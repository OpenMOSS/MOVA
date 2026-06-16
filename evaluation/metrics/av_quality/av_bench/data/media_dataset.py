import logging
from pathlib import Path
from typing import List
import warnings

import librosa
import numpy as np
import yaml
from einops import rearrange
import cv2

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# torio.io.StreamingMediaDecoder was removed in torchaudio >= 2.10
# Fall back to decord for video decoding
try:
    from torio.io import StreamingMediaDecoder
    _HAS_TORIO = True
except (ModuleNotFoundError, ImportError):
    _HAS_TORIO = False
    import decord

import torchaudio
import torchvision.transforms.v2 as v2
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

from av_bench.data.ib_data import SpatialCrop
from dover.datasets import spatial_temporal_view_decomposition, UnifiedFrameSampler
from dover.models import DOVER

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


def fuse_results(results: list):
    ## results[0]: aesthetic, results[1]: technical
    ## thank @dknyxh for raising the issue
    t, a = (results[1] - 0.1107) / 0.07355, (results[0] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1 / (1 + np.exp(-a)),
        "technical": 1 / (1 + np.exp(-t)),
        "overall": 1 / (1 + np.exp(-x)),
    }

log = logging.getLogger()

# https://github.com/facebookresearch/ImageBind/blob/main/imagebind/data.py
# https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/transforms/functional.html
_IMAGEBIND_SIZE = 224
_IMAGEBIND_FPS = 0.5

_SYNC_SIZE = 224
_SYNC_FPS = 25.0

def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        warnings.warn("Received an entire batch of None. Returning empty list.")
        return []
    return default_collate(batch)

# from ImageBind
def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


# from ImageBind
def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=10,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank

def calculate_snr_silence(waveform, sample_rate, silence_threshold_db=-40):
    """SNR and silence ratio"""
    # signal power
    signal_power = np.mean(waveform**2)
    
    # detect silence segments as noise reference
    intervals = librosa.effects.split(
        waveform, 
        top_db=abs(silence_threshold_db),
        frame_length=2048,
        hop_length=512
    )   # this is the non-silence segments
    
    # get silence intervals
    current_silence_start = 0
    silence_intervals = []
    for start, end in intervals:
        if start ==0:
            current_silence_start = end
        else:
            silence_intervals.append((current_silence_start, start))
            current_silence_start = end
    silence_intervals.append((current_silence_start, len(waveform)))

    # extract all silence segments
    noise_segments = []
    silence_length = 0
    for start, end in silence_intervals:
        if start == 0 or end == len(waveform):
            continue
        noise_segments.append(waveform[start:end])
        silence_length += end - start

    silence_ratio = silence_length / len(waveform)

    if not noise_segments:
        return float('inf'), silence_ratio  # no silence segments detected
    
    noise_signal = np.concatenate(noise_segments)
    noise_power = np.mean(noise_signal**2)
    
    # avoid division by zero
    if noise_power < 1e-10:
        return float('inf'), silence_ratio
    
    # calculate SNR (dB)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr, silence_ratio

def calculate_bandwidth(waveform, sample_rate):
    spec_bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)
    return np.mean(spec_bw)

# Function to detect audio peaks using the Onset Detection algorithm
def detect_audio_peaks(waveform, sample_rate):
    """
    Detect audio peaks using the Onset Detection algorithm.

    Args:
        waveform (torch.Tensor): Audio waveform.
        sample_rate (int): Sample rate of the audio.

    Returns:
        onset_times (list): List of times (in seconds) where audio peaks occur.
    """
    # Calculate the onset envelope
    onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
    # Get the onset events
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate)
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
    return onset_times


# Function to find local maxima in a list
def find_local_max_indexes(arr, fps):
    """
    Find local maxima in a list.

    Note:
        In this implementation, local maxima with an optical flow magnitude less than 0.1
        are ignored. This has always been the case to prevent static scenes from being
        incorrectly calculated as peaks due to very small optical flow.

    Args:
        arr (list): List of values to find local maxima in.
        fps (float): Frames per second, used to convert indexes to time.

    Returns:
        local_extrema_indexes (list): List of times (in seconds) where local maxima occur.
    """
    local_extrema_indexes = []
    n = len(arr)
    for i in range(1, n - 1):
        # Only consider local maxima with magnitude at least 0.1
        if arr[i - 1] < arr[i] > arr[i + 1] and arr[i] >= 0.1:
            local_extrema_indexes.append(i / fps)
    return local_extrema_indexes


# Function to detect video peaks using Optical Flow
def detect_video_peaks(frames, fps):
    """
    Detect video peaks using Optical Flow.

    Args:
        frames (list): List of video frames.
        fps (float): Frame rate of the video.

    Returns:
        flow_trajectory (list): List of optical flow magnitudes for each frame.
        video_peaks (list): List of times (in seconds) where video peaks occur.
    """
    flow_trajectory = [compute_of(frames[0], frames[1])] + [compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    video_peaks = find_local_max_indexes(flow_trajectory, fps)
    return flow_trajectory, video_peaks


# Function to compute the optical flow magnitude between two frames
def compute_of(img1, img2):
    """
    Compute the optical flow magnitude between two video frames.

    Args:
        img1 (numpy.ndarray): First video frame.
        img2 (numpy.ndarray): Second video frame.

    Returns:
        avg_magnitude (float): Average optical flow magnitude for the frame pair.
    """
    # Calculate the optical flow
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude of the optical flow vectors
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    avg_magnitude = cv2.mean(magnitude)[0]
    return avg_magnitude


# Function to calculate Intersection over Union (IoU) for audio and video peaks
def calc_intersection_over_union(audio_peaks, video_peaks, fps):
    """
    Calculate Intersection over Union (IoU) between audio and video peaks.

    Note:
        A video peak is matched to at most one audio peak, as has always been the case,
        ensuring that a single video peak does not correspond to multiple audio peaks.

    Args:
        audio_peaks (list): List of audio peak times (in seconds).
        video_peaks (list): List of video peak times (in seconds).
        fps (float): Frame rate of the video.

    Returns:
        iou (float): Intersection over Union score.
    """
    intersection_length = 0
    used_video_peaks = [False] * len(video_peaks)
    for audio_peak in audio_peaks:
        for j, video_peak in enumerate(video_peaks):
            if not used_video_peaks[j] and video_peak - 1 / fps < audio_peak < video_peak + 1 / fps:
                intersection_length += 1
                used_video_peaks[j] = True
                break
    return intersection_length / (len(audio_peaks) + len(video_peaks) - intersection_length + 1e-9) # in case of division by zero error

def calculate_av_align(waveform, sample_rate, video_frames, fps):
    # video frames from RGB to BGR
    video_frames = video_frames.numpy()[:, ::-1].transpose(0,2,3,1)
    # calculate the audio peaks
    audio_peaks = detect_audio_peaks(waveform, sample_rate)
    # calculate the video peaks
    flow_trajectory, video_peaks = detect_video_peaks(video_frames, fps)
    # calculate the intersection over union
    iou = calc_intersection_over_union(audio_peaks, video_peaks, fps)
    return iou

def imagebind_audio_transform(
    waveform,
    current_sr,
    audio_16k_sampler,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration,
                                                clips_per_video=clips_per_video)

    # resample if necessary
    # if sample_rate != current_sr:
    #     waveform = torchaudio.functional.resample(waveform, orig_freq=current_sr, new_freq=sample_rate)
    if current_sr != sample_rate:
        if current_sr not in audio_16k_sampler:
            audio_16k_sampler[current_sr] = torchaudio.transforms.Resample(
                current_sr,
                sample_rate
            )
        waveform = audio_16k_sampler[current_sr](waveform)

    all_clips_timepoints = get_clip_timepoints(clip_sampler, waveform.size(1) / sample_rate)
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = waveform[
            :,
            int(clip_timepoints[0] * sample_rate):int(clip_timepoints[1] * sample_rate),
        ]
        waveform_melspec = waveform2melspec(waveform_clip, sample_rate, num_mel_bins,
                                            target_length)
        all_clips.append(waveform_melspec)

    normalize = v2.Normalize(mean=[mean], std=[std])
    all_clips = [normalize(ac) for ac in all_clips]

    all_clips = torch.stack(all_clips, dim=0)
    audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)

def synchformer_audio_transform(waveform, sr, expected_length, audio_16k_sampler):
    waveform = waveform.mean(dim=0)

    # if sr != 16000:
    #     waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    if sr != 16000:
        if sr not in audio_16k_sampler:
            audio_16k_sampler[sr] = torchaudio.transforms.Resample(
                sr,
                16000
            )
        waveform = audio_16k_sampler[sr](waveform)

    waveform = waveform[:expected_length]
    if waveform.shape[0] != expected_length:
        raise ValueError(f'Audio is too short')

    waveform = waveform.squeeze()

    return waveform


def _decode_video_decord(video_path, target_fps, num_frames):
    """Decode video frames at a target FPS using decord. Returns (N, C, H, W) float tensor."""
    vr = decord.VideoReader(str(video_path))
    orig_fps = vr.get_avg_fps()
    # Sample frame indices at the target FPS
    duration = len(vr) / orig_fps
    target_ts = np.arange(num_frames) / target_fps
    # Clamp to video duration
    target_ts = np.clip(target_ts, 0, duration - 1e-6)
    frame_indices = (target_ts * orig_fps).astype(np.int64)
    frame_indices = np.clip(frame_indices, 0, len(vr) - 1)
    frames = vr.get_batch(frame_indices)  # (N, H, W, C) uint8
    if isinstance(frames, torch.Tensor):
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
    else:
        frames = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2).float() / 255.0
    return frames


def video_transform(video_path, duration_sec,
                    ib_transform, sync_transform, crop,
                    synchformer_resample=False):

    ib_expected_length = int(_IMAGEBIND_FPS * duration_sec)
    sync_expected_length = int(_SYNC_FPS * duration_sec)

    if _HAS_TORIO:
        reader = StreamingMediaDecoder(str(video_path))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_IMAGEBIND_FPS * duration_sec),
            frame_rate=_IMAGEBIND_FPS,
            format='rgb24',
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * duration_sec),
            frame_rate=_SYNC_FPS if synchformer_resample else None,
            format='rgb24',
        )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        ib_chunk = data_chunk[0]
        sync_chunk = data_chunk[1]
    else:
        # decord fallback: decode at both target FPS
        ib_chunk = _decode_video_decord(video_path, _IMAGEBIND_FPS, ib_expected_length)
        sync_chunk = _decode_video_decord(video_path, _SYNC_FPS, sync_expected_length)
    if ib_chunk is None:
        raise RuntimeError(f'IB video returned None {video_path}')
    if ib_chunk.shape[0] < ib_expected_length:
        last_frame = ib_chunk[-1:]  # 获取最后一帧
        repeat_count = ib_expected_length - ib_chunk.shape[0]
        padding = last_frame.repeat(repeat_count, 1, 1, 1)
        ib_chunk = torch.cat([ib_chunk, padding], dim=0)
    # if ib_chunk.shape[0] < ib_expected_length:
    #     raise RuntimeError(
    #         f'IB video too short {video_path}, expected {ib_expected_length}, got {ib_chunk.shape[0]}'
    #     ) 

    if sync_chunk is None:
        raise RuntimeError(f'Sync video returned None {video_path}')
    if sync_chunk.shape[0] < sync_expected_length:
        last_frame = sync_chunk[-1:]  # 获取最后一帧
        repeat_count = sync_expected_length - sync_chunk.shape[0] 
        padding = last_frame.repeat(repeat_count, 1, 1, 1)
        sync_chunk = torch.cat([sync_chunk, padding], dim=0)
        logging.warning(f'帧数补齐: {video_path}')
    # if sync_chunk.shape[0] < sync_expected_length:
    #     raise RuntimeError(
    #         f'Sync video too short {video_path}, expected {sync_expected_length}, got {sync_chunk.shape[0]}'
    #     )


    # truncate the video
    ib_chunk = ib_chunk[:ib_expected_length]
    if ib_chunk.shape[0] != ib_expected_length:
        raise RuntimeError(f'IB video wrong length {video_path}, '
                            f'expected {ib_expected_length}, '
                            f'got {ib_chunk.shape[0]}')
    ib_chunk = ib_transform(ib_chunk)

    sync_chunk = sync_chunk[:sync_expected_length]
    if sync_chunk.shape[0] != sync_expected_length:
        raise RuntimeError(f'Sync video wrong length {video_path}, '
                            f'expected {sync_expected_length}, '
                            f'got {sync_chunk.shape[0]}')
    sync_chunk = sync_transform(sync_chunk)

    ib_chunk = crop([ib_chunk])
    ib_chunk = torch.stack(ib_chunk)

    return ib_chunk, sync_chunk


class MediaDataset(Dataset):

    def __init__(
        self,
        dover_opt,
        datalist: List[Path],
        duration: float = 8.0,
        sr: int = 16000,
        limit_num=None,
        media_type='video',
        extract_imagebind_audio=True,
        extract_synchformer_audio=True,
        synchformer_resample=False,
    ):
        self.datalist = datalist
        if limit_num is not None:
            self.datalist = self.datalist[:limit_num]
        self.duration = duration
        self.sr = sr
        self.media_type = media_type

        # extract
        self.extract_imagebind_audio = extract_imagebind_audio
        self.extract_synchformer_audio = extract_synchformer_audio
        self.synchformer_resample = synchformer_resample

        # resampler
        self.resampler = {}
        self.audio_16k_sampler = {}
        
        # SynchformerAudioDataset
        self.synchformer_expected_length = int(16000 * duration)

        self.ib_transform = v2.Compose([
            v2.Resize(_IMAGEBIND_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.crop = SpatialCrop(_IMAGEBIND_SIZE, 3)

        # dover related
        self.sample_types = dover_opt["sample_types"]
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.dover_samplers = {}
        for stype, sopt in dover_opt["sample_types"].items():
            if "t_frag" not in sopt:
                # resized temporal sampling for TQE in DOVER
                self.dover_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                self.dover_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )
            print(
                stype + " branch sampled frames:",
                self.dover_samplers[stype](240, False),
            )

    def __len__(self):
        return len(self.datalist)

    def audio_read_from_file(self, audio_file):
        raw_waveform, sample_rate = torchaudio.load(str(audio_file))
        waveform = raw_waveform.mean(dim=0)  # mono
        waveform = waveform - waveform.mean()

        if sample_rate == self.sr:
            audio = waveform
        else:
            if sample_rate not in self.resampler:
                self.resampler[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate,
                    self.sr,
                )
            audio = self.resampler[sample_rate](waveform)

        audio = audio[:int(self.sr * self.duration)].unsqueeze(0)
        return audio, raw_waveform, sample_rate


    def __getitem__(self, idx: int):
        while True:
            try:
                filename = self.datalist[idx]
                # AudioDataset
                audio_waveform, raw_waveform, org_sample_rate = self.audio_read_from_file(filename)

                snr, silence_ratio = calculate_snr_silence(raw_waveform.mean(0).numpy(), org_sample_rate)
                bandwidth = calculate_bandwidth(raw_waveform.mean(0).numpy(), org_sample_rate)

                snr = torch.tensor(snr)
                silence_ratio = torch.tensor(silence_ratio)

                return_dict = {
                    "filename": filename.parent.name+'/'+filename.stem,  # this is for finevideo
                    "audio_waveform": audio_waveform,
                    "silence_ratio": silence_ratio,
                    "snr": snr,
                    "bandwidth": bandwidth,
                }

                # ImageBindAudioDataset
                if self.extract_imagebind_audio:
                    imagebind_audio = imagebind_audio_transform(raw_waveform, 
                                                                org_sample_rate, 
                                                                self.audio_16k_sampler)

                    return_dict["ib_audio"] = imagebind_audio

                # SynchformerAudioDataset
                if self.extract_synchformer_audio:
                    synchformer_audio = synchformer_audio_transform(raw_waveform, 
                                                                    org_sample_rate, 
                                                                    self.synchformer_expected_length, 
                                                                    self.audio_16k_sampler)
                    return_dict["sync_audio"] = synchformer_audio
                
                if self.media_type == 'video':
                    ib_chunk, sync_chunk = video_transform(filename, self.duration, 
                                                           self.ib_transform, self.sync_transform, self.crop,
                                                           synchformer_resample=self.synchformer_resample)
                    return_dict["ib_video"] = ib_chunk
                    return_dict["sync_video"] = sync_chunk
                    return_dict["av_align"] = calculate_av_align(raw_waveform.mean(0).numpy(), 
                                                                org_sample_rate, 
                                                                sync_chunk, 
                                                                25)

                    dover_data, frame_inds = spatial_temporal_view_decomposition(
                        str(filename),
                        self.sample_types,
                        self.dover_samplers,
                        False,
                        False,
                    )

                    for k, v in dover_data.items():
                        dover_data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(
                            3, 0, 1, 2
                        )

                    dover_data["num_clips"] = {}
                    for stype, sopt in self.sample_types.items():
                        dover_data["num_clips"][stype] = sopt["num_clips"]
                    dover_data["frame_inds"] = frame_inds
                    # dover_data["name"] = filename
                    return_dict["dover_data"] = dover_data

                # return all
                return return_dict

            except Exception as e:
                log.error(f'Error loading {self.datalist[idx]}: {e}')
                return None


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    with open(os.path.join(_repo_root, "models", "dover", "dover.yml"), "r") as f:
        opt = yaml.safe_load(f)

    dover_opt = opt["data"]["val-l1080p"]["args"]
    dataset = MediaDataset(dover_opt,
                            datalist=[Path('.'), Path('.'), Path('.'), Path('.')],
                            duration=8.0,
                            )
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=error_avoidance_collate)

    dover_model = DOVER(**opt["model"]["args"]).to(device)
    dover_model.load_state_dict(
        torch.load(os.path.join(_repo_root, 'models', 'dover', 'pretrained_weights', 'DOVER.pth'), map_location=device)
    )
    sample_types = ["aesthetic", "technical"]
    for data in data_loader:
        # print(data['dover_data'])
        if isinstance(data, list) and len(data) == 0:
            print("Empty batch, skip!!!")
            continue
        video = {}
        dover_data = data['dover_data']
        batch_size = None
        for key in sample_types:
            if key in dover_data:
                video[key] = dover_data[key].to(device)
                b, c, t, h, w = video[key].shape
                batch_size = b
                video[key] = (
                    video[key]
                    .reshape(
                        b, c, dover_data["num_clips"][key][0], t // dover_data["num_clips"][key][0], h, w
                    )   # Important! Assume same num_clips across batch here
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        b * dover_data["num_clips"][key][0], c, t // dover_data["num_clips"][key][0], h, w
                    )   # Important! Assume same num_clips across batch here
                )

        with torch.no_grad():
            results = dover_model(video, reduce_scores=False)
            results[0] = rearrange(results[0], '(b n) c t h w -> b n c t h w', b=batch_size)
            results[1] = rearrange(results[1], '(b n) c t h w -> b n c t h w', b=batch_size)
            results = [np.mean(l.cpu().numpy(), axis=tuple(range(1, len(l.shape)))) for l in results]
            rescaled_results = fuse_results(results)
        print(rescaled_results)
        print(data['av_align'])
        break
    # print(dataset[0]["dover_data"])