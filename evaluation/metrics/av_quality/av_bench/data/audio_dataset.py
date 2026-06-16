import logging
from pathlib import Path
from typing import List

import torch
import torchaudio
import torchvision.transforms.v2 as v2
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torch.utils.data import Dataset

log = logging.getLogger()


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


# from synchformer
def pad_or_truncate(audio: torch.Tensor,
                    max_spec_t: int,
                    pad_mode: str = 'constant',
                    pad_value: float = 0.0):
    difference = max_spec_t - audio.shape[-1]  # safe for batched input
    # pad or truncate, depending on difference
    if difference > 0:
        # pad the last dim (time) -> (..., n_mels, 0+time+difference)  # safe for batched input
        pad_dims = (0, difference)
        audio = torch.nn.functional.pad(audio, pad_dims, pad_mode, pad_value)
    elif difference < 0:
        log.warning(f'Truncating spec ({audio.shape}) to max_spec_t ({max_spec_t}).')
        audio = audio[..., :max_spec_t]  # safe for batched input
    return audio


def pad_short_audio(audio, min_samples=32000):
    if (audio.size(-1) < min_samples):
        audio = torch.nn.functional.pad(audio, (0, min_samples - audio.size(-1)),
                                        mode='constant',
                                        value=0.0)
    return audio


# from https://github.com/haoheliu/audioldm_eval
class AudioDataset(Dataset):

    def __init__(
        self,
        datalist: List[Path],
        audio_length: float = 8.0,
        sr: int = 16000,
        limit_num=None,
    ):
        self.datalist = datalist
        if limit_num is not None:
            self.datalist = self.datalist[:limit_num]
        self.sr = sr
        self.audio_length = audio_length

        self.resampler = {}

    def __getitem__(self, idx: int):
        while True:
            try:
                filename = self.datalist[idx]
                waveform = self.read_from_file(filename)
                break
            except Exception as e:
                log.error(f'Error loading {self.datalist[idx]}: {e}')
                idx = (idx + 1) % len(self.datalist)

        return waveform, filename.stem

    def __len__(self):
        return len(self.datalist)

    def read_from_file(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        waveform = waveform.mean(dim=0)  # mono
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

        audio = audio[:int(self.sr * self.audio_length)].unsqueeze(0)

        return audio


class ImageBindAudioDataset(Dataset):

    def __init__(self, datalist: List[Path]):
        self.datalist = datalist

    # from ImageBind
    def load_and_transform_audio_data(
        self,
        audio_path,
        num_mel_bins=128,
        target_length=204,
        sample_rate=16000,
        clip_duration=2,
        clips_per_video=3,  # some frames not sampled when 2s
        mean=-4.268,
        std=9.138,
    ):

        audio_outputs = []
        clip_sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration,
                                                    clips_per_video=clips_per_video)

        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
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

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx: int):
        filename = self.datalist[idx]
        return self.load_and_transform_audio_data(filename), filename.stem


class SynchformerAudioDataset(Dataset):

    def __init__(self, datalist: List[Path], duration: float = 8.0):
        self.datalist = datalist
        self.expected_length = int(16000 * duration)
        self.resampler = {}

    def __len__(self):
        return len(self.datalist)

    def sample(self, idx: int):
        filename = self.datalist[idx]
        waveform, sr = torchaudio.load(filename)
        waveform = waveform.mean(dim=0)

        if sr != 16000:
            if sr not in self.resampler:
                self.resampler[sr] = torchaudio.transforms.Resample(sr, 16000)
            waveform = self.resampler[sr](waveform)

        waveform = waveform[:self.expected_length]
        if waveform.shape[0] != self.expected_length:
            raise ValueError(f'Audio {filename} is too short')

        waveform = waveform.squeeze()

        return waveform, filename.stem

    def __getitem__(self, idx: int):
        while True:
            try:
                return self.sample(idx)
            except Exception as e:
                log.error(f'Error loading {self.datalist[idx]}: {e}')
                idx = (idx + 1) % len(self.datalist)
