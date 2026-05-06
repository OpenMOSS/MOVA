import logging
from pathlib import Path

import torch
import torchaudio
from imagebind.models.imagebind_model import ModalityType
from torch.utils.data import DataLoader
from tqdm import tqdm

from av_bench.data.audio_dataset import (AudioDataset, ImageBindAudioDataset,
                                         SynchformerAudioDataset, pad_or_truncate)
from av_bench.extraction_models import ExtractionModels
from av_bench.synchformer.synchformer import Synchformer

log = logging.getLogger()


def encode_audio_with_sync(synchformer: Synchformer, x: torch.Tensor,
                           mel: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    b, t = x.shape

    # partition the video
    segment_size = 10240
    step_size = 10240 // 2
    num_segments = (t - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size:i * step_size + segment_size])
    x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

    x = mel(x)
    x = torch.log(x + 1e-6)
    x = pad_or_truncate(x, 66)

    mean = -4.2677393
    std = 4.5689974
    x = (x - mean) / (2 * std)
    # x: B * S * 128 * 66
    x = synchformer.extract_afeats(x.unsqueeze(2))
    return x


@torch.inference_mode()
def extract(audio_path: Path,
            output_path: Path,
            *,
            audio_length: float = 8.0,
            batch_size: int = 128,
            num_workers: int = 32,
            device: str,
            skip_video_related: bool = False,
            skip_clap: bool = False):

    audios = sorted(list(audio_path.glob('*.wav')) + list(audio_path.glob('*.flac')),
                    key=lambda x: x.stem)

    log.info(f'{len(audios)} audios found in {audio_path}')

    models = ExtractionModels().to(device).eval()
    dataset = AudioDataset(audios, audio_length=audio_length, sr=16000)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # extract features with PANN
    out_dict = {}
    for wav, filename in tqdm(loader):
        wav = wav.squeeze(1).float().to(device)

        features = models.panns(wav)
        features = {k: v.cpu() for k, v in features.items()}
        for i, f_name in enumerate(filename):
            out_dict[f_name] = {k: v[i] for k, v in features.items()}

    output_path.mkdir(parents=True, exist_ok=True)
    pann_feature_path = output_path / 'pann_features.pth'
    log.info(f'Saving {len(out_dict)} features to {pann_feature_path}')
    torch.save(out_dict, pann_feature_path)
    del out_dict

    # extract features with VGGish
    out_dict = {}
    for wav, filename in tqdm(loader):
        wav = wav.squeeze(1).float()
        features = models.vggish(wav).cpu()
        for i, f_name in enumerate(filename):
            out_dict[f_name] = features[i]

    output_path.mkdir(parents=True, exist_ok=True)
    vggish_feature_path = output_path / 'vggish_features.pth'
    log.info(f'Saving {len(out_dict)} features to {vggish_feature_path}')
    torch.save(out_dict, vggish_feature_path)
    del out_dict

    if not skip_video_related:
        # extract features with ImageBind
        dataset = ImageBindAudioDataset(audios)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)
        out_dict = {}
        for wav, filename in tqdm(loader):
            wav = wav.squeeze(1).to(device)
            features = models.imagebind({ModalityType.AUDIO: wav})[ModalityType.AUDIO].cpu()
            for i, f_name in enumerate(filename):
                out_dict[f_name] = features[i]
        output_path.mkdir(parents=True, exist_ok=True)
        imagebind_feature_path = output_path / 'imagebind_audio.pth'
        log.info(f'Saving {len(out_dict)} features to {imagebind_feature_path}')
        torch.save(out_dict, imagebind_feature_path)

        # extract features with Synchformer
        dataset = SynchformerAudioDataset(audios, duration=audio_length)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)
        out_dict = {}
        for wav, filename in tqdm(loader):
            wav = wav.to(device)
            features = encode_audio_with_sync(models.synchformer, wav,
                                              models.sync_mel_spectrogram).cpu()
            for i, f_name in enumerate(filename):
                out_dict[f_name] = features[i]
        output_path.mkdir(parents=True, exist_ok=True)
        synchformer_feature_path = output_path / 'synchformer_audio.pth'
        log.info(f'Saving {len(out_dict)} features to {synchformer_feature_path}')
        torch.save(out_dict, synchformer_feature_path)

    if not skip_clap:
        # extract features with CLAP
        dataset = AudioDataset(audios, audio_length=audio_length, sr=48_000)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)
        out_dict = {}
        for wav, filename in tqdm(loader):
            wav = wav.squeeze(1).to(device)
            clap_features = models.laion_clap.get_audio_embedding_from_data(wav,
                                                                            use_tensor=True).cpu()
            for i, f_name in enumerate(filename):
                out_dict[f_name] = clap_features[i].squeeze()
        output_path.mkdir(parents=True, exist_ok=True)
        clap_feature_path = output_path / 'clap_laion_audio.pth'
        log.info(f'Saving {len(out_dict)} features to {clap_feature_path}')
        torch.save(out_dict, clap_feature_path)

        out_dict = {}
        for i in range(0, len(audios), batch_size):
            audio_paths = audios[i:i + batch_size]
            ms_features = models.ms_clap.get_audio_embeddings(audio_paths).cpu().squeeze()
            for f_name, f in zip(audio_paths, ms_features):
                out_dict[f_name.stem] = f
        ms_feature_path = output_path / 'clap_ms_audio.pth'
        log.info(f'Saving {len(out_dict)} features to {ms_feature_path}')
        torch.save(out_dict, ms_feature_path)

    # PassT, need 32K sampling rate
    dataset = AudioDataset(audios, audio_length=audio_length, sr=32_000)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    out_features = {}
    out_logits = {}
    for wav, filename in tqdm(loader):
        wav = wav.squeeze(1).float().to(device)

        if (wav.size(-1) >= 320000):
            wav = wav[..., :320000]
        else:
            wav = torch.nn.functional.pad(wav, (0, 320000 - wav.size(-1)))

        features = models.passt_model(wav).cpu()
        # see https://github.com/kkoutini/passt_hear21/blob/5f1cce6a54b88faf0abad82ed428355e7931213a/hear21passt/wrapper.py#L40
        # logits is 527 dim; features is 768
        logits = features[:, :527]
        features = features[:, 527:]
        for i, f_name in enumerate(filename):
            out_features[f_name] = features[i]
            out_logits[f_name] = logits[i]
    output_path.mkdir(parents=True, exist_ok=True)
    passt_feature_path = output_path / 'passt_features_embed.pth'
    log.info(f'Saving {len(out_features)} features to {passt_feature_path}')
    torch.save(out_features, passt_feature_path)

    passt_feature_path = output_path / 'passt_logits.pth'
    log.info(f'Saving {len(out_logits)} features to {passt_feature_path}')
    torch.save(out_logits, passt_feature_path)
