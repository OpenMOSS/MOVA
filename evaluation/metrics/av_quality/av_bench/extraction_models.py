from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from imagebind.models import imagebind_model

from av_bench.panns import Cnn14
from av_bench.synchformer.synchformer import Synchformer

_syncformer_ckpt_path = Path(__file__).parent.parent / 'weights' / 'synchformer_state_dict.pth'


class ExtractionModels(nn.Module):

    def __init__(self):
        super().__init__()

        features_list = ["2048", "logits"]

        print("[1/4] Loading PANNs (Cnn14) model...")
        self.panns = Cnn14(
            features_list=features_list,
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        )
        self.panns = self.panns.eval()
        print("PANNs loaded successfully.\n")

        print("[2/4] Loading ImageBind model...")
        self.imagebind = imagebind_model.imagebind_huge(pretrained=True).eval()
        print("ImageBind loaded successfully.\n")

        print("[3/4] Loading Synchformer model...")
        self.synchformer = Synchformer().eval()
        sd = torch.load(_syncformer_ckpt_path, map_location='cpu', weights_only=True)
        self.synchformer.load_state_dict(sd)
        print("Synchformer loaded successfully.\n")

        print("[4/4] Initializing MelSpectrogram...")
        self.sync_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            win_length=400,
            hop_length=160,
            n_fft=1024,
            n_mels=128,
        )
        print("MelSpectrogram initialized successfully.\n")
        print("All models loaded and ready.")
