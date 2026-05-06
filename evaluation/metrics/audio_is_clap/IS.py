import os
import sys
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
from torch.nn import functional as F

# panns_inference reads ~/panns_data/class_labels_indices.csv at import time
# to determine classes_num. If the file is missing/empty, classes_num=0 and
# Cnn14 fc_audioset has shape [0, 2048], causing a checkpoint load mismatch.
# Ensure the local CSV is copied to the expected location before importing.
_PANNS_CSV_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pann_home', 'class_labels_indices.csv')
_PANNS_CSV_DST = os.path.join(os.path.expanduser('~'), 'panns_data', 'class_labels_indices.csv')
if not os.path.isfile(_PANNS_CSV_DST) or os.path.getsize(_PANNS_CSV_DST) < 100:
    os.makedirs(os.path.dirname(_PANNS_CSV_DST), exist_ok=True)
    import shutil
    shutil.copy2(_PANNS_CSV_SRC, _PANNS_CSV_DST)

import panns_inference

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# Config
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sr = 32000  # PANNs pretrained model requires 32kHz
SOFTMAX_TEMPERATURE = 1.0


# ======================
# 1) Load audio
# ======================
def load_audio_files(directory, sr=32000):
    audio_data = []
    names = []
    for file in os.listdir(directory):
        if file.lower().endswith(".wav"):
            path = os.path.join(directory, file)
            wav, fs = torchaudio.load(path)
            if fs != sr:
                wav = torchaudio.transforms.Resample(fs, sr)(wav)
            wav = wav.mean(dim=0)  # Convert to mono
            audio_data.append(wav)
            names.append(file)
    print(f"[DEBUG] Loaded {len(audio_data)} audio files from {directory}")
    return names, audio_data


# ======================
# 2) Cnn14 (panns-inference wrapper)
# ======================
class Cnn14Extractor:
    def __init__(self):
        # Use panns_inference wrapper

        model_path = os.path.join(SCRIPT_DIR, 'pann_home', 'Cnn14_mAP=0.431.pth')
        # model = panns_inference.AudioTagging(checkpoint_path=model_path, device=device)

        self.model = panns_inference.AudioTagging(checkpoint_path=model_path, device=device)

        self.model.model.eval()

    def _ten_seconds(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        target_len = 320000
        if x.size(1) < target_len:
            x = F.pad(x, (0, target_len - x.size(1)))
        elif x.size(1) > target_len:
            x = x[:, :target_len]
        return x

    def __call__(self, x, temperature=1.0):
        # Convert to numpy for panns_inference
        x = self._ten_seconds(x).cpu().numpy()
        out = self.model.inference(x)
        
        if isinstance(out, dict):
            clipwise = out["clipwise_output"]
        elif isinstance(out, (tuple, list)):
            clipwise = out[0]
        else:
            raise TypeError(f"Unexpected model output type: {type(out)}")

        p = torch.tensor(clipwise, device=device)


        eps = 1e-8
        logits = torch.log(torch.clamp(p, eps, 1 - eps)) - torch.log(1 - torch.clamp(p, eps, 1 - eps))
        logits = logits / float(temperature)
        return torch.softmax(logits, dim=-1)


# ======================
# 3) Feature extraction
# ======================
def extract_softmax(audio_list, model, temperature=1.0):
    feats = []
    for wav in tqdm(audio_list, desc="Extracting softmax"):
        wav = wav.to(device)
        prob = model(wav, temperature=temperature).squeeze().cpu().numpy()
        feats.append(prob)
    return np.vstack(feats)


# ======================
# 4) Inception Score
# ======================
def calculate_inception_score(preds, eps=1e-10):
    preds = preds / preds.sum(axis=1, keepdims=True)
    p_y = np.mean(preds, axis=0, keepdims=True)
    kl = preds * (np.log(preds + eps) - np.log(p_y + eps))
    kl = np.mean(np.sum(kl, axis=1))
    return float(np.exp(kl))


# ======================
# 5) Main
# ======================
def main():
    root_eval_path = "./"
    subfolders = ["wav"]
    log_path = os.path.join(root_eval_path, "cnn14_is_softmax_results.log")

    print("Loading Cnn14 model (panns-inference, 32k)...")
    model = Cnn14Extractor()

    with open(log_path, "w") as f:
        for sub in subfolders:
            gen_path = os.path.join(root_eval_path, sub)
            print(f"\n=== Evaluating {sub} ===")
            names, gen_audio = load_audio_files(gen_path, sr)
            if len(gen_audio) == 0:
                print(f"[ERROR] No wav files found in {gen_path}")
                continue

            gen_softmax = extract_softmax(gen_audio, model, temperature=SOFTMAX_TEMPERATURE)
            is_score = calculate_inception_score(gen_softmax)

            line = f"{sub}: IS={is_score:.4f}\n"
            print(line.strip())
            f.write(line)

    print(f"\nEvaluation complete, results saved to {log_path}")


if __name__ == "__main__":
    main()


