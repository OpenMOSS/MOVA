# MOVA Evaluation Toolkit

Evaluation toolkit for [**MOVA**](https://github.com/OpenMOSS/MOVA) (**MO**SS **V**ideo and **A**udio), providing a comprehensive suite of metrics for assessing synchronized audio-visual video generation quality.

## Metrics Overview

| # | Metric | Script | What it measures | Higher/Lower is better |
|---|--------|--------|------------------|----------------------|
| 1 | **Audio Amplitude & Loudness** | `metrics/audio_amplitude/eval_audio_amplitude.py` | RMS amplitude, LUFS loudness | Context-dependent |
| 2 | **DNSMOS (P808 MOS)** | `metrics/dnsmos/eval_dnsmos.py` | Audio speech quality (no-reference) | Higher |
| 3 | **DeSync** | `metrics/av_quality/eval_av_quality.py` | Audio-video synchronization offset (Synchformer) | Lower |
| 4 | **IB-Score** | `metrics/av_quality/eval_av_quality.py` | Audio-visual semantic alignment (ImageBind cosine sim) | Higher |
| 5 | **AV-Align** | `metrics/av_quality/eval_av_quality.py` | Audio-onset / optical-flow alignment (IoU) | Higher |
| 6 | **VideoReward** | `metrics/video_reward/eval_video_reward.py` | Visual Quality (VQ), Motion Quality (MQ), Text Alignment (TA), Video Reward Overall | Higher |
| 7 | **IS (Inception Score)** | `metrics/audio_is_clap/eval_audio_is_clap.py` | Audio diversity/quality (PANNs Cnn14) | Higher |
| 8 | **CLAP Score** | `metrics/audio_is_clap/eval_audio_is_clap.py` | Audio-text alignment (cosine similarity) | Higher |
| 9 | **LSE-D** | `models/wav2lip/evaluation/syncnet_python/eval_lip_sync.py` | Lip sync error distance (SyncNet) | Lower |
| 10 | **LSE-C** | `models/wav2lip/evaluation/syncnet_python/eval_lip_sync.py` | Lip sync error confidence (SyncNet) | Higher |
| 11 | **cpCER** | `metrics/cpcer/eval_cpcer.py` | Multi-speaker conversational accuracy (MOSS Transcribe Diarize + Hungarian-matched CER) | Lower |

## Reproducibility Verification

We evaluate **MOVA-360p** (OpenVeo3-I2VA-A14B-1220, wsd variant) on the [Verse-Bench](https://huggingface.co/datasets/dorni/Verse-Bench) benchmark and [MyBench](#mybench) (our constructed benchmark) and compare this toolkit's scores against the MOVA technical report ([arXiv:2602.08794](https://arxiv.org/abs/2602.08794), Table 4).

To reproduce these results, ensure your input directory contains only a single checkpoint subdirectory that holds the category directories (e.g., `--input_dir /path/to/eval_videos/` with `checkpoint_name/multi-speaker/`, `checkpoint_name/movie/`, etc. inside). The subdirectory name is not constrained — any single top-level folder under `--input_dir` will be used.

### Verse-Bench

| Metric | Direction | Paper Score | mova_eval Score | Relative Error | Evaluated On |
|--------|-----------|-------------|-----------------|---------------|-------------|
| IS | ↑ | 4.269 | 4.269 | +0.0% | Verse-Bench (set1+2+3) |
| DNSMOS | ↑ | 3.797 | 3.798 | +0.0% | set3 (speech) |
| DeSync | ↓ | 0.475 | 0.475 | +0.0% | Verse-Bench (set1+2+3) |
| IB-Score | ↑ | 0.286 | 0.286 | +0.0% | Verse-Bench (set1+2+3) |
| LSE-D | ↓ | 8.098 | 8.098 | 0.0% | set3 (speech) |
| LSE-C | ↑ | 6.278 | 6.278 | 0.0% | set3 (speech) |
| AV-Align | ↑ | 0.238 | 0.238 | +0.0% | Verse-Bench (set1+2+3) |
| CLAP | ↑ | 0.268 | 0.268 | +0.0% | Verse-Bench (set1+2+3) |

### MyBench

| Metric | Direction | Paper Score | mova_eval Score | Relative Error | Evaluated On |
|--------|-----------|-------------|-----------------|---------------|-------------|
| cpCER | ↓ | 0.177 | 0.156 | -11.9% | MyBench (multi-speaker) |

**Notes:**
- Verse-Bench consists of set1 (205 videos), set2 (295 videos), and set3 (100 videos), totaling 600 items. When running with a Verse-Bench-only prompt JSON (containing only set1/set2/set3 categories), the `all` key in the output JSON equals the mean of per-category means for these three categories.
- IS, DeSync, and IB-Score are aggregated as the mean of per-category means across set1, set2, and set3.
- DNSMOS (a speech quality metric) is evaluated on set3 only, which contains exclusively speech videos.
- **LSE-D and LSE-C**: The paper-reported LSE-D/LSE-C scores are derived from the dedicated evaluation pipeline (i.e., the model variant and inference configuration used for benchmarking), rather than the Arena deployment. The Arena inference server employs a different production configuration optimized for serving, which may yield slightly different video outputs and thus different lip-sync scores. Since mova_eval replicates the evaluation pipeline, LSE-D and LSE-C achieve exact match (0.0% relative error) with the paper.
- **cpCER**: The cpCER score reported in the paper (0.177, Table 4) was measured using an internal beta version of the MOSS Transcribe Diarize (MTD) model. The publicly released MTD version adopted in this toolkit produces improved speaker diarization, yielding a lower (better) cpCER of 0.156. This discrepancy is therefore attributable to the ASR model version difference rather than evaluation methodology. Among the 132 MyBench items, only 27 belong to the `multi-speaker` category and contain multi-speaker conversational dialogues with ground-truth reference transcripts — cpCER can only be computed on these 27 videos, as the other categories (movie, shot-effect, anime, sports, games, others) do not necessarily feature multi-speaker conversations.

### MyBench

**MyBench** is our constructed evaluation benchmark with **132 samples** across **7 categories**: multi-speaker (27), movie (12), shot-effect (30), anime (20), sports (20), games (20), and others (3). It is designed to evaluate joint video–audio generation in realistic and challenging scenarios, complementing Verse-Bench with diverse video generation categories including multi-speaker interaction, movie-style narratives, sports competitions, game livestreams, camera motion sequences, and anime-style content. MyBench is also used in [MOVA Arena](https://mosi.cn/models/mova) comparisons. The multi-speaker reference transcripts for cpCER evaluation are provided at `metrics/cpcer/references.txt`.

## Directory Structure

```
evaluation/
├── scripts/
│   ├── run_eval.sh              # Main evaluation pipeline
│   ├── setup_envs.sh            # Create conda environments
│   ├── set_env.sh               # Set Python interpreter paths
│   ├── convert_prompts.py       # Convert prompt formats to prompts.json
│   └── download_weights.sh      # Download pretrained model weights
├── envs/
│   ├── requirements_visual.txt  # VideoReward dependencies
│   ├── requirements_av.txt      # AV Quality + Lip Sync dependencies
│   ├── requirements_audio.txt   # Audio IS + CLAP dependencies
│   ├── requirements_dnsmos.txt  # DNSMOS + Audio Amplitude dependencies
│   └── requirements_cpcer.txt   # cpCER dependencies
├── metrics/
│   ├── video_reward/            # VideoReward (VQ/MQ/TA/Video Reward Overall)
│   │   ├── eval_video_reward.py
│   │   ├── data.py
│   │   ├── prompt_template.py
│   │   ├── utils.py
│   │   ├── vision_process.py
│   │   ├── train_reward.py
│   │   └── checkpoints/         #   VideoReward weights (downloaded)
│   ├── av_quality/              # DeSync, IB-Score, AV-Align
│   │   ├── eval_av_quality.py
│   │   ├── av_align_score.py
│   │   ├── extract_video.py
│   │   ├── av_bench/            #   AV alignment sub-module
│   │   ├── pann_home/           #   PANNs Cnn14 checkpoint (16kHz)
│   │   └── weights/             #   Synchformer weights (downloaded)
│   ├── audio_is_clap/           # IS + CLAP Score
│   │   ├── eval_audio_is_clap.py
│   │   ├── IS.py
│   │   ├── data_fix.py
│   │   ├── pann_home/           #   PANNs Cnn14 checkpoint (32kHz)
│   │   └── clap_ckpt/           #   LAION-CLAP weights (downloaded)
│   ├── dnsmos/                  # DNSMOS P808 MOS
│   │   ├── eval_dnsmos.py
│   │   ├── DNSMOS/              #   ONNX models (4 files)
│   │   └── pDNSMOS/             #   P.835 variant ONNX
│   ├── audio_amplitude/         # RMS Amplitude + LUFS Loudness
│   │   └── eval_audio_amplitude.py
│   └── cpcer/                   # cpCER (multi-speaker conversational accuracy)
│       ├── eval_cpcer.py        #   Main evaluation script
│       ├── references.txt       #   27 multi-speaker reference transcripts
│       ├── speaker_metrics.py   #   cpCER computation (Hungarian matching + CER)
│       └── speaker_timestamp_utils.py
├── models/
│   ├── wav2lip/                 # Wav2Lip + SyncNet (lip sync eval)
│   │   ├── evaluation/
│   │   │   └── syncnet_python/
│   │   │       ├── eval_lip_sync.py
│   │   │       ├── SyncNetInstance.py
│   │   │       ├── SyncNetModel.py
│   │   │       ├── SyncNetInstance_calc_scores.py
│   │   │       ├── detectors/s3fd/    #   SFD face detector
│   │   │       └── data/              #   syncnet_v2.model (downloaded)
│   │   ├── models/              #   Wav2Lip model definitions
│   │   ├── audio.py
│   │   └── hparams.py
│   ├── pytorchvideo/            # PyTorchVideo (video backbone)
│   ├── clap/                    # LAION-CLAP (audio-text embedding)
│   │   └── src/laion_clap/
│   ├── imagebind/               # ImageBind (multi-modal embedding)
│   │   └── imagebind/
│   ├── dover/                   # DOVER (video quality assessment)
│   │   ├── dover/               #   Core model code
│   │   ├── models/              #   Model definitions
│   │   └── pretrained_weights/  #   DOVER.pth (downloaded)
│   ├── Qwen2-VL-2B-Instruct/   # Qwen2-VL (base model for VideoReward)
│   └── roberta-base/            # Roberta tokenizer/config (for CLAP)
├── assets/                      # Logo and images
├── verse_bench_prompts.json     # Verse-Bench prompt metadata
└── README.md
```

> **Note:** Weight files (`.pth`, `.pt`, `.model`) are excluded from git via `.gitignore`. They must be downloaded separately — see [Download Pretrained Weights](#2-download-pretrained-weights).

## Quick Start

### 1. Setup Environment

Each metric group requires specific dependencies. We provide two setup options:

#### Option A: Separate conda environments (recommended)

This avoids dependency conflicts between metric groups. We provide requirements files and setup scripts:

```bash
cd evaluation

# Create 5 conda environments (one per metric group)
bash scripts/setup_envs.sh

# Activate environment paths (must run before evaluation)
source scripts/set_env.sh
```

This creates:
| Environment | Metrics | Requirements |
|---|---|---|
| `mova_eval_visual` | VideoReward | `envs/requirements_visual.txt` |
| `mova_eval_av` | AV Quality + Lip Sync | `envs/requirements_av.txt` |

> ⚠️ **DeSync metric requires torio decoder**: The `mova_eval_av` environment must use `torchaudio<2.9.0` to ensure `torio.io.StreamingMediaDecoder` is available. If `torchaudio>=2.9.0` is installed, the DeSync metric falls back to the `decord` decoder, which produces inconsistent scores. The `requirements_av.txt` file enforces this version constraint.
| `mova_eval_audio` | Audio IS + CLAP | `envs/requirements_audio.txt` |
| `mova_eval_dnsmos` | DNSMOS + Audio Amplitude | `envs/requirements_dnsmos.txt` |
| `mova_eval_cpcer` | cpCER | `envs/requirements_cpcer.txt` |

#### Option B: Single environment

If you prefer one environment, install all dependencies together (may have version conflicts):

```bash
conda create -n mova_eval python=3.10 -y
conda activate mova_eval
pip install -r envs/requirements_visual.txt
pip install -r envs/requirements_av.txt
pip install -r envs/requirements_audio.txt
pip install -r envs/requirements_dnsmos.txt
pip install -r envs/requirements_cpcer.txt

# Install vendored model packages
pip install -e ./models/imagebind --no-deps
pip install -e ./models/dover --no-deps
pip install -e ./models/pytorchvideo --no-deps
pip install -e ./models/clap --no-deps
pip install -e ./metrics/av_quality/av_bench --no-deps
```

#### Using custom Python paths

If your environments are already set up elsewhere, set `PYTHON_*` environment variables to override:

```bash
export PYTHON_VIDEO_REWARD=/path/to/eval-visual/bin/python
export PYTHON_AV_QUALITY=/path/to/eval-av/bin/python
export PYTHON_AUDIO_QUALITY=/path/to/eval-audio/bin/python
export PYTHON_DNSMOS=/path/to/eval-dnsmos/bin/python
export PYTHON_AUDIO_AMPLITUDE="$PYTHON_DNSMOS"   # same as DNSMOS
export PYTHON_LIP_QUALITY="$PYTHON_AV_QUALITY"    # same as AV Quality
export PYTHON_CPCER=/path/to/eval-cpcer/bin/python
```

### 2. Download Pretrained Weights

Some evaluation metrics require pretrained model checkpoints that are **not bundled** with this repo (due to licensing and size).

```bash
# Interactive download with instructions (downloads all)
bash scripts/download_weights.sh

# Download specific metrics only
bash scripts/download_weights.sh --video_reward_only
bash scripts/download_weights.sh --av_quality_only
bash scripts/download_weights.sh --audio_is_clap_only
bash scripts/download_weights.sh --lip_sync_only
bash scripts/download_weights.sh --dover_only
```

Weights that need manual download (if auto-download fails):
- **PANNs Cnn14** → `metrics/audio_is_clap/pann_home/Cnn14_mAP=0.431.pth`

Weights auto-downloaded by the script:
- **Synchformer** (AudioSet-trained) → `metrics/av_quality/weights/synchformer_state_dict.pth` (from [a3s.fi](https://github.com/v-iashin/Synchformer), auto-converted to state_dict format)
- **VideoReward** checkpoint → `metrics/video_reward/checkpoints/` (from [KlingTeam/VideoReward](https://huggingface.co/KlingTeam/VideoReward))
- **LAION-CLAP** → `metrics/audio_is_clap/clap_ckpt/630k-audioset-fusion-best.pt` (from [Siyanl/clap-630k-audioset-fusion-best](https://huggingface.co/Siyanl/clap-630k-audioset-fusion-best))
- **SyncNet v2** → `models/wav2lip/evaluation/syncnet_python/data/syncnet_v2.model`
- **SFD face detector** → `models/wav2lip/evaluation/syncnet_python/detectors/s3fd/weights/sfd_face.pth`
- **ImageBind** → `$TORCH_HOME/hub/checkpoints/imagebind_huge.pth` (from `dl.fbaipublicfiles.com`)
- **DOVER** → `models/dover/pretrained_weights/DOVER.pth` (from [GitHub releases](https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth))
- **PANNs Cnn14 for av_bench** → `metrics/av_quality/pann_home/Cnn14_16k_mAP=0.438.pth` (16kHz, used by DeSync) and `Cnn14_mAP=0.431.pth` (32kHz)

Weights auto-downloaded on first use (by model code, requires network):
- **Qwen2-VL-2B-Instruct** (base model for VideoReward) — from [HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), or set `QWEN2VL_LOCAL_PATH`
- **Roberta-base** (for CLAP text branch) — from HuggingFace, or place at `models/roberta-base/`

Weights already included (committed to repo):
- **DNSMOS ONNX** models → `metrics/dnsmos/DNSMOS/` (4 ONNX files, ~4MB total)
- **Roberta-base tokenizer + config** → `models/roberta-base/` (~5MB; `pytorch_model.bin` excluded via `.gitignore`, will be auto-downloaded from HuggingFace on first use)

### 3. Run Evaluation

#### Basic Usage

Use `--input_dir` to specify the directory containing generated videos organized in the step/category/ layout:

```bash
bash scripts/run_eval.sh \
    --input_dir /path/to/experiment/eval_video_8s \
    --prompt_meta_json /path/to/benchmark.json \
    --video_second 8
```

Alternatively, specify `--checkpoint_dir` and `--video_save_path_subdir_name` separately:

```bash
bash scripts/run_eval.sh \
    --checkpoint_dir /path/to/experiment \
    --video_save_path_subdir_name eval_video_8s \
    --prompt_meta_json /path/to/benchmark.json \
    --video_second 8
```

Resume (skip already-completed metrics):

```bash
bash scripts/run_eval.sh \
    --input_dir /path/to/experiment/eval_video_8s \
    --prompt_meta_json /path/to/benchmark.json \
    --video_second 8 \
    --resume
```

> **Note:** This toolkit is evaluation-only. To generate videos with MOVA, see the [MOVA repository root](../).

### Evaluation Types

| `--eval_type` | Metrics Run |
|----------------|-------------|
| `eval` (default) | All 7 metric groups |
| `without_lip` | All except Lip Sync |
| `lip_only` | Lip Sync only |
| `amp_only` | Audio Amplitude + DNSMOS only |
| `video_only` | VideoReward only |
| `cpcer_only` | cpCER only |

### Offline Mode

By default, missing model weights are auto-downloaded from HuggingFace on first use. To force offline mode (no network access):

```bash
bash scripts/run_eval.sh --offline --input_dir /path/to/videos ...
# or set environment variable:
MOVA_EVAL_OFFLINE=1 bash scripts/run_eval.sh --input_dir /path/to/videos ...
```

## Input Data Format

### Checkpoint directory structure

```
checkpoint_dir/
├── video_save_path_subdir_name/
│   ├── checkpoint_name/
│   │   ├── category_a/
│   │   │   ├── video_001.mp4
│   │   │   └── video_002.mp4
│   │   └── category_b/
│   │       └── video_003.mp4
│   └── another_checkpoint/
│       └── ...
```

### Verse-Bench prompt format

We provide `verse_bench_prompts.json` in the repo root with all 600 items ready to use. It uses a nested JSON format mapping categories and item IDs to prompts:

```json
{
    "set1": {
        "00000000": {
            "prompt": "In a bright classroom, a smiling woman stands before a teal chalkboard...",
            "video_prompt": "The video depicts a woman in a classroom setting...",
            "audio_prompt": ["The sound of chalk writing."],
            "speech_prompt": {"speaker": "the woman", "text": "Attention Is All You Need."}
        }
    }
}
```

Only `prompt` is required by the evaluation pipeline. The other fields (`video_prompt`, `audio_prompt`, `speech_prompt`) are for video generation and not consumed by any evaluation metric.

```bash
bash scripts/run_eval.sh \
    --input_dir /path/to/your/eval_video_8s \
    --prompt_meta_json verse_bench_prompts.json \
    --video_second 8
```

For MyBench cpCER evaluation, we provide `metrics/cpcer/references.txt` containing the 27 multi-speaker reference transcripts. See [cpCER Metric](#cpcer-metric) for details.

## cpCER Metric

**cpCER** (Concatenated Permutation Character Error Rate) evaluates multi-speaker conversational accuracy by comparing ASR transcriptions against reference transcripts with Hungarian-matched speaker alignment. Lower is better. Only applicable to the 27 multi-speaker videos in MyBench.

### API Configuration

The script uses the `moss-transcribe-diarize` model on the studio.mosi.cn transcription API. Key configuration details:

- **Endpoint**: `https://studio.mosi.cn/api/v1/audio/transcriptions`
- **Model**: `moss-transcribe-diarize`
- **Prompt parameter**: Use `"prompt"` (not `"text"`) in the API request payload. The `"text"` field is treated as a Whisper-style initial prompt and does not trigger diarization, resulting in speaker-unaware transcriptions and inflated cpCER.
- **Response format**: The API returns structured JSON with `asr_transcription_result.segments` containing per-segment speaker labels and timestamps. The `full_text` field is plain text without speaker tags — the script reconstructs `[S01] [S02]` tagged text from `segments`.

Environment variables:
```bash
export CPCER_ASR_API_URL="https://studio.mosi.cn/api/v1/audio/transcriptions"  # default
export CPCER_ASR_API_KEY="sk-your-api-key-here"
export CPCER_ASR_MODEL="moss-transcribe-diarize"  # default
```

### Getting the API Key

cpCER requires **MOSS Transcribe Diarize** API access ([studio.mosi.cn](https://studio.mosi.cn)):

1. Register at [https://studio.mosi.cn](https://studio.mosi.cn) (phone or WeChat login)
2. Go to **API Keys** → **Create API Key**, copy the key
3. Configure:

   ```bash
   # Option 1: Environment variable
   export CPCER_ASR_API_KEY="sk-your-api-key-here"

   # Option 2: Command-line argument
   python metrics/cpcer/eval_cpcer.py \
       --video_dir /path/to/multi-speaker/videos \
       --references_txt metrics/cpcer/references.txt \
       --asr_api_key "sk-your-api-key-here"
   ```

### Running cpCER

```bash
# Standalone
python metrics/cpcer/eval_cpcer.py \
    --video_dir /path/to/multi-speaker/videos \
    --references_txt metrics/cpcer/references.txt \
    --asr_api_key YOUR_API_KEY

# Via run_eval.sh
bash scripts/run_eval.sh \
    --input_dir /path/to/experiment/eval_video_8s \
    --prompt_meta_json verse_bench_prompts.json \
    --video_second 8 \
    --eval_type cpcer_only \
    --references_txt metrics/cpcer/references.txt
```

Videos in `--video_dir` should have numeric prefixes (e.g., `001_xxx.mp4`) aligned with lines in `references.txt`. When using `run_eval.sh`, the default directory is `$INPUT_DIR/multi-speaker`.

## Third-Party Components

This toolkit includes code from the following open-source projects:

- **PyTorchVideo** (Meta) - Apache 2.0 - Video understanding models
- **LAION-CLAP** - MIT - Contrastive language-audio pretraining
- **ImageBind** (Meta) - CC-BY-NC-4.0 - Multi-modal embedding
- **Wav2Lip** - BSD-3-Clause - Lip sync generation and evaluation
- **DOVER** - Apache 2.0 - Video quality assessment
- **DNSMOS** - MIT - Deep noise suppression MOS
- **Synchformer** - Apache 2.0 - Audio-video synchronization

Please refer to each component's directory for specific license terms.

> **Note on licensing:** While this toolkit is released under Apache 2.0, the ImageBind component uses CC-BY-NC-4.0, which restricts commercial use. If you plan to use this toolkit commercially, you must exclude the ImageBind-based metric (IB-Score) or obtain a separate license for ImageBind.

## Citation

If you find our work helpful, please cite the MOVA paper and this evaluation toolkit:

```bibtex
@misc{openmoss_mova_2026,
  title         = {MOVA: Towards Scalable and Synchronized Video-Audio Generation},
  author        = {{SII-OpenMOSS Team} and Donghua Yu and Mingshu Chen and Qi Chen and Qi Luo and Qianyi Wu and Qinyuan Cheng and Ruixiao Li and Tianyi Liang and Wenbo Zhang and Wenming Tu and Xiangyu Peng and Yang Gao and Yanru Huo and Ying Zhu and Yinze Luo and Yiyang Zhang and Yuerong Song and Zhe Xu and Zhiyu Zhang and Chenchen Yang and Cheng Chang and Chushu Zhou and Hanfu Chen and Hongnan Ma and Jiaxi Li and Jingqi Tong and Junxi Liu and Ke Chen and Shimin Li and Songlin Wang and Wei Jiang and Zhaoye Fei and Zhiyuan Ning and Chunguo Li and Chenhui Li and Ziwei He and Zengfeng Huang and Xie Chen and Xipeng Qiu},
  year          = {2026},
  month         = feb,
  eprint        = {2602.08794},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2602.08794},
  url           = {https://arxiv.org/abs/2602.08794},
  note          = {Technical report. Corresponding authors: Xie Chen and Xipeng Qiu. Project leaders: Qinyuan Cheng and Tianyi Liang.}
}
```

This toolkit builds on the following papers. If you use specific metrics, please also cite the corresponding work:

```bibtex
@inproceedings{girdhar2023imagebind,
  title     = {ImageBind: One Embedding Space To Bind Them All},
  author    = {Rohit Girdhar and Alaaeldin El-Nouby and Zhuang Liu and Mannat Singh and Kalyan Vasudev Alwala and Armand Joulin and Ishan Misra},
  booktitle = {CVPR},
  year      = {2023}
}

@inproceedings{kong2020panns,
  title     = {PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition},
  author    = {Qiuqiang Kong and Yin Cao and Turab Iqbal and Yuxuan Wang and Wenwu Wang and Mark D. Plumbley},
  booktitle = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year      = {2020}
}

@inproceedings{iashin2024synchformer,
  title     = {Synchformer: Efficient Synchronization from Sparse Cues},
  author    = {Vladimir Iashin and Weidi Xie and Esa Rahtu and Andrew Zisserman},
  booktitle = {ICASSP},
  year      = {2024}
}

@article{wu2023clap,
  title   = {Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author  = {Yusong Wu and Ke Chen and Tianyu Zhang and Yuchen Hui and Marianna Nezhurina and Taylor Berg-Kirkpatrick and Shlomo Dubnov},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2023},
  volume  = {31},
  pages   = {3254--3268}
}

@inproceedings{prajwal2020wav2lip,
  title     = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
  author    = {K R Prajwal and Rudrabha Mukhopadhyay and Vinay P. Namboodiri and C. V. Jawahar},
  booktitle = {ACM MM},
  year      = {2020}
}

@inproceedings{wu2023dover,
  title     = {Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives},
  author    = {Haoning Wu and Erli Zhang and Liang Liao and Chaofeng Chen and Jingwen Hou and Annan Wang and Wenxiu Sun and Qiong Yan and Weisi Lin},
  booktitle = {ICCV},
  year      = {2023}
}

@inproceedings{reddy2022dnsmos,
  title     = {DNSMOS: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors},
  author    = {Chandan K. A. Reddy and Vishak Gopal and Ross Cutler},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2022}
}

@article{liu2025videoreward,
  title   = {Improving Video Generation with Human Feedback},
  author  = {Jie Liu and Gongye Liu and Jiajun Liang and Ziyang Yuan and Xiaokun Liu and Mingwu Zheng and Xiele Wu and Qiulin Wang and Menghan Xia and Xintao Wang and Xiaohong Liu and Fei Yang and Pengfei Wan and Di Zhang and Kun Gai and Yujiu Yang and Wanli Ouyang},
  journal = {arXiv preprint arXiv:2501.13918},
  year    = {2025}
}

@article{mosstranscribe2025,
  title   = {MOSS Transcribe Diarize Technical Report},
  author  = {Donghua Yu and Zhengyuan Lin and Chen Yang and Yiyang Zhang and Hanfu Chen and Jingqi Chen and Ke Chen and Liwei Fan and Yi Jiang and Jie Zhu and Muchen Li and Wenxuan Wang and Yang Wang and Zhe Xu and Yitian Gong and Yuqian Zhang and Wenbo Zhang and Songlin Wang and Zhiyu Wu and Zhaoye Fei and Qinyuan Cheng and Shimin Li and Xipeng Qiu},
  journal = {arXiv preprint arXiv:2601.01554},
  year    = {2025}
}
```

## License

This evaluation toolkit is released under the Apache 2.0 License. Individual components may have their own licenses - see above.
