#!/bin/bash
# ======================================
# Download Pretrained Weights for MOVA Eval
# ======================================
# This script downloads all required pretrained model checkpoints.
# By default, weights are placed in the same directory as each
# evaluation script (which is where the scripts look first).
#
# Usage:
#   bash scripts/download_weights.sh
#   bash scripts/download_weights.sh --video_reward_only
# ======================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DOWNLOAD_ALL=1
DOWNLOAD_VIDEO_REWARD=0
DOWNLOAD_AV_QUALITY=0
DOWNLOAD_AUDIO_IS_CLAP=0
DOWNLOAD_LIP_SYNC=0
DOWNLOAD_DNSMOS=0
DOWNLOAD_DOVER=0

if [ $# -eq 0 ]; then
    echo "Downloading all weights..."
else
    DOWNLOAD_ALL=0
    for arg in "$@"; do
        case "$arg" in
            --video_reward_only)  DOWNLOAD_VIDEO_REWARD=1 ;;
            --av_quality_only)    DOWNLOAD_AV_QUALITY=1 ;;
            --audio_is_clap_only) DOWNLOAD_AUDIO_IS_CLAP=1 ;;
            --lip_sync_only)      DOWNLOAD_LIP_SYNC=1 ;;
            --dnsmos_only)        DOWNLOAD_DNSMOS=1 ;;  # ONNX models already included, but allow explicit request
            --dover_only)         DOWNLOAD_DOVER=1 ;;
            --all)                DOWNLOAD_ALL=1 ;;
            *)                    echo "Unknown option: $arg"; exit 1 ;;
        esac
    done
fi

if [ "$DOWNLOAD_ALL" -eq 1 ]; then
    DOWNLOAD_VIDEO_REWARD=1
    DOWNLOAD_AV_QUALITY=1
    DOWNLOAD_AUDIO_IS_CLAP=1
    DOWNLOAD_LIP_SYNC=1
    DOWNLOAD_DOVER=1
fi

echo "========================================"
echo "MOVA Eval - Download Pretrained Weights"
echo "========================================"

# ----- VideoReward -----
if [ "$DOWNLOAD_VIDEO_REWARD" -eq 1 ]; then
    echo ""
    echo ">>> Downloading VideoReward model weights..."
    CKPT_DIR="$REPO_ROOT/metrics/video_reward/checkpoints"
    mkdir -p "$CKPT_DIR/checkpoint-11352"

    # Auto-download from HuggingFace (KlingTeam/VideoReward)
    MODEL_FILE="$CKPT_DIR/checkpoint-11352/model.pth"
    CONFIG_FILE="$CKPT_DIR/model_config.json"
    if [ ! -f "$MODEL_FILE" ]; then
        echo "Downloading VideoReward checkpoint from HuggingFace (KlingTeam/VideoReward)..."
        pip install -q huggingface_hub 2>/dev/null
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='KlingTeam/VideoReward',
    local_dir='$CKPT_DIR',
    repo_type='model',
    allow_patterns=['model_config.json', 'checkpoint-11352/*'],
)
print('Download complete.')
" || echo "  Download failed. Please download manually from https://huggingface.co/KlingTeam/VideoReward"
    else
        echo "VideoReward checkpoint already exists: $MODEL_FILE"
    fi

    # Qwen2-VL-2B-Instruct (required base model, auto-downloaded by transformers if not present)
    echo ""
    echo "Qwen2-VL-2B-Instruct (base model for VideoReward):"
    echo "  Set QWEN2VL_LOCAL_PATH to a local directory, or it will be auto-downloaded"
    echo "  from HuggingFace (Qwen/Qwen2-VL-2B-Instruct) on first use."
    echo "  To pre-download: huggingface-cli download Qwen/Qwen2-VL-2B-Instruct"
fi

# ----- AV Quality (Synchformer + ImageBind) -----
if [ "$DOWNLOAD_AV_QUALITY" -eq 1 ]; then
    echo ""
    echo ">>> Downloading AV Quality model weights..."
    WEIGHTS_DIR="$REPO_ROOT/metrics/av_quality/weights"
    mkdir -p "$WEIGHTS_DIR"

    # Synchformer weights (download from official a3s.fi, then extract state_dict)
    SYNC_FILE="$WEIGHTS_DIR/synchformer_state_dict.pth"
    if [ ! -f "$SYNC_FILE" ]; then
        echo "Downloading Synchformer checkpoint (AudioSet-trained, from a3s.fi)..."
        SYNC_RAW="$WEIGHTS_DIR/24-01-04T16-39-21.pt"
        if [ ! -f "$SYNC_RAW" ]; then
            wget -q "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt" \
                -O "$SYNC_RAW" || {
                echo "  Download failed. Please download manually from:"
                echo "  https://github.com/v-iashin/Synchformer"
                echo "  and convert to synchformer_state_dict.pth (see README)"
                rm -f "$SYNC_RAW"
            }
        fi
        if [ -f "$SYNC_RAW" ]; then
            echo "Converting checkpoint to state_dict format..."
            python3 -c "
import torch
ckpt = torch.load('$SYNC_RAW', map_location='cpu', weights_only=False)
sd = ckpt['state_dict']
# Strip 'module.' prefix from DataParallel keys
clean_sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
torch.save(clean_sd, '$SYNC_FILE')
print('Converted and saved to $SYNC_FILE')
" && rm -f "$SYNC_RAW" || echo "  Conversion failed. Please convert manually."
        fi
    else
        echo "Synchformer weights already exist: $SYNC_FILE"
    fi

    # ImageBind (auto-downloaded by torch.hub on first use, or download here)
    echo ""
    IB_CACHE_DIR="${TORCH_HOME:-$HOME/.cache/torch}/checkpoints"
    IB_FILE="$IB_CACHE_DIR/imagebind_huge.pth"
    if [ ! -f "$IB_FILE" ]; then
        echo "Downloading ImageBind weights..."
        mkdir -p "$IB_CACHE_DIR"
        wget -q "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth" \
            -O "$IB_FILE" || echo "  Download failed. Will be auto-downloaded on first use."
    else
        echo "ImageBind weights already exist: $IB_FILE"
    fi

    # PANNs Cnn14 for av_bench (used by AV Quality extraction_models with sample_rate=16000)
    # Code looks at metrics/av_quality/pann_home/ (../../pann_home relative to av_bench/panns/)
    AV_PANN_DIR="$REPO_ROOT/metrics/av_quality/pann_home"
    AV_PANN_FILE_16K="$AV_PANN_DIR/Cnn14_16k_mAP=0.438.pth"
    AV_PANN_FILE_32K="$AV_PANN_DIR/Cnn14_mAP=0.431.pth"
    mkdir -p "$AV_PANN_DIR"
    if [ ! -f "$AV_PANN_FILE_16K" ]; then
        echo "Downloading PANNs Cnn14 16kHz checkpoint for av_bench..."
        wget -q "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth" \
            -O "$AV_PANN_FILE_16K" || echo "  Download failed. Cnn14 16k for av_bench not available."
    else
        echo "PANNs Cnn14 16kHz for av_bench already exists: $AV_PANN_FILE_16K"
    fi
    if [ ! -f "$AV_PANN_FILE_32K" ]; then
        echo "Downloading PANNs Cnn14 32kHz checkpoint for av_bench..."
        wget -q "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth" \
            -O "$AV_PANN_FILE_32K" || echo "  Download failed. Cnn14 32k for av_bench not available."
    else
        echo "PANNs Cnn14 32kHz for av_bench already exists: $AV_PANN_FILE_32K"
    fi
fi

# ----- Audio IS + CLAP -----
if [ "$DOWNLOAD_AUDIO_IS_CLAP" -eq 1 ]; then
    echo ""
    echo ">>> Downloading Audio IS + CLAP model weights..."
    PANN_DIR="$REPO_ROOT/metrics/audio_is_clap/pann_home"
    CLAP_DIR="$REPO_ROOT/metrics/audio_is_clap/clap_ckpt"
    ROBERTA_DIR="$REPO_ROOT/models/roberta-base"
    mkdir -p "$PANN_DIR" "$CLAP_DIR" "$ROBERTA_DIR"

    # PANNs Cnn14 checkpoint
    PANN_FILE="$PANN_DIR/Cnn14_mAP=0.431.pth"
    if [ ! -f "$PANN_FILE" ]; then
        echo "Downloading PANNs Cnn14 checkpoint..."
        wget -q "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth" \
            -O "$PANN_FILE" || echo "  Download failed. Please download manually from https://zenodo.org/record/3987831"
    else
        echo "PANNs Cnn14 checkpoint already exists: $PANN_FILE"
    fi
    echo ""

    # CLAP checkpoint
    CLAP_FILE="$CLAP_DIR/630k-audioset-fusion-best.pt"
    if [ ! -f "$CLAP_FILE" ]; then
        echo "Downloading LAION-CLAP checkpoint..."
        # Try Siyanl/clap-630k-audioset-fusion-best (public mirror)
        pip install -q huggingface_hub 2>/dev/null
        python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='Siyanl/clap-630k-audioset-fusion-best',
    filename='pytorch_model.bin',
    local_dir='$CLAP_DIR',
)
import shutil, os
# Rename to expected filename
shutil.move(path, '$CLAP_FILE')
print('Download complete.')
" || echo "  Download failed. Please download manually from https://huggingface.co/Siyanl/clap-630k-audioset-fusion-best"
    else
        echo "CLAP checkpoint already exists: $CLAP_FILE"
    fi
    echo ""

    # Roberta-base (auto-downloaded from HuggingFace on first use, or use local)
    echo "Roberta-base tokenizer/config:"
    echo "  Will be auto-downloaded from HuggingFace on first use."
    echo "  Or place manually at: $ROBERTA_DIR/"
fi

# ----- Lip Sync (SyncNet + SFD) -----
if [ "$DOWNLOAD_LIP_SYNC" -eq 1 ]; then
    echo ""
    echo ">>> Downloading Lip Sync model weights..."
    SYNCNET_DIR="$REPO_ROOT/models/wav2lip/evaluation/syncnet_python/data"
    SFD_DIR="$REPO_ROOT/models/wav2lip/evaluation/syncnet_python/detectors/s3fd/weights"
    mkdir -p "$SYNCNET_DIR" "$SFD_DIR"

    # SyncNet v2 model (from VGG/Oxford)
    SYNCNET_FILE="$SYNCNET_DIR/syncnet_v2.model"
    if [ ! -f "$SYNCNET_FILE" ]; then
        echo "Downloading SyncNet v2 model..."
        wget -q "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model" \
            -O "$SYNCNET_FILE" || echo "  Download failed. Please download manually from https://github.com/joonson/syncnet_python"
    else
        echo "SyncNet v2 model already exists: $SYNCNET_FILE"
    fi
    echo ""

    # SFD face detector
    SFD_FILE="$SFD_DIR/sfd_face.pth"
    if [ ! -f "$SFD_FILE" ]; then
        echo "Downloading SFD face detector..."
        wget -q "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth" \
            -O "$SFD_FILE" || wget -q "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
            -O "$SFD_FILE" || echo "  Download failed. Please download manually."
    else
        echo "SFD face detector already exists: $SFD_FILE"
    fi

    # Example video for SyncNet testing
    EXAMPLE_FILE="$SYNCNET_DIR/example.avi"
    if [ ! -f "$EXAMPLE_FILE" ]; then
        wget -q "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi" \
            -O "$EXAMPLE_FILE" 2>/dev/null || true
    fi
fi

# ----- DOVER -----
if [ "$DOWNLOAD_DOVER" -eq 1 ]; then
echo ""
echo ">>> DOVER pretrained weights..."
DOVER_DIR="$REPO_ROOT/models/dover/pretrained_weights"
mkdir -p "$DOVER_DIR"

DOVER_FILE="$DOVER_DIR/DOVER.pth"
if [ ! -f "$DOVER_FILE" ]; then
    echo "Downloading DOVER weights..."
    wget -q "https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth" \
        -O "$DOVER_FILE" || echo "  Download failed. Please download manually from https://github.com/QualityAssessment/DOVER"
else
    echo "DOVER weights already exist: $DOVER_FILE"
fi
fi  # DOWNLOAD_DOVER

echo ""
echo "========================================"
echo "Download instructions complete."
echo "See README.md for more details."
echo "========================================"
