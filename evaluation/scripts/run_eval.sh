#!/bin/bash

# ======================================
# MOVA Evaluation Pipeline
# ======================================
# Evaluates generated videos across multiple dimensions:
#   1. Audio Amplitude & Loudness (RMS, LUFS)
#   2. DNSMOS Audio Quality (P808 MOS)
#   3. AV Quality (DeSync, IB-Score, AV-Align)
#   4. VideoReward (VQ, MQ, TA, Overall)
#   5. Audio IS + CLAP Score
#   6. Lip Sync (LSE-D, LSE-C)
#   7. cpCER (multi-speaker conversational accuracy)
#
# Usage:
#   bash run_eval.sh \
#     --input_dir /path/to/experiment/eval_video_8s \
#     --prompt_meta_json /path/to/benchmark.json \
#     --video_second 8
#
# Options:
#   --input_dir <path>            Direct path to eval video directory (shorthand)
#   --eval_type eval              eval / without_lip / lip_only / amp_only / video_only
#   --resume                      Skip already-completed metrics
#   --offline                     Force offline mode (no HuggingFace downloads)
#   -h, --help                    Show this help
#
# Environment variables (override defaults):
#   PYTHON_VIDEO_REWARD           Python for VideoReward metric
#   PYTHON_AV_QUALITY             Python for AV Quality + Lip Sync
#   PYTHON_AUDIO_QUALITY          Python for Audio IS + CLAP
#   PYTHON_DNSMOS                 Python for DNSMOS + Audio Amplitude
#   QWEN2VL_LOCAL_PATH            Local path to Qwen2-VL-2B-Instruct
#   TORCH_HOME                    Root dir for torch model caches (ImageBind, etc.)
# ======================================

# Track failures across metrics
FAILED_METRICS=""

# ----- Resolve this script's directory -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ----- Default Python interpreters -----
# Each metric group can use a separate conda/virtual environment.
# Set PYTHON_* environment variables to override, or source scripts/set_env.sh.
PYTHON_VIDEO_REWARD="${PYTHON_VIDEO_REWARD:-python}"
PYTHON_AV_QUALITY="${PYTHON_AV_QUALITY:-python}"
PYTHON_AUDIO_QUALITY="${PYTHON_AUDIO_QUALITY:-python}"
PYTHON_DNSMOS="${PYTHON_DNSMOS:-python}"
PYTHON_AUDIO_AMPLITUDE="${PYTHON_AUDIO_AMPLITUDE:-$PYTHON_DNSMOS}"
PYTHON_LIP_QUALITY="${PYTHON_LIP_QUALITY:-$PYTHON_AV_QUALITY}"
PYTHON_CPCER="${PYTHON_CPCER:-python}"

# ----- Script paths -----
SCRIPT_VIDEO_REWARD="$REPO_ROOT/metrics/video_reward/eval_video_reward.py"
SCRIPT_AV_QUALITY="$REPO_ROOT/metrics/av_quality/eval_av_quality.py"
SCRIPT_AUDIO_QUALITY="$REPO_ROOT/metrics/audio_is_clap/eval_audio_is_clap.py"
SCRIPT_LIP_QUALITY="$REPO_ROOT/models/wav2lip/evaluation/syncnet_python/eval_lip_sync.py"
SCRIPT_DNSMOS="$REPO_ROOT/metrics/dnsmos/eval_dnsmos.py"
SCRIPT_AUDIO_AMPLITUDE="$REPO_ROOT/metrics/audio_amplitude/eval_audio_amplitude.py"
SCRIPT_CPCER="$REPO_ROOT/metrics/cpcer/eval_cpcer.py"

# ----- Required parameters -----
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
VIDEO_SAVE_PATH_SUBDIR_NAME="${VIDEO_SAVE_PATH_SUBDIR_NAME:-}"
PROMPT_META_JSON="${PROMPT_META_JSON:-}"
VIDEO_SECOND="${VIDEO_SECOND:-8}"

# ----- Optional parameters -----
INPUT_DIR="${INPUT_DIR:-}"
EVAL_TYPE="${EVAL_TYPE:-eval}"
RESUME="${RESUME:-}"
VIDEO_REWARD_FPS="${VIDEO_REWARD_FPS:-2}"
REFERENCES_TXT="${REFERENCES_TXT:-}"
VIDEO_DIR_CPCER="${VIDEO_DIR_CPCER:-}"
ASR_API_KEY="${ASR_API_KEY:-}"

# ----- Parse command-line arguments -----
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint_dir=*)   CHECKPOINT_DIR="${1#*=}"; shift ;;
    --checkpoint_dir)     CHECKPOINT_DIR="$2"; shift 2 ;;
    --video_save_path_subdir_name=*) VIDEO_SAVE_PATH_SUBDIR_NAME="${1#*=}"; shift ;;
    --video_save_path_subdir_name)   VIDEO_SAVE_PATH_SUBDIR_NAME="$2"; shift 2 ;;
    --prompt_meta_json=*) PROMPT_META_JSON="${1#*=}"; shift ;;
    --prompt_meta_json)   PROMPT_META_JSON="$2"; shift 2 ;;
    --video_second=*)     VIDEO_SECOND="${1#*=}"; shift ;;
    --video_second)       VIDEO_SECOND="$2"; shift 2 ;;
    --input_dir=*)        INPUT_DIR="${1#*=}"; shift ;;
    --input_dir)          INPUT_DIR="$2"; shift 2 ;;
    --references_txt=*)  REFERENCES_TXT="${1#*=}"; shift ;;
    --references_txt)    REFERENCES_TXT="$2"; shift 2 ;;
    --video_dir_cpcer=*) VIDEO_DIR_CPCER="${1#*=}"; shift ;;
    --video_dir_cpcer)   VIDEO_DIR_CPCER="$2"; shift 2 ;;
    --asr_api_key=*)     ASR_API_KEY="${1#*=}"; shift ;;
    --asr_api_key)       ASR_API_KEY="$2"; shift 2 ;;
    --eval_type=*)        EVAL_TYPE="${1#*=}"; shift ;;
    --eval_type)          EVAL_TYPE="$2"; shift 2 ;;
    --resume)             RESUME=1; shift ;;
    --video_reward_fps=*) VIDEO_REWARD_FPS="${1#*=}"; shift ;;
    --offline)            MOVA_EVAL_OFFLINE=1; shift ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Required:"
      echo "  --prompt_meta_json <path>              Nested benchmark prompt JSON"
      echo "  --video_second <int>                   Video duration in seconds (default: 8)"
      echo ""
      echo "Input (one of):"
      echo "  --input_dir <path>                     Direct path to eval video directory"
      echo "  --checkpoint_dir + --video_save_path_subdir_name   Combined = eval video dir"
      echo ""
      echo "Optional:"
      echo "  --eval_type <type>                     eval / without_lip / lip_only / amp_only / video_only / cpcer_only"
      echo "  --references_txt <path>                Path to references.txt for cpCER (one reference per line)"
      echo "  --video_dir_cpcer <path>               Path to multi-speaker video directory for cpCER (default: input_dir/multi-speaker)"
      echo "  --video_reward_fps <fps>               FPS for VideoReward sampling (default: 2)"
      echo "  --resume                               Skip already-completed metrics"
      echo "  --offline                              Force offline mode"
      echo "  -h, --help                             Show this help"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ----- Resolve input path -----
# --input_dir is a shorthand: input_dir = checkpoint_dir/video_save_path_subdir_name
if [ -n "$INPUT_DIR" ]; then
    CHECKPOINT_DIR="$(dirname "$INPUT_DIR")"
    VIDEO_SAVE_PATH_SUBDIR_NAME="$(basename "$INPUT_DIR")"
fi

# ----- Validate required parameters -----
if [ -z "$CHECKPOINT_DIR" ] || [ -z "$VIDEO_SAVE_PATH_SUBDIR_NAME" ] || [ -z "$PROMPT_META_JSON" ]; then
    echo "Error: --prompt_meta_json is required, and either --input_dir or both" >&2
    echo "       --checkpoint_dir and --video_save_path_subdir_name must be provided." >&2
    exit 1
fi

if [ ! -f "$PROMPT_META_JSON" ]; then
    echo "Error: prompt_meta_json not found: $PROMPT_META_JSON" >&2
    exit 1
fi

# Set ulimit for many open files
ulimit -n 65535 2>/dev/null || true

# Offline mode
MOVA_EVAL_OFFLINE="${MOVA_EVAL_OFFLINE:-0}"
if [ "$MOVA_EVAL_OFFLINE" -eq 1 ]; then
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
fi

# Local Qwen2-VL model path for VideoReward
if [ -z "$QWEN2VL_LOCAL_PATH" ]; then
    _qwen_candidates=(
        "$REPO_ROOT/models/Qwen2-VL-2B-Instruct"
    )
    for _c in "${_qwen_candidates[@]}"; do
        if [ -d "$_c" ]; then
            export QWEN2VL_LOCAL_PATH="$_c"
            break
        fi
    done
fi

# ----- Define which metrics to run based on EVAL_TYPE -----
RUN_AMPLITUDE=0
RUN_DNSMOS=0
RUN_AV_QUALITY=0
RUN_REWARD=0
RUN_AUDIO_IS_CLAP=0
RUN_LIP=0
RUN_CPCER=0

case "$EVAL_TYPE" in
    eval)
        RUN_AMPLITUDE=1; RUN_DNSMOS=1; RUN_AV_QUALITY=1; RUN_REWARD=1; RUN_AUDIO_IS_CLAP=1; RUN_LIP=1; RUN_CPCER=1
        ;;
    without_lip)
        RUN_AMPLITUDE=1; RUN_DNSMOS=1; RUN_AV_QUALITY=1; RUN_REWARD=1; RUN_AUDIO_IS_CLAP=1; RUN_CPCER=1
        ;;
    lip_only)
        RUN_LIP=1
        ;;
    amp_only)
        RUN_AMPLITUDE=1; RUN_DNSMOS=1
        ;;
    video_only)
        RUN_REWARD=1
        ;;
    cpcer_only)
        RUN_CPCER=1
        ;;
    *)
        echo "Error: Invalid EVAL_TYPE: $EVAL_TYPE" >&2
        echo "Supported: eval / without_lip / lip_only / amp_only / video_only" >&2
        exit 1
        ;;
esac

# ----- Build common args -----
EXP_ROOT="${CHECKPOINT_DIR%/}"

# ===== Print config =====
echo "========================================"
echo "MOVA Evaluation Pipeline"
echo "========================================"
echo "Checkpoint:      $CHECKPOINT_DIR"
echo "Video subdir:    $VIDEO_SAVE_PATH_SUBDIR_NAME"
echo "Prompt JSON:     $PROMPT_META_JSON"
echo "Video seconds:   $VIDEO_SECOND"
echo "Eval type:       $EVAL_TYPE"
echo "Input dir:       ${INPUT_DIR:-$CHECKPOINT_DIR/$VIDEO_SAVE_PATH_SUBDIR_NAME}"
echo "========================================"

# ===== 1. Audio Amplitude & Loudness =====
if [ "$RUN_AMPLITUDE" -eq 1 ]; then
    if [ -n "$RESUME" ]; then
        _done="$EXP_ROOT/$VIDEO_SAVE_PATH_SUBDIR_NAME/eval_audio_amplitude_scores_means.json"
        if [ -f "$_done" ]; then
            echo "[RESUME] Audio Amplitude already done, skipping."
            RUN_AMPLITUDE=0
        fi
    fi
fi

if [ "$RUN_AMPLITUDE" -eq 1 ]; then
    if [ ! -f "$SCRIPT_AUDIO_AMPLITUDE" ]; then
        echo "[SKIP] Audio Amplitude script not found: $SCRIPT_AUDIO_AMPLITUDE"
        FAILED_METRICS="$FAILED_METRICS AudioAmplitude(script_missing)"
    else
        echo ""
        echo "======== Running: Audio Amplitude & Loudness ========"
        if $PYTHON_AUDIO_AMPLITUDE "$SCRIPT_AUDIO_AMPLITUDE" \
                -i "$EXP_ROOT" \
                --prompt_meta_json "$PROMPT_META_JSON" \
                --video_save_path_subdir_name "$VIDEO_SAVE_PATH_SUBDIR_NAME"; then
            echo "  [OK] Audio Amplitude & Loudness"
        else
            echo "  [FAILED] Audio Amplitude & Loudness"
            FAILED_METRICS="$FAILED_METRICS AudioAmplitude"
        fi
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
fi

# ===== 2. DNSMOS =====
if [ "$RUN_DNSMOS" -eq 1 ]; then
    if [ -n "$RESUME" ]; then
        _done="$EXP_ROOT/$VIDEO_SAVE_PATH_SUBDIR_NAME/eval_dnsmos_scores_means.json"
        if [ -f "$_done" ]; then
            echo "[RESUME] DNSMOS already done, skipping."
            RUN_DNSMOS=0
        fi
    fi
fi

if [ "$RUN_DNSMOS" -eq 1 ]; then
    if [ ! -f "$SCRIPT_DNSMOS" ]; then
        echo "[SKIP] DNSMOS script not found: $SCRIPT_DNSMOS"
        FAILED_METRICS="$FAILED_METRICS DNSMOS(script_missing)"
    else
        echo ""
        echo "======== Running: DNSMOS (P808 MOS) ========"
        if $PYTHON_DNSMOS "$SCRIPT_DNSMOS" \
                -i "$EXP_ROOT" \
                --prompt_meta_json "$PROMPT_META_JSON" \
                --video_save_path_subdir_name "$VIDEO_SAVE_PATH_SUBDIR_NAME"; then
            echo "  [OK] DNSMOS"
        else
            echo "  [FAILED] DNSMOS"
            FAILED_METRICS="$FAILED_METRICS DNSMOS"
        fi
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
fi

# ===== 3. AV Quality (DeSync, IB-Score, AV-Align) =====
if [ "$RUN_AV_QUALITY" -eq 1 ]; then
    if [ -n "$RESUME" ]; then
        _done="$EXP_ROOT/$VIDEO_SAVE_PATH_SUBDIR_NAME/eval_av_scores_means.json"
        if [ -f "$_done" ]; then
            echo "[RESUME] AV Quality already done, skipping."
            RUN_AV_QUALITY=0
        fi
    fi
fi

if [ "$RUN_AV_QUALITY" -eq 1 ]; then
    if [ ! -f "$SCRIPT_AV_QUALITY" ]; then
        echo "[SKIP] AV Quality script not found: $SCRIPT_AV_QUALITY"
        FAILED_METRICS="$FAILED_METRICS AVQuality(script_missing)"
    else
        echo ""
        echo "======== Running: AV Quality (DeSync, IB-Score, AV-Align) ========"
        _dover_opt="$REPO_ROOT/models/dover/dover.yml"
        export PYTHONPATH="$REPO_ROOT/metrics/av_quality:$REPO_ROOT/models/imagebind:$REPO_ROOT/models/pytorchvideo:$REPO_ROOT/models/dover"
        # TORCH_HOME: search candidate dirs for ImageBind weights
        if [ -z "$TORCH_HOME" ]; then
            _torch_home_candidates=(
                "$REPO_ROOT/models/imagebind"
                "$HOME/.cache/torch"
            )
            for _c in "${_torch_home_candidates[@]}"; do
                if [ -f "$_c/checkpoints/imagebind_huge.pth" ]; then
                    export TORCH_HOME="$_c"
                    break
                fi
            done
        fi
        if $PYTHON_AV_QUALITY "$SCRIPT_AV_QUALITY" \
                -i "$EXP_ROOT" \
                --prompt_meta_json "$PROMPT_META_JSON" \
                --video_save_path_subdir_name "$VIDEO_SAVE_PATH_SUBDIR_NAME" \
                --dover_opt "$_dover_opt" \
                --duration "$(python3 -c "print(${VIDEO_SECOND} - 0.1)")"; then
            echo "  [OK] AV Quality"
        else
            echo "  [FAILED] AV Quality"
            FAILED_METRICS="$FAILED_METRICS AVQuality"
        fi
        unset PYTHONPATH
        unset TORCH_HOME
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
fi

# ===== 4. VideoReward (VQ, MQ, TA, Overall) =====
if [ "$RUN_REWARD" -eq 1 ]; then
    if [ -n "$RESUME" ]; then
        _done="$EXP_ROOT/$VIDEO_SAVE_PATH_SUBDIR_NAME/eval_video_reward_results.json"
        if [ -f "$_done" ]; then
            echo "[RESUME] VideoReward already done, skipping."
            RUN_REWARD=0
        fi
    fi
fi

if [ "$RUN_REWARD" -eq 1 ]; then
    if [ ! -f "$SCRIPT_VIDEO_REWARD" ]; then
        echo "[SKIP] VideoReward script not found: $SCRIPT_VIDEO_REWARD"
        FAILED_METRICS="$FAILED_METRICS VideoReward(script_missing)"
    else
        echo ""
        echo "======== Running: VideoReward (VQ, MQ, TA) ========"
        _reward_model="$REPO_ROOT/metrics/video_reward/checkpoints"
        if $PYTHON_VIDEO_REWARD "$SCRIPT_VIDEO_REWARD" \
                -i "$EXP_ROOT" \
                --prompt_meta_json "$PROMPT_META_JSON" \
                --video_save_path_subdir_name "$VIDEO_SAVE_PATH_SUBDIR_NAME" \
                --reward_model "$_reward_model" \
                --fps "$VIDEO_REWARD_FPS"; then
            echo "  [OK] VideoReward"
        else
            echo "  [FAILED] VideoReward"
            FAILED_METRICS="$FAILED_METRICS VideoReward"
        fi
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
fi

# ===== 5. Audio IS + CLAP =====
if [ "$RUN_AUDIO_IS_CLAP" -eq 1 ]; then
    if [ -n "$RESUME" ]; then
        _done="$EXP_ROOT/$VIDEO_SAVE_PATH_SUBDIR_NAME/eval_audio_scores_means.json"
        if [ -f "$_done" ]; then
            echo "[RESUME] Audio IS + CLAP already done, skipping."
            RUN_AUDIO_IS_CLAP=0
        fi
    fi
fi

if [ "$RUN_AUDIO_IS_CLAP" -eq 1 ]; then
    if [ ! -f "$SCRIPT_AUDIO_QUALITY" ]; then
        echo "[SKIP] Audio IS + CLAP script not found: $SCRIPT_AUDIO_QUALITY"
        FAILED_METRICS="$FAILED_METRICS AudioISCLAP(script_missing)"
    else
        echo ""
        echo "======== Running: Audio IS + CLAP ========"
        export PYTHONPATH="$REPO_ROOT/metrics/audio_is_clap:$REPO_ROOT/models/clap/src:$REPO_ROOT/models/clap/src/laion_clap"
        if $PYTHON_AUDIO_QUALITY "$SCRIPT_AUDIO_QUALITY" \
                -i "$EXP_ROOT" \
                --prompt_meta_json "$PROMPT_META_JSON" \
                --video_save_path_subdir_name "$VIDEO_SAVE_PATH_SUBDIR_NAME"; then
            echo "  [OK] Audio IS + CLAP"
        else
            echo "  [FAILED] Audio IS + CLAP"
            FAILED_METRICS="$FAILED_METRICS AudioISCLAP"
        fi
        unset PYTHONPATH
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
fi

# ===== 6. Lip Sync (LSE-D, LSE-C) =====
if [ "$RUN_LIP" -eq 1 ]; then
    if [ -n "$RESUME" ]; then
        _done="$EXP_ROOT/$VIDEO_SAVE_PATH_SUBDIR_NAME/eval_lip_scores_means.json"
        if [ -f "$_done" ]; then
            echo "[RESUME] Lip Sync already done, skipping."
            RUN_LIP=0
        fi
    fi
fi

if [ "$RUN_LIP" -eq 1 ]; then
    if [ ! -f "$SCRIPT_LIP_QUALITY" ]; then
        echo "[SKIP] Lip Sync script not found: $SCRIPT_LIP_QUALITY"
        FAILED_METRICS="$FAILED_METRICS LipSync(script_missing)"
    else
        echo ""
        echo "======== Running: Lip Sync (LSE-D, LSE-C) ========"
        if $PYTHON_LIP_QUALITY "$SCRIPT_LIP_QUALITY" \
                -i "$EXP_ROOT" \
                --prompt_meta_json "$PROMPT_META_JSON" \
                --video_save_path_subdir_name "$VIDEO_SAVE_PATH_SUBDIR_NAME"; then
            echo "  [OK] Lip Sync"
        else
            echo "  [FAILED] Lip Sync"
            FAILED_METRICS="$FAILED_METRICS LipSync"
        fi
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
fi

# ===== 7. cpCER (multi-speaker conversational accuracy) =====
if [ "$RUN_CPCER" -eq 1 ]; then
    if [ -n "$RESUME" ]; then
        _done="$EXP_ROOT/$VIDEO_SAVE_PATH_SUBDIR_NAME/eval_cpcer_scores_means.json"
        if [ -f "$_done" ]; then
            echo "[RESUME] cpCER already done, skipping."
            RUN_CPCER=0
        fi
    fi
fi

if [ "$RUN_CPCER" -eq 1 ]; then
    if [ ! -f "$SCRIPT_CPCER" ]; then
        echo "[SKIP] cpCER script not found: $SCRIPT_CPCER"
        FAILED_METRICS="$FAILED_METRICS cpCER(script_missing)"
    elif [ -z "$REFERENCES_TXT" ]; then
        echo "[SKIP] cpCER: --references_txt is required"
        FAILED_METRICS="$FAILED_METRICS cpCER(no_references_txt)"
    else
        # Resolve video directory: explicit --video_dir_cpcer takes precedence,
        # otherwise search for multi-speaker subdir
        _cpcer_video_dir="${VIDEO_DIR_CPCER:-}"
        if [ -z "$_cpcer_video_dir" ]; then
            _base="${INPUT_DIR:-$CHECKPOINT_DIR/$VIDEO_SAVE_PATH_SUBDIR_NAME}"
            # Try subdirectories first (e.g., input_dir/checkpoint_name/multi-speaker)
            for _subdir in "$_base"/*/; do
                if [ -d "$_subdir/multi-speaker" ]; then
                    _cpcer_video_dir="$_subdir/multi-speaker"
                    break
                fi
            done
            # Fallback: direct multi-speaker under base dir
            if [ -z "$_cpcer_video_dir" ] && [ -d "$_base/multi-speaker" ]; then
                _cpcer_video_dir="$_base/multi-speaker"
            fi
        fi

        if [ ! -d "$_cpcer_video_dir" ]; then
            echo "[SKIP] cpCER: video directory not found: $_cpcer_video_dir"
            FAILED_METRICS="$FAILED_METRICS cpCER(video_dir_missing)"
        else
            echo ""
            echo "======== Running: cpCER (multi-speaker conversational accuracy) ========"
            _cpcer_cmd="$PYTHON_CPCER $SCRIPT_CPCER --video_dir $_cpcer_video_dir --references_txt $REFERENCES_TXT"
            if [ -n "$ASR_API_KEY" ]; then
                _cpcer_cmd="$_cpcer_cmd --asr_api_key $ASR_API_KEY"
            fi
            _cpcer_output="$EXP_ROOT/$VIDEO_SAVE_PATH_SUBDIR_NAME/eval_cpcer_scores_means.json"
            _cpcer_cmd="$_cpcer_cmd --output_json $_cpcer_output"
            if eval $_cpcer_cmd; then
                echo "  [OK] cpCER"
            else
                echo "  [FAILED] cpCER"
                FAILED_METRICS="$FAILED_METRICS cpCER"
            fi
            python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        fi
    fi
fi
echo ""
echo "========================================"
echo "Evaluation Complete"
echo "========================================"
if [ -n "$FAILED_METRICS" ]; then
    echo "Failed metrics:$FAILED_METRICS"
    exit 1
fi
