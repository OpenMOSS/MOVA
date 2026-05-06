#!/bin/bash
# ======================================
# Create conda environments for MOVA Evaluation
# ======================================
# Usage:
#   bash scripts/setup_envs.sh
#
# Creates 5 conda environments:
#   mova_eval_visual  → VideoReward
#   mova_eval_av      → AV Quality + Lip Sync
#   mova_eval_audio   → Audio IS + CLAP
#   mova_eval_dnsmos  → DNSMOS + Audio Amplitude
#   mova_eval_cpcer   → cpCER
#
# After setup, source scripts/set_env.sh before running evaluation:
#   source scripts/set_env.sh
#   bash scripts/run_eval.sh ...
# ======================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVS_DIR="$REPO_ROOT/envs"

# Check conda is available
if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Please install conda first." >&2
    exit 1
fi

echo "========================================"
echo "MOVA Eval - Conda Environment Setup"
echo "========================================"

create_env() {
    local env_name="$1"
    local req_file="$2"
    shift 2
    local local_pkgs=("$@")

    if conda env list | grep -q "^$env_name "; then
        echo "[SKIP] $env_name already exists"
        return 0
    fi

    echo ""
    echo "Creating: $env_name"
    conda create -n "$env_name" python=3.10 -y

    # Install vendored local packages first (so pip doesn't try to fetch from PyPI)
    for pkg in "${local_pkgs[@]}"; do
        if [ -d "$REPO_ROOT/$pkg" ]; then
            echo "  Installing vendored package: $pkg"
            conda run -n "$env_name" pip install -e "$REPO_ROOT/$pkg" --no-deps
        fi
    done

    conda run -n "$env_name" pip install -r "$req_file"
    echo "[OK] $env_name created"
}

create_env "mova_eval_visual" "$ENVS_DIR/requirements_visual.txt"
create_env "mova_eval_av"     "$ENVS_DIR/requirements_av.txt" \
    "models/imagebind" "models/dover" "models/pytorchvideo" \
    "models/clap" "metrics/av_quality/av_bench"
create_env "mova_eval_audio"  "$ENVS_DIR/requirements_audio.txt" \
    "models/clap"
create_env "mova_eval_dnsmos" "$ENVS_DIR/requirements_dnsmos.txt"
create_env "mova_eval_cpcer"  "$ENVS_DIR/requirements_cpcer.txt"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Before running evaluation, activate the environments:"
echo "  source scripts/set_env.sh"
echo ""
echo "Then run:"
echo "  bash scripts/run_eval.sh --checkpoint_dir ... --video_save_path_subdir_name ... --prompt_meta_json ... --video_second 8"
