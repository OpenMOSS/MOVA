#!/bin/bash
# ======================================
# Set Python interpreter paths for MOVA Evaluation
# ======================================
# Usage:
#   source scripts/set_env.sh
#
# This script auto-detects conda environments by name and sets
# PYTHON_* variables so that run_eval.sh uses the correct
# interpreter for each metric group.
#
# Requires: conda initialized in the current shell
# ======================================

# Find conda base directory
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "Error: conda not found. Please activate conda first." >&2
    return 1 2>/dev/null || exit 1
fi

# Environment names (must match the names in envs/*.yml)
ENV_VISUAL="mova_eval_visual"
ENV_AV="mova_eval_av"
ENV_AUDIO="mova_eval_audio"
ENV_DNSMOS="mova_eval_dnsmos"
ENV_CPCER="mova_eval_cpcer"

# Auto-detect python path for each environment
find_python() {
    local env_name="$1"
    local python_path="$CONDA_BASE/envs/$env_name/bin/python"
    if [ -x "$python_path" ]; then
        echo "$python_path"
    else
        echo ""
    fi
}

# Set PYTHON_* (skip if already set by user)
if [ -z "$PYTHON_VIDEO_REWARD" ]; then
    PYTHON_VIDEO_REWARD=$(find_python "$ENV_VISUAL")
fi
if [ -z "$PYTHON_AV_QUALITY" ]; then
    PYTHON_AV_QUALITY=$(find_python "$ENV_AV")
fi
if [ -z "$PYTHON_AUDIO_QUALITY" ]; then
    PYTHON_AUDIO_QUALITY=$(find_python "$ENV_AUDIO")
fi
if [ -z "$PYTHON_DNSMOS" ]; then
    PYTHON_DNSMOS=$(find_python "$ENV_DNSMOS")
fi
if [ -z "$PYTHON_CPCER" ]; then
    PYTHON_CPCER=$(find_python "$ENV_CPCER")
fi

# Derived: Amplitude uses DNSMOS env, Lip uses AV env
if [ -z "$PYTHON_AUDIO_AMPLITUDE" ]; then
    PYTHON_AUDIO_AMPLITUDE="$PYTHON_DNSMOS"
fi
if [ -z "$PYTHON_LIP_QUALITY" ]; then
    PYTHON_LIP_QUALITY="$PYTHON_AV_QUALITY"
fi

export PYTHON_VIDEO_REWARD PYTHON_AV_QUALITY PYTHON_AUDIO_QUALITY
export PYTHON_DNSMOS PYTHON_AUDIO_AMPLITUDE PYTHON_LIP_QUALITY PYTHON_CPCER

# Verify
echo "MOVA Eval Environment:"
echo "  VideoReward:   ${PYTHON_VIDEO_REWARD:-NOT FOUND}"
echo "  AV + Lip:      ${PYTHON_AV_QUALITY:-NOT FOUND}"
echo "  Audio IS+CLAP: ${PYTHON_AUDIO_QUALITY:-NOT FOUND}"
echo "  DNSMOS + Amp:  ${PYTHON_DNSMOS:-NOT FOUND}"
echo "  cpCER:         ${PYTHON_CPCER:-NOT FOUND}"
