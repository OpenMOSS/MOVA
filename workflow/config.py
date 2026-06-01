#!/usr/bin/env python3
"""
Configuration for SGLang Streamlit Workflow

Contains server endpoints, API keys, and default parameters.
"""

import os

# ============================================================================
# SGLang Server Configuration
# ============================================================================

#fill the base_url with forward adress, e.g. http://localhost:30000 or http://notebook.../proxy/30000
#SGLANG_SERVERS can be filled with one or multiple servers
SGLANG_SERVERS = {
    "mova-360p": {
        "name": "MOVA 360p",
        "base_url": "http://localhost:30000",
        "default_size": "640x360",
        "description": "MOVA 360p model (faster, lower resolution)"
    },
    "mova-720p": {
        "name": "MOVA 720p", 
        "base_url": "http://notebook.../proxy/30000",
        "default_size": "1280x720",
        "description": "MOVA 720p model (slower, higher resolution)"
    }
}

# ============================================================================
# Default Video Generation Parameters
# ============================================================================

DEFAULT_VIDEO_PARAMS = {
    "num_frames": 193,
    "fps": 24,
    "seed": 0,
    "guidance_scale": 5.0,
    "num_inference_steps": 50
}

# Size options for different resolutions
SIZE_OPTIONS = {
    "360p": {
        "landscape": "640x360",
        "portrait": "360x640"
    },
    "720p": {
        "landscape": "1280x720",
        "portrait": "720x1280"
    }
}

# ============================================================================
# Gemini API Configuration (for Full Workflow Mode)
# ============================================================================

GEMINI_API_URL = os.environ.get(
    'GEMINI_API_URL',
    # 填入Gemini API URL
    ''
)

GEMINI_API_KEY = os.environ.get(
    'GEMINI_API_KEY',
    # 填入Gemini API Key
    ''
)

GEMINI_MODEL = os.environ.get(
    'GEMINI_MODEL',
    'gemini-2.5-pro'
)

# ============================================================================
# MiniMax API Configuration (OpenAI-compatible alternative for prompt tasks)
# ============================================================================

MINIMAX_API_KEY = os.environ.get(
    'MINIMAX_API_KEY',
    ''
)

MINIMAX_BASE_URL = os.environ.get(
    'MINIMAX_BASE_URL',
    'https://api.minimax.io/v1'
)

MINIMAX_MODEL = os.environ.get(
    'MINIMAX_MODEL',
    'MiniMax-M3'
)

# ============================================================================
# DashScope API Configuration (Qwen VL / Z-Image / qwen-plus 等)
# 北京地域默认；可设 DASHSCOPE_BASE_URL 换地域（如新加坡、美国）
# ============================================================================

DASHSCOPE_BASE_URL = os.environ.get(
    'DASHSCOPE_BASE_URL',
    # '填入DashScope API URL'
    ''
).rstrip('/')

# 在 base 上拼接具体 API 路径（规范：base 统一，URL 由此派生）
DASHSCOPE_TEXT_GENERATION_URL = f"{DASHSCOPE_BASE_URL}/services/aigc/text-generation/generation"
DASHSCOPE_MULTIMODAL_GENERATION_URL = f"{DASHSCOPE_BASE_URL}/services/aigc/multimodal-generation/generation"

# Qwen VL（视觉元素提取）模型
QWEN_VL_MODEL = os.environ.get(
    'QWEN_VL_MODEL',
    'qwen3-vl-flash'
)

QWEN_VL_API_KEY = os.environ.get(
    'DASHSCOPE_API_KEY',
    # 填入DashScope API Key
    ''
)

# ============================================================================
# Application Settings
# ============================================================================

# Task polling interval (seconds)
POLL_INTERVAL = 5

# Task timeout (seconds) - 30 minutes
TASK_TIMEOUT = 1800

# Maximum number of tasks to keep in history
MAX_TASK_HISTORY = 1000

# Output directory
OUTPUT_DIR = "outputs"

# Tasks file
TASKS_FILE = "tasks.json"
