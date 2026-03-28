#!/usr/bin/env python3
"""
API 工具函数 - DashScope / Gemini 等 API 调用的公共逻辑
"""

import os
from typing import Optional, Tuple

from config import DASHSCOPE_BASE_URL


def resolve_api_keys(
    api_key: Optional[str] = None,
    qwen_api_key: Optional[str] = None,
    minimax_api_key: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    统一解析 Gemini、DashScope/Qwen 和 MiniMax API Key。
    优先级：参数 > 环境变量

    Returns:
        (gemini_key, qwen_key, minimax_key)
    """
    gemini = (api_key or "").strip() or os.getenv("GEMINI_API_KEY", "")
    qwen = (qwen_api_key or "").strip() or os.getenv("DASHSCOPE_API_KEY", "")
    minimax = (minimax_api_key or "").strip() or os.getenv("MINIMAX_API_KEY", "")
    return gemini, qwen, minimax


def setup_dashscope_url(base_url: Optional[str] = None) -> None:
    """
    设置 dashscope 的 base_http_api_url。
    参数 base_url 优先，否则使用 config.DASHSCOPE_BASE_URL。
    """
    import dashscope
    dashscope.base_http_api_url = (base_url or DASHSCOPE_BASE_URL).rstrip('/')
