#!/usr/bin/env python3
"""
首帧图生成脚本（合并版）

功能:
1. prompt: 将用户描述改写为首帧图生成提示词 (Gemini 2.5 Pro 或 通义千问 qwen-plus)
2. generate: 使用提示词生成首帧图 (Gemini 2.5 Flash Image 或 通义 Z-Image)
3. full: 完整流程（prompt + generate）一次完成

提示词生成：有 GEMINI_API_KEY 用 Gemini；否则用 DashScope qwen-plus。
图片生成：有 GEMINI_API_KEY 用 Gemini Flash Image；否则用通义文生图 Z-Image。
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None  # 仅在使用 Gemini 时需要

try:
    import requests
except ImportError:
    requests = None

try:
    import dashscope
except ImportError:
    dashscope = None

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None

from config import DASHSCOPE_MULTIMODAL_GENERATION_URL, GEMINI_API_URL, GEMINI_MODEL, MINIMAX_BASE_URL, MINIMAX_MODEL
from api_utils import setup_dashscope_url, resolve_api_keys


# ============================================================================
# System Prompt - 引导生成首帧图提示词
# ============================================================================

IMAGE_PROMPT_SYSTEM = """You are an expert at converting user descriptions into first-frame image generation prompts for video content.

### Your Task:
Analyze the user's description and convert it into a detailed image generation prompt that captures the FIRST FRAME of their envisioned video. You MUST include ALL visual elements explicitly mentioned in the user's description. Your goal is to faithfully represent what the user described while translating it into a static image description suitable for image generation models.

### Guidelines:

1. **Analyze Visual Style from User Description**:
   - Look for any explicit mentions of visual style, cinematography, color grading, or mood in the user's description
   - If the user mentions a film/show style, incorporate that aesthetic into your prompt
   - Infer appropriate visual characteristics based on the content: lighting (natural/artificial, warm/cool), color palette, composition style, production quality
   - If no style is mentioned, choose a style that fits the content naturally

2. **Identify and Include ALL Visual Elements from User Description (REQUIRED)**:
   - You MUST include every visual element explicitly mentioned by the user
   - Main subjects: people, characters, objects - include their appearance, position, and pose as described
   - Setting/Environment: location, background, spatial layout mentioned in the description
   - Any UI elements: text, logos, watermarks if mentioned in the description
   - Actions/States: if the user describes an action or state, capture it as the initial moment (e.g., "struggling" becomes "showing effort", "trips" becomes "about to trip")
   - Do not omit any visual elements the user mentioned, even if they seem minor

3. **Specify Camera Parameters (REQUIRED)**:
   You MUST explicitly specify camera parameters based on the scene described:
   - **Shot Size**: Choose from extreme close-up, close-up, medium close-up, medium shot, medium long shot, long shot, extreme long shot, or wide shot. Base your choice on what the user wants to emphasize (face, upper body, full body, environment, etc.)
   - **Camera Angle**: Specify eye-level, high angle (looking down), low angle (looking up), bird's eye view, worm's eye view, or Dutch angle. Consider the emotional tone and what perspective best serves the scene
   - **Camera Movement/Position**: If relevant, mention static shot, tracking shot, or specific camera position (front, side, back, etc.)
   - **Framing**: Describe how subjects are positioned within the frame (rule of thirds, centered, etc.)
   
   If the user doesn't specify these, infer the most appropriate camera parameters based on the content and emotional tone of the description.

4. **Format for Image Generation**:
   - Write in present tense, describing what IS visible in the first frame
   - Be concrete and specific about visual details based on what the user described
   - Include foreground, midground, and background elements only if mentioned or clearly implied
   - Describe spatial relationships and composition
   - Keep focused on the INITIAL visual state (no actions or temporal progression)
   - Expand user descriptions appropriately but avoid adding excessive details not mentioned or implied
   - Stay faithful to the user's description - do not invent major visual elements that weren't mentioned

5. **What NOT to Include**:
   - No temporal sequences ("then", "next", "after") - only describe the first moment
   - No audio descriptions - focus purely on visual elements
   - No story progression beyond the first moment
   - No abstract concepts unless they can be visually represented

### Output Format:
Provide a single, detailed paragraph (100-250 words) that describes the first frame as a static image. Structure your prompt as follows:
1. Start with camera parameters: shot size and camera angle (e.g., "A medium shot from eye-level angle...")
2. Then describe visual style: lighting, color palette, mood
3. Then describe the content: subjects, setting, composition details
4. End with any additional visual details or atmosphere

Always explicitly state the shot size and camera angle - these are essential for accurate image generation. Use descriptive, visual language suitable for image generation models."""

IMAGE_PROMPT_USER = """
### User's Description:
{user_input}

Please generate a detailed first-frame image prompt for this description."""


# ============================================================================
# 核心功能函数
# ============================================================================

def generate_image_prompt(
    user_input: str,
    api_base_url: str,
    api_key: str,
    model: str = "gemini-2.5-pro"
) -> str:
    """
    使用 Gemini API 生成首帧图提示词

    Args:
        user_input: 用户原始输入
        api_base_url: API 基础 URL
        api_key: API 密钥
        model: 使用的模型名称

    Returns:
        首帧图生成提示词
    """
    os.environ['GOOGLE_GEMINI_BASE_URL'] = api_base_url
    os.environ['GEMINI_API_KEY'] = api_key

    client = genai.Client()
    user_prompt = IMAGE_PROMPT_USER.format(user_input=user_input)

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=IMAGE_PROMPT_SYSTEM
        ),
        contents=[user_prompt],
    )

    image_prompt = response.text.strip()
    if not image_prompt:
        raise ValueError("API返回了空的提示词")

    return image_prompt


def generate_image_prompt_qwen(
    user_input: str,
    api_key: str,
    model: str = "qwen-plus",
    base_url: str = None,
) -> str:
    """
    使用 DashScope 通义千问（qwen-plus）生成首帧图提示词。
    参考: https://help.aliyun.com/zh/model-studio/qwen-api-via-dashscope

    Args:
        user_input: 用户原始输入
        api_key: DashScope API Key
        model: 模型名称，默认 qwen-plus
        base_url: API base URL（可选，默认北京地域）

    Returns:
        首帧图生成提示词
    """
    if dashscope is None:
        raise ImportError("请先安装 dashscope: pip install dashscope")

    setup_dashscope_url(base_url)
    user_prompt = IMAGE_PROMPT_USER.format(user_input=user_input)
    messages = [
        {"role": "system", "content": IMAGE_PROMPT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    response = dashscope.Generation.call(
        api_key=api_key,
        model=model,
        messages=messages,
        result_format="message",
    )

    if response.status_code != 200:
        raise RuntimeError(f"DashScope API 错误: {getattr(response, 'message', response.code or response.status_code)}")

    content = response.output.choices[0].message.content
    image_prompt = (content or "").strip()
    if not image_prompt:
        raise ValueError("API返回了空的提示词")

    return image_prompt


def generate_image_prompt_minimax(
    user_input: str,
    api_key: str,
    model: str = "MiniMax-M3",
    base_url: str = None,
) -> str:
    """
    使用 MiniMax OpenAI-compatible API 生成首帧图提示词。

    Args:
        user_input: 用户原始输入
        api_key: MiniMax API Key
        model: 模型名称，默认 MiniMax-M3
        base_url: API base URL（可选，默认 https://api.minimax.io/v1）

    Returns:
        首帧图生成提示词
    """
    if _OpenAI is None:
        raise ImportError("使用 MiniMax 需安装 openai: pip install openai")

    client = _OpenAI(
        api_key=api_key,
        base_url=base_url or MINIMAX_BASE_URL,
    )

    user_prompt = IMAGE_PROMPT_USER.format(user_input=user_input)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": IMAGE_PROMPT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content or ""
    image_prompt = content.strip()
    if not image_prompt:
        raise ValueError("API返回了空的提示词")

    return image_prompt


# Z-Image 宽高比到 size 的映射（参考 https://help.aliyun.com/zh/model-studio/z-image-api-reference）
ZIMAGE_ASPECT_TO_SIZE = {
    "16:9": "1280*720",
    "9:16": "720*1280",
}


def generate_image_zimage(
    image_prompt: str,
    api_key: str,
    output_path: str,
    aspect_ratio: str = "16:9",
    base_url: str = None,
) -> str:
    """
    使用阿里云通义文生图 Z-Image 生成首帧图

    API 参考: https://help.aliyun.com/zh/model-studio/z-image-api-reference

    Args:
        image_prompt: 图片生成提示词（不超过 800 字符）
        api_key: DashScope API Key
        output_path: 输出图片路径
        aspect_ratio: 宽高比 "16:9" 或 "9:16"
        base_url: 完整 API URL（可选，默认使用 config 中 multimodal-generation 地址）

    Returns:
        生成的图片路径
    """
    if requests is None:
        raise ImportError("请先安装 requests: pip install requests")

    size = ZIMAGE_ASPECT_TO_SIZE.get(aspect_ratio, "1280*720")
    # 完整 URL：base 上拼 multimodal-generation 路径（可由 config 传入或使用 config 默认）
    url = base_url if base_url else DASHSCOPE_MULTIMODAL_GENERATION_URL

    # 提示词超过 800 字符时截断（Z-Image 限制）
    if len(image_prompt) > 800:
        image_prompt = image_prompt[:797] + "..."

    print(f"🎨 正在使用通义 Z-Image 生成首帧图 (宽高比: {aspect_ratio}, size: {size})...", file=sys.stderr)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": "z-image-turbo",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": image_prompt}],
                }
            ]
        },
        "parameters": {
            "prompt_extend": False,
            "size": size,
        },
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if "code" in data and data["code"]:
        raise ValueError(f"Z-Image API 错误: {data.get('message', data)}")

    content = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", [])
    image_url = None
    for item in content:
        if isinstance(item, dict) and "image" in item:
            image_url = item["image"]
            break

    if not image_url:
        raise ValueError("Z-Image 未返回图片 URL")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    urllib.request.urlretrieve(image_url, output_path)
    print(f"✅ 首帧图已保存到: {output_path}", file=sys.stderr)
    return output_path


def generate_image(
    image_prompt: str,
    api_key: str,
    output_path: str,
    aspect_ratio: str = "16:9",
    api_base_url: str = "https://generativelanguage.googleapis.com",
    qwen_api_key: str = None,
) -> str:
    """
    生成首帧图。若无 GEMINI_API_KEY 则使用通义 Z-Image（需 DASHSCOPE_API_KEY）。

    Args:
        image_prompt: 图片生成提示词
        api_key: Gemini API key（可选，无则用 qwen_api_key 调用 Z-Image）
        output_path: 输出图片路径
        aspect_ratio: 宽高比 "16:9" 或 "9:16"
        api_base_url: Gemini API base URL（仅 Gemini 使用）
        qwen_api_key: DashScope/Qwen API key（无 Gemini key 时用于 Z-Image）

    Returns:
        生成的图片路径
    """
    qwen_key = (qwen_api_key or "").strip() or os.getenv("DASHSCOPE_API_KEY", "")
    gemini_key = (api_key or "").strip()

    # 优先使用 Gemini；仅当无 Gemini key 时才用 Qwen (Z-Image)
    if gemini_key:
        # 使用 Gemini 2.5 Flash Image
        if genai is None or types is None:
            raise ImportError("使用 Gemini 需安装 google-genai: pip install google-genai")
        print(f"🎨 正在使用 Gemini Flash Image 生成首帧图 (宽高比: {aspect_ratio})...", file=sys.stderr)
        os.environ['GOOGLE_GEMINI_BASE_URL'] = api_base_url
        os.environ['GEMINI_API_KEY'] = gemini_key

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[image_prompt],
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                )
            )
        )

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                image.save(output_path)
                print(f"✅ 首帧图已保存到: {output_path}", file=sys.stderr)
                return output_path

        raise ValueError("Gemini 未返回图片数据")

    elif qwen_key:
        # 使用通义 Z-Image
        return generate_image_zimage(
            image_prompt=image_prompt,
            api_key=qwen_key,
            output_path=output_path,
            aspect_ratio=aspect_ratio,
        )
    else:
        raise ValueError(
            "未提供图片生成 API Key。请设置 GEMINI_API_KEY 或 DASHSCOPE_API_KEY（或 --qwen-api-key）"
        )


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="首帧图生成脚本（提示词生成 + 图片生成）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（用户描述 → 提示词 → 首帧图）
  python3 generate_first_frame.py full \\
      --user-input "一只猫在弹钢琴" \\
      --output first_frame.png \\
      --api-url "http://example.com" \\
      --api-key YOUR_KEY

  # 仅生成提示词（输出到 stdout）
  python3 generate_first_frame.py prompt \\
      --user-input "一只猫在弹钢琴" \\
      --api-url "http://example.com" \\
      --api-key YOUR_KEY

  # 仅生成图片
  python3 generate_first_frame.py generate \\
      --image-prompt "A medium shot..." \\
      --output first_frame.png \\
      --api-key YOUR_KEY
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # ========== full 命令 ==========
    full_parser = subparsers.add_parser('full', help='完整流程：用户描述 → 首帧图')
    full_parser.add_argument('--user-input', type=str, required=True, help='用户描述')
    full_parser.add_argument('--output', type=str, default=None, help='输出图片路径（默认: generated_first_frame.png）')
    full_parser.add_argument('--output-image-prompt', type=str, default=None, help='将 image generation prompt 写入此文件（可选）')
    full_parser.add_argument('--api-key', type=str, default=None, help='Gemini API key（与 --qwen-api-key 二选一）')
    full_parser.add_argument('--api-url', type=str, default=None, help='Gemini API base URL（使用 Gemini 时必填）')
    full_parser.add_argument('--qwen-api-key', type=str, default=None,
                            help='DashScope/Qwen API key，用于 Z-Image；无 Gemini key 时使用')
    full_parser.add_argument('--minimax-api-key', type=str, default=None,
                            help='MiniMax API key，用于提示词生成（无 Gemini/Qwen key 时使用）')
    full_parser.add_argument('--pro-model', type=str, default=None, help='提示词生成模型（默认: gemini-2.5-pro，仅 Gemini）')
    full_parser.add_argument('--aspect-ratio', type=str, choices=['16:9', '9:16'], default='16:9',
                            help='宽高比 (默认: 16:9)')

    # ========== prompt 命令 ==========
    prompt_parser = subparsers.add_parser('prompt', help='仅生成首帧图提示词')
    prompt_parser.add_argument('--user-input', type=str, required=True, help='用户描述')
    prompt_parser.add_argument('--api-url', type=str, default=None, help='Gemini API base URL（使用 Gemini 时必填）')
    prompt_parser.add_argument('--api-key', type=str, default=None, help='Gemini API key（与 --qwen-api-key 二选一）')
    prompt_parser.add_argument('--qwen-api-key', type=str, default=None, help='DashScope API key，使用 qwen-plus 生成提示词')
    prompt_parser.add_argument('--minimax-api-key', type=str, default=None, help='MiniMax API key，用于提示词生成')
    prompt_parser.add_argument('--model', type=str, default=None, help='模型名称（Gemini 默认: gemini-2.5-pro；Qwen 默认: qwen-plus）')
    prompt_parser.add_argument('--output', type=str, default=None, help='输出文件路径（不指定则输出到 stdout）')

    # ========== generate 命令 ==========
    gen_parser = subparsers.add_parser('generate', help='仅生成首帧图')
    gen_parser.add_argument('--image-prompt', type=str, required=True, help='首帧图生成提示词')
    gen_parser.add_argument('--output', type=str, default=None, help='输出图片路径（默认: generated_first_frame.png）')
    gen_parser.add_argument('--image-api-key', type=str, default=None, help='Gemini API key（与 --qwen-api-key 二选一）')
    gen_parser.add_argument('--image-api-url', type=str, default=None,
                            help='Gemini API base URL（默认: https://generativelanguage.googleapis.com）')
    gen_parser.add_argument('--qwen-api-key', type=str, default=None,
                            help='DashScope/Qwen API key，用于 Z-Image；无 Gemini key 时使用')
    gen_parser.add_argument('--aspect-ratio', type=str, choices=['16:9', '9:16'], default='16:9',
                            help='宽高比 (默认: 16:9)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    DEFAULT_OUTPUT = 'generated_first_frame.png'

    try:
        if args.command == 'full':
            output_path = args.output or DEFAULT_OUTPUT
            gemini_key, qwen_key, minimax_key = resolve_api_keys(
                args.api_key,
                getattr(args, "qwen_api_key", None),
                getattr(args, "minimax_api_key", None),
            )

            if gemini_key and args.api_url:
                # 有 Gemini key：先生成提示词，再用 Gemini 或 Z-Image 生图
                image_prompt = generate_image_prompt(
                    user_input=args.user_input,
                    api_base_url=args.api_url,
                    api_key=gemini_key,
                    model=args.pro_model or GEMINI_MODEL
                )
            elif qwen_key:
                # 无 Gemini key 但有 Qwen key：用 qwen-plus 生成提示词，再用 Z-Image 生图
                print("📝 未提供 Gemini API Key，使用通义千问 qwen-plus 生成首帧图提示词，再调用 Z-Image 生图", file=sys.stderr)
                image_prompt = generate_image_prompt_qwen(
                    user_input=args.user_input,
                    api_key=qwen_key,
                    model="qwen-plus",
                )
            elif minimax_key:
                # 使用 MiniMax 生成提示词（仍需要 Gemini/Qwen key 生成图片）
                print("📝 使用 MiniMax 生成首帧图提示词", file=sys.stderr)
                image_prompt = generate_image_prompt_minimax(
                    user_input=args.user_input,
                    api_key=minimax_key,
                    model=MINIMAX_MODEL,
                )
            else:
                raise ValueError("请提供 --api-key (Gemini)、--qwen-api-key (DashScope) 或 --minimax-api-key (MiniMax)")

            aspect_ratio = args.aspect_ratio  # Use explicit parameter or default '16:9'
            image_path = generate_image(
                image_prompt=image_prompt,
                api_key=gemini_key,
                output_path=output_path,
                aspect_ratio=aspect_ratio,
                api_base_url=args.api_url or GEMINI_API_URL,
                qwen_api_key=qwen_key,
            )
            # 若指定了 --output-image-prompt，写入 image prompt 到文件
            output_prompt_path = getattr(args, 'output_image_prompt', None)
            if output_prompt_path:
                Path(output_prompt_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_prompt_path, 'w', encoding='utf-8') as f:
                    f.write(image_prompt)
            print(f"\n✅ 首帧图已保存: {image_path}")

        elif args.command == 'prompt':
            gemini_key, qwen_key, minimax_key = resolve_api_keys(
                args.api_key,
                getattr(args, "qwen_api_key", None),
                getattr(args, "minimax_api_key", None),
            )
            if gemini_key and args.api_url:
                image_prompt = generate_image_prompt(
                    user_input=args.user_input,
                    api_base_url=args.api_url,
                    api_key=gemini_key,
                    model=args.model or GEMINI_MODEL
                )
            elif qwen_key:
                image_prompt = generate_image_prompt_qwen(
                    user_input=args.user_input,
                    api_key=qwen_key,
                    model=args.model or "qwen-plus",
                )
            elif minimax_key:
                image_prompt = generate_image_prompt_minimax(
                    user_input=args.user_input,
                    api_key=minimax_key,
                    model=args.model or MINIMAX_MODEL,
                )
            else:
                raise ValueError("请提供 --api-key (Gemini)、--qwen-api-key (DashScope) 或 --minimax-api-key (MiniMax)，或设置环境变量")
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(image_prompt)
                print(f"✅ 提示词已保存到: {args.output}", file=sys.stderr)
            else:
                print(image_prompt)

        elif args.command == 'generate':
            output_path = args.output or DEFAULT_OUTPUT
            image_api_url = args.image_api_url or GEMINI_API_URL
            gemini_key, qwen_key = resolve_api_keys(args.image_api_key, getattr(args, "qwen_api_key", None))
            if not gemini_key and not qwen_key:
                raise ValueError("请提供 --image-api-key (Gemini) 或 --qwen-api-key (DashScope)，或设置环境变量")
            aspect_ratio = args.aspect_ratio  # Use explicit parameter or default '16:9'
            image_path = generate_image(
                image_prompt=args.image_prompt,
                api_key=gemini_key,
                output_path=output_path,
                aspect_ratio=aspect_ratio,
                api_base_url=image_api_url,
                qwen_api_key=qwen_key,
            )
            print(f"\n✅ 图片生成成功: {image_path}")

    except Exception as e:
        print(f"\n❌ 执行失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
