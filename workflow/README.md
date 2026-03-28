# MOVA Video Generation Workflow

**Language / 语言:** [English](#english) | [中文](#中文)

---

<a name="english"></a>

## English

### 1. What This Workflow Does

This workflow provides an **AI-assisted end-to-end pipeline** for video generation using the MOVA model. It offers two modes:

- **Full Workflow Mode**: User describes a scene in text (optionally uploads a first frame) → if no image provided, AI generates first frame; otherwise skips this step → extracts visual elements → rewrites into video description → MOVA generates video
- **Simple Mode**: User provides prompt + first frame image directly → MOVA generates video

> **Note (First Frame):** If you do **not** provide an image in Full Workflow mode, the system will automatically call an image generation model to create the first frame:
> - With **Gemini API Key**: Gemini 2.5 Flash Image
> - Without Gemini (Qwen only): 通义文生图 Z-Image + qwen-plus for prompt generation

> **Note (T2VA Testing):** To test MOVA's **text-to-video-audio (T2VA)** capability (i.e., generation from text only without a meaningful first frame), upload a **pure white image** as the first frame and provide your text prompt.

### 2. How to Use

#### Step 1: Start SGLang Server

Set the path to your MOVA checkpoint directory, then start the backend:

```bash
cd workflow
export MOVA_MODEL_PATH="/path/to/your/MOVA-360p-hf"   # or MOVA-720p-hf
./launch_sglang_server.sh
```

Ensure the server is running and accessible (default port: 30000).

#### Step 2: Configure `config.py`

Edit `config.py` to set:

- **SGLANG_SERVERS**: Update `base_url` for each model (360p / 720p) to match your SGLang server address.
- **API Keys** (optional, for Full Workflow):
  - `GEMINI_API_KEY` + `GEMINI_API_URL`: Gemini API (recommended; fill base URL when using proxy)
  - `DASHSCOPE_API_KEY` + `DASHSCOPE_BASE_URL`: Qwen/DashScope API (base URL for region, e.g. Singapore/US)
  - `MINIMAX_API_KEY`: [MiniMax](https://platform.minimaxi.com/) API for prompt generation (OpenAI-compatible; uses MiniMax-M2.7 by default)

> **Note (API Key Fallback):** If you do **not** provide a Gemini API key, the workflow will automatically use the Qwen (DashScope) API key to call qwen-plus and Z-Image for prompt generation and first frame generation. Alternatively, you can use a **MiniMax API key** (`MINIMAX_API_KEY`) for prompt rewriting and image prompt generation via MiniMax's OpenAI-compatible API. **We still recommend using a Gemini API key** for best quality.

#### Step 3: Start Streamlit App

```bash
./launch_streamlit.sh
# Or: streamlit run app.py --server.port 8500 --server.address 0.0.0.0
```

Open the URL shown in the terminal (e.g., http://localhost:8500) and use the web interface.

### Requirements

- See `requirements.txt` for Python dependencies.

### File Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `api_utils.py` | Shared API helpers (DashScope URL setup, API key resolution) |
| `config.py` | Server URLs, API keys, default parameters |
| `sglang_client.py` | SGLang video API client |
| `generate_first_frame.py` | Image prompt + first frame generation (Gemini, qwen-plus, or MiniMax + Z-Image) |
| `qwen_vl_api.py` | Visual element extraction from image (Qwen3-VL) |
| `prompt_rewriter_with_image.py` | Video description rewriting (Gemini, qwen-plus, or MiniMax) |
| `launch_sglang_server.sh` | SGLang server launcher (set `MOVA_MODEL_PATH` to your checkpoint dir) |
| `launch_streamlit.sh` | Streamlit app launcher |

---

<a name="中文"></a>

## 中文

### 1. 工作流简介

本工作流提供基于 **MOVA 模型** 的 AI 辅助端到端视频生成管道，支持两种模式：

- **完整工作流模式**：用户用文字描述场景（可选择性上传首帧图）→ 若未提供图片则 AI 生成首帧图，否则跳过此步 → 提取视觉元素 → 改写为视频描述 → MOVA 生成视频
- **简单模式**：用户直接提供提示词 + 首帧图 → MOVA 生成视频

> **说明（首帧图）：** 在完整工作流中，如果你**未**提供图片，系统会自动调用图像生成模型生成首帧图：
> - 有 **Gemini API Key** 时：使用 Gemini 2.5 Flash Image
> - 无 Gemini（仅 Qwen）时：使用通义文生图 Z-Image，搭配 qwen-plus 生成提示词

> **说明（T2VA 测试）：** 若要测试 MOVA 的**纯文本生成视频（T2VA）**能力（即无有效首帧图、仅凭文本生成），请上传**纯白图片**作为首帧图，并输入你的文本提示词。

### 2. 使用步骤

#### 第一步：启动 SGLang 服务

设置 MOVA 模型 checkpoint 目录后启动后端：

```bash
cd workflow
export MOVA_MODEL_PATH="/path/to/your/MOVA-360p-hf"   # 或 MOVA-720p-hf
./launch_sglang_server.sh
```

确保服务已启动且可访问（默认端口：30000）。

#### 第二步：配置 `config.py`

编辑 `config.py`，设置：

- **SGLANG_SERVERS**：根据你的 SGLang 服务地址，修改各模型（360p / 720p）的 `base_url`
- **API 密钥**（可选，完整工作流需要）：
  - `GEMINI_API_KEY` + `GEMINI_API_URL`：Gemini API（推荐；使用代理时需填写 base URL）
  - `DASHSCOPE_API_KEY` + `DASHSCOPE_BASE_URL`：通义千问 / DashScope API（base URL 可选，用于指定地域如新加坡/美国）
  - `MINIMAX_API_KEY`：[MiniMax](https://platform.minimaxi.com/) API，用于提示词生成（OpenAI 兼容接口；默认使用 MiniMax-M2.7）

> **说明（API 密钥回退）：** 若不提供 Gemini API 密钥，工作流会自动使用 Qwen（DashScope）API 密钥调用 qwen-plus 和 Z-Image 完成提示词生成和首帧图生成。也可使用 **MiniMax API 密钥**（`MINIMAX_API_KEY`）通过 MiniMax 的 OpenAI 兼容 API 完成提示词改写和首帧图提示词生成。**我们仍推荐使用 Gemini API 密钥**以获得更好效果。

#### 第三步：启动 Streamlit 应用

```bash
./launch_streamlit.sh
# 或：streamlit run app.py --server.port 8500 --server.address 0.0.0.0
```

在终端显示的 URL（如 http://localhost:8500）打开浏览器即可使用。

### 环境要求

- Python 依赖见 `requirements.txt`。

### 文件结构

| 文件 | 说明 |
|------|------|
| `app.py` | Streamlit Web 应用 |
| `api_utils.py` | API 工具（DashScope URL 设置、API Key 解析等） |
| `config.py` | 服务 URL、API 密钥、默认参数 |
| `sglang_client.py` | SGLang 视频 API 客户端 |
| `generate_first_frame.py` | 首帧图提示词与生成（Gemini、qwen-plus 或 MiniMax + Z-Image） |
| `qwen_vl_api.py` | 从图片提取视觉元素（Qwen3-VL） |
| `prompt_rewriter_with_image.py` | 视频描述改写（Gemini、qwen-plus 或 MiniMax） |
| `launch_sglang_server.sh` | SGLang 服务启动脚本（需设置环境变量 `MOVA_MODEL_PATH` 为模型目录） |
| `launch_streamlit.sh` | Streamlit 应用启动脚本 |
