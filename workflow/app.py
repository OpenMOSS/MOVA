#!/usr/bin/env python3
"""
SGLang Video Generation Workflow - Streamlit App

A web interface for video generation using SGLang server.
Supports both Full Workflow (with AI-assisted prompt generation) and Simple Mode.
"""

import streamlit as st
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from sglang_client import SGLangClient, VideoTask
from config import (
    SGLANG_SERVERS,
    DEFAULT_VIDEO_PARAMS,
    GEMINI_API_URL,
    GEMINI_API_KEY,
    QWEN_VL_API_KEY,
    MINIMAX_API_KEY,
    POLL_INTERVAL,
    TASK_TIMEOUT,
    OUTPUT_DIR,
    TASKS_FILE
)

# ============================================================================
# Constants
# ============================================================================

SCRIPT_DIR = Path(__file__).parent


def _normalize_task_dir(task_dir: Optional[str]) -> Optional[Path]:
    """
    Normalize task_dir path to absolute path.
    If task_dir is relative, resolve it relative to SCRIPT_DIR.
    If task_dir is None, return None.
    """
    if not task_dir:
        return None
    task_path = Path(task_dir)
    if task_path.is_absolute():
        return task_path
    # If relative, resolve relative to SCRIPT_DIR
    return SCRIPT_DIR / task_path


def _create_task_dir() -> Tuple[Path, str]:
    """Create task output directory and return (task_dir, timestamp)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir = SCRIPT_DIR / OUTPUT_DIR / f"task_{timestamp}"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir, timestamp


def _get_valid_server_key(server_key: Optional[str] = None) -> str:
    """Return a valid server key from config, for fallback when loading tasks."""
    if server_key and server_key in SGLANG_SERVERS:
        return server_key
    return list(SGLANG_SERVERS.keys())[0] if SGLANG_SERVERS else 'mova-360p'


# ============================================================================
# Session State Management
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'page': 'generate',
        'generating': False,
        'generation_complete': False,
        'current_task_id': None,
        'video_result': None,
        'poll_mode': 'task_queue',  # 'poll_wait' or 'task_queue'
        'workflow_mode': 'full',    # 'full' or 'simple'
        'show_task_details': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# Task Storage
# ============================================================================

def load_tasks() -> Dict[str, Any]:
    """Load tasks from JSON file."""
    tasks_path = SCRIPT_DIR / TASKS_FILE
    if tasks_path.exists():
        try:
            with open(tasks_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_tasks(tasks: Dict[str, Any]):
    """Save tasks to JSON file."""
    tasks_path = SCRIPT_DIR / TASKS_FILE
    with open(tasks_path, 'w') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)


def add_task(task_data: Dict[str, Any]):
    """Add a new task."""
    tasks = load_tasks()
    tasks[task_data['id']] = task_data
    save_tasks(tasks)


def update_task(task_id: str, updates: Dict[str, Any]):
    """Update an existing task."""
    tasks = load_tasks()
    if task_id in tasks:
        tasks[task_id].update(updates)
        save_tasks(tasks)


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific task."""
    tasks = load_tasks()
    return tasks.get(task_id)


# ============================================================================
# Sidebar Configuration
# ============================================================================

def _get_video_size_from_server(server_key: str, orientation: str) -> str:
    """Derive video size from server's default_size and orientation."""
    default_size = SGLANG_SERVERS[server_key].get('default_size', '640x360')
    try:
        w, h = default_size.lower().split('x')
        w, h = int(w), int(h)
        if orientation == 'portrait':
            return f"{min(w, h)}x{max(w, h)}"
        return f"{max(w, h)}x{min(w, h)}"
    except (ValueError, AttributeError):
        return default_size


def render_sidebar() -> Dict[str, Any]:
    """Render sidebar and return configuration."""
    st.sidebar.title("Configuration")
    
    server_keys = list(SGLANG_SERVERS.keys())
    
    # Server selection: hide dropdown when only 1 model, show when multiple
    if len(server_keys) == 1:
        server_key = server_keys[0]
        server_info = SGLANG_SERVERS[server_key]
        st.sidebar.markdown(f"**Model:** {server_info['name']}")
        if server_info.get('description'):
            st.sidebar.caption(server_info['description'])
    else:
        server_key = st.sidebar.selectbox(
            "Model Server",
            options=server_keys,
            format_func=lambda x: SGLANG_SERVERS[x]['name']
        )
    
    # Video orientation
    orientation = st.sidebar.radio(
        "Orientation",
        options=['landscape', 'portrait'],
        horizontal=True
    )
    
    # Get video size from server config
    video_size = _get_video_size_from_server(server_key, orientation)
    st.sidebar.info(f"Video Size: {video_size}")
    
    # Advanced parameters
    with st.sidebar.expander("Advanced Parameters"):
        num_frames = st.number_input(
            "Number of Frames",
            min_value=1,
            max_value=500,
            value=DEFAULT_VIDEO_PARAMS['num_frames']
        )
        
        fps = st.number_input(
            "FPS",
            min_value=1,
            max_value=60,
            value=DEFAULT_VIDEO_PARAMS['fps']
        )
        
        seed = st.number_input(
            "Seed (0 = random)",
            min_value=0,
            max_value=999999999,
            value=0,
            help="Set to 0 for random seed each time"
        )
        
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=DEFAULT_VIDEO_PARAMS['guidance_scale'],
            step=0.5
        )
        
        num_inference_steps = st.number_input(
            "Inference Steps",
            min_value=1,
            max_value=100,
            value=DEFAULT_VIDEO_PARAMS['num_inference_steps']
        )
    
    # API Keys (for full workflow)
    with st.sidebar.expander("API Keys"):
        gemini_url = st.text_input(
            "Gemini API URL",
            value=GEMINI_API_URL,
            type="default"
        )

        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=GEMINI_API_KEY or "",
            type="password"
        )

        qwen_key = st.text_input(
            "Qwen API Key",
            value=QWEN_VL_API_KEY or "",
            type="password"
        )

        minimax_key = st.text_input(
            "MiniMax API Key",
            value=MINIMAX_API_KEY or "",
            type="password",
            help="MiniMax API key for prompt generation (OpenAI-compatible). Get one at https://platform.minimaxi.com/"
        )

    return {
        'server_key': server_key,
        'video_size': video_size,
        'orientation': orientation,
        'num_frames': num_frames,
        'fps': fps,
        'seed': seed,
        'guidance_scale': guidance_scale,
        'num_inference_steps': num_inference_steps,
        'gemini_url': gemini_url,
        'gemini_api_key': gemini_api_key,
        'qwen_key': qwen_key,
        'minimax_key': minimax_key
    }


# ============================================================================
# Generate Page
# ============================================================================

def render_generate_page(config: Dict[str, Any]):
    """Render the generate page."""
    st.title("Video Generation")
    
    # Check if generation is in progress
    if st.session_state.generating:
        if st.session_state.poll_mode == 'poll_wait' and st.session_state.current_task_id:
            render_poll_wait_status(config)
        return
    
    # Check if generation is complete
    if st.session_state.generation_complete and st.session_state.video_result:
        render_generation_results()
        return
    
    # Mode selection
    col1, col2 = st.columns(2)
    
    with col1:
        workflow_mode = st.radio(
            "Workflow Mode",
            options=['full', 'simple'],
            format_func=lambda x: 'Full Workflow (AI-assisted)' if x == 'full' else 'Simple Mode (Direct)',
            index=0 if st.session_state.workflow_mode == 'full' else 1,
            horizontal=True
        )
        st.session_state.workflow_mode = workflow_mode
    
    with col2:
        poll_mode = st.radio(
            "Async Mode",
            options=['task_queue', 'poll_wait'],
            format_func=lambda x: 'Task Queue (Submit & Check Later)' if x == 'task_queue' else 'Poll & Wait (Real-time)',
            index=0 if st.session_state.poll_mode == 'task_queue' else 1,
            horizontal=True
        )
        st.session_state.poll_mode = poll_mode
    
    st.markdown("---")
    
    # Render appropriate form
    if workflow_mode == 'simple':
        render_simple_mode_form(config)
    else:
        render_full_workflow_form(config)


def render_simple_mode_form(config: Dict[str, Any]):
    """Render simple mode input form."""
    st.subheader("Simple Mode")
    st.info("Provide a prompt and first frame image directly.")
    
    prompt = st.text_area(
        "Video Prompt",
        height=150,
        placeholder="Describe the video you want to generate..."
    )
    
    uploaded_file = st.file_uploader(
        "First Frame Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload the first frame for the video"
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="First Frame Preview", width=300)
    
    if st.button("Generate Video", type="primary", disabled=not (prompt and uploaded_file)):
        submit_simple_mode_task(prompt, uploaded_file, config)


def render_full_workflow_form(config: Dict[str, Any]):
    """Render full workflow input form."""
    st.subheader("Full Workflow")
    st.info("Enter a description and optionally provide an image. AI will help generate the prompt and first frame.")
    
    user_input = st.text_area(
        "Video Description",
        height=150,
        placeholder="Describe what you want in the video..."
    )
    
    uploaded_file = st.file_uploader(
        "Reference Image (Optional)",
        type=['png', 'jpg', 'jpeg'],
        help="Optionally provide a reference image"
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="Reference Image", width=300)
    
    if st.button("Start Full Workflow", type="primary", disabled=not user_input):
        run_full_workflow(user_input, uploaded_file, config)


def submit_simple_mode_task(prompt: str, uploaded_file, config: Dict[str, Any]):
    """Submit a simple mode task."""
    st.session_state.generating = True
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        task_dir, timestamp = _create_task_dir()
        
        # Save first frame
        first_frame_path = task_dir / f"first_frame_{timestamp}.png"
        with open(first_frame_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Save user prompt (input in simple mode)
        user_input_path = task_dir / f"user_input_{timestamp}.txt"
        with open(user_input_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        progress_bar.progress(30)
        status_text.text("Submitting to SGLang server...")
        
        # Submit to SGLang
        client = SGLangClient(server_key=config['server_key'])
        # Generate random seed if seed is 0, so we can save the actual seed used
        import random
        actual_seed = config['seed'] if config['seed'] > 0 else random.randint(1, 2**31 - 1)
        # Pass actual_seed (even if random) to ensure client uses the same seed
        task = client.submit_video_task(
            prompt=prompt,
            image_path=str(first_frame_path),
            size=config['video_size'],
            num_frames=config['num_frames'],
            fps=config['fps'],
            seed=actual_seed,  # Use actual_seed (random if original was 0)
            guidance_scale=config['guidance_scale'],
            num_inference_steps=config['num_inference_steps']
        )
        
        st.success(f"Task submitted! ID: {task.id}")
        progress_bar.progress(50)
        
        # Save task to storage
        task_data = {
            'id': task.id,
            'status': task.status,
            'progress': task.progress,
            'created_at': task.created_at,
            'submitted_at': datetime.now().isoformat(),
            'image_generation_prompt': prompt,
            'original_input': prompt,
            'user_input_path': str(user_input_path),
            'server': config['server_key'],
            'params': {
                'size': config['video_size'],
                'num_frames': config['num_frames'],
                'fps': config['fps'],
                'seed': actual_seed,  # Save actual seed used (random if original was 0)
                'guidance_scale': config['guidance_scale'],
                'num_inference_steps': config['num_inference_steps']
            },
            'workflow_mode': 'simple',
            'task_dir': str(task_dir.resolve()),  # Ensure absolute path
            'first_frame_path': str(first_frame_path.resolve()),  # Ensure absolute path
            'video_path': None,
            'download_url': None
        }
        add_task(task_data)
        
        st.session_state.current_task_id = task.id
        
        # Handle based on async mode
        if st.session_state.poll_mode == 'poll_wait':
            st.rerun()
        else:
            st.session_state.generating = False
            progress_bar.progress(100)
            st.success("Task added to queue. Check the Task List page for status.")
            time.sleep(2)
            st.rerun()
    
    except Exception as e:
        st.session_state.generating = False
        st.error(f"Error: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def run_full_workflow(user_input: str, uploaded_file, config: Dict[str, Any]):
    """Run the full AI-assisted workflow."""
    st.session_state.generating = True
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        task_dir, timestamp = _create_task_dir()
        
        # Save user input to task_dir
        user_input_path = task_dir / f"user_input_{timestamp}.txt"
        with open(user_input_path, 'w', encoding='utf-8') as f:
            f.write(user_input)
        
        # Step 1: Handle first frame
        status_text.markdown("### 📍 Step 1/5: Preparing First Frame")
        progress_bar.progress(5)
        
        if uploaded_file:
            # User provided image
            first_frame_path = task_dir / f"first_frame_{timestamp}.png"
            with st.spinner("Saving uploaded image..."):
                with open(first_frame_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
            st.success("✅ Using provided image as first frame")
            st.image(str(first_frame_path), caption="First Frame", width=400)
            image_generation_prompt = None  # 用户上传图片，无 AI 生成的 image prompt
        else:
            # Generate first frame using AI (prompt + image in one call)
            status_text.markdown("### 📍 Step 1-2/5: Generating First Frame (AI)")
            progress_bar.progress(10)
            
            first_frame_path = task_dir / f"first_frame_{timestamp}.png"
            image_prompt_path = task_dir / f"image_generation_prompt_{timestamp}.txt"
            # 优先使用 Gemini：有 Gemini key 时始终用 Gemini；仅当无 Gemini 时才用 Qwen/Z-Image
            gemini_key = (config.get('gemini_api_key') or '').strip()
            qwen_key = (config.get('qwen_key') or '').strip()
            minimax_key = (config.get('minimax_key') or '').strip()
            spinner_msg = "🔄 Generating image prompt and first frame..." + (
                " (Gemini)" if gemini_key else (" (MiniMax + Z-Image)" if minimax_key and not qwen_key else " (通义 qwen-plus + Z-Image)")
            )
            with st.spinner(spinner_msg):
                # Map orientation to aspect ratio for first frame generation
                aspect_ratio = '9:16' if config['orientation'] == 'portrait' else '16:9'
                cmd = [
                    'python', str(SCRIPT_DIR / 'generate_first_frame.py'),
                    'full',
                    '--user-input', user_input,
                    '--output', str(first_frame_path),
                    '--output-image-prompt', str(image_prompt_path),
                    '--aspect-ratio', aspect_ratio,  # Use frontend orientation setting
                ]
                if gemini_key:
                    cmd.extend(['--api-key', gemini_key, '--api-url', config['gemini_url']])
                if qwen_key:
                    cmd.extend(['--qwen-api-key', qwen_key])
                if minimax_key:
                    cmd.extend(['--minimax-api-key', minimax_key])
                first_frame_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',  # Replace invalid characters instead of raising error
                    cwd=str(SCRIPT_DIR)
                )
            
            if first_frame_result.returncode != 0:
                error_msg = first_frame_result.stderr.strip() if first_frame_result.stderr else first_frame_result.stdout.strip()
                if not error_msg:
                    error_msg = f"Exit code: {first_frame_result.returncode}"
                raise Exception(f"Failed to generate first frame: {error_msg}")
            
            st.success("✅ First frame generated")
            st.image(str(first_frame_path), caption="Generated First Frame", width=400)
            
            # 读取 image generation prompt 供后续保存到 task_data
            image_generation_prompt = None
            if image_prompt_path.exists():
                image_generation_prompt = image_prompt_path.read_text(encoding='utf-8').strip()
        
        progress_bar.progress(30)
        
        # Step 3: Extract visual elements
        status_text.markdown("### 📍 Step 3/5: Extracting Visual Elements")
        
        # Save image elements to file
        elements_path = task_dir / f"image_elements_{timestamp}.json"
        
        with st.spinner("🔄 Analyzing image with Qwen3-VL..."):
            qwen_result = subprocess.run(
                ['python', str(SCRIPT_DIR / 'qwen_vl_api.py'),
                 '--image', str(first_frame_path),
                 '--api-key', config['qwen_key'],
                 '--output', str(elements_path),
                 '--format', 'json'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace invalid characters instead of raising error
                cwd=str(SCRIPT_DIR)
            )
        
        if qwen_result.returncode != 0:
            st.warning(f"⚠️ Visual element extraction failed: {qwen_result.stderr}")
            with open(elements_path, 'w', encoding='utf-8') as f:
                json.dump({"visual_description": ""}, f, ensure_ascii=False)
        else:
            try:
                with open(elements_path, 'r', encoding='utf-8') as f:
                    elements_data = json.load(f)
                st.success("✅ Visual elements extracted")
                with st.expander("View Visual Elements"):
                    st.json(elements_data)
            except Exception as e:
                st.warning(f"⚠️ Failed to read image elements: {e}")
                with open(elements_path, 'w', encoding='utf-8') as f:
                    json.dump({"visual_description": ""}, f, ensure_ascii=False)
        
        progress_bar.progress(45)
        
        # Step 4: Generate video description
        status_text.markdown("### 📍 Step 4/5: Generating Video Description")
        
        with st.spinner("🔄 Rewriting prompt with visual elements..."):
            # 优先使用 Gemini；无 Gemini key 时使用 qwen-plus
            rewriter_cmd = [
                'python', str(SCRIPT_DIR / 'prompt_rewriter_with_image.py'),
                '--user-input', user_input,
                '--image-elements-file', str(elements_path),
            ]
            if (config.get('gemini_api_key') or '').strip():
                rewriter_cmd.extend(['--api-url', config['gemini_url'], '--api-key', config['gemini_api_key']])
            if (config.get('qwen_key') or '').strip():
                rewriter_cmd.extend(['--qwen-api-key', config['qwen_key']])
            if (config.get('minimax_key') or '').strip():
                rewriter_cmd.extend(['--minimax-api-key', config['minimax_key']])
            rewriter_result = subprocess.run(
                rewriter_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace invalid characters instead of raising error
                cwd=str(SCRIPT_DIR)
            )
        
        if rewriter_result.returncode != 0:
            st.warning("⚠️ Video description generation failed, using original input.")
            video_description = user_input
        else:
            video_description = rewriter_result.stdout.strip()
            # If stdout is empty, fall back to user_input
            if not video_description:
                st.warning("⚠️ Video description is empty, using original input.")
                video_description = user_input
            else:
                word_count = len(video_description.split())
                st.success(f"✅ Video description generated ({word_count} words)")
                with st.expander("View Video Description"):
                    st.text_area("Description", video_description, height=200, disabled=True)
        
        # Validate video_description is not empty
        if not video_description or not video_description.strip():
            raise ValueError("Video description cannot be empty. Please provide a valid description.")
        
        # Validate first_frame_path exists
        if not Path(first_frame_path).exists():
            raise FileNotFoundError(f"First frame image not found: {first_frame_path}")
        
        # Save video generation prompt
        desc_path = task_dir / f"video_generation_prompt_{timestamp}.txt"
        with open(desc_path, 'w', encoding='utf-8') as f:
            f.write(video_description)
        
        progress_bar.progress(60)
        
        # Step 5: Submit to SGLang
        status_text.markdown("### 📍 Step 5/5: Submitting to SGLang Server")
        
        # Validate all required parameters before submission
        if not video_description or not video_description.strip():
            raise ValueError("Prompt cannot be empty")
        if not Path(first_frame_path).exists():
            raise FileNotFoundError(f"Image file not found: {first_frame_path}")
        
        # Debug: Show submission parameters
        st.info(f"📋 Submission Parameters:\n- Prompt length: {len(video_description)} chars\n- Image: {first_frame_path}\n- Size: {config['video_size']}\n- Server: {config['server_key']}")
        
        with st.spinner("🔄 Submitting video generation task..."):
            try:
                client = SGLangClient(server_key=config['server_key'])
                # Generate random seed if seed is 0, so we can save the actual seed used
                import random
                actual_seed = config['seed'] if config['seed'] > 0 else random.randint(1, 2**31 - 1)
                # Pass actual_seed (even if random) to ensure client uses the same seed
                
                task = client.submit_video_task(
                    prompt=video_description,
                    image_path=str(first_frame_path),
                    size=config['video_size'],
                    num_frames=config['num_frames'],
                    fps=config['fps'],
                    seed=actual_seed,  # Use actual_seed (random if original was 0)
                    guidance_scale=config['guidance_scale'],
                    num_inference_steps=config['num_inference_steps']
                )
            except Exception as submit_error:
                st.error(f"❌ Failed to submit task: {str(submit_error)}")
                import traceback
                with st.expander("Submission Error Details"):
                    st.code(traceback.format_exc())
                raise
        
        st.success(f"✅ Task submitted! ID: `{task.id}`")
        st.info(f"📊 Status: {task.status} | Progress: {task.progress}%")
        progress_bar.progress(70)
        
        # Save task to storage
        task_data = {
            'id': task.id,
            'status': task.status,
            'progress': task.progress,
            'created_at': task.created_at,
            'submitted_at': datetime.now().isoformat(),
            'original_input': user_input,
            'server': config['server_key'],
            'params': {
                'size': config['video_size'],
                'num_frames': config['num_frames'],
                'fps': config['fps'],
                'seed': actual_seed,  # Save actual seed used (random if original was 0)
                'guidance_scale': config['guidance_scale'],
                'num_inference_steps': config['num_inference_steps']
            },
            'workflow_mode': 'full',
            'task_dir': str(task_dir.resolve()),  # Ensure absolute path
            'user_input_path': str(user_input_path.resolve()),  # Ensure absolute path
            'first_frame_path': str(first_frame_path.resolve()),  # Ensure absolute path
            'image_elements_path': str(elements_path.resolve()),  # Ensure absolute path
            **({'image_generation_prompt': image_generation_prompt} if image_generation_prompt else {}),
            'video_generation_prompt': video_description,
            'video_path': None,
            'download_url': None
        }
        add_task(task_data)
        
        st.session_state.current_task_id = task.id
        
        # Handle based on async mode
        if st.session_state.poll_mode == 'poll_wait':
            status_text.markdown("### Step 5/5: Waiting for Video Generation")
            st.rerun()
        else:
            # Task queue mode
            st.session_state.generating = False
            progress_bar.progress(100)
            status_text.markdown("### Task Queued!")
            st.success("Task added to queue. Check the Task List page for status.")
            time.sleep(2)
            st.rerun()
    
    except Exception as e:
        st.session_state.generating = False
        st.error(f"Error in workflow: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


# ============================================================================
# Poll Wait Mode
# ============================================================================

def render_poll_wait_status(config: Dict[str, Any]):
    """Render poll wait status page."""
    task_id = st.session_state.current_task_id
    
    if not task_id:
        st.error("No task ID found")
        st.session_state.generating = False
        return
    
    st.subheader(f"Waiting for Task: {task_id}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    info_container = st.container()
    
    try:
        client = SGLangClient(server_key=config['server_key'])
        
        # Poll for status
        start_time = time.time()
        last_status = None
        
        while True:
            task = client.get_task_status(task_id)
            
            # Update display
            progress = task.progress if task.progress else 0
            progress_bar.progress(progress / 100)
            
            status_text.markdown(f"**Status:** {task.status} | **Progress:** {progress}%")
            
            with info_container:
                if task.status != last_status:
                    last_status = task.status
                    st.info(f"Task status: {task.status}")
            
            # Check completion
            if task.status == 'completed':
                # Update task in storage
                download_url = client.get_download_url(task_id)
                
                # Get current task data for task_dir
                current_task_data = get_task(task_id)
                task_dir_str = current_task_data.get('task_dir') if current_task_data else None
                task_dir = _normalize_task_dir(task_dir_str)
                
                updates = {
                    'status': 'completed',
                    'progress': 100,
                    'completed_at': task.completed_at,
                    'download_url': download_url,
                    'inference_time_s': task.inference_time_s,
                    'peak_memory_mb': task.peak_memory_mb
                }
                
                # Download video to local storage
                video_path = None
                if task_dir:
                    video_path = task_dir / f"{task_id}.mp4"
                    try:
                        st.info(f"Downloading video to {video_path}...")
                        client.download_video(task_id, str(video_path))
                        updates['video_path'] = str(video_path.resolve())  # Ensure absolute path
                        st.success(f"Video saved to {video_path}")
                    except Exception as download_error:
                        error_msg = str(download_error)
                        st.error(f"Could not download video: {error_msg}")
                        import traceback
                        st.code(traceback.format_exc())
                        print(f"Download error for task {task_id}: {error_msg}")
                        print(traceback.format_exc())
                
                update_task(task_id, updates)
                
                # Set results
                task_data = get_task(task_id)
                st.session_state.video_result = {
                    'task_id': task_id,
                    'download_url': download_url,
                    'video_path': str(video_path) if video_path else None,
                    'task_data': task_data,
                    'inference_time': task.inference_time_s,
                    'first_frame_path': task_data.get('first_frame_path') if task_data else None
                }
                st.session_state.generation_complete = True
                st.session_state.generating = False
                st.rerun()
                break
            
            if task.status == 'error' or task.status == 'failed':
                update_task(task_id, {'status': 'error', 'error': task.error})
                st.session_state.generating = False
                st.error(f"Task failed: {task.error}")
                break
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > TASK_TIMEOUT:
                st.session_state.generating = False
                st.error(f"Task timed out after {TASK_TIMEOUT} seconds")
                break
            
            # Wait before next poll
            time.sleep(POLL_INTERVAL)
            st.rerun()
    
    except Exception as e:
        st.session_state.generating = False
        st.error(f"Error polling task: {str(e)}")


def render_generation_results():
    """Render generation results."""
    st.title("Generation Complete!")
    
    result = st.session_state.video_result
    
    # Video display
    st.subheader("Generated Video")
    
    # Prefer local video_path over download_url
    video_source = result.get('video_path') or result.get('download_url')
    if video_source:
        try:
            # Check if it's a local file path
            if result.get('video_path') and Path(result['video_path']).exists():
                st.video(result['video_path'])
                st.info(f"📁 Local file: `{result['video_path']}`")
            elif result.get('download_url'):
                st.video(result['download_url'])
        except Exception as e:
            st.warning(f"Could not display video directly: {e}")
    
    # Show both paths if available
    if result.get('video_path'):
        st.markdown(f"**Local File:** `{result['video_path']}`")
    if result.get('download_url'):
        st.markdown(f"**Download URL:** [{result.get('download_url', 'N/A')}]({result.get('download_url', '#')})")
    
    # Task info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Task ID:** `{result.get('task_id')}`")
        if result.get('inference_time'):
            st.markdown(f"**Inference Time:** {result.get('inference_time'):.2f}s")
    
    with col2:
        if result.get('first_frame_path') and Path(result['first_frame_path']).exists():
            st.image(result['first_frame_path'], caption="First Frame", width=200)
    
    # Task details
    if result.get('task_data'):
        with st.expander("Full Task Details"):
            st.json(result['task_data'])
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Another Video"):
            st.session_state.generation_complete = False
            st.session_state.video_result = None
            st.session_state.current_task_id = None
            st.rerun()
    
    with col2:
        if st.button("View Task List"):
            st.session_state.page = 'tasks'
            st.session_state.generation_complete = False
            st.session_state.video_result = None
            st.rerun()


# ============================================================================
# Task List Page
# ============================================================================

def render_task_list_page(config: Dict[str, Any]):
    """Render the task list page."""
    st.title("Task List")
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Refresh All Status"):
            refresh_all_tasks()
            st.rerun()
    
    tasks = load_tasks()
    
    if not tasks:
        st.info("No tasks yet. Generate your first video!")
        return
    
    # Sort by submission time (newest first)
    sorted_tasks = sorted(
        tasks.items(),
        key=lambda x: x[1].get('submitted_at', ''),
        reverse=True
    )
    
    for task_id, task_data in sorted_tasks:
        render_task_card(task_id, task_data, config)


def refresh_all_tasks():
    """Refresh status for all non-completed tasks."""
    tasks = load_tasks()
    
    for task_id, task_data in tasks.items():
        if task_data.get('status') not in ['completed', 'error', 'failed']:
            try:
                server_key = _get_valid_server_key(task_data.get('server'))
                client = SGLangClient(server_key=server_key)
                task = client.get_task_status(task_id)
                
                updates = {
                    'status': task.status,
                    'progress': task.progress
                }
                
                if task.status == 'completed':
                    updates['completed_at'] = task.completed_at
                    updates['download_url'] = client.get_download_url(task_id)
                    updates['inference_time_s'] = task.inference_time_s
                    updates['peak_memory_mb'] = task.peak_memory_mb
                    
                    # Download video to local storage
                    task_dir_str = task_data.get('task_dir')
                    task_dir = _normalize_task_dir(task_dir_str)
                    if task_dir:
                        video_path = task_dir / f"{task_id}.mp4"
                        try:
                            st.info(f"Downloading video to {video_path}...")
                            client.download_video(task_id, str(video_path))
                            updates['video_path'] = str(video_path.resolve())  # Ensure absolute path
                            st.success(f"Video saved to {video_path}")
                        except Exception as download_error:
                            error_msg = str(download_error)
                            st.error(f"Could not download video: {error_msg}")
                            import traceback
                            st.code(traceback.format_exc())
                            print(f"Download error for task {task_id}: {error_msg}")
                            print(traceback.format_exc())
                
                update_task(task_id, updates)
            except Exception as e:
                st.warning(f"Could not refresh task {task_id[:8]}...: {str(e)}")


def render_task_card(task_id: str, task_data: Dict[str, Any], config: Dict[str, Any]):
    """Render a single task card with video preview."""
    status = task_data.get('status', 'unknown')
    progress = task_data.get('progress', 0)
    
    # Status mapping - SGLang API confirmed behavior:
    # - Only returns: queued, completed, failed
    # - NO intermediate states (no running/in_progress/processing)
    # - progress stays 0 until completion (jumps to 100)
    
    # Status color and display name
    # Note: "queued" actually means "排队或生成中" since SGLang doesn't distinguish
    status_display = {
        'queued': ('🔵', '排队/生成中'),  # Could be waiting OR actively processing
        'running': ('🔵', '生成中'),       # For compatibility if SGLang adds this later
        'processing': ('🔵', '生成中'),
        'in_progress': ('🔵', '生成中'),
        'generating': ('🔵', '生成中'),
        'completed': ('🟢', '已完成'),
        'error': ('🔴', '失败'),
        'failed': ('🔴', '失败'),
        'unknown': ('⚪', '未知')
    }
    
    # Get display info, default to showing the raw status with unknown icon
    status_icon, status_text = status_display.get(status, ('⚪', status))
    
    # Task card
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        
        with col1:
            st.markdown(f"**{task_id[:12]}...**")
            st.caption(task_data.get('submitted_at', 'Unknown time'))
        
        with col2:
            st.markdown(f"{status_icon} {status_text}")
            if status == 'queued':
                st.progress(0)
        
        with col3:
            # Refresh single task
            if st.button("Refresh", key=f"refresh_{task_id}"):
                refresh_single_task(task_id, task_data, config)
                st.rerun()
        
        with col4:
            # Download button
            if status == 'completed':
                download_url = task_data.get('download_url')
                if download_url:
                    st.markdown(f"[Download]({download_url})")
            
            # Details button
            if st.button("Details", key=f"details_{task_id}"):
                st.session_state.show_task_details = task_id
        
        # Video preview for completed tasks (always visible)
        # Prefer local video_path over download_url
        video_path = task_data.get('video_path')
        download_url = task_data.get('download_url')
        if status == 'completed' and (video_path or download_url):
            col_video, col_info = st.columns([2, 1])
            with col_video:
                try:
                    # Prefer local file if available
                    if video_path and Path(video_path).exists():
                        st.video(video_path)
                        st.caption(f"📁 Local: `{video_path}`")
                    elif download_url:
                        st.video(download_url)
                except Exception as e:
                    st.warning(f"Video preview not available: {e}")
            with col_info:
                st.markdown(f"**Size:** {task_data.get('params', {}).get('size', 'N/A')}")
                st.markdown(f"**Frames:** {task_data.get('params', {}).get('num_frames', 'N/A')}")
                st.markdown(f"**Seed:** {task_data.get('params', {}).get('seed', 'N/A')}")
                if task_data.get('inference_time_s'):
                    st.markdown(f"**Time:** {task_data['inference_time_s']:.1f}s")
        
        # Show details if selected
        if st.session_state.show_task_details == task_id:
            with st.expander("Task Details", expanded=True):
                # Show original user input (Full workflow)
                if task_data.get('workflow_mode') == 'full' and task_data.get('original_input'):
                    st.markdown("**User Input:**")
                    st.text_area("User Input", task_data['original_input'], height=80, disabled=True, key=f"input_{task_id}", label_visibility="collapsed")
                # Show Image Generation Prompt（simple 模式或 full 模式有则显示）
                img_prompt = task_data.get('image_generation_prompt') or task_data.get('prompt')
                if img_prompt:
                    disp = img_prompt[:500] + ("..." if len(img_prompt) > 500 else "")
                    st.markdown("**Image Generation Prompt:**")
                    st.text_area("Image Generation Prompt", disp, height=80, disabled=True, key=f"img_prompt_{task_id}", label_visibility="collapsed")
                # Show Video Generation Prompt（full 模式）
                vid_prompt = task_data.get('video_generation_prompt') or task_data.get('video_description')
                if vid_prompt:
                    disp = vid_prompt[:500] + ("..." if len(vid_prompt) > 500 else "")
                    st.markdown("**Video Generation Prompt:**")
                    st.text_area("Video Generation Prompt", disp, height=100, disabled=True, key=f"vid_prompt_{task_id}", label_visibility="collapsed")
                
                # Show first frame if available
                first_frame = task_data.get('first_frame_path')
                if first_frame:
                    # Normalize path (handle both relative and absolute paths)
                    first_frame_path = Path(first_frame)
                    if not first_frame_path.is_absolute():
                        first_frame_path = SCRIPT_DIR / first_frame_path
                    if first_frame_path.exists():
                        st.markdown("**First Frame:**")
                        st.image(str(first_frame_path), width=300)
                
                # Full JSON data
                st.markdown("**Raw Data:**")
                st.json(task_data)
                
                if st.button("Close Details", key=f"close_{task_id}"):
                    st.session_state.show_task_details = None
                    st.rerun()
        
        st.markdown("---")


def refresh_single_task(task_id: str, task_data: Dict[str, Any], config: Dict[str, Any]):
    """Refresh status for a single task."""
    try:
        server_key = _get_valid_server_key(task_data.get('server'))
        client = SGLangClient(server_key=server_key)
        task = client.get_task_status(task_id)
        
        updates = {
            'status': task.status,
            'progress': task.progress
        }
        
        if task.status == 'completed':
            updates['completed_at'] = task.completed_at
            updates['download_url'] = client.get_download_url(task_id)
            updates['inference_time_s'] = task.inference_time_s
            updates['peak_memory_mb'] = task.peak_memory_mb
            
            # Download video to local storage
            task_dir_str = task_data.get('task_dir')
            task_dir = _normalize_task_dir(task_dir_str)
            if task_dir:
                video_path = task_dir / f"{task_id}.mp4"
                try:
                    st.info(f"Downloading video to {video_path}...")
                    client.download_video(task_id, str(video_path))
                    updates['video_path'] = str(video_path.resolve())  # Ensure absolute path
                    st.success(f"Video saved to {video_path}")
                except Exception as download_error:
                    error_msg = str(download_error)
                    st.error(f"Could not download video: {error_msg}")
                    import traceback
                    st.code(traceback.format_exc())
                    print(f"Download error for task {task_id}: {error_msg}")
                    print(traceback.format_exc())
        
        update_task(task_id, updates)
        st.success(f"Task {task_id[:8]}... refreshed: {task.status}")
    except Exception as e:
        st.error(f"Error refreshing task: {str(e)}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="SGLang Video Generation",
        page_icon="🎬",
        layout="wide"
    )
    
    init_session_state()
    config = render_sidebar()
    
    # Navigation buttons at the top
    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        if st.button("🎬 Generate Video", use_container_width=True, type="primary" if st.session_state.page == 'generate' else "secondary"):
            st.session_state.page = 'generate'
            st.rerun()
    with col2:
        if st.button("📋 Task List", use_container_width=True, type="primary" if st.session_state.page == 'tasks' else "secondary"):
            st.session_state.page = 'tasks'
            st.rerun()
    
    st.markdown("---")
    
    # Route to appropriate page
    if st.session_state.page == 'generate':
        render_generate_page(config)
    elif st.session_state.page == 'tasks':
        render_task_list_page(config)


if __name__ == "__main__":
    main()
