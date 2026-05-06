import ast
import json
import os
from collections.abc import Mapping
import pandas as pd
import torch
from vision_process import process_vision_info
from data import DataConfig
from utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from utils import load_model_from_checkpoint
from train_reward import create_model_and_processor
from prompt_template import build_prompt

def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return config_dict["data_config"], None, config_dict["model_config"], config_dict["peft_lora_config"], \
           config_dict["inference_config"] if "inference_config" in config_dict else None

class VideoVLMRewardInference():
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=False,
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )
        
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            reward['MQ'] = (reward['MQ'] - self.inference_config['MQ_mean']) / self.inference_config['MQ_std']
            reward['TA'] = (reward['TA'] - self.inference_config['TA_mean']) / self.inference_config['TA_std']
            return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None,):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """
        Inputs:
            video_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."

        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)

        # Get reward token IDs for extracting hidden states at special token positions
        reward_token_ids = [
            self.processor.tokenizer.convert_tokens_to_ids(t)
            for t in ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
        ]

        # Forward pass with hidden states to extract reward values via rm_head
        outputs = self.model(
            return_dict=True,
            output_hidden_states=True,
            **batch
        )
        # Use last-layer hidden states; shape: (batch, seq_len, hidden_size)
        hidden_states = outputs.hidden_states[-1]

        # Find positions of reward special tokens in the input_ids
        input_ids = batch["input_ids"]  # (batch, seq_len)
        rewards = []
        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            reward_values = []
            for rtid in reward_token_ids:
                positions = (ids == rtid).nonzero(as_tuple=True)
                if len(positions[0]) > 0:
                    pos = positions[0][0].item()
                    h = hidden_states[b, pos]  # (hidden_size,)
                    reward_values.append(self.model.rm_head(h).item())
                else:
                    # Fallback: use last token hidden state
                    reward_values.append(self.model.rm_head(hidden_states[b, -1]).item())
            rewards.append({'VQ': reward_values[0], 'MQ': reward_values[1], 'TA': reward_values[2]})

        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']

        return rewards

# Helpers for structure and clarity

def compute_overall_means(rewards_list):
    return {
        'VQ': sum(r['VQ'] for r in rewards_list)/len(rewards_list),
        'MQ': sum(r['MQ'] for r in rewards_list)/len(rewards_list),
        'TA': sum(r['TA'] for r in rewards_list)/len(rewards_list),
        'Overall': sum(r['VQ'] + r['MQ'] + r['TA'] for r in rewards_list)/len(rewards_list),
    }


def get_video_frame_count(video_file_path):
    """Return total frame count for a video using OpenCV if available; otherwise None."""
    try:
        import cv2
    except Exception:
        return None
    try:
        cap = cv2.VideoCapture(str(video_file_path))
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count if frame_count > 0 else None
    except Exception:
        return None


def format_suffix(fps_val, frames_val):
    if fps_val is not None:
        if float(fps_val).is_integer():
            return f"_fps_{int(fps_val)}"
        return f"_fps_{fps_val}"
    if frames_val is not None:
        return f"_frames_{int(frames_val)}"
    return ""


def merge_reward_results_files(score_root):
    """Merge multiple eval_video_reward_results_steps_*.json files into eval_video_reward_results_steps_all_step.json"""
    import glob
    from collections import defaultdict
    
    # Find all matching step files
    pattern = os.path.join(score_root, "eval_video_reward_results_steps_*.json")
    step_files = glob.glob(pattern)
    
    # Filter out the all_step file itself
    step_files = [f for f in step_files if not f.endswith("_all_step.json")]
    
    if len(step_files) <= 0:
        print(f"📄 Found {len(step_files)} step files, no need to merge")
        return
    
    print(f"🔄 Found {len(step_files)} step files, merging...")
    
    # Merge all data
    merged_data = {}
    
    for step_file in step_files:
        try:
            with open(step_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract step info from filename
            filename = os.path.basename(step_file)
            # From "eval_video_reward_results_steps_1000_2000.json" extract "1000_2000"
            step_info = filename.replace("eval_video_reward_results_steps_", "").replace(".json", "")
            
            # Merge data
            for step, step_data in data.items():
                merged_key = f"{step}"  # Create unique key
                # merged_key = f"{step}_{step_info}"  # Create unique key
                merged_data[merged_key] = step_data
                    
        except Exception as e:
            print(f"⚠️ Error reading {step_file}: {e}")
            continue
    
    # Save merged file
    merged_file = os.path.join(score_root, "eval_video_reward_results.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Merged {len(step_files)} files into {merged_file}")
    
    # Optional: delete original step files
    # for step_file in step_files:
    #     os.remove(step_file)
    #     print(f"🗑️ Removed {step_file}")


def eval_direct_videos_with_prompts(inferencer, input_dir, state_json_name, direct_eval_save_path, meta_prompts, fps=None, frames=None, batch_size=50):
    """Read generation_state.json jobs to build (video_path, prompt) and evaluate all videos.
    Save per-video rewards to input_dir, and a combined JSON as eval_video_reward_results.json.
    """
    print(f"\n🎬 Direct mode evaluation")
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {direct_eval_save_path}")
    print(f"🔧 Batch size: {batch_size}")
    
    state_json_path = os.path.join(input_dir, state_json_name)
    print(f"📋 Loading: {state_json_path}")
    
    if not os.path.exists(state_json_path):
        print(f"❌ generation_state.json not found: {state_json_path}")
        return False
    try:
        with open(state_json_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {state_json_path}: {e}")
        return False

    jobs = state_data.get('jobs', [])
    if not jobs:
        print(f"❌ No jobs found in {state_json_path}")
        return False
    
    print(f"📊 Found {len(jobs)} jobs")

    videos, prompts, video_names = [], [], []
    missing_video_files = []
    for job in jobs:
        file_path = job.get('file_path')
        prompt_text = job.get('prompt', "")
        if not file_path:
            continue
        video_file_name = os.path.basename(file_path)
        video_full_path = os.path.join(input_dir, video_file_name)
        if os.path.exists(video_full_path):
            videos.append(video_full_path)
            prompts.append(prompt_text)
            video_names.append(video_file_name)
        else:
            missing_video_files.append(video_full_path)

    if not videos:
        print(f"❌ No videos resolved from jobs in {state_json_path}")
        return False

    print(f"📹 Processing {len(videos)} videos")
    if missing_video_files:
        print(f"⚠️ Missing videos from jobs: {len(missing_video_files)}")
        print(f"⚠️ Sample missing paths: {missing_video_files[:3]}")
    print(f"📋 Sample videos: {video_names[:3]}{'...' if len(video_names) > 3 else ''}")
    
    os.makedirs(direct_eval_save_path, exist_ok=True)
    all_rewards = []
    per_video_suffix = {}

    if fps is not None and frames is not None:
        print("❌ Please provide only one of fps or frames in direct mode, not both.")
        return False

    if fps is not None:
        for i in range(0, len(videos), batch_size):
            batch_videos = videos[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            with torch.no_grad():
                batch_rewards = inferencer.reward(batch_videos, batch_prompts, fps=fps, use_norm=True)
            all_rewards.extend(batch_rewards)
        for p in videos:
            per_video_suffix[p] = format_suffix(fps, None)
    elif frames is not None:
        for i in range(0, len(videos), batch_size):
            batch_videos = videos[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            with torch.no_grad():
                batch_rewards = inferencer.reward(batch_videos, batch_prompts, num_frames=frames, use_norm=True)
            all_rewards.extend(batch_rewards)
        for p in videos:
            per_video_suffix[p] = format_suffix(None, frames)
    else:
        # Auto-detect per-video frame counts; evaluate in batches (per video within batch)
        for i in range(0, len(videos), batch_size):
            batch_videos = videos[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            for p, prm in zip(batch_videos, batch_prompts):
                frames_val = get_video_frame_count(p)
                with torch.no_grad():
                    res = inferencer.reward([p], [prm], num_frames=frames_val, use_norm=True)
                all_rewards.extend(res)
                per_video_suffix[p] = format_suffix(None, frames_val)

    if len(all_rewards) == 0:
        print("❌ No rewards produced in direct mode.")
        return False

    for vid_path, vid_name, reward in zip(videos, video_names, all_rewards):
        base_name = os.path.splitext(os.path.basename(vid_name))[0]
        suffix = per_video_suffix.get(vid_path, "")
        vid_score_path = os.path.join(direct_eval_save_path, f"{base_name}{suffix}_visual_score.json")
        with open(vid_score_path, 'w', encoding='utf-8') as f:
            json.dump(reward, f, ensure_ascii=False, indent=4)

    overall = compute_overall_means(all_rewards)
    all_results = {'direct': {'all': overall}}
    output_json = os.path.join(direct_eval_save_path, "000eval_video_reward_results.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"✅ Direct reward evaluation finished. Results saved to {output_json}")
    return True


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import json
    import os
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--exp_root', type=str, default=None, help="Experiment root directory")
    parser.add_argument("--prompt_meta_json", type=str, default=None, help="Path to input meta JSON template")
    parser.add_argument('--reward_model', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints'), help="Reward model directory")
    parser.add_argument('--reward_step', type=int, default=11352, help="Reward model checkpoint step")
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument(
        "--video_save_path_subdir_name", type=str, default="eval_videos",
        help="Subdirectory name under exp_root where generated videos are saved"
    )
    parser.add_argument(
        "--video_eval_video_quality_output_subdir_name", type=str, default=None,
        help="Subdirectory name under exp_root where video quality evaluation results will be saved (default: same as --video_save_path_subdir_name)"
    )
    parser.add_argument("--input_dir_direct", type=str, default=None, help="If set to a directory path, read videos/prompts directly from this directory without step/category structure")
    parser.add_argument("--generate_state_json_name", type=str, default="generation_state.json", help="Filename of generation state JSON under --input_dir_direct")
    parser.add_argument("--specific_steps", type=int, nargs="+", default=None, help="Specific step checkpoints to evaluate (e.g., --specific_steps 1000 2000 3000)")
    parser.add_argument("--fps", type=float, default=None, help="FPS sampling rate; cannot be used with --frames")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to sample; cannot be used with --fps")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for reward inference (default: 50)")

    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16

    print(f"🚀 Starting VideoReward evaluation")
    print(f"📊 Device: {device}")
    print(f"🔧 Batch size: {args.batch_size}")
    if args.fps is not None:
        print(f"🎞️ FPS sampling: {args.fps}")
    elif args.frames is not None:
        print(f"🎞️ Frame sampling: {args.frames}")
    else:
        print(f"🎞️ Auto-detect FPS/frames per video")

    # Load reward model
    print(f"🔧 Loading reward model from: {args.reward_model}")
    print(f"🔧 Checkpoint step: {args.reward_step}")
    inferencer = VideoVLMRewardInference(load_from_pretrained=args.reward_model, load_from_pretrained_step=args.reward_step, device=device, dtype=dtype)
    print("✅ Reward model loaded successfully")



    # Validate fps/frames mutual exclusivity
    if args.fps is not None and args.frames is not None:
        print("❌ Please provide only one of --fps or --frames, not both.")
        import sys
        sys.exit(1)

    # Direct directory mode: read generation_state.json jobs from the specified directory to determine videos and prompts
    if args.input_dir_direct is not None and len(args.input_dir_direct) > 0:
        print(f"🔍 Using direct directory mode with input directory: {args.input_dir_direct}")
        input_dir = args.input_dir_direct
        state_json_name = args.generate_state_json_name
        meta_prompts = {} # Placeholder, not used in direct mode
        ok = eval_direct_videos_with_prompts(
            inferencer, input_dir, state_json_name, input_dir, meta_prompts,
            fps=args.fps, frames=args.frames, batch_size=args.batch_size
        )
        if not ok:
            print("❌ Direct mode evaluation failed. Please check logs above.")
            import sys
            sys.exit(1)
        import sys
        sys.exit(0)




    # Load meta_json
    with open(args.prompt_meta_json, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)

    prompt_categories = {}  # {category: [(meta_id, prompt_text)]}
    for category in meta_dict:
        prompt_categories[category] = [(k, v['prompt']) for k, v in meta_dict[category].items()]

    # Iterate over each step in the experiment directory

    if args.video_eval_video_quality_output_subdir_name is None:
        args.video_eval_video_quality_output_subdir_name = args.video_save_path_subdir_name

    eval_root = os.path.join(args.exp_root, args.video_save_path_subdir_name)
    score_root = os.path.join(args.exp_root, args.video_eval_video_quality_output_subdir_name)
    print(f"📁 eval_root: {eval_root}")
    print(f"📁 score_root: {score_root}")
    if not os.path.exists(eval_root):
        print(f"❌ eval_root does not exist: {eval_root}")
        import sys
        sys.exit(1)

    # eval_root = os.path.join(args.exp_root, args.video_save_path_subdir_name)
    # score_root = os.path.join(args.exp_root, "eval_video_reward")

    os.makedirs(score_root, exist_ok=True)

    all_steps = [d for d in os.listdir(eval_root) if d.startswith("step_")]
    all_steps.sort(key=lambda x: int(x.split("_")[1]))
    print(f"📊 Found step directories: {len(all_steps)}")
    
    # Filter steps based on specific_steps if provided
    if args.specific_steps is not None:
        steps = []
        for step in args.specific_steps:
            step_dir_name = f"step_{step}"
            step_dir_name_padded = f"step_{step:06d}"
            if step_dir_name in all_steps:
                steps.append(step_dir_name)
            elif step_dir_name_padded in all_steps:
                # Compatible with 6-digit zero-padded format (e.g., MoVA 480p inference output)
                steps.append(step_dir_name_padded)
            else:
                print(f"⚠️ Warning: Step directory {step_dir_name} not found in {eval_root}")
        if not steps:
            print("❌ None of the requested steps are available")
            # return
            import sys
            sys.exit(1)
        print(f"📌 Using specific steps: {[int(s.split('_')[1]) for s in steps]}")
    else:
        steps = all_steps

    all_results = {}
    
    print(f"\n📊 Processing {len(steps)} steps")
    print(f"📌 Steps: {[int(s.split('_')[1]) for s in steps]}")

    for step in steps:
        print(f"\n🔄 Processing step: {step}")
        step_path = os.path.join(eval_root, step)
        step_score_path = os.path.join(score_root, step)
        os.makedirs(step_score_path, exist_ok=True)

        step_results = {}
        for category in meta_dict:
            category_path = os.path.join(step_path, category)
            if not os.path.exists(category_path):
                print(f"[DEBUG] Step {step}, category {category} path does not exist: {category_path}")
                continue

            video_files = [f for f in os.listdir(category_path) if f.endswith(".mp4")]
            if not video_files:
                print(f"[DEBUG] Step {step}, category {category} has no videos.")
                continue
            
            print(f"📁 Processing category: {category} ({len(video_files)} videos)")

            meta_id_to_video = {vf.split("_video_")[0]: os.path.join(category_path, vf) for vf in video_files}

            videos, prompts, video_names = [], [], []
            for meta_id, prompt_text in prompt_categories.get(category, []):
                if meta_id in meta_id_to_video:
                    videos.append(meta_id_to_video[meta_id])
                    prompts.append(prompt_text)
                    video_names.append(os.path.basename(meta_id_to_video[meta_id]))

            if not videos:
                available_meta = [vf.split("_")[0] for vf in video_files]
                expected_meta = [mid for mid, _ in prompt_categories.get(category, [])]
                print(f"[DEBUG] Step {step}, category {category} no matching videos for meta_ids.")
                print(
                    f"[DEBUG]   video_files={len(video_files)}, prompt_meta={len(expected_meta)}, "
                    f"matched=0"
                )
                print(f"[DEBUG]   sample video names: {video_files[:5]}")
                print(f"[DEBUG]   sample parsed video meta_ids: {available_meta[:5]}")
                print(f"[DEBUG]   sample prompt meta_ids: {expected_meta[:5]}")
                # Helpful stats to locate naming mismatch quickly.
                available_meta_set = set(available_meta)
                expected_meta_set = set(expected_meta)
                inter = available_meta_set.intersection(expected_meta_set)
                print(
                    f"[DEBUG]   meta_id overlap: {len(inter)} / "
                    f"video_meta={len(available_meta_set)} / prompt_meta={len(expected_meta_set)}"
                )
                if len(inter) > 0:
                    print(f"[DEBUG]   sample overlap meta_ids: {list(sorted(inter))[:5]}")
                continue
            print(f"[DEBUG] Step {step}, category {category} matched videos with prompts: {len(videos)}")

            batch_size = args.batch_size

            # Run reward evaluation
            from tqdm import tqdm
            all_rewards = []
            per_video_suffix = {}
            num_batches = (len(videos) + batch_size - 1) // batch_size
            print(f"🚀 [{step}/{category}] Starting reward inference: {len(videos)} videos in {num_batches} batches")
            
            if args.fps is not None:
                print(f"🎞️ Using fixed FPS: {args.fps}")
                for i in tqdm(range(0, len(videos), batch_size), desc=f"{step}/{category}", total=num_batches):
                    batch_videos = videos[i:i+batch_size]
                    batch_prompts = prompts[i:i+batch_size]
                    with torch.no_grad():
                        batch_rewards = inferencer.reward(batch_videos, batch_prompts, fps=args.fps, use_norm=True)
                    all_rewards.extend(batch_rewards)
                    if i == 0 and len(batch_rewards) > 0:
                        print(f"📊 Sample reward: {batch_rewards[0]}")
                for p in videos:
                    per_video_suffix[p] = format_suffix(args.fps, None)
            elif args.frames is not None:
                print(f"🎞️ Using fixed frames: {args.frames}")
                for i in tqdm(range(0, len(videos), batch_size), desc=f"{step}/{category}", total=num_batches):
                    batch_videos = videos[i:i+batch_size]
                    batch_prompts = prompts[i:i+batch_size]
                    with torch.no_grad():
                        batch_rewards = inferencer.reward(batch_videos, batch_prompts, num_frames=args.frames, use_norm=True)
                    all_rewards.extend(batch_rewards)
                    if i == 0 and len(batch_rewards) > 0:
                        print(f"📊 Sample reward: {batch_rewards[0]}")
                for p in videos:
                    per_video_suffix[p] = format_suffix(None, args.frames)
            else:
                # Auto-detect per-video frame counts; evaluate in batches (per video within batch)
                print(f"🎞️ Auto-detecting frames per video")
                for i in tqdm(range(0, len(videos), batch_size), desc=f"{step}/{category}", total=num_batches):
                    batch_videos = videos[i:i+batch_size]
                    batch_prompts = prompts[i:i+batch_size]
                    for p, prm in zip(batch_videos, batch_prompts):
                        frames_val = get_video_frame_count(p)
                        with torch.no_grad():
                            res = inferencer.reward([p], [prm], num_frames=frames_val, use_norm=True)
                        all_rewards.extend(res)
                        per_video_suffix[p] = format_suffix(None, frames_val)
                if len(all_rewards) > 0:
                    print(f"📊 Sample reward: {all_rewards[0]}")

            # Save per-video reward
            category_score_path = os.path.join(step_score_path, category)
            os.makedirs(category_score_path, exist_ok=True)
            for vid_path, vid_name, reward in zip(videos, video_names, all_rewards):

                base_name = os.path.splitext(os.path.basename(vid_name))[0]
                suffix = per_video_suffix.get(vid_path, "")
                vid_score_path = os.path.join(category_score_path, f"{base_name}{suffix}_visual_score.json")

                with open(vid_score_path, 'w', encoding='utf-8') as f:
                    json.dump(reward, f, ensure_ascii=False, indent=4)

            # Compute category means
            mean_reward = {
                'VQ': sum(r['VQ'] for r in all_rewards)/len(all_rewards),
                'MQ': sum(r['MQ'] for r in all_rewards)/len(all_rewards),
                'TA': sum(r['TA'] for r in all_rewards)/len(all_rewards),
            }
            step_results[category] = mean_reward

        # Compute means across all categories
        all_cat_rewards = [v for k, v in step_results.items() if k != 'all']
        if all_cat_rewards:
            step_results['all'] = {
                'VQ': sum(r['VQ'] for r in all_cat_rewards)/len(all_cat_rewards),
                'MQ': sum(r['MQ'] for r in all_cat_rewards)/len(all_cat_rewards),
                'TA': sum(r['TA'] for r in all_cat_rewards)/len(all_cat_rewards),
                'Overall': sum(r['VQ'] + r['MQ'] + r['TA'] for r in all_cat_rewards)/len(all_cat_rewards),
            }

        all_results[step] = step_results
        if len(step_results) == 0:
            print(f"⚠️ Step {step} produced no category scores.")
        print(f"✅ Step {step} completed")

    # Save step summary scores JSON
    print(f"\n💾 Saving results...")
    # Check if step info needs to be added to filename
    if (args.video_save_path_subdir_name == args.video_eval_video_quality_output_subdir_name and 
        args.specific_steps is not None and len(args.specific_steps) > 0):
        steps_str = "_".join(map(str, sorted(args.specific_steps)))
        output_json_filename = f"eval_video_reward_results_steps_{steps_str}.json"
    else:
        output_json_filename = "eval_video_reward_results.json"
    
    output_json = os.path.join(score_root, output_json_filename)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"💾 Results saved to: {output_json}")
    if len(all_results) == 0:
        print("⚠️ Final results are empty. Usually this means no valid step/category/video was matched.")

    # Check if multiple step files need to be merged
    if (args.video_save_path_subdir_name == args.video_eval_video_quality_output_subdir_name and 
        args.specific_steps is not None and len(args.specific_steps) > 0):
        print(f"🔄 Merging step files...")
        merge_reward_results_files(score_root)

    print(f"\n🎉 ===  VideoReward Evaluation Completed ===")
    print(f"✅ Processed {len(steps)} steps")
    print(f"📊 Results saved to: {output_json}")
