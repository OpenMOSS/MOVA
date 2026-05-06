import os
import io
import json
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
import librosa
import pyloudnorm as pyln

from IS import Cnn14Extractor, calculate_inception_score, device, sr

import laion_clap
from clap_module.factory import load_state_dict
from transformers import RobertaTokenizer, RobertaModel


torch.serialization.add_safe_globals([np.core.multiarray.scalar])


def extract_audio_from_video_memory(video_path: str, target_sr: int = 44100) -> np.ndarray:
    cmd = [
        "ffmpeg", "-i", video_path, "-f", "wav", "-acodec", "pcm_s16le",
        "-ar", str(target_sr), "-ac", "2", "pipe:1", "-loglevel", "error"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = proc.communicate()
    audio, _ = librosa.load(io.BytesIO(out), sr=target_sr, mono=True)
    return audio


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype(np.int16)


def clap_single_score(model, text_emb: torch.Tensor, audio: np.ndarray, sr: int = 48000) -> float:
    audio = pyln.normalize.peak(audio, -1.0)
    audio = audio.reshape(1, -1)
    audio = torch.from_numpy(int16_to_float32(float32_to_int16(audio))).float().to(device)

    with torch.no_grad():
        audio_embeddings = model.get_audio_embedding_from_data(x=audio, use_tensor=True).to(device)
    cosine_sim = torch.nn.functional.cosine_similarity(
        audio_embeddings,
        text_emb.unsqueeze(0).to(device),
        dim=1,
        eps=1e-8,
    )[0]
    return float(cosine_sim.item())


def extract_numeric_prefix(filename: str) -> str:
    prefix = ""
    for c in filename:
        if c.isdigit():
            prefix += c
        else:
            break
    return prefix


def build_id_to_prompt_from_meta(meta_dict: Dict) -> Dict[str, str]:
    """Build mapping from meta ID to prompt"""
    id_to_prompt: Dict[str, str] = {}
    for category, items in meta_dict.items():
        if isinstance(items, dict):
            for item_id, item_data in items.items():
                if isinstance(item_data, dict) and "prompt" in item_data:
                    id_to_prompt[item_id] = item_data["prompt"]
    return id_to_prompt


def build_filename_to_prompt_mapping(id_to_prompt: Dict[str, str], video_files: List[str]) -> Dict[str, str]:
    """1127 revision: direct mapping from full filename to prompt"""
    filename_to_prompt = {}
    
    for video_file in video_files:
        filename = os.path.splitext(os.path.basename(video_file))[0]  # Remove extension
        
        matched_prompt = None
        
        # Match directly using meta ID (starting with clip_xxx)
        for meta_id in id_to_prompt.keys():
            if filename.startswith(meta_id):
                matched_prompt = id_to_prompt[meta_id]
                break
        
        # Extract clip ID portion from filename
        if matched_prompt is None:
            # Match "clip_xxx" format
            import re
            clip_match = re.search(r'(clip_[a-f0-9]+)', filename)
            if clip_match:
                clip_id = clip_match.group(1)
                if clip_id in id_to_prompt:
                    matched_prompt = id_to_prompt[clip_id]
        
        # Extract numeric prefix from filename
        if matched_prompt is None:
            numeric_prefix = extract_numeric_prefix(filename)
            if numeric_prefix and numeric_prefix in id_to_prompt:
                matched_prompt = id_to_prompt[numeric_prefix]
        
        # Debug output
        if matched_prompt is None:
            print(f"[DEBUG] File '{filename}' could not match any meta ID")
            print(f"[DEBUG] Available meta ID examples: {list(id_to_prompt.keys())[:5]}")  # Show first 5
        
        if matched_prompt:
            filename_to_prompt[filename] = matched_prompt
        else:
            print(f"[WARN] Could not find matching prompt for file {filename}")
    
    return filename_to_prompt

def setup_models() -> Tuple[Cnn14Extractor, laion_clap.CLAP_Module]:
    print("Loading Cnn14 model for IS...")
    is_model = Cnn14Extractor()

    print("Loading CLAP model...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=True, device=str(device))

    roberta_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'roberta-base')
    clap_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clap_ckpt')
    clap_ckpt = "630k-audioset-fusion-best.pt"

    if os.path.isdir(roberta_local):
        print(f"[INFO] Using local roberta-base: {roberta_local}")
        tokenizer = RobertaTokenizer.from_pretrained(roberta_local, local_files_only=True)
        text_encoder = RobertaModel.from_pretrained(roberta_local, local_files_only=True).to(device)
    else:
        print("[WARN] Local roberta-base not found, falling back to HuggingFace Hub download 'roberta-base'")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        text_encoder = RobertaModel.from_pretrained("roberta-base").to(device)

    clap_model.tokenize = tokenizer
    clap_model.model.text_branch = text_encoder

    pkg = load_state_dict(os.path.join(clap_local, clap_ckpt))
    pkg.pop("text_branch.embeddings.position_ids", None)
    clap_model.model.load_state_dict(pkg, strict=False)
    clap_model.eval()

    return is_model, clap_model


def precompute_text_embeddings(clap_model, prompts: List[str]) -> Dict[str, torch.Tensor]:
    """Pre-compute text embeddings for all prompts"""
    print("Extracting text embeddings from prompts...")
    text_embs: Dict[str, torch.Tensor] = {}
    for prompt in tqdm(prompts, desc="Computing text embeddings"):
        with torch.no_grad():
            emb = clap_model.get_text_embedding([prompt], use_tensor=True)[0].to(device)
        text_embs[prompt] = emb
    return text_embs


def evaluate_video_list(
    is_model: Cnn14Extractor,
    clap_model,
    id_to_prompt: Dict[str, str],
    video_paths: List[str],
    sample_rate: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns:
        is_score (float): Inception Score computed across the given list.
        clap_scores (dict): mapping from absolute video path -> CLAP score.
    """
    all_is_preds: List[np.ndarray] = []
    clap_scores: Dict[str, float] = {}

    # Build filename-to-prompt mapping
    filename_to_prompt = build_filename_to_prompt_mapping(id_to_prompt, video_paths)
    
    # Collect all unique prompts
    unique_prompts = list(set(filename_to_prompt.values()))
    
    # Pre-compute text embeddings for all prompts
    text_embs = precompute_text_embeddings(clap_model, unique_prompts)

    for video_path in tqdm(video_paths, desc="Evaluating videos (IS+CLAP)"):
        if not video_path.endswith(".mp4"):
            continue
        
        fname = os.path.basename(video_path)
        filename_key = os.path.splitext(fname)[0]  # Full filename without extension
        
        # Check if there is a matching prompt
        if filename_key not in filename_to_prompt:
            print(f"[WARN] {fname} has no matching prompt, skipping CLAP")
            # Still compute IS for inclusion in IS set
            try:
                audio = extract_audio_from_video_memory(video_path, target_sr=sample_rate)
                wav_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
                prob = is_model(wav_tensor).squeeze().detach().cpu().numpy()
                all_is_preds.append(prob)
            except Exception as e:
                print(f"[WARN] Error processing {fname} (IS stage): {e}")
            continue

        try:
            audio = extract_audio_from_video_memory(video_path, target_sr=sample_rate)
        except Exception as e:
            print(f"[WARN] Failed to extract audio for {fname}: {e}")
            continue

        # IS
        try:
            wav_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
            prob = is_model(wav_tensor).squeeze().detach().cpu().numpy()
            all_is_preds.append(prob)
        except Exception as e:
            print(f"[WARN] Failed to compute IS for {fname}: {e}")

        # CLAP
        try:
            prompt = filename_to_prompt[filename_key]
            text_emb = text_embs[prompt]
            clap_val = clap_single_score(clap_model, text_emb, audio, sr)
            clap_scores[video_path] = clap_val
        except Exception as e:
            print(f"[WARN] Failed to compute CLAP for {fname}: {e}")

    if len(all_is_preds) == 0:
        is_score = 0.0
    else:
        all_is_array = np.vstack(all_is_preds)
        is_score = float(calculate_inception_score(all_is_array))

    return is_score, clap_scores


def save_per_video_scores(
    output_dir: str,
    is_score: float,
    clap_scores: Dict[str, float],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for abs_video_path, clap_val in clap_scores.items():
        base = os.path.basename(abs_video_path)
        save_name = base[:-4] if base.endswith('.mp4') else base
        out_json = os.path.join(output_dir, f"{save_name}_eval_audio_score.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({"IS": is_score, "CLAP": float(clap_val)}, f, ensure_ascii=False, indent=4)


def evaluate_direct_directory(input_dir: str, id_to_prompt: Dict[str, str]) -> None:
    print("🎬 Direct directory evaluation mode (audio)")
    print(f"📁 Input directory: {input_dir}")
    video_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4')]
    video_paths = sorted(video_paths)
    if not video_paths:
        print(f"❌ No .mp4 files found in {input_dir}")
        import sys
        sys.exit(0)

    is_model, clap_model = setup_models()
    is_score, clap_scores = evaluate_video_list(is_model, clap_model, id_to_prompt, video_paths, sr)
    save_per_video_scores(input_dir, is_score, clap_scores)

    # Means
    clap_mean = float(np.mean(list(clap_scores.values()))) if len(clap_scores) > 0 else 0.0
    nested_means = {
        'IS': {'direct': {'all': float(is_score)}},
        'CLAP': {'direct': {'all': clap_mean}},
    }
    mean_json_path = os.path.join(input_dir, '000eval_audio_scores_means.json')
    with open(mean_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_means, f, ensure_ascii=False, indent=4)
    print(f"💾 Saved mean scores to: {mean_json_path}")
    print("✅ Direct evaluation completed.")


def merge_audio_scores_files(output_root: str) -> None:
    import glob
    from collections import defaultdict

    pattern = os.path.join(output_root, "eval_audio_scores_means_steps_*.json")
    step_files = [f for f in glob.glob(pattern) if not f.endswith("_all_step.json")]
    if len(step_files) == 0:
        print(f"📄 Found {len(step_files)} step files, no need to merge")
        return

    print(f"🔄 Found {len(step_files)} step files, merging...")
    merged_data = defaultdict(lambda: defaultdict(dict))
    for step_file in step_files:
        try:
            with open(step_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            filename = os.path.basename(step_file)
            step_info = filename.replace("eval_audio_scores_means_steps_", "").replace(".json", "")
            for score_name, step_data in data.items():
                for step, category_data in step_data.items():
                    merged_key = f"{step}"
                    merged_data[score_name][merged_key] = category_data
        except Exception as e:
            print(f"⚠️ Error reading {step_file}: {e}")

    merged_file = os.path.join(output_root, "eval_audio_scores_means.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    print(f"✅ Merged {len(step_files)} files into {merged_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Audio IS + CLAP Evaluation with step/category support")
    parser.add_argument('-i', "--exp_root", type=str,
                        default=None,
                        help="experiment root directory")
    parser.add_argument("--prompt_meta_json", type=str,
                        default=None,
                        help="Path to input meta JSON template")

    parser.add_argument("--video_save_path_subdir_name", type=str, default="eval_videos",
                        help="Subdirectory under exp_root where generated videos are saved")
    parser.add_argument("--video_eval_audio_output_subdir_name", type=str, default=None,
                        help="Subdirectory under exp_root where audio evaluation results will be saved (default: same as --video_save_path_subdir_name)")
    parser.add_argument("--input_dir_direct", type=str, default=None,
                        help="If set to a directory path, read videos directly from this directory without step/category structure")
    parser.add_argument("--specific_steps", type=int, nargs="+", default=None,
                        help="Specific step checkpoints to evaluate (e.g., --specific_steps 1000 2000 3000)")

    args = parser.parse_args()

    # Load meta and build id_to_prompt (only needed in pipeline mode)
    id_to_prompt = {}
    if args.input_dir_direct is None and args.prompt_meta_json is not None:
        with open(args.prompt_meta_json, 'r', encoding='utf-8') as f:
            meta_dict = json.load(f)
        id_to_prompt = build_id_to_prompt_from_meta(meta_dict)

    # Direct mode
    if args.input_dir_direct is not None and len(args.input_dir_direct) > 0:
        evaluate_direct_directory(args.input_dir_direct, id_to_prompt)
        import sys
        sys.exit(0)

    # Step/category mode
    if args.video_eval_audio_output_subdir_name is None:
        args.video_eval_audio_output_subdir_name = args.video_save_path_subdir_name

    input_root = os.path.join(args.exp_root, args.video_save_path_subdir_name)
    output_root = os.path.join(args.exp_root, args.video_eval_audio_output_subdir_name)
    os.makedirs(output_root, exist_ok=True)

    all_step_dirs = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])

    # Filter steps based on specific_steps if provided
    if args.specific_steps is not None:
        step_dirs = []
        for step in args.specific_steps:
            step_dir_name = f"step_{step}"
            if step_dir_name in all_step_dirs:
                step_dirs.append(step_dir_name)
            else:
                print(f"⚠️ Warning: Step directory {step_dir_name} not found in {input_root}")
        if not step_dirs:
            print("❌ None of the requested steps are available")
            import sys
            sys.exit(0)
        print(f"📌 Using specific steps: {[int(d.split('_')[1]) for d in step_dirs]}")
    else:
        step_dirs = all_step_dirs

    # Prepare models once
    is_model, clap_model = setup_models()

    from collections import defaultdict
    nested_means = defaultdict(lambda: defaultdict(dict))  # {score_name: {step: {category: mean, 'all': mean}}}

    # Determine categories from meta (keys that have prompt entries)
    categories = []
    for category, items in meta_dict.items():
        if isinstance(items, dict):
            prompts = [(k, v['prompt']) for k, v in items.items() if isinstance(v, dict) and 'prompt' in v]
            if prompts:
                categories.append(category)

    for step_dir in tqdm(step_dirs, desc="Processing steps"):
        print(f"\n🔄 Processing step: {step_dir}")
        input_step_dir = os.path.join(input_root, step_dir)
        output_step_dir = os.path.join(output_root, step_dir)
        os.makedirs(output_step_dir, exist_ok=True)

        for category in categories:
            print(f"📁 Processing category: {category}")
            category_input_dir = os.path.join(input_step_dir, category)
            category_output_dir = os.path.join(output_step_dir, category)
            os.makedirs(category_output_dir, exist_ok=True)

            # Gather videos under category
            all_paths: List[str] = []
            if os.path.exists(category_input_dir):
                for root, _, files in os.walk(category_input_dir):
                    for f in files:
                        if f.endswith('.mp4'):
                            all_paths.append(os.path.join(root, f))
            all_paths = sorted(all_paths)

            if not all_paths:
                print(f"⚠️ No video files found in {category_input_dir}")
                continue

            print(f"📹 Found {len(all_paths)} videos in {category}")

            is_score, clap_scores = evaluate_video_list(is_model, clap_model, id_to_prompt, all_paths, sr)
            save_per_video_scores(category_output_dir, is_score, clap_scores)

            # Means for this category
            clap_mean = float(np.mean(list(clap_scores.values()))) if len(clap_scores) > 0 else 0.0
            nested_means['IS'][step_dir][category] = float(is_score)
            nested_means['CLAP'][step_dir][category] = clap_mean
            print(f"📈 Category means -> IS: {is_score:.4f}, CLAP: {clap_mean:.4f}")

        # After finishing categories in this step, compute overall means
        print(f"📊 Computing overall means for {step_dir}...")
        for score_name in ['IS', 'CLAP']:
            step_dict = nested_means[score_name][step_dir]
            if step_dict:  # at least one category
                all_mean = float(np.mean(list(step_dict.values())))
                step_dict['all'] = all_mean
                print(f"📈 {score_name} overall mean: {all_mean:.4f}")
        print(f"✅ Step {step_dir} completed")

    # Save final means file
    if (args.video_save_path_subdir_name == args.video_eval_audio_output_subdir_name and
        args.specific_steps is not None and len(args.specific_steps) > 0):
        steps_str = "_".join(map(str, sorted(args.specific_steps)))
        mean_json_filename = f"eval_audio_scores_means_steps_{steps_str}.json"
    else:
        mean_json_filename = "eval_audio_scores_means.json"

    mean_json_path = os.path.join(output_root, mean_json_filename)
    print(f"💾 Saving final results to: {mean_json_path}")
    with open(mean_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_means, f, ensure_ascii=False, indent=4)

    # Merge multiple step files if needed
    if (args.video_save_path_subdir_name == args.video_eval_audio_output_subdir_name and
        args.specific_steps is not None and len(args.specific_steps) > 0):
        print("🔄 Merging step files...")
        merge_audio_scores_files(output_root)

    print("\n🎉 === Audio Evaluation Completed ===")


if __name__ == "__main__":
    main()


