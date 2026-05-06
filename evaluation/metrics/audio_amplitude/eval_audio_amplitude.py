#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audio Amplitude and Loudness Evaluation Script
Computes audio amplitude (RMS) and loudness metrics for video audio tracks

Note: This script uses the same audio extraction method as train_dual_tower.py
to ensure consistency between training and evaluation.
"""

import argparse
import io
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from torchcodec.decoders import AudioDecoder

# Try to import audiotools, with fallback for loudness calculation
try:
    from audiotools import AudioSignal
    AUDIOTOOLS_AVAILABLE = True
    print("✅ audiotools available")
except ImportError:
    AUDIOTOOLS_AVAILABLE = False
    print("⚠️ audiotools not available, will use pyloudnorm as fallback for loudness")
    try:
        import pyloudnorm as pyln
        PYLOUDNORM_AVAILABLE = True
    except ImportError:
        PYLOUDNORM_AVAILABLE = False
        print("⚠️ pyloudnorm not available either, loudness calculation disabled")

# Default sampling rate (same as train_dual_tower.py default: 22050)
SAMPLING_RATE = 48000


def extract_audio_from_video(video_path: str, target_sr: int = 22050, clip_duration_s=None):
    """Extract audio from video using AudioDecoder (same as train_dual_tower.py).
    
    This implementation matches train_dual_tower.py's extract_audio_from_video method
    to ensure consistency between training and evaluation.
    
    Args:
        video_path: Path to the input video file
        target_sr: Target sampling rate for audio
        clip_duration_s: Optional clip duration in seconds (if None, extracts full audio)
        
    Returns:
        Tuple of (waveform, sample_rate) where waveform is a torch.Tensor of shape (channels, samples)
    """
    try:
        # Use AudioDecoder to extract audio from video (same as training)
        audio_decoder = AudioDecoder(video_path)
        samples = audio_decoder.get_samples_played_in_range(0, clip_duration_s)
        waveform, original_sr = samples.data, samples.sample_rate
    except (ValueError, RuntimeError) as e:
        print(f"[ERROR] Failed to extract audio from {video_path}: {e}")
        raise
    
    # Resample if needed (same as training)
    if target_sr and original_sr != target_sr:
        from torchaudio.functional import resample
        waveform = resample(waveform, original_sr, target_sr)
        original_sr = target_sr
    
    # Handle positive PTS offset by padding silence (same as training)
    if samples.pts_seconds > 0:
        pad_samples = int(round(samples.pts_seconds * original_sr))
        if pad_samples > 0:
            pad = torch.zeros((waveform.size(0), pad_samples), dtype=waveform.dtype, device=waveform.device)
            waveform = torch.cat([pad, waveform], dim=1)
    
    # Convert to mono (same as training)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform, original_sr


def compute_audio_amplitude(waveform) -> float:
    """Compute RMS amplitude of audio waveform.
    
    Args:
        waveform: Audio waveform (torch.Tensor or numpy array)
        
    Returns:
        RMS amplitude value
    """
    # Convert to numpy if needed
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    rms = np.sqrt(np.mean(waveform ** 2))
    return float(rms)


def compute_audio_loudness(waveform, sr: int) -> float:
    """Compute audio loudness using audiotools or pyloudnorm.
    
    Args:
        waveform: Audio waveform (torch.Tensor or numpy array)
        sr: Sample rate
        
    Returns:
        Loudness value in LUFS
    """
    # Convert to numpy if needed
    if isinstance(waveform, torch.Tensor):
        waveform_np = waveform.cpu().numpy()
    else:
        waveform_np = waveform
    
    if AUDIOTOOLS_AVAILABLE:
        # Use audiotools (preferred method, same as training)
        # AudioSignal expects shape (channels, samples)
        if waveform_np.ndim == 1:
            waveform_2d = waveform_np[np.newaxis, :]
        else:
            waveform_2d = waveform_np.T if waveform_np.shape[0] > waveform_np.shape[1] else waveform_np
        
        audio_signal = AudioSignal(waveform_2d, sr)
        loudness = audio_signal.loudness()
        return float(loudness)
    
    elif PYLOUDNORM_AVAILABLE:
        # Fallback to pyloudnorm
        # Convert to 1D if needed
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.flatten()
        
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(waveform_np)
        return float(loudness)
    
    else:
        # No loudness library available
        return 0.0




def evaluate_video_list(
    video_paths: List[str],
    sample_rate: int = SAMPLING_RATE,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a list of videos and return amplitude/loudness scores.
    
    Args:
        video_paths: List of video file paths
        sample_rate: Audio sampling rate
        
    Returns:
        Dictionary mapping video paths to scores.
        Failed videos are skipped (not included in results) to avoid affecting mean calculations.
    """
    scores: Dict[str, Dict[str, float]] = {}
    
    for video_path in tqdm(video_paths, desc="Evaluating audio amplitude/loudness"):
        if not video_path.endswith(".mp4"):
            continue
        
        try:
            # Extract audio from video
            waveform, sr = extract_audio_from_video(video_path, target_sr=sample_rate)
            
            # Compute amplitude (RMS)
            amplitude = compute_audio_amplitude(waveform)
            
            # Compute loudness
            loudness = compute_audio_loudness(waveform, sr)
            
            # Only add to scores if successful
            scores[video_path] = {
                'amplitude_rms': amplitude,
                'loudness_lufs': loudness
            }
        except Exception as e:
            # Skip failed videos (like IS/CLAP and Lip sync do)
            # This prevents failed videos from affecting the mean calculation
            print(f"[WARN] Failed to process {os.path.basename(video_path)}: {e}. Skipping.")
            continue
    
    return scores


def save_per_video_scores(
    output_dir: str,
    scores: Dict[str, Dict[str, float]],
) -> None:
    """Save individual JSON files for each video with amplitude/loudness scores."""
    os.makedirs(output_dir, exist_ok=True)
    for abs_video_path, score_dict in scores.items():
        base = os.path.basename(abs_video_path)
        save_name = base[:-4] if base.endswith('.mp4') else base
        out_json = os.path.join(output_dir, f"{save_name}_eval_audio_amplitude_score.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(score_dict, f, ensure_ascii=False, indent=4)


def evaluate_direct_directory(
    input_dir: str,
) -> None:
    """Evaluate all videos in a flat directory (direct mode)."""
    print("🎬 Direct directory evaluation mode (Audio Amplitude/Loudness)")
    print(f"📁 Input directory: {input_dir}")
    
    video_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4')]
    video_paths = sorted(video_paths)
    
    if not video_paths:
        print(f"❌ No .mp4 files found in {input_dir}")
        import sys
        sys.exit(0)

    print(f"📹 Found {len(video_paths)} videos")
    
    # Evaluate videos
    scores = evaluate_video_list(video_paths, SAMPLING_RATE)
    
    # Save per-video scores
    save_per_video_scores(input_dir, scores)
    
    # Compute and save means
    amplitude_values = [s['amplitude_rms'] for s in scores.values()]
    loudness_values = [s['loudness_lufs'] for s in scores.values()]
    
    amplitude_mean = float(np.mean(amplitude_values)) if len(amplitude_values) > 0 else 0.0
    loudness_mean = float(np.mean(loudness_values)) if len(loudness_values) > 0 else 0.0
    
    nested_means = {
        'amplitude_rms': {'direct': {'all': amplitude_mean}},
        'loudness_lufs': {'direct': {'all': loudness_mean}},
    }
    
    mean_json_path = os.path.join(input_dir, '000eval_audio_amplitude_scores_means.json')
    with open(mean_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_means, f, ensure_ascii=False, indent=4)
    
    print(f"💾 Saved mean scores to: {mean_json_path}")
    print(f"📈 Amplitude RMS mean: {amplitude_mean:.6f}")
    print(f"📈 Loudness LUFS mean: {loudness_mean:.2f}")
    print("✅ Direct evaluation completed.")


def merge_scores_files(output_root: str) -> None:
    """Merge multiple eval_audio_amplitude_scores_means_steps_*.json files."""
    import glob
    
    pattern = os.path.join(output_root, "eval_audio_amplitude_scores_means_steps_*.json")
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
            step_info = filename.replace("eval_audio_amplitude_scores_means_steps_", "").replace(".json", "")
            for score_name, step_data in data.items():
                for step, category_data in step_data.items():
                    merged_key = f"{step}"
                    merged_data[score_name][merged_key] = category_data
        except Exception as e:
            print(f"⚠️ Error reading {step_file}: {e}")
    
    merged_file = os.path.join(output_root, "eval_audio_amplitude_scores_means.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    print(f"✅ Merged {len(step_files)} files into {merged_file}")


def main():
    parser = argparse.ArgumentParser(description="Audio Amplitude and Loudness Evaluation with step/category support")
    parser.add_argument('-i', "--exp_root", type=str,
                        default=None,
                        help="Experiment root directory, e.g., /path/to/experiment")
    parser.add_argument("--prompt_meta_json", type=str,
                        default=None,
                        help="Path to input meta JSON template")
    parser.add_argument("--video_save_path_subdir_name", type=str, default="eval_videos",
                        help="Subdirectory under exp_root where generated videos are saved")
    parser.add_argument("--video_eval_audio_output_subdir_name", type=str, default=None,
                        help="Subdirectory under exp_root where evaluation results will be saved (default: same as --video_save_path_subdir_name)")
    parser.add_argument("--input_dir_direct", type=str, default=None,
                        help="If set to a directory path, read videos directly from this directory without step/category structure")
    parser.add_argument("--specific_steps", type=int, nargs="+", default=None,
                        help="Specific step checkpoints to evaluate (e.g., --specific_steps 1000 2000 3000)")

    args = parser.parse_args()

    # Direct mode
    if args.input_dir_direct is not None and len(args.input_dir_direct) > 0:
        evaluate_direct_directory(args.input_dir_direct)
        import sys
        sys.exit(0)

    # Load meta and get categories
    with open(args.prompt_meta_json, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)

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

    # Determine categories from meta
    categories = []
    for category, items in meta_dict.items():
        if isinstance(items, dict):
            prompts = [(k, v['prompt']) for k, v in items.items() if isinstance(v, dict) and 'prompt' in v]
            if prompts:
                categories.append(category)

    nested_means = defaultdict(lambda: defaultdict(dict))

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

            # Evaluate videos
            scores = evaluate_video_list(all_paths, SAMPLING_RATE)
            
            # Save per-video scores
            save_per_video_scores(category_output_dir, scores)

            # Compute means for this category
            amplitude_values = [s['amplitude_rms'] for s in scores.values()]
            loudness_values = [s['loudness_lufs'] for s in scores.values()]
            
            amplitude_mean = float(np.mean(amplitude_values)) if len(amplitude_values) > 0 else 0.0
            loudness_mean = float(np.mean(loudness_values)) if len(loudness_values) > 0 else 0.0
            
            nested_means['amplitude_rms'][step_dir][category] = amplitude_mean
            nested_means['loudness_lufs'][step_dir][category] = loudness_mean
            
            print(f"📈 Category mean -> Amplitude RMS: {amplitude_mean:.6f}, Loudness LUFS: {loudness_mean:.2f}")

        # After finishing categories in this step, compute overall means
        print(f"📊 Computing overall means for {step_dir}...")
        amp_step_dict = nested_means['amplitude_rms'][step_dir]
        loud_step_dict = nested_means['loudness_lufs'][step_dir]
        
        if amp_step_dict:  # at least one category
            amp_all_mean = float(np.mean(list(amp_step_dict.values())))
            amp_step_dict['all'] = amp_all_mean
            print(f"📈 Amplitude RMS overall mean: {amp_all_mean:.6f}")
        
        if loud_step_dict:  # at least one category
            loud_all_mean = float(np.mean(list(loud_step_dict.values())))
            loud_step_dict['all'] = loud_all_mean
            print(f"📈 Loudness LUFS overall mean: {loud_all_mean:.2f}")
        
        print(f"✅ Step {step_dir} completed")

    # Save final means file
    if (args.video_save_path_subdir_name == args.video_eval_audio_output_subdir_name and
        args.specific_steps is not None and len(args.specific_steps) > 0):
        steps_str = "_".join(map(str, sorted(args.specific_steps)))
        mean_json_filename = f"eval_audio_amplitude_scores_means_steps_{steps_str}.json"
    else:
        mean_json_filename = "eval_audio_amplitude_scores_means.json"

    mean_json_path = os.path.join(output_root, mean_json_filename)
    print(f"💾 Saving final results to: {mean_json_path}")
    with open(mean_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_means, f, ensure_ascii=False, indent=4)

    # Merge multiple step files if needed
    if (args.video_save_path_subdir_name == args.video_eval_audio_output_subdir_name and
        args.specific_steps is not None and len(args.specific_steps) > 0):
        print("🔄 Merging step files...")
        merge_scores_files(output_root)

    print("\n🎉 === Audio Amplitude/Loudness Evaluation Completed ===")


if __name__ == "__main__":
    main()

