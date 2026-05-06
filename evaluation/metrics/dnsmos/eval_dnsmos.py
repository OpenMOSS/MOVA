#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNSMOS Evaluation Script for Video Audio Quality Assessment
Integrates with the evaluation pipeline for step/category-based evaluation
"""

import argparse
import io
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


def extract_audio_from_video_to_file(video_path: str, target_sr: int = 16000) -> str:
    """Extract audio from video and save to a temporary WAV file.
    
    Args:
        video_path: Path to the input video file
        target_sr: Target sampling rate for audio
        
    Returns:
        Path to the temporary WAV file
    """
    # Create a temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()
    
    cmd = [
        "ffmpeg", "-i", video_path, "-f", "wav", "-acodec", "pcm_s16le",
        "-ar", str(target_sr), "-ac", "1", temp_wav_path, "-loglevel", "error", "-y"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_wav_path
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to extract audio from {video_path}: {e}")
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        raise


class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path, use_gpu=True) -> None:
        """Initialize DNSMOS scorer with ONNX models.
        
        Args:
            primary_model_path: Path to primary ONNX model
            p808_model_path: Path to P808 ONNX model
            use_gpu: If True, attempt to use GPU acceleration (CUDA). Automatically falls back to CPU if unavailable.
        """
        # Try GPU first if requested
        if use_gpu:
            print("🚀 Attempting to use GPU acceleration for ONNX models...")
            try:
                # Check if CUDA provider is available
                available_providers = ort.get_available_providers()
                print(f"📋 Available providers: {available_providers}")
                
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.onnx_sess = ort.InferenceSession(primary_model_path, providers=providers)
                    self.p808_onnx_sess = ort.InferenceSession(p808_model_path, providers=providers)
                    
                    # Check which provider is actually being used
                    primary_provider = self.onnx_sess.get_providers()[0]
                    p808_provider = self.p808_onnx_sess.get_providers()[0]
                    
                    if 'CUDA' in primary_provider:
                        print(f"✅ Successfully using GPU acceleration!")
                        print(f"   Primary model: {primary_provider}")
                        print(f"   P808 model: {p808_provider}")
                        return
                    else:
                        print(f"⚠️  GPU requested but not used, provider: {primary_provider}")
                        print("   Falling back to CPU...")
                else:
                    print("⚠️  CUDAExecutionProvider not available in onnxruntime")
                    print("   This usually means:")
                    print("   1. onnxruntime-gpu is not installed (only onnxruntime)")
                    print("   2. CUDA is not properly installed")
                    print("   3. GPU is not available")
                    print("   Falling back to CPU...")
            except Exception as e:
                print(f"⚠️  Failed to initialize GPU: {e}")
                print("   Falling back to CPU...")
        else:
            print("💻 Using CPU for ONNX models (as requested)...")
        
        # CPU fallback
        providers = ['CPUExecutionProvider']
        self.onnx_sess = ort.InferenceSession(primary_model_path, providers=providers)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path, providers=providers)
        print(f"✅ Models loaded successfully on CPU")
        print(f"   Tip: Install onnxruntime-gpu and CUDA to enable GPU acceleration")
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate, is_personalized_MOS, only_p808=True):
        """Compute DNSMOS scores for an audio file.
        
        Args:
            fpath: Path to audio file
            sampling_rate: Target sampling rate
            is_personalized_MOS: Whether to use personalized MOS
            only_p808: If True, only compute P808_MOS (faster)
            
        Returns:
            Dictionary with scores
        """
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_p808_mos = []
        
        # Only compute other metrics if requested
        if not only_p808:
            predicted_mos_sig_seg_raw = []
            predicted_mos_bak_seg_raw = []
            predicted_mos_ovr_seg_raw = []
            predicted_mos_sig_seg = []
            predicted_mos_bak_seg = []
            predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            # Always compute P808_MOS
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            predicted_p808_mos.append(p808_mos)
            
            # Compute other metrics only if requested
            if not only_p808:
                input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
                oi = {'input_1': input_features}
                mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
                mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)
                predicted_mos_sig_seg_raw.append(mos_sig_raw)
                predicted_mos_bak_seg_raw.append(mos_bak_raw)
                predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
                predicted_mos_sig_seg.append(mos_sig)
                predicted_mos_bak_seg.append(mos_bak)
                predicted_mos_ovr_seg.append(mos_ovr)

        clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr': fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['P808_MOS'] = float(np.mean(predicted_p808_mos))
        
        # Add other metrics only if computed
        if not only_p808:
            clip_dict['OVRL_raw'] = float(np.mean(predicted_mos_ovr_seg_raw))
            clip_dict['SIG_raw'] = float(np.mean(predicted_mos_sig_seg_raw))
            clip_dict['BAK_raw'] = float(np.mean(predicted_mos_bak_seg_raw))
            clip_dict['OVRL'] = float(np.mean(predicted_mos_ovr_seg))
            clip_dict['SIG'] = float(np.mean(predicted_mos_sig_seg))
            clip_dict['BAK'] = float(np.mean(predicted_mos_bak_seg))
        
        return clip_dict


def evaluate_video_list(
    compute_score: ComputeScore,
    video_paths: List[str],
    sample_rate: int = SAMPLING_RATE,
    is_personalized: bool = False,
    only_p808: bool = True,
) -> Dict[str, float]:
    """Evaluate a list of videos and return DNSMOS scores.
    
    Args:
        compute_score: ComputeScore instance
        video_paths: List of video file paths
        sample_rate: Audio sampling rate
        is_personalized: Whether to use personalized MOS
        only_p808: If True, only compute P808_MOS
        
    Returns:
        Dictionary mapping video paths to P808_MOS scores.
        Failed videos are skipped (not included in results) to avoid affecting mean calculations.
    """
    dnsmos_scores: Dict[str, float] = {}
    
    for video_path in tqdm(video_paths, desc="Evaluating videos (DNSMOS)"):
        if not video_path.endswith(".mp4"):
            continue
        
        temp_wav_path = None
        try:
            # Extract audio from video
            temp_wav_path = extract_audio_from_video_to_file(video_path, target_sr=sample_rate)
            
            # Compute DNSMOS score
            result = compute_score(temp_wav_path, sample_rate, is_personalized, only_p808=only_p808)
            dnsmos_scores[video_path] = result['P808_MOS']
            
        except Exception as e:
            # Skip failed videos instead of returning 0.0
            # This prevents failed videos from affecting the mean calculation
            print(f"[WARN] Failed to process {os.path.basename(video_path)}: {e}. Skipping.")
            # Don't add to dnsmos_scores - just continue to next video
        finally:
            # Clean up temporary file
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
    
    return dnsmos_scores


def save_per_video_scores(
    output_dir: str,
    dnsmos_scores: Dict[str, float],
) -> None:
    """Save individual JSON files for each video with DNSMOS scores."""
    os.makedirs(output_dir, exist_ok=True)
    for abs_video_path, p808_score in dnsmos_scores.items():
        base = os.path.basename(abs_video_path)
        save_name = base[:-4] if base.endswith('.mp4') else base
        out_json = os.path.join(output_dir, f"{save_name}_eval_dnsmos_score.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({"P808_MOS": float(p808_score)}, f, ensure_ascii=False, indent=4)


def evaluate_direct_directory(
    input_dir: str,
    compute_score: ComputeScore,
    is_personalized: bool = False,
) -> None:
    """Evaluate all videos in a flat directory (direct mode)."""
    print("🎬 Direct directory evaluation mode (DNSMOS)")
    print(f"📁 Input directory: {input_dir}")
    
    video_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4')]
    video_paths = sorted(video_paths)
    
    if not video_paths:
        print(f"❌ No .mp4 files found in {input_dir}")
        import sys
        sys.exit(0)

    print(f"📹 Found {len(video_paths)} videos")
    
    # Evaluate videos
    dnsmos_scores = evaluate_video_list(compute_score, video_paths, SAMPLING_RATE, is_personalized)
    
    # Save per-video scores
    save_per_video_scores(input_dir, dnsmos_scores)
    
    # Compute and save means
    p808_mean = float(np.mean(list(dnsmos_scores.values()))) if len(dnsmos_scores) > 0 else 0.0
    nested_means = {
        'P808_MOS': {'direct': {'all': p808_mean}},
    }
    
    mean_json_path = os.path.join(input_dir, '000eval_dnsmos_scores_means.json')
    with open(mean_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_means, f, ensure_ascii=False, indent=4)
    
    print(f"💾 Saved mean scores to: {mean_json_path}")
    print(f"📈 P808_MOS mean: {p808_mean:.4f}")
    print("✅ Direct evaluation completed.")


def merge_dnsmos_scores_files(output_root: str) -> None:
    """Merge multiple eval_dnsmos_scores_means_steps_*.json files."""
    import glob
    
    pattern = os.path.join(output_root, "eval_dnsmos_scores_means_steps_*.json")
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
            step_info = filename.replace("eval_dnsmos_scores_means_steps_", "").replace(".json", "")
            for score_name, step_data in data.items():
                for step, category_data in step_data.items():
                    merged_key = f"{step}"
                    merged_data[score_name][merged_key] = category_data
        except Exception as e:
            print(f"⚠️ Error reading {step_file}: {e}")
    
    merged_file = os.path.join(output_root, "eval_dnsmos_scores_means.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    print(f"✅ Merged {len(step_files)} files into {merged_file}")


def main():
    parser = argparse.ArgumentParser(description="DNSMOS Audio Quality Evaluation with step/category support")
    parser.add_argument('-i', "--exp_root", type=str,
                        default=None,
                        help="Experiment root directory, e.g., /path/to/experiment")
    parser.add_argument("--prompt_meta_json", type=str,
                        default=None,
                        help="Path to input meta JSON template")
    parser.add_argument("--video_save_path_subdir_name", type=str, default="eval_videos",
                        help="Subdirectory under exp_root where generated videos are saved")
    parser.add_argument("--video_eval_audio_output_subdir_name", type=str, default=None,
                        help="Subdirectory under exp_root where DNSMOS evaluation results will be saved (default: same as --video_save_path_subdir_name)")
    parser.add_argument("--input_dir_direct", type=str, default=None,
                        help="If set to a directory path, read videos directly from this directory without step/category structure")
    parser.add_argument("--specific_steps", type=int, nargs="+", default=None,
                        help="Specific step checkpoints to evaluate (e.g., --specific_steps 1000 2000 3000)")
    parser.add_argument("--personalized_MOS", action='store_true',
                        help="Use personalized MOS model")
    parser.add_argument("--use_gpu", action='store_true', default=True,
                        help="Use GPU acceleration for ONNX model inference (requires onnxruntime-gpu and CUDA)")

    args = parser.parse_args()

    # Get script directory for model paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p808_model_path = os.path.join(script_dir, 'DNSMOS', 'model_v8.onnx')

    if args.personalized_MOS:
        primary_model_path = os.path.join(script_dir, 'pDNSMOS', 'sig_bak_ovr.onnx')
    else:
        primary_model_path = os.path.join(script_dir, 'DNSMOS', 'sig_bak_ovr.onnx')

    print(f"🔧 Loading DNSMOS models...")
    print(f"  Primary model: {primary_model_path}")
    print(f"  P808 model: {p808_model_path}")

    # Initialize ComputeScore
    compute_score = ComputeScore(primary_model_path, p808_model_path, use_gpu=args.use_gpu)
    print(f"✅ Models loaded successfully")

    # Direct mode
    if args.input_dir_direct is not None and len(args.input_dir_direct) > 0:
        evaluate_direct_directory(args.input_dir_direct, compute_score, args.personalized_MOS)
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
            dnsmos_scores = evaluate_video_list(compute_score, all_paths, SAMPLING_RATE, args.personalized_MOS)
            
            # Save per-video scores
            save_per_video_scores(category_output_dir, dnsmos_scores)

            # Compute means for this category
            p808_mean = float(np.mean(list(dnsmos_scores.values()))) if len(dnsmos_scores) > 0 else 0.0
            nested_means['P808_MOS'][step_dir][category] = p808_mean
            print(f"📈 Category mean -> P808_MOS: {p808_mean:.4f}")

        # After finishing categories in this step, compute overall means
        print(f"📊 Computing overall means for {step_dir}...")
        step_dict = nested_means['P808_MOS'][step_dir]
        if step_dict:  # at least one category
            all_mean = float(np.mean(list(step_dict.values())))
            step_dict['all'] = all_mean
            print(f"📈 P808_MOS overall mean: {all_mean:.4f}")
        print(f"✅ Step {step_dir} completed")

    # Save final means file
    if (args.video_save_path_subdir_name == args.video_eval_audio_output_subdir_name and
        args.specific_steps is not None and len(args.specific_steps) > 0):
        steps_str = "_".join(map(str, sorted(args.specific_steps)))
        mean_json_filename = f"eval_dnsmos_scores_means_steps_{steps_str}.json"
    else:
        mean_json_filename = "eval_dnsmos_scores_means.json"

    mean_json_path = os.path.join(output_root, mean_json_filename)
    print(f"💾 Saving final results to: {mean_json_path}")
    with open(mean_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_means, f, ensure_ascii=False, indent=4)

    # Merge multiple step files if needed
    if (args.video_save_path_subdir_name == args.video_eval_audio_output_subdir_name and
        args.specific_steps is not None and len(args.specific_steps) > 0):
        print("🔄 Merging step files...")
        merge_dnsmos_scores_files(output_root)

    print("\n🎉 === DNSMOS Evaluation Completed ===")


if __name__ == "__main__":
    main()

