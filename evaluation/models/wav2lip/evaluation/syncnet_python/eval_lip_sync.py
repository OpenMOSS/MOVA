#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import shutil
import subprocess
from types import SimpleNamespace
from collections import defaultdict
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# Ensure local modules are importable when executed from elsewhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SyncNetInstance_calc_scores import SyncNetInstance  # noqa: E402


def run_pipeline_for_video(video_path: str,
                           tmp_root: str,
                           reference: str = "wav2lip",
                           facedet_scale: float = 0.25,
                           min_face_size: int = 100,
                           conf_th: float = 0.9,
                           use_multi_scale: bool = False) -> None:
    """Run face detection/tracking/cropping pipeline to produce per-person crops.

    Outputs under tmp_root: pyavi, pyframes, pywork, pycrop, pytmp.
    """
    os.makedirs(tmp_root, exist_ok=True)
    run_py = os.path.join(SCRIPT_DIR, 'run_pipeline.py')
    cmd = [
        sys.executable, run_py,
        '--videofile', video_path,
        '--reference', reference,
        '--data_dir', tmp_root,
        '--facedet_scale', str(facedet_scale),
        '--min_face_size', str(min_face_size),
        '--conf_th', str(conf_th),
    ]
    if use_multi_scale:
        cmd.append('--use_multi_scale')
    subprocess.run(cmd, check=True)


def compute_lip_scores_from_crops(tmp_root: str,
                                  batch_size: int = 20,
                                  vshift: int = 15,
                                  reference: str = 'wav2lip') -> Tuple[Dict[str, float], float, float]:
    """Compute LSE-D (distance) and LSE-C (confidence) for each cropped person.

    Returns:
        per_person: dict mapping keys like 'person0_LSE-D', 'person0_LSE-C', ...
        mean_lse_d: mean of LSE-D across persons
        mean_lse_c: mean of LSE-C across persons
    """
    crop_dir = os.path.join(tmp_root, 'pycrop', reference)
    tmp_dir = os.path.join(tmp_root, 'pytmp')

    flist = glob.glob(os.path.join(crop_dir, '0*.avi'))
    flist.sort()

    per_person: Dict[str, float] = {}
    lse_ds: List[float] = []
    lse_cs: List[float] = []

    if len(flist) == 0:
        return per_person, 0.0, 0.0

    # Prepare SyncNet
    model_path = os.path.join(SCRIPT_DIR, 'data', 'syncnet_v2.model')
    s = SyncNetInstance()
    s.loadParameters(model_path)

    # Minimal opt namespace for evaluate()
    opt = SimpleNamespace(
        tmp_dir=tmp_dir,
        reference=reference,
        vshift=vshift,
        batch_size=batch_size,
    )
    max_C_minux_D_value = - 100.0
    max_C = 0.0
    min_D = 0.0
    for idx, avi in enumerate(flist):
        offset, conf, dist = s.evaluate(opt, videofile=avi)
        # Map as specified: conf -> LSE-C, dist -> LSE-D
        per_person[f'person{idx}_LSE-D'] = float(dist)
        per_person[f'person{idx}_LSE-C'] = float(conf)
        if float(conf) - float(dist) > max_C_minux_D_value:
            max_C_minux_D_value = float(conf) - float(dist)
            max_C = float(conf)
            min_D = float(dist)
        lse_ds.append(float(dist))
        lse_cs.append(float(conf))

    mean_lse_d = float(np.mean(lse_ds)) if len(lse_ds) > 0 else 0.0
    mean_lse_c = float(np.mean(lse_cs)) if len(lse_cs) > 0 else 0.0
    return per_person, mean_lse_d, mean_lse_c, max_C, min_D


def save_per_video_json(output_dir: str,
                        video_path: str,
                        per_person: Dict[str, float],
                        mean_lse_d: float,
                        mean_lse_c: float,
                        max_C: float,
                        min_D: float) -> None:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(video_path)
    save_name = base[:-4] if base.endswith('.mp4') else base
    out_json = os.path.join(output_dir, f"{save_name}_lip_score.json")
    to_save = dict(per_person)
    to_save['mean_LSE-D'] = float(mean_lse_d)
    to_save['mean_LSE-C'] = float(mean_lse_c)
    to_save['max_C_inperson'] = float(max_C)
    to_save['min_D_inperson'] = float(min_D)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(to_save, f, ensure_ascii=False, indent=4)


def evaluate_one_video(video_path: str,
                       tmp_work_root: str,
                       batch_size: int,
                       facedet_scale: float,
                       min_face_size: int,
                       conf_th: float,
                       use_multi_scale: bool) -> Tuple[Dict[str, float] | None, float | None, float | None]:
    reference = 'wav2lip'
    # Make a dedicated tmp folder per video to avoid conflicts
    vid_stem = os.path.splitext(os.path.basename(video_path))[0]
    tmp_root = os.path.join(tmp_work_root, f"tmp_{vid_stem}")
    if os.path.exists(tmp_root):
        try:
            shutil.rmtree(tmp_root)
        except (OSError, FileNotFoundError) as e:
            # Ignore errors if directory is already being deleted or doesn't exist
            print(f"[DEBUG] Failed to remove {tmp_root}: {e}, continuing...")
    os.makedirs(tmp_root, exist_ok=True)

    try:
        # Run pipeline; if it fails (e.g., detector crash), skip gracefully
        try:
            run_pipeline_for_video(
                video_path, tmp_root, reference=reference,
                facedet_scale=facedet_scale,
                min_face_size=min_face_size,
                conf_th=conf_th,
                use_multi_scale=use_multi_scale,
            )
        except Exception as e:
            print(f"[WARN] Pipeline failed for {video_path}: {e}. Skip AV sync eval.")
            return None, None, None, None, None

        # Check if any frames were extracted; if not, skip
        frames_dir = os.path.join(tmp_root, 'pyframes', reference)
        frames_jpg = glob.glob(os.path.join(frames_dir, '*.jpg'))
        wav2lip_log = os.path.join(tmp_root, 'wav2lip.log')
        if (len(frames_jpg) == 0) or (os.path.isfile(wav2lip_log) and '0 dets' in open(wav2lip_log, 'r', encoding='utf-8', errors='ignore').read()):
            print(f"[WARN] No face frames detected for {video_path}, skip AV sync eval.")
            return None, None, None, None, None

        # Check if any crops were produced (face tracks)
        crop_dir = os.path.join(tmp_root, 'pycrop', reference)
        flist = glob.glob(os.path.join(crop_dir, '0*.avi'))
        if len(flist) == 0:
            print(f"[WARN] No face detected in {video_path}, skip AV sync eval.")
            return None, None, None, None, None

        per_person, mean_lse_d, mean_lse_c, max_C, min_D = compute_lip_scores_from_crops(
            tmp_root, batch_size=batch_size, vshift=15, reference=reference
        )
    finally:
        # Clean temporary artifacts for this video
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)
    return per_person, mean_lse_d, mean_lse_c, max_C, min_D


def evaluate_direct_directory(input_dir: str,
                              batch_size: int = 20,
                              facedet_scale: float = 0.25,
                              min_face_size: int = 100,
                              conf_th: float = 0.9,
                              use_multi_scale: bool = False,
                              max_workers: int = 1) -> None:
    print("🎬 Direct directory evaluation mode (lip)")
    print(f"📁 Input directory: {input_dir}")
    print(f"🔧 Max workers for parallel processing: {max_workers}")
    video_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4')]
    video_paths = sorted(video_paths)
    if not video_paths:
        print(f"❌ No .mp4 files found in {input_dir}")
        sys.exit(0)

    # Use a working tmp dir under input directory to avoid conflicts across runs
    tmp_work_root = os.path.join(input_dir, 'tmp_dir_eval')
    if os.path.exists(tmp_work_root):
        shutil.rmtree(tmp_work_root, ignore_errors=True)
    os.makedirs(tmp_work_root, exist_ok=True)

    per_video_means_d: List[float] = []
    per_video_means_c: List[float] = []
    per_video_max_C: List[float] = []
    per_video_min_D: List[float] = []
    
    if max_workers > 1:
        # Parallel processing with multiple workers
        print(f"🚀 Using {max_workers} workers for parallel processing")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_one_video,
                    video_path, tmp_work_root, batch_size,
                    facedet_scale, min_face_size, conf_th, use_multi_scale
                ): video_path for video_path in video_paths
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating videos (lip)"):
                video_path = futures[future]
                try:
                    per_person, mean_lse_d, mean_lse_c, max_C, min_D = future.result()
                    if per_person is None:
                        continue  # skip saving and aggregation
                    save_per_video_json(input_dir, video_path, per_person, mean_lse_d, mean_lse_c, max_C, min_D)
                    per_video_means_d.append(mean_lse_d)
                    per_video_means_c.append(mean_lse_c)
                    per_video_max_C.append(max_C)
                    per_video_min_D.append(min_D)
                except Exception as e:
                    print(f"[ERROR] Failed to process {video_path}: {e}")
    else:
        # Sequential processing (original behavior)
        print("🔄 Using sequential processing (single worker)")
        for video_path in tqdm(video_paths, desc="Evaluating videos (lip)"):
            per_person, mean_lse_d, mean_lse_c, max_C, min_D = evaluate_one_video(
                video_path, tmp_work_root, batch_size,
                facedet_scale, min_face_size, conf_th, use_multi_scale
            )
            if per_person is None:
                continue  # skip saving and aggregation
            save_per_video_json(input_dir, video_path, per_person, mean_lse_d, mean_lse_c, max_C, min_D)
            per_video_means_d.append(mean_lse_d)
            per_video_means_c.append(mean_lse_c)
            per_video_max_C.append(max_C)
            per_video_min_D.append(min_D)
    # Save flat means without 'direct'/'all'
    nested_means = {
        'mean_LSE-D': float(np.mean(per_video_means_d)) if per_video_means_d else 0.0,
        'mean_LSE-C': float(np.mean(per_video_means_c)) if per_video_means_c else 0.0,
        'max_C_inperson': float(np.mean(per_video_max_C)) if per_video_max_C else 0.0,
        'min_D_inperson': float(np.mean(per_video_min_D)) if per_video_min_D else 0.0,
    }
    mean_json_path = os.path.join(input_dir, '000eval_lip_scores_means.json')
    with open(mean_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_means, f, ensure_ascii=False, indent=4)
    print(f"💾 Saved mean scores to: {mean_json_path}")
    print("✅ Direct evaluation completed.")


def build_categories_from_meta(meta_dict: Dict) -> List[str]:
    categories: List[str] = []
    for category, items in meta_dict.items():
        if isinstance(items, dict):
            prompts = [(k, v['prompt']) for k, v in items.items() if isinstance(v, dict) and 'prompt' in v]
            if prompts:
                categories.append(category)
    return categories


def merge_lip_scores_files(output_root: str) -> None:
    import glob
    pattern = os.path.join(output_root, "eval_lip_scores_means_steps_*.json")
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
            for score_name, step_data in data.items():
                for step, category_data in step_data.items():
                    merged_data[score_name][f"{step}"] = category_data
        except Exception as e:
            print(f"⚠️ Error reading {step_file}: {e}")
    merged_file = os.path.join(output_root, "eval_lip_scores_means.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    print(f"✅ Merged {len(step_files)} files into {merged_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lip sync (LSE-D/LSE-C) evaluation without multinode")
    parser.add_argument('-i', '--exp_root', type=str,
                        default=None,
                        help='Experiment root directory')
    parser.add_argument('--prompt_meta_json', type=str,
                        default=None,
                        help='Path to input meta JSON template')

    parser.add_argument('--video_save_path_subdir_name', type=str, default='eval_videos',
                        help='Subdir under exp_root where generated videos are saved')
    parser.add_argument('--video_eval_video_quality_output_subdir_name', type=str, default=None,
                        help='Output subdir under exp_root where lip evaluation results are saved (default: same as --video_save_path_subdir_name)')

    parser.add_argument('--input_dir_direct', type=str, default=None,
                        help='If set, evaluate videos directly from this directory')
    parser.add_argument('--specific_steps', type=int, nargs='+', default=None,
                        help='Specific step checkpoints to evaluate (e.g., --specific_steps 1000 2000)')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for SyncNet evaluation')
    # Face detection controls
    parser.add_argument('--facedet_scale', type=float, default=0.25, help='Scale factor for face detection')
    parser.add_argument('--min_face_size', type=int, default=100, help='Minimum face size in pixels')
    parser.add_argument('--conf_th', type=float, default=0.9, help='Face detection confidence threshold')
    parser.add_argument('--use_multi_scale', action='store_true', help='Enable multi-scale face detection (adds fixed scales)')
    
    # Parallel processing control
    parser.add_argument('--max_workers', type=int, default=8, 
                        help='Number of parallel workers for video processing. Recommended: 2-4 for single GPU. Default: 1 (sequential)')



# Lower detection confidence threshold: change conf_th=0.9 to 0.6-0.7.
# Increase --facedet_scale: default 0.25 shrinks the image, easily missing small faces; change to 0.5 or 1.0.
# Lower --min_face_size: e.g., from 100 to 40-60.

    args = parser.parse_args()
    
    # Validate max_workers
    if args.max_workers < 1:
        print("⚠️ Warning: max_workers must be >= 1, setting to 1")
        args.max_workers = 1
    elif args.max_workers > 15:
        print(f"⚠️ Warning: max_workers={args.max_workers} is very high, may cause GPU OOM. Consider using 2-4.")
    
    print(f"🔧 Configuration:")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Face detection scale: {args.facedet_scale}")
    print(f"  Min face size: {args.min_face_size}")
    print(f"  Confidence threshold: {args.conf_th}")

    # Direct mode
    if args.input_dir_direct is not None and len(args.input_dir_direct) > 0:
        evaluate_direct_directory(args.input_dir_direct, batch_size=args.batch_size,
                                  facedet_scale=args.facedet_scale,
                                  min_face_size=args.min_face_size,
                                  conf_th=args.conf_th,
                                  use_multi_scale=args.use_multi_scale,
                                  max_workers=args.max_workers)
        sys.exit(0)

    # Step/category mode
    if args.video_eval_video_quality_output_subdir_name is None:
        args.video_eval_video_quality_output_subdir_name = args.video_save_path_subdir_name

    input_root = os.path.join(args.exp_root, args.video_save_path_subdir_name)
    output_root = os.path.join(args.exp_root, args.video_eval_video_quality_output_subdir_name)
    os.makedirs(output_root, exist_ok=True)

    all_step_dirs = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])

    if args.specific_steps is not None:
        step_dirs = []
        for step in args.specific_steps:
            sdir = f"step_{step}"
            if sdir in all_step_dirs:
                step_dirs.append(sdir)
            else:
                print(f"⚠️ Warning: Step directory {sdir} not found in {input_root}")
        if not step_dirs:
            print("❌ None of the requested steps are available")
            sys.exit(0)
        print(f"📌 Using specific steps: {[int(d.split('_')[1]) for d in step_dirs]}")
    else:
        step_dirs = all_step_dirs

    with open(args.prompt_meta_json, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)
    categories = build_categories_from_meta(meta_dict)

    print(f"🚀 Using {args.max_workers} workers for parallel processing")
    nested_means = defaultdict(lambda: defaultdict(dict))  # {metric: {step: {category: mean, 'all': mean}}}

    for step_dir in tqdm(step_dirs, desc="Processing steps"):
        print(f"\n🔄 Processing step: {step_dir}")
        input_step_dir = os.path.join(input_root, step_dir)
        output_step_dir = os.path.join(output_root, step_dir)
        os.makedirs(output_step_dir, exist_ok=True)

        step_means_d: List[float] = []
        step_means_c: List[float] = []

        for category in categories:

            tmp_work_root_step_category = os.path.join(input_step_dir, f'tmp_dir_eval_step_{step_dir.split("_")[1]}_{category}')
            if os.path.exists(tmp_work_root_step_category):
                shutil.rmtree(tmp_work_root_step_category, ignore_errors=True)
            os.makedirs(tmp_work_root_step_category, exist_ok=True)

            print(f"📁 Processing category: {category}")
            category_input_dir = os.path.join(input_step_dir, category)
            category_output_dir = os.path.join(output_step_dir, category)
            # os.makedirs(category_output_dir, exist_ok=True)

            # Gather videos
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

            cat_video_means_d: List[float] = []
            cat_video_means_c: List[float] = []
            cat_video_max_C: List[float] = []
            cat_video_min_D: List[float] = []
            
            if args.max_workers > 1:
                # Parallel processing with multiple workers
                with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                    futures = {
                        executor.submit(
                            evaluate_one_video,
                            video_path, tmp_work_root_step_category, args.batch_size,
                            args.facedet_scale, args.min_face_size, args.conf_th, args.use_multi_scale
                        ): video_path for video_path in all_paths
                    }
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating videos ({step_dir}/{category})"):
                        video_path = futures[future]
                        try:
                            res = future.result()
                            per_person, mean_lse_d, mean_lse_c, max_C, min_D = res
                            if per_person is None:
                                continue  # skip
                            save_per_video_json(category_output_dir, video_path, per_person, mean_lse_d, mean_lse_c, max_C, min_D)
                            cat_video_max_C.append(max_C)
                            cat_video_min_D.append(min_D)
                        except Exception as e:
                            print(f"[ERROR] Failed to process {video_path}: {e}")
            else:
                # Sequential processing (original behavior)
                for video_path in tqdm(all_paths, desc=f"Evaluating videos ({step_dir}/{category})"):
                    res = evaluate_one_video(
                        video_path, tmp_work_root_step_category, args.batch_size,
                        args.facedet_scale, args.min_face_size, args.conf_th, args.use_multi_scale
                    )
                    per_person, mean_lse_d, mean_lse_c, max_C, min_D = res
                    if per_person is None:
                        continue  # skip
                    save_per_video_json(category_output_dir, video_path, per_person, mean_lse_d, mean_lse_c, max_C, min_D)
                    # cat_video_means_d.append(mean_lse_d)
                    # cat_video_means_c.append(mean_lse_c)
                    cat_video_max_C.append(max_C)
                    cat_video_min_D.append(min_D)
            # Category means for this step
            # cat_mean_d = float(np.mean(cat_video_means_d)) if cat_video_means_d else 0.0
            # cat_mean_c = float(np.mean(cat_video_means_c)) if cat_video_means_c else 0.0
            cat_mean_max_C = float(np.mean(cat_video_max_C)) if cat_video_max_C else 0.0
            cat_mean_min_D = float(np.mean(cat_video_min_D)) if cat_video_min_D else 0.0
            # nested_means['LSE-D'][step_dir][category] = cat_mean_d
            # nested_means['LSE-C'][step_dir][category] = cat_mean_c
            nested_means['LSE-C'][step_dir][category] = cat_mean_max_C
            nested_means['LSE-D'][step_dir][category] = cat_mean_min_D
            step_means_d.extend(cat_video_min_D)
            step_means_c.extend(cat_video_max_C)

        # overall means for this step across categories
        if step_means_d:
            nested_means['LSE-D'][step_dir]['all'] = float(np.mean(step_means_d))
        if step_means_c:
            nested_means['LSE-C'][step_dir]['all'] = float(np.mean(step_means_c))
        print(f"✅ Step {step_dir} completed")

    # Save final means file
    if (args.video_save_path_subdir_name == args.video_eval_video_quality_output_subdir_name and
        args.specific_steps is not None and len(args.specific_steps) > 0):
        steps_str = "_".join(map(str, sorted(args.specific_steps)))
        mean_json_filename = f"eval_lip_scores_means_steps_{steps_str}.json"
    else:
        mean_json_filename = "eval_lip_scores_means.json"

    mean_json_path = os.path.join(output_root, mean_json_filename)
    print(f"💾 Saving final results to: {mean_json_path}")
    with open(mean_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_means, f, ensure_ascii=False, indent=4)

    # Merge multiple step files, if applicable
    if (args.video_save_path_subdir_name == args.video_eval_video_quality_output_subdir_name and
        args.specific_steps is not None and len(args.specific_steps) > 0):
        print("🔄 Merging step files...")
        merge_lip_scores_files(output_root)

    print("\n🎉 === Lip Score Evaluation Completed ===")


if __name__ == '__main__':
    main()


