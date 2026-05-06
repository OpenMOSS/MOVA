import logging
from pathlib import Path
from typing import Dict, Optional
import time
from typing import Union
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from einops import rearrange
import argparse
import yaml
import re

import torch
# import torchaudio
from torch.utils.data import DataLoader
torch.backends.cuda.enable_flash_sdp(True)  # PyTorch 2.0+
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True

from imagebind.models.imagebind_model import ModalityType
from av_bench.data.audio_dataset import (AudioDataset, ImageBindAudioDataset,
                                         SynchformerAudioDataset, pad_or_truncate)
from av_bench.data.media_dataset import MediaDataset, error_avoidance_collate
from av_bench.extraction_models import ExtractionModels
from av_bench.metrics import compute_isc#, compute_fd, compute_kl
from av_bench.synchformer.synchformer import make_class_grid
# from av_bench.utils import (unroll_dict, unroll_dict_all_keys, unroll_paired_dict,
#                             unroll_paired_dict_with_key)
from av_bench.extract import encode_audio_with_sync
from extract_video import encode_video_with_sync, encode_video_with_imagebind

from av_align_score import compute_av_align_for_videos_parallel


mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)

import os
import json


# os.environ["HF_HUB_OFFLINE"] = "1"



 


def merge_av_scores_files(output_root):
    """Merge multiple eval_av_scores_means_steps_*.json files into eval_av_scores_means_steps_all_step.json"""
    import glob
    from collections import defaultdict
    
    # Find all matching step files
    pattern = os.path.join(output_root, "eval_av_scores_means_steps_*.json")
    step_files = glob.glob(pattern)
    
    # Filter out the all_step file itself
    step_files = [f for f in step_files if not f.endswith("_all_step.json")]
    
    if len(step_files) <= 0:
        print(f"📄 Found {len(step_files)} step files, no need to merge")
        return
    
    print(f"🔄 Found {len(step_files)} step files, merging...")
    
    # Merge all data
    merged_data = defaultdict(lambda: defaultdict(dict))
    
    for step_file in step_files:
        try:
            with open(step_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract step info from filename
            filename = os.path.basename(step_file)
            # From "eval_av_scores_means_steps_1000_2000.json" extract "1000_2000"
            step_info = filename.replace("eval_av_scores_means_steps_", "").replace(".json", "")
            
            # Merge data
            for score_name, step_data in data.items():
                for step, category_data in step_data.items():
                    merged_key = f"{step}"  # Create unique key
                    # merged_key = f"{step}_{step_info}"  # Create unique key
                    merged_data[score_name][merged_key] = category_data
                    
        except Exception as e:
            print(f"⚠️ Error reading {step_file}: {e}")
            continue
    
    # Save merged file
    merged_file = os.path.join(output_root, "eval_av_scores_means.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Merged {len(step_files)} files into {merged_file}")
    
    # Optional: delete original step files
    # for step_file in step_files:
    #     os.remove(step_file)
    #     print(f"🗑️ Removed {step_file}")


# ============ Duration helpers ==========
def infer_duration_from_subdir_name(subdir_name: str) -> Optional[float]:
    """Infer clip duration from a subdir name pattern like '5s' or '8s'.

    Rule: find the first number before 's' (case-insensitive), subtract 0.2 seconds,
    and round to one decimal place. Examples: '5s' -> 4.8, '8s' -> 7.8.
    """
    if not subdir_name:
        print(f"🔍 Duration inference: subdir_name is empty")
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*s", subdir_name, flags=re.IGNORECASE)
    if not match:
        print(f"🔍 Duration inference: no 'Xs' pattern found in '{subdir_name}'")
        return None
    try:
        nominal = float(match.group(1))
        adjusted = round(nominal - 0.2, 1)
        result = max(adjusted, 0.0)
        print(f"🔍 Duration inference: '{subdir_name}' -> {nominal}s -> {result}s")
        return result
    except ValueError:
        print(f"🔍 Duration inference: failed to parse number from '{subdir_name}'")
        return None

# ============ Helpers (structure/clarity) ============
def setup_models_and_opts(dover_opt_path: str, device: str):
    print(f"🔧 Setting up models from {dover_opt_path}")
    with open(dover_opt_path, "r") as f:
        dover_opt = yaml.safe_load(f)
    print(f"🔧 Loading ExtractionModels to {device}")
    models = ExtractionModels().to(device).eval()
    dopt = dover_opt["data"]["val-l1080p"]["args"]
    print(f"🔧 Models loaded successfully")
    return models, dopt


def evaluate_direct_directory(input_dir: str, direct_eval_save_path: str, device: str,
                              models, dopt,
                              batch_size: int = 16,
                              duration: float = 8.0) -> None:
    """Evaluate a flat directory of .mp4 files and save results under direct_eval_save_path."""
    print(f"🎬 Direct directory evaluation mode")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {direct_eval_save_path}")
    print(f"⏱️ Duration: {duration}s")
    print(f"🔧 Batch size: {batch_size}")
    
    # Collect videos
    all_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4')]
    all_paths = sorted(all_paths)
    if not all_paths:
        print(f"❌ No .mp4 files found in {input_dir}")
        import sys
        sys.exit(0)

    os.makedirs(direct_eval_save_path, exist_ok=True)
    print(f"📄 Direct input_dir mode: {len(all_paths)} videos under {input_dir}")
    print(f"📋 Video files: {[os.path.basename(p) for p in all_paths[:5]]}{'...' if len(all_paths) > 5 else ''}")


    # Run batch evaluation and save CSV
    print(f"🚀 Starting audio-video evaluation...")
    output_path = os.path.join(direct_eval_save_path, 'quality_assess_direct.csv')
    print(f"📊 Output CSV: {output_path}")
    scores = evaluate_audio_video(
        models, dopt,
        all_paths,
        output_path=output_path,
        duration=duration,
        device=device,
        media_type='video',
        batch_size=batch_size,
        num_workers=8,
        synchformer_resample=True
    )
    print(f"✅ Audio-video evaluation completed")

    # Compute AV-Align scores
    print(f"🔄 Computing AV-Align scores...")
    av_align_scores = compute_av_align_for_videos_parallel(all_paths)
    print(f"✅ AV-Align scores computed for {len(av_align_scores)} videos")
    print(f"🔍 AV-Align keys: {list(av_align_scores.keys())[:5]}...")
    print(f"🔍 AV-Align sample values: {[(k, v) for k, v in list(av_align_scores.items())[:3]]}")

    score_list = ['DeSync', 'IB-Score', 'AV-Align']
    print(f"📊 Computing metrics: {score_list}")

    # Save per-video JSON
    print(f"💾 Saving individual video scores...")
    for i, name in enumerate(scores['names']):
        vid_name = os.path.basename(str(name))
        # Keys in av_align_scores are full basenames (with .mp4), need to ensure matching
        # If vid_name does not have extension, try adding .mp4
        if not vid_name.endswith('.mp4'):
            vid_name_with_ext = vid_name + '.mp4'
        else:
            vid_name_with_ext = vid_name
        
        vid_score_dict = {}
        for score_name in score_list:
            if score_name == 'AV-Align':
                # Try both matching: with and without extension
                av_score = av_align_scores.get(vid_name_with_ext, av_align_scores.get(vid_name, 0.0))
                vid_score_dict[score_name] = av_score
                if i < 3:  # Debug first 3
                    print(f"🔍 Matching '{vid_name}' (tried '{vid_name_with_ext}') -> AV-Align score: {av_score}")
            else:
                vid_score_dict[score_name] = scores[score_name][i]
        
        # Save using the original vid_name (may not have .mp4)
        save_name = vid_name if not vid_name.endswith('.mp4') else vid_name[:-4]
        vid_score_path = os.path.join(direct_eval_save_path, f"{save_name}_av_score.json")
        with open(vid_score_path, 'w', encoding='utf-8') as f:
            json.dump(vid_score_dict, f, ensure_ascii=False, indent=4)
        if i < 3:  # Show first 3 examples
            print(f"📄 {save_name}: {vid_score_dict}")
    print(f"✅ Saved {len(scores['names'])} individual video score files")

    # Compute summary means and save overall JSON
    print(f"📊 Computing mean scores...")
    from collections import defaultdict
    nested_means = defaultdict(lambda: defaultdict(dict))
    for score_name in score_list:
        if score_name == 'AV-Align':
            # av_align_scores contains 'AV-Align' key storing pre-computed average, can be used directly
            # Or filter out that key and recompute
            individual_scores = [v for k, v in av_align_scores.items() if k != 'AV-Align']
            mean_score = float(np.mean(individual_scores)) if individual_scores else 0.0
            print(f"🔍 AV-Align: {len(individual_scores)} videos, mean: {mean_score:.4f}")
            print(f"🔍 Pre-computed average: {av_align_scores.get('AV-Align', 'NOT_FOUND')}")
        else:
            mean_score = float(np.mean(scores[score_name]))
        nested_means[score_name]['direct']['all'] = mean_score
        print(f"📈 {score_name} mean: {mean_score:.4f}")

    mean_json_path = os.path.join(direct_eval_save_path, '000eval_av_scores_means.json')
    with open(mean_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_means, f, ensure_ascii=False, indent=4)
    print(f"💾 Saved mean scores to: {mean_json_path}")

    print("✅ Direct evaluation completed.")
    print("📊 Final Results:")
    for score_name, data in nested_means.items():
        print(f"  {score_name}: {data['direct']['all']:.4f}")
    import sys
    sys.exit(0)



log = logging.getLogger()
device = 'cuda'


@torch.inference_mode()
@torch.no_grad()
def evaluate_audio_video(
    models,
    dopt,
    data_paths: list[Union[Path, str]],
    output_path: str = None,
    duration: float = 8.0,
    device: str = device,
    media_type: str = 'video',
    batch_size: int = 4,
    num_workers: int = 32,
    synchformer_resample: bool = False,
) -> Dict[str, float]:

    # --- 1. loading data ---
    print(f"📂 Loading {len(data_paths)} videos with duration {duration}s...")
    start = time.time()
    data_paths = [Path(path) for path in data_paths]
    dataset = MediaDataset(dopt, data_paths, duration, media_type=media_type, synchformer_resample=synchformer_resample)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, prefetch_factor = 4,
                            collate_fn=error_avoidance_collate)
    print(f"⏱️ Data loading took {time.time()-start:.2f}s")

    # --- 2. initializing models ---
    # log.info("Initializing models...")
    # start = time.time()

    # models = ExtractionModels().to(device).eval()

    # cmp_encode_video_with_sync = torch.compile(encode_video_with_sync)
    # cmp_encode_video_with_imagebind = torch.compile(encode_video_with_imagebind)
    # cmp_encode_audio_with_sync = torch.compile(encode_audio_with_sync)

    # log.info(f'Take time: {time.time()-start}')

    # --- 3. computing metrics ---
    print(f"🚀 Computing metrics for {len(data_paths)} videos...")
    print(f"📊 Metrics: DeSync (Synchformer), IB-Score (ImageBind)")

    output_metrics = {
        'names': [],
        'silence_ratio': [],
        'snr': [],
        'bandwidth': [],
        'IB-Score': [],
        'DeSync': [],
        # 'AV-Align': [],
    }

    for batch in tqdm(data_loader):
        if isinstance(batch, list) and len(batch) == 0:
            print("Empty batch, skip!!!")
            continue
        filename = batch['filename']
        ib_audio = batch['ib_audio']
        sync_audio = batch['sync_audio']
        ib_video = batch['ib_video']
        sync_video = batch['sync_video']
        silence_ratio = batch['silence_ratio']
        snr = batch['snr']
        bandwidth = batch['bandwidth']
        # av_align = batch['av_align']

        output_metrics['names'].extend(filename)
        output_metrics['silence_ratio'].extend(silence_ratio.tolist())
        output_metrics['snr'].extend(snr.tolist())
        output_metrics['bandwidth'].extend(bandwidth.tolist())
        # output_metrics['AV-Align'].extend(av_align.tolist())

        # Imagebind scores 0.3s
        start = time.time()
        print(f"🔍 Computing ImageBind scores...")

        ib_video = ib_video.to(device)
        ib_audio = ib_audio.squeeze(1).to(device)

        with torch.amp.autocast("cuda"):
            ib_video_features = encode_video_with_imagebind(models.imagebind, ib_video).cpu()
            ib_audio_features = models.imagebind({ModalityType.AUDIO: ib_audio})[ModalityType.AUDIO].cpu().detach()

        ib_score = torch.cosine_similarity(ib_video_features,
                                           ib_audio_features,
                                           dim=-1)

        print(f"📊 IB-Score shape: {ib_score.shape}, range: [{ib_score.min().item():.3f}, {ib_score.max().item():.3f}]")
        output_metrics['IB-Score'].extend([s.item() for s in ib_score])

        print(f"⏱️ ImageBind scores took {time.time()-start:.2f}s")

        # Synchformer scores 1s, compile later
        start = time.time()
        print(f"🔍 Computing Synchformer (DeSync) scores...")

        sync_audio = sync_audio.to(device)
        sync_video = sync_video.to(device)
        

        with torch.amp.autocast("cuda"):
            sync_audio_features = encode_audio_with_sync(models.synchformer, sync_audio, models.sync_mel_spectrogram).cpu()
            sync_video_features = encode_video_with_sync(models.synchformer, sync_video).cpu().detach()

        total_samples = sync_video_features.shape[0]
        batch_size = min(total_samples, 32)
        total_sync_scores1 = []
        total_sync_scores2 = []
        sync_grid = make_class_grid(-2, 2, 21)

        print(f"📊 Processing {total_samples} samples in batches of {batch_size}")

        with torch.amp.autocast("cuda"):
            for i in range(0, total_samples, batch_size):
                sync_video_batch = sync_video_features[i:i + batch_size].to(device)
                sync_audio_batch = sync_audio_features[i:i + batch_size].to(device)

                print(f"🔍 Batch {i//batch_size + 1}: video {sync_video_batch.shape}, audio {sync_audio_batch.shape}")

                logits = models.synchformer.compare_v_a(sync_video_batch[:, :14], sync_audio_batch[:, :14])
                top_id = torch.argmax(logits, dim=-1).cpu().numpy()
                for j in range(sync_video_batch.shape[0]):
                    total_sync_scores1.append(abs(sync_grid[top_id[j]].item()))

                logits = models.synchformer.compare_v_a(sync_video_batch[:, -14:], sync_audio_batch[:, -14:])
                top_id = torch.argmax(logits, dim=-1).cpu().numpy()
                for j in range(sync_video_batch.shape[0]):
                    total_sync_scores2.append(abs(sync_grid[top_id[j]].item()))

        total_sync_scores = ((np.array(total_sync_scores1) + np.array(total_sync_scores2))/2).tolist()

        print(f"📊 DeSync scores: {len(total_sync_scores)} samples, range: [{min(total_sync_scores):.3f}, {max(total_sync_scores):.3f}]")
        output_metrics['DeSync'].extend(total_sync_scores)

        log.info(f'Synchformer scores take time: {time.time()-start}')

    # --- 4. save results ---
    log.info("Saving results...")
    start = time.time()
    output_metrics = pd.DataFrame(output_metrics)
    output_metrics.to_csv(output_path, index=False)
    log.info(f'Save results take time: {time.time()-start}')

    # # LAION-CLAP
    # gt_audio_48k = _load_and_preprocess_audio_tensor(gt_audio_tensor, gt_audio_sr, 48000, audio_length, device)
    # features['clap_laion_audio'] = models.laion_clap.get_audio_embedding_from_data(gt_audio_48k.unsqueeze(0), use_tensor=True)
    
    # # MS-CLAP
    # features['clap_ms_audio'] = models.ms_clap.get_audio_embeddings([str(gt_audio_path)])

    # # DO NOT extract text features for now
    # features['clap_laion_text'] = models.laion_clap.get_text_embedding([text_prompt], use_tensor=True)
    # features['clap_ms_text'] = models.ms_clap.get_text_embeddings([text_prompt])

    # # CLAP Scores
    # output_metrics['LAION-CLAP-Score'] = torch.cosine_similarity(
    #     features['clap_laion_text'], features['clap_laion_audio']
    # ).item()
    # output_metrics['MS-CLAP-Score'] = torch.cosine_similarity(
    #     features['clap_ms_text'], features['clap_ms_audio']
    # ).item()

    return output_metrics




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--exp_root", type=str, default=None, help="Experiment root directory, e.g., /path/to/experiment")
    parser.add_argument('-d', '--dover_opt', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'dover', 'dover.yml'))
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--prompt_meta_json", type=str, default=None, help="Path to input meta JSON template")

    parser.add_argument(
        "--video_save_path_subdir_name", type=str, default="eval_videos",
        help="Subdirectory name under exp_root where generated videos are saved"
    )
    parser.add_argument(
        "--video_eval_av_output_subdir_name", type=str, default=None,
        help="Subdirectory name under exp_root where AV evaluation results will be saved (default: same as --video_save_path_subdir_name)"
    )
    parser.add_argument("--input_dir_direct", type=str, default=None, help="If set to a directory path, read videos directly from this directory without step/category structure")
    parser.add_argument("--specific_steps", type=int, nargs="+", default=None, help="Specific step checkpoints to evaluate (e.g., --specific_steps 1000 2000 3000)")
    parser.add_argument("--duration", type=float, default=None, help="Clip duration in seconds. If not set, inferred from subdir name like '8s' -> 7.8")

    args = parser.parse_args()



    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    log = logging.getLogger("quality_assess")


    device = args.device
    print(f"🚀 Starting evaluation with device: {device}")
    
    # Derive effective duration: CLI > infer from save subdir > default 7.9 (old behavior)
    print(f"⏱️ Duration inference:")
    print(f"  CLI --duration: {args.duration}")
    print(f"  Subdir name: {args.video_save_path_subdir_name}")
    
    effective_duration = args.duration if args.duration is not None else infer_duration_from_subdir_name(args.video_save_path_subdir_name)
    if effective_duration is None:
        effective_duration = 7.9
        print(f"  Using default: {effective_duration}s")
    
    print(f"✅ Final duration: {effective_duration}s")
    models, dopt = setup_models_and_opts(args.dover_opt, device)


    score_list = ['DeSync', 'IB-Score', 'AV-Align']  # List of scores to save
    # step_means = {}

    # Direct directory mode: read video files from the specified directory
    if args.input_dir_direct is not None and len(args.input_dir_direct) > 0:
        print(f"🔍 Using direct directory mode with input directory: {args.input_dir_direct}")
        print(f"📊 Computing metrics: {score_list}")
        evaluate_direct_directory(
            input_dir=args.input_dir_direct,
            direct_eval_save_path=args.input_dir_direct,
            device=device,
            models=models,
            dopt=dopt,
            batch_size=16,
            duration=effective_duration,
        )




    if args.video_eval_av_output_subdir_name is None:
        args.video_eval_av_output_subdir_name = args.video_save_path_subdir_name

    input_root = os.path.join(args.exp_root, args.video_save_path_subdir_name)
    output_root = os.path.join(args.exp_root, args.video_eval_av_output_subdir_name)

    os.makedirs(output_root, exist_ok=True)


    all_step_dirs = sorted([d for d in os.listdir(input_root)
                           if os.path.isdir(os.path.join(input_root, d))])
    
    # Filter step directories based on specific_steps if provided
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
            # return
            import sys
            sys.exit(0) 
        print(f"📌 Using specific steps: {[int(d.split('_')[1]) for d in step_dirs]}")
    else:
        step_dirs = all_step_dirs


    # ====== Load meta JSON ======
    with open(args.prompt_meta_json, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)

    prompt_categories = {}
    for category, items in meta_dict.items():
        if isinstance(items, dict):  # Ensure it is a dict
            # Only collect entries with 'prompt'
            prompts = [(k, v['prompt']) for k, v in items.items() if isinstance(v, dict) and 'prompt' in v]
            if prompts:
                prompt_categories[category] = prompts



    from collections import defaultdict

    # Final master dictionary {score_name: {step: {category: mean, "all": mean}}}
    nested_means = defaultdict(lambda: defaultdict(dict))

    for step_dir in tqdm(step_dirs, desc="Processing steps"):
        print(f"\n🔄 Processing step: {step_dir}")
        input_step_dir = os.path.join(input_root, step_dir)
        output_step_dir = os.path.join(output_root, step_dir)
        os.makedirs(output_step_dir, exist_ok=True)

        for category, meta_list in prompt_categories.items():
            print(f"📁 Processing category: {category}")
            category_input_dir = os.path.join(input_step_dir, category)
            category_output_dir = os.path.join(output_step_dir, category)
            os.makedirs(category_output_dir, exist_ok=True)

            # Collect video paths
            all_paths = []
            if os.path.exists(category_input_dir):
                for root, _, files in os.walk(category_input_dir):
                    for f in files:
                        if f.endswith(".mp4"):
                            all_paths.append(os.path.join(root, f))
            all_paths = sorted(all_paths)

            if not all_paths:
                print(f"⚠️ No video files found in {category_input_dir}")
                continue

            print(f"📹 Found {len(all_paths)} videos in {category}")
            print(f"📋 Sample videos: {[os.path.basename(p) for p in all_paths[:3]]}{'...' if len(all_paths) > 3 else ''}")

            # Compute AV-Align scores
            print(f"🔄 Computing AV-Align scores for {len(all_paths)} videos...")
            av_align_scores = compute_av_align_for_videos_parallel(all_paths)
            print(f"✅ AV-Align scores computed for {len(av_align_scores)} videos")
            print(f"🔍 AV-Align keys: {list(av_align_scores.keys())[:5]}...")
            print(f"🔍 AV-Align sample values: {[(k, v) for k, v in list(av_align_scores.items())[:3]]}")

            output_path = os.path.join(category_output_dir,
                                    f'quality_assess_{step_dir}_{category}.csv')
            print(f"📊 Output CSV: {output_path}")

            print(f"🚀 Starting audio-video evaluation for {category}...")
            start = time.time()
            scores = evaluate_audio_video(
                models, dopt,
                all_paths,
                output_path=output_path,
                duration=effective_duration,
                device=device,
                media_type='video',
                batch_size=16,
                num_workers=8,
                synchformer_resample=True
            )
            print(f"⏱️ Evaluation took {time.time()-start:.2f}s")
            print(f"📊 Scores computed: {list(scores.keys())}")

            # Create a dict containing all scores for each video
            print(f"💾 Saving individual video scores...")
            for i, name in enumerate(scores['names']):
                vid_name = str(name).split('/')[-1]  # Take filename
                # Keys in av_align_scores are full basenames (with .mp4), need to ensure matching
                # If vid_name does not have extension, try adding .mp4
                if not vid_name.endswith('.mp4'):
                    vid_name_with_ext = vid_name + '.mp4'
                else:
                    vid_name_with_ext = vid_name
                
                # Create a dict containing all scores for each video
                vid_score_dict = {}
                for score_name in score_list:
                    if score_name == 'AV-Align':
                        # Try both matching: with and without extension
                        av_score = av_align_scores.get(vid_name_with_ext, av_align_scores.get(vid_name, 0.0))
                        vid_score_dict[score_name] = av_score
                        if i < 3:  # Debug first 3
                            print(f"🔍 Matching '{vid_name}' (tried '{vid_name_with_ext}') -> AV-Align score: {av_score}")
                    else:
                        vid_score_dict[score_name] = scores[score_name][i]
                
                # Save individual video score file
                save_name = vid_name if not vid_name.endswith('.mp4') else vid_name[:-4]
                vid_score_path = os.path.join(category_output_dir, f"{save_name}_av_score.json")
                with open(vid_score_path, 'w', encoding='utf-8') as f:
                    json.dump(vid_score_dict, f, ensure_ascii=False, indent=4)
                
                if i < 3:  # Show first 3 examples
                    print(f"📄 {save_name}: {vid_score_dict}")
            print(f"✅ Saved {len(scores['names'])} individual video score files")

            # Compute mean for each score
            print(f"📊 Computing mean scores for {category}...")
            for score_name in score_list:
                if score_name == 'AV-Align':
                    # av_align_scores contains 'AV-Align' key storing pre-computed average
                    # Filter out that key and only compute mean from individual video scores
                    individual_scores = [v for k, v in av_align_scores.items() if k != 'AV-Align']
                    mean_score = float(np.mean(individual_scores)) if individual_scores else 0.0
                    print(f"🔍 AV-Align: {len(individual_scores)} videos, mean: {mean_score:.4f}")
                    print(f"🔍 Pre-computed average: {av_align_scores.get('AV-Align', 'NOT_FOUND')}")
                else:
                    mean_score = float(np.mean(scores[score_name]))
                nested_means[score_name][step_dir][category] = mean_score
                print(f"📈 {score_name} mean: {mean_score:.4f}")
            print(f"✅ Category {category} completed")

        # After finishing the step, add an "all" entry
        print(f"📊 Computing overall means for {step_dir}...")
        for score_name in score_list:
            step_dict = nested_means[score_name][step_dir]
            if step_dict:  # At least one category
                all_mean = float(np.mean(list(step_dict.values())))
                step_dict["all"] = all_mean
                print(f"📈 {score_name} overall mean: {all_mean:.4f}")
        print(f"✅ Step {step_dir} completed")

    # ====== Save ======
    # Check if step info needs to be added to filename
    if (args.video_save_path_subdir_name == args.video_eval_av_output_subdir_name and 
        args.specific_steps is not None and len(args.specific_steps) > 0):
        steps_str = "_".join(map(str, sorted(args.specific_steps)))
        mean_json_filename = f"eval_av_scores_means_steps_{steps_str}.json"
    else:
        mean_json_filename = "eval_av_scores_means.json"
    
    mean_json_path = os.path.join(output_root, mean_json_filename)
    print(f"💾 Saving final results to: {mean_json_path}")
    with open(mean_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_means, f, ensure_ascii=False, indent=4)

    # Check if multiple step files need to be merged
    if (args.video_save_path_subdir_name == args.video_eval_av_output_subdir_name and 
        args.specific_steps is not None and len(args.specific_steps) > 0):
        print(f"🔄 Merging step files...")
        merge_av_scores_files(output_root)

    print("\n🎉 === Final Results Summary ===")
    print("📊 Step/Category-wise Scores means:")
    for score_name, step_data in nested_means.items():
        print(f"\n📈 {score_name}:")
        for step, category_data in step_data.items():
            print(f"  {step}: {category_data}")
    print("\n✅ Evaluation completed successfully!")
