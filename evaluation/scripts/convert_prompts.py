#!/usr/bin/env python3
"""
Convert prompt metadata files to the flat prompts.json format expected by MOVA eval scripts.

Supports three input formats:
  1. Nested prompt_meta_json ({category: {item_id: {prompt: "..."}}})
     - Matches video files by meta_id prefix in filenames (same logic as original code)
  2. generation_state.json ({jobs: [{file_path: "...", prompt: "..."}]})
  3. Flat JSON that is already in the correct format

Usage:
    # From nested prompt_meta_json with video directory:
    python convert_prompts.py --prompt_meta_json benchmark.json --video_dir /path/to/videos --output prompts.json

    # From generation_state.json:
    python convert_prompts.py --generation_state generation_state.json --output prompts.json

    # From a flat JSON (just validate and copy):
    python convert_prompts.py --flat_prompts existing_prompts.json --output prompts.json
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


def convert_from_generation_state(gs_path: str) -> Dict[str, str]:
    """Convert generation_state.json to flat prompts.json."""
    with open(gs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    jobs = data.get('jobs', [])
    prompts = {}
    for job in jobs:
        file_path = job.get('file_path', '')
        prompt = job.get('prompt', '')
        if not file_path:
            continue
        # Use basename without .mp4 extension as key
        basename = os.path.basename(file_path)
        stem = basename[:-4] if basename.endswith('.mp4') else basename
        prompts[stem] = prompt

    return prompts


def _match_videos_to_items(video_dir: str, meta_dict: dict, prompt_field: str = 'video_prompt') -> Dict[str, str]:
    """Match video files in video_dir to items in the nested meta_dict.

    Uses the same matching logic as the original evaluation code:
    - Video filename format: {meta_id}_video_{prompt_text}_step{N}_seed{S}.mp4
    - Extract meta_id via vf.split("_")[0]
    - Match meta_id to the key in meta_dict[category]

    This handles the case where one benchmark item generates multiple videos
    (different seeds) — all videos with the same meta_id prefix get the same prompt.
    """
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    if not video_files:
        return {}

    # Build meta_id -> prompt mapping from the nested meta_dict
    # This matches the original code's prompt_categories logic:
    #   prompt_categories[category] = [(k, v['prompt']) for k, v in meta_dict[category].items()]
    meta_id_to_prompt: Dict[str, str] = {}
    for category, items in meta_dict.items():
        if not isinstance(items, dict):
            continue
        for item_id, item_data in items.items():
            if not isinstance(item_data, dict):
                continue
            prompt = item_data.get(prompt_field, '') or item_data.get('video_prompt', '') or item_data.get('prompt', '')
            if prompt:
                meta_id_to_prompt[item_id] = prompt

    if not meta_id_to_prompt:
        return {}

    # Match videos: extract meta_id from filename and look up prompt
    # Same as original: meta_id_to_video = {vf.split("_")[0]: path for vf in video_files}
    prompts = {}
    for vf in video_files:
        stem = vf[:-4]  # remove .mp4
        meta_id = vf.split("_")[0]
        if meta_id in meta_id_to_prompt:
            prompts[stem] = meta_id_to_prompt[meta_id]

    return prompts


def convert_from_prompt_meta_json(meta_path: str, video_dir: Optional[str] = None, prompt_field: str = 'video_prompt') -> Dict[str, str]:
    """Convert nested prompt_meta_json to flat prompts.json.

    If video_dir is provided, matches video files to prompt items.
    If not, creates a flat dict using "category/item_id" as keys.

    prompt_field: which field to extract from each item (default: 'prompt',
    matching the original evaluation code's behavior).
    """
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)

    if video_dir and os.path.isdir(video_dir):
        return _match_videos_to_items(video_dir, meta_dict, prompt_field)

    # No video directory: create flat dict with category/item_id keys
    prompts = {}
    for category, items in meta_dict.items():
        if not isinstance(items, dict):
            continue
        for item_id, item_data in items.items():
            if not isinstance(item_data, dict):
                continue
            prompt = item_data.get(prompt_field, '') or item_data.get('video_prompt', '') or item_data.get('prompt', '')
            if prompt:
                key = f"{category}/{item_id}"
                prompts[key] = prompt

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Convert prompt metadata to flat prompts.json format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--prompt_meta_json', type=str,
        help='Nested benchmark JSON ({category: {item_id: {prompt: "..."}}})'
    )
    group.add_argument(
        '--generation_state', type=str,
        help='generation_state.json ({jobs: [{file_path: "...", prompt: "..."}]})'
    )
    group.add_argument(
        '--flat_prompts', type=str,
        help='Already-flat prompts JSON (just validate and output)'
    )

    parser.add_argument(
        '--video_dir', type=str, default=None,
        help='Directory of .mp4 files (required with --prompt_meta_json for filename matching)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output path for prompts.json (default: <video_dir>/prompts.json or stdout)'
    )
    parser.add_argument(
        '--prompt_field', type=str, default='video_prompt',
        help='Field to extract from each item in prompt_meta_json (default: video_prompt). '
             'Falls back to "prompt" if video_prompt is not found in an item.'
    )

    args = parser.parse_args()

    if args.prompt_meta_json:
        prompts = convert_from_prompt_meta_json(args.prompt_meta_json, args.video_dir, args.prompt_field)
    elif args.generation_state:
        prompts = convert_from_generation_state(args.generation_state)
    elif args.flat_prompts:
        with open(args.flat_prompts, 'r', encoding='utf-8') as f:
            prompts = json.load(f)

    if not prompts:
        print("[WARN] No prompts extracted. Check your input files.", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    output_path = args.output
    if output_path is None and args.video_dir:
        output_path = os.path.join(args.video_dir, 'prompts.json')

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote {len(prompts)} prompts to {output_path}")
    else:
        json.dump(prompts, sys.stdout, ensure_ascii=False, indent=2)
        print()


if __name__ == '__main__':
    main()
