#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cpCER (Concatenated Permutation Character Error Rate) Evaluation Script

Evaluates multi-speaker conversational accuracy in generated videos by:
1. Extracting audio from videos
2. Running ASR (MOSS Transcribe Diarize) to transcribe with speaker tags ([S01], [S02], etc.)
3. Comparing predictions against reference transcripts using cpCER

cpCER measures:
- Whether the conversational content is accurate
- Whether the speaker assignments are correct

Lower is better.

Usage:
  python eval_cpcer.py \
    --video_dir /path/to/multi-speaker/videos \
    --references_txt /path/to/references.txt \
    --asr_api_key YOUR_API_KEY
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List

import requests

# Ensure local modules are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from speaker_metrics import cp_cer  # noqa: E402

# =========================
# ASR Configuration
# =========================
DEFAULT_API_URL = os.environ.get(
    "CPCER_ASR_API_URL",
    "https://studio.mosi.cn/api/v1/audio/transcriptions"
)
DEFAULT_API_KEY = os.environ.get(
    "CPCER_ASR_API_KEY",
    ""
)
DEFAULT_MODEL = os.environ.get(
    "CPCER_ASR_MODEL",
    "moss-transcribe-diarize"
)

FIXED_PROMPT = (
    "转录为文本，使用 [S01] [S02] [MULTI]等说话人标签。"
)

MAX_AUDIO_DURATION = 30  # seconds
TARGET_SAMPLE_RATE = 16000


# =========================
# Utility functions
# =========================
def read_txt_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_duration(file_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return float(out.strip())
    except Exception:
        return 0.0


def file_to_audio_bytes(file_path: str, duration_limit: float) -> bytes:
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", file_path,
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        "-t", str(duration_limit),
        "-f", "wav",
        "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    return proc.stdout


def file_to_data_uri(file_path: str) -> str:
    duration = get_duration(file_path)
    duration_limit = min(duration, MAX_AUDIO_DURATION)

    audio_bytes = file_to_audio_bytes(file_path, duration_limit)
    b64 = base64.b64encode(audio_bytes).decode("utf-8")

    print(f"[DEBUG] audio size = {len(b64)/1024/1024:.2f} MB")
    return f"data:audio/wav;base64,{b64}"


def call_remote_asr(audio_data_uri: str, api_url: str, api_key: str, model: str) -> str:
    """Call MOSS Transcribe Diarize API and return formatted prediction text."""
    payload = {
        "prompt": FIXED_PROMPT,
        "model": model,
        "audio_data": audio_data_uri,
        "sampling_params": {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "top_k": 20,
            "top_p": 1.0,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=180
            )
            break
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"[WARNING] API timeout, retrying ({attempt+2}/{max_retries})...")
                import time; time.sleep(5)
            else:
                raise

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()

    # Fallback: try raw text fields (same as original eval_cpcer_check.py)
    for k in ("text", "result", "transcription", "output", "generated_text"):
        if k in data and isinstance(data[k], str) and ("[S0" in data[k] or "[S1" in data[k]):
            return data[k]

    # Structured response from studio.mosi.cn API: reconstruct with speaker tags from segments
    asr_result = data.get("asr_transcription_result", {})
    segments = asr_result.get("segments", [])
    if segments:
        parts = []
        for seg in segments:
            spk = seg.get("speaker", "S01")
            # Normalize speaker tag format: S1 -> S01, S02 stays S02
            spk_num = re.search(r'S(\d+)', spk)
            if spk_num:
                spk_tag = f"[S{int(spk_num.group(1)):02d}]"
            else:
                spk_tag = f"[{spk}]"
            text = seg.get("text", "").strip()
            if text:
                parts.append(f"{spk_tag} {text}")
        if parts:
            return " ".join(parts)

    # Last fallback: plain full_text without speaker tags
    full_text = asr_result.get("full_text", "")
    if full_text:
        return full_text

    return json.dumps(data, ensure_ascii=False)


def post_process(text: str) -> str:
    """Post-process ASR output: remove non-speaker tags, merge consecutive same-speaker segments."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s*(\[[Ss]\d+\])\s*', r'\n\1 ', text).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    results = []
    cur_spk = None
    cur_buf = []

    for line in lines:
        m = re.match(r'^(\[[Ss]\d+\])\s*(.*)$', line)
        if m:
            spk = m.group(1).upper()
            content = m.group(2).strip()

            if spk == cur_spk:
                if content:
                    cur_buf.append(content)
            else:
                if cur_spk and cur_buf:
                    results.append(f"{cur_spk} {' '.join(cur_buf)}")
                cur_spk = spk
                cur_buf = [content] if content else []
        else:
            if cur_spk:
                cur_buf.append(line)

    if cur_spk and cur_buf:
        results.append(f"{cur_spk} {' '.join(cur_buf)}")

    out = "\n".join(results)
    out = re.sub(r'\s+([，。！？；：])', r'\1', out)
    out = re.sub(r'\s+', ' ', out)
    out = out.replace(" \n", "\n").strip()
    return out


def collect_sorted_videos(video_dir: str) -> List[str]:
    """Collect and sort videos by numeric prefix in filename (e.g., 01_xxx.mp4, 02_xxx.mp4)."""
    video_dir = Path(video_dir)
    assert video_dir.exists(), f"Directory not found: {video_dir}"

    videos = []
    for p in video_dir.iterdir():
        if p.suffix.lower() == ".mp4":
            m = re.match(r"(\d+)_", p.name)
            if m:
                videos.append((int(m.group(1)), str(p)))

    videos.sort(key=lambda x: x[0])
    return [v[1] for v in videos]


# =========================
# Main evaluation flow
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="cpCER Evaluation - Multi-speaker conversational accuracy assessment"
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to directory containing multi-speaker videos")
    parser.add_argument("--references_txt", type=str, required=True,
                        help="Path to references.txt file (one reference per line, ordered by video index)")
    # ASR configuration
    parser.add_argument("--asr_api_url", type=str, default=None,
                        help="ASR API URL (overrides CPCER_ASR_API_URL env var)")
    parser.add_argument("--asr_api_key", type=str, default=None,
                        help="ASR API key (overrides CPCER_ASR_API_KEY env var)")
    parser.add_argument("--asr_model", type=str, default=None,
                        help="ASR model name (overrides CPCER_ASR_MODEL env var)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results as JSON (for run_eval.sh --resume support)")

    args = parser.parse_args()

    # Resolve ASR config
    api_url = args.asr_api_url or DEFAULT_API_URL
    api_key = args.asr_api_key or DEFAULT_API_KEY
    model = args.asr_model or DEFAULT_MODEL

    if not api_key:
        print("Error: ASR API key is required. Set CPCER_ASR_API_KEY env var or use --asr_api_key.")
        print("  Get your API key at: https://studio.mosi.cn")
        sys.exit(1)

    print("cpCER Evaluation Configuration:")
    print(f"  Video directory: {args.video_dir}")
    print(f"  References file: {args.references_txt}")
    print(f"  ASR API URL: {api_url}")
    print(f"  ASR Model: {model}")
    print(f"  API Key: {'***' + api_key[-8:] if len(api_key) > 8 else '***'}")

    # Collect sorted videos and references
    video_list = collect_sorted_videos(args.video_dir)
    references = read_txt_lines(args.references_txt)

    assert len(video_list) == len(references), (
        f"Number of videos ({len(video_list)}) != number of references ({len(references)})"
    )

    print(f"  Found {len(video_list)} videos and {len(references)} references")

    predictions = []
    sample_scores: List[dict] = []

    for idx, video_path in enumerate(video_list):
        print(f"\n===== Sample {idx} =====")
        print(f"Video: {video_path}")

        audio_uri = file_to_data_uri(video_path)
        raw_pred = call_remote_asr(audio_uri, api_url, api_key, model)
        pred = post_process(raw_pred)
        ref = references[idx]

        predictions.append(pred)

        sample_cpcer = cp_cer([pred], [ref])

        sample_scores.append({
            "idx": idx,
            "video": video_path,
            "cpcer": sample_cpcer,
            "prediction": pred,
            "reference": ref
        })

        print("Prediction:")
        print(pred)
        print("Reference:")
        print(ref)
        print(f"[Sample cpCER]: {sample_cpcer:.4f}%")

    # Compute overall cpCER
    final_score = cp_cer(predictions, references)

    print("\n==============================")
    print(f"Final cpCER (ALL): {final_score:.4f}%")
    print("==============================")

    # Save results to JSON if requested (for run_eval.sh --resume support)
    if args.output_json:
        output_data = {
            "cpCER": {
                "multi-speaker": final_score,
            },
            "per_sample": [
                {
                    "idx": s["idx"],
                    "video": os.path.basename(s["video"]),
                    "cpcer": s["cpcer"],
                }
                for s in sample_scores
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
