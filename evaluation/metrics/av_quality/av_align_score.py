import argparse
import cv2
import librosa
import numpy as np
import subprocess
import os
import glob
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------- Video Processing ----------------------
def extract_frames(video_path: str):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

# ---------------------- Audio Processing ----------------------
def detect_audio_peaks_from_video(video_path: str, sr: int = 44100, onset_interval_s: float = 0.032, use_hop_length: bool = False):
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(sr),
        "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_bytes = proc.stdout
    audio = np.frombuffer(audio_bytes, dtype=np.float32)

    if use_hop_length:
        hop_length = int(sr * onset_interval_s)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        all_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    else:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_env = np.array(onset_env)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        all_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

    return onset_times, all_times, onset_env

# ---------------------- Video Optical Flow ----------------------
def compute_of(img1, img2):
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    avg_magnitude = cv2.mean(magnitude)[0]
    return avg_magnitude

def detect_video_peaks(frames, fps):
    flow_trajectory = [compute_of(frames[0], frames[1])] + \
                      [compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    flow_trajectory = np.array(flow_trajectory)
    video_peaks = []
    n = len(flow_trajectory)
    for i in range(1, n - 1):
        if flow_trajectory[i - 1] < flow_trajectory[i] > flow_trajectory[i + 1] and flow_trajectory[i] >= 0.1:
            video_peaks.append(i / fps)
    video_peaks = np.array(video_peaks)
    return flow_trajectory, video_peaks

# ---------------------- AV-Align Computation ----------------------
def calc_intersection_over_union(audio_peaks, video_peaks, fps):
    intersection_length = 0
    used_video_peaks = [False] * len(video_peaks)
    for audio_peak in audio_peaks:
        for j, video_peak in enumerate(video_peaks):
            if not used_video_peaks[j] and video_peak - 1 / fps < audio_peak < video_peak + 1 / fps:
                intersection_length += 1
                used_video_peaks[j] = True
                break
    denominator = len(audio_peaks) + len(video_peaks) - intersection_length
    if denominator == 0:
        return 0.0
    return intersection_length / denominator

# ---------------------- Visualization ----------------------
def visualize_av_alignment(video_path, frames, fps, audio_peaks, audio_times, onset_env, flow_trajectory, video_peaks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)

    max_onset = max(onset_env) if len(onset_env) > 0 else 1.0
    max_flow = max(flow_trajectory) if len(flow_trajectory) > 0 else 1.0

    # Stacked subplot (audio + video)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(audio_times, onset_env, label="Audio Onset Strength")
    ax[0].vlines(audio_peaks, ymin=0, ymax=max_onset, color='r', alpha=0.7, label="Audio Peaks")
    ax[0].set_title("Audio Onset Detection")
    ax[0].legend()
    times_video = np.arange(len(flow_trajectory)) / fps
    ax[1].plot(times_video, flow_trajectory, label="Video Flow Magnitude")
    ax[1].vlines(video_peaks, ymin=0, ymax=max_flow, color='g', alpha=0.7, label="Video Peaks")
    ax[1].set_title("Video Optical Flow")
    ax[1].legend()
    plt.xlabel("Time (s)")
    plt.tight_layout()
    fig_path1 = os.path.join(output_dir, f"{video_name}_av_alignment_2rows.png")
    plt.savefig(fig_path1)
    plt.close()

    # Single-line comparison plot
    plt.figure(figsize=(12, 3))
    plt.plot(audio_times, onset_env, label="Audio Onset Strength")
    normalized_flow = flow_trajectory / max_flow * max_onset
    plt.plot(times_video, normalized_flow, label="Video Flow Magnitude (normalized)")
    plt.vlines(audio_peaks, ymin=0, ymax=max_onset, color='r', alpha=0.7, label="Audio Peaks")
    plt.vlines(video_peaks, ymin=0, ymax=max_onset, color='g', alpha=0.7, label="Video Peaks")
    plt.title("Audio vs Video Peaks (Single Line)")
    plt.xlabel("Time (s)")
    plt.legend()
    fig_path2 = os.path.join(output_dir, f"{video_name}_av_alignment_single_line.png")
    plt.savefig(fig_path2)
    plt.close()

    # Concatenate video frames and visualization into a new video
    vis_img = cv2.imread(fig_path1)
    h_vis, w_vis, _ = vis_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out_path = os.path.join(output_dir, f"{video_name}_av_visualization.mp4")
    out = cv2.VideoWriter(video_out_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0] + h_vis))

    for frame in frames:
        frame_resized = cv2.resize(frame, (frames[0].shape[1], frames[0].shape[0]))
        vis_resized = cv2.resize(vis_img, (frames[0].shape[1], h_vis))
        combined = np.vstack((frame_resized, vis_resized))
        out.write(combined)
    out.release()

    # Save a copy of the original video
    orig_video_out = os.path.join(output_dir, video_name)
    if not os.path.exists(orig_video_out):
        subprocess.run(["cp", video_path, orig_video_out])

# ---------------------- Single Video Processing ----------------------
def compute_single_video_av(video_path: str, sr: int = 44100, onset_interval_s: float = 0.032,
                            use_hop_length: bool = False, visualize: bool = False, output_dir: str = None):
    try:
        frames, fps = extract_frames(video_path)
        if len(frames) < 2:
            return os.path.basename(video_path), 0.0

        audio_peaks, audio_times, onset_env = detect_audio_peaks_from_video(
            video_path, sr=sr, onset_interval_s=onset_interval_s, use_hop_length=use_hop_length
        )
        flow_trajectory, video_peaks = detect_video_peaks(frames, fps)
        video_score = calc_intersection_over_union(audio_peaks, video_peaks, fps)

        if visualize and output_dir:
            visualize_av_alignment(video_path, frames, fps, audio_peaks, audio_times, onset_env, flow_trajectory, video_peaks, output_dir)

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        video_score = 0.0

    return os.path.basename(video_path), video_score

# ---------------------- Parallel Processing ----------------------
def compute_av_align_for_videos_parallel(video_paths: List[str], sr: int = 44100, onset_interval_s: float = 0.032,
                                         use_hop_length: bool = False, visualize: bool = False, output_dir: str = None, max_workers: int = 15):
    av_scores = {}
    total_score = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_single_video_av, path, sr, onset_interval_s,
                                   use_hop_length, visualize, output_dir) for path in video_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            video_name, video_score = future.result()
            av_scores[video_name] = video_score
            total_score += video_score

    avg_score = total_score / len(video_paths) if video_paths else 0.0
    av_scores['AV-Align'] = avg_score
    return av_scores

# ---------------------- CLI Entry Point ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_dir", type=str, default='', help='Videos folder path')
    parser.add_argument('-w', "--workers", type=int, default=15, help='Number of parallel workers')
    parser.add_argument('--sr', type=int, default=44100, help='Audio sample rate')
    parser.add_argument('--onset_interval', type=float, default=0.032, help='Audio onset detection interval in seconds')
    parser.add_argument('--use_hop_length', action='store_true', help='Whether to use hop_length for onset detection')
    parser.add_argument('--visualize', action='store_true', help='Whether to generate visualizations')
    parser.add_argument('--output_dir', type=str, default="./av_align_output", help='Output path for visualizations')
    args = parser.parse_args()


    video_list = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    if not video_list:
        print(f"No .mp4 files found in {args.input_dir}")
        exit(1)

    scores_dict = compute_av_align_for_videos_parallel(
        video_list, sr=args.sr, onset_interval_s=args.onset_interval,
        use_hop_length=args.use_hop_length, visualize=args.visualize,
        output_dir=args.output_dir, max_workers=args.workers
    )
    print("AV-Align scores:", scores_dict)
