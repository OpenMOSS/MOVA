import torch

import argparse
import pickle as pkl

import decord
import numpy as np
import yaml

import onnxruntime as ort

from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


# 4-parameter sigmoid rescaling, as adviced by ITU
def fuse_results(results: list):
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (
        results[1] + 0.08285
    ) / 0.03774 * 0.3896
    print(x)
    return 1 / (1 + np.exp(-x))


def gaussian_rescale(pr):
    # The results should follow N(0,1)
    pr = (pr - np.mean(pr)) / np.std(pr)
    return pr


def uniform_rescale(pr):
    # The result scores should follow U(0,1)
    return np.arange(len(pr))[np.argsort(pr).argsort()] / len(pr)


def rescale_results(results: list, vname="undefined"):
    dbs = {
        "livevqc": "LIVE_VQC",
        "kv1k": "KoNViD-1k",
        "ltest": "LSVQ_Test",
        "l1080p": "LSVQ_1080P",
        "ytugc": "YouTube_UGC",
    }
    for abbr, full_name in dbs.items():
        with open(f"dover_predictions/val-{abbr}.pkl", "rb") as f:
            pr_labels = pkl.load(f)
        aqe_score_set = pr_labels["resize"]
        tqe_score_set = pr_labels["fragments"]
        tqe_score_set_p = np.concatenate((np.array([results[0]]), tqe_score_set), 0)
        aqe_score_set_p = np.concatenate((np.array([results[1]]), aqe_score_set), 0)
        tqe_nscore = gaussian_rescale(tqe_score_set_p)[0]
        tqe_uscore = uniform_rescale(tqe_score_set_p)[0]
        print(f"Compared with all videos in the {full_name} dataset:")
        print(
            f"-- the technical quality of video [{vname}] is better than {int(tqe_uscore*100)}% of videos, with normalized score {tqe_nscore:.2f}."
        )
        aqe_nscore = gaussian_rescale(aqe_score_set_p)[0]
        aqe_uscore = uniform_rescale(aqe_score_set_p)[0]
        print(
            f"-- the aesthetic quality of video [{vname}] is better than {int(aqe_uscore*100)}% of videos, with normalized score {aqe_nscore:.2f}."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--opt", type=str, default="./dover.yml", help="the option file"
    )

    ## can be your own
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        default="./demo/1724.mp4",
        help="the input video path",
    )


    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)


    dopt = opt["data"]["val-l1080p"]["args"]

    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            # resized temporal sampling for TQE in DOVER
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            # temporal sampling for AQE in DOVER
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )

    ### View Decomposition
    views, _ = spatial_temporal_view_decomposition(
        args.video_path, dopt["sample_types"], temporal_samplers
    )

    for k, v in views.items():
        num_clips = dopt["sample_types"][k].get("num_clips", 1)
        views[k] = (
            ((v.permute(1, 2, 3, 0) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
            .transpose(0, 1)
        )

    
    aes_input = views["aesthetic"]
    tech_input = views["technical"]
    ort_session = ort.InferenceSession("onnx_dover.onnx")
    
    import time
    
    s = time.time()
    predictions = ort_session.run(None, {"aes_view": aes_input.numpy(),
                           "tech_view": tech_input.numpy()})
    
    scores = [np.mean(s) for s in predictions]
    print(f"Inference time cost: {time.time() - s:.3f}s.")
    # predict fused overall score, with default score-level fusion parameters
    print(f"Normalized fused overall score (scale in [0,1]): {fuse_results(scores):.3f}")
        
