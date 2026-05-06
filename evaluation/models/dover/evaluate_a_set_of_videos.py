import torch

import argparse
import os
import pickle as pkl

import decord
import numpy as np
import yaml
from tqdm import tqdm

from dover.datasets import (
    UnifiedFrameSampler,
    ViewDecompositionDataset,
    spatial_temporal_view_decomposition,
)
from dover.models import DOVER

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


def fuse_results(results: list):
    ## results[0]: aesthetic, results[1]: technical
    ## thank @dknyxh for raising the issue
    t, a = (results[1] - 0.1107) / 0.07355, (results[0] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1 / (1 + np.exp(-a)),
        "technical": 1 / (1 + np.exp(-t)),
        "overall": 1 / (1 + np.exp(-x)),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--opt", type=str, default="./dover.yml", help="the option file"
    )

    ## can be your own
    parser.add_argument(
        "-in",
        "--input_video_dir",
        type=str,
        default="./demo",
        help="the input video dir",
    )

    parser.add_argument(
        "-out",
        "--output_result_csv",
        type=str,
        default="./dover_predictions/demo.csv",
        help="the input video dir",
    )

    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="the running device"
    )

    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Load DOVER
    evaluator = DOVER(**opt["model"]["args"]).to(args.device)
    evaluator.load_state_dict(
        torch.load(opt["test_load_path"], map_location=args.device)
    )

    video_paths = []
    all_results = {}

    with open(args.output_result_csv, "w") as w:
        w.write(f"path, aesthetic score, technical score, overall/final score\n")

    dopt = opt["data"]["val-l1080p"]["args"]

    dopt["anno_file"] = None
    dopt["data_prefix"] = args.input_video_dir

    dataset = ViewDecompositionDataset(dopt)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
    )

    try:
        with open(
            f"dover_predictions/val-custom_{args.input_video_dir.split('/')[-1]}.pkl",
            "rb",
        ) as rf:
            all_results = pkl.dump(all_results, rf)
        print(f"Starting from {len(all_results)}.")
    except:
        print("Starting over.")

    sample_types = ["aesthetic", "technical"]

    for i, data in enumerate(tqdm(dataloader, desc="Testing")):
        if len(data.keys()) == 1:
            ##  failed data
            continue

        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(args.device)
                b, c, t, h, w = video[key].shape
                video[key] = (
                    video[key]
                    .reshape(
                        b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                    )
                )

        with torch.no_grad():
            results = evaluator(video, reduce_scores=False)
            results = [np.mean(l.cpu().numpy()) for l in results]

        rescaled_results = fuse_results(results)
        # all_results[data["name"][0]] = rescaled_results

        # with open(
        #    f"dover_predictions/val-custom_{args.input_video_dir.split('/')[-1]}.pkl", "wb"
        # ) as wf:
        # pkl.dump(all_results, wf)
        
        with open("zero_shot_res_sensehdr.txt","a") as wf:
            wf.write(f'{data["name"][0].split("/")[-1]},{rescaled_results["aesthetic"]*100:4f}, {rescaled_results["technical"]*100:4f},{rescaled_results["overall"]*100:4f}\n')

        with open(args.output_result_csv, "a") as w:
            w.write(
                f'{data["name"][0]}, {rescaled_results["aesthetic"]*100:4f}, {rescaled_results["technical"]*100:4f},{rescaled_results["overall"]*100:4f}\n'
            )
