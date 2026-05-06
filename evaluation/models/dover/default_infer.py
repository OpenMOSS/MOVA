import torch

import argparse
import math
import os.path as osp
import pickle
import random
from time import time

import cv2
import numpy as np
import torch
import yaml
from scipy.stats import kendalltau as kendallr
from scipy.stats import pearsonr, spearmanr
from thop import profile
from tqdm import tqdm

import dover.datasets as datasets
import dover.models as models
import wandb


def rescale(pr, gt=None):
    if gt is None:
        print("mean", np.mean(pr), "std", np.std(pr))
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        print(np.mean(pr), np.std(pr), np.std(gt), np.mean(gt))
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr


sample_types = ["aesthetic", "technical"]


def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device)
            c, t, h, w = video[key].shape
            video[key] = (
                video[key]
                .reshape(
                    1, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                )
                .permute(0, 2, 1, 3, 4, 5)
                .reshape(data["num_clips"][key], c, t // data["num_clips"][key], h, w)
            )
    with torch.no_grad():
        flops, params = profile(model, (video,))
    print(
        f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M."
    )


def inference_set(
    inf_loader, model, device, best_, save_model=False, suffix="s", set_name="na"
):
    print(f"Validating for {set_name}.")
    results = []
    try:
        model = torch.compile(model)
    except:
        print("You may try to accelerate your model with torch 2.0")

    best_s, best_p, best_k, best_r = best_

    names = []
    keys = []

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video = {}
        for key in sample_types:
            if key not in keys:
                keys.append(key)
            if key in data:
                video[key] = data[key].to(device)
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
            labels = model(video, reduce_scores=False)
            labels = [np.mean(l.cpu().numpy()) for l in labels]
            result["pr_labels"] = labels
        result["gt_label"] = data["gt_label"].item()
        result["name"] = data["name"]
        names.append(data["name"][0])
        # result['frame_inds'] = data['frame_inds']
        # del data
        results.append(result)

    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = 0
    pr_dict = {}
    weights = [1 - inf_loader.dataset.weight, inf_loader.dataset.weight]
    for i, w, key in zip(range(len(results[0]["pr_labels"])), weights, keys):
        key_pr_labels = rescale([np.mean(r["pr_labels"][i]) for r in results])
        pr_labels += key_pr_labels * w
        pr_dict[key] = key_pr_labels

    # with open(f"dover_predictions/{set_name}.pkl", "wb") as f:
    #    pickle.dump(pr_dict, f)
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    results = sorted(results, key=lambda x: x["pr_labels"])

    try:
        wandb.log(
            {
                f"val/SRCC-{suffix}": s,
                f"val/PLCC-{suffix}": p,
                f"val/KRCC-{suffix}": k,
                f"val/RMSE-{suffix}": r,
            }
        )
    except:
        pass

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    try:
        wandb.log(
            {
                f"val/best_SRCC-{suffix}": best_s,
                f"val/best_PLCC-{suffix}": best_p,
                f"val/best_KRCC-{suffix}": best_k,
                f"val/best_RMSE-{suffix}": best_r,
            }
        )
    except:
        pass
    print(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )
    return best_s, best_p, best_k, best_r, pr_labels, names


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="dover.yml", help="the option file"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="the running device"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    ## adaptively choose the device
    device = args.device

    ## defining model and loading checkpoint

    bests_ = []

    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)

    state_dict = torch.load(
        opt["test_load_path"], map_location=device
    )  # ["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    for key in opt["data"].keys():

        if "val" not in key and "test" not in key:
            continue

        run = wandb.init(
            project=opt["wandb"]["project_name"],
            name=opt["name"] + "_Test_" + key,
            reinit=True,
            settings=wandb.Settings(start_method='thread'),
        )

        val_dataset = getattr(datasets, opt["data"][key]["type"])(
            opt["data"][key]["args"]
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )

        profile_inference(val_dataset, model, device)

        # test the model
        print(len(val_loader))

        best_ = -1, -1, -1, 1000

        best_ = inference_set(val_loader, model, device, best_, set_name=key,)

        print(
            f"""Testing result on: [{len(val_loader)}] videos:
            SROCC: {best_[0]:.4f}
            PLCC:  {best_[1]:.4f}
            KROCC: {best_[2]:.4f}
            RMSE:  {best_[3]:.4f}."""
        )

        run.finish()


if __name__ == "__main__":
    main()
