from typing import List

import torch
import torch.nn.functional as F


def compute_kl(paired_logits: List[torch.Tensor], gt_logits: torch.Tensor):

    # KL is computed as KL(gt || generated)
    # note the input ordering in PyTorch's kl_div is (Q, P) for KL (P || Q)
    EPS = 1e-6
    for paired_samples in paired_logits:

        # kl_ref = F.kl_div(
        #     (paired_samples.softmax(dim=1) + EPS).log(),
        #     gt_logits.softmax(dim=1),
        #     reduction="none",
        # ) / len(paired_samples)
        # kl_ref = torch.mean(kl_ref, dim=-1)

        # AudioGen use this formulation
        kl_softmax = F.kl_div(
            # (paired_samples.softmax(dim=1) + EPS).log(),
            F.log_softmax(paired_samples, dim=1),
            # gt_logits.softmax(dim=1),
            F.log_softmax(gt_logits, dim=1),
            reduction="sum",
            log_target=True,
        ) / len(paired_samples)

        # For multi-class audio clips, this formulation could be better
        kl_sigmoid = F.kl_div(
            # (paired_samples.sigmoid() + EPS).log(),
            F.logsigmoid(paired_samples),
            # gt_logits.sigmoid(),
            F.logsigmoid(gt_logits),
            reduction="sum",
            log_target=True,
        ) / len(paired_samples)

    return {
        "kl_sigmoid": float(kl_sigmoid),
        "kl_softmax": float(kl_softmax),
        # "kl_ref": float(kl_ref),
    }
