# audioldm_eval

import numpy as np
import torch


def compute_isc(featuresdict, feat_layer_name, rng_seed, samples_shuffle, splits):
    if feat_layer_name is not None:
        features = featuresdict[feat_layer_name]
    else:
        features = featuresdict

    assert torch.is_tensor(features) and features.dim() == 2
    N, C = features.shape
    if samples_shuffle:
        rng = np.random.RandomState(rng_seed)
        features = features[rng.permutation(N), :]
    features = features.double()

    p = features.softmax(dim=1)
    log_p = features.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        p_chunk = p[(i * N // splits):((i + 1) * N // splits), :]
        log_p_chunk = log_p[(i * N // splits):((i + 1) * N // splits), :]
        q_chunk = p_chunk.mean(dim=0, keepdim=True)
        kl = p_chunk * (log_p_chunk - q_chunk.log())
        kl = kl.sum(dim=1).mean().exp().item()
        scores.append(kl)

    return {
        "inception_score_mean": float(np.mean(scores)),
        "inception_score_std": float(np.std(scores)),
    }
