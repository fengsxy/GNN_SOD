import random
import numpy as np
import torch


def check_model_version(checkpoint):
    if checkpoint.get("format_version") != 2:
        raise ValueError("Incompatible decoder checkpoint: version 2 uses sequential fusion; retrain the decoder")


def configure(device="cpu", threads=2, seed=0):
    if threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    return device


def check_config(expected, actual):
    keys = ("schema_version", "size", "segments", "knn", "encoder", "feature_signature",
            "node_channels", "node_activation", "topology", "preprocess")
    mismatches = [k for k in keys if expected.get(k) != actual.get(k)]
    if mismatches:
        raise ValueError("Incompatible cache/checkpoint configuration: " + ", ".join(mismatches))
