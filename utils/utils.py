import torch
import os
import json
import numpy as np
import random
import cv2


def load_checkpoint(filename, optimizer_key=None):
    assert filename.endswith(".ckpt"), "Error: filename is not a pth file"
    assert os.path.isfile(filename), "Error: checkpoint file not found"

    checkpoint = torch.load(filename)
    optimizer = None
    if optimizer_key is not None:
        if optimizer_key in checkpoint.keys():
            optimizer = checkpoint[optimizer_key]
            del checkpoint[optimizer_key]
        else:
            print("Warning: optimizer not found in checkpoint")

    if "state_dict" not in checkpoint:
        state_dict = checkpoint
        info = None
    else:
        state_dict = checkpoint["state_dict"]
        del checkpoint["state_dict"]
        info = checkpoint

    return state_dict, optimizer, info


def save_checkpoint(checkpoint, filename, qat=False):
    assert filename.endswith(".ckpt"), "Error: filename is not a pth file"

    torch.save(checkpoint, filename)
    print("Checkpoint saved to {}".format(filename))


def set_seed(seed):
    assert isinstance(seed, int), "Error: seed must be an integer"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cv2.setRNGSeed(seed)


def save_json(config_file, filename):
    assert str(filename).endswith(".json"), (
        f"Error: filename is not a json file: {filename}"
    )

    with open(filename, "w") as f:
        json.dump(config_file, f, indent=4, sort_keys=True)
        print("Saved to {}".format(filename))


def load_json(filename):
    assert filename.endswith(".json"), f"Error: filename is not a json file: {filename}"
    assert os.path.isfile(filename), "Error: json file not found"

    with open(filename, "r") as f:
        return json.load(f)


def print_all_results(results_dict):
    for k in results_dict:
        print(k, "results:")
        print_results(results_dict[k])


def print_results(results_dict):
    """Print results as a .md table but transposed"""
    metrics = list(results_dict.keys())
    header = "| | " + " | ".join(metrics) + " |"
    separator = "|---|" + "---|" * len(metrics)
    values = "| " + " | ".join(["%.2f" % results_dict[k] for k in metrics]) + " |"
    print(header)
    print(separator)
    print(values)
