"""Shared constants, paths, device selection and seeding."""

import os
import random
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
SEEDS = (0, 1, 2)
DROP_LAYERS = (4, 6, 8)          # token removed from the output of layer l (1-indexed)
EPSILONS = (0.005, 0.01, 0.02)   # 0.01 is the main threshold
PRUNE_FRACTIONS = {"M0": 0.0, "M25": 0.25, "M50": 0.50}

# Outputs go to Google Drive on Colab so they survive disconnects; override with TR_OUTPUT_DIR.
_DRIVE = Path("/content/drive/MyDrive/token-redundancy")
OUTPUT_DIR = Path(os.environ.get("TR_OUTPUT_DIR", _DRIVE if _DRIVE.parent.exists() else "outputs"))
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
