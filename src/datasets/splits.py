import os
import pickle
from pathlib import Path
from typing import Tuple

import torch

from ..utils.io import ensure_dir


def _split_file(root: str, name: str, val_split: float, seed: int) -> Path:
    split_dir = ensure_dir(Path(root) / "splits")
    fname = f"{name}_val{val_split}_seed{seed}.pkl"
    return split_dir / fname


def make_split_indices(
    dataset_len: int,
    root: str,
    name: str,
    val_split: float,
    seed: int,
) -> Tuple[list[int], list[int]]:
    split_path = _split_file(root, name, val_split, seed)
    if split_path.exists():
        with open(split_path, "rb") as f:
            data = pickle.load(f)
        return data["train"], data["val"]

    n_val = int(dataset_len * val_split)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_len, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    with open(split_path, "wb") as f:
        pickle.dump({"train": train_idx, "val": val_idx}, f)
    return train_idx, val_idx


def make_split_indices_threeway(
    dataset_len: int,
    root: str,
    name: str,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[list[int], list[int], list[int]]:
    split_dir = ensure_dir(Path(root) / "splits")
    fname = f"{name}_val{val_split}_test{test_split}_seed{seed}.pkl"
    split_path = split_dir / fname
    if split_path.exists():
        with open(split_path, "rb") as f:
            data = pickle.load(f)
        return data["train"], data["val"], data["test"]

    n_test = int(dataset_len * test_split)
    n_trainval = dataset_len - n_test
    n_val = int(n_trainval * val_split)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_len, generator=g).tolist()
    test_idx = perm[:n_test]
    trainval_idx = perm[n_test:]
    val_idx = trainval_idx[:n_val]
    train_idx = trainval_idx[n_val:]

    with open(split_path, "wb") as f:
        pickle.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)
    return train_idx, val_idx, test_idx
