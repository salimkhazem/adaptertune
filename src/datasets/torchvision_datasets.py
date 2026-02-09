from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .splits import make_split_indices
from ..utils.seed import seed_worker, make_generator

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 224):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def _get_dataset(name: str, root: str, train: bool, transform):
    if name == "CIFAR10":
        return datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
    if name == "CIFAR100":
        return datasets.CIFAR100(root=root, train=train, transform=transform, download=True)
    if name == "SVHN":
        split = "train" if train else "test"
        return datasets.SVHN(root=root, split=split, transform=transform, download=True)
    if name == "OxfordIIITPet":
        split = "trainval" if train else "test"
        return datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="category",
            transform=transform,
            download=True,
        )
    if name == "Food101":
        split = "train" if train else "test"
        return datasets.Food101(root=root, split=split, transform=transform, download=True)
    raise ValueError(f"Unknown dataset: {name}")


def _num_classes(name: str) -> int:
    if name == "CIFAR10":
        return 10
    if name == "CIFAR100":
        return 100
    if name == "SVHN":
        return 10
    if name == "OxfordIIITPet":
        return 37
    if name == "Food101":
        return 101
    raise ValueError(f"Unknown dataset: {name}")


def create_dataloaders(cfg: dict, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    name = cfg["dataset"]
    root = cfg["root"]
    val_split = float(cfg.get("val_split", 0.1))
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 8))
    pin_memory = bool(cfg.get("pin_memory", True))
    img_size = int(cfg.get("img_size", 224))

    train_tf, eval_tf = build_transforms(img_size)

    train_full = _get_dataset(name, root, train=True, transform=train_tf)
    test_set = _get_dataset(name, root, train=False, transform=eval_tf)

    train_idx, val_idx = make_split_indices(
        dataset_len=len(train_full),
        root=root,
        name=name,
        val_split=val_split,
        seed=seed,
    )
    train_set = Subset(train_full, train_idx)
    val_set = Subset(_get_dataset(name, root, train=True, transform=eval_tf), val_idx)

    g = make_generator(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader, test_loader, _num_classes(name)
