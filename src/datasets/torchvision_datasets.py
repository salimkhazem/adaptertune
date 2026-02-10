from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

from .splits import make_split_indices, make_split_indices_threeway
from ..utils.seed import make_generator, seed_worker

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


def _prepare_tiny_imagenet_val(root: str) -> Path:
    tiny_root = Path(root) / "tiny-imagenet-200"
    val_dir = tiny_root / "val"
    images_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"
    organized = val_dir / "organized_by_class"

    if organized.exists():
        return organized

    if not images_dir.exists() or not ann_path.exists():
        raise FileNotFoundError(
            f"TinyImageNet val split not found under {val_dir}. Expected images/ and val_annotations.txt."
        )

    organized.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, class_name = parts[0], parts[1]
            src = images_dir / image_name
            class_dir = organized / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            dst = class_dir / image_name
            if dst.exists() or not src.exists():
                continue
            try:
                dst.symlink_to(src)
            except Exception:
                import shutil

                shutil.copy2(src, dst)
    return organized


def _imagenet_r_paths(root: str) -> tuple[Path, Path | None]:
    base = Path(root) / "imagenet-r"
    if not base.exists():
        raise FileNotFoundError(
            f"ImageNet-R folder not found at {base}. Place class folders there or use train/test subfolders."
        )
    train_dir = base / "train"
    test_dir = base / "test"
    if train_dir.exists():
        return train_dir, test_dir if test_dir.exists() else None
    return base, test_dir if test_dir.exists() else None


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
    if name == "Flowers102":
        if train:
            train_set = datasets.Flowers102(root=root, split="train", transform=transform, download=True)
            val_set = datasets.Flowers102(root=root, split="val", transform=transform, download=True)
            return ConcatDataset([train_set, val_set])
        return datasets.Flowers102(root=root, split="test", transform=transform, download=True)
    if name == "FGVCAircraft":
        split = "trainval" if train else "test"
        return datasets.FGVCAircraft(
            root=root,
            split=split,
            annotation_level="variant",
            transform=transform,
            download=True,
        )
    if name == "TinyImageNet":
        tiny_root = Path(root) / "tiny-imagenet-200"
        if train:
            train_root = tiny_root / "train"
            if not train_root.exists():
                raise FileNotFoundError(
                    f"TinyImageNet train split not found at {train_root}. Download/extract tiny-imagenet-200 under {root}."
                )
            return datasets.ImageFolder(str(train_root), transform=transform)
        return datasets.ImageFolder(str(_prepare_tiny_imagenet_val(root)), transform=transform)
    if name == "ImageNetR":
        train_root, test_root = _imagenet_r_paths(root)
        if train:
            return datasets.ImageFolder(str(train_root), transform=transform)
        if test_root is not None:
            return datasets.ImageFolder(str(test_root), transform=transform)
        # Fallback: full dataset used when create_dataloaders performs deterministic 3-way split.
        return datasets.ImageFolder(str(train_root), transform=transform)
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
    if name == "Flowers102":
        return 102
    if name == "FGVCAircraft":
        return 100
    if name == "TinyImageNet":
        return 200
    if name == "ImageNetR":
        return 200
    raise ValueError(f"Unknown dataset: {name}")


def create_dataloaders(cfg: dict, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    name = cfg["dataset"]
    root = cfg["root"]
    val_split = float(cfg.get("val_split", 0.1))
    test_split = float(cfg.get("test_split", 0.2))
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 8))
    pin_memory = bool(cfg.get("pin_memory", True))
    img_size = int(cfg.get("img_size", 224))

    train_tf, eval_tf = build_transforms(img_size)

    if name == "ImageNetR":
        _, explicit_test_root = _imagenet_r_paths(root)
        if explicit_test_root is None:
            train_full = _get_dataset(name, root, train=True, transform=train_tf)
            eval_full = _get_dataset(name, root, train=True, transform=eval_tf)
            train_idx, val_idx, test_idx = make_split_indices_threeway(
                dataset_len=len(train_full),
                root=root,
                name=name,
                val_split=val_split,
                test_split=test_split,
                seed=seed,
            )
            train_set = Subset(train_full, train_idx)
            val_set = Subset(eval_full, val_idx)
            test_set = Subset(eval_full, test_idx)
        else:
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
    else:
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
