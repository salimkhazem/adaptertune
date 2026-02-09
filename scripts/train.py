import argparse
import os
from pathlib import Path

import torch

from src.datasets.torchvision_datasets import create_dataloaders
from src.models.timm_backbones import prepare_model
from src.train.optim import create_optimizer
from src.train.sched import create_scheduler
from src.train.trainer import Trainer
from src.utils.io import load_config
from src.utils.logging import RunLogger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=[], help="Config paths (can be repeated)")
    parser.add_argument("--override", action="append", default=[], help="Overrides like train.epochs=5")
    parser.add_argument("--device", default=None, help="Single CUDA device index or 'cpu'")
    parser.add_argument("--exp-name", default=None, help="Override experiment name")
    args = parser.parse_args()

    if not args.config:
        raise SystemExit("Provide at least one --config")

    cfg = load_config(args.config, args.override)
    if args.exp_name:
        cfg.setdefault("logging", {})["exp_name"] = args.exp_name

    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    set_seed(seed, deterministic)

    device_str = args.device or cfg.get("system", {}).get("devices", "0").split(",")[0]
    if device_str == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_str}")

    logger = RunLogger(cfg["logging"]["output_dir"], cfg["logging"]["exp_name"])
    if cfg.get("repro", {}).get("save_config", True):
        logger.log_config(cfg)
    if cfg.get("repro", {}).get("save_env", True):
        logger.log_env()
    if cfg.get("repro", {}).get("save_code_state", True):
        logger.log_code_state()

    train_loader, val_loader, test_loader, num_classes = create_dataloaders(cfg["data"], seed)

    try:
        model = prepare_model(cfg["model"], cfg["method"], num_classes)
    except ValueError as e:
        if "deit_tiny" in str(e):
            print("Warning: deit_tiny_patch16_224 not available in timm, skipping run.")
            return
        raise

    model.to(device)

    optimizer = create_optimizer(cfg["train"], model)
    scheduler = create_scheduler(cfg["train"], optimizer, steps_per_epoch=len(train_loader))

    trainer = Trainer(
        model,
        (train_loader, val_loader, test_loader),
        cfg,
        logger,
        num_classes,
        device,
    )
    summary = trainer.fit(optimizer, scheduler)
    logger.close()
    print("Run complete:", summary)


if __name__ == "__main__":
    main()
