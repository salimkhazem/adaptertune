import argparse

import torch

from src.datasets.torchvision_datasets import create_dataloaders
from src.models.timm_backbones import prepare_model
from src.train.trainer import Trainer
from src.utils.io import load_config
from src.utils.logging import RunLogger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=[], help="Config paths (can be repeated)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (best.pt or last.pt)")
    parser.add_argument("--device", default=None, help="Single CUDA device index or 'cpu'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    set_seed(seed, deterministic)

    device_str = args.device or cfg.get("system", {}).get("devices", "0").split(",")[0]
    if device_str == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_str}")

    logger = RunLogger(cfg["logging"]["output_dir"], cfg["logging"]["exp_name"] + "_eval")

    train_loader, val_loader, test_loader, num_classes = create_dataloaders(cfg["data"], seed)
    model = prepare_model(cfg["model"], cfg["method"], num_classes)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    trainer = Trainer(model, (train_loader, val_loader, test_loader), cfg, logger, num_classes, device)
    metrics = trainer.evaluate("test")
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
