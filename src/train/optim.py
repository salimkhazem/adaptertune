from typing import Dict

import torch


def create_optimizer(cfg: Dict, model: torch.nn.Module):
    opt_name = cfg.get("opt", "adamw").lower()
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {opt_name}")
