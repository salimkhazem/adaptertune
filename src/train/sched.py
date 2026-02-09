import math
from typing import Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(cfg: Dict, optimizer: Optimizer, steps_per_epoch: int):
    sched = cfg.get("sched", "cosine").lower()
    epochs = int(cfg.get("epochs", 50))
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = warmup_epochs * steps_per_epoch

    if sched == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    if sched == "none":
        return None

    raise ValueError(f"Unknown scheduler: {sched}")
