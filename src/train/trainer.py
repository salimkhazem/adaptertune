from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..eval.metrics import accuracy
from ..utils.logging import RunLogger, count_trainable_params, count_total_params


@dataclass
class TrainState:
    epoch: int = 0
    best_val: float = 0.0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loaders: Tuple[DataLoader, DataLoader, DataLoader],
        cfg: Dict,
        logger: RunLogger,
        num_classes: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.cfg = cfg
        self.logger = logger
        self.num_classes = num_classes
        self.device = device
        self.state = TrainState()

        self.topk = [k for k in cfg.get("metrics", {}).get("topk", [1]) if k <= num_classes]
        if not self.topk:
            self.topk = [1]

        label_smoothing = float(cfg.get("train", {}).get("label_smoothing", 0.0))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _step(self, batch):
        images, targets = batch
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        outputs = self.model(images)
        loss = self.criterion(outputs, targets)
        return loss, outputs, targets

    def train_one_epoch(self, optimizer, scheduler=None) -> Dict:
        self.model.train()
        running_loss = 0.0
        correct = {k: 0.0 for k in self.topk}
        total = 0

        pbar = tqdm(self.train_loader, desc=f"train {self.state.epoch}", leave=False)
        for step, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)
            loss, outputs, targets = self._step(batch)
            loss.backward()
            grad_clip = float(self.cfg.get("train", {}).get("grad_clip", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            accs = accuracy(outputs, targets, self.topk)
            for k, acc in zip(self.topk, accs):
                correct[k] += acc.item() * targets.size(0) / 100.0
            if step % int(self.cfg.get("logging", {}).get("log_every", 50)) == 0:
                pbar.set_postfix({"loss": loss.item()})

        metrics = {"loss": running_loss / total}
        for k in self.topk:
            metrics[f"acc@{k}"] = 100.0 * correct[k] / total
        return metrics

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> Dict:
        self.model.eval()
        loader = self.val_loader if split == "val" else self.test_loader
        running_loss = 0.0
        correct = {k: 0.0 for k in self.topk}
        total = 0

        for batch in tqdm(loader, desc=f"{split} {self.state.epoch}", leave=False):
            loss, outputs, targets = self._step(batch)
            running_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            accs = accuracy(outputs, targets, self.topk)
            for k, acc in zip(self.topk, accs):
                correct[k] += acc.item() * targets.size(0) / 100.0

        metrics = {"loss": running_loss / total}
        for k in self.topk:
            metrics[f"acc@{k}"] = 100.0 * correct[k] / total
        return metrics

    def fit(self, optimizer, scheduler=None) -> Dict:
        epochs = int(self.cfg.get("train", {}).get("epochs", 1))
        save_ckpt = bool(self.cfg.get("logging", {}).get("save_ckpt", True))

        trainable = count_trainable_params(self.model)
        total_params = count_total_params(self.model)
        self.logger.log_metrics({
            "epoch": -1,
            "split": "meta",
            "trainable_params": trainable,
            "total_params": total_params,
        })

        best_summary = {}
        for epoch in range(epochs):
            self.state.epoch = epoch
            train_metrics = self.train_one_epoch(optimizer, scheduler)
            val_metrics = self.evaluate("val")

            row = {
                "epoch": epoch,
                "split": "train",
                **train_metrics,
            }
            self.logger.log_metrics(row)
            row = {
                "epoch": epoch,
                "split": "val",
                **val_metrics,
            }
            self.logger.log_metrics(row)

            val_acc = val_metrics.get("acc@1", 0.0)
            if val_acc > self.state.best_val:
                self.state.best_val = val_acc
                best_summary = {
                    "best_epoch": epoch,
                    "best_val_acc": val_acc,
                }
                if save_ckpt:
                    ckpt = {
                        "model": self.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(ckpt, self.logger.run_dir / "best.pt")

        if save_ckpt:
            ckpt = {"model": self.model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epochs - 1}
            torch.save(ckpt, self.logger.run_dir / "last.pt")

        test_metrics = self.evaluate("test")
        self.logger.log_metrics({"epoch": epochs, "split": "test", **test_metrics})

        summary = {
            "best_val_acc": self.state.best_val,
            "test_acc": test_metrics.get("acc@1", 0.0),
            "trainable_params": trainable,
            "total_params": total_params,
            **best_summary,
        }
        self.logger.log_summary(summary)
        return summary
