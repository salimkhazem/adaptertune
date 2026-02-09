from typing import Dict

import timm
import torch
from torch import nn

from .vit_adapter import attach_adapters, iter_adapters


def _get_head(model: nn.Module) -> nn.Module:
    if hasattr(model, "head"):
        return model.head
    if hasattr(model, "fc"):
        return model.fc
    raise ValueError("Model does not have a recognized head (head or fc).")


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def _pick_pretrained_variant(backbone: str) -> str | None:
    candidates = timm.list_models(f"{backbone}*", pretrained=True)
    if not candidates:
        return None
    # Prefer ImageNet-1k variants if present.
    prefer = [c for c in candidates if "in1k" in c]
    pool = prefer or candidates
    return sorted(pool)[0]


def create_backbone(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    if backbone not in timm.list_models():
        matches = timm.list_models(f"{backbone}*")
        hint = f" Closest matches: {matches[:5]}" if matches else ""
        raise ValueError(f"Backbone {backbone} not available in timm.{hint}")

    backbone_used = backbone
    if pretrained and backbone not in timm.list_models(pretrained=True):
        alt = _pick_pretrained_variant(backbone)
        if alt is not None:
            backbone_used = alt
            print(
                f"Warning: pretrained weights not registered for {backbone}; using {backbone_used} instead."
            )

    try:
        model = timm.create_model(backbone_used, pretrained=pretrained, num_classes=num_classes)
    except Exception as exc:
        if pretrained:
            print(
                f"Warning: pretrained weights not available for {backbone_used}; falling back to pretrained=False."
            )
            model = timm.create_model(backbone_used, pretrained=False, num_classes=num_classes)
        else:
            raise exc
    # Track the actual backbone used for logging/debugging.
    model._backbone_used = backbone_used
    return model


def prepare_model(cfg_model: Dict, cfg_method: Dict, num_classes: int) -> nn.Module:
    backbone = cfg_model["backbone"]
    pretrained = bool(cfg_model.get("pretrained", True))

    model = create_backbone(backbone, num_classes=num_classes, pretrained=pretrained)
    method = cfg_method["name"]

    if method == "full_finetune":
        _set_trainable(model, True)
        return model

    if method == "head_only":
        _set_trainable(model, False)
        head = _get_head(model)
        _set_trainable(head, True)
        return model

    if method == "adapter_tune":
        rank = int(cfg_method.get("adapter_rank", 16))
        every = int(cfg_method.get("adapter_every", 1))
        init = str(cfg_method.get("adapter_init", "zero"))
        alpha = float(cfg_method.get("adapter_alpha", 1.0))
        attach_adapters(model, rank=rank, every=every, init=init, alpha=alpha)
        _set_trainable(model, False)
        for adapter in iter_adapters(model):
            _set_trainable(adapter, True)
        head = _get_head(model)
        _set_trainable(head, True)
        return model

    raise ValueError(f"Unknown method: {method}")
