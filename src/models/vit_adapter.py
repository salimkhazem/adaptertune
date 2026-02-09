from typing import List, Optional

import torch
from torch import nn

from .adapters import Adapter


class AdapterBlock(nn.Module):
    def __init__(self, block: nn.Module, adapter: Optional[Adapter]) -> None:
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.adapter is not None:
            x = x + self.adapter(x)
        return x


def attach_adapters(
    model: nn.Module,
    rank: int,
    every: int = 1,
    init: str = "zero",
    alpha: float = 1.0,
) -> nn.Module:
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not have transformer blocks attribute 'blocks'.")
    blocks = []
    for idx, block in enumerate(model.blocks):
        if idx % every == 0:
            adapter = Adapter(model.embed_dim, rank, init=init, alpha=alpha)
        else:
            adapter = None
        blocks.append(AdapterBlock(block, adapter))
    # timm ViT expects self.blocks to be callable (Sequential-like).
    model.blocks = nn.Sequential(*blocks)
    return model


def iter_adapters(model: nn.Module) -> List[Adapter]:
    adapters: List[Adapter] = []
    for module in model.modules():
        if isinstance(module, Adapter):
            adapters.append(module)
    return adapters
