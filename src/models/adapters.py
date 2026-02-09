from typing import Literal

import torch
from torch import nn


class Adapter(nn.Module):
    def __init__(self, dim: int, rank: int, init: Literal["zero", "small"] = "zero", alpha: float = 1.0):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.alpha = alpha
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        self.act = nn.GELU()
        self.reset_parameters(init)

    def reset_parameters(self, init: Literal["zero", "small"] = "zero") -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        if init == "zero":
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        elif init == "small":
            nn.init.normal_(self.up.weight, mean=0.0, std=1e-4)
            nn.init.normal_(self.up.bias, mean=0.0, std=1e-4)
        else:
            raise ValueError(f"Unknown adapter init: {init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.up(self.act(self.down(x)))
