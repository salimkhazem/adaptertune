import torch

from src.models.adapters import Adapter
from src.utils.seed import set_seed


def test_seed_reproducibility():
    x = torch.randn(2, 4, 8)
    set_seed(123, deterministic=True)
    a1 = Adapter(dim=8, rank=4, init="small")
    y1 = a1(x)

    set_seed(123, deterministic=True)
    a2 = Adapter(dim=8, rank=4, init="small")
    y2 = a2(x)

    assert torch.allclose(y1, y2, atol=1e-6)
