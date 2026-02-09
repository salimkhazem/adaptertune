import torch

from src.models.adapters import Adapter


def test_zero_init_outputs_zero():
    adapter = Adapter(dim=8, rank=4, init="zero")
    x = torch.randn(2, 3, 8)
    y = adapter(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_small_init_not_all_zero():
    adapter = Adapter(dim=8, rank=4, init="small")
    x = torch.randn(2, 3, 8)
    y = adapter(x)
    assert not torch.allclose(y, torch.zeros_like(y))
