from typing import Iterable, List

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Iterable[int] = (1,)) -> List[torch.Tensor]:
    maxk = max(topk)
    with torch.no_grad():
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / target.size(0)))
        return res
