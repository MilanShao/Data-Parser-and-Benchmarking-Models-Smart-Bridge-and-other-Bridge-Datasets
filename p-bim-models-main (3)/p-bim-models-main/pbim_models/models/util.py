from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


Stage = Literal["train", "val", "test"]


class LossWrapper(nn.Module):
    def __init__(self, name: str, reduction: str = "mean", multi_step: bool = False):
        super().__init__()
        self.name = name
        self.reduction = reduction
        self.multi_step = multi_step

    def _base_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.name == "mse":
            return F.mse_loss(x, y, reduction="none")
        elif self.name == "mae":
            return F.l1_loss(x, y, reduction="none")
        elif self.name == "bce":
            return F.binary_cross_entropy(x, y, reduction="none")
        else:
            raise ValueError(f"Unknown loss type '{self.name}'.")

    def _reduction(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        else:
            raise ValueError(f"Unknown reduction with '{self.reduction}'.")

    def forward(self, x: torch.Tensor, y: torch.Tensor, reduce=True) -> torch.Tensor:
        loss = self._base_loss(x, y).sum(dim=-1)
        if self.multi_step:
            loss = loss.sum(-1)
        return self._reduction(loss) if reduce else loss
