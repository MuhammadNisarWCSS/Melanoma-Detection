from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary Focal Loss for highly imbalanced classification.

    Focal Loss down-weights easy negatives (confident correct predictions) so
    gradient signal concentrates on hard, misclassified examples — critical for
    the 1.76% positive rate in ISIC 2020.

    Lin et al., 2017: https://arxiv.org/abs/1708.02002

    Args:
        gamma: Focusing parameter. 0 → standard BCE. 2 is the canonical default.
        alpha: Prior probability of the positive class. Balances pos/neg in loss.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}'")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  Raw logits, shape (B,)
            targets: Binary labels in {0, 1}, shape (B,)
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)  # probability of the correct class

        # alpha_t weights positive samples by alpha and negatives by (1-alpha)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal
