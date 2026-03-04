#!/usr/bin/env python

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ValueModel(nn.Module):
    """Distributional value model backed by the policy's System-2 (Qwen) value head."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if not hasattr(self.policy, "forward_value"):
            raise ValueError(
                "Policy does not expose forward_value(batch). "
                "Use GrootCoTPolicy with recap_value_head_enable=true."
            )
        logits, value_scalar = self.policy.forward_value(batch)
        return logits, value_scalar


def distributional_value_ce_loss(value_logits: Tensor, value_target_bins: Tensor) -> Tensor:
    target = value_target_bins.long().view(-1)
    logits = value_logits.view(target.shape[0], -1)
    return F.cross_entropy(logits, target)

