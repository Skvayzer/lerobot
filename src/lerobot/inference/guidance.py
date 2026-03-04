#!/usr/bin/env python

from __future__ import annotations

import torch


def apply_cfg_guidance_velocity(
    cond_velocity: torch.Tensor,
    uncond_velocity: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    """Classifier-free guidance mixing for diffusion velocity predictions."""
    if cond_velocity.shape != uncond_velocity.shape:
        raise ValueError(
            "cond_velocity and uncond_velocity must have the same shape, "
            f"got {tuple(cond_velocity.shape)} vs {tuple(uncond_velocity.shape)}."
        )
    scale = float(cfg_scale)
    return uncond_velocity + scale * (cond_velocity - uncond_velocity)

