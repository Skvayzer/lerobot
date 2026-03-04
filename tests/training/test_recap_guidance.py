#!/usr/bin/env python

import pytest
import torch

from lerobot.inference.guidance import apply_cfg_guidance_velocity


def test_apply_cfg_guidance_velocity():
    cond = torch.tensor([[2.0, 4.0]])
    uncond = torch.tensor([[1.0, 1.0]])
    guided = apply_cfg_guidance_velocity(cond, uncond, cfg_scale=2.0)
    assert torch.allclose(guided, torch.tensor([[3.0, 7.0]]))


def test_apply_cfg_guidance_velocity_shape_mismatch():
    with pytest.raises(ValueError):
        apply_cfg_guidance_velocity(torch.zeros(1, 2), torch.zeros(1, 3), cfg_scale=1.0)

