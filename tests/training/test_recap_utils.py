#!/usr/bin/env python

import torch

from lerobot.training.recap_utils import apply_indicator_dropout, gather_values_by_index


def test_gather_values_by_index():
    index = torch.tensor([5, 8, 13], dtype=torch.long)
    values = gather_values_by_index(
        index,
        index_to_value={5: 1.0, 13: 0.0},
        default_value=-1.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert values.shape == (3, 1)
    assert torch.allclose(values[:, 0], torch.tensor([1.0, -1.0, 0.0]))


def test_apply_indicator_dropout_eval_mode():
    indicator = torch.ones((4, 1))
    out = apply_indicator_dropout(indicator, dropout_p=0.5, null_value=-1.0, training=False)
    assert torch.equal(out, indicator)


def test_apply_indicator_dropout_train_mode():
    torch.manual_seed(0)
    indicator = torch.ones((16, 1))
    out = apply_indicator_dropout(indicator, dropout_p=0.5, null_value=-1.0, training=True)
    num_null = int((out == -1.0).sum().item())
    assert num_null > 0
    assert num_null < 16

