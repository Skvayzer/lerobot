#!/usr/bin/env python

import numpy as np
import pandas as pd

from lerobot.data.label_experience import (
    bin_returns,
    compute_advantage_nstep,
    compute_indicator_by_task_percentile,
    compute_returns,
    compute_rewards,
    label_experience_dataframe,
)


def test_compute_rewards_sparse_terminal():
    rewards_success = compute_rewards(True, episode_length=4, c_fail=50.0)
    rewards_fail = compute_rewards(False, episode_length=4, c_fail=50.0)
    assert rewards_success.tolist() == [-1.0, -1.0, -1.0, 0.0]
    assert rewards_fail.tolist() == [-1.0, -1.0, -1.0, -50.0]


def test_compute_returns():
    rewards = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
    returns = compute_returns(rewards)
    assert np.allclose(returns, np.array([-2.0, -1.0, 0.0], dtype=np.float32))


def test_bin_returns_bounds():
    returns = np.array([-2.0, -1.0, -0.5, 0.0, 1.0], dtype=np.float32)
    bins = bin_returns(returns, vmin=-1.0, vmax=0.0, bins=5)
    assert bins.tolist() == [0, 0, 2, 4, 4]


def test_advantage_nstep():
    rewards = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
    values = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    advantages = compute_advantage_nstep(rewards, values, n_step=2)
    assert np.allclose(advantages, np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_indicator_percentile_with_corrections():
    advantages = np.array([0.1, 0.5, 0.9, 0.2], dtype=np.float32)
    task_ids = np.array(["a", "a", "b", "b"])
    corrections = np.array([False, False, False, True])
    indicators, thresholds = compute_indicator_by_task_percentile(
        advantages, task_ids, target_pos_rate=0.5, corrections=corrections
    )
    assert set(thresholds.keys()) == {"a", "b"}
    # correction row must always be positive
    assert indicators[-1] == 1


def test_label_experience_dataframe_adds_columns():
    df = pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1],
            "frame_index": [0, 1, 0, 1],
            "task_name": ["pick", "pick", "place", "place"],
            "success": [1, 1, 0, 0],
            "value": [-2.0, -1.0, -2.0, -52.0],
            "is_correction": [0, 0, 0, 1],
        }
    )
    labeled, summary = label_experience_dataframe(
        df,
        c_fail=50.0,
        n_step=2,
        target_pos_rate=0.5,
        vmin=-1.0,
        vmax=0.0,
        bins=11,
    )
    for col in ["reward", "return", "value_target_bin", "advantage", "indicator"]:
        assert col in labeled.columns
    assert summary["num_rows"] == 4

