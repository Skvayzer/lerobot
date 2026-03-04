#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import types
from types import SimpleNamespace

import torch

from lerobot.policies.grootCoT.modeling_groot import GrootCoTPolicy


def _make_policy(*, async_enable: bool) -> GrootCoTPolicy:
    policy = GrootCoTPolicy.__new__(GrootCoTPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        n_action_steps=4,
        output_features={"action": SimpleNamespace(shape=(4,))},
        recap_enable=False,
        dual_rate_enable=True,
        dual_rate_apply_in_train=False,
        system2_hz=20.0,
        system2_async_enable=async_enable,
        system2_async_prefetch_chunks=1,
        system2_async_startup_warmup=False,
        system2_async_max_observation_age_s=0.5,
        system2_async_log_every_n_steps=10,
        system2_async_wall_clock=True,
        system1_min_queue_size=0,
        system1_replan_every_n_steps=0,
        dual_rate_force_backbone_refresh_on_reset=True,
        use_bf16=False,
        recap_adv_indicator_use_cfg=False,
    )
    # Ensure fallback path has a known device if state is absent.
    policy.register_parameter("_dummy_param", torch.nn.Parameter(torch.zeros(1)))
    policy.reset()
    return policy


def test_async_select_action_never_blocks_when_worker_is_slow():
    policy = _make_policy(async_enable=True)

    def _slow_compute(self, **kwargs):
        time.sleep(0.2)
        return torch.ones((1, 2, 4), dtype=torch.float32)

    policy._compute_action_chunk_sync = types.MethodType(_slow_compute, policy)
    try:
        start = time.perf_counter()
        action = policy.select_action({"state": torch.zeros((1, 4), dtype=torch.float32)})
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1
        assert action.shape == (1, 4)
        # First step should use state-based hold fallback while worker computes.
        assert torch.allclose(action, torch.zeros((1, 4), dtype=torch.float32))
    finally:
        policy._shutdown_async_worker()


def test_async_transitions_from_fallback_to_ready_chunk():
    policy = _make_policy(async_enable=True)

    def _fast_compute(self, **kwargs):
        return torch.full((1, 2, 4), 7.0, dtype=torch.float32)

    policy._compute_action_chunk_sync = types.MethodType(_fast_compute, policy)
    try:
        first = policy.select_action({"state": torch.zeros((1, 4), dtype=torch.float32)})
        assert torch.allclose(first, torch.zeros((1, 4), dtype=torch.float32))

        got_chunk = False
        for _ in range(20):
            time.sleep(0.01)
            nxt = policy.select_action({"state": torch.zeros((1, 4), dtype=torch.float32)})
            if torch.allclose(nxt, torch.full((1, 4), 7.0, dtype=torch.float32)):
                got_chunk = True
                break
        assert got_chunk
    finally:
        policy._shutdown_async_worker()


def test_reset_stops_worker_and_clears_queue():
    policy = _make_policy(async_enable=True)

    def _compute(self, **kwargs):
        return torch.full((1, 2, 4), 3.0, dtype=torch.float32)

    policy._compute_action_chunk_sync = types.MethodType(_compute, policy)
    policy.select_action({"state": torch.zeros((1, 4), dtype=torch.float32)})
    time.sleep(0.02)

    worker = policy._async_thread
    assert worker is not None and worker.is_alive()

    policy.reset()
    assert policy._async_thread is None
    assert len(policy._action_queue) == 0


def test_sync_mode_behavior_is_unchanged():
    policy = _make_policy(async_enable=False)
    calls = {"n": 0}

    def _compute(self, **kwargs):
        calls["n"] += 1
        return torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
            dtype=torch.float32,
        )

    policy._compute_action_chunk_sync = types.MethodType(_compute, policy)
    try:
        a1 = policy.select_action({"state": torch.zeros((1, 4), dtype=torch.float32)})
        a2 = policy.select_action({"state": torch.zeros((1, 4), dtype=torch.float32)})
        assert calls["n"] == 1
        assert torch.allclose(a1, torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32))
        assert torch.allclose(a2, torch.tensor([[5.0, 6.0, 7.0, 8.0]], dtype=torch.float32))
    finally:
        policy._shutdown_async_worker()


def test_async_runtime_stats_are_exposed():
    policy = _make_policy(async_enable=True)

    def _compute(self, **kwargs):
        return torch.full((1, 2, 4), 2.0, dtype=torch.float32)

    policy._compute_action_chunk_sync = types.MethodType(_compute, policy)
    try:
        for _ in range(3):
            policy.select_action({"state": torch.zeros((1, 4), dtype=torch.float32)})
            time.sleep(0.01)

        stats = policy.get_dual_rate_runtime_stats()
        expected_keys = {
            "s1_inference_hz_est",
            "s1_fallback_count",
            "s1_queue_len",
            "s2_worker_alive",
            "s2_worker_busy",
            "s2_chunks_ready",
            "s2_last_refresh_age_ms",
            "s2_refresh_count",
        }
        assert expected_keys.issubset(set(stats.keys()))
    finally:
        policy._shutdown_async_worker()
