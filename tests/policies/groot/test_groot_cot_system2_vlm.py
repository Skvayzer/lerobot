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

import pytest
import draccus

from lerobot.policies.grootCoT.configuration_groot import GrootCoTConfig
from lerobot.policies.grootCoT.processor_groot import (
    GrootQwenCollateStep,
    GrootQwenEncodeStep,
    make_groot_pre_post_processors,
)
from lerobot.policies.grootCoT.system2_vlm_registry import (
    get_system2_vlm_model_id_from_preset,
    resolve_system2_vlm_model_id,
)


def test_registry_maps_instruct_preset():
    model_id = get_system2_vlm_model_id_from_preset("qwen3_vl_8b_instruct")
    assert model_id == "Qwen/Qwen3-VL-8B-Instruct"


def test_registry_rejects_unknown_preset():
    with pytest.raises(ValueError, match="Unsupported system2_vlm_preset"):
        get_system2_vlm_model_id_from_preset("qwen3_vl_8b_unknown")


def test_config_resolves_preset_and_mirrors_legacy_field():
    cfg = GrootCoTConfig(system2_vlm_preset="qwen3_vl_8b_instruct")
    assert cfg.resolved_system2_vlm_model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg.resolved_system2_vlm_source == "preset"
    assert cfg.vlm_processor_model_id == "Qwen/Qwen3-VL-8B-Instruct"


def test_config_explicit_model_id_has_highest_priority():
    cfg = GrootCoTConfig(
        system2_vlm_preset="qwen3_vl_8b_thinking",
        system2_vlm_model_id="Qwen/Qwen3-VL-8B-Instruct",
        vlm_processor_model_id="Qwen/Qwen3-VL-8B-Thinking",
    )
    assert cfg.resolved_system2_vlm_model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg.resolved_system2_vlm_source == "explicit_model_id"
    assert cfg.vlm_processor_model_id == "Qwen/Qwen3-VL-8B-Instruct"


def test_config_keeps_legacy_value_for_old_custom_configs():
    cfg = GrootCoTConfig(
        # Keep preset at default; old checkpoints often only set this legacy key.
        vlm_processor_model_id="Qwen/Qwen2-VL-2B-Instruct",
    )
    assert cfg.resolved_system2_vlm_model_id == "Qwen/Qwen2-VL-2B-Instruct"
    assert cfg.resolved_system2_vlm_source == "legacy_vlm_processor_model_id"
    assert cfg.vlm_processor_model_id == "Qwen/Qwen2-VL-2B-Instruct"


def test_config_rejects_unknown_preset():
    with pytest.raises(ValueError, match="Unsupported system2_vlm_preset"):
        GrootCoTConfig(system2_vlm_preset="invalid")


def test_preprocessor_uses_same_resolved_model_id_for_encode_and_collate():
    cfg = GrootCoTConfig(system2_vlm_preset="qwen3_vl_8b_instruct")
    preprocessor, _ = make_groot_pre_post_processors(cfg, dataset_stats=None)

    encode_steps = [step for step in preprocessor.steps if isinstance(step, GrootQwenEncodeStep)]
    collate_steps = [step for step in preprocessor.steps if isinstance(step, GrootQwenCollateStep)]
    assert len(encode_steps) == 1
    assert len(collate_steps) == 1

    expected_model_id = "Qwen/Qwen3-VL-8B-Instruct"
    assert encode_steps[0].processor_model_id == expected_model_id
    assert collate_steps[0].processor_model_id == expected_model_id


def test_resolver_prefers_preset_then_legacy():
    resolved, source, normalized_preset = resolve_system2_vlm_model_id(
        preset="qwen3_vl_8b_instruct",
        explicit_model_id=None,
        legacy_model_id="Qwen/Qwen2-VL-2B-Instruct",
    )
    assert normalized_preset == "qwen3_vl_8b_instruct"
    assert source == "preset"
    assert resolved == "Qwen/Qwen3-VL-8B-Instruct"


def test_cli_parse_smoke_for_instruct_preset():
    cfg = draccus.parse(
        GrootCoTConfig,
        args=["--system2_vlm_preset=qwen3_vl_8b_instruct"],
    )
    assert cfg.resolved_system2_vlm_model_id == "Qwen/Qwen3-VL-8B-Instruct"
