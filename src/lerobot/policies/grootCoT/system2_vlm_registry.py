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

from typing import Final


SYSTEM2_VLM_PRESET_TO_MODEL_ID: Final[dict[str, str]] = {
    "qwen3_vl_8b_thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3_vl_8b_instruct": "Qwen/Qwen3-VL-8B-Instruct",
}

DEFAULT_SYSTEM2_VLM_PRESET: Final[str] = "qwen3_vl_8b_thinking"
DEFAULT_SYSTEM2_VLM_MODEL_ID: Final[str] = SYSTEM2_VLM_PRESET_TO_MODEL_ID[DEFAULT_SYSTEM2_VLM_PRESET]


def normalize_system2_vlm_preset(preset: str | None) -> str:
    value = (preset or "").strip().lower()
    return value or DEFAULT_SYSTEM2_VLM_PRESET


def list_system2_vlm_presets() -> tuple[str, ...]:
    return tuple(sorted(SYSTEM2_VLM_PRESET_TO_MODEL_ID.keys()))


def validate_system2_vlm_preset(preset: str | None) -> str:
    normalized = normalize_system2_vlm_preset(preset)
    if normalized not in SYSTEM2_VLM_PRESET_TO_MODEL_ID:
        raise ValueError(
            f"Unsupported system2_vlm_preset='{preset}'. "
            f"Expected one of {list_system2_vlm_presets()}."
        )
    return normalized


def get_system2_vlm_model_id_from_preset(preset: str | None) -> str:
    normalized = validate_system2_vlm_preset(preset)
    return SYSTEM2_VLM_PRESET_TO_MODEL_ID[normalized]


def _clean_model_id(model_id: str | None) -> str | None:
    if model_id is None:
        return None
    value = str(model_id).strip()
    return value or None


def resolve_system2_vlm_model_id(
    *,
    preset: str | None,
    explicit_model_id: str | None,
    legacy_model_id: str | None,
) -> tuple[str, str, str]:
    """Resolve a canonical System-2 VLM model id.

    Precedence:
    1) explicit model id (system2_vlm_model_id)
    2) preset mapping (system2_vlm_preset)
    3) legacy model id (vlm_processor_model_id)

    Backward compatibility:
    - If preset is the default preset and legacy differs from the default model id,
      prefer legacy so old checkpoint configs continue to use their historical value.
    """

    normalized_preset = validate_system2_vlm_preset(preset)
    explicit = _clean_model_id(explicit_model_id)
    legacy = _clean_model_id(legacy_model_id)

    if explicit is not None:
        return explicit, "explicit_model_id", normalized_preset

    preset_model_id = SYSTEM2_VLM_PRESET_TO_MODEL_ID[normalized_preset]
    default_model_id = SYSTEM2_VLM_PRESET_TO_MODEL_ID[DEFAULT_SYSTEM2_VLM_PRESET]
    if (
        legacy is not None
        and normalized_preset == DEFAULT_SYSTEM2_VLM_PRESET
        and legacy != default_model_id
    ):
        return legacy, "legacy_vlm_processor_model_id", normalized_preset

    if preset_model_id:
        return preset_model_id, "preset", normalized_preset

    if legacy is not None:
        return legacy, "legacy_vlm_processor_model_id", normalized_preset

    return DEFAULT_SYSTEM2_VLM_MODEL_ID, "preset", normalized_preset
