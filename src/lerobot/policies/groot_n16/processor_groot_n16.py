#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""
GR00T N1.6 Processor for LeRobot Integration

Adapts the N1.5 processor pipeline for N1.6:
- N1.6 uses pixel_values/input_ids/attention_mask (no eagle_ prefix)
- max_state_dim=29, max_action_dim=29
- Embodiment mapping: unitree_g1 → 8 (from N1.6 EMBODIMENT_TAG_TO_PROJECTOR_INDEX)
- Uses gr00t's Gr00tN1d6DataCollator for VLM collation
"""

from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import ProcessorMixin
else:
    ProcessorMixin = object

from lerobot.configs.types import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
)
from lerobot.policies.groot_n16.configuration_groot_n16 import GrootN16Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

# N1.6 embodiment mapping (from gr00t EMBODIMENT_TAG_TO_PROJECTOR_INDEX)
N16_EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "robocasa_panda_omron": 13,
    "gr1": 20,
    "behavior_r1_pro": 24,
    "unitree_g1": 8,
    "libero_panda": 2,
    "oxe_google": 0,
    "oxe_widowx": 1,
    "oxe_droid": 16,
    "new_embodiment": 10,
}


def _build_n16_eagle_processor(model_name: str = "nvidia/Eagle-Block2A-2B-v2") -> ProcessorMixin:
    """Load Eagle processor from gr00t package (installed at /home/cosmos/Isaac-GR00T)."""
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import build_processor
    proc = build_processor(model_name, transformers_loading_kwargs={"trust_remote_code": True})
    proc.tokenizer.padding_side = "left"
    return proc


def make_groot_n16_pre_post_processors(
    config: GrootN16Config, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create preprocessor and postprocessor for GR00T N1.6 policy.

    Processing pipeline:
    1. Rename observations (dataset-specific key mapping)
    2. Add batch dimension
    3. Infer observation.state from obs keys if missing
    4. Pack video/state/action/language/embodiment (N1.6 dims: 29/29)
    5. Encode video+language → vlm_content using Eagle processor from gr00t
    6. Collate vlm_content → pixel_values/input_ids/attention_mask tensors
    7. Move to device
    """
    state_horizon = 1
    action_horizon = config.chunk_size  # must match checkpoint action_horizon (50 for GR00T-N1.6-3B)
    max_state_dim = config.max_state_dim
    max_action_dim = config.max_action_dim

    padded_stats = dataset_stats or {}

    try:
        env_action_dim = int(config.output_features[ACTION].shape[0])
    except Exception:
        env_action_dim = 0

    # Import from groot N1.5 processor for shared steps
    from lerobot.policies.groot.processor_groot import GrootInferStateFromObsStep

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        GrootInferStateFromObsStep(
            max_state_dim=max_state_dim,
            infer_state_from_obs=getattr(config, "infer_state_from_obs", True),
            state_keys=getattr(config, "state_keys", None),
            state_key_regex=getattr(config, "state_key_regex", None),
            state_key_exclude_regex=getattr(config, "state_key_exclude_regex", None),
        ),
        GrootN16PackInputsStep(
            state_horizon=state_horizon,
            action_horizon=action_horizon,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            language_key="task",
            formalize_language=False,
            embodiment_tag=config.embodiment_tag,
            normalize_min_max=True,
            stats=padded_stats,
        ),
        GrootN16EagleEncodeStep(),
        GrootN16EagleCollateStep(),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        GrootN16ActionUnpackUnnormalizeStep(
            env_action_dim=env_action_dim,
            stats=padded_stats,
            normalize_min_max=True,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def _to_uint8_np_bhwc(img_t: torch.Tensor) -> np.ndarray:
    if img_t.dtype.is_floating_point:
        img_t = (img_t.clamp(0, 1) * 255.0).to(torch.uint8)
    return rearrange(img_t.cpu().numpy(), "b c h w -> b h w c")


@dataclass
@ProcessorStepRegistry.register(name="groot_n16_pack_inputs_v1")
class GrootN16PackInputsStep(ProcessorStep):
    """Pack video/state/action for N1.6.

    Actual nvidia/GR00T-N1.6-3B checkpoint: max_state_dim=128, max_action_dim=128,
    action_horizon=50. State is zero-padded to max_state_dim.
    """

    state_horizon: int = 1
    action_horizon: int = 50
    max_state_dim: int = 128
    max_action_dim: int = 128
    language_key: str = "task"
    formalize_language: bool = False
    embodiment_tag: str = "unitree_g1"
    embodiment_mapping: dict[str, int] = field(
        default_factory=lambda: dict(N16_EMBODIMENT_TAG_TO_PROJECTOR_INDEX)
    )
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        def _align_vec(vec: Any, target_dim: int, *, default: float) -> torch.Tensor:
            t = torch.as_tensor(vec)
            t = t.flatten().to(
                dtype=torch.float32,
                device=next(
                    (v.device for v in obs.values() if isinstance(v, torch.Tensor)), torch.device("cpu")
                ),
            )
            d = int(t.shape[-1]) if t.numel() > 0 else 0
            if d == target_dim:
                return t
            if d < target_dim:
                pad = torch.full((target_dim - d,), default, dtype=t.dtype, device=t.device)
                return torch.cat([t, pad], dim=0)
            return t[:target_dim]

        def _min_max_norm(x: torch.Tensor, key: str) -> torch.Tensor:
            if not self.normalize_min_max:
                return x
            if self.stats is None or key not in self.stats:
                return x
            stats_k = self.stats[key]
            last_dim = x.shape[-1]
            min_v = _align_vec(stats_k.get("min", torch.zeros(last_dim)), last_dim, default=0.0)
            max_v = _align_vec(stats_k.get("max", torch.ones(last_dim)), last_dim, default=1.0)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            mapped = 2 * (x - min_v) / safe_denom - 1
            return torch.where(mask, mapped, torch.zeros_like(mapped))

        # 1) Video -> (B, 1, V, C, H, W) uint8 numpy
        img_keys = sorted([k for k in obs if k.startswith(OBS_IMAGES)])
        if not img_keys and OBS_IMAGE in obs:
            img_keys = [OBS_IMAGE]
        if img_keys:
            cams = [_to_uint8_np_bhwc(obs[k]) for k in img_keys]
            video = np.stack(cams, axis=1)   # (B, V, H, W, C)
            video = np.expand_dims(video, axis=1)  # (B, 1, V, H, W, C)
            video = np.transpose(video, (0, 1, 2, 5, 3, 4))  # (B, 1, V, C, H, W)
            obs["video"] = video
            for k in img_keys:
                obs.pop(k, None)

        # 2) Language
        lang = comp.get(self.language_key)
        if isinstance(lang, list):
            lang = lang[0] if len(lang) > 0 else None
        if not lang:
            lang = "Perform the task."
        if self.formalize_language:
            lang = (lang or "").lower()
            lang = "".join(ch for ch in lang if ch.isalnum() or ch.isspace())
        comp["language"] = lang

        # 3) State/state_mask -> (B, 1, max_state_dim)
        if OBS_STATE in obs:
            state = obs[OBS_STATE]
            if state.dim() != 2:
                raise ValueError(f"state must be (B, D), got {tuple(state.shape)}")
            bsz, d = state.shape
            if self.normalize_min_max:
                state = _min_max_norm(state, OBS_STATE)
            state = state.unsqueeze(1)  # (B, 1, D)
            if d > self.max_state_dim:
                state = state[:, :, : self.max_state_dim]
                d = self.max_state_dim
            elif d < self.max_state_dim:
                pad = torch.zeros(bsz, 1, self.max_state_dim - d, dtype=state.dtype, device=state.device)
                state = torch.cat([state, pad], dim=2)
            state_mask = torch.zeros(bsz, 1, self.max_state_dim, dtype=torch.bool, device=state.device)
            state_mask[:, :, :d] = True
            obs["state"] = state
            obs["state_mask"] = state_mask

        # 4) Action/action_mask -> (B, action_horizon, max_action_dim)
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            if self.normalize_min_max:
                if action.dim() == 2:
                    action = _min_max_norm(action, ACTION)
                elif action.dim() == 3:
                    b, t, d = action.shape
                    flat = action.reshape(b * t, d)
                    flat = _min_max_norm(flat, ACTION)
                    action = flat.view(b, t, d)
            if action.dim() == 2:
                action = action.unsqueeze(1).repeat(1, self.action_horizon, 1)
            elif action.dim() == 3:
                b, t, d = action.shape
                if t < self.action_horizon:
                    last = action[:, -1:, :]
                    pad = last.repeat(1, self.action_horizon - t, 1)
                    action = torch.cat([action, pad], dim=1)
                elif t > self.action_horizon:
                    action = action[:, : self.action_horizon, :]
            else:
                raise ValueError(f"action must be (B, D) or (B, T, D), got {tuple(action.shape)}")

            b, t, d = action.shape
            if d > self.max_action_dim:
                action = action[:, :, : self.max_action_dim]
                d = self.max_action_dim
            elif d < self.max_action_dim:
                pad = torch.zeros(b, t, self.max_action_dim - d, dtype=action.dtype, device=action.device)
                action = torch.cat([action, pad], dim=2)
            action_mask = torch.zeros(b, t, self.max_action_dim, dtype=torch.bool, device=action.device)
            action_mask[:, :, :d] = True
            transition[TransitionKey.ACTION] = action
            comp["action_mask"] = action_mask

        # 5) Embodiment id
        emb_id = self.embodiment_mapping.get(self.embodiment_tag, 0)
        bsz = None
        device = torch.device("cpu")
        for v in list(obs.values()) + [transition.get(TransitionKey.ACTION)]:
            if isinstance(v, torch.Tensor):
                bsz = v.shape[0]
                device = v.device
                break
        if bsz is None and "video" in obs and isinstance(obs["video"], np.ndarray):
            bsz = obs["video"].shape[0]
        if bsz is None:
            bsz = 1
        comp["embodiment_id"] = torch.full((bsz,), emb_id, dtype=torch.long, device=device)

        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "state_horizon": self.state_horizon,
            "action_horizon": self.action_horizon,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "language_key": self.language_key,
            "formalize_language": self.formalize_language,
            "embodiment_tag": self.embodiment_tag,
            "embodiment_mapping": self.embodiment_mapping,
            "normalize_min_max": self.normalize_min_max,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        if not self.stats:
            return {}
        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if not state:
            return
        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor
        if reconstructed:
            self.stats = reconstructed


@dataclass
@ProcessorStepRegistry.register(name="groot_n16_eagle_encode_v1")
class GrootN16EagleEncodeStep(ProcessorStep):
    """Encode video+language into N1.6 VLM content format.

    Produces 'vlm_content' entries per batch item, which the collate step
    converts to pixel_values/input_ids/attention_mask tensors.

    image_max_pixels: caps image resolution fed to Eagle processor. Training
    images may be large (e.g. 480x640 = 307K pixels). Siglip2 concatenates all
    patches from every image in the batch into one sequence before attention, so
    oversized images → OOM. Default 224*224=50176 keeps patches/GPU manageable.
    """

    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    image_max_pixels: int = 224 * 224  # ~50K pixels; limits patches per image
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_n16_eagle_processor(self.model_name)
        return self._proc

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        if "video" not in obs:
            return transition

        video = obs["video"]  # (B, T, V, C, H, W) uint8 numpy
        lang = comp.get("language", "Perform the task.")
        if isinstance(lang, list):
            lang = lang[0] if len(lang) > 0 else "Perform the task."

        bsz = video.shape[0]
        vlm_contents: list[dict[str, Any]] = []

        for b in range(bsz):
            vt = video[b]  # (T, V, C, H, W)
            if vt.ndim == 5:
                t, v, c, h, w = vt.shape
                flat = rearrange(vt, "t v c h w -> (t v) h w c")
            else:
                # Fallback: assume (T, V, H, W, C)
                t, v, h, w, c = vt.shape
                flat = rearrange(vt, "t v h w c -> (t v) h w c")

            pil_images = [Image.fromarray(flat[i]) for i in range(t * v)]

            # Build conversation in the format N1.6 processor expects.
            # max_pixels limits image resolution so Eagle's smart_resize keeps
            # patch count small enough to avoid Siglip2 full-attention OOM
            # (Siglip2 concatenates ALL batch images into one sequence).
            lang_text = str([lang])  # Match original GR00T format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": lang_text},
                        *[
                            {"type": "image", "image": img, "max_pixels": self.image_max_pixels}
                            for img in pil_images
                        ],
                    ],
                }
            ]

            text = self.proc.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )

            vlm_contents.append({
                "vlm_content": {
                    "text": text,
                    "images": pil_images,
                    "conversation": conversation,
                }
            })

        comp["n16_vlm_content"] = vlm_contents
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_n16_eagle_collate_v1")
class GrootN16EagleCollateStep(ProcessorStep):
    """Collate N1.6 VLM content into pixel_values/input_ids/attention_mask tensors.

    Uses gr00t's Gr00tN1d6DataCollator to produce the correct tensor format
    for the N1.6 backbone (no eagle_ prefix on keys).
    """

    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    _collator: Any | None = field(default=None, init=False, repr=False)

    @property
    def collator(self):
        if self._collator is None:
            from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6DataCollator
            self._collator = Gr00tN1d6DataCollator(
                model_name=self.model_name,
                model_type="eagle",
                transformers_loading_kwargs={"trust_remote_code": True},
            )
        return self._collator

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        vlm_contents = comp.get("n16_vlm_content")
        if not vlm_contents:
            return transition

        # Use Gr00tN1d6DataCollator: input is list of dicts with 'vlm_content'
        # Output is BatchFeature with 'inputs' key containing pixel_values/input_ids/attention_mask
        batched = self.collator(vlm_contents)["inputs"]

        # Inject VLM tensors into comp (no eagle_ prefix, as N1.6 expects)
        for k, v in batched.items():
            if k in ("pixel_values", "input_ids", "attention_mask", "image_grid_thw"):
                comp[k] = v

        comp.pop("n16_vlm_content", None)
        obs.pop("video", None)

        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_n16_action_unpack_v1")
class GrootN16ActionUnpackUnnormalizeStep(ProcessorStep):
    """Slice and unnormalize N1.6 action predictions."""

    env_action_dim: int = 0
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition

        if action.dim() == 3:
            action = action[:, -1, :]  # Take last timestep
        if self.env_action_dim and action.shape[-1] >= self.env_action_dim:
            action = action[..., : self.env_action_dim]

        if self.normalize_min_max and self.stats is not None:
            stats_k = self.stats.get(ACTION, {})
            d = action.shape[-1]
            min_v = torch.as_tensor(
                stats_k.get("min", torch.zeros(d)), dtype=action.dtype, device=action.device
            )
            max_v = torch.as_tensor(
                stats_k.get("max", torch.ones(d)), dtype=action.dtype, device=action.device
            )
            if min_v.numel() != d:
                min_v = torch.nn.functional.pad(min_v.flatten()[:d], (0, max(0, d - min_v.numel())))
                min_v = min_v.to(action.device, dtype=action.dtype)
            if max_v.numel() != d:
                max_v = torch.nn.functional.pad(max_v.flatten()[:d], (0, max(0, d - max_v.numel())))
                max_v = max_v.to(action.device, dtype=action.dtype)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            inv = (action + 1.0) * 0.5 * safe_denom + min_v
            action = torch.where(mask, inv, min_v)

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "env_action_dim": self.env_action_dim,
            "normalize_min_max": self.normalize_min_max,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        if not self.stats:
            return {}
        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if not state:
            return
        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor
        if reconstructed:
            self.stats = reconstructed
