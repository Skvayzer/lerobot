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

from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, ProcessorMixin
else:
    AutoProcessor = None
    ProcessorMixin = object

from lerobot.configs.types import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
)
from lerobot.policies.grootCoT.configuration_groot import GrootCoTConfig
from lerobot.policies.grootCoT.system2_vlm_registry import DEFAULT_SYSTEM2_VLM_MODEL_ID
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
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

# Defaults for VLM processor locations
DEFAULT_QWEN_PROCESSOR_MODEL_ID = DEFAULT_SYSTEM2_VLM_MODEL_ID
SUMMARY_TOKEN = "<SUMMARY>"
SUMMARY_PROMPT = "Think briefly (<=20 tokens), then summarize. " + SUMMARY_TOKEN
QWEN_CONTENT_KEY = "qwen_content"
LEGACY_EAGLE_CONTENT_KEY = "eagle_content"
QWEN_INPUT_PREFIX = "qwen_"
LEGACY_EAGLE_INPUT_PREFIX = "eagle_"
DEFAULT_DEX3_CANONICAL_CAMERA_ORDER = [
    "observation.images.cam_left_high",
    "observation.images.cam_right_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
DEFAULT_DEX3_CAMERA_ALIASES = {
    "observation.images.cam_left_high": ["observation.images.left_high", "observation.images.left_high_rgb"],
    "observation.images.cam_right_high": ["observation.images.right_high", "observation.images.right_high_rgb"],
    "observation.images.cam_left_wrist": ["observation.images.left_wrist", "observation.images.left_wrist_rgb"],
    "observation.images.cam_right_wrist": ["observation.images.right_wrist", "observation.images.right_wrist_rgb"],
}


def make_groot_pre_post_processors(
    config: GrootCoTConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create preprocessor and postprocessor for Groot policy.

    This creates a processing pipeline that transforms LeRobot data format into
    the format expected by Isaac-GR00T models:

    Preprocessing steps:
    1. Optional key renaming (dataset-specific key mapping)
    2. Add batch dimension to unbatched data
    3. Pack video/state/action/language/embodiment and apply optional min-max normalization before padding
    4. Encode video+language with Qwen VLM into intermediate qwen_content
    5. Collate qwen_content into batched qwen_* tensors
    6. Move tensors to device (GPU)

    NOTE: We optionally apply min-max normalization to STATE and ACTION using
    dataset-provided statistics prior to padding, mapping values to [-1, 1].
    This mirrors SO100-style preprocessing and keeps scales consistent with GR00T.

    Args:
        config: Groot configuration containing data_config, embodiment_tag, etc.
        dataset_stats: Optional per-key min/max statistics for normalization before padding.

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """

    # Get horizon/dimension parameters from config
    # These should match the config used for the pretrained model
    # Default values match most GR00T configs (state_horizon=1, action_horizon=16)
    state_horizon = 1
    # CRITICAL: Pretrained GR00T models use action_horizon=16 max!
    # The model architecture hardcodes this limit
    action_horizon = min(config.chunk_size, 16)
    max_state_dim = config.max_state_dim
    max_action_dim = config.max_action_dim

    # Pass raw dataset_stats; normalization will occur inside pack step before padding
    padded_stats = dataset_stats or {}

    # Define feature specs for optional normalization steps
    _features: dict[str, PolicyFeature] = {
        # Observation features (only add those we may normalize)
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_horizon, max_state_dim)),
        # Action feature
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_horizon, max_action_dim)),
    }

    # Normalize STATE and ACTION with min_max (SO100-like default)
    _norm_map = {
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
        FeatureType.STATE: NormalizationMode.MIN_MAX,
    }

    # Determine env action dimension from config (simple, object-like PolicyFeature)
    try:
        env_action_dim = int(config.output_features["action"].shape[0])
    except Exception:
        env_action_dim = 0

    input_steps: list[ProcessorStep] = [
        # 1. Rename keys if needed (e.g., dataset-specific camera names)
        # Leave empty for now - add mappings if your dataset uses different key names
        RenameObservationsProcessorStep(rename_map={}),
        # 2. Add batch dimension for single samples
        AddBatchDimensionProcessorStep(),
        # 2.5 Infer observation.state from other observation keys if missing
        GrootInferStateFromObsStep(
            max_state_dim=max_state_dim,
            infer_state_from_obs=getattr(config, "infer_state_from_obs", True),
            state_keys=getattr(config, "state_keys", None),
            state_key_regex=getattr(config, "state_key_regex", None),
            state_key_exclude_regex=getattr(config, "state_key_exclude_regex", None),
        ),
        # 3. Pack video/state/action/language/embodiment; apply optional min-max normalization before padding
        GrootPackInputsStep(
            state_horizon=state_horizon,
            action_horizon=action_horizon,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            language_key="task",
            formalize_language=False,
            embodiment_tag=config.embodiment_tag,
            normalize_min_max=True,
            stats=padded_stats,
            enforce_dex3_canonical_camera_order=getattr(
                config, "enforce_dex3_canonical_camera_order", True
            ),
            dex3_canonical_camera_order=list(
                getattr(config, "dex3_canonical_camera_order", DEFAULT_DEX3_CANONICAL_CAMERA_ORDER)
            ),
            dex3_missing_camera_policy=getattr(config, "dex3_missing_camera_policy", "zero_fill"),
            use_grounded_reference_frame=getattr(config, "use_grounded_reference_frame", False),
        ),
        # 4. Qwen encode (creates qwen_content)
        GrootQwenEncodeStep(
            processor_model_id=config.vlm_processor_model_id,
        ),
        # 5. Collate qwen_content -> qwen_* tensors
        GrootQwenCollateStep(
            processor_model_id=config.vlm_processor_model_id,
        ),
        # 6. Move to device
        DeviceProcessorStep(device=config.device),
    ]

    # Postprocessing: slice to env action dim and unnormalize to env scale, then move to CPU
    output_steps: list[ProcessorStep] = [
        GrootActionUnpackUnnormalizeStep(
            env_action_dim=env_action_dim,
            stats=padded_stats,
            normalize_min_max=True,
        ),
        # Finally, move to CPU for env interaction
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


# GR00T specific processor steps


def _to_uint8_np_bhwc(img_t: torch.Tensor) -> np.ndarray:
    # img_t: (B, C, H, W) float in [0,1] or uint8
    if img_t.dtype.is_floating_point:
        img_t = (img_t.clamp(0, 1) * 255.0).to(torch.uint8)
    return rearrange(img_t.cpu().numpy(), "b c h w -> b h w c")


def _build_qwen_processor(processor_model_id: str = DEFAULT_QWEN_PROCESSOR_MODEL_ID) -> ProcessorMixin:
    proc = AutoProcessor.from_pretrained(processor_model_id, trust_remote_code=True)
    # Align padding with autoregressive usage
    if hasattr(proc, "tokenizer") and proc.tokenizer:
        proc.tokenizer.padding_side = "left"
    return proc


@dataclass
@ProcessorStepRegistry.register(name="groot_cot_infer_state_v1")
class GrootInferStateFromObsStep(ProcessorStep):
    """Create observation.state from other observation keys when missing.

    Heuristic: concatenate 1D observation tensors (excluding images) in a stable key order.
    Use explicit state_keys or regex selection when provided.
    """

    max_state_dim: int = 64
    infer_state_from_obs: bool = True
    state_keys: list[str] | None = None
    state_key_regex: str | None = None
    state_key_exclude_regex: str | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.infer_state_from_obs:
            return transition

        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        if "observation.state" in obs:
            return transition

        # Build candidate key list
        if self.state_keys:
            keys = [k for k in self.state_keys if k in obs]
        else:
            keys = [k for k in obs.keys() if k.startswith("observation.")]
            if self.state_key_regex:
                pattern = re.compile(self.state_key_regex)
                keys = [k for k in keys if pattern.search(k)]
            if self.state_key_exclude_regex:
                exclude = re.compile(self.state_key_exclude_regex)
                keys = [k for k in keys if not exclude.search(k)]

        # Default heuristic: exclude image-like keys
        keys = [k for k in keys if not k.startswith("observation.images.") and k != "observation.image"]

        # Deterministic ordering
        keys = sorted(keys)

        parts: list[torch.Tensor] = []
        for k in keys:
            v = obs.get(k)
            if v is None:
                continue
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if not isinstance(v, torch.Tensor):
                continue

            # Expect (B, D). Skip non-vector observations (e.g., images, depth, lidar).
            if v.dim() == 1:
                v = v.unsqueeze(1)
            if v.dim() != 2:
                continue

            parts.append(v.to(dtype=torch.float32))

        if not parts:
            return transition

        state = torch.cat(parts, dim=1)
        if state.shape[1] > self.max_state_dim:
            state = state[:, : self.max_state_dim]

        obs["observation.state"] = state
        return transition

    # Pipeline API requirement: declare how features change.
    # This step infers runtime values only; it does not change static feature specs.
    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_cot_pack_inputs_v3")
class GrootPackInputsStep(ProcessorStep):
    state_horizon: int = 1
    action_horizon: int = 16
    max_state_dim: int = 64
    max_action_dim: int = 32
    language_key: str = "task"
    formalize_language: bool = False
    embodiment_tag: str = "new_embodiment"
    embodiment_mapping: dict[str, int] = field(
        default_factory=lambda: {
            "new_embodiment": 31,  # Match original GR00T EMBODIMENT_TAG_MAPPING
            "oxe_droid": 17,
            "agibot_genie1": 26,
            "gr1": 24,
            "so100": 2,
            "unitree_g1": 3,
        }
    )
    # Min-max normalization (SO100-like) applied BEFORE padding
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None
    # Dex3 canonical multi-view packing support.
    enforce_dex3_canonical_camera_order: bool = True
    dex3_canonical_camera_order: list[str] = field(
        default_factory=lambda: list(DEFAULT_DEX3_CANONICAL_CAMERA_ORDER)
    )
    dex3_camera_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_DEX3_CAMERA_ALIASES.items()}
    )
    dex3_missing_camera_policy: str = "zero_fill"
    use_grounded_reference_frame: bool = False

    def _resolve_dex3_camera_key(self, obs: dict[str, Any], canonical_key: str) -> str | None:
        candidates = [canonical_key, *(self.dex3_camera_aliases.get(canonical_key, []))]
        for key in candidates:
            if key in obs:
                return key
        return None

    def _use_dex3_canonical_order(self, obs: dict[str, Any]) -> bool:
        if not self.enforce_dex3_canonical_camera_order:
            return False
        return any(
            self._resolve_dex3_camera_key(obs, canonical_key) is not None
            for canonical_key in self.dex3_canonical_camera_order
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.dex3_missing_camera_policy not in {"zero_fill", "error"}:
            raise ValueError(
                "dex3_missing_camera_policy must be one of {'zero_fill', 'error'}."
            )

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

        # 1) Video (B, T=1, V, H, W, C) uint8
        available_img_keys = sorted([k for k in obs if k.startswith("observation.images.")])
        img_tensors: list[torch.Tensor] = []
        if self._use_dex3_canonical_order(obs):
            ref_img = next(
                (obs[k] for k in available_img_keys if isinstance(obs.get(k), torch.Tensor)),
                None,
            )
            for canonical_key in self.dex3_canonical_camera_order:
                resolved_key = self._resolve_dex3_camera_key(obs, canonical_key)
                if resolved_key is None:
                    if self.dex3_missing_camera_policy == "error":
                        raise ValueError(
                            "Missing canonical Dex3 camera input while strict policy is enabled: "
                            f"{canonical_key}. Available={available_img_keys}"
                        )
                    if ref_img is None:
                        raise ValueError(
                            "Cannot zero-fill missing canonical Dex3 camera because no reference "
                            f"image tensor is available. Missing={canonical_key}"
                        )
                    img_tensors.append(torch.zeros_like(ref_img))
                else:
                    img_tensors.append(obs[resolved_key])
        else:
            fallback_keys = available_img_keys
            if not fallback_keys and "observation.image" in obs:
                fallback_keys = ["observation.image"]
            img_tensors = [obs[k] for k in fallback_keys]

        # Optionally inject a grounded reference frame with bounding box overlay.
        # target_bbox can be a single bbox (applied to all batch elements) or a list
        # of bboxes (one per batch element) for batch-varying targets during training.
        if self.use_grounded_reference_frame and img_tensors:
            bbox_data = comp.get("target_bbox")
            if bbox_data is not None:
                from lerobot.policies.grootCoT.grounding_utils import render_grounded_frame
                ref_base = img_tensors[0]
                ref_np = ref_base.cpu().numpy() if isinstance(ref_base, torch.Tensor) else ref_base
                # Ensure BHWC uint8 for rendering
                if ref_np.ndim == 4 and ref_np.shape[1] <= 4:  # BCHW
                    ref_np = np.transpose(ref_np, (0, 2, 3, 1))
                if ref_np.dtype != np.uint8:
                    ref_np = (ref_np * 255).clip(0, 255).astype(np.uint8)
                bsz = ref_np.shape[0]
                # Normalize bbox_data to a list of per-element bboxes
                if isinstance(bbox_data, (list, tuple)) and len(bbox_data) > 0 and isinstance(bbox_data[0], (list, tuple, np.ndarray)):
                    # List of bboxes, one per batch element
                    bboxes = bbox_data
                else:
                    # Single bbox applied to all batch elements
                    bboxes = [bbox_data] * bsz
                # Render bbox on each batch element with its own bbox
                grounded_batch = np.stack(
                    [render_grounded_frame(ref_np[i], bboxes[i]) for i in range(bsz)],
                    axis=0,
                )  # (B, H, W, C)
                # Convert back to BCHW tensor to match img_tensors format
                grounded_tensor = torch.from_numpy(
                    np.transpose(grounded_batch, (0, 3, 1, 2))
                ).to(img_tensors[0].device)
                img_tensors.append(grounded_tensor)

        if img_tensors:
            cams = [_to_uint8_np_bhwc(img) for img in img_tensors]
            video = np.stack(cams, axis=1)  # (B, V, H, W, C)
            video = np.expand_dims(video, axis=1)  # (B, 1, V, H, W, C)
            # GR00T validates that video.shape[3] == 3 (channels), so reorder to (B, T, V, C, H, W)
            video = np.transpose(video, (0, 1, 2, 5, 3, 4))  # (B, 1, V, C, H, W)
            obs["video"] = video
            # Drop raw images to avoid confusion downstream
            for k in available_img_keys:
                obs.pop(k, None)
            obs.pop("observation.image", None)

        # 2) Language (string)
        lang = comp.get(self.language_key)
        if isinstance(lang, list):
            lang = lang[0] if len(lang) > 0 else None
        if not lang:
            lang = "Perform the task."
        # Append summary prompt/sentinel for downstream summary-state pooling
        lang = f"{lang}\n{SUMMARY_PROMPT}"
        if self.formalize_language:
            lang = (lang or "").lower()
            lang = "".join(ch for ch in lang if ch.isalnum() or ch.isspace())
        comp["language"] = lang

        # 3) State/state_mask -> (B, 1, max_state_dim)
        if "observation.state" in obs:
            state = obs["observation.state"]  # (B, D)
            if state.dim() != 2:
                raise ValueError(f"state must be (B, D), got {tuple(state.shape)}")
            bsz, d = state.shape
            # Normalize BEFORE padding
            if self.normalize_min_max:
                state = _min_max_norm(state, "observation.state")
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
            # Normalize BEFORE temporal expansion/padding
            if self.normalize_min_max:
                if action.dim() == 2:
                    action = _min_max_norm(action, "action")
                elif action.dim() == 3:
                    b, t, d = action.shape
                    flat = action.reshape(b * t, d)
                    flat = _min_max_norm(flat, "action")
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

        # 5) Embodiment id as LongTensor (B,)
        emb_id = self.embodiment_mapping.get(self.embodiment_tag, 0)
        # Infer batch size/device from any tensor in obs or action
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

    # Pipeline API requirement: declare how features change (we keep it simple)
    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        Excludes 'stats' since they are saved separately via state_dict().
        """
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
            "enforce_dex3_canonical_camera_order": self.enforce_dex3_canonical_camera_order,
            "dex3_canonical_camera_order": self.dex3_canonical_camera_order,
            "dex3_camera_aliases": self.dex3_camera_aliases,
            "dex3_missing_camera_policy": self.dex3_missing_camera_policy,
            "use_grounded_reference_frame": self.use_grounded_reference_frame,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns normalization statistics as a flat state dictionary.

        This enables saving stats to safetensors files, similar to normalizer_processor.
        """
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        Loads normalization statistics from a flat state dictionary.

        This enables loading stats from safetensors files during from_pretrained.
        """
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
@ProcessorStepRegistry.register(name="groot_cot_qwen_encode_v1")
@ProcessorStepRegistry.register(name="groot_cot_eagle_encode_v3")
class GrootQwenEncodeStep(ProcessorStep):
    processor_model_id: str = DEFAULT_QWEN_PROCESSOR_MODEL_ID
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_qwen_processor(self.processor_model_id)
        return self._proc

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        if "video" not in obs:
            return transition

        video = obs["video"]  # (B, T, V, H, W, C) uint8
        lang = comp.get("language", "Perform the task.")
        if isinstance(lang, list):
            lang = lang[0] if len(lang) > 0 else "Perform the task."

        bsz = video.shape[0]
        qwen_contents: list[dict[str, Any]] = []
        for b in range(bsz):
            vt = video[b]  # (T, V, C, H, W) after reorder
            if vt.ndim != 5:
                # Fallback: assume (T, V, H, W, C)
                t, v, h, w, c = vt.shape
                flat = rearrange(vt, "t v h w c -> (t v) h w c")
            else:
                t, v, c, h, w = vt.shape
                flat = rearrange(vt, "t v c h w -> (t v) h w c")
            images = [Image.fromarray(flat[i]) for i in range(t * v)]
            # Format language as string list representation to match Original GROOT
            lang_formatted = str([lang])
            text_content = [{"type": "text", "text": lang_formatted}]
            image_content = [{"type": "image", "image": img} for img in images]
            conv = [{"role": "user", "content": image_content + text_content}]
            prompt = self.proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            qwen_contents.append(
                {
                    "text": prompt,
                    "images": images,
                }
            )

        comp[QWEN_CONTENT_KEY] = qwen_contents
        comp.pop(LEGACY_EAGLE_CONTENT_KEY, None)
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    # Pipeline API requirement: declare how features change (no schema change here)
    def transform_features(self, features):
        return features


def collate_multimodal(
    features: list[dict[str, Any]],
    processor: ProcessorMixin,
    *,
    content_key: str,
    output_prefix: str,
) -> dict[str, Any]:
    """Collate multimodal chat content into prefixed tensor keys.

    This is used by Qwen-backed pipelines (qwen_*) and kept flexible so legacy
    Eagle-prefixed checkpoints can still be loaded.
    """
    batch: dict[str, Any] = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == content_key:
            text_list: list[str] = []
            image_inputs: list[Any] = []
            num_images_per_sample: list[int] = []
            for v in values:
                text_list.append(v["text"])
                image_inputs.append(v["images"])
                num_images_per_sample.append(len(v["images"]))
            model_inputs = processor(text=text_list, images=image_inputs, return_tensors="pt", padding=True)
            total_imgs = sum(num_images_per_sample)
            # Reshape flattened vision outputs when processor returns (sum_imgs*tokens, dim) and grid (B, num_imgs, 3)
            if "pixel_values" in model_inputs and "image_grid_thw" in model_inputs:
                pv = model_inputs["pixel_values"]
                grid = model_inputs["image_grid_thw"]
                # If pixel_values is 2D (flattened tokens), try to reconstruct (B, num_imgs, tokens, dim)
                if pv.dim() == 2 and grid.dim() == 3:
                    tokens_per_img = (grid[..., 1] * grid[..., 2]).view(-1).tolist()
                    if sum(tokens_per_img) == pv.shape[0]:
                        splits = torch.split(pv, tokens_per_img, dim=0)
                        regrouped = []
                        idx = 0
                        for n_img in num_images_per_sample:
                            regrouped.append(torch.stack(splits[idx : idx + n_img], dim=0))
                            idx += n_img
                        model_inputs["pixel_values"] = torch.stack(regrouped, dim=0)  # (B, num_imgs, tokens, dim)
                    else:
                        print(
                            f"[GROOT][WARN] Unexpected pixel_values shape {tuple(pv.shape)} for grids {tuple(grid.shape)}; leaving as-is."
                        )
                # Ensure grid is shaped (B, num_imgs, 3)
                if grid.shape[0] != len(num_images_per_sample) or grid.dim() != 3:
                    if grid.shape[0] == total_imgs:
                        splits = torch.split(grid, num_images_per_sample, dim=0)
                        model_inputs["image_grid_thw"] = torch.stack(splits, dim=0)
                    else:
                        print(f"[GROOT][WARN] Unexpected image_grid_thw shape {tuple(grid.shape)}; leaving as-is.")
            for k, v in model_inputs.items():
                k = output_prefix + k
                batch[k] = v
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


# Backward-compatible helper kept for external imports/tests.
def collate(features: list[dict[str, Any]], eagle_processor: ProcessorMixin) -> dict[str, Any]:
    return collate_multimodal(
        features,
        eagle_processor,
        content_key=LEGACY_EAGLE_CONTENT_KEY,
        output_prefix=LEGACY_EAGLE_INPUT_PREFIX,
    )


@dataclass
@ProcessorStepRegistry.register(name="groot_cot_qwen_collate_v1")
@ProcessorStepRegistry.register(name="groot_cot_eagle_collate_v3")
class GrootQwenCollateStep(ProcessorStep):
    processor_model_id: str = DEFAULT_QWEN_PROCESSOR_MODEL_ID
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_qwen_processor(self.processor_model_id)
        return self._proc

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        content_key = QWEN_CONTENT_KEY
        contents = comp.get(content_key)
        if not contents:
            # Backward compatibility for older checkpoints
            content_key = LEGACY_EAGLE_CONTENT_KEY
            contents = comp.get(content_key)
        if not contents:
            return transition

        # Build features list as original API expects: one dict per batch item
        features = [{content_key: content} for content in contents]
        batched = collate_multimodal(
            features,
            self.proc,
            content_key=content_key,
            output_prefix=QWEN_INPUT_PREFIX,
        )

        # Compute summary position (last non-pad token) for each sample
        qwen_ids_key = QWEN_INPUT_PREFIX + "input_ids"
        qwen_attn_key = QWEN_INPUT_PREFIX + "attention_mask"
        if qwen_ids_key in batched and qwen_attn_key in batched:
            attn = batched[qwen_attn_key]
            summary_pos = attn.sum(dim=1) - 1  # last non-pad token index
            batched[QWEN_INPUT_PREFIX + "summary_pos"] = summary_pos.to(torch.long)

        # Inject qwen_* tensors and remove temporary content/raw video to free memory
        for k, v in batched.items():
            comp[k] = v
        comp.pop(QWEN_CONTENT_KEY, None)
        comp.pop(LEGACY_EAGLE_CONTENT_KEY, None)
        obs.pop(
            "video", None
        )  # Raw video is encoded into qwen_* tensors; no need to keep it.
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_cot_action_unpack_unnormalize_v1")
class GrootActionUnpackUnnormalizeStep(ProcessorStep):
    env_action_dim: int = 0
    # Apply inverse of min-max normalization if it was used in preprocessor
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Expect model outputs to be in TransitionKey.ACTION as (B, T, D_model)
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition

        # Select last timestep and slice to env dimension
        if action.dim() == 3:
            action = action[:, -1, :]
        # Now action is (B, D_model)
        if self.env_action_dim and action.shape[-1] >= self.env_action_dim:
            action = action[..., : self.env_action_dim]

        # Inverse min-max normalization mirroring _min_max_norm:
        # forward: y = 2 * (x - min) / denom - 1, with y=0 when denom==0
        # inverse: x = (y+1)/2 * denom + min, and when denom==0 -> x = min
        if self.normalize_min_max and self.stats is not None:
            stats_k = self.stats.get("action", {})
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
        """
        Returns a serializable dictionary of the processor's configuration.

        Excludes 'stats' since they are saved separately via state_dict().
        """
        return {
            "env_action_dim": self.env_action_dim,
            "normalize_min_max": self.normalize_min_max,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns normalization statistics as a flat state dictionary.

        This enables saving stats to safetensors files, similar to normalizer_processor.
        """
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        Loads normalization statistics from a flat state dictionary.

        This enables loading stats from safetensors files during from_pretrained.
        """
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


# Alias for factory.py dynamic loading
make_groot_cot_pre_post_processors = make_groot_pre_post_processors

# Backward-compatible class aliases for older imports/checkpoint codepaths.
GrootEagleEncodeStep = GrootQwenEncodeStep
GrootEagleCollateStep = GrootQwenCollateStep
