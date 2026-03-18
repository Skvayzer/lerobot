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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.grootCoT.system2_vlm_registry import (
    DEFAULT_SYSTEM2_VLM_MODEL_ID,
    DEFAULT_SYSTEM2_VLM_PRESET,
    resolve_system2_vlm_model_id,
    validate_system2_vlm_preset,
)

DEFAULT_PRIMARY_ACTION_GROUP_INDICES = list(range(14))
DEFAULT_SECONDARY_ACTION_GROUP_INDICES = list(range(14, 28))
DEFAULT_PRIMARY_ACTION_GROUP_LOSS_WEIGHT = 1.0
DEFAULT_SECONDARY_ACTION_GROUP_LOSS_WEIGHT = 1.0


@PreTrainedConfig.register_subclass("groot_cot")
@dataclass
class GrootCoTConfig(PreTrainedConfig):
    """Configuration for Groot policy wrapper."""

    # Basic policy settings
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Dimension settings (must match pretrained GR00T model expectations)
    # Maximum state dimension. Shorter states will be zero-padded.
    max_state_dim: int = 64

    # Maximum action dimension. Shorter actions will be zero-padded.
    max_action_dim: int = 32

    # Normalization (start with identity, adjust as needed)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Image preprocessing (adjust to match Groot's expected input)
    image_size: tuple[int, int] = (224, 224)

    # Groot-specific model parameters (from groot_finetune_script.py)

    # Path or HuggingFace model ID for the base Groot model
    base_model_path: str = "nvidia/GR00T-N1.5-3B"

    # HF repo ID (or local path) that hosts vocab.json and merges.txt for Eagle tokenizer.
    tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5"
    # Explicit System-2 VLM selector:
    # - system2_vlm_model_id: direct HF model id (highest priority)
    # - system2_vlm_preset: registry preset (second priority)
    # - vlm_processor_model_id: legacy field (third priority; retained for compatibility)
    system2_vlm_preset: str = DEFAULT_SYSTEM2_VLM_PRESET
    system2_vlm_model_id: str | None = None
    vlm_processor_model_id: str = DEFAULT_SYSTEM2_VLM_MODEL_ID
    # Resolved canonical model id and resolution source for logging/debug.
    resolved_system2_vlm_model_id: str = field(default="", init=False)
    resolved_system2_vlm_source: str = field(default="", init=False)
    
    # Attention implementation for Qwen VLM backbone ("eager", "sdpa", or "flash_attention_2")
    # Use "eager" if experiencing NaN issues with flash attention
    attn_implementation: str | None = None

    # Embodiment tag to use for training (e.g. 'new_embodiment', 'gr1')
    embodiment_tag: str = "new_embodiment"

    # Optional state inference from observation keys when dataset does not provide observation.state
    # If None, a heuristic will concatenate all 1D observation tensors (excluding images).
    infer_state_from_obs: bool = True
    state_keys: list[str] | None = None
    state_key_regex: str | None = None
    state_key_exclude_regex: str | None = None

    # Optional canonical camera packing for Dex3-style datasets.
    # When enabled, if any canonical Dex3 camera key is present in a sample, the
    # preprocessor will always pack exactly these 4 views in this fixed order.
    # Missing views are handled by `dex3_missing_camera_policy`.
    enforce_dex3_canonical_camera_order: bool = True
    dex3_canonical_camera_order: list[str] = field(
        default_factory=lambda: [
            "observation.images.cam_left_high",
            "observation.images.depth_cam_left_high",
            "observation.images.cam_left_wrist",
            "observation.images.depth_cam_left_wrist",
            "observation.images.cam_right_wrist",
            "observation.images.depth_cam_right_wrist",
        ]
    )
    # Missing-view policy for canonical Dex3 camera packing.
    # - "zero_fill": substitute missing camera with a zero image tensor.
    # - "error": fail fast if any canonical camera is missing.
    dex3_missing_camera_policy: str = "zero_fill"

    # Optional action-group split for weighted action loss.
    # If only primary indices are provided, secondary is treated as complement.
    # If only secondary indices are provided, primary is treated as complement.
    # If both are empty, all joints are treated as primary.
    # Default split matches current 28-dim G1 action packing used in this project:
    # first 14 dims (primary group), last 14 dims (secondary group).
    primary_action_group_indices: list[int] = field(
        default_factory=lambda: list(DEFAULT_PRIMARY_ACTION_GROUP_INDICES)
    )
    secondary_action_group_indices: list[int] = field(
        default_factory=lambda: list(DEFAULT_SECONDARY_ACTION_GROUP_INDICES)
    )
    primary_action_group_loss_weight: float = DEFAULT_PRIMARY_ACTION_GROUP_LOSS_WEIGHT
    secondary_action_group_loss_weight: float = DEFAULT_SECONDARY_ACTION_GROUP_LOSS_WEIGHT
    # Legacy aliases retained for backward compatibility with old configs/checkpoints.
    upper_body_joint_indices: list[int] = field(
        default_factory=lambda: list(DEFAULT_PRIMARY_ACTION_GROUP_INDICES)
    )
    lower_body_joint_indices: list[int] = field(
        default_factory=lambda: list(DEFAULT_SECONDARY_ACTION_GROUP_INDICES)
    )
    upper_body_loss_weight: float = DEFAULT_PRIMARY_ACTION_GROUP_LOSS_WEIGHT
    lower_body_loss_weight: float = DEFAULT_SECONDARY_ACTION_GROUP_LOSS_WEIGHT

    # Fine-tuning control arguments

    # Optional training-stage preset to avoid repeating many tune flags.
    # Stages:
    # - manual: honor explicit tune_* flags as provided.
    # - backbone_align: adapt backbone/projector first, freeze action head.
    # - policy_adapt: freeze backbone, adapt action head.
    # - joint_refine: co-train action head with top LLM layers/projector.
    training_stage: str = "manual"

    # Whether to fine-tune the llm backbone
    tune_llm: bool = False

    # Whether to fine-tune the vision tower
    tune_visual: bool = False

    # Whether to fine-tune the backbone hidden->action-head projector.
    tune_vlm_projector: bool = True

    # Number of top language-model layers to unfreeze in Qwen when LoRA is disabled.
    # 0 means no selective unfreezing (use full unfreeze if tune_llm/tune_visual are enabled).
    tune_top_llm_layers: int = 0

    # Whether to fine-tune the projector
    tune_projector: bool = True

    # Whether to fine-tune the diffusion model
    tune_diffusion_model: bool = False

    # Whether to tune action-head visual-language normalization/adapter blocks (vlln + vl_self_attention).
    tune_vlln: bool = True

    # Freeze backbone and policy; train only the VLM->actor projector
    train_vlm_projector_only: bool = True

    # N1.6 action head: "n15" uses existing FlowmatchingActionHead, "n16" uses Gr00tN1d6ActionHead
    action_head_version: str = "n15"
    # Path to extracted N1.6 action head pretrained weights (.pt file)
    n16_action_head_weights_path: str | None = None

    # IK prior source distribution for flow matching training
    ik_prior_prob: float = 0.0  # 0.0 = disabled (standard noise), 0.4 = recommended for Stage 2
    ik_prior_noise_scale: float = 0.15
    ik_prior_arm_dim: int = 14  # first 14 dims = arm joints for G1

    # Relative actions: subtract current state from target action during training,
    # add current state back during inference
    use_relative_actions: bool = False
    # Path to JSON with pre-computed relative action stats (min/max) for normalization.
    # Required when use_relative_actions=true. Generate with compute_relative_action_stats.py
    relative_action_stats_path: str | None = None

    # FLARE future prediction
    flare_enable: bool = False
    flare_coefficient: float = 0.2
    flare_target_dim: int = 4096
    flare_future_offset: int = 30
    num_future_tokens: int = 32

    # LoRA parameters (from groot_finetune_script.py)
    # Rank for the LORA model. If 0, no LORA will be used.
    lora_rank: int = 0

    # Alpha value for the LORA model
    lora_alpha: int = 16

    # Dropout rate for the LORA model
    lora_dropout: float = 0.1

    # Whether to use the full model for LORA
    lora_full_model: bool = False

    # LoRA target modules for the backbone (list of module names to apply LoRA to)
    # If None, it will default to linear layers in attention/mlp depending on model type
    lora_target_modules: list[str] | None = None

    # LoRA parameters for the Action Head
    # Rank for the Action Head LoRA model. If 0, no LoRA will be used.
    action_head_lora_rank: int = 0
    # Alpha value for the Action Head LoRA model
    action_head_lora_alpha: int = 16
    # Dropout rate for the Action Head LoRA model
    action_head_lora_dropout: float = 0.1
    # LoRA target modules for the Action Head
    action_head_lora_target_modules: list[str] | None = None

    # Training parameters (matching groot_finetune_script.py)
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 1.0
    warmup_ratio: float = 0.05
    scheduler_num_decay_steps: int = 0
    scheduler_decay_lr_ratio: float = 0.1
    # Legacy compatibility fields retained so old checkpoint configs can be loaded.
    use_amp: bool = False
    use_bf16: bool = True
    use_peft: bool = False

    # Dataset parameters
    # Video backend to use for training ('decord' or 'torchvision_av')
    video_backend: str = "decord"

    # Whether to balance dataset weights in mixture datasets
    balance_dataset_weights: bool = True

    # Whether to sample trajectories weighted by their length
    balance_trajectory_weights: bool = True

    # Optional dataset paths for delegating training to Isaac-GR00T runner
    dataset_paths: list[str] | None = None
    output_dir: str = "./tmp/gr00t"
    save_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 32
    dataloader_num_workers: int = 8
    report_to: str = "wandb"
    resume: bool = False

    # Dual-rate execution (System-2 backbone + System-1 action head).
    # When enabled, Qwen backbone latents are refreshed at a lower rate and reused
    # by System-1 chunk prediction between refreshes.
    dual_rate_enable: bool = True
    # Optional train-time simulation of stale System-2 latents.
    dual_rate_apply_in_train: bool = False
    # Control-loop and System-2 target rates used to derive refresh intervals.
    control_hz: float = 30.0
    system2_hz: float = 8.0
    # If > 0, use this explicit System-2 refresh interval in control steps.
    system2_update_every_n_steps: int = 0
    # Optional replanning cadence for System-1 chunk prediction. If 0, only replan
    # when queue is depleted or below `system1_min_queue_size`.
    system1_replan_every_n_steps: int = 0
    # Trigger chunk replanning when queue length is <= this threshold.
    system1_min_queue_size: int = 0
    # Reset-time behavior for cached System-2 latents.
    dual_rate_force_backbone_refresh_on_reset: bool = True
    # Optional non-blocking async System-2 updates for inference/eval.
    # When enabled, System-2 chunk generation runs in a background thread
    # scheduled by wall-clock `system2_hz`, while System-1 keeps emitting actions.
    system2_async_enable: bool = False
    # Max number of precomputed chunks buffered from async System-2 worker.
    system2_async_prefetch_chunks: int = 1
    # If True, block once at startup until first async chunk is ready.
    # Keep False to avoid blocking control loop.
    system2_async_startup_warmup: bool = False
    # Drop stale observation snapshots older than this threshold in async worker.
    system2_async_max_observation_age_s: float = 0.5
    # Async worker diagnostic log interval (in select_action control steps).
    system2_async_log_every_n_steps: int = 100
    # Explicitly mark async scheduling policy (currently wall-clock only).
    system2_async_wall_clock: bool = True

    # Visual feature dropout probability during training.
    # When > 0, randomly zeroes fresh visual features to teach the DiT
    # to function with cached System 2 features alone.
    visual_dropout_p: float = 0.2

    # Whether to render and inject a grounded reference frame
    # (with bounding box) as an additional camera view.
    use_grounded_reference_frame: bool = False

    # Which layer to extract System 1 visual features from.
    # "vit" = pure ViT output (fastest, most spatial).
    # An integer = specific LLM layer index (more semantic, slower).
    system1_visual_source: str = "vit"

    # RECAP-style improvement conditioning.
    recap_enable: bool = False
    recap_adv_indicator_key: str = "observation.extra.adv_indicator"
    recap_adv_indicator_null_value: float = -1.0
    recap_adv_indicator_cond_value: float = 1.0
    # Classifier-free style dropout probability for indicator conditioning.
    recap_adv_indicator_dropout_p: float = 0.3
    # Optional labels sidecar (parquet/jsonl) keyed by dataset index for offline fine-tuning.
    recap_labels_path: str | None = None
    # Optional CFG at inference for velocity predictions.
    recap_adv_indicator_use_cfg: bool = False
    recap_cfg_scale: float = 1.0

    # Value head on top of System-2 (Qwen) embeddings.
    recap_value_head_enable: bool = False
    recap_tune_value_head: bool = True
    recap_value_head_bins: int = 201
    recap_value_head_vmin: float = -1.0
    recap_value_head_vmax: float = 0.0
    # One of {"masked_mean", "last_token"}.
    recap_value_head_pooling: str = "masked_mean"

    def __post_init__(self):
        super().__post_init__()
        self._apply_training_stage_preset()
        self._resolve_system2_vlm()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )
        if self.tune_top_llm_layers < 0:
            raise ValueError("tune_top_llm_layers must be >= 0")
        if self.dex3_missing_camera_policy not in {"zero_fill", "error"}:
            raise ValueError(
                "dex3_missing_camera_policy must be one of {'zero_fill', 'error'}."
            )
        if len(self.dex3_canonical_camera_order) == 0:
            raise ValueError("dex3_canonical_camera_order must contain at least one camera key.")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be > 0.")
        if self.system2_hz <= 0:
            raise ValueError("system2_hz must be > 0.")
        if self.system2_update_every_n_steps < 0:
            raise ValueError("system2_update_every_n_steps must be >= 0.")
        if self.system1_replan_every_n_steps < 0:
            raise ValueError("system1_replan_every_n_steps must be >= 0.")
        if self.system1_min_queue_size < 0:
            raise ValueError("system1_min_queue_size must be >= 0.")
        if self.system2_async_prefetch_chunks < 1:
            raise ValueError("system2_async_prefetch_chunks must be >= 1.")
        if self.system2_async_max_observation_age_s <= 0.0:
            raise ValueError("system2_async_max_observation_age_s must be > 0.")
        if self.system2_async_log_every_n_steps < 1:
            raise ValueError("system2_async_log_every_n_steps must be >= 1.")
        if not (0.0 <= self.visual_dropout_p <= 1.0):
            raise ValueError("visual_dropout_p must be in [0, 1].")
        if self.system1_visual_source != "vit":
            try:
                int(self.system1_visual_source)
            except ValueError:
                raise ValueError(
                    f"system1_visual_source must be 'vit' or an integer layer index, "
                    f"got '{self.system1_visual_source}'"
                )
        if not (0.0 <= self.recap_adv_indicator_dropout_p <= 1.0):
            raise ValueError("recap_adv_indicator_dropout_p must be in [0, 1].")
        if self.recap_value_head_bins < 2:
            raise ValueError("recap_value_head_bins must be >= 2.")
        if self.recap_value_head_vmax <= self.recap_value_head_vmin:
            raise ValueError("recap_value_head_vmax must be > recap_value_head_vmin.")
        if self.recap_value_head_pooling not in {"masked_mean", "last_token"}:
            raise ValueError(
                "recap_value_head_pooling must be one of {'masked_mean', 'last_token'}."
            )
        if self.recap_cfg_scale < 0.0:
            raise ValueError("recap_cfg_scale must be >= 0.")

        # Auto-configure dimensions for N1.6 action head
        if self.action_head_version == "n16":
            if self.max_state_dim < 128:
                self.max_state_dim = 128
            if self.max_action_dim < 128:
                self.max_action_dim = 128

        def _normalize_joint_indices(indices: list[int], name: str) -> list[int]:
            normalized: list[int] = []
            for idx in indices:
                try:
                    value = int(idx)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} contains non-integer value: {idx}") from exc
                if value < 0:
                    raise ValueError(f"{name} contains negative index: {value}")
                normalized.append(value)
            return sorted(set(normalized))

        def _is_non_default_list(value: list[int], default_value: list[int]) -> bool:
            return list(value) != list(default_value)

        def _is_non_default_float(value: float, default_value: float) -> bool:
            return float(value) != float(default_value)

        primary_new_set = _is_non_default_list(
            self.primary_action_group_indices, DEFAULT_PRIMARY_ACTION_GROUP_INDICES
        )
        primary_legacy_set = _is_non_default_list(
            self.upper_body_joint_indices, DEFAULT_PRIMARY_ACTION_GROUP_INDICES
        )
        secondary_new_set = _is_non_default_list(
            self.secondary_action_group_indices, DEFAULT_SECONDARY_ACTION_GROUP_INDICES
        )
        secondary_legacy_set = _is_non_default_list(
            self.lower_body_joint_indices, DEFAULT_SECONDARY_ACTION_GROUP_INDICES
        )

        primary_weight_new_set = _is_non_default_float(
            self.primary_action_group_loss_weight, DEFAULT_PRIMARY_ACTION_GROUP_LOSS_WEIGHT
        )
        primary_weight_legacy_set = _is_non_default_float(
            self.upper_body_loss_weight, DEFAULT_PRIMARY_ACTION_GROUP_LOSS_WEIGHT
        )
        secondary_weight_new_set = _is_non_default_float(
            self.secondary_action_group_loss_weight, DEFAULT_SECONDARY_ACTION_GROUP_LOSS_WEIGHT
        )
        secondary_weight_legacy_set = _is_non_default_float(
            self.lower_body_loss_weight, DEFAULT_SECONDARY_ACTION_GROUP_LOSS_WEIGHT
        )

        if (
            primary_new_set
            and primary_legacy_set
            and list(self.primary_action_group_indices) != list(self.upper_body_joint_indices)
        ):
            print(
                "[GrootCoTConfig] Both primary_action_group_indices and upper_body_joint_indices are set "
                "with different values. Using primary_action_group_indices."
            )
        if (
            secondary_new_set
            and secondary_legacy_set
            and list(self.secondary_action_group_indices) != list(self.lower_body_joint_indices)
        ):
            print(
                "[GrootCoTConfig] Both secondary_action_group_indices and lower_body_joint_indices are set "
                "with different values. Using secondary_action_group_indices."
            )
        if (
            primary_weight_new_set
            and primary_weight_legacy_set
            and float(self.primary_action_group_loss_weight) != float(self.upper_body_loss_weight)
        ):
            print(
                "[GrootCoTConfig] Both primary_action_group_loss_weight and upper_body_loss_weight are set "
                "with different values. Using primary_action_group_loss_weight."
            )
        if (
            secondary_weight_new_set
            and secondary_weight_legacy_set
            and float(self.secondary_action_group_loss_weight) != float(self.lower_body_loss_weight)
        ):
            print(
                "[GrootCoTConfig] Both secondary_action_group_loss_weight and lower_body_loss_weight are set "
                "with different values. Using secondary_action_group_loss_weight."
            )

        primary_indices_raw = list(self.primary_action_group_indices)
        if not primary_new_set and primary_legacy_set:
            primary_indices_raw = list(self.upper_body_joint_indices)

        secondary_indices_raw = list(self.secondary_action_group_indices)
        if not secondary_new_set and secondary_legacy_set:
            secondary_indices_raw = list(self.lower_body_joint_indices)

        primary_weight = float(self.primary_action_group_loss_weight)
        if not primary_weight_new_set and primary_weight_legacy_set:
            primary_weight = float(self.upper_body_loss_weight)

        secondary_weight = float(self.secondary_action_group_loss_weight)
        if not secondary_weight_new_set and secondary_weight_legacy_set:
            secondary_weight = float(self.lower_body_loss_weight)

        if primary_weight < 0.0:
            raise ValueError("primary_action_group_loss_weight must be >= 0.")
        if secondary_weight < 0.0:
            raise ValueError("secondary_action_group_loss_weight must be >= 0.")
        if primary_weight == 0.0 and secondary_weight == 0.0:
            raise ValueError(
                "At least one of primary_action_group_loss_weight or "
                "secondary_action_group_loss_weight must be > 0."
            )

        primary_indices = _normalize_joint_indices(
            primary_indices_raw, "primary_action_group_indices"
        )
        secondary_indices = _normalize_joint_indices(
            secondary_indices_raw, "secondary_action_group_indices"
        )
        overlap = set(primary_indices).intersection(secondary_indices)
        if overlap:
            raise ValueError(
                "primary_action_group_indices and secondary_action_group_indices overlap: "
                f"{sorted(overlap)}"
            )

        # Canonicalize to clear names and mirror to legacy aliases for compatibility.
        self.primary_action_group_indices = primary_indices
        self.secondary_action_group_indices = secondary_indices
        self.primary_action_group_loss_weight = primary_weight
        self.secondary_action_group_loss_weight = secondary_weight
        self.upper_body_joint_indices = list(primary_indices)
        self.lower_body_joint_indices = list(secondary_indices)
        self.upper_body_loss_weight = float(primary_weight)
        self.lower_body_loss_weight = float(secondary_weight)

        # groot_repo_path is now optional since we ported the components
        # No validation needed

    def _resolve_system2_vlm(self) -> None:
        # Validate early to provide clear config-time errors.
        self.system2_vlm_preset = validate_system2_vlm_preset(self.system2_vlm_preset)
        model_id, source, normalized_preset = resolve_system2_vlm_model_id(
            preset=self.system2_vlm_preset,
            explicit_model_id=self.system2_vlm_model_id,
            legacy_model_id=self.vlm_processor_model_id,
        )
        self.system2_vlm_preset = normalized_preset
        self.resolved_system2_vlm_model_id = model_id
        self.resolved_system2_vlm_source = source
        # Mirror for backward compatibility with existing code paths.
        self.vlm_processor_model_id = model_id

    def _apply_training_stage_preset(self) -> None:
        stage = (self.training_stage or "manual").strip().lower()
        if stage == "":
            stage = "manual"

        valid_stages = {"manual", "backbone_align", "policy_adapt", "joint_refine"}
        if stage not in valid_stages:
            raise ValueError(
                f"Unsupported training_stage='{self.training_stage}'. "
                f"Expected one of {sorted(valid_stages)}."
            )

        if stage == "manual":
            self.training_stage = "manual"
            return

        # Stage presets explicitly control freeze/unfreeze behavior.
        self.train_vlm_projector_only = False

        if stage == "backbone_align":
            # Train backbone projector first; optionally train top language layers.
            top_layers = self.tune_top_llm_layers if self.tune_top_llm_layers > 0 else 4
            self.tune_llm = True
            self.tune_visual = False
            self.tune_vlm_projector = True
            self.tune_top_llm_layers = top_layers
            self.tune_projector = False
            self.tune_vlln = False
            self.tune_diffusion_model = False
        elif stage == "policy_adapt":
            # Freeze backbone and adapt policy/action head.
            self.tune_llm = False
            self.tune_visual = False
            self.tune_vlm_projector = False
            self.tune_top_llm_layers = 0
            self.tune_projector = True
            self.tune_vlln = True
            self.tune_diffusion_model = True
        elif stage == "joint_refine":
            # Jointly refine with minimal backbone adaptation.
            top_layers = self.tune_top_llm_layers if self.tune_top_llm_layers > 0 else 4
            self.tune_llm = True
            self.tune_visual = False
            self.tune_vlm_projector = True
            self.tune_top_llm_layers = top_layers
            self.tune_projector = True
            self.tune_vlln = True
            self.tune_diffusion_model = True

        self.training_stage = stage

    def validate_features(self) -> None:
        """Validate and set up input/output features for Groot."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "Groot policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature
        else:
            state_shape = self.input_features["observation.state"].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature
        else:
            action_shape = self.output_features["action"].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        """Return optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Return scheduler configuration.

        When scheduler_num_decay_steps == 0 (default), we set num_decay_steps
        to a very large sentinel value so that the scheduler's auto-scaling
        logic (triggered when num_training_steps < num_decay_steps) will
        shrink it to exactly match the actual training length.
        """
        num_decay = self.scheduler_num_decay_steps if self.scheduler_num_decay_steps > 0 else 10_000_000
        num_warmup = int(num_decay * self.warmup_ratio)
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=num_warmup,
            num_decay_steps=num_decay,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * self.scheduler_decay_lr_ratio,
        )

    def get_system2_update_interval_steps(self) -> int:
        if self.system2_update_every_n_steps > 0:
            return int(self.system2_update_every_n_steps)
        return max(1, int(round(float(self.control_hz) / float(self.system2_hz))))

    @property
    def observation_delta_indices(self) -> None:
        """Return indices for delta observations (None for Groot)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return indices for delta actions."""
        return list(range(min(self.chunk_size, 16)))

    @property
    def reward_delta_indices(self) -> None:
        """Return indices for delta rewards (None for Groot)."""
        return None
