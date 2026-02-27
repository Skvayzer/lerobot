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
    # Vision-language processor/model ID for Qwen-based backbone
    vlm_processor_model_id: str = "Qwen/Qwen3-VL-8B-Thinking"
    
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
            "observation.images.cam_right_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
    )
    # Missing-view policy for canonical Dex3 camera packing.
    # - "zero_fill": substitute missing camera with a zero image tensor.
    # - "error": fail fast if any canonical camera is missing.
    dex3_missing_camera_policy: str = "zero_fill"

    # Optional joint split for weighted action loss.
    # If only upper indices are provided, lower is treated as complement.
    # If only lower indices are provided, upper is treated as complement.
    # If both are empty, all joints are treated as upper.
    # Default split matches current 28-dim G1 action packing used in this project:
    # first 14 dims (upper group), last 14 dims (lower group).
    upper_body_joint_indices: list[int] = field(default_factory=lambda: list(range(14)))
    lower_body_joint_indices: list[int] = field(default_factory=lambda: list(range(14, 28)))
    upper_body_loss_weight: float = 1.0
    lower_body_loss_weight: float = 1.0

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
    warmup_ratio: float = 0.05
    use_bf16: bool = True

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

    def __post_init__(self):
        super().__post_init__()
        self._apply_training_stage_preset()

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
        if self.upper_body_loss_weight < 0.0:
            raise ValueError("upper_body_loss_weight must be >= 0.")
        if self.lower_body_loss_weight < 0.0:
            raise ValueError("lower_body_loss_weight must be >= 0.")
        if self.upper_body_loss_weight == 0.0 and self.lower_body_loss_weight == 0.0:
            raise ValueError("At least one of upper_body_loss_weight or lower_body_loss_weight must be > 0.")
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

        self.upper_body_joint_indices = _normalize_joint_indices(
            self.upper_body_joint_indices, "upper_body_joint_indices"
        )
        self.lower_body_joint_indices = _normalize_joint_indices(
            self.lower_body_joint_indices, "lower_body_joint_indices"
        )
        overlap = set(self.upper_body_joint_indices).intersection(self.lower_body_joint_indices)
        if overlap:
            raise ValueError(
                "upper_body_joint_indices and lower_body_joint_indices overlap: "
                f"{sorted(overlap)}"
            )

        # groot_repo_path is now optional since we ported the components
        # No validation needed

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
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Return scheduler configuration."""
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=int(10000 * self.warmup_ratio),  # 5% warmup by default
            num_decay_steps=10000,  # Adjust based on training steps
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
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
