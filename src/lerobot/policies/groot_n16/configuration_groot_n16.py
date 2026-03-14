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
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("groot_n16")
@dataclass
class GrootN16Config(PreTrainedConfig):
    """Configuration for GR00T N1.6 policy wrapper.

    N1.6 differences from N1.5:
    - 32-layer AlternateVLDiT (vs 16-layer DiT)
    - Eagle-Block2A-2B-v2 backbone (same family, different integration)
    - max_state_dim=29, max_action_dim=29 (vs 64/32)
    - No eagle_ prefix on VLM keys (uses pixel_values, input_ids, attention_mask)
    - State-relative action representation
    """

    # Basic policy settings
    n_obs_steps: int = 1
    chunk_size: int = 50   # Matches nvidia/GR00T-N1.6-3B action_horizon=50
    n_action_steps: int = 50

    # Dimension settings — actual nvidia/GR00T-N1.6-3B checkpoint values.
    # apply_sincos_state_encoding=True in the checkpoint: raw 64-dim state → 128-dim sincos.
    # Our processor zero-pads state to max_state_dim; sincos can be added for better alignment.
    max_state_dim: int = 128
    max_action_dim: int = 128

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Image preprocessing
    image_size: tuple[int, int] = (224, 224)

    # Path or HuggingFace model ID for the base GR00T N1.6 model
    base_model_path: str = "nvidia/GR00T-N1.6-3B"

    # Embodiment tag to use for training
    embodiment_tag: str = "unitree_g1"

    # Optional state inference from observation keys when dataset does not provide observation.state
    infer_state_from_obs: bool = True
    state_keys: list[str] | None = None
    state_key_regex: str | None = None
    state_key_exclude_regex: str | None = None

    # Fine-tuning control arguments
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # Training parameters
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 1.0
    warmup_ratio: float = 0.05
    scheduler_num_decay_steps: int = 0
    scheduler_decay_lr_ratio: float = 0.1
    use_bf16: bool = True

    # ROCm compatibility: set False to disable flash attention
    # Note: N1.6 EagleBackbone asserts use_flash_attention=True at init time,
    # so we always pass True but then patch to eager attn post-load when this is False.
    use_flash_attention: bool = False

    # Dataset parameters
    video_backend: str = "torchvision_av"
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True

    # Optional dataset paths
    dataset_paths: list[str] | None = None
    output_dir: str = "./tmp/gr00t_n16"
    save_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 32
    dataloader_num_workers: int = 8
    report_to: str = "wandb"
    resume: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features for GR00T N1.6."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "GrootN16 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature
        else:
            state_shape = self.input_features[OBS_STATE].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature
        else:
            action_shape = self.output_features[ACTION].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        num_decay = self.scheduler_num_decay_steps if self.scheduler_num_decay_steps > 0 else 10_000_000
        num_warmup = int(num_decay * self.warmup_ratio)
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=num_warmup,
            num_decay_steps=num_decay,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * self.scheduler_decay_lr_ratio,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
