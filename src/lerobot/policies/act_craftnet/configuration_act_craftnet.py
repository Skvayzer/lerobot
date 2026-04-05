"""ACT-CraftNet configuration.

ACT-CraftNet extends the standard ACT policy with:
  - FrozenDINOv2 ViT-B/14 visual backbone (replaces ResNet-18)
  - DepthCNNEncoder: 2D CNN over depth images (3 views: head + 2 wrists)
  - System0Policy: MoE reactive finger correction with bidirectional S1↔S0 connections
    - 4 experts, top-2 routing, 686D input (enc_out[0] + intent + tactile + finger + s1_fingers)
    - Two-phase: encode_feedback() before encoder, forward_delta() after decoder
    - Time-varying delta (B, T, 14) — per-timestep residual vs S1 finger targets
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_craftnet")
@dataclass
class ACTCraftNetConfig(PreTrainedConfig):
    """Configuration for ACT-CraftNet policy.

    Expected input_features keys:
      - "observation.state"              : FeatureType.STATE  (28D: arm×2 + finger×2)
      - "observation.tactile"            : FeatureType.STATE  (18D)
      - "observation.environment_state"  : FeatureType.ENV    (9D)
      - "observation.images.<cam>"       : FeatureType.VISUAL (3, 224, 224) × n_cameras
      - "observation.depth.<cam>"        : FeatureType.STATE  (1, 120, 160) × n_depth_views

    Expected output_features keys:
      - "action"                         : FeatureType.ACTION (28D)
    """

    # ── Observation structure ──────────────────────────────────────────────────
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Keys for depth views (order matters: head, left_wrist, right_wrist)
    depth_feature_keys: list = field(
        default_factory=lambda: [
            "observation.depth.head",
            "observation.depth.left_wrist",
            "observation.depth.right_wrist",
        ]
    )
    # Tactile observation key
    tactile_feature_key: str = "observation.tactile"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ── Architecture ───────────────────────────────────────────────────────────
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_enc_layers: int = 4
    n_dec_layers: int = 7
    latent_dim: int = 32
    dropout: float = 0.1

    # Depth encoder
    depth_max: float = 2.0         # metres — clips and normalises depth to [0, 1]

    # System 0 MoE
    physical_intent_dim: int = 128
    feedback_dim: int = 64

    # ── Loss ───────────────────────────────────────────────────────────────────
    kl_weight: float = 1.0

    # ── Optimiser ──────────────────────────────────────────────────────────────
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("At least one VISUAL input feature is required.")

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None
