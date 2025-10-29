# === IsaacGr00t START ===
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


_DEFAULT_MOTOR_ORDER = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
]

_DEFAULT_STATE_GROUPS = {
    "left_arm": list(range(0, 7)),
    "right_arm": list(range(7, 14)),
    "left_hand": list(range(14, 21)),
    "right_hand": list(range(21, 28)),
}


@PreTrainedConfig.register_subclass("isaac_gr00t")
@dataclass
class IsaacGr00tConfig(PreTrainedConfig):
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    data_config: str = "unitree_g1"
    embodiment_tag: str = "new_embodiment"
    motor_order: list[str] = field(default_factory=lambda: list(_DEFAULT_MOTOR_ORDER))
    state_groups: dict[str, list[int]] = field(default_factory=lambda: dict(_DEFAULT_STATE_GROUPS))
    camera_key: str = "observation.images.cam_left_high"
    language_key: str | None = "annotation.human.task_description"
    default_language: str = "Perform the default behavior."
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 1
    denoising_steps: int | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 5e-6

    def __post_init__(self):
        super().__post_init__()
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be > 0")
        if set().union(*self.state_groups.values()) - set(range(len(self.motor_order))):
            raise ValueError("state_groups indices exceed motor_order length")

    def validate_features(self) -> None:
        missing = []
        if self.camera_key not in self.input_features:
            missing.append(self.camera_key)
        if "observation.state" not in self.input_features:
            missing.append("observation.state")
        if "action" not in self.output_features:
            missing.append("action")
        if missing:
            raise ValueError(f"Required features missing from dataset: {missing}")

        state_dim = self.input_features["observation.state"].shape[0]
        if state_dim != len(self.motor_order):
            raise ValueError(
                "Dataset state dimension does not match expected motor order length: "
                f"{state_dim} vs {len(self.motor_order)}"
            )
        for group, indices in self.state_groups.items():
            if not indices:
                raise ValueError(f"State group '{group}' has no indices configured.")
            if max(indices) >= state_dim or min(indices) < 0:
                raise ValueError(f"State group '{group}' indices out of range: {indices}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.n_obs_steps))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
# === IsaacGr00t END ===
