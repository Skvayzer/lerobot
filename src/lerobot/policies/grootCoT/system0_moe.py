"""
System 0: Single MoE reactive policy for tactile-aware action refinement.

NOT separate skills. The MoE router, conditioned on physical intent from
System 1 (DiT layer 14), automatically specializes experts by contact regime.

Training: PPO in Isaac Lab with dense sim rewards. No teleop needed.
Inference: <1ms forward pass on robot GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class System0Config:
    # Input dimensions
    joint_dim: int = 28          # joint positions
    vel_dim: int = 28            # joint velocities
    tactile_dim: int = 66        # 33 per hand x 2
    torque_dim: int = 28         # joint torques
    target_dim: int = 28         # coarse targets from System 1
    intent_dim: int = 128        # physical intent from DiT layer 14
    # Set to 0 when training standalone (Stage 3 without System 1 intent)
    # Set to 128 when co-training with System 1 (Stage 4)

    # MoE config
    hidden_dim: int = 256
    n_experts: int = 8
    top_k: int = 2

    # Output
    action_dim: int = 28         # delta_q corrections
    feedback_dim: int = 64       # tactile state feedback to System 1

    # Safety
    max_delta_q: float = 0.1     # clamp corrections to +/-0.1 rad
    kp_base_arm: float = 80.0
    kp_base_finger: float = 1.5
    kd_base_arm: float = 5.0
    kd_base_finger: float = 0.1

    @property
    def input_dim(self) -> int:
        return (self.joint_dim + self.vel_dim + self.tactile_dim +
                self.torque_dim + self.target_dim + self.intent_dim)


class MoEFFN(nn.Module):
    """Single MoE feed-forward layer with top-k routing."""

    def __init__(self, hidden_dim: int, n_experts: int, top_k: int, intent_dim: int):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts

        # Each expert: hidden -> 2*hidden -> hidden
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(n_experts)
        ])

        # Router conditioned on hidden + physical_intent
        router_input_dim = hidden_dim + intent_dim if intent_dim > 0 else hidden_dim
        self.router = nn.Linear(router_input_dim, n_experts)
        self.has_intent = intent_dim > 0

    def forward(self, x: torch.Tensor, intent: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, hidden_dim) hidden state
            intent: (B, intent_dim) physical intent from System 1, or None
        Returns:
            (B, hidden_dim) MoE output
        """
        B = x.shape[0]

        # Compute routing weights
        if self.has_intent and intent is not None:
            router_input = torch.cat([x, intent], dim=-1)
        else:
            router_input = x
        logits = self.router(router_input)  # (B, n_experts)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # renormalize

        # Compute weighted expert outputs
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            for e_idx in range(self.n_experts):
                mask = indices[:, k] == e_idx  # (B,) bool
                if mask.any():
                    expert_out = self.experts[e_idx](x[mask])  # (n_selected, hidden)
                    output[mask] += weights[mask, k:k+1] * expert_out

        return output


class System0MoEPolicy(nn.Module):
    """
    Single reactive policy with MoE for automatic behavior specialization.

    Inputs: proprioception + tactile + targets + physical_intent
    Outputs: delta_q corrections + stiffness/damping adjustments + feedback

    At 100Hz on robot, <1ms forward pass.
    """

    def __init__(self, config: System0Config):
        super().__init__()
        self.config = config

        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )

        # Two MoE layers with residual connections
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.moe1 = MoEFFN(config.hidden_dim, config.n_experts, config.top_k, config.intent_dim)

        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.moe2 = MoEFFN(config.hidden_dim, config.n_experts, config.top_k, config.intent_dim)

        # Output heads
        self.delta_q_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.delta_kp_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.delta_kd_head = nn.Linear(config.hidden_dim, config.action_dim)

        # Feedback encoder: tactile+torques -> compact vector for System 1
        self.feedback_encoder = nn.Sequential(
            nn.Linear(config.tactile_dim + config.torque_dim, 128),
            nn.SiLU(),
            nn.Linear(128, config.feedback_dim),
        )

        # Base gains as buffers (not parameters)
        kp_base = torch.cat([
            torch.full((14,), config.kp_base_arm),    # arm joints 0-13
            torch.full((14,), config.kp_base_finger),  # finger joints 14-27
        ])
        kd_base = torch.cat([
            torch.full((14,), config.kd_base_arm),
            torch.full((14,), config.kd_base_finger),
        ])
        self.register_buffer("kp_base", kp_base)
        self.register_buffer("kd_base", kd_base)

    def forward(self, obs: torch.Tensor, intent: torch.Tensor | None = None) -> dict:
        """
        Args:
            obs: (B, input_dim - intent_dim) or (B, input_dim) if intent concatenated
                 Order: joint_pos, joint_vel, tactile, torques, coarse_targets, [intent]
            intent: (B, intent_dim) physical intent from System 1, or None
                    If None and config.intent_dim > 0, zeros are used.

        Returns dict with:
            delta_q: (B, 28) joint corrections, clamped +/-max_delta_q
            kp: (B, 28) stiffness commands
            kd: (B, 28) damping commands
            feedback: (B, feedback_dim) tactile state for System 1
        """
        B = obs.shape[0]
        device = obs.device

        # Handle intent
        if intent is None and self.config.intent_dim > 0:
            intent = torch.zeros(B, self.config.intent_dim, device=device)

        # Build full input
        if intent is not None and self.config.intent_dim > 0:
            if obs.shape[-1] == self.config.input_dim - self.config.intent_dim:
                full_input = torch.cat([obs, intent], dim=-1)
            else:
                full_input = obs  # intent already included
        else:
            full_input = obs

        # Encode
        x = self.input_encoder(full_input)

        # MoE layers with residual
        x = x + self.moe1(self.norm1(x), intent)
        x = x + self.moe2(self.norm2(x), intent)

        # Output heads
        delta_q = self.delta_q_head(x).clamp(-self.config.max_delta_q, self.config.max_delta_q)
        delta_kp = self.delta_kp_head(x)
        delta_kd = self.delta_kd_head(x)

        # Compute actual Kp/Kd commands: base x sigmoid(delta) x 10
        kp = self.kp_base * torch.sigmoid(delta_kp) * 10.0
        kd = self.kd_base * torch.sigmoid(delta_kd) * 10.0

        # Feedback encoder (tactile + torques -> compact state for System 1)
        tactile_start = self.config.joint_dim + self.config.vel_dim
        tactile_end = tactile_start + self.config.tactile_dim
        torque_end = tactile_end + self.config.torque_dim
        tactile = obs[:, tactile_start:tactile_end] if obs.shape[-1] >= torque_end else torch.zeros(B, self.config.tactile_dim, device=device)
        torques = obs[:, tactile_end:torque_end] if obs.shape[-1] >= torque_end else torch.zeros(B, self.config.torque_dim, device=device)
        feedback = self.feedback_encoder(torch.cat([tactile, torques], dim=-1))

        return {
            "delta_q": delta_q,
            "kp": kp,
            "kd": kd,
            "feedback": feedback,
        }

    def compute_motor_commands(self, coarse_targets: torch.Tensor,
                                obs: torch.Tensor,
                                intent: torch.Tensor | None = None) -> dict:
        """Convenience method for deployment at 100Hz."""
        out = self.forward(obs, intent)
        q_cmd = coarse_targets + out["delta_q"]
        return {
            "q": q_cmd,
            "dq": torch.zeros_like(q_cmd),  # velocity target = 0 for position control
            "tau_ff": torch.zeros_like(q_cmd),  # no feedforward torque
            "kp": out["kp"],
            "kd": out["kd"],
            "feedback": out["feedback"],
        }

    def freeze_experts(self, expert_indices: list[int]):
        """Freeze specific experts for skill expansion."""
        for moe in [self.moe1, self.moe2]:
            for idx in expert_indices:
                if idx < len(moe.experts):
                    for p in moe.experts[idx].parameters():
                        p.requires_grad_(False)

    def add_experts(self, n_new: int = 2):
        """Add new experts to both MoE layers. Freeze existing first."""
        self.freeze_experts(list(range(self.config.n_experts)))
        for moe in [self.moe1, self.moe2]:
            for _ in range(n_new):
                new_expert = nn.Sequential(
                    nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                )
                moe.experts.append(new_expert)
            # Expand router
            old_n = moe.router.out_features
            new_n = old_n + n_new
            old_router = moe.router
            new_router = nn.Linear(old_router.in_features, new_n)
            new_router.weight.data[:old_n] = old_router.weight.data
            new_router.bias.data[:old_n] = old_router.bias.data
            new_router.weight.data[old_n:] = 0.01 * torch.randn(n_new, old_router.in_features)
            new_router.bias.data[old_n:] = -2.0  # low initial routing probability
            moe.router = new_router
        self.config.n_experts += n_new
