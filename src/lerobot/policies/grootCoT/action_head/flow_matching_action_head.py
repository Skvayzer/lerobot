# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributions import Beta

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import PretrainedConfig
    from transformers.feature_extraction_utils import BatchFeature
else:
    PretrainedConfig = object
    BatchFeature = None

from lerobot.policies.groot.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_w = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        b, t, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == b:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, t)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maximum Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    tune_vlln: bool = field(
        default=True,
        metadata={"help": "Whether to tune visual-language normalization/adapter blocks."},
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})
    # Optional extra observation vectors (e.g., IMU, odometry, tactile) projected into state token.
    extra_observation_dims: dict[str, int] = field(default_factory=dict)
    # Optional action-group split for weighted action reconstruction loss.
    primary_action_group_indices: list[int] = field(default_factory=list)
    secondary_action_group_indices: list[int] = field(default_factory=list)
    primary_action_group_loss_weight: float = field(default=1.0)
    secondary_action_group_loss_weight: float = field(default=0.0)
    # Legacy aliases retained for backward compatibility with old checkpoints.
    upper_body_joint_indices: list[int] = field(default_factory=list)
    lower_body_joint_indices: list[int] = field(default_factory=list)
    upper_body_loss_weight: float = field(default=1.0)
    lower_body_loss_weight: float = field(default=0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
        lora_config: dict | None = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.extra_observation_dims: dict[str, int] = {}
        self._extra_obs_key_to_module_key: dict[str, str] = {}
        self.extra_observation_projectors = nn.ModuleDict()
        # Older pretrained configs may not carry this newly added field yet.
        extra_observation_dims_cfg = getattr(config, "extra_observation_dims", None)
        if not isinstance(extra_observation_dims_cfg, dict):
            extra_observation_dims_cfg = {}
        for obs_key, raw_dim in sorted(extra_observation_dims_cfg.items()):
            try:
                obs_dim = int(raw_dim)
            except (TypeError, ValueError):
                continue
            if obs_dim <= 0:
                continue
            module_key = obs_key.replace(".", "__")
            self.extra_observation_dims[obs_key] = obs_dim
            self._extra_obs_key_to_module_key[obs_key] = module_key
            self.extra_observation_projectors[module_key] = nn.Sequential(
                nn.LayerNorm(obs_dim),
                nn.Linear(obs_dim, self.input_embedding_dim),
            )
        if self.extra_observation_dims:
            print(
                "[FlowmatchingActionHead] Extra observation projectors initialized for keys: "
                f"{sorted(self.extra_observation_dims.keys())}"
            )

        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self._configure_joint_loss_split()
        
        # Apply LoRA if configured
        self.lora_config = lora_config
        if self.lora_config is not None and self.lora_config.get("r", 0) > 0:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                # Convert dict to LoraConfig object
                peft_config = LoraConfig(
                    r=self.lora_config.get("r"),
                    lora_alpha=self.lora_config.get("lora_alpha", 16),
                    lora_dropout=self.lora_config.get("lora_dropout", 0.05),
                    target_modules=self.lora_config.get("target_modules"),
                    bias="none",
                    task_type=None, 
                )
                
                print(f"[FlowmatchingActionHead] Applying LoRA to DiT with config: {peft_config}")
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()
            except ImportError:
                print("[FlowmatchingActionHead] Warning: peft not installed, cannot apply LoRA.")
        
        self.set_trainable_parameters(
            config.tune_projector,
            config.tune_diffusion_model,
            tune_vlln=getattr(config, "tune_vlln", True),
        )

    def set_trainable_parameters(
        self,
        tune_projector: bool,
        tune_diffusion_model: bool,
        tune_vlln: bool | None = None,
        lora_config: dict | None = None,
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = getattr(self.config, "tune_vlln", True) if tune_vlln is None else bool(tune_vlln)

        # Preserve backwards compatibility for callers that still pass lora_config here.
        del lora_config

        is_peft = hasattr(self.model, "peft_config")

        # Strict staged training semantics: freeze everything first, then whitelist trainable blocks.
        for p in self.parameters():
            p.requires_grad = False

        if tune_projector:
            self.state_encoder.requires_grad_(True)
            self.action_encoder.requires_grad_(True)
            self.action_decoder.requires_grad_(True)
            self.extra_observation_projectors.requires_grad_(True)
            self.future_tokens.requires_grad_(True)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(True)

        # vlln/vl_self_attention are orthogonal to projector and diffusion toggles.
        if self.config.use_vlln and self.tune_vlln:
            self.vlln.requires_grad_(True)
            self.vl_self_attention.requires_grad_(True)

        if tune_diffusion_model:
            if is_peft:
                # Keep only adapter params trainable in PEFT mode.
                for name, p in self.model.named_parameters():
                    if ("lora_" in name) or ("modules_to_save" in name):
                        p.requires_grad = True
            else:
                self.model.requires_grad_(True)
        else:
            self.model.requires_grad_(False)

        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head vlln: {self.tune_vlln and self.config.use_vlln}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model} (LoRA: {is_peft})")

        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                self.extra_observation_projectors.eval()
                self.future_tokens.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if self.config.use_vlln and not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    @staticmethod
    def _normalize_joint_indices(indices: list[int] | None, action_dim: int, name: str) -> list[int]:
        normalized: list[int] = []
        if not indices:
            return normalized

        for raw_idx in indices:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                print(f"[FlowmatchingActionHead] Ignoring non-integer index in {name}: {raw_idx}")
                continue
            if idx < 0:
                print(f"[FlowmatchingActionHead] Ignoring negative index in {name}: {idx}")
                continue
            if idx >= action_dim:
                print(
                    "[FlowmatchingActionHead] Ignoring out-of-range index in "
                    f"{name}: {idx} (action_dim={action_dim})"
                )
                continue
            normalized.append(idx)
        return sorted(set(normalized))

    def _configure_joint_loss_split(self) -> None:
        action_dim = int(self.action_dim)
        primary_weight = float(getattr(self.config, "primary_action_group_loss_weight", 1.0))
        secondary_weight = float(getattr(self.config, "secondary_action_group_loss_weight", 0.0))
        legacy_primary_weight = float(getattr(self.config, "upper_body_loss_weight", primary_weight))
        legacy_secondary_weight = float(getattr(self.config, "lower_body_loss_weight", secondary_weight))
        if primary_weight == 1.0 and legacy_primary_weight != 1.0:
            primary_weight = legacy_primary_weight
        if secondary_weight == 0.0 and legacy_secondary_weight != 0.0:
            secondary_weight = legacy_secondary_weight
        primary_indices_raw = list(getattr(self.config, "primary_action_group_indices", []) or [])
        secondary_indices_raw = list(getattr(self.config, "secondary_action_group_indices", []) or [])
        legacy_primary_indices = list(getattr(self.config, "upper_body_joint_indices", []) or [])
        legacy_secondary_indices = list(getattr(self.config, "lower_body_joint_indices", []) or [])
        if not primary_indices_raw and legacy_primary_indices:
            primary_indices_raw = legacy_primary_indices
        if not secondary_indices_raw and legacy_secondary_indices:
            secondary_indices_raw = legacy_secondary_indices

        primary_indices = self._normalize_joint_indices(
            primary_indices_raw, action_dim, "primary_action_group_indices"
        )
        secondary_indices = self._normalize_joint_indices(
            secondary_indices_raw, action_dim, "secondary_action_group_indices"
        )

        overlap = sorted(set(primary_indices).intersection(secondary_indices))
        if overlap:
            raise ValueError(
                "primary_action_group_indices and secondary_action_group_indices overlap in FlowmatchingActionHead: "
                f"{overlap}"
            )

        both_sets_provided = bool(primary_indices) and bool(secondary_indices)
        primary_selector = torch.zeros(action_dim, dtype=torch.float32)
        secondary_selector = torch.zeros(action_dim, dtype=torch.float32)

        if both_sets_provided:
            primary_selector[primary_indices] = 1.0
            secondary_selector[secondary_indices] = 1.0
        elif primary_indices:
            primary_selector[primary_indices] = 1.0
            secondary_selector = 1.0 - primary_selector
        elif secondary_indices:
            secondary_selector[secondary_indices] = 1.0
            primary_selector = 1.0 - secondary_selector
        else:
            primary_selector[:] = 1.0
            secondary_selector[:] = 0.0

        if primary_weight < 0.0 or secondary_weight < 0.0:
            raise ValueError(
                "primary_action_group_loss_weight and secondary_action_group_loss_weight must be non-negative. "
                f"Got ({primary_weight}, {secondary_weight})."
            )
        if primary_weight == 0.0 and secondary_weight == 0.0:
            raise ValueError(
                "At least one of primary_action_group_loss_weight or "
                "secondary_action_group_loss_weight must be > 0."
            )

        self.primary_action_group_loss_weight = primary_weight
        self.secondary_action_group_loss_weight = secondary_weight
        self.primary_action_group_indices = primary_indices
        self.secondary_action_group_indices = secondary_indices
        # Legacy aliases for existing logs/consumers.
        self.upper_body_loss_weight = primary_weight
        self.lower_body_loss_weight = secondary_weight
        self.upper_body_joint_indices = primary_indices
        self.lower_body_joint_indices = secondary_indices
        self.register_buffer("_primary_joint_selector", primary_selector.view(1, 1, -1), persistent=False)
        self.register_buffer("_secondary_joint_selector", secondary_selector.view(1, 1, -1), persistent=False)

        print(
            "[FlowmatchingActionHead] Joint loss split: "
            f"primary_dims={int(primary_selector.sum().item())}, "
            f"secondary_dims={int(secondary_selector.sum().item())}, "
            f"primary_weight={self.primary_action_group_loss_weight}, "
            f"secondary_weight={self.secondary_action_group_loss_weight}"
        )
        if primary_indices or secondary_indices:
            print(
                "[FlowmatchingActionHead] Joint index sets: "
                f"primary={primary_indices}, secondary={secondary_indices}"
            )

    def sample_time(self, batch_size, device, dtype):
        # Beta/Dirichlet sampling is not implemented for bfloat16 in PyTorch.
        # Match original groot behavior: sample in fp32, then cast.
        alpha = torch.tensor(self.config.noise_beta_alpha, device=device, dtype=torch.float32)
        beta = torch.tensor(self.config.noise_beta_beta, device=device, dtype=torch.float32)
        dist = Beta(alpha, beta)
        sample = dist.sample([batch_size]).to(device=device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _encode_extra_observations(self, action_input: BatchFeature, target_dtype: torch.dtype) -> torch.Tensor | None:
        if not self.extra_observation_projectors:
            return None

        accum: torch.Tensor | None = None
        count = 0
        device = action_input.state.device

        for obs_key, module_key in self._extra_obs_key_to_module_key.items():
            if obs_key not in action_input:
                continue

            obs_tensor = action_input[obs_key]
            if not isinstance(obs_tensor, torch.Tensor):
                continue

            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            elif obs_tensor.dim() == 3 and obs_tensor.shape[1] == 1:
                obs_tensor = obs_tensor[:, 0, :]
            elif obs_tensor.dim() > 2:
                obs_tensor = obs_tensor.reshape(obs_tensor.shape[0], -1)

            obs_tensor = obs_tensor.to(device=device, dtype=torch.float32)

            expected_dim = self.extra_observation_dims[obs_key]
            if obs_tensor.shape[-1] > expected_dim:
                obs_tensor = obs_tensor[:, :expected_dim]
            elif obs_tensor.shape[-1] < expected_dim:
                pad = torch.zeros(
                    (obs_tensor.shape[0], expected_dim - obs_tensor.shape[-1]),
                    dtype=obs_tensor.dtype,
                    device=obs_tensor.device,
                )
                obs_tensor = torch.cat([obs_tensor, pad], dim=1)

            encoded = self.extra_observation_projectors[module_key](obs_tensor).to(dtype=target_dtype)
            accum = encoded if accum is None else accum + encoded
            count += 1

        if accum is None or count == 0:
            return None
        return (accum / float(count)).unsqueeze(1)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        extra_state_features = self._encode_extra_observations(action_input, target_dtype=state_features.dtype)
        if extra_state_features is not None:
            state_features = state_features + extra_state_features

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Weighted split loss across primary and secondary action groups.
        action_mask = action_input.action_mask.to(dtype=pred_actions.dtype)
        mse = F.mse_loss(pred_actions, velocity, reduction="none")

        primary_selector = self._primary_joint_selector.to(device=pred_actions.device, dtype=pred_actions.dtype)
        secondary_selector = self._secondary_joint_selector.to(device=pred_actions.device, dtype=pred_actions.dtype)
        primary_mask = action_mask * primary_selector
        secondary_mask = action_mask * secondary_selector

        primary_denom = primary_mask.sum()
        secondary_denom = secondary_mask.sum()

        zero = mse.new_zeros(())
        primary_loss = (mse * primary_mask).sum() / primary_denom if primary_denom.item() > 0 else zero
        secondary_loss = (mse * secondary_mask).sum() / secondary_denom if secondary_denom.item() > 0 else zero
        loss = (
            self.primary_action_group_loss_weight * primary_loss
            + self.secondary_action_group_loss_weight * secondary_loss
        )

        if primary_denom.item() == 0 and self.primary_action_group_loss_weight > 0:
            print(
                "[GROOT][DEBUG] primary action-group loss mask is empty; "
                f"primary_selector_sum={primary_selector.sum().detach().float().cpu().item()}, "
                f"action_mask_sum={action_mask.sum().detach().float().cpu().item()}"
            )
        if secondary_denom.item() == 0 and self.secondary_action_group_loss_weight > 0:
            print(
                "[GROOT][DEBUG] secondary action-group loss mask is empty; "
                f"secondary_selector_sum={secondary_selector.sum().detach().float().cpu().item()}, "
                f"action_mask_sum={action_mask.sum().detach().float().cpu().item()}"
            )
        if torch.isnan(loss):
            with torch.no_grad():
                print(
                    "[GROOT][DEBUG] NaN loss detected.",
                    f"primary_loss={primary_loss.detach().float().cpu().item()}",
                    f"secondary_loss={secondary_loss.detach().float().cpu().item()}",
                    f"primary_denom={primary_denom.detach().float().cpu().item()}",
                    f"secondary_denom={secondary_denom.detach().float().cpu().item()}",
                    f"pred_actions_stats=(min={pred_actions.min().detach().float().cpu().item()}, "
                    f"max={pred_actions.max().detach().float().cpu().item()}, "
                    f"mean={pred_actions.mean().detach().float().cpu().item()})",
                    f"velocity_stats=(min={velocity.min().detach().float().cpu().item()}, "
                    f"max={velocity.max().detach().float().cpu().item()}, "
                    f"mean={velocity.mean().detach().float().cpu().item()})",
                )
        output_dict = {
            "loss": loss,
            "loss_primary": primary_loss.detach(),
            "loss_secondary": secondary_loss.detach(),
            "loss_primary_weighted": (self.primary_action_group_loss_weight * primary_loss).detach(),
            "loss_secondary_weighted": (self.secondary_action_group_loss_weight * secondary_loss).detach(),
            # Legacy keys kept for backward-compatible logging.
            "loss_upper": primary_loss.detach(),
            "loss_lower": secondary_loss.detach(),
            "loss_upper_weighted": (self.primary_action_group_loss_weight * primary_loss).detach(),
            "loss_lower_weighted": (self.secondary_action_group_loss_weight * secondary_loss).detach(),
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        extra_state_features = self._encode_extra_observations(action_input, target_dtype=state_features.dtype)
        if extra_state_features is not None:
            state_features = state_features + extra_state_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
