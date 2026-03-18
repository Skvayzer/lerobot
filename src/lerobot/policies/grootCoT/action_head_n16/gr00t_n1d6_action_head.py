"""N1.6 Action Head for CraftNet.

Ported from Isaac-GR00T gr00t/model/gr00t_n1d6/gr00t_n1d6.py (Gr00tN1d6ActionHead only).
Imports rewritten to use local modules instead of gr00t package.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from lerobot.policies.grootCoT.action_head_n16.dit import AlternateVLDiT, DiT
from lerobot.policies.grootCoT.action_head_n16.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


@dataclass
class N16ActionHeadConfig:
    """Config for N1.6 action head, extracted from Gr00tN1d6Config defaults."""

    backbone_embedding_dim: int = 2048
    input_embedding_dim: int = 1536
    hidden_size: int = 1024
    max_state_dim: int = 128
    max_action_dim: int = 128
    action_horizon: int = 50
    use_alternate_vl_dit: bool = True
    attend_text_every_n_blocks: int = 2
    add_pos_embed: bool = True
    use_vlln: bool = True
    max_seq_len: int = 1024
    max_num_embodiments: int = 32
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    state_dropout_prob: float = 0.0
    state_additive_noise_scale: float = 0.0
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    attn_dropout: float = 0.2

    # IK prior source distribution settings
    ik_prior_prob: float = 0.0
    ik_prior_noise_scale: float = 0.15
    ik_prior_arm_dim: int = 14

    diffusion_model_cfg: dict = field(default_factory=lambda: {
        "positional_embeddings": None,
        "num_layers": 32,
        "num_attention_heads": 32,
        "attention_head_dim": 48,
        "norm_type": "ada_norm",
        "dropout": 0.2,
        "final_dropout": True,
        "output_dim": 1024,
        "interleave_self_attention": True,
    })


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for N1.6 flow matching diffusion policy.

    Ported from Isaac-GR00T. Uses AlternateVLDiT (default) or DiT for denoising,
    with multi-embodiment action/state encoders.
    """

    supports_gradient_checkpointing = True

    def __init__(self, config: N16ActionHeadConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for N1.6 diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            print("Using DiT for N1.6 diffusion model")

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def _sample_source(self, actions: torch.Tensor) -> torch.Tensor:
        """Sample source distribution x_0: IK linear prior for arm joints OR Gaussian noise."""
        B, H, D = actions.shape
        device = actions.device
        dtype = actions.dtype
        arm_dim = min(self.config.ik_prior_arm_dim, D)
        p_prior = self.config.ik_prior_prob
        noise_scale = self.config.ik_prior_noise_scale

        x_0 = torch.randn(B, H, D, device=device, dtype=dtype)

        if p_prior <= 0 or not self.training or arm_dim <= 0:
            return x_0

        use_prior = (torch.rand(B, device=device) < p_prior)
        if not use_prior.any():
            return x_0

        arm_start = actions[:, 0:1, :arm_dim]
        arm_end = actions[:, -1:, :arm_dim]
        alphas = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1, H, 1)
        arm_linear = arm_start + alphas * (arm_end - arm_start)
        arm_prior = arm_linear + noise_scale * torch.randn_like(arm_linear)

        prior_mask = use_prior.view(B, 1, 1).expand(B, H, arm_dim)
        x_0[:, :, :arm_dim] = torch.where(prior_mask, arm_prior, x_0[:, :, :arm_dim])

        return x_0

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        # Ensure attention masks are bool for diffusers SDPA compatibility
        if "backbone_attention_mask" in backbone_output and backbone_output["backbone_attention_mask"] is not None:
            backbone_output["backbone_attention_mask"] = backbone_output["backbone_attention_mask"].bool()
        if "image_mask" in backbone_output and backbone_output["image_mask"] is not None:
            backbone_output["image_mask"] = backbone_output["image_mask"].bool()
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Forward pass (training).

        Args:
            backbone_output: {backbone_features, backbone_attention_mask, image_mask}
            action_input: {state, action, embodiment_id, action_mask}
        Returns:
            dict with loss and diagnostics
        """
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        embodiment_id = action_input.embodiment_id

        # Encode state
        state_features = self.state_encoder(action_input.state, embodiment_id)

        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        if self.training and self.state_additive_noise_scale > 0:
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Flow matching: sample source distribution
        actions = action_input.action
        x_0 = self._sample_source(actions)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * x_0 + t * actions
        velocity = actions - x_0

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1]:]

        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)
        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        ik_trajectory: torch.Tensor | None = None,
    ) -> BatchFeature:
        """Generate actions via flow matching denoising."""
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device

        if ik_trajectory is not None:
            arm_dim = min(self.config.ik_prior_arm_dim, self.action_dim)
            actions = torch.randn(
                size=(batch_size, self.config.action_horizon, self.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
            )
            ik_traj = ik_trajectory.to(device=device, dtype=vl_embeds.dtype)
            if ik_traj.shape[1] != self.config.action_horizon:
                ik_traj = torch.nn.functional.interpolate(
                    ik_traj.permute(0, 2, 1),
                    size=self.config.action_horizon,
                    mode='linear',
                    align_corners=True,
                ).permute(0, 2, 1)
            actions[:, :, :arm_dim] = ik_traj[:, :, :arm_dim]
        else:
            actions = torch.randn(
                size=(batch_size, self.config.action_horizon, self.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
            )

        dt = 1.0 / self.num_inference_timesteps

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Generate actions (inference)."""
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
