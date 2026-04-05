"""ACT-CraftNet Policy — lerobot integration.

Wraps ACTCraftNet (bimanual G1+Dex3, 28D) as a lerobot PreTrainedPolicy.

Batch key mapping (lerobot → internal):
  observation.state              → state      (B, 28)
  observation.tactile            → tactile    (B, 18)
  observation.environment_state  → env_state  (B, 9)
  observation.images.*           → images     (B, n_cam, 3, 224, 224)  stacked in config order
  observation.depth.*            → depths     (B, n_views, 1, H, W)    stacked in config order
  action                         → action     (B, chunk, 28)
  action_is_pad                  → action_is_pad

Architecture internals — see _ACTCraftNet and related classes below.
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.act_craftnet.configuration_act_craftnet import ACTCraftNetConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


# ──────────────────────────────────────────────────────────────────────────────
# Policy wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ACTCraftNetPolicy(PreTrainedPolicy):
    """ACT-CraftNet as a lerobot PreTrainedPolicy.

    Extends standard ACT with FrozenDINOv2 visual backbone, CNN depth encoder,
    and System0 MoE reactive finger correction with bidirectional S1↔S0 links.
    """

    config_class = ACTCraftNetConfig
    name = "act_craftnet"

    def __init__(self, config: ACTCraftNetConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Build internal model config from lerobot config
        n_cameras = len(config.image_features)
        n_depth_views = len(config.depth_feature_keys)

        mcfg = _ModelCfg(
            chunk_size=config.chunk_size,
            dim_model=config.dim_model,
            n_heads=config.n_heads,
            dim_ff=config.dim_feedforward,
            n_enc_layers=config.n_enc_layers,
            n_dec_layers=config.n_dec_layers,
            latent_dim=config.latent_dim,
            kl_weight=config.kl_weight,
            dropout=config.dropout,
            n_cameras=n_cameras,
            n_depth_views=n_depth_views,
            physical_intent_dim=config.physical_intent_dim,
            feedback_dim=config.feedback_dim,
            depth_max=config.depth_max,
        )
        self.model = _ACTCraftNet(mcfg)
        self._mcfg = mcfg

        # Image key order (fixed once at init — determines stacking order)
        self._image_keys = sorted(config.image_features.keys())
        self._depth_keys = config.depth_feature_keys

        self._action_queue: deque = deque([], maxlen=config.n_action_steps)

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _build_model_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Map lerobot batch keys to the internal model batch format."""
        mb: dict[str, Tensor] = {}

        # State (28D)
        mb["state"] = batch[OBS_STATE].float()

        # Tactile (18D) — from dedicated key or zero-fallback
        tac_key = self.config.tactile_feature_key
        if tac_key in batch:
            mb["tactile"] = batch[tac_key].float()
        else:
            B = mb["state"].shape[0]
            mb["tactile"] = torch.zeros(B, 18, device=mb["state"].device)

        # Environment state (9D)
        if OBS_ENV_STATE in batch:
            mb["env_state"] = batch[OBS_ENV_STATE].float()
        else:
            B = mb["state"].shape[0]
            mb["env_state"] = torch.zeros(B, 9, device=mb["state"].device)

        # RGB images → (B, n_cam, 3, 224, 224)
        imgs = torch.stack([batch[k].float() for k in self._image_keys], dim=1)
        mb["images"] = imgs

        # Depth images → (B, n_views, 1, H, W)
        depth_list = []
        for k in self._depth_keys:
            if k in batch:
                d = batch[k].float()        # (B, 1, H, W) or (B, H, W)
                if d.dim() == 3:
                    d = d.unsqueeze(1)      # ensure (B, 1, H, W)
                depth_list.append(d)
            else:
                B = mb["state"].shape[0]
                depth_list.append(torch.zeros(B, 1, 120, 160, device=mb["state"].device))
        mb["depths"] = torch.stack(depth_list, dim=1)   # (B, n_views, 1, H, W)

        # Actions (training only)
        if ACTION in batch:
            mb["action"] = batch[ACTION].float()
        if "action_is_pad" in batch:
            mb["action_is_pad"] = batch["action_is_pad"]

        return mb

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return the full action chunk (B, chunk_size, 28) for a given observation."""
        self.eval()
        mb = self._build_model_batch(batch)
        actions, _, _ = self.model(mb)   # (B, chunk, 28)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return a single (B, action_dim) action for environment execution."""
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)   # (B, chunk, 28)
            for t in range(self.config.n_action_steps):
                self._action_queue.append(actions[:, t])
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training/validation forward. Returns (loss, loss_dict)."""
        mb = self._build_model_batch(batch)
        actions_hat, mu, log_sigma = self.model(mb)

        gt = mb["action"]
        pad_mask = mb.get("action_is_pad", torch.zeros(gt.shape[:2], dtype=torch.bool, device=gt.device))
        mask = (~pad_mask).unsqueeze(-1).float()

        l1 = (F.l1_loss(actions_hat, gt, reduction="none") * mask).sum() / mask.sum()

        loss_dict = {"l1_loss": l1.item()}

        kl = torch.tensor(0.0, device=l1.device)
        if mu is not None and log_sigma is not None:
            kl = (-0.5 * (1 + log_sigma - mu.pow(2) - log_sigma.exp())).sum(-1).mean()
            loss_dict["kl_loss"] = kl.item()

        loss = l1 + self.config.kl_weight * kl
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def get_optim_params(self) -> list[dict]:
        # DINOv2 params are frozen (requires_grad=False) — excluded automatically
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]


# ──────────────────────────────────────────────────────────────────────────────
# Internal model config (not a lerobot config — just a plain object)
# ──────────────────────────────────────────────────────────────────────────────

class _ModelCfg:
    state_dim:    int   = 28
    action_dim:   int   = 28
    tactile_dim:  int   = 18
    env_dim:      int   = 9
    finger_dim:   int   = 14
    depth_h:      int   = 120
    depth_w:      int   = 160

    # Overridable
    chunk_size:           int   = 50
    latent_dim:           int   = 32
    dim_model:            int   = 512
    n_heads:              int   = 8
    dim_ff:               int   = 3200
    n_enc_layers:         int   = 4
    n_dec_layers:         int   = 7
    dropout:              float = 0.1
    kl_weight:            float = 1.0
    n_cameras:            int   = 3
    n_depth_views:        int   = 3
    physical_intent_dim:  int   = 128
    feedback_dim:         int   = 64
    depth_max:            float = 2.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sinusoidal_embed(seq_len: int, dim: int, device) -> Tensor:
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    i   = torch.arange(0, dim, 2, device=device).float()
    div = torch.exp(i * (-math.log(10000.0) / dim))
    pe  = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, seq, dim)


class _EncoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=False)
        self.ff   = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.dp1  = nn.Dropout(dropout)
        self.dp2  = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.n1(x + self.dp1(a))
        x = self.n2(x + self.dp2(self.ff(x)))
        return x


class _DecoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, dropout):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=False)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d)
        self.dp1 = nn.Dropout(dropout); self.dp2 = nn.Dropout(dropout); self.dp3 = nn.Dropout(dropout)

    def forward(self, tgt, mem, mem_key_padding_mask=None):
        a, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.n1(tgt + self.dp1(a))
        a, _ = self.cross_attn(tgt, mem, mem, key_padding_mask=mem_key_padding_mask)
        tgt = self.n2(tgt + self.dp2(a))
        tgt = self.n3(tgt + self.dp3(self.ff(tgt)))
        return tgt


# ──────────────────────────────────────────────────────────────────────────────
# FrozenDINOv2 visual encoder
# ──────────────────────────────────────────────────────────────────────────────

class _FrozenDINOv2(nn.Module):
    """DINOv2 ViT-B/14 frozen backbone. All 86.6M params frozen."""
    def __init__(self, out_dim: int = 512):
        super().__init__()
        from transformers import Dinov2Model
        self.dino = Dinov2Model.from_pretrained("facebook/dinov2-base")
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(768, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, 3, 224, 224) → (B, out_dim)"""
        with torch.no_grad():
            out = self.dino(pixel_values=x)
        patch_tokens = out.last_hidden_state[:, 1:]   # (B, 256, 768)
        return self.proj(patch_tokens.mean(dim=1))    # (B, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# DepthCNNEncoder
# ──────────────────────────────────────────────────────────────────────────────

class _DepthCNNEncoder(nn.Module):
    """2D CNN depth encoder. No intrinsics required.

    Input:  (B, n_views, 1, H, W) float32 depth in metres
    Output: (B, 256)
    """
    def __init__(self, n_views: int = 3, out_dim: int = 256, depth_max: float = 2.0):
        super().__init__()
        self.n_views   = n_views
        self.depth_max = depth_max
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(
            nn.Linear(n_views * 448, 512), nn.LayerNorm(512), nn.ReLU(True),
            nn.Linear(512, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, depths: Tensor) -> Tensor:
        view_feats = []
        for v in range(self.n_views):
            d = depths[:, v].clamp(0, self.depth_max) / self.depth_max  # (B,1,H,W) → [0,1]
            x1 = self.conv1(d); x2 = self.conv2(x1); x3 = self.conv3(x2)
            view_feats.append(torch.cat([
                self.gap(x1).flatten(1),   # (B, 64)
                self.gap(x2).flatten(1),   # (B, 128)
                self.gap(x3).flatten(1),   # (B, 256)
            ], dim=-1))                    # (B, 448)
        return self.fusion(torch.cat(view_feats, dim=-1))  # (B, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# System 0: MoE reactive finger correction
# ──────────────────────────────────────────────────────────────────────────────

class _System0Policy(nn.Module):
    """MoE reactive finger corrector with bidirectional S1↔S0 links.

    Two-phase:
      encode_feedback(tactile, finger_state) → tactile_feedback (B, feedback_dim)   [before encoder]
      forward_delta(enc_out[0], intent, tactile, finger_state, s1_fingers)
                                             → delta_finger (B, T, 14)               [after decoder]

    Expert input: enc_out[0](512) + intent(128) + tactile(18) + finger_state(14) + s1_fingers(14) = 686D
    """
    N_EXPERTS     = 4
    TOP_K         = 2
    FINGER_DIM    = 14
    EXPERT_HIDDEN = 256

    def __init__(self, hidden_dim=512, physical_intent_dim=128, tactile_dim=18, feedback_dim=64):
        super().__init__()
        expert_in = hidden_dim + physical_intent_dim + tactile_dim + self.FINGER_DIM + self.FINGER_DIM  # 686
        self.router  = nn.Linear(expert_in, self.N_EXPERTS)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_in, self.EXPERT_HIDDEN), nn.ReLU(True),
                          nn.Linear(self.EXPERT_HIDDEN, self.FINGER_DIM))
            for _ in range(self.N_EXPERTS)
        ])
        self.feedback_encoder = nn.Sequential(
            nn.Linear(tactile_dim + self.FINGER_DIM, 128), nn.ReLU(True),
            nn.Linear(128, feedback_dim),
        )
        self._init_near_zero()

    def _init_near_zero(self):
        for exp in self.experts:
            nn.init.normal_(exp[-1].weight, std=0.01); nn.init.zeros_(exp[-1].bias)
        nn.init.normal_(self.feedback_encoder[-1].weight, std=0.01)
        nn.init.zeros_(self.feedback_encoder[-1].bias)

    def encode_feedback(self, tactile: Tensor, finger_state: Tensor) -> Tensor:
        """Phase 1 (before encoder). Returns (B, feedback_dim)."""
        return self.feedback_encoder(torch.cat([tactile, finger_state], dim=-1))

    def forward_delta(self, hidden_state: Tensor, physical_intent: Tensor,
                      tactile: Tensor, finger_state: Tensor, s1_fingers: Tensor) -> Tensor:
        """Phase 2 (after decoder). Returns delta_finger (B, T, 14)."""
        T = s1_fingers.shape[1]
        h  = hidden_state.unsqueeze(1).expand(-1, T, -1)
        pi = physical_intent.unsqueeze(1).expand(-1, T, -1)
        ta = tactile.unsqueeze(1).expand(-1, T, -1)
        fs = finger_state.unsqueeze(1).expand(-1, T, -1)
        ctx = torch.cat([h, pi, ta, fs, s1_fingers], dim=-1)  # (B, T, 686)

        logits              = self.router(ctx)
        topk_vals, topk_idx = logits.topk(self.TOP_K, dim=-1)
        gates               = F.softmax(topk_vals, dim=-1)      # (B, T, TOP_K)

        B = s1_fingers.shape[0]
        ctx_flat = ctx.reshape(B * T, -1)
        all_out = torch.stack(
            [exp(ctx_flat).reshape(B, T, self.FINGER_DIM) for exp in self.experts], dim=2
        )                                                        # (B, T, N, 14)
        idx_exp  = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.FINGER_DIM)
        selected = all_out.gather(2, idx_exp)                    # (B, T, TOP_K, 14)
        return (selected * gates.unsqueeze(-1)).sum(2)           # (B, T, 14)


# ──────────────────────────────────────────────────────────────────────────────
# ACTCraftNet core model
# ──────────────────────────────────────────────────────────────────────────────

class _ACTCraftNet(nn.Module):
    """Core ACT-CraftNet model (not a lerobot policy — used inside ACTCraftNetPolicy)."""

    def __init__(self, cfg: _ModelCfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim_model

        # Visual backbone: FrozenDINOv2 (shared across all cameras)
        self.dino_cam = _FrozenDINOv2(out_dim=D)

        # Depth encoder: CNN 2D (3 views → 1 token)
        self.depth_encoder = _DepthCNNEncoder(n_views=cfg.n_depth_views, out_dim=256, depth_max=cfg.depth_max)
        self.depth_proj    = nn.Linear(256, D)

        # System 0: MoE reactive finger correction
        self.system0 = _System0Policy(
            hidden_dim=D, physical_intent_dim=cfg.physical_intent_dim,
            tactile_dim=cfg.tactile_dim, feedback_dim=cfg.feedback_dim,
        )

        # S1→S0: enc_out[0] (512D) → physical_intent (128D)
        self.physical_intent_proj = nn.Linear(D, cfg.physical_intent_dim)

        # S0→S1: tactile_feedback (64D) → encoder token (512D), near-zero init
        self.tactile_feedback_proj = nn.Linear(cfg.feedback_dim, D)
        nn.init.normal_(self.tactile_feedback_proj.weight, std=0.01)
        nn.init.zeros_(self.tactile_feedback_proj.bias)

        # Low-dim projections
        self.state_proj  = nn.Linear(cfg.state_dim, D)
        self.env_proj    = nn.Linear(cfg.env_dim, D)
        self.latent_proj = nn.Linear(cfg.latent_dim, D)

        # VAE encoder: [CLS | state | actions…]
        n_vae = 1 + 1 + cfg.chunk_size
        self.vae_cls_embed   = nn.Embedding(1, D)
        self.vae_state_proj  = nn.Linear(cfg.state_dim, D)
        self.vae_action_proj = nn.Linear(cfg.action_dim, D)
        self.register_buffer("vae_pos_enc",
            _sinusoidal_embed(n_vae, D, torch.device("cpu")).squeeze(0))
        vae_enc_layer = nn.TransformerEncoderLayer(D, cfg.n_heads, cfg.dim_ff, cfg.dropout,
                                                    batch_first=False, norm_first=False)
        self.vae_encoder     = nn.TransformerEncoder(vae_enc_layer, num_layers=4)
        self.vae_latent_proj = nn.Linear(D, cfg.latent_dim * 2)

        # Main encoder: 8 tokens [z|state|tac_fb|env|depth|cam0|cam1|cam2]
        n_enc_tokens = 5 + cfg.n_cameras
        self.register_buffer("enc_pos_enc",
            _sinusoidal_embed(n_enc_tokens, D, torch.device("cpu")).squeeze(0))
        enc_layer = nn.TransformerEncoderLayer(D, cfg.n_heads, cfg.dim_ff, cfg.dropout,
                                                batch_first=False, norm_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_enc_layers)

        # Main decoder
        self.query_embed  = nn.Embedding(cfg.chunk_size, D)
        self.decoder      = nn.ModuleList([
            _DecoderLayer(D, cfg.n_heads, cfg.dim_ff, cfg.dropout)
            for _ in range(cfg.n_dec_layers)
        ])
        self.decoder_norm = nn.LayerNorm(D)
        self.action_head  = nn.Linear(D, cfg.action_dim)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "dino_cam.dino" in name:
                continue
            if "system0" in name or "tactile_feedback_proj" in name:
                continue
            if p.dim() > 1 and "embed" not in name:
                nn.init.xavier_uniform_(p)

    def _encode_vae(self, state, action_chunk, action_is_pad):
        B, D = state.shape[0], self.cfg.dim_model
        device = state.device
        cls  = self.vae_cls_embed.weight.unsqueeze(0).expand(B, 1, -1)
        st   = self.vae_state_proj(state).unsqueeze(1)
        acts = self.vae_action_proj(action_chunk)
        seq  = torch.cat([cls, st, acts], dim=1) + self.vae_pos_enc.unsqueeze(0)
        pad  = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=device), action_is_pad], dim=1)
        out  = self.vae_encoder(seq.permute(1, 0, 2), src_key_padding_mask=pad)[0]
        params    = self.vae_latent_proj(out)
        mu        = params[:, :self.cfg.latent_dim]
        log_sigma = params[:, self.cfg.latent_dim:]
        z = mu + log_sigma.div(2).exp() * torch.randn_like(mu)
        return mu, log_sigma, z

    def _finger_state(self, state: Tensor) -> Tensor:
        return torch.cat([state[:, 7:14], state[:, 21:28]], dim=-1)  # (B, 14)

    def forward(self, batch: dict) -> tuple:
        """Returns (actions_hat (B,chunk,28), mu, log_sigma)."""
        device    = batch["state"].device
        B         = batch["state"].shape[0]
        D         = self.cfg.dim_model
        cfg       = self.cfg

        state     = batch["state"].float()
        tactile   = batch["tactile"].float()
        env_state = batch["env_state"].float()
        images    = batch["images"].float()
        depths    = batch["depths"].float()

        # CVAE encode (training only)
        if self.training and "action" in batch:
            mu, log_sigma, z = self._encode_vae(state, batch["action"].float(), batch["action_is_pad"])
        else:
            mu = log_sigma = None
            z  = torch.zeros(B, cfg.latent_dim, device=device)

        # S0 Phase 1: tactile feedback token (before encoder)
        finger_state     = self._finger_state(state)
        tactile_feedback = self.system0.encode_feedback(tactile, finger_state)   # (B, 64)
        tac_fb_token     = self.tactile_feedback_proj(tactile_feedback)           # (B, D)

        # DINOv2 camera features
        imgs_flat      = images.reshape(B * cfg.n_cameras, 3, 224, 224)
        cam_feats_flat = self.dino_cam(imgs_flat)                                  # (B*n, D)
        cam_feats      = cam_feats_flat.reshape(B, cfg.n_cameras, D)

        # Depth token
        depth_tok = self.depth_proj(self.depth_encoder(depths))                   # (B, D)

        # 8-token encoder sequence: [z|state|tac_fb|env|depth|cam0|cam1|cam2]
        tokens = [
            self.latent_proj(z),
            self.state_proj(state),
            tac_fb_token,
            self.env_proj(env_state),
            depth_tok,
        ] + [cam_feats[:, ci] for ci in range(cfg.n_cameras)]

        seq = torch.stack(tokens, dim=0)             # (8, B, D)
        seq = seq + self.enc_pos_enc.unsqueeze(1)
        mem = self.encoder(seq)                       # (8, B, D)

        # S0 Phase 2a: physical intent from enc_out[0]
        physical_intent = self.physical_intent_proj(mem[0])  # (B, 128)

        # Decode
        queries = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)  # (chunk, B, D)
        out = queries
        for layer in self.decoder:
            out = layer(out, mem)
        out = self.decoder_norm(out).permute(1, 0, 2)  # (B, chunk, D)
        actions_hat = self.action_head(out)             # (B, chunk, 28)

        # S0 Phase 2b: per-timestep residual finger correction
        s1_fingers = torch.cat([actions_hat[:, :, 7:14], actions_hat[:, :, 21:28]], dim=-1)  # (B,T,14)
        delta_finger = self.system0.forward_delta(
            mem[0], physical_intent, tactile, finger_state, s1_fingers
        )                                                                           # (B, T, 14)

        actions_hat = actions_hat.clone()
        actions_hat[:, :, 7:14]  = actions_hat[:, :, 7:14]  + delta_finger[:, :, :7]
        actions_hat[:, :, 21:28] = actions_hat[:, :, 21:28] + delta_finger[:, :, 7:]

        return actions_hat, mu, log_sigma
