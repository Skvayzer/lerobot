#!/usr/bin/env python
"""
iDP3-style depth encoder: converts raw depth images to point clouds,
then encodes them with a lightweight PointNet into a fixed-length vector
that is injected as an extra observation into the GR00T action head.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DepthEncoderConfig:
    """Configuration for the depth point-cloud encoder."""

    num_views: int = 2  # number of depth cameras
    num_points: int = 512  # points sampled per view
    out_dim: int = 64  # output feature dimension per view
    img_height: int = 224
    img_width: int = 224
    # Simple pinhole intrinsics (can be overridden per-camera later)
    fx: float = 200.0
    fy: float = 200.0
    cx: float = 112.0  # img_width / 2
    cy: float = 112.0  # img_height / 2
    depth_max: float = 3.0  # metres; pixels beyond this are masked


class PointCloudDepthEncoder(nn.Module):
    """
    Per-view pipeline:
      1. Back-project depth pixels → 3-D points (using pinhole intrinsics).
      2. Mask invalid / out-of-range points.
      3. FPS-style random subsample to `num_points`.
      4. Shared PointNet (per-point MLP → max-pool) → `out_dim` vector.

    All views share the same PointNet weights so parameter count stays low.
    Final output is the concatenation of per-view vectors:
        shape = (B, num_views * out_dim)
    """

    def __init__(self, cfg: DepthEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Shared PointNet encoder (input: xyz = 3)
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.out_dim),
        )

        # Pre-compute pixel grid for back-projection (not a parameter)
        v, u = torch.meshgrid(
            torch.arange(cfg.img_height, dtype=torch.float32),
            torch.arange(cfg.img_width, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("_u", u.reshape(-1), persistent=False)  # (H*W,)
        self.register_buffer("_v", v.reshape(-1), persistent=False)

    # ------------------------------------------------------------------
    def _backproject(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Back-project a single depth map to 3-D points.

        Args:
            depth: (B, H, W) float32 depth in metres.

        Returns:
            xyz:  (B, H*W, 3) point cloud.
            mask: (B, H*W) bool — True for valid points.
        """
        cfg = self.cfg
        B = depth.shape[0]
        d = depth.reshape(B, -1)  # (B, N)

        mask = (d > 0) & (d < cfg.depth_max)

        x = (self._u.unsqueeze(0) - cfg.cx) * d / cfg.fx  # (B, N)
        y = (self._v.unsqueeze(0) - cfg.cy) * d / cfg.fy
        z = d

        xyz = torch.stack([x, y, z], dim=-1)  # (B, N, 3)
        return xyz, mask

    def _subsample(self, xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Random subsample to cfg.num_points per batch element.

        Invalid points are replaced with zeros.
        """
        B, N, _ = xyz.shape
        K = self.cfg.num_points
        device = xyz.device

        out = torch.zeros(B, K, 3, device=device, dtype=xyz.dtype)
        for b in range(B):
            valid_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            n_valid = valid_idx.numel()
            if n_valid == 0:
                continue
            if n_valid >= K:
                perm = torch.randperm(n_valid, device=device)[:K]
            else:
                perm = torch.randint(n_valid, (K,), device=device)
            out[b] = xyz[b, valid_idx[perm]]
        return out  # (B, K, 3)

    def _encode_single_view(self, depth: torch.Tensor) -> torch.Tensor:
        """Encode one depth image → (B, out_dim)."""
        xyz, mask = self._backproject(depth)
        pts = self._subsample(xyz, mask)  # (B, K, 3)
        feat = self.point_mlp(pts)  # (B, K, out_dim)
        return feat.max(dim=1).values  # (B, out_dim)

    def forward(self, depths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depths: (B, V, H, W) float32 depth in metres,
                    V == cfg.num_views.

        Returns:
            (B, V * out_dim) concatenated per-view feature vectors.
        """
        B, V = depths.shape[:2]
        feats = [self._encode_single_view(depths[:, v]) for v in range(V)]
        return torch.cat(feats, dim=-1)  # (B, V * out_dim)
