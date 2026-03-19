"""
iDP3-style egocentric point cloud encoder for depth images.

Architecture follows PointVLA (arXiv:2503.07511) which uses the iDP3 encoder:
- Hierarchical 1D conv layers extract multi-level features
- Max pooling between layers progressively reduces point density
- Global max pool at EACH level, concatenate → multi-scale representation
- This captures both low-level geometry (edges) and high-level structure (objects)

Reference: iDP3 (Ze et al., IROS 2025), PointVLA (Li et al., 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class DepthEncoderConfig:
    num_views: int = 3
    num_points_per_view: int = 512
    input_height: int = 480
    input_width: int = 640
    conv_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    output_dim: int = 256
    # Intel D405 approximate intrinsics (encoder is robust to small errors)
    default_fx: float = 430.0
    default_fy: float = 430.0
    default_cx: float = 320.0
    default_cy: float = 240.0


class PointCloudDepthEncoder(nn.Module):
    """
    Multi-scale egocentric point cloud encoder.

    Pipeline per view:
      1. Depth image (B, 1, H, W) → back-project to 3D in camera frame
      2. Random subsample to N points, filter invalid (zero depth)
      3. Center points (subtract centroid for translation invariance)
      4. Multi-scale 1D conv pyramid with progressive density reduction:
         Conv(3→64) → pool → Conv(64→128) → pool → Conv(128→256) → pool
      5. Global max pool EACH layer → concat → multi-scale feature
    All views combined → Linear → (B, output_dim)

    Total params: ~0.6M
    """

    def __init__(self, config: DepthEncoderConfig):
        super().__init__()
        self.config = config
        self.num_views = config.num_views
        self.num_points = config.num_points_per_view
        self.output_dim = config.output_dim

        # Multi-scale 1D conv pyramid (iDP3/PointVLA architecture)
        self.conv_layers = nn.ModuleList()
        in_ch = 3
        for out_ch in config.conv_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        # Multi-scale feature dim: sum of ALL channel dims
        # e.g., [64, 128, 256] → 64 + 128 + 256 = 448
        total_feat_dim = sum(config.conv_channels)

        # Combine all views → output_dim
        self.view_combiner = nn.Sequential(
            nn.Linear(total_feat_dim * config.num_views, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, config.output_dim),
        )

        # Default intrinsics as buffer
        self.register_buffer(
            "default_intrinsics",
            torch.tensor([config.default_fx, config.default_fy,
                          config.default_cx, config.default_cy]),
            persistent=False,
        )

        self._pixel_grid_cache: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_pixel_grid(self, H: int, W: int, device: torch.device):
        cache_key = (H, W, str(device))
        if cache_key not in self._pixel_grid_cache:
            u = torch.arange(W, device=device, dtype=torch.float32)
            v = torch.arange(H, device=device, dtype=torch.float32)
            grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
            self._pixel_grid_cache[cache_key] = (grid_u, grid_v)
        return self._pixel_grid_cache[cache_key]

    def depth_to_pointcloud(self, depth, fx, fy, cx, cy):
        """Back-project depth (B, 1, H, W) to point cloud (B, N, 3) in camera frame."""
        B, _, H, W = depth.shape
        device = depth.device
        grid_u, grid_v = self._get_pixel_grid(H, W, device)

        z = depth[:, 0, :, :]  # (B, H, W)
        x = (grid_u.unsqueeze(0) - cx) * z / max(fx, 1e-6)
        y = (grid_v.unsqueeze(0) - cy) * z / max(fy, 1e-6)

        points = torch.stack([x, y, z], dim=-1).reshape(B, H * W, 3)
        valid_mask = z.reshape(B, H * W) > 0.01

        sampled = []
        for b in range(B):
            valid_idx = valid_mask[b].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                sampled.append(torch.zeros(self.num_points, 3, device=device))
                continue
            if valid_idx.numel() >= self.num_points:
                perm = torch.randperm(valid_idx.numel(), device=device)[:self.num_points]
                idx = valid_idx[perm]
            else:
                idx = valid_idx[torch.randint(valid_idx.numel(), (self.num_points,), device=device)]
            sampled.append(points[b, idx, :])

        return torch.stack(sampled, dim=0)

    def encode_single_view(self, points):
        """
        Multi-scale point cloud encoding following iDP3/PointVLA.

        Features from EACH conv layer are globally max-pooled and concatenated.
        This gives multi-level 3D representation: low-level edges (layer 1)
        + mid-level surfaces (layer 2) + high-level objects (layer 3).
        Progressive max_pool1d between layers reduces point density.
        """
        centroid = points.mean(dim=1, keepdim=True)
        x = (points - centroid).transpose(1, 2)  # (B, 3, N)

        multi_scale_features = []
        for conv in self.conv_layers:
            x = conv(x)  # (B, C_i, N_i)
            level_feat = x.max(dim=2).values  # (B, C_i) — global max pool this level
            multi_scale_features.append(level_feat)
            # Progressive density reduction
            if x.shape[2] > 1:
                x = F.max_pool1d(x, kernel_size=2, ceil_mode=True)

        return torch.cat(multi_scale_features, dim=1)  # (B, sum(conv_channels))

    def forward(self, depth_images, intrinsics=None):
        """
        Process multiple depth views into a single feature vector.

        Args:
            depth_images: list of (B, 1, H, W) or (B, H, W) depth tensors
                          OR (B, V, H, W) stacked tensor
            intrinsics: optional list of (fx, fy, cx, cy) per view

        Returns:
            (B, output_dim) feature vector for DiT state conditioning
        """
        # Handle stacked tensor input (B, V, H, W)
        if isinstance(depth_images, torch.Tensor):
            if depth_images.dim() == 4:
                # (B, V, H, W) → list of (B, 1, H, W)
                V = min(self.num_views, depth_images.shape[1])
                depth_images = [depth_images[:, i:i+1, :, :] for i in range(V)]
            elif depth_images.dim() == 3:
                # (B, H, W) → single view
                depth_images = [depth_images.unsqueeze(1)]

        if intrinsics is None:
            fx, fy, cx, cy = self.default_intrinsics.tolist()
            intrinsics = [(fx, fy, cx, cy)] * len(depth_images)

        view_features = []
        for i, depth in enumerate(depth_images):
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            depth = depth.float()
            if depth.max() > 100.0:
                depth = depth / depth.max().clamp(min=1.0)

            _, _, H, W = depth.shape
            cfg_H, cfg_W = self.config.input_height, self.config.input_width
            if H != cfg_H or W != cfg_W:
                depth = F.interpolate(depth, size=(cfg_H, cfg_W), mode='nearest')
                sx, sy = cfg_W / W, cfg_H / H
                ofx, ofy, ocx, ocy = intrinsics[i]
                intrinsics[i] = (ofx * sx, ofy * sy, ocx * sx, ocy * sy)

            fx_i, fy_i, cx_i, cy_i = intrinsics[i]
            points = self.depth_to_pointcloud(depth, fx_i, fy_i, cx_i, cy_i)
            feat = self.encode_single_view(points)
            view_features.append(feat)

        combined = torch.cat(view_features, dim=1)
        return self.view_combiner(combined)
