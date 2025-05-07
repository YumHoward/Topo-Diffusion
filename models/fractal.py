# models/fractal.py

import math
import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import ResidualBlock


class FractalFeatureExtractor(nn.Module):
    """Extracts multi-scale fractal features using box-counting and modulation"""

    def __init__(self, num_scales=3, base_scale=16):
        super().__init__()
        self.num_scales = num_scales
        self.base_scale = base_scale

        # Scale-adaptive convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=2 ** i * base_scale + 1,
                      padding=(2 ** i * base_scale) // 2)
            for i in range(num_scales)
        ])

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(num_scales, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU()
        )

    def box_counting(self, x, scale):
        """Perform box-counting on binary images at given scale"""
        b, c, h, w = x.shape
        if scale > h or scale > w:
            return torch.zeros(b, c, device=x.device)

        # Ensure float32 precision
        x = x.to(torch.float32)

        # Adaptive binarization
        median = torch.median(x.view(b, c, -1), dim=2)[0]
        binary = (x > median.view(b, c, 1, 1)).float()

        # Box counting
        x_pool = F.max_pool2d(binary, kernel_size=scale, stride=scale)
        return x_pool.sum(dim=(2, 3))  # [B, C]

    def forward(self, x):
        features = []
        batch_size = x.shape[0]

        for i, conv in enumerate(self.conv_layers):
            scale = 2 ** i * self.base_scale
            with torch.no_grad():
                # Generate candidate scales
                scales = [scale // 4, scale // 2, scale, scale * 2, scale * 4]
                valid_scales = [s for s in scales if s <= x.shape[2] and s <= x.shape[3]]

                if len(valid_scales) < 3:
                    continue

                # Compute log(n) and log(1/s)
                log_n = []
                log_s_inv = []
                for s in valid_scales:
                    count = self.box_counting(x, s)  # [B, 1]
                    log_n.append(torch.log(count.to(torch.float32) + 1e-6))
                    log_s_inv.append(np.log(1 / s))

                # Build regression matrix A [num_scales, 2]
                log_s_inv = torch.tensor(log_s_inv, device=x.device, dtype=torch.float32)
                A = torch.stack([log_s_inv, torch.ones_like(log_s_inv)], dim=1)

                # Stack log_n values [num_scales, batch_size]
                log_n_stack = torch.stack(log_n, dim=0).squeeze(-1)

                # Linear regression to estimate fractal dimension
                fd = torch.linalg.lstsq(A, log_n_stack).solution[0]  # [batch_size]

            # Apply convolution and modulate by fractal dimension
            conv_feat = conv(x.to(torch.float32))  # [B, 1, H, W]
            modulated = conv_feat * fd.view(-1, 1, 1, 1)  # Broadcast
            features.append(modulated)

        return self.fusion(torch.cat(features, dim=1))


class FractalResidualBlock(ResidualBlock):
    """Residual block augmented with fractal feature conditioning"""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__(in_ch, out_ch, t_dim)

        # Fractal feature processing
        self.fractal_mlp = nn.Sequential(
            nn.Conv2d(16, out_ch, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch * 2, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                fractal_feat: torch.Tensor = None) -> torch.Tensor:
        h = super().forward(x, t)

        if fractal_feat is not None:
            f_params = self.fractal_mlp(fractal_feat)
            gamma, beta = torch.chunk(f_params, 2, dim=1)
            h = h * (gamma + 1) + beta

        return h