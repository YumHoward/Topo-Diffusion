# models/blocks.py

import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.graph import GraphBuilder


class ResidualBlock(nn.Module):
    """Basic residual block with time conditioning"""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.out_ch = out_ch

        # Time conditioning network
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, out_ch * 2)
        )

        # Main convolutional path
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)

        # Shortcut connection
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Generate time-conditioned parameters
        t_params = self.mlp(t).view(B, -1, 1, 1)
        gamma = t_params[:, :self.out_ch] + 1  # Scale parameter
        beta = t_params[:, self.out_ch:]       # Shift parameter

        # Forward through main path
        h = self.conv1(x)
        h = self.norm1(h)
        h = h * gamma + beta
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return h + self.shortcut(x)


class GCNResidualBlock(nn.Module):
    """Residual block enhanced with GCN for global context"""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()

        # Regular convolution branch
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

        # GCN branch
        self.gcn = GCNConv(in_ch + 2, out_ch)  # +2 for coordinate features
        self.gcn_norm = nn.LayerNorm(out_ch)

        # Gated fusion layer
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
            nn.Sigmoid()
        )

        # Time conditioning projection
        self.time_proj = nn.Linear(t_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Convolutional features
        conv_feat = self.conv_path(x)

        # Build graph structure
        graph_builder = GraphBuilder(patch_size=16)
        nodes, edge_index = graph_builder.build(x)

        # GCN processing
        gcn_feat = self.gcn(nodes, edge_index)
        gcn_feat = self.gcn_norm(gcn_feat)
        gcn_feat = gcn_feat.view(B, -1, H // 16, W // 16)
        gcn_feat = F.interpolate(gcn_feat, (H, W), mode='bilinear')

        # Gated fusion
        combined = torch.cat([conv_feat, gcn_feat], dim=1)
        gate = self.gate(combined)
        merged = gate * conv_feat + (1 - gate) * gcn_feat

        # Apply time conditioning
        t_params = self.time_proj(t).view(B, -1, 1, 1)
        gamma, beta = torch.chunk(t_params, 2, dim=1)

        return merged * (gamma + 1) + beta