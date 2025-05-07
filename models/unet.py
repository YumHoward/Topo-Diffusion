# models/unet.py

import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import torch.nn as nn
from typing import List

# Import custom modules
from models.blocks import ResidualBlock, GCNResidualBlock
from models.fractal import FractalFeatureExtractor, FractalResidualBlock
from models.graph import GraphBuilder
from diffusion.time_embedding import TimeEmbedding


class ConfigurableUNet(nn.Module):
    """Configurable UNet architecture supporting multiple block types"""

    def __init__(
        self,
        img_size: int = 256,
        down_blocks: List[str] = ['residual'] * 5,
        mid_block: str = 'residual',
        up_blocks: List[str] = ['residual'] * 4,
        channels: List[int] = [64, 128, 256, 512, 1024],
        t_dim: int = 256,
        fractal: dict = {'num_scales': 3, 'base_scale': 16},
        gcn: dict = {'patch_size': 16, 'neighbor': 4}
    ):
        super().__init__()
        self.img_size = img_size
        self.channels = channels

        # Time embedding network
        self.time_mlp = TimeEmbedding(t_dim)

        # Fractal feature extractor (only if fractal blocks are used)
        self.fractal_extractor = None
        if 'fractal' in down_blocks + [mid_block] + up_blocks:
            self.fractal_extractor = FractalFeatureExtractor(**fractal)

        # Down-sampling path
        self.down_path = nn.ModuleList()
        in_chs = [1] + channels[:-1]
        for i, (in_ch, out_ch, block_type) in enumerate(zip(in_chs, channels, down_blocks)):
            self.down_path.append(self._build_down_block(in_ch, out_ch, block_type, i))

        # Middle block
        self.mid_block = self._build_mid_block(channels[-1], mid_block)
        # Additional middle convolution layer
        self.mid_conv = nn.Conv2d(channels[-1], channels[-1], 3, padding=1)

        # Up-sampling path
        self.up_path = nn.ModuleList()
        reversed_chs = channels[::-1]
        for i, block_type in enumerate(up_blocks):
            in_ch = reversed_chs[i]
            out_ch = reversed_chs[i + 1] if i < len(reversed_chs) - 1 else reversed_chs[-1]
            self.up_path.append(self._build_up_block(in_ch, out_ch, block_type, i))

        # Final output layer
        self.final_conv = nn.Conv2d(channels[0], 1, 3, padding=1)

    def _build_down_block(self, in_ch: int, out_ch: int, block_type: str, stage: int) -> nn.Module:
        """Build a down-sampling block"""
        layers = nn.ModuleList()
        # First stage does not downsample
        stride = 2 if stage > 0 else 1
        layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1))
        layers.append(self._get_block(block_type, out_ch, out_ch))
        return layers

    def _build_mid_block(self, ch: int, block_type: str) -> nn.Module:
        """Build the middle block"""
        return self._get_block(block_type, ch, ch)

    def _build_up_block(self, in_ch: int, out_ch: int, block_type: str, stage: int) -> nn.Module:
        """Build an up-sampling block"""
        layers = nn.ModuleList()
        layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
        layers.append(self._get_block(block_type, out_ch * 2, out_ch))  # With skip connection
        return layers

    def _get_block(self, block_type: str, in_ch: int, out_ch: int) -> nn.Module:
        """Dispatcher for different block types"""
        if block_type == 'residual':
            return ResidualBlock(in_ch, out_ch, self.time_mlp.dim)
        elif block_type == 'gcn':
            return GCNResidualBlock(in_ch, out_ch, self.time_mlp.dim)
        elif block_type == 'fractal':
            return FractalResidualBlock(in_ch, out_ch, self.time_mlp.dim)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_mlp(t)

        # Fractal feature extraction
        fractal_feat = None
        if self.fractal_extractor is not None:
            fractal_feat = self.fractal_extractor(x)

        # Down-sampling path
        skips = []
        for i, stage in enumerate(self.down_path):
            x = stage[0](x)
            if isinstance(stage[1], FractalResidualBlock):
                x = stage[1](x, t_emb, fractal_feat)
            else:
                x = stage[1](x, t_emb)
            # Save features for skip connections (except last one)
            if i < len(self.down_path) - 1:
                skips.append(x)

        # Middle block
        if isinstance(self.mid_block, FractalResidualBlock):
            x = self.mid_block(x, t_emb, fractal_feat)
        else:
            x = self.mid_block(x, t_emb)
        # Apply middle convolution
        x = self.mid_conv(x)

        # Up-sampling path
        for stage in self.up_path:
            x = stage[0](x)
            skip = skips.pop() if skips else None
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            if isinstance(stage[1], FractalResidualBlock):
                x = stage[1](x, t_emb, fractal_feat)
            else:
                x = stage[1](x, t_emb)

        return self.final_conv(x)