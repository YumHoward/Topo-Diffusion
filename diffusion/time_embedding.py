# diffusion/time_embedding.py

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Encodes discrete timestep information into feature vectors"""

    def __init__(self, dim: int = 256):
        """
        Args:
            dim: Output embedding dimension
        """
        super().__init__()
        self.dim = dim

        # Time embedding network
        self.proj = nn.Sequential(
            nn.Linear(1, dim // 2),
            nn.SiLU(),  # Activation function
            nn.Linear(dim // 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor [batch_size]
        Returns:
            Embedded vectors [batch_size, dim]
        """
        return self.proj(t.float().view(-1, 1))