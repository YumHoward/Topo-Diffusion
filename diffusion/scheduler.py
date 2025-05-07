# diffusion/scheduler.py

import math
import torch


class BetaScheduler:
    """Manages timestep noise scheduling for the diffusion process"""

    def __init__(self, timesteps: int = 1000, device: str = 'cpu', schedule_type: str = 'cosine'):
        """
        Args:
            timesteps: Total number of diffusion steps
            device: Computation device
            schedule_type: Type of noise schedule ['cosine', 'linear']
        """
        self.timesteps = timesteps
        self.device = device
        self.schedule_type = schedule_type

        # Precompute schedule and related parameters
        self.betas = self._create_schedule()
        self._precompute_parameters()

    def _create_schedule(self) -> torch.Tensor:
        """Generates beta schedule based on selected strategy"""
        if self.schedule_type == 'cosine':
            return self._cosine_schedule()
        elif self.schedule_type == 'linear':
            return self._linear_schedule()
        else:
            raise ValueError(f"Unsupported schedule type: {self.schedule_type}")

    def _cosine_schedule(self) -> torch.Tensor:
        """Cosine noise scheduling strategy"""
        steps = self.timesteps
        x = torch.linspace(0, steps, steps + 1)
        alphas = torch.cos((x / steps + 0.01) * math.pi / 2) ** 2
        alphas_cumprod = alphas / alphas[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.02).to(self.device)

    def _linear_schedule(self) -> torch.Tensor:
        """Linear noise scheduling strategy"""
        return torch.linspace(0.0001, 0.02, self.timesteps).to(self.device)

    def _precompute_parameters(self):
        """Precompute diffusion-related parameters"""
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def get_ddim_timesteps(self, ddim_steps: int) -> list:
        """Generate time steps for DDIM sampling (with stride)"""
        c = self.timesteps // ddim_steps
        return list(reversed(range(0, self.timesteps, c)))