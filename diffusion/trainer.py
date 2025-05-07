# diffusion/trainer.py
import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from typing import Dict
import torch.nn.functional as F

# Import custom modules
from models.unet import ConfigurableUNet
from data.dataset import PaperCutDataset
from diffusion.scheduler import BetaScheduler
from diffusion.time_embedding import TimeEmbedding


class DiffusionTrainer:
    """Main trainer for the diffusion model"""

    def __init__(self, config: Dict, device: torch.device):
        """
        Args:
            config: Configuration dictionary
            device: Computation device (CPU or GPU)
        """
        self.config = config
        self.device = device

        # Initialize output directories
        self._init_dirs()

        # Model components
        self.model = ConfigurableUNet(**config['model']).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        # Diffusion process components
        self.scheduler = BetaScheduler(
            timesteps=config['timesteps'],
            device=device,
            schedule_type=config['schedule_type']
        )

        # Dataset and DataLoader
        self.dataset = PaperCutDataset(
            img_dir=config['dataset_dir'],
            img_size=config['img_size']
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        # Optimization setup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['lr'],
            total_steps=config['epochs'] * len(self.dataloader),
            pct_start=0.3
        )

        # Training state tracking
        self.epoch_losses = []
        self.epoch_gradient_norms = []

    def _init_dirs(self):
        """Initialize directory structure for outputs"""
        base_dir = os.path.join(
            self.config['output_root'],
            self.config['run_name']
        )
        self.model_dir = os.path.join(base_dir, 'models')
        self.sample_dir = os.path.join(base_dir, 'samples')
        self.loss_plot_dir = os.path.join(base_dir, 'loss_plots')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.loss_plot_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        """Run a single training epoch"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.config['epochs']}")
        for batch in progress_bar:
            # Preprocess data
            x = batch.to(self.device)

            # Sample random timesteps
            t = torch.randint(0, self.config['timesteps'], (x.size(0),), device=self.device)

            # Forward diffusion process
            noise = torch.randn_like(x)
            sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t][:, None, None, None]
            sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            noisy_x = sqrt_alpha * x + sqrt_one_minus_alpha * noise

            # Noise prediction
            pred_noise = self.model(noisy_x, t)

            # Loss computation
            loss = F.mse_loss(pred_noise, noise)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Record metrics
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # Update EMA model
            self._update_ema()
            self.lr_scheduler.step()

        return total_loss / len(self.dataloader)

    def _update_ema(self):
        """Update EMA model weights"""
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.config['ema_decay']).add_(param.data, alpha=1 - self.config['ema_decay'])

    def generate_samples(self, epoch: int, num_samples: int = 4):
        """Generate sample images using DDIM sampling"""
        self.ema_model.eval()
        with torch.no_grad():
            # Start from random noise
            samples = torch.randn(num_samples, 1, self.config['img_size'], self.config['img_size'], device=self.device)

            # Get DDIM step sequence
            timesteps = self.scheduler.get_ddim_timesteps(self.config['ddim_steps'])

            for t, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1):
                # Predict noise
                ts = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                pred_noise = self.ema_model(samples, ts)

                # Compute predicted x0
                alpha_bar = self.scheduler.alphas_cumprod[t]
                alpha_bar_next = self.scheduler.alphas_cumprod[t_next] if t_next >= 0 else 1.0
                pred_x0 = (samples - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

                # Update samples
                samples = torch.sqrt(alpha_bar_next) * pred_x0 + \
                          torch.sqrt(1 - alpha_bar_next - self.config['eta'] ** 2) * pred_noise + \
                          self.config['eta'] * torch.randn_like(samples)

            # Save results
            samples = (samples.clamp(-1, 1) + 1) / 2  # Convert to [0, 1]
            save_path = os.path.join(self.sample_dir, f"epoch_{epoch + 1:04d}.png")
            save_image(samples, save_path, nrow=2, padding=2)

    def save_model(self, epoch: int):
        """Save model checkpoint"""
        save_path = os.path.join(self.model_dir, f"model_epoch_{epoch + 1:04d}.pth")
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'ema_state': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, save_path)

    def model_summary(self) -> str:
        """Return model architecture summary string"""
        from utils.helpers import model_summary  # Lazy import to avoid circular dependency
        return model_summary(self.model)