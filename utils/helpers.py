# utils/helpers.py

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torchvision
from torch import nn
from torchvision.utils import make_grid


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"training_{timestamp}.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def save_loss_plot(
        train_losses: List[float],
        val_losses: List[float] = None,
        output_dir: str = "loss_plots",
        filename: str = "loss_curve.png"
) -> None:
    """保存训练损失曲线图"""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(train_losses, label="Training Loss", color="blue")

    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss", color="red")

    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved loss plot at: {save_path}")


def model_summary(model: nn.Module) -> str:
    """生成模型参数统计摘要"""
    summary = []
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        summary.append(f"{name:<60} | {num_params:>10,} | {param.requires_grad}")
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    summary_str = "\n".join(summary)
    return (
        f"Model Architecture Summary:\n"
        f"{'Layer':<60} | {'Params':>10} | Trainable\n"
        f"{'-' * 80}\n"
        f"{summary_str}\n"
        f"{'-' * 80}\n"
        f"Total Parameters: {total_params:,}\n"
        f"Trainable Parameters: {trainable_params:,}"
    )


def save_checkpoint(
        state: Dict,
        directory: str = "checkpoints",
        filename: str = "model.pth"
) -> str:
    """保存模型检查点"""
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, filename)

    torch.save(state, save_path)
    print(f"Checkpoint saved to: {save_path}")
    return save_path


def load_checkpoint(
        filepath: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        device: str = "cpu"
) -> Tuple[nn.Module, torch.optim.Optimizer, int]:
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from epoch {epoch}")
    return model, optimizer, epoch


def visualize_samples(
        samples: torch.Tensor,
        nrow: int = 4,
        denormalize: bool = True
) -> np.ndarray:
    """可视化样本图像"""
    if denormalize:
        samples = (samples + 1) / 2  # [-1,1] → [0,1]

    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    np_grid = grid.cpu().numpy().transpose(1, 2, 0)
    return np.clip(np_grid, 0, 1)


def seed_everything(seed: int = 42) -> None:
    """固定所有随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_config(config: Dict, path: str) -> None:
    """保存配置文件"""
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to: {path}")


def load_config(path: str) -> Dict:
    """加载配置文件"""
    with open(path, 'r') as f:
        config = json.load(f)
    print(f"Config loaded from: {path}")
    return config