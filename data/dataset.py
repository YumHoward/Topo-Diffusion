# data/dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Lambda


class PaperCutDataset(Dataset):
    """Dataset loader for papercut art style images"""

    def __init__(self, img_dir: str, img_size: int = 256):
        """
        Args:
            img_dir: Directory path containing papercut images
            img_size: Output image size (square)
        """
        super().__init__()
        self.img_dir = img_dir
        self.img_size = img_size

        # Get all image paths in the directory
        self.img_paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # Define image preprocessing pipeline
        self.transform = Compose([
            Resize((img_size, img_size)),  # Resize to target size
            ToTensor(),  # Convert to tensor [0,1]
            Lambda(lambda x: (x > 0.5).float()),  # Binarize image
            Lambda(lambda x: x * 2 - 1)  # Normalize to [-1, 1]
        ])

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and preprocess a single image"""
        img_path = self.img_paths[idx]

        # Load image in grayscale
        img = Image.open(img_path).convert('L')

        # Apply transforms
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self) -> int:
        """Return total number of samples in the dataset"""
        return len(self.img_paths)

    @property
    def num_samples(self) -> int:
        """Number of available samples (same as __len__)"""
        return len(self)