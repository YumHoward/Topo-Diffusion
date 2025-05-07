# configs/default.py
import os

# Training configuration
training_config = {
    # Dataset settings
    "dataset_dir": "dataset",  # Path to dataset
    "img_size": 256,  # Input image size
    "num_workers": 4 if os.name != 'nt' else 0,  # Number of data loading workers

    # Training hyperparameters
    "batch_size": 8,  # Batch size (adjust based on GPU memory)
    "lr": 1e-4,  # Initial learning rate
    "weight_decay": 1e-4,  # Weight decay
    "epochs": 5000,  # Total number of training epochs
    "ema_decay": 0.995,  # EMA decay rate

    # Diffusion process parameters
    "timesteps": 1000,  # Total diffusion steps
    "schedule_type": "cosine",  # Noise schedule strategy

    # Output settings
    "output_root": "result",  # Root directory for outputs
    "run_name": "256_experiment",  # Experiment name
    "save_interval": 100,  # Model saving interval (in epochs)
    "sample_interval": 100,  # Sampling interval (in epochs)

    # DDIM inference parameters
    "ddim_steps": 50,  # Number of sampling steps
    "eta": 0.0  # Randomness coefficient (0 means deterministic)
}

# Model architecture configuration
model_config = {
    "unet": {
        # Basic parameters
        "img_size": 256,  # Input image size
        "t_dim": 256,  # Time embedding dimension

        # Down-sampling path configuration
        "down_blocks": [
            "fractal", "fractal",
            "fractal", "gcn", "gcn"  # Block types for each layer
        ],

        # Mid-block configuration
        "mid_block": "gcn",  # Block type for mid-layer

        # Up-sampling path configuration
        "up_blocks": [
            "residual", "residual",
            "residual", "residual"  # Block types for each layer
        ],

        # Channel configuration
        "channels": [64, 128, 256, 512, 1024],  # Number of channels per layer

        # Fractal feature parameters
        "fractal": {
            "num_scales": 3,  # Number of multi-scales
            "base_scale": 16  # Base scale size
        },

        # GCN parameters
        "gcn": {
            "patch_size": 16,  # Patch size for graph partitioning
            "neighbor": 4  # Number of adjacent nodes
        }
    }
}

# Optimizer configuration (optional independent settings)
optimizer_config = {
    "betas": (0.9, 0.999),  # Adam optimizer parameters
    "amsgrad": False  # Whether to use AMSGrad
}