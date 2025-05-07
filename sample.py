# generate.py

import torch
from torchvision.utils import save_image
from configs.default import training_config, model_config
from diffusion.trainer import DiffusionTrainer


def generate_samples(
        checkpoint_path: str,
        output_dir: str = "generated_samples",
        num_samples: int = 16,
        img_size: int = 256
):
    """Generate samples using a trained diffusion model from a checkpoint"""

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build full config from training and model configs
    config = {
        **training_config,
        "model": model_config["unet"]
    }

    # Initialize trainer (with model)
    trainer = DiffusionTrainer(config, device)

    # Load checkpoint
    trainer.load_checkpoint(checkpoint_path)

    # Generate samples using EMA model
    with torch.no_grad():
        noise = torch.randn(num_samples, 1, img_size, img_size, device=device)
        samples = trainer.ema_model.ddim_sample(noise)
        samples = (samples.clamp(-1, 1) + 1) / 2  # Convert to [0, 1] range

    # Create output directory and save results
    os.makedirs(output_dir, exist_ok=True)
    save_image(samples, os.path.join(output_dir, "final_samples.png"), nrow=4)
    print(f"Generated samples saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample generation script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")

    args = parser.parse_args()

    generate_samples(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )