# train.py

import logging
import torch
from configs.default import training_config, model_config
from diffusion.trainer import DiffusionTrainer


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Initializing PaperCut Diffusion Training")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Merge configurations
    config = {
        **training_config,
        "model": model_config["unet"]
    }

    # Initialize trainer
    trainer = DiffusionTrainer(config, device)

    # Print model summary
    logger.info("\nModel Summary:\n" + trainer.model_summary())

    # Training loop
    for epoch in range(config["epochs"]):
        avg_loss = trainer.train_epoch(epoch)

        # Save model checkpoint
        if (epoch + 1) % config["save_interval"] == 0:
            trainer.save_model(epoch)

        # Generate sample images
        if (epoch + 1) % config["sample_interval"] == 0:
            trainer.generate_samples(epoch)

        logger.info(f"Epoch [{epoch + 1}/{config['epochs']}] Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()