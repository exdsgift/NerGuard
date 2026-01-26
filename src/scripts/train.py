#!/usr/bin/env python
"""
Training entry point for NerGuard.

Usage:
    python -m src.scripts.train
    python -m src.scripts.train --batch-size 16 --epochs 5

For multi-GPU training:
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 -m src.scripts.train
"""

import argparse
from src.training.trainer import main as train_main
from src.core.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATA_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
)


def main():
    parser = argparse.ArgumentParser(description="Train NerGuard PII detection model")

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model name (HuggingFace)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to tokenized dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/mdeberta-pii-safe",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    args = parser.parse_args()

    train_main(
        model_name=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
