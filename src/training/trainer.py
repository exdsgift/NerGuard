"""
Training script for NerGuard PII detection model.

This script handles distributed training of the mDeBERTa-based NER model
with support for multi-GPU training via PyTorch DDP.

Usage:
    # Single GPU
    python -m src.training.trainer

    # Multi-GPU (DDP)
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 -m src.training.trainer
"""

import os
import json
import warnings

import torch
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)

from src.training.encoder import PIIEncoder
from src.core.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATA_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
)
from src.utils.logging_config import setup_logging

warnings.filterwarnings("ignore")

logger = setup_logging("ModelTraining")


def compute_metrics_func(p, id2label, metric):
    """
    Compute evaluation metrics for token classification.

    Uses seqeval for proper NER evaluation at the entity level.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Filter out ignored tokens (-100)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": float(results["overall_precision"]),
        "recall": float(results["overall_recall"]),
        "f1": float(results["overall_f1"]),
        "accuracy": float(results["overall_accuracy"]),
    }


def main(
    model_name: str = DEFAULT_BASE_MODEL,
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = "./models/mdeberta-pii-safe",
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    gradient_accumulation: int = 1,
    use_wandb: bool = True,
):
    """
    Main training function.

    Args:
        model_name: HuggingFace model name
        data_path: Path to tokenized dataset
        output_dir: Directory to save model checkpoints
        batch_size: Per-device batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        gradient_accumulation: Gradient accumulation steps
        use_wandb: Whether to use Weights & Biases logging
    """
    # Setup environment
    if use_wandb:
        os.environ["WANDB_PROJECT"] = "ner-guard-pii"
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_WATCH"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Check if we're the main process in distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank == -1 or local_rank == 0

    # Load dataset
    if is_main_process:
        logger.info(f"Loading dataset from {data_path}...")

    try:
        dataset = load_from_disk(data_path, keep_in_memory=False)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Load label mappings
    label_path = os.path.dirname(data_path)
    try:
        with open(os.path.join(label_path, "label2id.json"), "r") as f:
            label2id = json.load(f)
        with open(os.path.join(label_path, "id2label.json"), "r") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        if is_main_process:
            logger.error(f"Error loading label mappings: {e}")
        return

    # Initialize model
    if is_main_process:
        logger.info(f"Initializing model: {model_name}")

    encoder = PIIEncoder(model_name)
    tokenizer = encoder.get_tokenizer()
    model = encoder.get_model(
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        dropout_rate=0.1,
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    # Evaluation metric
    metric = evaluate.load("seqeval")

    def compute_metrics_wrapper(p):
        return compute_metrics_func(p, id2label, metric)

    # Training arguments
    run_name = f"mdeberta-bs{batch_size}-lr{learning_rate}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        # Batching
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=gradient_accumulation,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        eval_accumulation_steps=1,
        # Precision
        fp16=True,
        bf16=False,
        # DDP & Checkpointing
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=num_epochs,
        group_by_length=True,
        # Evaluation strategy
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # Logging
        logging_dir="./logs",
        logging_steps=100,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        # Cleanup
        save_total_limit=2,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=4,
                early_stopping_threshold=0.001,
            )
        ],
    )

    # Start training
    if is_main_process:
        n_gpus = torch.cuda.device_count()
        logger.info(f"Starting training on {n_gpus} GPU(s) (Run: {run_name})")

    trainer.train()

    # Save final model
    if is_main_process:
        logger.info("Saving final model...")
        final_path = os.path.join(output_dir, "final")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 -m src.training.trainer
