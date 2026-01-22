"""
Calibration Analysis Utilities for NER Dataset

This module provides tools for analyzing and debugging the alignment between
tokenized inputs and their corresponding entity labels in the processed NER dataset.
It includes functions for visual inspection of token-label pairs and decoding
entity mentions with highlighted annotations.
"""

import logging
from datasets import load_from_disk
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class Colors:
    OKGREEN = "\033[92m"
    FAIL = "\033[91m"
    WARNING = "\033[93m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def check_data_alignment():
    """
    Visually inspect the alignment between tokens and labels in the dataset.

    Displays the first 3 training examples that contain entity annotations,
    showing token-label pairs for non-O (outside entity) labels.
    """
    DATA_PATH = "./data/processed/tokenized_data"
    dataset = load_from_disk(DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")

    logger.info("--- VISUAL DATASET INSPECTION ---")
    found = 0
    for i in range(len(dataset["train"])):
        row = dataset["train"][i]
        labels = row["labels"]

        # Check if there's at least one entity label (not 0=O or -100=IGN)
        if any(l > 0 for l in labels):
            tokens = tokenizer.convert_ids_to_tokens(row["input_ids"])
            logger.info(f"\nSample {i} (Train Set):")

            for tok, lbl in zip(tokens, labels):
                if lbl == -100:
                    lbl_str = "IGN"
                elif lbl == 0:
                    lbl_str = "O"
                else:
                    lbl_str = f"ENTITY-{lbl}"
                if lbl_str != "O" or "ENTITY" in lbl_str:
                    logger.info(f"{tok:<15} | {lbl_str}")

            found += 1
            if found >= 3:
                break


def debug_decode_entities(dataset, tokenizer, num_samples=10):
    """
    Decode and display entity-annotated text samples with color highlighting.

    Args:
       dataset: The tokenized dataset to analyze
       tokenizer: The tokenizer used for encoding
       num_samples: Maximum number of samples to display (default: 10)

    Reconstructs text from tokens, highlighting entities in green.
    Skips ignored tokens (label=-100) and rejoins WordPiece subwords.
    """
    logger.info(f"\n{Colors.BOLD} - DECODING CHECK - {Colors.ENDC}")
    count = 0
    for i in range(len(dataset["train"])):
        labels = dataset["train"][i]["labels"]
        if not any(l > 0 for l in labels):
            continue

        ids = dataset["train"][i]["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)

        logger.info(f"\nExample {i}:")
        reconstructed = []
        for token, label in zip(tokens, labels):
            if label > 0:
                reconstructed.append(f"{Colors.OKGREEN}{token}{Colors.ENDC}")
            elif label == -100:
                pass  # Skip ignored tokens (e.g., special tokens, subword continuations)
            else:
                reconstructed.append(token)

        logger.info(" ".join(reconstructed).replace(" ##", ""))

        count += 1
        if count >= num_samples:
            break


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    DATA_PATH = "./data/processed/tokenized_data"
    dataset = load_from_disk(DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")

    # Uncomment to check token-label alignment
    # check_data_alignment()

    # Display decoded entities with highlighting
    debug_decode_entities(dataset, tokenizer)
