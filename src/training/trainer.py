import os
import sys
import json
import torch
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
try:
    from src.components.encoder import PIIEncoder
except ImportError:
    print("Error importing PIIEncoder. Check path.")
    sys.exit(1)

def compute_metrics_func(p, id2label, metric):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

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

def main():
    os.environ["WANDB_PROJECT"] = "ner-guard-pii"
    os.environ["WANDB_LOG_MODEL"] = "false" # Set true if you want to upload model checkpoints to cloud (slow)
    os.environ["WANDB_WATCH"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(f"[Process {local_rank}] Assegnato a GPU Logica {local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Single Process] Assegnato a {device}")

    is_main_process = (local_rank == -1 or local_rank == 0)

    MODEL_NAME = "microsoft/mdeberta-v3-base"
    DATA_PATH = "./data/processed/tokenized_data"
    OUTPUT_DIR = "./models/mdeberta-pii-safe"
    LOG_DIR = "./logs"
    
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 2
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 2
    
    if is_main_process:
        print(f"Loading dataset (Epochs: {NUM_EPOCHS})...")
    
    dataset = load_from_disk(DATA_PATH, keep_in_memory=False)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    try:
        with open("./data/processed/label2id.json", "r") as f:
            label2id = json.load(f)
        with open("./data/processed/id2label.json", "r") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        if is_main_process:
            print(f"Error loading json: {e}")
        return

    if is_main_process:
        print("Initializing Model...")
    
    encoder = PIIEncoder(MODEL_NAME)
    tokenizer = encoder.get_tokenizer()
    model = encoder.get_model(
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        dropout_rate=0.1
    )
    
    model = model.to(device)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    metric = evaluate.load("seqeval")
    def compute_metrics_wrapper(p):
        return compute_metrics_func(p, id2label, metric)

    run_name = f"mdeberta-rtx6000-fp16-bs{BATCH_SIZE}"

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        
        fp16=True,
        bf16=False,

        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
        dataloader_num_workers=4,
        
        num_train_epochs=NUM_EPOCHS,
        
        # Strategy
        eval_strategy="steps", 
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        logging_dir=LOG_DIR,
        logging_steps=50,
        report_to="wandb",
        run_name=run_name,

        eval_accumulation_steps=1,
        
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=4,
            early_stopping_threshold=0.001
            )
        ]
    )

    if is_main_process: print(f"--> Start training loop (Run: {run_name})")
    trainer.train()
    
    if is_main_process:
        print("--> Saving model")
        trainer.save_model(os.path.join(OUTPUT_DIR, "final"))

if __name__ == "__main__":
    main()