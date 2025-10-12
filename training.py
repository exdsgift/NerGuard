from transformers import TrainingArguments, Trainer, AutoTokenizer
import pprint as pp
import torch
import os

from tokenizer import upl_tokenizer
from encoder import upload_encoder_model
from data_processing import upload_dataset
from padding import NERCollatorSoftmax, compute_metrics_softmax


tokenizer = upl_tokenizer()
model, label2id, id2label = upload_encoder_model()
collator = NERCollatorSoftmax(tokenizer)
ds = upload_dataset("data/processed_dataset")
train_ds = ds["train"]
eval_ds = ds["validation"]
test_ds = ds["test"]

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


args = TrainingArguments(
    output_dir="pii_mdeberta_softmax",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=32,
    # Learning rate and schedule
    learning_rate=1e-5,  # 2e-5
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    # Evaluation and saving
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    # Memory and performance optimizations
    fp16=True,
    bf16=False,
    fp16_full_eval=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.5,
    # DataLoader settings
    dataloader_num_workers=1,
    dataloader_pin_memory=True,
    # Additional settings
    logging_steps=25,
    logging_strategy="steps",
    report_to="none",
    save_safetensors=True,
    # Reproducibility
    seed=42,
    data_seed=42,
)


def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


model.config.use_cache = False

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# clean_memory()


# Create a closure for compute_metrics that includes id2label
def get_compute_metrics(id2label):
    def compute_metrics_wrapper(eval_preds):
        return compute_metrics_softmax(eval_preds, id2label)

    return compute_metrics_wrapper


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    processing_class=tokenizer,
    compute_metrics=get_compute_metrics(id2label),  # Updated this
)


def run_training():
    try:
        # clean_memory()
        trainer.train()
        # clean_memory()
        # Save the best model
        trainer.save_model("pii_mdeberta_softmax/outputs/best")
        tokenizer.save_pretrained("pii_mdeberta_softmax/outputs/best")

    except Exception as e:
        print(f"Training error occurred: {str(e)}")
        clean_memory()
        raise
    finally:
        clean_memory()


def training_metr():
    val_metrics = trainer.evaluate()
    if test_ds is not None:
        test_metrics = trainer.evaluate(test_ds)
    return val_metrics, test_metrics


if __name__ == "__main__":
    print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    run_training()
    val_metrics, test_metrics = training_metr()
    print("_____________________________")
    pp.pprint(val_metrics)
    print("_____________________________")
    pp.pprint(test_metrics)
