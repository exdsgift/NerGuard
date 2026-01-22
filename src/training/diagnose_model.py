import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk
from sklearn.metrics import classification_report
from tqdm import tqdm

# --- CONFIGURAZIONE ---
MODEL_PATH = "./models/mdeberta-pii-safe/final"
DATA_PATH = "./data/processed/tokenized_data"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Colors:
    HEADER = "\033[95m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def load_resources():
    print(f"Uploading model from:  {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"{Colors.FAIL}Error while loading model: {e}{Colors.ENDC}")
        exit(1)


def analyze_sentence(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    preds = torch.argmax(probs, dim=-1)
    id2label = model.config.id2label

    print(f"\n{Colors.HEADER}--- Sentence: '{text}' ---{Colors.ENDC}")
    print(f"{'TOKEN':<15} | {'PRED':<20} | {'CONF %':<8} | {'2nd CHOICE (MARGIN)'}")
    print("-" * 85)

    for i, token in enumerate(tokens):
        if token in tokenizer.all_special_tokens:
            continue

        pred_idx = preds[i].item()
        pred_label = id2label[pred_idx]
        confidence = probs[i][pred_idx].item()

        top2 = torch.topk(probs[i], 2)
        if len(top2.values) > 1:
            second_prob = top2.values[1].item()
            second_label = id2label[top2.indices[1].item()]
            margin = confidence - second_prob
            alt_info = f"{second_label} ({second_prob:.2%})" if margin < 0.9 else ""
        else:
            alt_info = ""

        color = Colors.OKGREEN if pred_label != "O" else Colors.ENDC
        if confidence < 0.6:
            color = Colors.WARNING

        display_token = token.replace(" ", " ")
        print(
            f"{color}{display_token:<15} | {pred_label:<20} | {confidence:.1%}   | {alt_info}{Colors.ENDC}"
        )


def generate_full_report(tokenizer, model):
    print(f"\n{Colors.HEADER}--> Stats Validation Set{Colors.ENDC}")

    try:
        dataset = load_from_disk(DATA_PATH)
        eval_data = dataset["validation"]
    except:
        print("Dataset not found or could not be loaded.")
        return

    id2label = model.config.id2label
    true_labels = []
    pred_labels = []

    for i in tqdm(range(0, len(eval_data), BATCH_SIZE), desc="Processing"):
        batch = eval_data[i : i + BATCH_SIZE]

        inputs = tokenizer.pad(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        labels = batch["labels"]
        if hasattr(labels, "numpy"):
            labels = labels.numpy()

        for j in range(len(labels)):
            preds = predictions[j]
            lbls = labels[j]

            for p, l in zip(preds, lbls):
                if l != -100:
                    true_labels.append(id2label[l])
                    pred_labels.append(id2label[p])

    print("\n" + "=" * 60)
    print(f"{Colors.BOLD} Class Stats {Colors.ENDC}")
    print("=" * 60)

    unique_labels = sorted(list(set(true_labels) | set(pred_labels)))
    print(
        classification_report(
            true_labels, pred_labels, labels=unique_labels, digits=3, zero_division=0
        )
    )


def main():
    tokenizer, model = load_resources()

    print(f"\n{Colors.OKGREEN}- Sanity Check -{Colors.ENDC}")
    generate_full_report(tokenizer, model)


if __name__ == "__main__":
    main()
