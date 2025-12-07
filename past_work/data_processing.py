from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer

from functools import partial
from pathlib import Path
from past_work.tokenizer import get_tokenizer
import json
import pprint as pp
from typing import List, Dict, Tuple
import unicodedata

DEFAULT_TOKENIZER = get_tokenizer()
TEXT_COL = "generated_text"
SPAN_COL = "pii_spans"


def download_dataset(
    path: Path,
    save_to_disk: bool = False,
):
    "download dataset from HF"
    ds = load_dataset(path, trust_remote_code=True)
    if save_to_disk:
        ds.save_to_disk(f"data/{path}")
    return ds


def upload_dataset(path: Path):
    # Carica il dataset da disco
    ds = load_from_disk(path)
    return ds


def create_validation_set(ds):
    if "validation" not in ds:
        tmp = ds["test"].train_test_split(test_size=0.4, seed=42)
        ds = DatasetDict(
            {"train": ds["train"], "validation": tmp["test"], "test": tmp["train"]}
        )
    ds.save_to_disk("data/synthetic_pii_finance_multilingual_split")
    return ds


def _parse_spans(spans):
    """Parse the pii_spans field, which can be a JSON string or a list of dicts."""
    if isinstance(spans, str):
        try:
            spans = json.loads(spans)
        except json.JSONDecodeError:
            return []
    if not isinstance(spans, list):
        return []
    out = []
    for s in spans:
        if isinstance(s, dict) and all(k in s for k in ("start", "end", "label")):
            out.append(s)
    return out


def _collect_label_set(dataset_dict, span_col: str = SPAN_COL):
    """Collect the set of unique labels in the dataset."""
    labels = set()
    for split in dataset_dict.keys():
        for ex in dataset_dict[split]:
            for s in _parse_spans(ex[span_col]):
                labels.add(s["label"])
    return sorted(labels)


def assign_labels(ds):
    "Assign BIO labels to the entities in the dataset"
    entity_labels = _collect_label_set(
        ds, span_col=SPAN_COL
    )  # es: ['company','date','name','street_address', ...]
    bio_labels = (
        ["O"] + [f"B-{l}" for l in entity_labels] + [f"I-{l}" for l in entity_labels]
    )
    label2id = {l: i for i, l in enumerate(bio_labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def _spans_to_bio(
    text: str, spans, label2id: Dict[str, int], tokenizer, max_length: int = 512
) -> Tuple[
    List[int], List[int], List[int]
]:  # First the max_lenght was 512 but is too much for Colab
    # normalizza per coerenza degli indici carattere
    text = unicodedata.normalize("NFC", text)
    spans = _parse_spans(spans)

    n = len(text)
    char_tags = ["O"] * n
    for s in spans:
        start, end, lab = s["start"], s["end"], s["label"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(lab, str)
        ):
            continue
        if start < 0 or end <= start or end > n:
            continue
        if f"B-{lab}" not in label2id:
            continue
        char_tags[start] = f"B-{lab}"
        for i in range(start + 1, end):
            char_tags[i] = f"I-{lab}"

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,  # need to try changing this!
        truncation=True,
        max_length=max_length,
    )
    offsets = enc["offset_mapping"]
    tags = []
    for a, b in offsets:
        if a == b:
            tags.append("O")
            continue
        start_tag = char_tags[a] if 0 <= a < n else "O"
        if start_tag.startswith("B-"):
            lab = start_tag
        else:
            sl = {t for t in char_tags[a:b] if t != "O"}
            lab = ("I-" + next(iter(sl)).split("-", 1)[1]) if sl else "O"
        tags.append(lab)

    y_ids = [label2id[t] for t in tags]
    return enc["input_ids"], enc["attention_mask"], y_ids


def _preprocess_batch(
    batch,
    label2id: Dict[str, int],
    tokenizer,
    text_col=TEXT_COL,
    span_col=SPAN_COL,
    max_length=512,
):
    input_ids, attention_mask, labels = [], [], []
    for text, spans in zip(batch[text_col], batch[span_col]):
        ids, mask, y = _spans_to_bio(
            text,
            spans,
            label2id,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(y)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def ds_processing(
    ds,
    tokenizer=DEFAULT_TOKENIZER,
    text_col=TEXT_COL,
    span_col=SPAN_COL,
    output_path="data/processed_dataset",
):
    label2id, id2label = assign_labels(ds, span_col=span_col)

    remove_cols = list(ds["train"].column_names)
    processed = DatasetDict()
    # Bind parameters explicitly so the map fn is self-contained
    map_fn = partial(
        _preprocess_batch,
        label2id=label2id,
        tokenizer=tokenizer,
        text_col=text_col,
        span_col=span_col,
        max_length=512,
    )
    for split in ds.keys():
        processed[split] = (
            ds[split]
            .map(map_fn, batched=True, remove_columns=remove_cols)
            .with_format(
                "torch",
                columns=["input_ids", "attention_mask", "labels"],
                output_all_columns=True,
            )
        )

    processed.save_to_disk(output_path)

    return (
        processed["train"],
        processed["validation"],
        processed.get("test", None),
        label2id,
        id2label,
    )


# if __name__ == "__main__":
#     ds = upload_dataset("data/processed_dataset")
#     pp.pprint(ds)
