from transformers import AutoTokenizer, AutoModelForTokenClassification
from data_processing import assign_labels, upload_dataset
from pathlib import Path
import pprint as pp


def upload_encoder_model(
    model_name="microsoft/mdeberta-v3-base",
    ds_path: Path = "data/synthetic_pii_finance_multilingual_split",
):
    ds = upload_dataset(ds_path)
    label2id, id2label = assign_labels(ds)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    return model, label2id, id2label


if __name__ == "__main__":
    model, label2id, id2label = upload_encoder_model()
    pp.pprint(model)
