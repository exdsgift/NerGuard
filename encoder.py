from transformers import AutoTokenizer, AutoModelForTokenClassification
from data_processing import assign_labels, upload_dataset

def upload_encoder_model(model_name="microsoft/mdeberta-v3-base",
                         ):
   ds = upload_dataset("data/synthetic_pii_finance_multilingual_split")
   label2id, id2label = assign_labels(ds)
   model =  AutoModelForTokenClassification.from_pretrained(
      model_name,
      num_labels=len(label2id),
      id2label=id2label,
      label2id=label2id
   )
   return model, label2id, id2label

# if __name__ == "__main__":
#    print("All packages imported.")