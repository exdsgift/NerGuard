import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer
)
from typing import Dict

class PIIEncoder:

    def __init__(self, model_name: str = "microsoft/mdeberta-v3-base"):
        self.model_name = model_name
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """
        Load the tokenizer. We use DebertaV2TokenizerFast for efficiency.
        mDeBERTa v3 requires the use_fast=True flag and sentencepiece installed.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_fast=True,
                add_prefix_space=True
            )
            return tokenizer
        except OSError as e:
            raise OSError(f"Error during tokenizer uploading for: {self.model_name}.\nMake sure to install it first. Info:\n{e}")

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(
        self, 
        num_labels: int, 
        id2label: Dict[int, str], 
        label2id: Dict[str, int],
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False
    ) -> nn.Module:
        """
        Initializes and returns the configured HuggingFace model.
        
        Args:
            num_labels (int): Total number of PII classes.
            id2label (Dict): Mapping from ID to label name.
            label2id (Dict): Mapping from label name to ID.
            dropout_rate (float): Dropout probability for the classification head.
            freeze_backbone (bool): If True, freeze the encoder weights (excluding the head).
        
        Returns:
            model: An instance of AutoModelForTokenClassification.
        """
        
        # Base config
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )

        # Pretrained model
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            config=config,
            ignore_mismatched_sizes=True # Utile se ricarichiamo checkpoint con head diverse
        )

        if freeze_backbone:
            self._freeze_layers(model)

        return model

    def _freeze_layers(self, model):
        """
        Freeze the parameters of the mDeBERTa encoder, leaving only the classifier head trainable.
        """
        if hasattr(model, "deberta"):
            for param in model.deberta.parameters():
                param.requires_grad = False
            print(f"--> Backbone {self.model_name} frozen. Only the head will be trained.")
        else:
            print("Unable to find the ‘deberta’ module for automatic freezing.")

# Debug
if __name__ == "__main__":
    # Mock labels for test
    mock_labels = {"O": 0, "B-PER": 1, "I-PER": 2}
    mock_id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
    
    print("PIIEncoder initialization ...")
    encoder_factory = PIIEncoder()
    
    print("Uploading model...")
    model = encoder_factory.get_model(
        num_labels=len(mock_labels),
        id2label=mock_id2label,
        label2id=mock_labels,
        dropout_rate=0.2
    )
    
    print(f"Model uploaded:\n{type(model)}")
    print(f"Tokenizer uploaded:\n{encoder_factory.get_tokenizer()}")
    
    # Test
    sample_text = "Il signor Rossi vive a Roma."
    tokens = encoder_factory.get_tokenizer()(sample_text)
    print(f"Test tokenizzazione: {tokens.input_ids}")