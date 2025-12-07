from transformers import AutoTokenizer


def get_tokenizer(
    model_name="microsoft/mdeberta-v3-base", use_fast=True, padding_side="right"
):
    "Return the tokenizer"
    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast,
        padding_side=padding_side,
    )
    tok.is_fast, "Serve tokenizer fast per offset_mapping"
    return tok


def upl_tokenizer(MODEL_ID="microsoft/mdeberta-v3-base"):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,  # per avere offset_mapping affidabile
        model_max_length=256,  # allinea alla tua max_length
    )
    assert tokenizer.is_fast, "Serve un fast tokenizer per gli offset"
    return tokenizer
