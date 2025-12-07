## Datasets
already done:


to check:
- 


thesis-pii-finance/
├── data/
│   ├── raw/                  # Dataset originali (Gretel, etc.)
│   └── processed/            # Dataset unificato e tokenizzato (Arrow/Parquet)
├── src/
│   ├── components/
│   │   ├── encoder.py        # mDeBERTa wrapper
│   │   ├── uncertainty.py    # Logica di calcolo Entropia/Calibration
│   │   └── llm_verifier.py   # Classe per chiamare l'LLM (con json mode)
│   ├── pipeline/
│   │   └── hybrid_system.py  # La classe che unisce tutto
│   ├── training/
│   │   └── trainer.py        # Loop di training custom
│   └── utils/
│       ├── metrics.py        # Calcolo F1, Precision, Recall
│       └── validators.py     # Regex e checksums (Luhn, IBAN check)
├── notebooks/                # Solo per esplorazione, non per il codice core
├── config.yaml               # Iperparametri
└── main.py                   # Entry point

