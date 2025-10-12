from data_processing import download_dataset, upload_dataset
import pprint as pp

# ds1 = upload_dataset()
# ds2 = upload_dataset("beki/privy")

if __name__ == "__main__":
   ds1 = upload_dataset("data/synthetic_pii_finance_multilingual_split")
   ds2 = upload_dataset("data/beki/privy")
   ds3 = upload_dataset("data/processed_dataset")
   pp.pprint(ds1)
   pp.pprint(ds2)
   pp.pprint(ds3)
