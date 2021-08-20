import csv
import os
import pandas as pd

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
stats = pd.read_csv(BAND_STATISTICS_FILE)
stats['std'][:-1] = stats['std'][:-1] / 371
stats.to_csv(os.path.join(DATASET_DIR, "band_statistics_train_FIXED.csv"))
