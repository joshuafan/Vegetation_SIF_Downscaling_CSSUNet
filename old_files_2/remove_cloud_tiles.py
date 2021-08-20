"""
Creates a filtered dataset containing only large tiles with cloud cover fraction less than
"MAX_CLOUD_COVER"
"""

import csv
import numpy as np
import os
import pandas as pd

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
INFO_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_info_train.csv"),
                  "val": os.path.join(DATASET_DIR, "tile_info_val.csv")}
TILE_AVERAGE_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_averages_train.csv"),
                         "val": os.path.join(DATASET_DIR, "tile_averages_val.csv")}
FILTERED_INFO_CSV_FILES = {"train": os.path.join(DATASET_DIR, "filtered_tile_info_train.csv"),
                           "val": os.path.join(DATASET_DIR, "filtered_tile_info_val.csv")}
FILTERED_TILE_AVERAGE_CSV_FILES = {"train": os.path.join(DATASET_DIR, "filtered_tile_averages_train.csv"),
                         "val": os.path.join(DATASET_DIR, "filtered_tile_averages_val.csv")}

MAX_CLOUD_COVER = 0.1

for split, info_file in INFO_CSV_FILES.items():
    tile_metadata = pd.read_csv(info_file)
    tile_averages = pd.read_csv(TILE_AVERAGE_CSV_FILES[split])
    print('=========================')
    print('Split', split)
    print('Originally:', str(len(tile_metadata)), 'tiles')
    assert(len(tile_metadata) == len(tile_averages))
    filtered_averages = tile_averages[tile_averages['missing_reflectance'] <= MAX_CLOUD_COVER]
    filtered_tiles = tile_metadata[tile_metadata.index.isin(filtered_averages.index)]
    assert(len(filtered_tiles) == len(filtered_averages))
    print('After filtering:', str(len(filtered_tiles)), 'tiles')
    filtered_tiles.to_csv(FILTERED_INFO_CSV_FILES[split])
    filtered_averages.to_csv(FILTERED_TILE_AVERAGE_CSV_FILES[split])

