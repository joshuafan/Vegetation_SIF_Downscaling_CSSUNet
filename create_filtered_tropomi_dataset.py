"""
Filters train and validation tiles where the data is too noisy or incomplete (e.g. too much cloud cover, or too low SIF)
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.model_selection
from sif_utils import plot_histogram
import torch

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
UNFILTERED_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_info_unfiltered_train.csv"),
                        "val": os.path.join(DATASET_DIR, "tile_info_unfiltered_val.csv")}
FILTERED_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_info_train.csv"),
                      "val": os.path.join(DATASET_DIR, "tile_info_val.csv")}
BAND_STATISTICS_CSV_FILES = {"train": os.path.join(DATASET_DIR, "band_statistics_train.csv"),
                             "val": os.path.join(DATASET_DIR, "band_statistics_val.csv")}

STATISTICS_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils', 'missing_reflectance', 'SIF']

MAX_LANDSAT_CLOUD_COVER = 0.1
MAX_TROPOMI_CLOUD_COVER = 0.2
MIN_TROPOMI_N = 5
MIN_SIF = 0.2

for split in ["train", "val"]:
    tile_metadata = pd.read_csv(UNFILTERED_CSV_FILES[split])
    print('Split:', split)
    print('Number of tiles:', len(tile_metadata))

    # Restrict to only tiles where TROPOMI (SIF) cloud cover is low
    tile_metadata = tile_metadata.loc[tile_metadata['cloud_fraction'] < MAX_TROPOMI_CLOUD_COVER]
    print('After filtering cloudy TROPOMI SIF:', len(tile_metadata))  

    # Restrict to only tiles where TROPOMI has enough soundings
    tile_metadata = tile_metadata.loc[tile_metadata['num_soundings'] >= MIN_TROPOMI_N]
    print('After filtering low # of soundings:', len(tile_metadata))

    # Restrict to only tiles where SIF is at least 0.2 (low SIF tiles are noisy)
    tile_metadata = tile_metadata.loc[tile_metadata['SIF'] >= MIN_SIF]
    print('After filtering low SIF:', len(tile_metadata))

    # Restrict to only tiles where Landsat (reflectance) cloud cover is low
    tile_metadata = tile_metadata.loc[tile_metadata['missing_reflectance'] < MAX_LANDSAT_CLOUD_COVER]
    print('After filtering missing reflectance:', len(tile_metadata))

    # Write filtered dataset to csv file
    tile_metadata.reset_index(drop=True, inplace=True)
    tile_metadata.to_csv(FILTERED_CSV_FILES[split])

    # Compute averages for each band
    selected_columns = tile_metadata[STATISTICS_COLUMNS]
    print("Band values ARRAY shape", selected_columns.shape)
    band_means = selected_columns.mean(axis=0)
    band_stds = selected_columns.std(axis=0)
    print("Band means", band_means)
    print("Band stds", band_stds)

    # Write band averages to file
    statistics_rows = [['mean', 'std']]
    for i in range(len(band_means)):
        statistics_rows.append([band_means[i], band_stds[i]])
    with open(BAND_STATISTICS_CSV_FILES[split], 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in statistics_rows:
            csv_writer.writerow(row)