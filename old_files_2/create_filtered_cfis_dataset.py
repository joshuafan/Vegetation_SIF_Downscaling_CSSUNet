"""
Filters train and validation tiles where the data is too noisy or incomplete (e.g. too much cloud cover, or too low SIF).
Intended to be used for CFIS dataset.
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import sklearn.model_selection
from sif_utils import plot_histogram, determine_split, determine_split_random
import torch

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATES = ["2016-08-01"]
DATASET_DIRS = [os.path.join(DATA_DIR, "dataset_" + date) for date in DATES]
UNFILTERED_CFIS_FILES = [os.path.join(dataset_dir, "eval_subtiles.csv") for dataset_dir in DATASET_DIRS]
PROCESSED_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_all_2")
FILTERED_CFIS_FILE = os.path.join(PROCESSED_DATASET_DIR, "cfis_subtiles_filtered_1000soundings.csv")

CDL_COLUMNS = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']

MIN_CDL_COVERAGE = 0.8
MIN_NUM_SOUNDINGS = 1000
MIN_SIF = 0.2
MAX_LANDSAT_CLOUD_COVER = 0.1

# Read CFIS files (the loop is only necessary if we have multiple files)
cfis_frames = []
for info_file in UNFILTERED_CFIS_FILES:
    cfis_frame = pd.read_csv(info_file) 
    cfis_frames.append(cfis_frame)
cfis_metadata = pd.concat(cfis_frames)
cfis_metadata.reset_index(drop=True, inplace=True)

print("CFIS average SIF", cfis_metadata['SIF'].mean())
print('CFIS tiles before filter', len(cfis_metadata))

# Remove tiles with little CDL coverage (for the crops we're interested in)
cdl_coverage = cfis_metadata[CDL_COLUMNS].sum(axis=1)
print(cdl_coverage.head())
plot_histogram(cdl_coverage.to_numpy(), "cfis_cdl_coverage.png")
cfis_metadata = cfis_metadata.loc[cdl_coverage >= MIN_CDL_COVERAGE]
print('CFIS tiles after filtering missing CDL:', len(cfis_metadata))

# Restrict to only tiles where Landsat (reflectance) cloud cover is low
cfis_metadata = cfis_metadata.loc[cfis_metadata['missing_reflectance'] <= MAX_LANDSAT_CLOUD_COVER]
print('CFIS tiles after filtering missing reflectance:', len(cfis_metadata))

# Restrict to only tiles where CFIS has enough soundings
cfis_metadata = cfis_metadata.loc[cfis_metadata['num_soundings'] >= MIN_NUM_SOUNDINGS]
print('CFIS tiles after filtering low # of soundings:', len(cfis_metadata))

# Restrict to only tiles where SIF is at least 0.2 (low SIF tiles are noisy)
cfis_metadata = cfis_metadata.loc[cfis_metadata['SIF'] >= MIN_SIF]
print('CFIS tiles after filtering low SIF:', len(cfis_metadata))

cfis_metadata.reset_index(drop=True, inplace=True)
cfis_metadata.to_csv(FILTERED_CFIS_FILE)

