"""
Filters train and validation tiles where the data is too noisy or incomplete (e.g. too much cloud cover, or too low SIF)
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
DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
         "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
         "2018-09-16"]

DATASET_DIRS = [os.path.join(DATA_DIR, "dataset_" + date) for date in DATES]
UNFILTERED_TROPOMI_FILES = [os.path.join(dataset_dir, "reflectance_cover_to_sif.csv") for dataset_dir in DATASET_DIRS]
UNFILTERED_OCO2_FILES = [os.path.join(dataset_dir, "oco2_eval_subtiles.csv") for dataset_dir in DATASET_DIRS]
# print("Dataset dirs", DATASET_DIRS)
# print("Unfiltered Tropomi files:", UNFILTERED_TROPOMI_FILES)
# print("Unfiltered OCO2 files:", UNFILTERED_OCO2_FILES)

# Create a folder for the processed dataset
# PROCESSED_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset")
PROCESSED_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_all_2")
if not os.path.exists(PROCESSED_DATASET_DIR):
    os.makedirs(PROCESSED_DATASET_DIR)

# Record the split of large grid areas between train/val/test
TRAIN_VAL_TEST_SPLIT_FILE = os.path.join(PROCESSED_DATASET_DIR, "data_split_random.csv")

# Resulting filtered csv files
FILTERED_CSV_FILES = {"train": os.path.join(PROCESSED_DATASET_DIR, "tile_info_train.csv"),
                      "val": os.path.join(PROCESSED_DATASET_DIR, "tile_info_val.csv"),
                      "test": os.path.join(PROCESSED_DATASET_DIR, "tile_info_test.csv")}
BAND_STATISTICS_CSV_FILES = {"train": os.path.join(PROCESSED_DATASET_DIR, "band_statistics_train.csv"),
                             "val": os.path.join(PROCESSED_DATASET_DIR, "band_statistics_val.csv"),
                             "test": os.path.join(PROCESSED_DATASET_DIR, "band_statistics_test.csv")}

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
CDL_COLUMNS = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']

MIN_CDL_COVERAGE = 0.8
MAX_LANDSAT_CLOUD_COVER = 0.1
MAX_TROPOMI_CLOUD_COVER = 0.2
MIN_TROPOMI_NUM_SOUNDINGS = 5
MIN_OCO2_NUM_SOUNDINGS = 5
MIN_SIF = 0.2
TROPOMI_FRACTION_VAL = 0.1
OCO2_FRACTION_VAL = 0.4
OCO2_FRACTION_TEST = 0.5
OCO2_SCALING_FACTOR = 1.69


# Divide the region into 1x1 degree large grid areas. Split them between train/val/test
# LONS = list(range(-108, -82))  # These lat/lons are the UPPER LEFT corner of the large grid areas
# LATS = list(range(39, 50))
# large_grid_areas = dict()
# for lon in LONS:
#     for lat in LATS:
#         random_number = random.random()
#         if random_number < 1 - FRACTION_TEST - FRACTION_VAL:
#             # split = 'train'
#             split = 'val'
#         elif random_number < 1 - FRACTION_TEST:
#             split = 'val'
#         else:
#             split = 'test'
#         large_grid_areas[(lon, lat)] = split

# # Save the split to a file
# with open(TRAIN_VAL_TEST_SPLIT_FILE, 'wb') as f:
#     pickle.dump(large_grid_areas, f, pickle.HIGHEST_PROTOCOL)

# Read TROPOMI files
tropomi_frames = []
for info_file in UNFILTERED_TROPOMI_FILES:
    tropomi_frame = pd.read_csv(info_file) 
    tropomi_frame['source'] = 'TROPOMI'
    tropomi_frames.append(tropomi_frame)
tropomi_metadata = pd.concat(tropomi_frames)
tropomi_metadata.reset_index(drop=True, inplace=True)

print("TROPOMI average SIF", tropomi_metadata['SIF'].mean())
print('TROPOMI tiles before filter', len(tropomi_metadata))

# Remove tiles with little CDL coverage (for the crops we're interested in)
tropomi_cdl_coverage = tropomi_metadata[CDL_COLUMNS].sum(axis=1)
print(tropomi_cdl_coverage.head())
plot_histogram(tropomi_cdl_coverage.to_numpy(), "tropomi_cdl_coverage.png")
tropomi_metadata = tropomi_metadata.loc[tropomi_cdl_coverage >= MIN_CDL_COVERAGE]
print('TROPOMI tiles after filtering missing CDL:', len(tropomi_metadata))

# Restrict to only tiles where Landsat (reflectance) cloud cover is low
tropomi_metadata = tropomi_metadata.loc[tropomi_metadata['missing_reflectance'] <= MAX_LANDSAT_CLOUD_COVER]
print('TROPOMI tiles after filtering missing reflectance:', len(tropomi_metadata))

# Restrict to only tiles where TROPOMI (SIF) cloud cover is low
tropomi_metadata = tropomi_metadata.loc[tropomi_metadata['tropomi_cloud_fraction'] <= MAX_TROPOMI_CLOUD_COVER]
print('TROPOMI tiles after filtering cloudy TROPOMI SIF:', len(tropomi_metadata))  

# Restrict to only tiles where TROPOMI has enough soundings
tropomi_metadata = tropomi_metadata.loc[tropomi_metadata['num_soundings'] >= MIN_TROPOMI_NUM_SOUNDINGS]
print('TROPOMI tiles after filtering low # of soundings:', len(tropomi_metadata))

# Restrict to only tiles where SIF is at least 0.2 (low SIF tiles are noisy)
tropomi_metadata = tropomi_metadata.loc[tropomi_metadata['SIF'] >= MIN_SIF]
print('TROPOMI tiles after filtering low SIF:', len(tropomi_metadata))

# Split TROPOMI datapoints into train, val
tropomi_val_start_idx = int((1 - TROPOMI_FRACTION_VAL) * len(tropomi_metadata))
tropomi_train_metadata, tropomi_val_metadata = np.split(tropomi_metadata.sample(frac=1), [tropomi_val_start_idx])
print('===========================================')
print('TROPOMI train samples:', len(tropomi_train_metadata))
print('TROPOMI val samples:', len(tropomi_val_metadata))

# Read OCO-2 files
oco2_frames = []
for info_file in UNFILTERED_OCO2_FILES:
    oco2_frame = pd.read_csv(info_file) 
    oco2_frame['source'] = 'OCO2'
    oco2_frame['SIF'] *= OCO2_SCALING_FACTOR  # OCO-2 SIF needs to be scaled to match TROPOMI
    oco2_frames.append(oco2_frame)
oco2_metadata = pd.concat(oco2_frames)
print('=============================================================')
print("OCO2 average SIF", oco2_metadata['SIF'].mean())
print("OCO2 tiles before filter", len(oco2_metadata))

# Remove tiles with little CDL coverage (for the crops we're interested in)
oco2_cdl_coverage = oco2_metadata[CDL_COLUMNS].sum(axis=1)
plot_histogram(oco2_cdl_coverage.to_numpy(), "oco2_cdl_coverage.png")
oco2_metadata = oco2_metadata.loc[oco2_cdl_coverage >= MIN_CDL_COVERAGE]
print('OCO2 tiles after filtering missing CDL:', len(oco2_metadata))

# Restrict to only tiles where Landsat (reflectance) cloud cover is low
oco2_metadata = oco2_metadata.loc[oco2_metadata['missing_reflectance'] <= MAX_LANDSAT_CLOUD_COVER]
print('OCO2 tiles after filtering missing reflectance:', len(oco2_metadata))

# Restrict to only tiles where OCO-2 has enough soundings
oco2_metadata = oco2_metadata.loc[oco2_metadata['num_soundings'] >= MIN_OCO2_NUM_SOUNDINGS]
print('OCO2 tiles after filtering low # of soundings:', len(oco2_metadata))

# Restrict to only tiles where SIF is at least 0.2 (low SIF tiles are noisy)
oco2_metadata = oco2_metadata.loc[oco2_metadata['SIF'] >= MIN_SIF]
print('OCO2 tiles after filtering low SIF:', len(oco2_metadata))

# Shuffle OCO-2 data
oco2_metadata = oco2_metadata.sample(frac=1).reset_index(drop=True)

# Split OCO-2 datapoints into train, val, test
oco2_val_start_idx = int((1 - OCO2_FRACTION_VAL - OCO2_FRACTION_TEST) * len(oco2_metadata))
oco2_test_start_idx = int((1 - OCO2_FRACTION_TEST) * len(oco2_metadata))
oco2_train_metadata, oco2_val_metadata, oco2_test_metadata = np.split(oco2_metadata.sample(frac=1), [oco2_val_start_idx, oco2_test_start_idx])
print('===========================================')
print('OCO2 train samples:', len(oco2_train_metadata))
print('OCO2 val samples:', len(oco2_val_metadata))
print('OCO2 test samples:', len(oco2_test_metadata))

# Combine OCO-2 and TROPOMI train/val points
train_metadata = pd.concat([tropomi_train_metadata, oco2_train_metadata])
train_metadata.reset_index(drop=True, inplace=True)
train_metadata = train_metadata.sample(frac=1).reset_index(drop=True)
val_metadata = pd.concat([tropomi_val_metadata, oco2_val_metadata])
val_metadata.reset_index(drop=True, inplace=True)
val_metadata = val_metadata.sample(frac=1).reset_index(drop=True)
split_metadata = {'train': train_metadata,
                  'val': val_metadata,
                  'test': oco2_test_metadata}



# Train on TROPOMI, val/test on OCO-2
# oco2_metadata['split'] = oco2_metadata.apply(lambda row: determine_split(large_grid_areas, row), axis=1)
# split_metadata = {'train': tropomi_metadata,
#                   'val': oco2_metadata[oco2_metadata['split'] == 'val'],
#                   'test': oco2_metadata[oco2_metadata['split'] == 'test']}

# Combine OCO2 and TROPOMI tiles into a single dataframe
# tile_metadata = pd.concat([tropomi_metadata, oco2_metadata])
# tile_metadata.reset_index(drop=True, inplace=True)

# # Shuffle rows to mix OCO2/TROPOMI together
# tile_metadata = tile_metadata.sample(frac=1).reset_index(drop=True)
# # print('After all filtering:', tile_metadata['source'].value_counts())
# print('TROPOMI by date', tropomi_metadata['date'].value_counts())
# print('OCO2 by date', oco2_metadata['date'].value_counts())

# # For each row, determine whether it's in a train, val, or test large grid region
# tile_metadata['split'] = tile_metadata.apply(lambda row: determine_split(large_grid_areas, row), axis=1)
# tile_metadata['split'] = tile_metadata.apply(lambda row: determine_split_random(row, FRACTION_VAL, FRACTION_TEST), axis=1)

# split_metadata = {'train': tile_metadata[tile_metadata['split'] == 'train'],
#                   'val': tile_metadata[tile_metadata['split'] == 'val'],
#                   'test': tile_metadata[tile_metadata['split'] == 'test']}

# For each split, write the processed dataset to a file
for split, metadata in split_metadata.items():
    metadata.reset_index(drop=True, inplace=True)
    metadata.to_csv(FILTERED_CSV_FILES[split])

    # Number of pure pixels
    pure_grassland_pasture = metadata.loc[(metadata['grassland_pasture'] > 0.6) & (metadata['source'] == 'OCO2')]
    pure_corn = metadata.loc[(metadata['corn'] > 0.6) & (metadata['source'] == 'OCO2')]
    pure_soybean = metadata.loc[(metadata['soybean'] > 0.6) & (metadata['source'] == 'OCO2')]
    pure_deciduous_forest = metadata.loc[(metadata['deciduous_forest'] > 0.6) & (metadata['source'] == 'OCO2')]
    print('===================== Split', split, '=====================')
    print("OCO2 pure grassland/pasture", len(pure_grassland_pasture))
    print("OCO2 pure corn", len(pure_corn))
    print("OCO2 pure soybean", len(pure_soybean))
    print("OCO2 pure deciduous forest", len(pure_deciduous_forest))

    # Compute averages for each band
    selected_columns = metadata[STATISTICS_COLUMNS]
    # print("Band values ARRAY shape", selected_columns.shape)
    band_means = selected_columns.mean(axis=0)
    band_stds = selected_columns.std(axis=0)
    # print("Band means", band_means)
    # print("Band stds", band_stds)

    # Write band averages to file
    statistics_rows = [['mean', 'std']]
    for i in range(len(band_means)):
        statistics_rows.append([band_means[i], band_stds[i]])
    with open(BAND_STATISTICS_CSV_FILES[split], 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in statistics_rows:
            csv_writer.writerow(row)
