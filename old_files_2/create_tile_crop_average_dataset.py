"""
(Run this after create_filtered_tropomi_dataset.py)
Creates dataset with average reflectance values for each crop, for each tile.
Standardizes feature values!!!!!
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.model_selection
from sif_utils import plot_histogram
import tile_transforms
import torch
from crop_type_averages_dataset import CropTypeAveragesFromTileDataset

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
INFO_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_info_train.csv"),
                  "val": os.path.join(DATASET_DIR, "tile_info_val.csv")}
CROP_AVERAGE_FILES = {"train": os.path.join(DATASET_DIR, "tile_info_train_crops.csv"),
                      "val": os.path.join(DATASET_DIR, "tile_info_val_crops.csv")}
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")

CROP_TYPES = {'grassland_pasture': 12,
              'corn': 13,
              'soybean': 14,
              'shrubland': 15,
              'deciduous_forest': 16,
              'evergreen_forest': 17,
              'spring_wheat': 18,
              'developed_open_space': 19,
              'other_hay_non_alfalfa': 20,
              'winter_wheat': 21,
              'herbaceous_wetlands': 22,
              'woody_wetlands': 23,
              'open_water': 24,
              'alfalfa': 25,
              'fallow_idle_cropland': 26,
              'sorghum': 27,
              'developed_low_intensity': 28,
              'barren': 29,
              'durum_wheat': 30,
              'canola': 31,
              'sunflower': 32,
              'dry_beans': 33,
              'developed_med_intensity': 34,
              'millet': 35,
              'sugarbeets': 36,
              'oats': 37,
              'mixed_forest': 38,
              'peas': 39,
              'barley': 40,
              'lentils': 41}
CROP_INDICES = list(CROP_TYPES.values())

FLOAT_EQUALITY_TOLERANCE = 1e-5
FEATURES = {'ref_1': 0,
            'ref_2': 1,
            'ref_3': 2,
            'ref_4': 3,
            'ref_5': 4,
            'ref_6': 5,
            'ref_7': 6,
            'ref_10': 7,
            'ref_11': 8,
            'Rainf_f_tavg': 9,
            'SWdown_f_tavg': 10,
            'Tair_f_tavg': 11}
MISSING_IDX = 42

# Construct list of features. We compute features for pixels of each crop
# type, as well as "other" (pixels that are not occupied by a listed crop type)
NEW_COLUMNS = ['lon', 'lat', 'date', 'tile_file']
for crop_type in list(CROP_TYPES.keys()) + ['other']:
    NEW_COLUMNS.append(crop_type + '_cover')
    for feature_name in FEATURES:
        NEW_COLUMNS.append(crop_type + '_' + feature_name)
NEW_COLUMNS.extend(['SIF'])
print(NEW_COLUMNS)

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
print("Means", train_means)
print("Stds", train_stds)
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]
transform = tile_transforms.StandardizeTile(band_means, band_stds)


for split in ["train", "val"]:
    # Create a dataset of tile-average values
    tile_metadata = pd.read_csv(INFO_CSV_FILES[split])
    dataset = CropTypeAveragesFromTileDataset(tile_metadata, CROP_TYPES, FEATURES, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    csv_rows = []
    csv_rows.append(NEW_COLUMNS)
    i = -1

    # Loop through dataset
    for sample in dataloader:
        i += 1
        if i % 100 == 0:
            print('Processing tile', i)

        batch_size = len(sample['SIF'])
        crop_to_features = sample['features']
        area_fractions = sample['cover_fractions']
        print('Area fractions', area_fractions)

        # Loop through each tile in sample
        for j in range(batch_size):
            tile_features = []
            total_crop_cover = 0

            # For each crop, append its fractional area cover & feature list to the csv row
            for crop_type, crop_specific_features in crop_to_features.items():
                area_fraction = area_fractions[crop_type][j].item()
                total_crop_cover += area_fraction
                tile_features.append(area_fraction)
                tile_features.extend(crop_specific_features[j].tolist())

            # Create the csv row for this tile
            csv_row = [sample['lon'][j].item(), sample['lat'][j].item(), sample['date'][j], sample['tile_file'][j]] + tile_features + [sample['SIF'][j].item()]
            csv_rows.append(csv_row)

            # Ensure covers add up to 1
            assert(abs(total_crop_cover - 1) < FLOAT_EQUALITY_TOLERANCE)

    # Write rows to .csv file
    with open(CROP_AVERAGE_FILES[split], "w") as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            csv_writer.writerow(row)


    # for index, row in dataset.iterrows():
    #     i += 1
    #     if i % 100 == 0:
    #         print('Processing tile', i)
    #     tile_features = []

    #     # Tile assumed to be (band x lat x long)
    #     tile = np.load(row.loc['tile_file'])
    #     tile = transform(tile)
    #     # print('tile shape', tile.shape)

    #     # Reshape tile into (pixel x band)
    #     pixels = np.moveaxis(tile, 0, -1)
    #     pixels = pixels.reshape((-1, pixels.shape[2]))
    #     # print('after reshape', pixels.shape)
    #     # print('Random pixel', pixels[10000, :])
    #     # print('other random pixel', pixels[10001, :])

    #     for crop_type, crop_idx in CROP_TYPES.items():
    #         # Compute fraction land cover for this crop
    #         crop_cover = np.mean(pixels[:, crop_idx])
    #         tile_features.append(crop_cover)

    #         # Extract pixels belonging to this crop & are not obscured by clouds
    #         crop_pixels = pixels[(pixels[:, crop_idx] == 1) & (pixels[:, MISSING_IDX] == 1)]
    #         # print('crop cover', crop_cover)
    #         # print('Crop type', crop_type, 'shape', crop_pixels.shape)
    #         # print(crop_pixels.shape)
    #         for feature_name, feature_idx in FEATURES.items():
    #             if crop_pixels.shape[0] > 0:
    #                 feature_mean = np.mean(crop_pixels[:, feature_idx])
    #             else:
    #                 feature_mean = float('NaN')
    #             tile_features.append(feature_mean)

    #     # "any_cover": 1 if pixel is covered by any of the crop types; 0 if it is not
    #     any_cover = np.sum(pixels[:, CROP_INDICES], axis=1)
    #     other_pixels = pixels[any_cover == 0]  # Pixels not covered by any of the given crop types

    #     # Compute values for features that are not covered by any crop
    #     other_cover = 1 - np.mean(any_cover)
    #     tile_features.append(other_cover)
    #     for feature_name, feature_idx in FEATURES.items():
    #         if other_pixels.shape[0] > 0:
    #             feature_mean = np.mean(other_pixels[:, feature_idx])
    #         else:
    #             feature_mean = float('NaN')
    #         tile_features.append(feature_mean)

    #     csv_row = [row.loc['lon'], row.loc['lat'], row.loc['date'], row.loc['tile_file']] + tile_features + [row.loc['SIF'], row.loc['cloud_fraction'], row.loc['num_soundings']]
    #     csv_rows.append(csv_row)





