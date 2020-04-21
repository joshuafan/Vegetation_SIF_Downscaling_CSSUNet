"""
Randomly splits tiles into 80% train, 20% validation.
Throws out tiles with cloud cover fraction exceeding "MAX_CLOUD_COVER".
Creates a dataset of tile averages.
Computes mean/std of each band
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.model_selection
from sif_utils import plot_histogram

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-07-17")
INFO_CSV_FILE = os.path.join(DATASET_DIR, "reflectance_cover_to_sif.csv")
SPLIT_INFO_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_info_train.csv"),
                        "val": os.path.join(DATASET_DIR, "tile_info_val.csv")}
TILE_AVERAGE_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_averages_train.csv"),
                          "val": os.path.join(DATASET_DIR, "tile_averages_val.csv")}
BAND_STATISTICS_CSV_FILES = {"train": os.path.join(DATASET_DIR, "band_statistics_train.csv"),
                             "val": os.path.join(DATASET_DIR, "band_statistics_val.csv")}
MAX_CLOUD_COVER = 0.1
MIN_SIF = 0.2

tile_metadata = pd.read_csv(INFO_CSV_FILE)

# Split into train/val, and write the split to a file (to ensure that all methods use the same
# train/val split)
train_set, val_set = sklearn.model_selection.train_test_split(tile_metadata, test_size=0.2)
datasets = {"train": train_set, "val": val_set}

for split in ["train", "val"]:
    # Create a dataset of tile-average values. Also, compute mean/std of each band
    csv_rows = []
    column_names = ['lat', 'lon', 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils', 'missing_reflectance', 'SIF']
    csv_rows.append(column_names)

    band_averages_all_tiles = []
    dataset = datasets[split]
    valid_indices = []  # Indices of tiles which have low cloud cover
    sifs = []

    i = 0
    for index, row in dataset.iterrows():
        # Tile assumed to be (band x lat x long)
        tile = np.load(row.loc['tile_file'])
        # print('Tile', tile.shape, 'dtype', tile.dtype)
        tile_averages = np.mean(tile, axis=(1,2))
        
        # If too much of this pixel is covered by clouds (reflectance
        # data is missing), throw this tile out
        if tile_averages[-1] > MAX_CLOUD_COVER:
            continue
        if float(row.loc['SIF']) < MIN_SIF:
            continue

        csv_row = [row.loc['lat'], row.loc['lon']] + tile_averages.tolist() + [row.loc['SIF']]
        csv_rows.append(csv_row)
        band_averages_all_tiles.append(tile_averages)
        valid_indices.append(index)
        sifs.append(row.loc['SIF'])
        i += 1
        print('Processing tile', i)

    # Select the rows that are mentioned in "valid indices" (low cloud cover)
    filtered_dataset = dataset.iloc[dataset.index.isin(valid_indices)]
    print('Filtered tiles', len(filtered_dataset))
    band_averages_array = np.stack(band_averages_all_tiles)
    print("Band values ARRAY shape", band_averages_array.shape)
    band_means = np.mean(band_averages_array, axis=0)
    band_stds = np.std(band_averages_array, axis=0)
    print("Band means", band_means)
    print("Band stds", band_stds)

    # Write filtered tile-info dataset to file, to record split
    filtered_dataset.to_csv(SPLIT_INFO_CSV_FILES[split])


    # Write rows to .csv file
    with open(TILE_AVERAGE_CSV_FILES[split], "w") as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            csv_writer.writerow(row)

    # Plot histogram of each column
    #data_array = np.array(csv_rows[1:])  # Ignore header row
    #for i in range(len(column_names)):
    #    print("Column:", column_names[i])
    #    plot_histogram(data_array[:, i], column_names[i] + "_" + split + ".png")

    # Write band averages to file
    statistics_rows = [['mean', 'std']]
    for i in range(len(band_means)):
        statistics_rows.append([band_means[i], band_stds[i]])

    statistics_rows.append([np.mean(sifs), np.std(sifs)])
    with open(BAND_STATISTICS_CSV_FILES[split], 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in statistics_rows:
            csv_writer.writerow(row)


