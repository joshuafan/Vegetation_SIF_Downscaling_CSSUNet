"""
Randomly splits tiles into 80% train, 20% validation.
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

DATASET_DIR = "datasets/dataset_2016-08-01"
INFO_CSV_FILE = os.path.join(DATASET_DIR, "reflectance_cover_to_sif.csv")
SPLIT_INFO_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_info_train.csv"),
                        "val": os.path.join(DATASET_DIR, "tile_info_val.csv")}
TILE_AVERAGE_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_averages_train.csv"),
                          "val": os.path.join(DATASET_DIR, "tile_averages_val.csv")}
BAND_STATISTICS_CSV_FILES = {"train": os.path.join(DATASET_DIR, "band_statistics_train.csv"),
                             "val": os.path.join(DATASET_DIR, "band_statistics_val.csv")}
tile_metadata = pd.read_csv(INFO_CSV_FILE)

# Split into train/val, and write the split to a file (to ensure that all methos use the same
# train/val split)
train_set, val_set = sklearn.model_selection.train_test_split(tile_metadata, test_size=0.2)
datasets = {"train": train_set, "val": val_set}

for split in ["train", "val"]:
    # Write original tile-info dataset to file, to record split
    datasets[split].to_csv(SPLIT_INFO_CSV_FILES[split])

    # Create a dataset of tile-average values. Also, compute mean/std of each band
    csv_rows = []
    column_names = ['lat', 'lon', 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'corn', 'soybean', 'grassland', 'deciduous_forest',
                    'percent_missing', 'SIF']
    csv_rows.append(column_names)

    band_values_all_tiles = []
    dataset = datasets[split]
    for index, row in dataset.iterrows():
        # Tile assumed to be (band x lat x long)
        tile = np.load(row.loc['tile_file'])
        print('Tile', tile.shape, 'dtype', tile.dtype)
        tile_averages = np.mean(tile, axis=(1,2))
        csv_row = [row.loc['lat'], row.loc['lon']] + tile_averages.tolist() + [row.loc['SIF']]
        csv_rows.append(csv_row)
        band_values = tile.reshape(tile.shape[0], -1)
        print("Band values shape", band_values.shape)
        band_values_all_tiles.append(band_values)
    band_values_array = np.concatenate(band_values_all_tiles, axis=1)
    print("Band values ARRAY shape", band_values_array.shape)
    band_means = np.mean(band_values_array, axis=1).flatten()
    band_stds = np.std(band_values_array, axis=1).flatten()
    print("Band means", band_means)

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
    sifs = dataset['SIF']
    statistics_rows.append([sifs.mean(), sifs.std()])
    with open(BAND_STATISTICS_CSV_FILES[split], 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in statistics_rows:
            csv_writer.writerow(row)


