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
import torch

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2_2018-08-01")
INFO_CSV_FILES = [os.path.join(DATASET_DIR, "reflectance_cover_to_sif.csv")] #,
                  #os.path.join(DATASET_DIR, "reflectance_cover_to_sif_r2.csv")]
#                  os.path.join(DATA_DIR, "dataset_2019-07-16/reflectance_cover_to_sif.csv")]
SPLIT_INFO_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_info_train.csv"),
                        "val": os.path.join(DATASET_DIR, "tile_info_val.csv")}
TILE_AVERAGE_CSV_FILES = {"train": os.path.join(DATASET_DIR, "tile_averages_train.csv"),
                          "val": os.path.join(DATASET_DIR, "tile_averages_val.csv")}
BAND_STATISTICS_CSV_FILES = {"train": os.path.join(DATASET_DIR, "band_statistics_train.csv"),
                             "val": os.path.join(DATASET_DIR, "band_statistics_val.csv")}
MAX_CLOUD_COVER = 0.1
CDL_BANDS = list(range(12, 42))
MIN_SIF = 0.2
MIN_CDL_COVERAGE = 0.5  # Throw out tiles if less than this raction of land cover is unknown

frames = []
for info_file in INFO_CSV_FILES:
    frames.append(pd.read_csv(info_file))
tile_metadata = pd.concat(frames)
tile_metadata.reset_index(drop=True, inplace=True)
print('(Unfiltered) number of large tiles:', len(tile_metadata))

# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)


# Split into train/val, and write the split to a file (to ensure that all methods use the same
# train/val split)
train_set, val_set = sklearn.model_selection.train_test_split(tile_metadata, test_size=0.2)
datasets = {"train": train_set, "val": val_set}

for split in ["train", "val"]:
    # Create a dataset of tile-average values. Also, compute mean/std of each band
    csv_rows = []
    column_names = ['date', 'tile_file', 'lat', 'lon', 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
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
    valid_indices = []  # Indices of tiles to include
    sifs = []
    dataset.reset_index(drop=True, inplace=True)
    print(dataset.head(5))
    i = -1
    for index, row in dataset.iterrows():
        i += 1
        if i % 100 == 0:
            print('Processing tile', i)
        # Tile assumed to be (band x lat x long)
        tile = np.load(row.loc['tile_file'])
        # print('Tile', tile.shape, 'dtype', tile.dtype)
        tile_averages = torch.mean(torch.tensor(tile).to(device), dim=(1,2)).cpu().numpy()

        # Remove tiles with any NaNs
        if np.isnan(tile_averages).any():
            print('tile contained nan:', row.loc['tile_file'])
            continue

        # Remove tiles with little CDL coverage (for the crops we're interested in)
        cdl_coverage = np.sum(tile_averages[CDL_BANDS])
        if cdl_coverage < MIN_CDL_COVERAGE:
            #print('CDL coverage too low:', cdl_coverage)
            #print(row.loc['tile_file'])
            continue
        #print('FLDAS average', tile_averages[10])
 
        # If too much of this pixel is covered by clouds (reflectance
        # data is missing), throw this tile out
        if tile_averages[-1] > MAX_CLOUD_COVER:
            continue

        # Remove tiles with low SIF (those observations may be unreliable)
        if float(row.loc['SIF']) < MIN_SIF:
            continue

        csv_row = [row.loc['date'], row.loc['tile_file'], row.loc['lat'], row.loc['lon']] + tile_averages.tolist() + [row.loc['SIF']]
        csv_rows.append(csv_row)
        band_averages_all_tiles.append(tile_averages)
        valid_indices.append(i)
        sifs.append(row.loc['SIF'])

    # Select the rows that are mentioned in "valid indices" (low cloud cover)
    filtered_dataset = dataset.iloc[valid_indices, :]
    print('Filtered tiles', len(filtered_dataset))
    band_averages_array = np.stack(band_averages_all_tiles)
    print("Band values ARRAY shape", band_averages_array.shape)
    band_means = np.mean(band_averages_array, axis=0)
    band_stds = np.std(band_averages_array, axis=0)
    print("Band means", band_means)
    print("Band stds", band_stds)

    # Write filtered tile-info dataset to file, to record split
    filtered_dataset.to_csv(SPLIT_INFO_CSV_FILES[split], index=False)

    # Write rows to .csv file
    with open(TILE_AVERAGE_CSV_FILES[split], "w") as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            csv_writer.writerow(row)

    # Write band averages to file
    statistics_rows = [['mean', 'std']]
    for i in range(len(band_means)):
        statistics_rows.append([band_means[i], band_stds[i]])

    statistics_rows.append([np.mean(sifs), np.std(sifs)])
    with open(BAND_STATISTICS_CSV_FILES[split], 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in statistics_rows:
            csv_writer.writerow(row)


