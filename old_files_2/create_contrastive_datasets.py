"""
Given large images (e.g. ~1-degree), create large tiles (of size 0.2, 0.33, and 0.5 degrees) 
that can be used for contrastive training.

The contrastive training method will take 2 random crops (of size 0.1 degrees) from a given large tile.

After producing these tiles, this script removes tiles with too much cloud cover or too little CDL coverage.
The script also filters the supervised training set (0.1 degree tiles with TROPOMI SIF labels) to remove
tiles with too much cloud cover, too little CDL coverage, or unreliable SIF.
"""
import csv
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import sif_utils

seed=3957472
np.random.seed(seed)

# File paths
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATES = ["2018-06-10", "2018-07-08", "2018-08-05", "2018-09-02"]

# , "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24",
#          "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
#          "2018-09-16"]

# Input files: labeled reflectance/cover/SIF tiles + images
DATASET_DIRS = [os.path.join(DATA_DIR, "dataset_" + date) for date in DATES]
UNFILTERED_TROPOMI_FILES = [os.path.join(dataset_dir, "reflectance_cover_to_sif.csv") for dataset_dir in DATASET_DIRS]
IMAGE_METADATA_FILES = [os.path.join(dataset_dir, "images.csv") for dataset_dir in DATASET_DIRS]

# Output files
PROCESSED_DATASET_DIR = os.path.join(DATA_DIR, "landsat_contrastive_datasets")
SPLIT_FILE = os.path.join(DATA_DIR, "split.pkl")
READ_SPLIT = True

if not os.path.exists(PROCESSED_DATASET_DIR):
    os.makedirs(PROCESSED_DATASET_DIR)
FILTERED_LABELED_FILES = {"train": os.path.join(PROCESSED_DATASET_DIR, "labeled_tiles_0.1_degrees_train.csv"),
                          "test": os.path.join(PROCESSED_DATASET_DIR, "labeled_tiles_0.1_degrees_test.csv")}
SPLIT_IMAGE_METADATA_FILES = {"train": os.path.join(PROCESSED_DATASET_DIR, "images_train.csv"),
                              "test": os.path.join(PROCESSED_DATASET_DIR, "images_test.csv")}
BAND_STATISTICS_CSV_FILES = {"train": os.path.join(PROCESSED_DATASET_DIR, "band_statistics_train.csv"),
                             "test": os.path.join(PROCESSED_DATASET_DIR, "band_statistics_test.csv")}

LARGE_TILE_DEGREE_SIZES = [0.2] #[0.33, 0.5]
UNLABELED_TILE_METADATA_TRAIN = [os.path.join(PROCESSED_DATASET_DIR, "unlabeled_tiles_" + str(tile_degrees) + "_degrees_train.csv")
                                 for tile_degrees in LARGE_TILE_DEGREE_SIZES]
UNLABELED_TILE_METADATA_TEST = [os.path.join(PROCESSED_DATASET_DIR, "unlabeled_tiles_" + str(tile_degrees) + "_degrees_test.csv")
                                for tile_degrees in LARGE_TILE_DEGREE_SIZES]
UNLABELED_TILE_DIRS = [os.path.join(DATA_DIR, "large_tiles_" + str(tile_degrees) + "_deg")
                             for tile_degrees in LARGE_TILE_DEGREE_SIZES]
for filename in UNLABELED_TILE_DIRS:
    if not os.path.exists(filename):
        os.makedirs(filename)


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

FRACTION_TEST = 0.3
RES = (0.00026949458523585647, 0.00026949458523585647)
CDL_INDICES = list(range(12, 42))
MISSING_REFLECTANCE_IDX = -1
MIN_CDL_COVERAGE = 0.5
MAX_LANDSAT_CLOUD_COVER = 0.5
MAX_TROPOMI_CLOUD_COVER = 0.2
MIN_TROPOMI_NUM_SOUNDINGS = 5
MIN_SIF = 0.2

if not READ_SPLIT:
    # Divide the region into 1x1 degree large grid areas. Split them between train/val/test
    LONS = list(range(-108, -82))  # These lat/lons are the UPPER LEFT corner of the large grid areas
    LATS = list(range(39, 50))
    large_grid_areas = dict()
    for lon in LONS:
        for lat in LATS:
            random_number = random.random()
            if random_number < 1 - FRACTION_TEST:
                split = 'train'
            else:
                split = 'test'
            large_grid_areas[(lon, lat)] = split
    with open(SPLIT_FILE, 'wb') as f:
        pickle.dump(large_grid_areas, f)
else:
    with open(SPLIT_FILE, 'rb') as f:
        large_grid_areas = pickle.load(f)

# Read TROPOMI files
tropomi_frames = []
for info_file in UNFILTERED_TROPOMI_FILES:
    tropomi_frame = pd.read_csv(info_file)
    tropomi_frame['source'] = 'TROPOMI'
    tropomi_frames.append(tropomi_frame)
tropomi_metadata = pd.concat(tropomi_frames)
tropomi_metadata.reset_index(drop=True, inplace=True)

# Read in all *large image* metadata
image_frames = []
for image_info_file in IMAGE_METADATA_FILES:
    image_frame = pd.read_csv(image_info_file)
    image_frames.append(image_frame)
image_metadata = pd.concat(image_frames)
image_metadata.reset_index(drop=True, inplace=True)

# Figure out which split each image/tile belongs to
tropomi_metadata['split'] = tropomi_metadata.apply(lambda row: sif_utils.determine_split(large_grid_areas, row), axis=1)
image_metadata['split'] = image_metadata.apply(lambda row: sif_utils.determine_split(large_grid_areas, row), axis=1)
print('Tropomi metadata', tropomi_metadata.head())

# Filter TROPOMI dataset
# Remove tiles with little CDL coverage (for the crops we're interested in)
tropomi_cdl_coverage = tropomi_metadata[CDL_COLUMNS].sum(axis=1)
sif_utils.plot_histogram(tropomi_cdl_coverage.to_numpy(), "tropomi_cdl_coverage.png")
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

# Create separate train/test TROPOMI datasets
split_tropomi_metadata = {'train': tropomi_metadata[tropomi_metadata['split'] == 'train'],
                          'test': tropomi_metadata[tropomi_metadata['split'] == 'test']}
for split, metadata in split_tropomi_metadata.items():
    # Write TROPOMI split to file
    print(split, len(metadata), 'tropomi')
    metadata.reset_index(drop=True, inplace=True)
    metadata.to_csv(FILTERED_LABELED_FILES[split])

    # Compute averages for each band
    selected_columns = metadata[STATISTICS_COLUMNS]
    band_means = selected_columns.mean(axis=0)
    band_stds = selected_columns.std(axis=0)

    # Write band averages to file
    statistics_rows = [['mean', 'std']]
    for i in range(len(band_means)):
        statistics_rows.append([band_means[i], band_stds[i]])
    with open(BAND_STATISTICS_CSV_FILES[split], 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in statistics_rows:
            csv_writer.writerow(row)

# Remove images with too much cloud cover, create separate train/test image datasets, and write to file
print('1-degree images before filtering', len(image_metadata))
image_metadata = image_metadata.loc[image_metadata['missing_reflectance'] <= MAX_LANDSAT_CLOUD_COVER]
print('After removing cloudy Landsat', len(image_metadata))
split_image_metadata = {'train': image_metadata[image_metadata['split'] == 'train'],
                        'test': image_metadata[image_metadata['split'] == 'test']}
for split, metadata in split_image_metadata.items():
    metadata.reset_index(drop=True, inplace=True)
    metadata.to_csv(SPLIT_IMAGE_METADATA_FILES[split])


# Select random rows
image_metadata = image_metadata.sample(frac=1)

# Build up train/test lists of "large tile info", for each tile size
unlabeled_tile_info_train = [[list(image_metadata.columns) + ['surrounding_image_file']] for tile_degrees in LARGE_TILE_DEGREE_SIZES]
unlabeled_tile_info_test = [[list(image_metadata.columns) + ['surrounding_image_file']] for tile_degrees in LARGE_TILE_DEGREE_SIZES]

# Loop through all image files
for idx, image_row in image_metadata.iterrows():
    # Load image
    image_filename = image_row['tile_file']
    image = np.load(image_filename)
    split = image_row['split']
    image_center_lat = image_row['lat']
    image_center_lon = image_row['lon']
    image_date = image_row['date']
    image_top_lat = image_center_lat + RES[0] * (image.shape[1] / 2)
    image_left_lon = image_center_lon - RES[1] * (image.shape[2] / 2)
    print('Image file', image_row['tile_file'], 'shape', image.shape, 'dtype', image.dtype)
    print('Image left lon, top lat', image_left_lon, image_top_lat)

    # Loop through all large-tile degree sizes we're trying
    for idx, large_tile_degrees in enumerate(LARGE_TILE_DEGREE_SIZES):
        LARGE_TILE_PIXELS = math.floor(large_tile_degrees / RES[0])
        
        # Get the appropriate output tile folder and metadata list (depending on if this is a train/test tile)
        if split == 'train':
            tile_info = unlabeled_tile_info_train[idx]
        else:
            tile_info = unlabeled_tile_info_test[idx]
        LARGE_TILE_DIR = UNLABELED_TILE_DIRS[idx]

        # Cut image into tiles of size "large_tile_degree_size" (there are multiple settings of this)
        bottom_idx_rounded = sif_utils.round_down(image.shape[1], LARGE_TILE_PIXELS)
        right_idx_rounded = sif_utils.round_down(image.shape[2], LARGE_TILE_PIXELS)
        top_indices = np.arange(0, bottom_idx_rounded, LARGE_TILE_PIXELS)
        left_indices = np.arange(0, right_idx_rounded, LARGE_TILE_PIXELS)
        print('Top indices', top_indices)
        print('Left indices', left_indices)
        print('Image shape', image.shape)
        for top_idx in top_indices:
            for left_idx in left_indices:
                bottom_idx = top_idx + LARGE_TILE_PIXELS
                right_idx = left_idx + LARGE_TILE_PIXELS
                if bottom_idx > image.shape[1]:
                    exit(1)
                if right_idx > image.shape[2]:
                    exit(1)

                # Extract tile from image
                large_tile = image[:, top_idx:bottom_idx, left_idx:right_idx]

                if np.isnan(large_tile).any():
                    print('ATTENTION: tile had NaNs!!!') # longitude', left_degrees, 'to', right_degrees, 'latitude', bottom_degrees, 'to', top_degrees)
                    continue
 
                # If too much is covered by clouds, or if CDL coverage is too low, remove this tile
                if np.mean(large_tile[MISSING_REFLECTANCE_IDX]) > MAX_LANDSAT_CLOUD_COVER:
                    # print('too much missing')
                    continue
                if np.mean(np.sum(large_tile[CDL_INDICES], axis=0)) < MIN_CDL_COVERAGE:
                    # print('too little cdl')
                    continue

                # Write large tile to .npy file
                large_tile_lon = round(image_left_lon + RES[1] * ((left_idx + right_idx) / 2), 2)
                large_tile_lat = round(image_top_lat - RES[0] * ((bottom_idx + top_idx) / 2), 2)
                # print('Top/left idx', top_idx, left_idx)
                large_tile_filename = os.path.join(LARGE_TILE_DIR, "image_lat_" + str(
                    large_tile_lat) + "_lon_" + str(large_tile_lon) + '_' + image_date + ".npy")
                print('Large tile filename', large_tile_filename)
                np.save(large_tile_filename, large_tile)

                # Record metadata in csv file
                average_input_features = sif_utils.compute_band_averages(large_tile, large_tile[MISSING_REFLECTANCE_IDX])
                csv_row = [large_tile_lon, large_tile_lat, image_date, large_tile_filename] + average_input_features.tolist() + [image_filename]
                tile_info.append(csv_row)

# Write info about each tile to the output csv file
for idx, tile_info in enumerate(unlabeled_tile_info_train):
    with open(UNLABELED_TILE_METADATA_TRAIN[idx], 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in tile_info:
            csv_writer.writerow(row)
for idx, tile_info in enumerate(unlabeled_tile_info_test):
    with open(UNLABELED_TILE_METADATA_TEST[idx], 'w') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in tile_info:
            csv_writer.writerow(row)


