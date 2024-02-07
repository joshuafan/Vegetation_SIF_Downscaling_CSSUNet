"""
Double-check what inductive biases tend to hold on fine-resolution SIF. For example pixels, compare the SIF distributions of

(1) Nearby pixels with same land cover
(2) Nearby pixels with different land cover
(3) Faraway pixels with same land cover and similar reflectance
(4) Faraway pixels with same land cover and different reflectance
(5) Faraway pixels with different land cover

This will inform the exact nature of the contrastive loss. For example, if (1) and (3) tend to have more similar SIFs
than (2), (4), and (5), then (1) and (3) could provide positive pairs (encouraging the representation of those
similar pixels to be similar), whereas (2), (4), (5) could provide negative pairs (whose representation should be different).
"""

import argparse
import json
import math
import os
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from scipy.stats import pearsonr, spearmanr
import scipy.spatial as spatial
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sif_utils import plot_histogram, print_stats, density_scatter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set random seed for data shuffling
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Folds
TRAIN_FOLDS = [0, 1, 2]
VAL_FOLDS = [3]
TEST_FOLDS = [4]

# Spread parameter in similarity function. The higher the value, the faster the function decays towards 0 as you get further away.
TAU = 0.5

# Directories
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF"
METADATA_DIR = os.path.join(DATA_DIR, "metadata/CFIS_OCO2_dataset")
PLOT_DIR = os.path.join(DATA_DIR, "exploratory_plots/euclidean_std_smoothness")
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Train files
RESOLUTION = 30
CFIS_COARSE_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_coarse_metadata.csv')
CFIS_FINE_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_metadata_' + str(RESOLUTION) + 'm.csv')
OCO2_METADATA_FILE = os.path.join(METADATA_DIR, 'oco2_metadata.csv')
BAND_STATISTICS_FILE = os.path.join(METADATA_DIR, 'cfis_band_statistics_train.csv')
NORMALIZED_BAND_STATISTICS_FILE = os.path.join(METADATA_DIR, 'normalized_cfis_band_statistics_train.csv')

# Only include CFIS tiles where at least this fraction of pixels have CFIS
# fine-resolution data
MIN_COARSE_FRACTION_VALID_PIXELS = 0.1

# Only EVALUATE on CFIS fine-resolution pixels with at least this number of soundings (measurements)
MIN_FINE_CFIS_SOUNDINGS = 30

# For resolutions greater than 30m, only evaluate on grid cells where at least this fraction
# of 30m pixels have any CFIS data
eps = 1e-5
MIN_FINE_FRACTION_VALID_PIXELS = 0.9-eps

DATES = ["2016-06-15", "2016-08-01"]
TRAIN_DATES = ["2016-06-15", "2016-08-01"]
TEST_DATES = ["2016-06-15", "2016-08-01"]

# For evaluation purposes, we consider a grid cell to be "pure" if at least this fraction
# of the cell is of a given land cover type
PURE_THRESHOLD = 0.7

# Only train on OCO-2 datapoints with at least this number of soundings
MIN_OCO2_SOUNDINGS = 3

# Remove OCO-2 and CFIS tiles with cloud cover that exceeds this threshold
MAX_OCO2_CLOUD_COVER = 0.5
MAX_CFIS_CLOUD_COVER = 0.5

# Clip inputs to this many standard deviations from mean
MIN_INPUT = -3
MAX_INPUT = 3

# Clip SIF predictions to be within this range, and exclude
# datapoints whose true SIF is outside this range
MIN_SIF_CLIP = 0.1
MAX_SIF_CLIP = None

# Range of SIF values to plot
MIN_SIF_PLOT = 0
MAX_SIF_PLOT = 1.5

INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7']
                # 'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
REFLECTANCE_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7']  #, 'ref_10', 'ref_11']
COLUMNS_TO_STANDARDIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7']
                         # 'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
# COLUMNS_TO_NORMALIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7']  #, 'ref_10', 'ref_11']
OUTPUT_COLUMN = ["SIF"]
ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest']

STATISTICS_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                      'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
                      'grassland_pasture', 'corn', 'soybean', 'shrubland',
                      'deciduous_forest', 'evergreen_forest', 'spring_wheat',
                      'developed_open_space', 'other_hay_non_alfalfa', 'winter_wheat',
                      'herbaceous_wetlands', 'woody_wetlands', 'open_water', 'alfalfa',
                      'fallow_idle_cropland', 'sorghum', 'developed_low_intensity',
                      'barren', 'durum_wheat',
                      'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                      'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                      'lentils', 'missing_reflectance', 'SIF']

# Order of columns in band averages file
BAND_AVERAGES_COLUMN_ORDER =  ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                                'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
                                'grassland_pasture', 'corn', 'soybean', 'shrubland',
                                'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                                'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                                'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                                'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                                'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                                'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                                'lentils', 'missing_reflectance']

# Filter OCO2 tiles
oco2_metadata = pd.read_csv(OCO2_METADATA_FILE)
oco2_metadata = oco2_metadata[(oco2_metadata['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                                (oco2_metadata['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                                (oco2_metadata['SIF'] >= MIN_SIF_CLIP)]
oco2_metadata = oco2_metadata[oco2_metadata[ALL_COVER_COLUMNS].sum(axis=1) >= 0.5]

# Read CFIS coarse datapoints - only include CFIS tiles with enough valid pixels
cfis_coarse_metadata = pd.read_csv(CFIS_COARSE_METADATA_FILE)
cfis_coarse_metadata = cfis_coarse_metadata[(cfis_coarse_metadata['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                            (cfis_coarse_metadata['SIF'] >= MIN_SIF_CLIP) &
                                            (cfis_coarse_metadata['missing_reflectance'] <= MAX_CFIS_CLOUD_COVER)]
cfis_coarse_metadata = cfis_coarse_metadata[cfis_coarse_metadata[ALL_COVER_COLUMNS].sum(axis=1) >= 0.5]

# Read CFIS fine datapoints - only include pixels with enough soundings and are in coarse tiles
cfis_fine_metadata = pd.read_csv(CFIS_FINE_METADATA_FILE)
cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['SIF'] >= MIN_SIF_CLIP) &
                                (cfis_fine_metadata['tile_file'].isin(set(cfis_coarse_metadata['tile_file'])))]
cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['num_soundings'] >= MIN_FINE_CFIS_SOUNDINGS) &
                                        (cfis_fine_metadata['fraction_valid'] >= MIN_FINE_FRACTION_VALID_PIXELS)]  # Avoid roundoff errors

# Compute NDVI
oco2_metadata["NDVI"] = (oco2_metadata["ref_5"] - oco2_metadata["ref_4"]) / (oco2_metadata["ref_5"] + oco2_metadata["ref_4"])
cfis_coarse_metadata["NDVI"] = (cfis_coarse_metadata["ref_5"] - cfis_coarse_metadata["ref_4"]) / (cfis_coarse_metadata["ref_5"] + cfis_coarse_metadata["ref_4"])
cfis_fine_metadata["NDVI"] = (cfis_fine_metadata["ref_5"] - cfis_fine_metadata["ref_4"]) / (cfis_fine_metadata["ref_5"] + cfis_fine_metadata["ref_4"])

# Compute ratio index
oco2_metadata["NIR_over_R"] = (oco2_metadata["ref_5"] / oco2_metadata["ref_4"])
cfis_coarse_metadata["NIR_over_R"] = (cfis_coarse_metadata["ref_5"] / cfis_coarse_metadata["ref_4"])
cfis_fine_metadata["NIR_over_R"] = (cfis_fine_metadata["ref_5"] / cfis_fine_metadata["ref_4"])

# Compute cover type
oco2_metadata["cover_type"] = oco2_metadata[ALL_COVER_COLUMNS].idxmax(axis=1)
cfis_coarse_metadata["cover_type"] = cfis_coarse_metadata[ALL_COVER_COLUMNS].idxmax(axis=1)
cfis_fine_metadata["cover_type"] = cfis_fine_metadata[ALL_COVER_COLUMNS].idxmax(axis=1)

# Read dataset splits
oco2_train_set = oco2_metadata[(oco2_metadata['fold'].isin(TRAIN_FOLDS)) &
                                (oco2_metadata['date'].isin(TRAIN_DATES))].copy()
oco2_val_set = oco2_metadata[(oco2_metadata['fold'].isin(VAL_FOLDS)) &
                                (oco2_metadata['date'].isin(TRAIN_DATES))].copy()
oco2_test_set = oco2_metadata[(oco2_metadata['fold'].isin(TEST_FOLDS)) &
                                (oco2_metadata['date'].isin(TEST_DATES))].copy()
coarse_train_set = cfis_coarse_metadata[(cfis_coarse_metadata['fold'].isin(TRAIN_FOLDS)) &
                                        (cfis_coarse_metadata['date'].isin(TRAIN_DATES))].copy()
coarse_val_set = cfis_coarse_metadata[(cfis_coarse_metadata['fold'].isin(VAL_FOLDS)) &
                                        (cfis_coarse_metadata['date'].isin(TRAIN_DATES))].copy()
coarse_test_set = cfis_coarse_metadata[(cfis_coarse_metadata['fold'].isin(TEST_FOLDS)) &
                                        (cfis_coarse_metadata['date'].isin(TEST_DATES))].copy()
fine_train_set = cfis_fine_metadata[(cfis_fine_metadata['fold'].isin(TRAIN_FOLDS)) &
                                        (cfis_fine_metadata['date'].isin(TRAIN_DATES))].copy()
fine_val_set = cfis_fine_metadata[(cfis_fine_metadata['fold'].isin(VAL_FOLDS)) &
                                        (cfis_fine_metadata['date'].isin(TRAIN_DATES))].copy()
fine_test_set = cfis_fine_metadata[(cfis_fine_metadata['fold'].isin(TEST_FOLDS)) &
                                        (cfis_fine_metadata['date'].isin(TEST_DATES))].copy()
train_set = pd.concat([oco2_train_set, coarse_train_set])

# Read band statistics
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

# # Try injecting mult noise
# fine_train_set = fine_train_set.reset_index(drop=True)
# for idx in range(len(fine_train_set)):
#     # print("Initial bands")
#     # print(train_set.loc[idx, REFLECTANCE_BANDS])
#     noise = 1 + np.random.normal(loc=0, scale=0.2)
#     fine_train_set.loc[idx, REFLECTANCE_COLUMNS] = fine_train_set.loc[idx, REFLECTANCE_COLUMNS] * noise


# ================= Standardize ===================
# OPTION 1: Standardize data based on pre-computed means/stds
for column in COLUMNS_TO_STANDARDIZE:
    idx = BAND_AVERAGES_COLUMN_ORDER.index(column)
    train_set[column] = np.clip((train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    coarse_val_set[column] = np.clip((coarse_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    fine_train_set[column] = np.clip((fine_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    fine_val_set[column] = np.clip((fine_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    fine_test_set[column] = np.clip((fine_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)

# # OPTION 2: Normalize brightness, then standardize
# # Normalize each pixel's vector to have length equal to the average
# fine_train_set[COLUMNS_TO_NORMALIZE] = fine_train_set[COLUMNS_TO_NORMALIZE] / np.linalg.norm(fine_train_set[COLUMNS_TO_NORMALIZE].values, axis=1, keepdims=True)
# fine_val_set[COLUMNS_TO_NORMALIZE] = fine_val_set[COLUMNS_TO_NORMALIZE] / np.linalg.norm(fine_val_set[COLUMNS_TO_NORMALIZE].values, axis=1, keepdims=True)
# print("After normalize", fine_train_set.head())

# # Compute means/stds of the normalized version
# selected_columns = fine_train_set[STATISTICS_COLUMNS]
# band_means = selected_columns.mean(axis=0)
# band_stds = selected_columns.std(axis=0)
# print("Band means", band_means)

# # Write band averages to file
# statistics_rows = [['mean', 'std']]
# for i, mean in enumerate(band_means):
#     statistics_rows.append([band_means[i], band_stds[i]])
# with open(NORMALIZED_BAND_STATISTICS_FILE, 'w') as output_csv_file:
#     csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
#     for row in statistics_rows:
#         csv_writer.writerow(row)

# # Now do StandardScaler or min-max scaler
# scaler = StandardScaler().fit(fine_train_set[COLUMNS_TO_STANDARDIZE])
# fine_train_set[COLUMNS_TO_STANDARDIZE] = scaler.transform(fine_train_set[COLUMNS_TO_STANDARDIZE])
# fine_val_set[COLUMNS_TO_STANDARDIZE] = scaler.transform(fine_val_set[COLUMNS_TO_STANDARDIZE])
# print("After both normalize and standardize", fine_train_set.head())

# ==================== Extract data =======================
# Extract X/Y
X_fine_train = fine_train_set[INPUT_COLUMNS].to_numpy()
Y_fine_train = fine_train_set[OUTPUT_COLUMN].values.ravel()
print("X fine train", X_fine_train.shape)
print("Y fine train", Y_fine_train.shape)
X_fine_val = fine_val_set[INPUT_COLUMNS].to_numpy()
Y_fine_val = fine_val_set[OUTPUT_COLUMN].values.ravel()

# List of lists, of SIF differences between these types of pixel pairs. Each inner list represents differences from a single anchor pixel.
nearby_same_cover = []
nearby_diff_cover = []
distant_same_cover_similar_reflectance = []
distant_same_cover_different_reflectance = []
distant_diff_cover = []

# Helper function to compute distance between this row's reflectance vector and an anchor's reflectance vector
def compute_reflectance_distance(row, anchor_reflectance_vector):
    return np.linalg.norm(row[REFLECTANCE_COLUMNS] - anchor_reflectance_vector)

def compute_reflectance_similarity(row, anchor_reflectance_vector, tau):
    distance = np.linalg.norm(row[REFLECTANCE_COLUMNS] - anchor_reflectance_vector)
    return np.exp(-tau * (distance ** 2))


for i in range(100):
    anchor_idx = random.randrange(len(fine_train_set))
    anchor_row = fine_train_set.iloc[anchor_idx]
    anchor_tile_file = anchor_row['tile_file']
    anchor_cover_type = anchor_row['cover_type']
    anchor_sif = anchor_row['SIF']
    anchor_lat = anchor_row['lat']
    anchor_lon = anchor_row['lon']
    anchor_reflectance_vector = anchor_row[REFLECTANCE_COLUMNS].to_numpy()

    nearby_rows = fine_train_set[(fine_train_set['tile_file'] == anchor_tile_file) &
                                 (fine_train_set['lat'] < anchor_lat + 0.005) &
                                 (fine_train_set['lat'] > anchor_lat - 0.005) &
                                 (fine_train_set['lon'] < anchor_lon + 0.005) &
                                 (fine_train_set['lon'] > anchor_lon - 0.005)]
    nearby_same_cover_rows = nearby_rows[nearby_rows['cover_type'] == anchor_cover_type]
    nearby_diff_cover_rows = nearby_rows[nearby_rows['cover_type'] != anchor_cover_type]
    distant_rows = fine_train_set[fine_train_set['tile_file'] != anchor_tile_file]
    distant_same_cover_rows = distant_rows[distant_rows['cover_type'] == anchor_cover_type].copy()
    distant_diff_cover_rows = distant_rows[distant_rows['cover_type'] != anchor_cover_type]

    # For distant same cover, compute distance between their reflectance vectors
    distant_same_cover_rows['reflectance_distance_to_anchor'] = distant_same_cover_rows.apply(lambda row: compute_reflectance_distance(row, anchor_reflectance_vector), axis=1)
    distant_same_cover_rows['reflectance_similarity_to_anchor'] = distant_same_cover_rows.apply(lambda row: compute_reflectance_similarity(row, anchor_reflectance_vector, TAU), axis=1)

    print("Reflectance distance quantiles.",
          '50%', np.quantile(distant_same_cover_rows['reflectance_distance_to_anchor'], 0.5),
          '10%', np.quantile(distant_same_cover_rows['reflectance_distance_to_anchor'], 0.1),
          '5%', np.quantile(distant_same_cover_rows['reflectance_distance_to_anchor'], 0.05))
    print("Reflectance similarity quantiles.",
          '50%', np.quantile(distant_same_cover_rows['reflectance_similarity_to_anchor'], 0.5),
          '90%', np.quantile(distant_same_cover_rows['reflectance_similarity_to_anchor'], 0.9),
          '95%', np.quantile(distant_same_cover_rows['reflectance_similarity_to_anchor'], 0.95))
    SIMILAR_REFLECTANCE_THRESHOLD = 0.8 #2 #np.quantile(distant_same_cover_rows['reflectance_distance_to_anchor'], 0.05)
    DIFFERENT_REFLECTANCE_THRESHOLD = 0.8  #2
    distant_same_cover_similar_reflectance_rows = distant_same_cover_rows[distant_same_cover_rows['reflectance_similarity_to_anchor'] > SIMILAR_REFLECTANCE_THRESHOLD]
    distant_same_cover_different_reflectance_rows = distant_same_cover_rows[distant_same_cover_rows['reflectance_similarity_to_anchor'] < DIFFERENT_REFLECTANCE_THRESHOLD]

    # Compute SIF differences and save them
    nearby_same_cover_differences = np.abs(nearby_same_cover_rows['SIF'] - anchor_sif)
    nearby_same_cover.append(nearby_same_cover_differences)
    nearby_diff_cover_differences = np.abs(nearby_diff_cover_rows['SIF'] - anchor_sif)
    nearby_diff_cover.append(nearby_diff_cover_differences)
    distant_same_cover_similar_reflectance_differences = np.abs(distant_same_cover_similar_reflectance_rows['SIF'] - anchor_sif)
    distant_same_cover_similar_reflectance.append(distant_same_cover_similar_reflectance_differences)
    distant_same_cover_different_reflectance_differences = np.abs(distant_same_cover_different_reflectance_rows['SIF'] - anchor_sif)
    distant_same_cover_different_reflectance.append(distant_same_cover_different_reflectance_differences)
    distant_diff_cover_differences = np.abs(distant_diff_cover_rows['SIF'] - anchor_sif)
    distant_diff_cover.append(distant_diff_cover_differences)

    # Plot histograms with common range, for single anchor pixel
    if i % 20 == 0:
        hist_range = (np.min(distant_diff_cover_differences), np.max(distant_diff_cover_differences))
        fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, figsize=(15, 11))
        ax1.axis('off')
        ax0.hist(nearby_same_cover_differences, range=hist_range, bins=20)
        ax0.set_title("Nearby same cover: mean " + str(round(np.mean(nearby_same_cover_differences), 3)))
        ax2.hist(nearby_diff_cover_differences, range=hist_range, bins=20)
        ax2.set_title("Nearby different cover: mean " + str(round(np.mean(nearby_diff_cover_differences), 3)))
        ax3.hist(distant_same_cover_similar_reflectance_differences, range=hist_range, bins=20)
        ax3.set_title("Distant same cover, similar reflectance: mean " + str(round(np.mean(distant_same_cover_similar_reflectance_differences), 3)))
        ax4.hist(distant_same_cover_different_reflectance_differences, range=hist_range, bins=20)
        ax4.set_title("Distant same cover, different reflectance: mean " + str(round(np.mean(distant_same_cover_different_reflectance_differences), 3)))
        ax5.hist(distant_diff_cover_differences, range=hist_range, bins=20)
        ax5.set_title("Distant different cover: mean " + str(round(np.mean(distant_diff_cover_differences), 3)))
        fig.suptitle("Differences with pixel, lat " + str(round(anchor_lat, 4)) + " lon " + str(round(anchor_lon, 4)))
                    # "\n Nearby = within a square of radius 0.005 degrees\nSimilar reflectance: Euclidean distance < 0.05\n Different reflectance: Euclidean distance > 0.05")
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(PLOT_DIR, "contrastive_test_" + os.path.basename(anchor_tile_file) + ".png"))
        plt.close()

# Plot histograms
nearby_same_cover = np.concatenate(nearby_same_cover)
nearby_diff_cover = np.concatenate(nearby_diff_cover)
distant_same_cover_similar_reflectance = np.concatenate(distant_same_cover_similar_reflectance)
distant_same_cover_different_reflectance = np.concatenate(distant_same_cover_different_reflectance)
distant_diff_cover = np.concatenate(distant_diff_cover)
print("Distant diff cover", distant_diff_cover)
print("Min", np.min(distant_diff_cover))

# Plot histograms with common range - for all anchor pixels
hist_range = (np.min(distant_diff_cover), np.max(distant_diff_cover))
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, figsize=(15, 11))
ax1.axis('off')
ax0.hist(nearby_same_cover, range=hist_range, bins=20)
ax0.set_title("Nearby same cover: mean " + str(round(np.mean(nearby_same_cover), 3)))
ax2.hist(nearby_diff_cover, range=hist_range, bins=20)
ax2.set_title("Nearby different cover: mean " + str(round(np.mean(nearby_diff_cover), 3)))
ax3.hist(distant_same_cover_similar_reflectance, range=hist_range, bins=20)
ax3.set_title("Distant same cover, similar reflectance: mean " + str(round(np.mean(distant_same_cover_similar_reflectance), 3)))
ax4.hist(distant_same_cover_different_reflectance, range=hist_range, bins=20)
ax4.set_title("Distant same cover, different reflectance: mean " + str(round(np.mean(distant_same_cover_different_reflectance), 3)))
ax5.hist(distant_diff_cover, range=hist_range, bins=20)
ax5.set_title("Distant different cover: mean " + str(round(np.mean(distant_diff_cover), 3)))
fig.suptitle("Differences combined")
fig.tight_layout()
fig.subplots_adjust(top=0.9)
plt.savefig(os.path.join(PLOT_DIR, "contrastive_test_all.png"))
plt.close()
