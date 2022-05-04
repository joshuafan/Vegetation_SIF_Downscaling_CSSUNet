"""
Creates plots of input similarity between pixels vs. SIF distances. The expectation is that pixels with high similarity should have smaller differences in SIF.
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
SIMILARITY_RANGES = [(0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1)]

# Directories
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF"
METADATA_DIR = os.path.join(DATA_DIR, "metadata/CFIS_OCO2_dataset")

# Train files
RESOLUTION = 30
CFIS_COARSE_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_coarse_metadata.csv')
CFIS_FINE_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_metadata_' + str(RESOLUTION) + 'm.csv')
OCO2_METADATA_FILE = os.path.join(METADATA_DIR, 'oco2_metadata.csv')
BAND_STATISTICS_FILE = os.path.join(METADATA_DIR, 'cfis_band_statistics_train.csv')

PLOT_DIR = os.path.join(DATA_DIR, "exploratory_plots/euclidean_std_similarity_tau0.5")
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

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

# # TODO - changed
# DATES = ["2016-06-15"]
# TRAIN_DATES = ["2016-06-15"]
# TEST_DATES = ["2016-06-15"]

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


# Input feature names - only include 7 reflectance bands
INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7']
#                'ref_10', 'ref_11']  #, 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
# INPUT_COLUMNS = ['NDVI']
# INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
#                     'ref_10', 'ref_11']
# COLUMNS_TO_STANDARDIZE = []

# INPUT_COLUMNS = ['ref_3', 'ref_4', 'ref_5']
# INPUT_COLUMNS = ['NDVI']

# COLUMNS_TO_STANDARDIZE = ['ref_4', 'ref_5', 'ref_6']
# INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
#                     'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
#                     'grassland_pasture', 'corn', 'soybean',
#                     'deciduous_forest', 'evergreen_forest', 'developed_open_space',
#                     'woody_wetlands', 'open_water', 'alfalfa',
#                     'developed_low_intensity', 'developed_med_intensity', 'missing_reflectance']
# COLUMNS_TO_NORMALIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7', 'ref_10', 'ref_11']
# COLUMNS_TO_STANDARDIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
#                           'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']  #, 'NDVI']
# COLUMNS_TO_NORMALIZE = list(range(0, 9))

COLUMNS_TO_STANDARDIZE = list(range(0, 7))
OUTPUT_COLUMN = ["SIF"]
ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest']




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
# fine_train_set = fine_train_set[(fine_train_set['soybean'] > 0.7)]
print("Length of fine train set", len(fine_train_set))
train_set = pd.concat([oco2_train_set, coarse_train_set])

# Read band statistics
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]



# # Standardize data
# for idx, column in enumerate(COLUMNS_TO_STANDARDIZE):
#     train_set[column] = np.clip((train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
#     coarse_val_set[column] = np.clip((coarse_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
#     fine_train_set[column] = np.clip((fine_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
#     fine_val_set[column] = np.clip((fine_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
#     fine_test_set[column] = np.clip((fine_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
#     # if False: #"ref" in column:
#     #     train_set[column] = train_set[column] / 10000
#     #     coarse_val_set[column] = coarse_val_set[column] / 10000
#     #     fine_train_set[column] = fine_train_set[column] / 10000
#     #     fine_val_set[column] = fine_val_set[column] / 10000
#     #     fine_test_set[column] = fine_test_set[column] / 10000
#     # else:

# # Standardize data
# for idx, column in enumerate(COLUMNS_TO_STANDARDIZE):
#     train_set[column] = train_set[column] / band_means[idx]
#     coarse_val_set[column] = coarse_val_set[column] / band_means[idx]
#     fine_train_set[column] = fine_train_set[column] / band_means[idx]
#     fine_val_set[column] = fine_val_set[column] / band_means[idx]
#     fine_test_set[column] = fine_test_set[column] / band_means[idx]

X_fine_train = fine_train_set[INPUT_COLUMNS].to_numpy()
Y_fine_train = fine_train_set[OUTPUT_COLUMN].values.ravel()
print("X fine train", X_fine_train.shape)
print("Y fine train", Y_fine_train.shape)
X_fine_val = fine_val_set[INPUT_COLUMNS].to_numpy()
Y_fine_val = fine_val_set[OUTPUT_COLUMN].values.ravel()

# # Set inf/NaN to 0
# X_fine_train = np.nan_to_num(X_fine_train, nan=0, posinf=0, neginf=0)
# X_fine_val = np.nan_to_num(X_fine_val, nan=0, posinf=0, neginf=0)

# # Normalize each pixel's vector to have length equal to the average
# # average_norm = np.linalg.norm(band_means[COLUMNS_TO_NORMALIZE])
# # print("average norm", average_norm)
# X_fine_train[:, COLUMNS_TO_NORMALIZE] = (X_fine_train[:, COLUMNS_TO_NORMALIZE] / np.linalg.norm(X_fine_train[:, COLUMNS_TO_NORMALIZE], axis=1, keepdims=True))
# X_fine_val[:, COLUMNS_TO_NORMALIZE] = (X_fine_val[:, COLUMNS_TO_NORMALIZE] / np.linalg.norm(X_fine_val[:, COLUMNS_TO_NORMALIZE], axis=1, keepdims=True))
# X_fine_train[:, COLUMNS_TO_NORMALIZE] = X_fine_train[:, COLUMNS_TO_NORMALIZE] / np.mean(X_fine_train[:, COLUMNS_TO_NORMALIZE], axis=0, keepdims=True)
# X_fine_val[:, COLUMNS_TO_NORMALIZE] = X_fine_val[:, COLUMNS_TO_NORMALIZE] / np.mean(X_fine_train[:, COLUMNS_TO_NORMALIZE], axis=0, keepdims=True)

# Now do StandardScaler or min-max scaler
scaler = StandardScaler().fit(X_fine_train[:, COLUMNS_TO_STANDARDIZE])
X_fine_train[:, COLUMNS_TO_STANDARDIZE] = scaler.transform(X_fine_train[:, COLUMNS_TO_STANDARDIZE])
X_fine_val[:, COLUMNS_TO_STANDARDIZE] = scaler.transform(X_fine_val[:, COLUMNS_TO_STANDARDIZE])

# # # TODO Instead, try to normalize data
# X_fine_train[:, COLUMNS_TO_NORMALIZE] = X_fine_train[:, COLUMNS_TO_NORMALIZE] / np.linalg.norm(X_fine_train[:, COLUMNS_TO_NORMALIZE], axis=1)
# X_fine_train[:, COLUMNS_TO_NORMALIZE] = X_fine_train[:, COLUMNS_TO_NORMALIZE] / np.linalg.norm(X_fine_train[:, COLUMNS_TO_NORMALIZE], axis=1)


# Train a separate model per crop
fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('True vs predicted SIF - separate models for each crop type')

for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    # Fit linear model on just this crop, to see how strong the relationship is
    X_train_crop = X_fine_train[fine_train_set[crop_type] > PURE_THRESHOLD]
    Y_train_crop = Y_fine_train[fine_train_set[crop_type] > PURE_THRESHOLD]
    X_val_crop = X_fine_val[fine_val_set[crop_type] > PURE_THRESHOLD]
    Y_val_crop = Y_fine_val[fine_val_set[crop_type] > PURE_THRESHOLD]
    crop_regression = Ridge().fit(X_train_crop, Y_train_crop)
    predicted_val_crop = crop_regression.predict(X_val_crop)

    # Print stats and plot results for this crop
    ax = axeslist.ravel()[idx]
    print(' ----- Crop specific regression: ' + crop_type + ' -----')
    print_stats(Y_val_crop, predicted_val_crop, sif_mean, ax=ax, fit_intercept=True)
    ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
    ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
    ax.set_title(crop_type)

plt.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(os.path.join(PLOT_DIR, "crop_specific_model_results.png"))
plt.close()

def pixel_similarity(pixel1, pixel2):
    return math.exp(-(np.linalg.norm(pixel1 - pixel2) ** 2) / 12)

def angle_similarity(pixel1, pixel2):
    sim = cosine_similarity(pixel1.reshape(1, -1), pixel2.reshape(1, -1))[0, 0]
    if sim < 0.2:
        print("====== Very low similarity:", sim, "=====")
        print("pixel1", pixel1)
        print("pixel2", pixel2)
    return sim

def neg_euclidean_distance(pixel1, pixel2):
    return -np.linalg.norm(pixel1 - pixel2)

def compute_reflectance_similarity(pixel1, pixel2, tau):
    distance = np.linalg.norm(pixel1 - pixel2)
    return np.exp(-tau * (distance ** 2))


#  Plot SIF distance vs. input similarity for pairs, for each crop type.
fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('SIF distance vs input similarity')
input_similarities_by_crop = dict()
output_distances_by_crop = dict()
for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    print("Crop type", crop_type)
    X_train_crop = X_fine_train[fine_train_set[crop_type] > PURE_THRESHOLD]
    Y_train_crop = Y_fine_train[fine_train_set[crop_type] > PURE_THRESHOLD]
    fine_train_set_crop = fine_train_set.loc[fine_train_set[crop_type] > PURE_THRESHOLD]

    # Plot SIF distance vs. input distance for random pairs. TODO try cosine distance
    index_pairs = np.random.randint(low=0, high=X_train_crop.shape[0], size=(10000, 2))
    input_similarities = []
    output_distances = []
    for i in range(index_pairs.shape[0]):
        idx1 = index_pairs[i, 0]
        idx2 = index_pairs[i, 1]
        if i % 500 == 0:
            print("X1", X_train_crop[idx1])
        similarity_between_inputs = compute_reflectance_similarity(X_train_crop[idx1], X_train_crop[idx2], TAU)
        # similarity_between_inputs = angle_similarity(X_train_crop[idx1], X_train_crop[idx2])  # np.linalg.norm(X_train_crop[idx1] - X_train_crop[idx2])
        # similarity_between_inputs = spatial.distance.cosine(X_fine_train[idx1], X_fine_train[idx2])
        distance_between_outputs = abs(Y_train_crop[idx1] - Y_train_crop[idx2])
        input_similarities.append(similarity_between_inputs)
        output_distances.append(distance_between_outputs)

    input_similarities = np.array(input_similarities)
    output_distances = np.array(output_distances)
    input_similarities_by_crop[crop_type] = input_similarities
    output_distances_by_crop[crop_type] = output_distances
    print("input distances", input_similarities[0:10])
    print("output distances", output_distances[0:10])
    corr, _ = pearsonr(input_similarities, output_distances)
    ax = axeslist.ravel()[idx]
    ax = density_scatter(input_similarities, output_distances, bins=[40, 40], ax=ax, s=5)
    ax.set(xlabel='Similarity between inputs', ylabel='Distance between SIF')
    ax.set_title(crop_type)  # + ": corr = " + str(round(corr, 3)))

plt.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(os.path.join(PLOT_DIR, "output_vs_input_similarities.png"))
plt.close()



# For each crop type - plot histograms of SIF similarity at different levels of input similarity
for crop_type in COVER_COLUMN_NAMES:
    input_similarities = input_similarities_by_crop[crop_type]
    output_distances = output_distances_by_crop[crop_type]
    hist_range = (np.min(output_distances), np.max(output_distances))
    cols = math.ceil(len(SIMILARITY_RANGES) / 2)
    fig, axeslist = plt.subplots(ncols=cols, nrows=2, figsize=(4*cols, 8))
    fig.suptitle('SIF vs NDVI')
    for idx, similarity_range in enumerate(SIMILARITY_RANGES):
        indices = (input_similarities >= similarity_range[0]) & (input_similarities <= similarity_range[1])
        output_distances_in_range = output_distances[indices]
        ax = axeslist.ravel()[idx]
        ax.hist(output_distances_in_range, range=hist_range, bins=20)
        ax.set_xlabel("SIF difference")
        ax.set_title("Similarity score between {} and {}\nMean: {:.3f}, Num datapoints: {}".format(
                similarity_range[0], similarity_range[1], np.mean(output_distances_in_range), len(output_distances_in_range)))
    fig.suptitle("SIF differences by input similarity: " + crop_type)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(PLOT_DIR, "sif_distances_histograms_" + crop_type + ".png"))
    plt.close()


# Also plot SIF vs NDVI for each crop type
fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('SIF vs NDVI')

for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    print("Crop type", crop_type)
    X_train_crop = X_fine_train[fine_train_set[crop_type] > PURE_THRESHOLD]
    Y_train_crop = Y_fine_train[fine_train_set[crop_type] > PURE_THRESHOLD]
    fine_train_set_crop = fine_train_set.loc[fine_train_set[crop_type] > PURE_THRESHOLD]

    # Plot SIF vs NDVI
    ax = axeslist.ravel()[idx]
    corr, _ = pearsonr(fine_train_set_crop["NDVI"].values.ravel(), Y_train_crop)
    # ax.scatter(input_similarities, output_distances, color="k", s=5)
    ax = density_scatter(fine_train_set_crop["NDVI"].values.ravel(), Y_train_crop, bins=[40, 40], ax=ax, s=5)
    ax.set(xlabel='NDVI', ylabel='SIF')
    ax.set_title(crop_type + ': corr = ' + str(round(corr, 3)))

plt.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(os.path.join(PLOT_DIR, "sif_vs_ndvi.png"))
plt.close()