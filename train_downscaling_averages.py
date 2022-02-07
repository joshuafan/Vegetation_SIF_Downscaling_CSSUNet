"""
Runs pre-built ML methods over the channel averages of each tile (e.g. linear regression or gradient boosted tree)
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
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sif_utils import plot_histogram, print_stats

parser = argparse.ArgumentParser()
parser.add_argument('-method', "--method", choices=["Ridge_Regression", "Gradient_Boosting_Regressor", "MLP", "Nearest_Neighbors"], type=str, help='Method type. MLP is the fully-connected artificial neural netwoprk.')
parser.add_argument('-multiplicative_noise', "--multiplicative_noise", action='store_true')
parser.add_argument('-mult_noise_std', "--mult_noise_std", default=0.2, type=float, help="If the 'multiplicative_noise' augmentation is used, multiply each channel by (1+eps), where eps ~ N(0, mult_noise_std)")
parser.add_argument('-mult_noise_repeats', "--mult_noise_repeats", default=10, type=int, help="How many times to repeat each example (with different multiplicative noise)")
parser.add_argument('-standardize', "--standardize", action='store_true', help='Whether to standardize features to mean 0, variance 1.')
parser.add_argument('-normalize', "--normalize", action='store_true', help='Whether to normalize the reflectance bands to have norm 1. If this is enabled, the reflectance bands are NOT standardized.')
parser.add_argument('-log', "--log", action='store_true', help='Whether to log the reflectance features and SIF.')
parser.add_argument('-compute_ratios', "--compute_ratios", action='store_true', help='Whether to compute and use all ratios')

args = parser.parse_args()
METHOD_READABLE = args.method.replace("_", " ")
if not args.multiplicative_noise:
    args.mult_noise_std = 0

# Set random seed for data shuffling
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Random seeds for model training
TRAIN_RANDOM_STATES = [0, 1, 2]
NUM_RUNS = len(TRAIN_RANDOM_STATES)

# Folds
TRAIN_FOLDS = [0, 1, 2]
VAL_FOLDS = [3]
TEST_FOLDS = [4]

# Directories
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF"
METADATA_DIR = os.path.join(DATA_DIR, "metadata/CFIS_OCO2_dataset")

# Train files
CFIS_COARSE_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_coarse_metadata.csv')
CFIS_FINE_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_fine_metadata.csv')
OCO2_METADATA_FILE = os.path.join(METADATA_DIR, 'oco2_metadata.csv')
BAND_STATISTICS_FILE = os.path.join(METADATA_DIR, 'cfis_band_statistics_train.csv')

# Only include CFIS tiles where at least this fraction of pixels have CFIS
# fine-resolution data
MIN_COARSE_FRACTION_VALID_PIXELS = [0.1]

# Only EVALUATE on CFIS fine-resolution pixels with at least this number of soundings (measurements)
MIN_FINE_CFIS_SOUNDINGS = [30] #[30] #[1, 5, 10, 20, 30] # # 100, 250] #[100, 300, 1000, 3000]
eps = 1e-5

# For resolutions greater than 30m, only evaluate on grid cells where at least this fraction
# of 30m pixels have any CFIS data
MIN_FINE_FRACTION_VALID_PIXELS = [0.9-eps] #[0.1, 0.3, 0.5, 0.7] # [0.5] #[0.5]

# Resolutions to consider
RESOLUTION_METERS = [30, 90, 150, 300, 600]

# Dates/sources
DATES = ["2016-06-15", "2016-08-01"]
TRAIN_DATES = ["2016-06-15", "2016-08-01"]
TEST_DATES = ["2016-06-15", "2016-08-01"]

# List of sources to use (either CFIS or OCO-2)
TRAIN_SOURCES = ['CFIS', 'OCO2']
print("METHOD:", args.method, "- SOURCES:", TRAIN_SOURCES)

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

# Result plots and files
RESULTS_DIR = os.path.join(DATA_DIR, "baseline_results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(RESULTS_DIR + "/plots"):
    os.makedirs(RESULTS_DIR + "/plots")
CFIS_TRUE_VS_PREDICTED_PLOT = os.path.join(RESULTS_DIR, 'plots/true_vs_predicted_sif_cfis_' + args.method)
if args.log:
    CFIS_TRUE_VS_PREDICTED_PLOT += "_log"
if args.normalize:
    CFIS_TRUE_VS_PREDICTED_PLOT += "_normalize"
if args.standardize:
    CFIS_TRUE_VS_PREDICTED_PLOT += "_standardize"
if args.compute_ratios:
    CFIS_TRUE_VS_PREDICTED_PLOT += "_ratios"

RESULTS_SUMMARY_FILE = os.path.join(RESULTS_DIR, "results_summary_BASELINE.csv")
results_rows = {s: [args.method, s, MIN_FINE_CFIS_SOUNDINGS[0], MIN_FINE_FRACTION_VALID_PIXELS[0], args.mult_noise_std] for s in TRAIN_RANDOM_STATES}

# Input feature names
INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',  # 'ref_10', 'ref_11', 
                    'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
                    'grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity', 'missing_reflectance']  # ['NDVI']
ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']
REFLECTANCE_BANDS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7']  # 'ref_10', 'ref_11']

# Order of column in band averages file
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
COLUMNS_TO_STANDARDIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7', # 'ref_10', 'ref_11', 
                          'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
COLUMNS_TO_LOG = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7']  #, 'ref_10', 'ref_11']
OUTPUT_COLUMN = ['SIF']

# Crop types to look at when analyzing results
COVER_COLUMNS_TO_PLOT = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest'] 


# Other options - out of date
# INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
#                     'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
#                     'grassland_pasture', 'corn', 'soybean', 'shrubland',
#                     'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
#                     'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
#                     'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
#                     'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
#                     'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
#                     'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
#                     'lentils', 'missing_reflectance']
# INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
#                     'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
# COVER_COLUMNS_TO_PLOT = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
#                     'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
#                     'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
#                     'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
#                     'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
#                     'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
#                     'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
#                     'lentils']





for min_coarse_fraction_valid in MIN_COARSE_FRACTION_VALID_PIXELS:
    for resolution in RESOLUTION_METERS:
        # Filter OCO2 tiles
        oco2_metadata = pd.read_csv(OCO2_METADATA_FILE)
        oco2_metadata = oco2_metadata[(oco2_metadata['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                                      (oco2_metadata['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                                      (oco2_metadata['SIF'] >= MIN_SIF_CLIP)]
        oco2_metadata = oco2_metadata[oco2_metadata[ALL_COVER_COLUMNS].sum(axis=1) >= 0.5]

        # Read CFIS coarse datapoints - only include CFIS tiles with enough valid pixels
        cfis_coarse_metadata = pd.read_csv(CFIS_COARSE_METADATA_FILE)
        cfis_coarse_metadata = cfis_coarse_metadata[(cfis_coarse_metadata['fraction_valid'] >= min_coarse_fraction_valid) &
                                                    (cfis_coarse_metadata['SIF'] >= MIN_SIF_CLIP) &
                                                    (cfis_coarse_metadata['missing_reflectance'] <= MAX_CFIS_CLOUD_COVER)]
        cfis_coarse_metadata = cfis_coarse_metadata[cfis_coarse_metadata[ALL_COVER_COLUMNS].sum(axis=1) >= 0.5]

        # Read fine metadata at particular resolution, and do initial filtering
        CFIS_FINE_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_metadata_' + str(resolution) + 'm.csv')
        cfis_fine_metadata = pd.read_csv(CFIS_FINE_METADATA_FILE)
        cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['SIF'] >= MIN_SIF_CLIP) &
                                        (cfis_fine_metadata['tile_file'].isin(set(cfis_coarse_metadata['tile_file'])))]
        cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['num_soundings'] >= min(MIN_FINE_CFIS_SOUNDINGS)) &
                                                (cfis_fine_metadata['fraction_valid'] >= min(MIN_FINE_FRACTION_VALID_PIXELS))]  # Avoid roundoff errors

        # Compute NDVI
        oco2_metadata["NDVI"] = (oco2_metadata["ref_5"] - oco2_metadata["ref_4"]) / (oco2_metadata["ref_5"] + oco2_metadata["ref_4"])
        cfis_coarse_metadata["NDVI"] = (cfis_coarse_metadata["ref_5"] - cfis_coarse_metadata["ref_4"]) / (cfis_coarse_metadata["ref_5"] + cfis_coarse_metadata["ref_4"])
        cfis_fine_metadata["NDVI"] = (cfis_fine_metadata["ref_5"] - cfis_fine_metadata["ref_4"]) / (cfis_fine_metadata["ref_5"] + cfis_fine_metadata["ref_4"])

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

        # print('Fine CFIS by fold:', cfis_fine_metadata['fold'].value_counts())
        # print('Fine CFIS by grid fold:', cfis_fine_metadata['grid_fold'].value_counts())

        # # Compute averages for each band
        # STATISTICS_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
        #                     'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
        #                     'grassland_pasture', 'corn', 'soybean', 'shrubland',
        #                     'deciduous_forest', 'evergreen_forest', 'spring_wheat',
        #                     'developed_open_space', 'other_hay_non_alfalfa', 'winter_wheat',
        #                     'herbaceous_wetlands', 'woody_wetlands', 'open_water', 'alfalfa',
        #                     'fallow_idle_cropland', 'sorghum', 'developed_low_intensity',
        #                     'barren', 'durum_wheat',
        #                     'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
        #                     'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
        #                     'lentils', 'missing_reflectance', 'SIF']
        # selected_columns_coarse = coarse_train_set[STATISTICS_COLUMNS]
        # band_means_coarse = selected_columns_coarse.mean(axis=0)
        # print("(Coarse train set) Band means", band_means_coarse)        
        # selected_columns = fine_train_set[STATISTICS_COLUMNS]
        # band_means = selected_columns.mean(axis=0)
        # band_stds = selected_columns.mean(axis=0)
        # print("(Fine train set) Band means", band_means)
        # selected_columns_test = fine_test_set[STATISTICS_COLUMNS]
        # band_means_test = selected_columns_test.mean(axis=0)
        # print("(Fine test set) Band means", band_means_test)
        # exit(0)

        # # Write band averages to file
        # statistics_rows = [['mean', 'std']]
        # for i, mean in enumerate(band_means):
        #     statistics_rows.append([band_means[i], band_stds[i]])
        # with open(BAND_STATISTICS_FILE, 'w') as output_csv_file:
        #     csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        #     for row in statistics_rows:
        #         csv_writer.writerow(row)


        # # Read coarse datasets
        # coarse_train_set = pd.read_csv(COARSE_AVERAGES_TRAIN_FILE)
        # coarse_val_set = pd.read_csv(COARSE_AVERAGES_VAL_FILE)
        # coarse_test_set = pd.read_csv(COARSE_AVERAGES_TEST_FILE)
        # oco2_train_set = pd.read_csv(OCO2_METADATA_TRAIN_FILE)
        # oco2_val_set = pd.read_csv(OCO2_METADATA_VAL_FILE)
        # oco2_test_set = pd.read_csv(OCO2_METADATA_TEST_FILE)

        # # Only include CFIS tiles with enough valid pixels
        # coarse_train_set = coarse_train_set[(coarse_train_set['fraction_valid'] >= min_coarse_fraction_valid) &
        #                                     (coarse_train_set['SIF'] >= MIN_SIF_CLIP) &
        #                                     (coarse_train_set['date'].isin(TRAIN_DATES))]
        # coarse_val_set = coarse_val_set[(coarse_val_set['fraction_valid'] >= min_coarse_fraction_valid) &
        #                                 (coarse_val_set['SIF'] >= MIN_SIF_CLIP) &
        #                                 (coarse_val_set['date'].isin(TRAIN_DATES))]
        # coarse_test_set = coarse_test_set[(coarse_test_set['fraction_valid'] >= min_coarse_fraction_valid) &
        #                                 (coarse_test_set['SIF'] >= MIN_SIF_CLIP) &
        #                                 (coarse_test_set['date'].isin(TEST_DATES))]

        # # Filter OCO2 sets
        # oco2_train_set = oco2_train_set[(oco2_train_set['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
        #                                 (oco2_train_set['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
        #                                 (oco2_train_set['SIF'] >= MIN_SIF_CLIP) &
        #                                 (oco2_train_set['date'].isin(TRAIN_DATES))]
        # oco2_val_set = oco2_val_set[(oco2_val_set['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
        #                             (oco2_val_set['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
        #                             (oco2_val_set['SIF'] >= MIN_SIF_CLIP) &
        #                             (oco2_val_set['date'].isin(TRAIN_DATES))]
        # oco2_test_set = oco2_test_set[(oco2_test_set['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
        #                             (oco2_test_set['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
        #                             (oco2_test_set['SIF'] >= MIN_SIF_CLIP) &
        #                             (oco2_test_set['date'].isin(TEST_DATES))]
        # oco2_train_set['SIF'] = oco2_train_set['SIF'] * OCO2_SCALING_FACTOR

        # # Read fine dataset at a particular resolution
        # FINE_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_averages_' + str(resolution) + 'm_train.csv')
        # FINE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_averages_' + str(resolution) + 'm_val.csv')
        # FINE_AVERAGES_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_averages_' + str(resolution) + 'm_test.csv')
        # fine_train_set = pd.read_csv(FINE_AVERAGES_TRAIN_FILE)
        # fine_val_set = pd.read_csv(FINE_AVERAGES_VAL_FILE)
        # fine_test_set = pd.read_csv(FINE_AVERAGES_TEST_FILE)

        # fine_train_set = fine_train_set[(fine_train_set['SIF'] >= MIN_SIF_CLIP) &
        #                                 (fine_train_set['date'].isin(TRAIN_DATES)) &
        #                                 (fine_train_set['tile_file'].isin(set(coarse_train_set['tile_file'])))]
        # fine_val_set = fine_val_set[(fine_val_set['SIF'] >= MIN_SIF_CLIP) &
        #                                 (fine_val_set['date'].isin(TRAIN_DATES)) &
        #                                (fine_val_set['tile_file'].isin(set(coarse_val_set['tile_file'])))]
        # fine_test_set = fine_test_set[(fine_test_set['SIF'] >= MIN_SIF_CLIP) &
        #                                 (fine_test_set['date'].isin(TEST_DATES)) &
        #                                 (fine_test_set['tile_file'].isin(set(coarse_test_set['tile_file'])))]
        # # fine_val_set = fine_val_set[(fine_val_set['num_soundings'] >= 10) &
        # #                                         (fine_val_set['fraction_valid'] >= 0.2)]

        # Construct combined train set
        print('CFIS Coarse train set:', len(coarse_train_set))
        print('OCO2 train set:', len(oco2_train_set))

        # Filter train set to only include desired sources
        if 'OCO2' in TRAIN_SOURCES and 'CFIS' in TRAIN_SOURCES:
            print('Using both OCO2 and CFIS')
            train_set = pd.concat([oco2_train_set, coarse_train_set])
        elif 'OCO2' in TRAIN_SOURCES:
            print('ONLY using OCO2')
            train_set = oco2_train_set
            coarse_val_set = oco2_val_set
        elif 'CFIS' in TRAIN_SOURCES:
            print('ONLY using CFIS')
            train_set = coarse_train_set
        else:
            print("Didn't specify valid sources :(")
            exit(0)

        # Shuffle train set
        train_set = train_set.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        # print('Train set samples', len(train_set))

        # Read band statistics
        train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
        train_means = train_statistics['mean'].values
        train_stds = train_statistics['std'].values
        band_means = train_means[:-1]
        sif_mean = train_means[-1]
        band_stds = train_stds[:-1]
        sif_std = train_stds[-1]

        # Print dataset info
        # print('Coarse val samples', len(coarse_val_set))
        # print('OCO2 train samples', len(oco2_train_set))
        # print('OCO2 val samples', len(oco2_val_set))
        # print('OCO2 test samples', len(oco2_test_set))
        # print('Average SIF (fine, train)', fine_train_set['SIF'].mean())
        # print('Average SIF (fine, val)', fine_val_set['SIF'].mean())
        # print('Average SIF (fine, test)', fine_test_set['SIF'].mean())
        # print('Average SIF (coarse, train)', coarse_train_set['SIF'].mean())
        # print('Average SIF (coarse, val)', coarse_val_set['SIF'].mean())
        # print('Average SIF (coarse, test)', coarse_test_set['SIF'].mean())
        # print('Average SIF (OCO2, train)', oco2_train_set['SIF'].mean())
        # print('Average SIF (OCO2, val)', oco2_val_set['SIF'].mean())
        # print('Average SIF (OCO2, test)', oco2_test_set['SIF'].mean())
        # print('Average SIf from band statistics file', sif_mean)

        # Inject multiplicative noise optionally
        if args.multiplicative_noise:
            train_set = pd.concat([train_set] * args.mult_noise_repeats)
            train_set = train_set.reset_index(drop=True)
            for idx in range(len(train_set)):
                noise = 1 + np.random.normal(loc=0, scale=args.mult_noise_std)
                train_set.loc[idx, REFLECTANCE_BANDS] = train_set.loc[idx, REFLECTANCE_BANDS] * noise

        # Feature engineering        
        if args.log:
            for idx, column in enumerate(COLUMNS_TO_LOG):
                log_column_values = np.nan_to_num(np.log(train_set[column] + 1e-5))
                log_mean = np.mean(log_column_values)
                log_std = np.std(log_column_values)
                log_column = "log_" + column
                INPUT_COLUMNS.append(log_column)
                train_set[log_column] = np.nan_to_num((np.log(train_set[column] + 1e-5) - log_mean) / log_std)
                coarse_val_set[log_column] = np.nan_to_num((np.log(coarse_val_set[column] + 1e-5) - log_mean) / log_std)
                fine_train_set[log_column] = np.nan_to_num((np.log(fine_train_set[column] + 1e-5) - log_mean) / log_std)
                fine_val_set[log_column] = np.nan_to_num((np.log(fine_val_set[column] + 1e-5) - log_mean) / log_std)
                fine_test_set[log_column] = np.nan_to_num((np.log(fine_test_set[column] + 1e-5) - log_mean) / log_std)

        if args.normalize:
            train_set[REFLECTANCE_BANDS] = normalize(train_set[REFLECTANCE_BANDS])
            coarse_val_set[REFLECTANCE_BANDS] = normalize(coarse_val_set[REFLECTANCE_BANDS])
            fine_train_set[REFLECTANCE_BANDS] = normalize(fine_train_set[REFLECTANCE_BANDS])
            fine_val_set[REFLECTANCE_BANDS] = normalize(fine_val_set[REFLECTANCE_BANDS])
            fine_test_set[REFLECTANCE_BANDS] = normalize(fine_test_set[REFLECTANCE_BANDS])

        if args.compute_ratios:
            for col1 in REFLECTANCE_BANDS:
                for col2 in REFLECTANCE_BANDS:
                    if col1 == col2:
                        continue
                    new_col_name = col1 + "_over_" + col2
                    INPUT_COLUMNS.append(new_col_name)
                    train_set[new_col_name] = np.nan_to_num(train_set[col1] / train_set[col2], posinf=0, neginf=0)
                    coarse_val_set[new_col_name] = np.nan_to_num(coarse_val_set[col1] / coarse_val_set[col2], posinf=0, neginf=0)
                    fine_train_set[new_col_name] = np.nan_to_num(fine_train_set[col1] / fine_train_set[col2], posinf=0, neginf=0)
                    fine_val_set[new_col_name] = np.nan_to_num(fine_val_set[col1] / fine_val_set[col2], posinf=0, neginf=0)
                    fine_test_set[new_col_name] = np.nan_to_num(fine_test_set[col1] / fine_test_set[col2], posinf=0, neginf=0)

        if args.standardize:
            for column in COLUMNS_TO_STANDARDIZE:
                # Note - this uses precomputed band means/stds. If desired you can compute them on the fly.
                idx = BAND_AVERAGES_COLUMN_ORDER.index(column)
                print("For debugging - column", column, "idx", idx)
                train_set[column] = np.clip((train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
                coarse_val_set[column] = np.clip((coarse_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
                fine_train_set[column] = np.clip((fine_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
                fine_val_set[column] = np.clip((fine_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
                fine_test_set[column] = np.clip((fine_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)

        # Select X/Y columns
        X_train = train_set[INPUT_COLUMNS]
        Y_train = train_set[OUTPUT_COLUMN].values.ravel()
        X_coarse_val = coarse_val_set[INPUT_COLUMNS]
        Y_coarse_val = coarse_val_set[OUTPUT_COLUMN].values.ravel()
        X_fine_val = fine_val_set[INPUT_COLUMNS]
        Y_fine_val = fine_val_set[OUTPUT_COLUMN].values.ravel()
        X_fine_test = fine_test_set[INPUT_COLUMNS]
        Y_fine_test = fine_test_set[OUTPUT_COLUMN].values.ravel()
        print("X_train shape", X_train.shape)

        # If "log" is set, log SIF also
        if args.log:
            Y_fine_train = np.log(Y_fine_train)
            Y_fine_val = np.log(Y_fine_val)
            Y_fine_test = np.log(Y_fine_test)

        # Fit models on band averages (with various hyperparam settings)
        regression_models = dict()
        if 'Linear_Regression' in args.method:
            regression_model = LinearRegression().fit(X_train, Y_train) #X_train, Y_train)
            regression_models['linear'] = [regression_model]
        elif 'Lasso' in args.method:
            alphas = [0.001, 0.01, 0.1, 1, 10, 100]
            for alpha in alphas:
                regression_model = Lasso(alpha=alpha).fit(X_train, Y_train)
                param_string = 'alpha=' + str(alpha)
                regression_models[param_string] = [regression_model]
        elif 'Ridge_Regression' in args.method:
            alphas = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            for alpha in alphas:
                models = []
                for random_state in TRAIN_RANDOM_STATES:
                    regression_model = Ridge(alpha=alpha, random_state=random_state).fit(X_train, Y_train) # HuberRegressor(alpha=alpha, max_iter=1000).fit(X_train, Y_train)
                    models.append(regression_model)
                param_string = 'alpha=' + str(alpha)
                regression_models[param_string] = models
        elif "Gradient_Boosting_Regressor" in args.method:
            max_iter_values = [100, 300, 1000] #
            max_depth_values = [2, 3, None]
            for max_iter in max_iter_values:
                for max_depth in max_depth_values:
                    models = []
                    for random_state in TRAIN_RANDOM_STATES:
                        regression_model = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=max_depth, learning_rate=0.1, random_state=random_state).fit(X_train, Y_train)
                        models.append(regression_model)
                    param_string = 'max_iter=' + str(max_iter) + ', max_depth=' + str(max_depth)
                    print(param_string)
                    regression_models[param_string] = models
        elif "MLP" in args.method:
            hidden_layer_sizes = [(100,), (20, 20), (100, 100), (100, 100, 100)] #[(100, 100)] # 
            learning_rate_inits =  [1e-2, 1e-3, 1e-4]  # [1e-3] #
            max_iter = 10000
            for hidden_layer_size in hidden_layer_sizes:
                for learning_rate_init in learning_rate_inits:
                    models = []
                    for random_state in TRAIN_RANDOM_STATES:
                        regression_model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, learning_rate_init=learning_rate_init, max_iter=max_iter, random_state=random_state).fit(X_train, Y_train)
                        models.append(regression_model)
                    param_string = 'hidden_layer_sizes=' + str(hidden_layer_size) + ', learning_rate_init=' + str(learning_rate_init)
                    print(param_string)
                    regression_models[param_string] = models
        elif "Nearest_Neighbors" in args.method:
            num_neighbors = [5, 10, 20]
            for n in num_neighbors:
                models = []
                for random_state in TRAIN_RANDOM_STATES:
                    regression_model = KNeighborsRegressor(n_neighbors=n, weights="distance").fit(X_train, Y_train)
                    models.append(regression_model)
                param_string = "n_neighbors=" + str(n)
                print(param_string)
                regression_models[param_string] = models
        else:
            raise ValueError("Unsupported method. Options for --method are 'Ridge_Regression', 'Gradient_Boosting_Regressor', 'MLP'.")


        # Loop through all hyperparameter settings we trained models for, and compute
        # loss on the FINE-RESOLUTION validation set
        best_loss = float('inf')
        best_params = 'N/A'
        average_losses_val = []
        for params, models in regression_models.items():
            losses_val = []
            for model in models:  # Loop through all model runs (trained with different seeds)
                predictions_val = model.predict(X_fine_val)
                loss_val = math.sqrt(mean_squared_error(Y_fine_val, predictions_val)) / sif_mean
                losses_val.append(loss_val)
            average_loss_val = sum(losses_val) / len(losses_val)
            print(params + ': avg val loss', round(average_loss_val, 4))
            if average_loss_val < best_loss:
                best_loss = average_loss_val
                best_params = params
                best_idx = np.argmin(losses_val)
            average_losses_val.append(average_loss_val)

        print('Best params:', best_params)

        # If we logged SIF, transform labels back to original values
        if args.log:
            Y_fine_train = np.exp(Y_fine_train)
            Y_fine_val = np.exp(Y_fine_val)
            Y_fine_test = np.exp(Y_fine_test)

        # Different ways of filtering fine pixels
        for min_fine_cfis_soundings in MIN_FINE_CFIS_SOUNDINGS:
            for min_fraction_valid_pixels in MIN_FINE_FRACTION_VALID_PIXELS:
                # Record performances for this setting
                all_r2 = {'all_coarse_val': [], 'all_fine_train': [], 'all_fine_val': [], 'all_fine_test': [], 'grassland_pasture': [], 'corn': [], 'soybean': [], 'deciduous_forest': []}
                all_nrmse = {'all_coarse_val': [], 'all_fine_train': [], 'all_fine_val': [], 'all_fine_test': [], 'grassland_pasture': [], 'corn': [], 'soybean': [], 'deciduous_forest': []}
                all_corr = {'all_coarse_val': [], 'all_fine_train': [], 'all_fine_val': [], 'all_fine_test': [], 'grassland_pasture': [], 'corn': [], 'soybean': [], 'deciduous_forest': []}

                print('========================================= FILTER ======================================================')
                print('*** Resolution', resolution)
                print('*** Min coarse fraction valid pixels', min_coarse_fraction_valid)
                print('*** Min fine soundings', min_fine_cfis_soundings)
                print('*** Min fine fraction valid pixels', min_fraction_valid_pixels)
                print('===================================================================================================')

                # Loop through trained models
                for idx, model in enumerate(regression_models[best_params]):
                    # Only plot graph for best model
                    is_best_model = (idx == best_idx)

                    # Use the best model to make predictions
                    predictions_train = model.predict(X_train)
                    predictions_coarse_val = model.predict(X_coarse_val)
                    predictions_train = np.clip(predictions_train, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
                    predictions_coarse_val = np.clip(predictions_coarse_val, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)

                    if is_best_model:
                        # Print NRMSE, correlation, R2 on train/validation set
                        print('============== Train set stats =====================')
                        print_stats(Y_train, predictions_train, sif_mean, fit_intercept=False)

                        print('============== Coarse val set stats =====================')
                        coarse_val_r2, coarse_val_nrmse, coarse_val_corr = print_stats(Y_coarse_val, predictions_coarse_val, sif_mean, ax=plt.gca(), fit_intercept=False)
                        plt.title('Coarse val set: true vs predicted SIF (' + METHOD_READABLE + ')')
                        plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                        plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                        plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_coarsefractionvalid' + str(min_coarse_fraction_valid) + '_coarse_val.png')
                        plt.close()
                    else:
                        coarse_val_r2, coarse_val_nrmse, coarse_val_corr = print_stats(Y_coarse_val, predictions_coarse_val, sif_mean,
                                                                                       ax=None, fit_intercept=False, print_report=False)
                    all_r2['all_coarse_val'].append(coarse_val_r2)
                    all_nrmse['all_coarse_val'].append(coarse_val_nrmse)
                    all_corr['all_coarse_val'].append(coarse_val_corr)

                    PLOT_PREFIX = CFIS_TRUE_VS_PREDICTED_PLOT + '_res' + str(resolution) #+ '_coarsefractionvalid' + str(min_coarse_fraction_valid) + '_finesoundings' + str(min_fine_cfis_soundings) + '_finefractionvalid' + str(min_fraction_valid_pixels)

                    # Further filter 
                    fine_train_set_filtered = fine_train_set[(fine_train_set['num_soundings'] >= min_fine_cfis_soundings) &
                                                             (fine_train_set['fraction_valid'] >= min_fraction_valid_pixels)]
                    fine_val_set_filtered = fine_val_set[(fine_val_set['num_soundings'] >= min_fine_cfis_soundings) &
                                                         (fine_val_set['fraction_valid'] >= min_fraction_valid_pixels)]
                    fine_test_set_filtered = fine_test_set[(fine_test_set['num_soundings'] >= min_fine_cfis_soundings) &
                                                           (fine_test_set['fraction_valid'] >= min_fraction_valid_pixels)]
                    X_fine_train_filtered = fine_train_set_filtered[INPUT_COLUMNS]
                    Y_fine_train_filtered = fine_train_set_filtered[OUTPUT_COLUMN].values.ravel()
                    X_fine_val_filtered = fine_val_set_filtered[INPUT_COLUMNS]
                    Y_fine_val_filtered = fine_val_set_filtered[OUTPUT_COLUMN].values.ravel()
                    X_fine_test_filtered = fine_test_set_filtered[INPUT_COLUMNS]
                    Y_fine_test_filtered = fine_test_set_filtered[OUTPUT_COLUMN].values.ravel()
                    # # Check distributions
                    # print('Coarse Train set means', train_set.mean(axis=0))
                    # print('Fine train set means', fine_train_set.mean(axis=0))
                    # print('Coarse val set means', coarse_val_set.mean(axis=0))
                    # print('Fine val set means', fine_val_set.mean(axis=0))
                    # print('Fine test set means', fine_test_set.mean(axis=0))

                    # Now generate predictions using trained model
                    predictions_fine_train_filtered = model.predict(X_fine_train_filtered)
                    predictions_fine_val_filtered = model.predict(X_fine_val_filtered)
                    predictions_fine_test_filtered = model.predict(X_fine_test_filtered)
                    predictions_fine_train_filtered = np.clip(predictions_fine_train_filtered, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
                    predictions_fine_val_filtered = np.clip(predictions_fine_val_filtered, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
                    predictions_fine_test_filtered = np.clip(predictions_fine_test_filtered, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)

                    # Print summary statistics for best model
                    if is_best_model:
                        print('============== CFIS fine train set stats =====================')
                        fine_train_r2, fine_train_nrmse, fine_train_corr = print_stats(Y_fine_train_filtered, predictions_fine_train_filtered, sif_mean, ax=plt.gca(), fit_intercept=False)
                        plt.title('True vs predicted SIF (' + METHOD_READABLE + '): ' + str(int(resolution)) + 'm pixels, train tiles')
                        plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                        plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                        plt.savefig(PLOT_PREFIX + '_fine_train.png')
                        plt.close()

                        print('============== CFIS fine val set stats =====================')
                        fine_val_r2, fine_val_nrmse, fine_val_corr = print_stats(Y_fine_val_filtered, predictions_fine_val_filtered, sif_mean, ax=plt.gca(), fit_intercept=False)
                        plt.title('True vs predicted SIF (' + METHOD_READABLE + '): ' + str(int(resolution)) + 'm pixels, val tiles')
                        plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                        plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                        plt.savefig(PLOT_PREFIX + '_fine_val.png')
                        plt.close()

                        print('============== CFIS fine test set stats =====================')
                        fine_test_r2, fine_test_nrmse, fine_test_corr = print_stats(Y_fine_test_filtered, predictions_fine_test_filtered, sif_mean, ax=plt.gca(), fit_intercept=False)
                        plt.title('True vs predicted SIF (' + METHOD_READABLE + '): ' + str(int(resolution)) + 'm pixels, test tiles')
                        plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                        plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                        plt.savefig(PLOT_PREFIX + '_fine_test.png')
                        plt.close()
                    else:
                        fine_train_r2, fine_train_nrmse, fine_train_corr = print_stats(Y_fine_train_filtered, predictions_fine_train_filtered, sif_mean,
                                                                                       ax=None, fit_intercept=False, print_report=False)
                        fine_val_r2, fine_val_nrmse, fine_val_corr = print_stats(Y_fine_val_filtered, predictions_fine_val_filtered, sif_mean,
                                                                                 ax=None, fit_intercept=False, print_report=False)
                        fine_test_r2, fine_test_nrmse, fine_test_corr = print_stats(Y_fine_test_filtered, predictions_fine_test_filtered, sif_mean,
                                                                                    ax=None, fit_intercept=False, print_report=False)

                    all_r2['all_fine_train'].append(fine_train_r2)
                    all_nrmse['all_fine_train'].append(fine_train_nrmse)
                    all_corr['all_fine_train'].append(fine_train_corr)
                    all_r2['all_fine_val'].append(fine_val_r2)
                    all_nrmse['all_fine_val'].append(fine_val_nrmse)
                    all_corr['all_fine_val'].append(fine_val_corr)
                    all_r2['all_fine_test'].append(fine_test_r2)
                    all_nrmse['all_fine_test'].append(fine_test_nrmse)
                    all_corr['all_fine_test'].append(fine_test_corr)

                    # Plot true vs. predicted for each crop on CFIS fine (for each crop)
                    if is_best_model:
                        fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
                        fig.suptitle('True vs predicted SIF by crop: ' + METHOD_READABLE)
                    for idx, crop_type in enumerate(COVER_COLUMNS_TO_PLOT):
                        # predicted = predictions_fine_train_filtered[fine_train_set_filtered[crop_type] > PURE_THRESHOLD]
                        # true = Y_fine_train_filtered[fine_train_set_filtered[crop_type] > PURE_THRESHOLD]
                        predicted = predictions_fine_test_filtered[fine_test_set_filtered[crop_type] > PURE_THRESHOLD]
                        true = Y_fine_test_filtered[fine_test_set_filtered[crop_type] > PURE_THRESHOLD]                        
                        if len(predicted) >= 2:
                            if is_best_model:
                                print('======================= (CFIS fine) CROP: ', crop_type, '==============================')
                                ax = axeslist.ravel()[idx]
                                crop_r2, crop_nrmse, crop_corr = print_stats(true, predicted, sif_mean, ax=ax, fit_intercept=False)
                                ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                                ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                                ax.set_title(crop_type)
                            else:
                                crop_r2, crop_nrmse, crop_corr = print_stats(true, predicted, sif_mean, ax=None, fit_intercept=False, print_report=False)
                            all_r2[crop_type].append(crop_r2)
                            all_nrmse[crop_type].append(crop_nrmse)
                            all_corr[crop_type].append(crop_corr)
                        
                            # Fit linear model on just this crop, to see how strong the relationship is
                            # X_train_crop = X_train.loc[train_set[crop_type] > PURE_THRESHOLD]
                            # Y_train_crop = Y_train[train_set[crop_type] > PURE_THRESHOLD]
                            # X_val_crop = X_val.loc[test_set[crop_type] > PURE_THRESHOLD]
                            # Y_val_crop = Y_val[test_set[crop_type] > PURE_THRESHOLD]
                            # crop_regression = LinearRegression().fit(X_train_crop, Y_train_crop)
                            # predicted_oco2_crop = crop_regression.predict(X_val_crop)
                            # print(' ----- Crop specific regression -----')
                            # #print('Coefficients:', crop_regression.coef_)
                            # print_stats(Y_oco2_crop, predicted_oco2_crop, sif_mean)

                        # Plot true vs. predicted for that specific crop
                        # axeslist.ravel()[idx].scatter(true, predicted)
                        # axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')

                    if is_best_model:
                        plt.tight_layout()
                        fig.subplots_adjust(top=0.92)
                        plt.savefig(PLOT_PREFIX + '_crop_types.png')
                        plt.close()


                    # Print statistics and plot by date
                    if is_best_model:
                        fig, axeslist = plt.subplots(ncols=1, nrows=len(DATES), figsize=(6, 6*len(DATES)))
                        fig.suptitle('True vs predicted SIF, by date: ' + METHOD_READABLE)
                        idx = 0
                        for date in DATES:
                            # Obtain global model's predictions for data points with this date
                            predicted = predictions_fine_train_filtered[fine_train_set_filtered['date'] == date]
                            true = Y_fine_train_filtered[fine_train_set_filtered['date'] == date]
                            print('=================== Date ' + date + ' ======================')
                            assert(len(predicted) == len(true))
                            if len(predicted) < 2:
                                idx += 1
                                continue

                            # Print stats (true vs predicted)
                            ax = axeslist.ravel()[idx]
                            print_stats(true, predicted, sif_mean, ax=ax)
                            ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                            ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                            ax.set_title(date)
                            idx += 1

                        plt.tight_layout()
                        fig.subplots_adjust(top=0.92)
                        plt.savefig(PLOT_PREFIX + '_dates.png')
                        plt.close()

                    # Trivial method: use surrounding coarse tile SIF
                    if is_best_model:
                        predictions_train_predict_coarse = np.clip(fine_train_set_filtered['coarse_sif'].to_numpy(), a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
                        predictions_val_predict_coarse = np.clip(fine_val_set_filtered['coarse_sif'].to_numpy(), a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
                        predictions_test_predict_coarse = np.clip(fine_test_set_filtered['coarse_sif'].to_numpy(), a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)

                        print('============= (TRIVIAL: PREDICT COARSE) - Fine train set stats =============')
                        print_stats(Y_fine_train_filtered, predictions_train_predict_coarse, sif_mean, fit_intercept=False, ax=plt.gca())
                        plt.title('Fine train set: true vs predicted SIF (predict coarse SIF)')
                        plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                        plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                        plt.savefig(PLOT_PREFIX + '_fine_vs_coarse_train.png')
                        plt.close()

                        print('============= (TRIVIAL: PREDICT COARSE) - Fine val set stats =============')
                        print_stats(Y_fine_val_filtered, predictions_val_predict_coarse, sif_mean, fit_intercept=False, ax=plt.gca())
                        plt.title('Fine val set: true vs predicted SIF (predict coarse SIF)')
                        plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                        plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                        plt.savefig(PLOT_PREFIX + '_fine_vs_coarse_val.png')
                        plt.close()

                        print('============= (TRIVIAL: PREDICT COARSE) - Fine test set stats =============')
                        print_stats(Y_fine_test_filtered, predictions_test_predict_coarse, sif_mean, fit_intercept=False, ax=plt.gca())
                        plt.title('Fine test set: true vs predicted SIF (predict coarse SIF)')
                        plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                        plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                        plt.savefig(PLOT_PREFIX + '_fine_vs_coarse_test.png')
                        plt.close()

                        # Plot fine vs coarse SIF for each crop on CFIS fine (for each crop)
                        fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
                        fig.suptitle('True vs predicted SIF by crop: ' + METHOD_READABLE)
                        for idx, crop_type in enumerate(COVER_COLUMNS_TO_PLOT):
                            predicted = predictions_train_predict_coarse[fine_train_set_filtered[crop_type] > PURE_THRESHOLD]
                            true = Y_fine_train_filtered[fine_train_set_filtered[crop_type] > PURE_THRESHOLD]
                            ax = axeslist.ravel()[idx]
                            print('======================= (TRIVIAL: PREDICT COARSE) - CROP: ', crop_type, '==============================')
                            if len(predicted) >= 2:
                                print_stats(true, predicted, sif_mean, ax=ax)
                                ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                                ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                                ax.set_title(crop_type)

                        plt.tight_layout()
                        fig.subplots_adjust(top=0.92)
                        plt.savefig(PLOT_PREFIX + '_fine_vs_coarse_crop_types.png')
                        plt.close()


                nrmse_mean = {}
                nrmse_std = {}
                r2_mean = {}
                r2_std = {}
                print('==================== Summary (resolution: ' + str(resolution) + 'm) ===================')
                for key in all_r2:
                    nrmse_mean[key] = round(np.mean(all_nrmse[key]), 3)
                    nrmse_std[key] = round(np.std(all_nrmse[key], ddof=1), 4)
                    r2_mean[key] = round(np.mean(all_r2[key]), 3)
                    r2_std[key] = round(np.std(all_r2[key], ddof=1), 4)
                    print(key + ": NRMSE = " + str(nrmse_mean[key]) + " (std: " + str(nrmse_std[key]) +"), R2 = " + str(r2_mean[key]) + " (std: " + str(r2_std[key]) + ")")
                print('=================================================================\n\n')
  


                # Write final results to file
                for i, seed in enumerate(TRAIN_RANDOM_STATES):
                    if resolution == 30:
                        results_rows[seed].extend([best_params,
                                                   all_nrmse["all_fine_train"][i],
                                                   all_r2["all_fine_train"][i],
                                                   all_corr["all_fine_train"][i],
                                                   all_nrmse["all_fine_test"][i],
                                                   all_r2["all_fine_test"][i],
                                                   all_corr["all_fine_test"][i],
                                                   all_nrmse["grassland_pasture"][i],
                                                   all_nrmse["corn"][i],
                                                   all_nrmse["soybean"][i],
                                                   all_nrmse["deciduous_forest"][i]])
                    else:
                        results_rows[seed].append(all_nrmse["all_fine_train"][i])


                # # TODO fix param string
                # PARAM_STRING = ''
                # PARAM_STRING += '================ RESULTS =================\n'
                # PARAM_STRING += 'R2 means: ' + json.dumps(r2_mean) + '\n'
                # PARAM_STRING += 'R2 stds: ' + json.dumps(r2_std) + '\n'
                # PARAM_STRING += 'NRMSE means: ' + json.dumps(nrmse_mean) + '\n'
                # PARAM_STRING += 'NRMSE stds: ' + json.dumps(nrmse_std) + '\n'
                # results_dir = os.path.join('results', args.method)
                # if not os.path.exists(results_dir):
                #     os.makedirs(results_dir)
                # filename = PLOT_PREFIX + '_' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
                # with open(os.path.join(results_dir, filename), mode='w') as f:
                #     f.write(PARAM_STRING)
                # print(PARAM_STRING)


header = ["method", "seed", "min_eval_cfis_soundings", "min_fraction_valid", 'mult_noise_std', 'best_params'
            "30m_train_nrmse", "30m_train_r2", "30m_train_corr", "30m_test_nrmse", "30m_test_r2", "30m_test_corr",
            "30m_grassland_nrmse", "30m_corn_nrmse", "30m_soybean_nrmse", "30m_deciduous_forest_nrmse",
            "90m_nrmse", "150m_nrmse", "300m_nrmse", "600m_nrmse"]

if not os.path.isfile(RESULTS_SUMMARY_FILE):
    with open(RESULTS_SUMMARY_FILE, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
with open(RESULTS_SUMMARY_FILE, mode='a+') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for seed, row in results_rows.items():
        csv_writer.writerow(row)