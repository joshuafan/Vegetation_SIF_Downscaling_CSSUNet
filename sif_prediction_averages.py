"""
Runs pre-built ML methods over the channel averages of each tile (e.g. linear regression or gradient boosted tree)
"""
import hydra

import numpy as np
import os
import pandas as pd
import random
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Lasso, Ridge, LinearRegression, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import json
import math
import tile_transforms
import time
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sif_utils import plot_histogram, print_stats, remove_pure_tiles
import visualization_utils

# Set random seed for data shuffling
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Random seeds for model training
TRAIN_RANDOM_STATES = [1, 2, 3]

# Directories
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
PLOTS_DIR = os.path.join(DATA_DIR, "exploratory_plots")

# Train files
PROCESSED_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_2degree_random0")
TILE_AVERAGE_TRAIN_FILE = os.path.join(PROCESSED_DATASET_DIR, "tile_info_train.csv")
TILE_AVERAGE_VAL_FILE = os.path.join(PROCESSED_DATASET_DIR, "tile_info_val.csv")
TILE_AVERAGE_TEST_FILE = os.path.join(PROCESSED_DATASET_DIR, "tile_info_test.csv")
BAND_STATISTICS_FILE = os.path.join(PROCESSED_DATASET_DIR, "band_statistics_train.csv") #"band_statistics_pixels.csv") #"band_statistics_train.csv")
RES = (0.00026949458523585647, 0.00026949458523585647)
TILE_PIXELS = 100
TILE_SIZE_DEGREES = RES[0] * TILE_PIXELS

# CFIS eval files
CFIS_RESOLUTION = 300
CFIS_COARSE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_metadata.csv')
CFIS_EVAL_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_metadata_' + str(RESOLUTION_METERS) + 'm.csv')
TEST_FOLDS = [4]

# Dates/sources
# TRAIN_TROPOMI_DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
#                        "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
#                        "2018-09-16"]
# TRAIN_OCO2_DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
#                     "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
#                     "2018-09-16"]
# TEST_DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
#              "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
#              "2018-09-16"]

TRAIN_TROPOMI_DATES = ["2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19"]
TRAIN_OCO2_DATES = ["2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19"]
TEST_DATES = ["2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19"]

# TRAIN_TROPOMI_DATES = ["2018-07-08", "2018-08-05"]
# TRAIN_OCO2_DATES = ["2018-07-08", "2018-08-05"]
# TEST_DATES = ["2018-07-22", "2018-08-19"]

# TRAIN_SOURCES = ["TROPOMI"]
# TRAIN_SOURCES = ["OCO2"]
TRAIN_SOURCES = ["TROPOMI", "OCO2"]
TEST_SOURCES = ["OCO2"]

# Method
# METHOD = "1a_tropomi_Ridge_Regression"
# METHOD = "1b_tropomi_Gradient_Boosting_Regressor"
# METHOD = "1c_tropomi_MLP"
# METHOD = "2a_both_Ridge_Regression"
# METHOD = "2b_both_Gradient_Boosting_Regressor" # + str(MIN_CFIS_SOUNDINGS) + "_soundings"
METHOD = "2c_both_MLP"
# METHOD = "2d_both_KNN"
# METHOD = "3a_oco2_Ridge_Regression"
# METHOD = "3b_oco2_Gradient_Boosting_Regressor"
# METHOD = "3c_oco2_MLP"
# METHOD = "3_cfis_Linear_Regression"

# Other dataset params
NUM_TROPOMI_SAMPLES = 1000
NUM_OCO2_SAMPLES = 1000
NUM_OCO2_REPEATS = 1 #round(NUM_TROPOMI_SAMPLES / NUM_OCO2_SAMPLES)
MIN_SOUNDINGS = 5
MIN_CFIS_SOUNDINGS = 10
MAX_CLOUD_COVER = 0.2
MIN_INPUT = -3
MAX_INPUT = 3
REMOVE_PURE_TRAIN = False #True
PURE_THRESHOLD_TRAIN = 0.6


# Input feature names
# INPUT_COLUMNS = ['ref_5', 'ref_6']
INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils', 'missing_reflectance']

# INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
#                     'ref_10', 'ref_11']
# INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
#                     'ref_10', 'ref_11',
#                     'grassland_pasture', 'corn', 'soybean', 'shrubland',
#                     'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
#                     'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
#                     'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
#                     'developed_low_intensity', 'missing_reflectance']

COLUMNS_TO_STANDARDIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
OUTPUT_COLUMN = ['SIF']

# Crop types to look at when analyzing results
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest'] #, 'evergreen_forest', 'spring_wheat']



# Evaluation parameters
PURE_THRESHOLD_EVAL = 0.6
MIN_SIF_CLIP = 0.1
MAX_SIF_CLIP = None
MIN_SIF_PLOT = 0
MAX_SIF_PLOT = 2

# True vs predicted plot
OCO2_TRUE_VS_PREDICTED_PLOT = os.path.join(PLOTS_DIR, 'true_vs_predicted_sif_oco2_' + METHOD)
CFIS_TRUE_VS_PREDICTED_PLOT = os.path.join(PLOTS_DIR, 'true_vs_predicted_sif_cfis_' + METHOD)
CFIS_VS_OCO2_LOSS_PLOT = os.path.join(DATA_DIR, 'loss_plots/losses_' + METHOD + '_nrmse_cfis_vs_val_oco2.png')




def main():
    # Read datasets
    train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE)
    val_set = pd.read_csv(TILE_AVERAGE_VAL_FILE)
    test_set = pd.read_csv(TILE_AVERAGE_TEST_FILE)

    # Read band statistics
    band_statistics = pd.read_csv(BAND_STATISTICS_FILE)
    average_sif = band_statistics['mean'].iloc[-1]
    band_means = band_statistics['mean'].values[:-1]
    band_stds = band_statistics['std'].values[:-1]

    # Filter. Note that most of these filters are redundant with create_filtered_dataset.py
    train_tropomi_set = train_set[(train_set['source'] == 'TROPOMI') &
                                (train_set['num_soundings'] >= MIN_SOUNDINGS) &
                                (train_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                                (train_set['SIF'] >= MIN_SIF_CLIP) &
                                (train_set['date'].isin(TRAIN_TROPOMI_DATES))].copy()
    train_oco2_set = train_set[(train_set['source'] == 'OCO2') &
                                (train_set['num_soundings'] >= MIN_SOUNDINGS) &
                                (train_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                                (train_set['SIF'] >= MIN_SIF_CLIP) &
                                (train_set['date'].isin(TRAIN_OCO2_DATES))].copy()
    val_tropomi_set = val_set[(val_set['source'] == 'TROPOMI') &
                                (val_set['num_soundings'] >= MIN_SOUNDINGS) &
                                (val_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                                (val_set['SIF'] >= MIN_SIF_CLIP) &
                                (val_set['date'].isin(TRAIN_TROPOMI_DATES))].copy()
    val_oco2_set = val_set[(val_set['source'] == 'OCO2') &
                                (val_set['num_soundings'] >= MIN_SOUNDINGS) &
                                (val_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                                (val_set['SIF'] >= MIN_SIF_CLIP) &
                                (val_set['date'].isin(TRAIN_OCO2_DATES))].copy()
    test_tropomi_set = test_set[(test_set['source'] == 'TROPOMI') &
                                (test_set['num_soundings'] >= MIN_SOUNDINGS) &
                                (test_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                                (test_set['SIF'] >= MIN_SIF_CLIP) &
                                (test_set['date'].isin(TRAIN_TROPOMI_DATES))].copy()
    test_oco2_set = test_set[(test_set['source'] == 'OCO2') &
                                (test_set['num_soundings'] >= MIN_SOUNDINGS) &
                                (test_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                                (test_set['SIF'] >= MIN_SIF_CLIP) &
                                (test_set['date'].isin(TEST_DATES))].copy()
    train_oco2_set['SIF'] /= 1.04
    val_oco2_set['SIF'] /= 1.04
    test_oco2_set['SIF'] /= 1.04
    # print('TROPOMI sizes', len(train_tropomi_set), len(val_tropomi_set), len(test_tropomi_set))
    # print('OCO2 sizes', len(train_oco2_set), len(val_oco2_set), len(test_oco2_set))
    # exit(0)

    # Artificially remove pure tiles
    if REMOVE_PURE_TRAIN:
        train_tropomi_set = remove_pure_tiles(train_tropomi_set, threshold=PURE_THRESHOLD_TRAIN)
        train_oco2_set = remove_pure_tiles(train_oco2_set, threshold=PURE_THRESHOLD_TRAIN)

    # TODO refactor in progress
    # Read CFIS dataset
    cfis_coarse_metadata = pd.read_csv(CFIS_COARSE_METADATA_FILE)

    # Only include CFIS tiles with enough valid pixels
    cfis_coarse_metadata = cfis_coarse_metadata[(cfis_coarse_metadata['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                        (cfis_coarse_metadata['SIF'] >= MIN_SIF_CLIP)]

    # Read fine metadata at particular resolution
    cfis_eval_metadata = pd.read_csv(CFIS_EVAL_METADATA_FILE)
    cfis_eval_metadata = cfis_eval_metadata[(cfis_eval_metadata['SIF'] >= MIN_SIF_CLIP) &
                                            (cfis_eval_metadata['num_soundings'] >= MIN_EVAL_CFIS_SOUNDINGS) &
                                            (cfis_eval_metadata['fraction_valid'] >= MIN_EVAL_FRACTION_VALID) &
                                            (cfis_eval_metadata['tile_file'].isin(set(cfis_coarse_metadata['tile_file'])))]

    # Read dataset splits
    test_coarse_cfis_set = cfis_coarse_metadata[(cfis_coarse_metadata['fold'].isin(TEST_FOLDS)) &
                                            (cfis_coarse_metadata['date'].isin(TEST_DATES))].copy()
    test_eval_cfis_set = cfis_eval_metadata[(cfis_eval_metadata['fold'].isin(TEST_FOLDS)) &
                                            (cfis_eval_metadata['date'].isin(TEST_DATES))].copy()
                                        

    # Create shuffled train sets
    # combined_tropomi_set = pd.concat([train_tropomi_set, val_tropomi_set, test_tropomi_set])
    shuffled_tropomi_set = train_tropomi_set.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) #.iloc[0:NUM_TROPOMI_SAMPLES]
    shuffled_oco2_set = train_oco2_set.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) #.iloc[0:NUM_OCO2_SAMPLES]
    print('Shuffled tropomi set', shuffled_tropomi_set)
    # Filter train set to only include desired sources
    if 'TROPOMI' in TRAIN_SOURCES and 'OCO2' in TRAIN_SOURCES:
        print('Using both TROPOMI and OCO2')
        shuffled_oco2_repeated = pd.concat([shuffled_oco2_set] * NUM_OCO2_REPEATS)
        train_set = pd.concat([shuffled_tropomi_set, shuffled_oco2_repeated]) #.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        # Repeat OCO2 so that there's roughly the same number of OCO2 and TROPOMI points
        # train_set = pd.concat([train_tropomi_set, val_tropomi_set, test_tropomi_set, train_oco2_repeated])
    elif 'TROPOMI' in TRAIN_SOURCES:
        print('ONLY using TROPOMI')
        train_set = shuffled_tropomi_set
    elif 'OCO2' in TRAIN_SOURCES:
        print('ONLY using OCO2')
        train_set = shuffled_oco2_set
    else:
        print("Didn't specify valid sources :(")
        exit(0)

    # Shuffle train set
    train_set = train_set.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Record params
    PARAM_STRING = ''
    PARAM_STRING += '============= DATASET PARAMS =============\n'
    PARAM_STRING += ('Dataset dir: ' + PROCESSED_DATASET_DIR + '\n')
    PARAM_STRING += ('Train sources: ' + str(TRAIN_SOURCES) + '\n')
    if 'TROPOMI' in TRAIN_SOURCES:
        PARAM_STRING += ('Train TROPOMI dates: ' + str(TRAIN_TROPOMI_DATES) + '\n')
        PARAM_STRING += ('Num TROPOMI samples: ' + str(len(shuffled_tropomi_set)) + '\n')
    if 'OCO2' in TRAIN_SOURCES:
        PARAM_STRING += ('Train OCO-2 dates: ' + str(TRAIN_OCO2_DATES) + '\n')
        PARAM_STRING += ('Num OCO-2 samples: ' + str(len(shuffled_oco2_set)) + ', repeated ' + str(NUM_OCO2_REPEATS) + ' times\n')
    PARAM_STRING += ('Test dates: ' + str(TEST_DATES) + '\n')
    PARAM_STRING += ('Min soundings: ' + str(MIN_SOUNDINGS) + '\n')
    PARAM_STRING += ('Min SIF clip: ' + str(MIN_SIF_CLIP) + '\n')
    PARAM_STRING += ('Max cloud cover: ' + str(MAX_CLOUD_COVER) + '\n')
    PARAM_STRING += ('Train features: ' + str(INPUT_COLUMNS) + '\n')
    PARAM_STRING += ("Clip input features: " + str(MIN_INPUT) + " to " + str(MAX_INPUT) + " standard deviations from mean\n")
    if REMOVE_PURE_TRAIN:
        PARAM_STRING += ('Removing pure train tiles above ' + str(PURE_THRESHOLD_TRAIN) + '\n')
    PARAM_STRING += ('Pure threshold (eval): ' + str(PURE_THRESHOLD_EVAL) + '\n')
    PARAM_STRING += ('================= METHOD ===============\n')
    PARAM_STRING += ('Method name: ' + METHOD + '\n')


    # Print dataset info
    print('Total train samples:', len(train_set))
    print('Val samples (OCO2):', len(val_oco2_set))
    print('Eval CFIS samples:', len(eval_cfis_set))
    print('average sif (train, according to band statistics file)', average_sif)
    print('average sif (train, large tiles)', train_set['SIF'].mean())
    print('average sif (train, TROPOMI)', train_tropomi_set['SIF'].mean())
    print('average sif (train, OCO2)', train_oco2_set['SIF'].mean())
    print('average sif (val, TROPOMI)', val_tropomi_set['SIF'].mean())
    print('average sif (val, OCO2)', val_oco2_set['SIF'].mean())
    print('average sif (test, OCO2)', test_oco2_set['SIF'].mean())
    print('average sif (CFIS subtiles)', eval_cfis_set['SIF'].mean())


    # Standardize data
    for idx, column in enumerate(COLUMNS_TO_STANDARDIZE):
        train_set[column] = np.clip((train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
        val_set[column] = np.clip((val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
        val_tropomi_set[column] = np.clip((val_tropomi_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
        val_oco2_set[column] = np.clip((val_oco2_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
        test_oco2_set[column] = np.clip((test_oco2_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
        eval_cfis_set[column] = np.clip((eval_cfis_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)

        # column_mean = train_set[column].mean()
        # column_std = train_set[column].std()
        # cfis_column_mean = eval_cfis_set[column].mean()
        # cfis_column_std = eval_cfis_set[column].mean()
        # train_set[column] = np.clip((train_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
        # val_set[column] = np.clip((val_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
        # val_tropomi_set[column] = np.clip((val_tropomi_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
        # val_oco2_set[column] = np.clip((val_oco2_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
        # test_oco2_set[column] = np.clip((test_oco2_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
        # eval_cfis_set[column] = np.clip((eval_cfis_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)

        # # Histograms of standardized columns - MOVE (also plot by date)
        # plot_histogram(train_set[column].to_numpy(), "histogram_clipped_std_" + column + "_train.png")
        # plot_histogram(val_tropomi_set[column].to_numpy(), "histogram_clipped_std_" + column + "_val_tropomi.png")
        # plot_histogram(val_oco2_set[column].to_numpy(), "histogram_clipped_std_" + column + "_val_oco2.png")

        # train_set[column] = (train_set[column] - column_mean) / column_std
        # val_set[column] = (val_set[column] - column_mean) / column_std
        # val_tropomi_set[column] = (val_tropomi_set[column] - column_mean) / column_std
        # val_oco2_set[column] = (val_oco2_set[column] - column_mean) / column_std
        # test_oco2_set[column] = (test_oco2_set[column] - column_mean) / column_std
        # eval_cfis_set[column] = (eval_cfis_set[column] - column_mean) / column_std

        # eval_cfis_set[column] = (eval_cfis_set[column] - cfis_column_mean) / cfis_column_std

        # train_set[column] = train_set[column] + np.random.normal(loc=0, scale=0.1, size=len(train_set[column]))
        # plot_histogram(eval_cfis_set[column].to_numpy(), "histogram_clipped_std_" + column + "_cfis.png")


    # # Print averages of each feature (standardized by train stats)
    # print("************************************************************")
    # print('FEATURE AVERAGES')
    # for column_name in INPUT_COLUMNS:
    #     print('================== Feature:', column_name, '=================')
    #     print('Train:', round(np.mean(train_set[column_name]), 3))
    #     print('Train TROPOMI:', round(np.mean(train_tropomi_set[column_name]), 3))
    #     print('Train OCO2:', round(np.mean(train_oco2_set[column_name]), 3))
    #     print('Val OCO2:', round(np.mean(val_set[column_name]), 3))
    #     print('CFIS:', round(np.mean(eval_cfis_set[column_name]), 3))
    # print("************************************************************")

    # print('============= train set ==============')
    # pd.set_option('display.max_columns', 500)
    # print(train_set.tail())
    # print('============= cfis set ==============')
    # print(eval_cfis_set.head())
    # exit(0)



    X_train = train_set[INPUT_COLUMNS]
    Y_train = train_set[OUTPUT_COLUMN].values.ravel()
    X_val_tropomi = val_tropomi_set[INPUT_COLUMNS]
    Y_val_tropomi = val_tropomi_set[OUTPUT_COLUMN].values.ravel()
    X_val_oco2 = val_oco2_set[INPUT_COLUMNS]
    Y_val_oco2 = val_oco2_set[OUTPUT_COLUMN].values.ravel()
    X_test_oco2 = test_oco2_set[INPUT_COLUMNS]
    Y_test_oco2 = test_oco2_set[OUTPUT_COLUMN].values.ravel()
    X_cfis = eval_cfis_set[INPUT_COLUMNS]
    Y_cfis = eval_cfis_set[OUTPUT_COLUMN].values.ravel()


    print('X train', X_train.shape)
    # Fit models on band averages (with various hyperparam settings)
    regression_models = dict()
    if 'Linear_Regression' in METHOD:
        regression_model = LinearRegression().fit(X_train, Y_train)
        regression_models['linear'] = [regression_model]
    elif 'Lasso' in METHOD:
        alphas = [0.001, 0.01, 0.1, 1, 10, 100]
        for alpha in alphas:
            regression_model = Lasso(alpha=alpha).fit(X_train, Y_train)
            param_string = 'alpha=' + str(alpha)
            regression_models[param_string] = [regression_model]    
    elif 'Ridge_Regression' in METHOD:
        alphas = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        for alpha in alphas:
            models = []
            for random_state in TRAIN_RANDOM_STATES:
                regression_model = Ridge(alpha=alpha).fit(X_train, Y_train) # HuberRegressor(alpha=alpha, max_iter=1000).fit(X_train, Y_train)
                models.append(regression_model)
            param_string = 'alpha=' + str(alpha)
            regression_models[param_string] = models
    elif "Gradient_Boosting_Regressor" in METHOD:
        max_iter_values = [30, 100, 300, 1000] #
        max_depth_values = [2, 3, None]
        # n_estimator_values = [700, 1000]
        # learning_rates = [0.01, 0.1, 0.5]
        # max_depths = [1, 10]
        for max_iter in max_iter_values:
            for max_depth in max_depth_values:
                models = []
                for random_state in TRAIN_RANDOM_STATES:
                    regression_model = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=max_depth, learning_rate=0.1, random_state=random_state).fit(X_train, Y_train)
                    models.append(regression_model)
                param_string = 'max_iter=' + str(max_iter) + ', max_depth=' + str(max_depth)
                print(param_string)
                regression_models[param_string] = models
    elif "MLP" in METHOD:
        hidden_layer_sizes = [(20,), (100,), (20, 20), (100, 100), (100, 100, 100)] #[(100, 100)] # 
        learning_rate_inits = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]  # [1e-3] #
        max_iter = 1000
        for hidden_layer_size in hidden_layer_sizes:
            for learning_rate_init in learning_rate_inits:
                models = []
                for random_state in TRAIN_RANDOM_STATES:
                    regression_model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, learning_rate_init=learning_rate_init, max_iter=max_iter, random_state=random_state).fit(X_train, Y_train)
                    models.append(regression_model)
                param_string = 'hidden_layer_sizes=' + str(hidden_layer_size) + ', learning_rate_init=' + str(learning_rate_init)
                print(param_string)
                regression_models[param_string] = models
    elif "KNN" in METHOD:
        n_neighbors_options = [3, 5, 10, 20]
        for n_neighbors in n_neighbors_options:
            regression_model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, Y_train)
            param_string = 'n_neighbors=' + str(n_neighbors)
            print(param_string)
            regression_models[param_string] = [regression_model]

    else:
        print("Unsupported method")
        exit(1)

    # print('Coefficients', regression_model.coef_)
    best_loss = float('inf')
    best_params = 'N/A'
    best_idx = -1

    # Loop through all hyperparameter settings we trained models for, and compute
    # loss on the validation set
    average_losses_val = []
    average_losses_cfis = []
    for params, models in regression_models.items():
        losses_val = []
        losses_cfis = []
        for model in models:
            predictions_val = model.predict(X_val_oco2)
            predictions_cfis = model.predict(X_cfis)
            loss_val = math.sqrt(mean_squared_error(Y_val_oco2, predictions_val)) / average_sif  
            loss_cfis = math.sqrt(mean_squared_error(Y_cfis, predictions_cfis)) / average_sif
            # if loss_val < best_loss:
            #     best_loss = loss_val
            #     best_params = params
            #     best_model = model
            losses_val.append(loss_val)
            losses_cfis.append(loss_cfis)
        average_loss_val = sum(losses_val) / len(losses_val)
        average_loss_cfis = sum(losses_cfis) / len(losses_cfis)
        print(params + ': avg val loss', round(average_loss_val, 4), 'avg CFIS loss', round(average_loss_cfis, 4))

        if average_loss_val < best_loss:
            best_loss = average_loss_val
            best_params = params
            best_idx = np.argmin(losses_val)
        average_losses_val.append(average_loss_val)
        average_losses_cfis.append(average_loss_cfis)

    print('Best params:', best_params)
    PARAM_STRING += ('Best params: ' + str(best_params) + '\n')

    print('================== CFIS vs OCO-2 performance =================')
    print_stats(average_losses_cfis, average_losses_val, average_sif, ax=plt.gca(), fit_intercept=True)
    plt.title('CFIS vs OCO-2 (' + METHOD + ')')
    plt.xlabel('Val OCO-2 NRMSE')
    plt.ylabel('Val CFIS NRMSE')
    plt.savefig(CFIS_VS_OCO2_LOSS_PLOT)
    plt.close()

    # Record performances
    columns = ['Val OCO-2', 'Test OCO-2', 'Test OCO-2 grassland', 'Test OCO-2 corn', 'Test OCO-2 soybean', 'Test OC0-2 deciduous forest']
    all_r2 = {'all_val': [], 'all_test': [], 'grassland_pasture': [], 'corn': [], 'soybean': [], 'deciduous_forest': []}
    all_nrmse = {'all_val': [], 'all_test': [], 'grassland_pasture': [], 'corn': [], 'soybean': [], 'deciduous_forest': []}

    for idx, model in enumerate(regression_models[best_params]):
        # Only plot graph for best model
        plot_results = (idx == best_idx)

        # Use the best model to make predictions
        predictions_train = model.predict(X_train)
        predictions_val_tropomi = model.predict(X_val_tropomi)
        predictions_val_oco2 = model.predict(X_val_oco2)
        predictions_test_oco2 = model.predict(X_test_oco2)
        predictions_cfis = model.predict(X_cfis)

        predictions_train = np.clip(predictions_train, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
        predictions_val_tropomi = np.clip(predictions_val_tropomi, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
        predictions_val_oco2 = np.clip(predictions_val_oco2, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
        predictions_test_oco2 = np.clip(predictions_test_oco2, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
        predictions_cfis = np.clip(predictions_cfis, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)

        #scale_factor = np.mean(linear_predictions_cfis) / np.mean(Y_cfis)
        #print('Scale factor', scale_factor)
        #Y_cfis *= scale_factor
        #print('Coef', linear_regression.coef_)

        # Quantile analysis
        #squared_errors = (Y_cfis - linear_predictions_cfis) ** 2
        #indices = squared_errors.argsort() #Ascending order of squared error

        #percentiles = [0, 0.05, 0.1, 0.2]
        #for percentile in percentiles:
        #    cutoff_idx = int((1 - percentile) * len(Y_cfis))
        #    indices_to_include = indices[:cutoff_idx]
        #    nrmse = math.sqrt(np.mean(squared_errors[indices_to_include])) / average_sif
        #    corr, _ = pearsonr(Y_cfis[indices_to_include], linear_predictions_cfis[indices_to_include])
        #    print('Excluding ' + str(int(percentile*100)) + '% worst predictions')
        #    print('NRMSE', round(nrmse, 3))
        #    print('Corr', round(corr, 3))


        if plot_results:
            # Print NRMSE, correlation, R2 on train/validation set
            print('============== Train set stats =====================')
            print_stats(Y_train, predictions_train, average_sif)

            # print('============== Val set stats (TROPOMI) =====================')
            # print_stats(Y_val_tropomi, predictions_val_tropomi, average_sif)

            print('============== Val set stats (OCO2) =====================')
            val_r2, val_nrmse = print_stats(Y_val_oco2, predictions_val_oco2, average_sif, ax=plt.gca(), fit_intercept=False)
            plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
            plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
            plt.title('OCO2 val set: true vs predicted SIF (' + METHOD + ')')
            plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_val.png')
            plt.close()
        else:
            val_r2, val_nrmse = print_stats(Y_val_oco2, predictions_val_oco2, average_sif, ax=None, fit_intercept=False, print_report=False)
        all_r2['all_val'].append(val_r2)
        all_nrmse['all_val'].append(val_nrmse)

        if plot_results:
            print('============== Test set stats (OCO2) =====================')
            test_r2, test_nrmse = print_stats(Y_test_oco2, predictions_test_oco2, average_sif, ax=plt.gca(), fit_intercept=False)
            plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
            plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
            plt.title('OCO2 test set: true vs predicted SIF (' + METHOD + ')')
            plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '.png')
            plt.close()
        else:
            test_r2, test_nrmse = print_stats(Y_test_oco2, predictions_test_oco2, average_sif, ax=None, fit_intercept=False, print_report=False)
        all_r2['all_test'].append(test_r2)
        all_nrmse['all_test'].append(test_nrmse)

        # Plot true vs. predicted for each crop on OCO-2 (for each crop)
        if plot_results:
            fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
            fig.suptitle('True vs predicted SIF (OCO-2) by crop: ' + METHOD)
        for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
            predicted = predictions_test_oco2[test_oco2_set[crop_type] > PURE_THRESHOLD_EVAL]
            true = Y_test_oco2[test_oco2_set[crop_type] > PURE_THRESHOLD_EVAL]
            if len(predicted) >= 2:
                if plot_results:
                    print('======================= (OCO2) CROP: ', crop_type, '==============================')
                    ax = axeslist.ravel()[idx]
                    crop_r2, crop_nrmse = print_stats(true, predicted, average_sif, ax=ax, fit_intercept=False)
                    ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                    ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                    ax.set_title(crop_type)
                else:
                    crop_r2, crop_nrmse = print_stats(true, predicted, average_sif, ax=None, fit_intercept=False)
                all_r2[crop_type].append(crop_r2)
                all_nrmse[crop_type].append(crop_nrmse)

            # Fit linear model on just this crop, to see how strong the relationship is
            # X_train_crop = X_train.loc[train_set[crop_type] > PURE_THRESHOLD]
            # Y_train_crop = Y_train[train_set[crop_type] > PURE_THRESHOLD]
            # X_val_crop = X_val.loc[test_set[crop_type] > PURE_THRESHOLD]
            # Y_val_crop = Y_val[test_set[crop_type] > PURE_THRESHOLD]
            # crop_regression = LinearRegression().fit(X_train_crop, Y_train_crop)
            # predicted_oco2_crop = crop_regression.predict(X_val_crop)
            # print(' ----- Crop specific regression -----')
            # #print('Coefficients:', crop_regression.coef_)
            # print_stats(Y_oco2_crop, predicted_oco2_crop, average_sif)
    
        # Plot true vs. predicted for that specific crop
        # axeslist.ravel()[idx].scatter(true, predicted)
        # axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
        if plot_results:
            plt.tight_layout()
            fig.subplots_adjust(top=0.92)
            plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
            plt.close()


        # Print stats for CFIS subtiles
        if plot_results:
            print('========== CFIS Eval subtile stats ===========')
            print_stats(Y_cfis, predictions_cfis, average_sif, ax=plt.gca(), fit_intercept=False)  #eval_cfis_set['SIF'].mean())  #average_sif)
            plt.title('CFIS: true vs predicted SIF (' + METHOD + ')')
            plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
            plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
            plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '.png')
            plt.close()

        # # Plot true vs. predicted for each crop on CFIS (for each crop)
        # fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
        # fig.suptitle('True vs predicted SIF (CFIS): ' + METHOD)
        # for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
        #     predicted = predictions_cfis[eval_cfis_set[crop_type] > PURE_THRESHOLD_EVAL]  # Find CFIS tiles which are "purely" this crop type
        #     true = Y_cfis[eval_cfis_set[crop_type] > PURE_THRESHOLD_EVAL]
        #     ax = axeslist.ravel()[idx]
        #     print('======================= (CFIS) CROP: ', crop_type, '==============================')
        #     if len(predicted) >= 2:
        #         print_stats(true, predicted, average_sif, ax=ax, fit_intercept=False)
        #         ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
        #         ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
        #         ax.set_title(crop_type)

        #         # # Fit linear model on just this crop, to see how strong the relationship is
        #         # X_train_crop = X_train.loc[X_train[crop_type] > 0.5]
        #         # Y_train_crop = Y_train[X_train[crop_type] > 0.5]
        #         # X_cfis_crop = X_cfis.loc[X_cfis[crop_type] > PURE_THRESHOLD]
        #         # Y_cfis_crop = Y_cfis[X_cfis[crop_type] > PURE_THRESHOLD]
        #         # crop_regression = LinearRegression().fit(X_cfis_crop, Y_cfis_crop)
        #         # predicted_cfis_crop = crop_regression.predict(X_cfis_crop)
        #         # print(' ----- CFIS Crop specific regression -----')
        #         # print('Coefficients:', crop_regression.coef_)
        #         # print_stats(Y_cfis_crop, predicted_cfis_crop, average_sif)
        
        #     # Plot true vs. predicted for that specific crop
        #     # axeslist.ravel()[idx].scatter(true, predicted)
        #     # axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
        #     # axeslist.ravel()[idx].set_xlim(left=0, right=2)
        #     # axeslist.ravel()[idx].set_ylim(bottom=0, top=2)
        #     # axeslist.ravel()[idx].set_title(crop_type)

        # plt.tight_layout()
        # fig.subplots_adjust(top=0.92)
        # plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
        # plt.close()

        # # Print summary statistics for each source
        # fig, axeslist = plt.subplots(ncols=len(SOURCES), nrows=1, figsize=(12, 6))
        # fig.suptitle('True vs predicted SIF, by source: ' + METHOD)

        # for idx, source in enumerate(SOURCES):
        #     # Obtain global model's predictions for data points with this source 
        #     print('=================== ALL DATES: ' + source + ' ======================')
        #     predicted = predictions_val[val_set['source'] == source]
        #     true = Y_val[val_set['source'] == source]
        #     print('Number of rows', len(predicted))
        #     if len(predicted) < 2:
        #         idx += 1
        #         continue

        #     print_stats(true, predicted, average_sif)

        #     # Scatter plot of true vs predicted
        #     axeslist.ravel()[idx].scatter(true, predicted)
        #     axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
        #     axeslist.ravel()[idx].set_xlim(left=MIN_SIF, right=MAX_SIF)
        #     axeslist.ravel()[idx].set_ylim(bottom=MIN_SIF, top=MAX_SIF)
        #     axeslist.ravel()[idx].set_title(source)

        #     # Fit linear model on just this source
        #     X_train_source = X_train.loc[train_set['source'] == source]
        #     Y_train_source = Y_train[train_set['source'] == source]
        #     X_val_source = X_val.loc[val_set['source'] == source]
        #     Y_val_source = Y_val[val_set['source'] == source]
        #     source_regression = LinearRegression().fit(X_train_source, Y_train_source)
        #     predicted_val_source = source_regression.predict(X_val_source)
        #     print(' ----- Source-specific regression -----')
        #     print('Coefficients:', source_regression.coef_)
        #     print_stats(Y_val_source, predicted_val_source, average_sif)

        # plt.tight_layout()
        # fig.subplots_adjust(top=0.92)
        # plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_sources.png')
        # plt.close()

        # Print statistics and plot by date and source
        if plot_results:
            fig, axeslist = plt.subplots(ncols=len(TEST_SOURCES), nrows=len(TEST_DATES), figsize=(7*len(TEST_SOURCES), 6*len(TEST_DATES)))
            fig.suptitle('True vs predicted SIF, by date/source: ' + METHOD)
            idx = 0
            for date in TEST_DATES:
                for source in TEST_SOURCES:
                    # Obtain global model's predictions for data points with this date/source 
                    predicted = predictions_test_oco2[(test_oco2_set['date'] == date) & (test_oco2_set['source'] == source)]
                    true = Y_test_oco2[(test_oco2_set['date'] == date) & (test_oco2_set['source'] == source)]
                    print('=================== Date ' + date + ', ' + source + ' ======================')
                    assert(len(predicted) == len(true))
                    if len(predicted) < 2:
                        idx += 1
                        continue

                    # Print stats (true vs predicted)
                    ax = axeslist.ravel()[idx]
                    print_stats(true, predicted, average_sif, ax=ax, fit_intercept=False)
                    ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                    ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                    ax.set_title(date + ', ' + source)
                    idx += 1

                    # Scatter plot of true vs predicted
                    # axeslist.ravel()[idx].scatter(true, predicted)
                    # axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
                    # axeslist.ravel()[idx].set_xlim(left=MIN_SIF, right=MAX_SIF)
                    # axeslist.ravel()[idx].set_ylim(bottom=MIN_SIF, top=MAX_SIF)
                    # axeslist.ravel()[idx].set_title(date + ', ' + source)

                    # Fit linear model on just this date
                    # X_train_date = X_train.loc[(train_set['date'] == date)] # & (train_set['source'] == source)]
                    # Y_train_date = Y_train[(train_set['date'] == date)] # & (train_set['source'] == source)]
                    # X_test_date = X_test.loc[(test_set['date'] == date) & (test_set['source'] == source)]
                    # Y_test_date = Y_test[(test_set['date'] == date) & (test_set['source'] == source)]
                    # date_regression = LinearRegression().fit(X_train_date, Y_train_date)
                    # predicted_test_date = date_regression.predict(X_test_date)
                    # print('----- Date and source-specific linear regression -----')
                    # print_stats(Y_test_date, predicted_test_date, average_sif)

            plt.tight_layout()
            fig.subplots_adjust(top=0.96)
            plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_dates_sources.png')
            plt.close()









    # # Plot true vs predicted for each date and source
    # fig, axeslist = plt.subplots(ncols=2, nrows=5, figsize=(12, 12))
    # fig.suptitle('True vs predicted SIF (OCO-2) by date: ' + METHOD)
    # for idx, date in enumerate(DATES):
    #     predicted = predictions_oco2[val_oco2_set['date'] == date]
    #     true = Y_oco2[val_oco2_set['date'] == date]
    #     print('======================= DATE: ', date, '==============================')
    #     print(len(predicted), 'tiles for date', date)
    #     if len(predicted) >= 2:
    #         print(' ----- All-date regression ------')
    #         print_stats(true, predicted, average_sif)

    #         # Fit linear model on just this date
    #         X_train_date = X_train.loc[train_set['date'] == date]
    #         Y_train_date = Y_train[train_set['date'] == date]
    #         X_oco2_date = X_oco2.loc[val_oco2_set['date'] == date]
    #         Y_oco2_date = Y_oco2[val_oco2_set['date'] == date]
    #         date_regression = LinearRegression().fit(X_train_date, Y_train_date)
    #         predicted_oco2_date = date_regression.predict(X_oco2_date)
    #         print(' ----- Date-specific regression -----')
    #         # print('Coefficients:', date_regression.coef_)
    #         print_stats(Y_oco2_date, predicted_oco2_date, average_sif)
    
    #     # Plot true vs. predicted for that specific date
    #     axeslist.ravel()[idx].scatter(true, predicted)
    #     axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    #     axeslist.ravel()[idx].set_xlim(left=0, right=2)
    #     axeslist.ravel()[idx].set_ylim(bottom=0, top=2)
    #     axeslist.ravel()[idx].set_title(crop_type)

    # plt.tight_layout()
    # fig.subplots_adjust(top=0.92)
    # plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_dates.png')
    # plt.close()


    # Plot "map" of test OCO-2 datapoints, colored by error
    sif_cmap = plt.get_cmap('RdYlGn')
    plt.figure(figsize=(30, 10))
    date = '2018-08-05'
    errors_date = predictions_test_oco2[test_oco2_set['date'] == date] - Y_test_oco2[test_oco2_set['date'] == date]
    train_oco2_set_date = train_oco2_set[train_oco2_set['date'] == date]
    test_oco2_set_date = test_oco2_set[test_oco2_set['date'] == date]
    train_tropomi_set_date = train_tropomi_set[train_tropomi_set['date'] == date]
    test_tropomi_set_date = test_tropomi_set[test_tropomi_set['date'] == date]
    scatterplot = plt.scatter(test_oco2_set_date['lon'], test_oco2_set_date['lat'], c=errors_date, cmap=sif_cmap, vmin=-0.5, vmax=0.5)
    plt.colorbar(scatterplot)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Prediction error: Test OCO-2 points, ' + date)
    plt.savefig(os.path.join(DATA_DIR, 'exploratory_plots/test_oco2_points_errors_' + date + '.png'))
    plt.close()

    # Plot "map" of train TROPOMI datapoints
    plt.figure(figsize=(30, 10))
    scatterplot = plt.scatter(train_tropomi_set_date['lon'], train_tropomi_set_date['lat'], c=train_tropomi_set_date['SIF'], cmap=sif_cmap, vmin=0, vmax=1.5)
    plt.colorbar(scatterplot)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Train TROPOMI points, ' + date)
    plt.savefig(os.path.join(DATA_DIR, 'exploratory_plots/train_tropomi_points_' + date + '.png'))
    plt.close()

    # Plot "map" of train OCO-2 datapoints
    plt.figure(figsize=(30, 10))
    scatterplot = plt.scatter(train_oco2_set_date['lon'], train_oco2_set_date['lat'], c=train_oco2_set_date['SIF'], cmap=sif_cmap, vmin=0, vmax=1.5)
    plt.colorbar(scatterplot)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Train OCO2 points, ' + date)
    plt.savefig(os.path.join(DATA_DIR, 'exploratory_plots/train_oco2_points_' + date + '.png'))
    plt.close()

    # Plot "map" of test TROPOMI datapoints
    plt.figure(figsize=(30, 10))
    scatterplot = plt.scatter(test_tropomi_set_date['lon'], test_tropomi_set_date['lat'], c=test_tropomi_set_date['SIF'], cmap=sif_cmap, vmin=0, vmax=1.5)
    plt.colorbar(scatterplot)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Test TROPOMI points, ' + date)
    plt.savefig(os.path.join(DATA_DIR, 'exploratory_plots/test_tropomi_points_' + date + '.png'))
    plt.close()

    # # Plot high error points
    # NUM_TILES = 10
    # errors = np.abs(Y_test_oco2 - predictions_test_oco2)
    # sorted_indices = errors.argsort()  # Ascending order of distance
    # high_error_indices = sorted_indices[-NUM_TILES:][::-1]

    # # Set up image transforms
    # transform_list = []
    # transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
    # transform_list.append(tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT))
    # transform = transforms.Compose(transform_list)

    # for high_error_idx in high_error_indices:
    #     row = test_oco2_set.iloc[high_error_idx]
    #     high_error_tile = transform(np.load(row['tile_file']))
    #     valid_mask = np.logical_not(high_error_tile[-1, :, :])
    #     tile_description = 'oco2_high_error_' + os.path.basename(row['tile_file'])
    #     title = 'Lat ' + str(round(row['lat'], 4)) + ', Lon ' + str(round(row['lon'], 4))
    #     if 'date' in row:
    #         title += (', Date ' + row['date'])
    #     title += ('\n(True SIF: ' + str(round(Y_test_oco2[high_error_idx], 3)) + ', Predicted SIF: ' + str(round(predictions_test_oco2[high_error_idx], 3)) + ')')
    #     print('Tile:', tile_description)
    #     print('header:', title)
    #     visualization_utils.plot_tile(high_error_tile, valid_mask, row['lon'], row['lat'], row['date'],
    #                                   TILE_SIZE_DEGREES, tile_description=tile_description, title=title)

    # Write final results to file
    r2_mean = {}
    r2_std = {}
    nrmse_mean = {}
    nrmse_std = {}
    for key in all_r2:
        r2_mean[key] = round(np.mean(all_r2[key]), 3)
        r2_std[key] = round(np.std(all_r2[key], ddof=1), 4)
        nrmse_mean[key] = round(np.mean(all_nrmse[key]), 3)
        nrmse_std[key] = round(np.std(all_nrmse[key], ddof=1), 4)
    print('R2 all runs:', all_r2)
    print('NRMSE all runs:', all_nrmse)


    PARAM_STRING += '================ RESULTS =================\n'
    PARAM_STRING += 'R2 means: ' + json.dumps(r2_mean) + '\n'
    PARAM_STRING += 'NRMSE means: ' + json.dumps(nrmse_mean) + '\n'
    PARAM_STRING += 'R2 stds: ' + json.dumps(r2_std) + '\n'
    PARAM_STRING += 'NRMSE stds: ' + json.dumps(nrmse_std) + '\n'
    results_dir = os.path.join('results', METHOD)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    filename = time.strftime("%Y%m%d-%H%M%S") + '.txt'
    with open(os.path.join(results_dir, filename), mode='w') as f:
        f.write(PARAM_STRING)
    print(PARAM_STRING)





if __name__ == "__main__":
    main()