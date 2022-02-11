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
MIN_COARSE_FRACTION_VALID_PIXELS = 0.1

# Only EVALUATE on CFIS fine-resolution pixels with at least this number of soundings (measurements)
MIN_FINE_CFIS_SOUNDINGS = 30 #[30] #[1, 5, 10, 20, 30] # # 100, 250] #[100, 300, 1000, 3000]
eps = 1e-5

# For resolutions greater than 30m, only evaluate on grid cells where at least this fraction
# of 30m pixels have any CFIS data
MIN_FINE_FRACTION_VALID_PIXELS = 0.9-eps #[0.1, 0.3, 0.5, 0.7] # [0.5] #[0.5]

# Resolutions to consider
RESOLUTION_METERS = 30 #, 90, 150, 300, 600]

# Dates/sources
DATES = ["2016-06-15", "2016-08-01"]
TRAIN_DATES = ["2016-06-15", "2016-08-01"]
TEST_DATES = ["2016-06-15", "2016-08-01"]

# List of sources to use (either CFIS or OCO-2)
TRAIN_SOURCES = ['CFIS_fine']  # ['CFIS', 'OCO2']
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
# COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
#                     'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
#                     'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
#                     'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
#                     'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
#                     'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
#                     'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
#                     'lentils']

COLUMNS_TO_STANDARDIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
COLUMNS_TO_LOG = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7', 'ref_10', 'ref_11']

# Input feature names
if args.log:
    INPUT_COLUMNS = ['log_ref_1', 'log_ref_2', 'log_ref_3', 'log_ref_4', 'log_ref_5', 'log_ref_6', 'log_ref_7']  #, 'log_ref_10', 'log_ref_11']
    OUTPUT_COLUMN = ['SIF']  # actually SIF will be logged too, but this is handled later
else:
    INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7', 'NDVI'] #,
                        # 'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
                        # 'grassland_pasture', 'corn', 'soybean',
                        # 'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                        # 'woody_wetlands', 'open_water', 'alfalfa',
                        # 'developed_low_intensity', 'developed_med_intensity', 'missing_reflectance']
    OUTPUT_COLUMN = ['SIF']

# Special column groups
ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']
REFLECTANCE_BANDS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7']
                    # 'ref_10', 'ref_11']

# Crop types to look at when analyzing results
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

# Read fine metadata at particular resolution, and do initial filtering
CFIS_FINE_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_metadata_' + str(RESOLUTION_METERS) + 'm.csv')
cfis_fine_metadata = pd.read_csv(CFIS_FINE_METADATA_FILE)
cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['SIF'] >= MIN_SIF_CLIP) &
                                (cfis_fine_metadata['tile_file'].isin(set(cfis_coarse_metadata['tile_file'])))]
cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['num_soundings'] >= MIN_FINE_CFIS_SOUNDINGS) &
                                        (cfis_fine_metadata['fraction_valid'] >= MIN_FINE_FRACTION_VALID_PIXELS)]  # Avoid roundoff errors

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



# Shuffle train set
fine_train_set = fine_train_set.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Read band statistics
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

# Inject multiplicative noise optionally
if args.multiplicative_noise:
    fine_train_set = pd.concat([fine_train_set] * args.mult_noise_repeats)
    fine_train_set = fine_train_set.reset_index(drop=True)
    for idx in range(len(fine_train_set)):
        noise = 1 + np.random.normal(loc=0, scale=args.mult_noise_std)
        fine_train_set.loc[idx, REFLECTANCE_BANDS] = fine_train_set.loc[idx, REFLECTANCE_BANDS] * noise

# Feature engineering        
if args.log:
    for idx, column in enumerate(COLUMNS_TO_LOG):
        log_column_values = np.nan_to_num(np.log(fine_train_set[column] + 1e-5))
        log_mean = np.mean(log_column_values)
        log_std = np.std(log_column_values)
        log_column = "log_" + column
        fine_train_set[log_column] = np.nan_to_num((np.log(fine_train_set[column] + 1e-5) - log_mean) / log_std)
        fine_val_set[log_column] = np.nan_to_num((np.log(fine_val_set[column] + 1e-5) - log_mean) / log_std)
        fine_test_set[log_column] = np.nan_to_num((np.log(fine_test_set[column] + 1e-5) - log_mean) / log_std)

if args.normalize:
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
            fine_train_set[new_col_name] = np.nan_to_num(fine_train_set[col1] / fine_train_set[col2], posinf=0, neginf=0)
            fine_val_set[new_col_name] = np.nan_to_num(fine_val_set[col1] / fine_val_set[col2], posinf=0, neginf=0)
            fine_test_set[new_col_name] = np.nan_to_num(fine_test_set[col1] / fine_test_set[col2], posinf=0, neginf=0)

if args.standardize:
    for idx, column in enumerate(COLUMNS_TO_STANDARDIZE):
        # TODO - This uses precomputed band means/stds. If desired you can compute them on the fly. 
        fine_train_set[column] = np.clip((fine_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
        fine_val_set[column] = np.clip((fine_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
        fine_test_set[column] = np.clip((fine_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)

# Record performances for this setting
all_r2 = {'grassland_pasture': [], 'corn': [], 'soybean': [], 'deciduous_forest': []}
all_nrmse = {'grassland_pasture': [], 'corn': [], 'soybean': [], 'deciduous_forest': []}
all_corr = {'grassland_pasture': [], 'corn': [], 'soybean': [], 'deciduous_forest': []}

print("Fine train set example")
print(fine_train_set.head())
# Create model per cover type
for cover_type in COVER_COLUMN_NAMES:
    print('==================================================================')
    print('************* LAND COVER:', cover_type, '****************')
    print('==================================================================')

    # Filter all sets for this cover type
    fine_train_set_cover = fine_train_set[fine_train_set[cover_type] >= PURE_THRESHOLD]
    fine_val_set_cover = fine_val_set[fine_val_set[cover_type] >= PURE_THRESHOLD]
    fine_test_set_cover = fine_test_set[fine_test_set[cover_type] >= PURE_THRESHOLD]

    # Select X/Y columns
    X_fine_train = fine_train_set_cover[INPUT_COLUMNS]
    Y_fine_train = fine_train_set_cover[OUTPUT_COLUMN].values.ravel()
    X_fine_val = fine_val_set_cover[INPUT_COLUMNS]
    Y_fine_val = fine_val_set_cover[OUTPUT_COLUMN].values.ravel()
    X_fine_test = fine_test_set_cover[INPUT_COLUMNS]
    Y_fine_test = fine_test_set_cover[OUTPUT_COLUMN].values.ravel()
    print("X_fine_train", X_fine_train.shape)
    print("Y_fine_train", Y_fine_train.shape)
    print("X_fine_test", X_fine_test.shape)
    print("Y_fine_test", Y_fine_test.shape)

    # If "log" is set, log SIF also
    if args.log:
        Y_fine_train = np.log(Y_fine_train)
        Y_fine_val = np.log(Y_fine_val)
        Y_fine_test = np.log(Y_fine_test)

    # Fit models on band averages (with various hyperparam settings)
    regression_models = dict()
    if 'Linear_Regression' in args.method:
        regression_model = LinearRegression().fit(X_fine_train, Y_fine_train) #X_train, Y_train)
        regression_models['linear'] = [regression_model]
    elif 'Lasso' in args.method:
        alphas = [0.001, 0.01, 0.1, 1, 10, 100]
        for alpha in alphas:
            regression_model = Lasso(alpha=alpha).fit(X_fine_train, Y_fine_train)
            param_string = 'alpha=' + str(alpha)
            regression_models[param_string] = [regression_model]
    elif 'Ridge_Regression' in args.method:
        alphas = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        for alpha in alphas:
            models = []
            for random_state in TRAIN_RANDOM_STATES:
                regression_model = Ridge(alpha=alpha, random_state=random_state).fit(X_fine_train, Y_fine_train) # HuberRegressor(alpha=alpha, max_iter=1000).fit(X_train, Y_train)
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
                    regression_model = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=max_depth, learning_rate=0.1, random_state=random_state).fit(X_fine_train, Y_fine_train)
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
                    regression_model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, learning_rate_init=learning_rate_init, max_iter=max_iter, random_state=random_state).fit(X_fine_train, Y_fine_train)
                    models.append(regression_model)
                param_string = 'hidden_layer_sizes=' + str(hidden_layer_size) + ', learning_rate_init=' + str(learning_rate_init)
                print(param_string)
                regression_models[param_string] = models
    elif "nearest_neighbors" in args.method:
        num_neighbors = [5, 10, 20]
        for n in num_neighbors:
            models = [KNeighborsRegressor(n_neighbors=n, weights="distance").fit(X_fine_train, Y_fine_train)]
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
    # print("Coefs:", regression_models[best_params][0].coef_)

    # If we logged SIF, transform labels back to original values
    if args.log:
        Y_fine_train = np.exp(Y_fine_train)
        Y_fine_val = np.exp(Y_fine_val)
        Y_fine_test = np.exp(Y_fine_test)

    # Loop through trained models
    for idx, model in enumerate(regression_models[best_params]):
        # Only plot graph for best model
        is_best_model = (idx == best_idx)

        # Use the best model to make predictions
        predictions_fine_train = model.predict(X_fine_train)
        predictions_fine_val = model.predict(X_fine_val)
        predictions_fine_test = model.predict(X_fine_test)

        PLOT_PREFIX = CFIS_TRUE_VS_PREDICTED_PLOT + '_' + cover_type

        # If we logged SIF, transform predictions back to original values
        if args.log:
            predictions_fine_train = np.exp(predictions_fine_train)
            predictions_fine_val = np.exp(predictions_fine_val)
            predictions_fine_test = np.exp(predictions_fine_test)

        if is_best_model:
            # print('============== CFIS fine train set stats =====================')
            # fine_train_r2, fine_train_nrmse, fine_train_corr = print_stats(Y_fine_train, predictions_fine_train, sif_mean, ax=plt.gca(), fit_intercept=False)
            # plt.title('True vs predicted SIF (' + METHOD_READABLE + '): ' + str(int(RESOLUTION_METERS)) + 'm pixels, train tiles')
            # plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
            # plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
            # plt.savefig(PLOT_PREFIX + '_fine_train.png')
            # plt.close()

            # print('============== CFIS fine val set stats =====================')
            # fine_val_r2, fine_val_nrmse, fine_val_corr = print_stats(Y_fine_val, predictions_fine_val, sif_mean, ax=plt.gca(), fit_intercept=False)
            # plt.title('True vs predicted SIF (' + METHOD_READABLE + '): ' + str(int(RESOLUTION_METERS)) + 'm pixels, val tiles')
            # plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
            # plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
            # plt.savefig(PLOT_PREFIX + '_fine_val.png')
            # plt.close()

            print('============== CFIS fine test set stats =====================')
            fine_test_r2, fine_test_nrmse, fine_test_corr = print_stats(Y_fine_test, predictions_fine_test, sif_mean, ax=plt.gca(), fit_intercept=False)
            plt.title('True vs predicted SIF (' + METHOD_READABLE + '): ' + str(int(RESOLUTION_METERS)) + 'm pixels, test tiles')
            plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
            plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
            plt.savefig(PLOT_PREFIX + '_fine_test.png')
            plt.close()
        else:
            fine_train_r2, fine_train_nrmse, fine_train_corr = print_stats(Y_fine_train, predictions_fine_train, sif_mean,
                                                                            ax=None, fit_intercept=False, print_report=False)
            fine_val_r2, fine_val_nrmse, fine_val_corr = print_stats(Y_fine_val, predictions_fine_val, sif_mean,
                                                                        ax=None, fit_intercept=False, print_report=False)
            fine_test_r2, fine_test_nrmse, fine_test_corr = print_stats(Y_fine_test, predictions_fine_test, sif_mean,
                                                                        ax=None, fit_intercept=False, print_report=False)

        all_r2[cover_type].append(fine_test_r2)
        all_nrmse[cover_type].append(fine_test_nrmse)
        all_corr[cover_type].append(fine_test_corr)

        # Trivial method: use surrounding coarse tile SIF
        if is_best_model:
            predictions_train_predict_coarse = np.clip(fine_train_set_cover['coarse_sif'].to_numpy(), a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
            predictions_val_predict_coarse = np.clip(fine_val_set_cover['coarse_sif'].to_numpy(), a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
            predictions_test_predict_coarse = np.clip(fine_test_set_cover['coarse_sif'].to_numpy(), a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)

            # print('============= (TRIVIAL: PREDICT COARSE) - Fine train set stats =============')
            # print_stats(Y_fine_train, predictions_train_predict_coarse, sif_mean, fit_intercept=False, ax=plt.gca())
            # plt.title('Fine train set: true vs predicted SIF (predict coarse SIF)')
            # plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
            # plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
            # plt.savefig(PLOT_PREFIX + '_fine_vs_coarse_train.png')
            # plt.close()

            # print('============= (TRIVIAL: PREDICT COARSE) - Fine val set stats =============')
            # print_stats(Y_fine_val, predictions_val_predict_coarse, sif_mean, fit_intercept=False, ax=plt.gca())
            # plt.title('Fine val set: true vs predicted SIF (predict coarse SIF)')
            # plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
            # plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
            # plt.savefig(PLOT_PREFIX + '_fine_vs_coarse_val.png')
            # plt.close()

            print('============= (TRIVIAL: PREDICT COARSE) - Fine test set stats =============')
            print_stats(Y_fine_test, predictions_test_predict_coarse, sif_mean, fit_intercept=False, ax=plt.gca())
            plt.title('Fine test set: true vs predicted SIF (predict coarse SIF)')
            plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
            plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
            plt.savefig(PLOT_PREFIX + '_fine_vs_coarse_test.png')
            plt.close()

