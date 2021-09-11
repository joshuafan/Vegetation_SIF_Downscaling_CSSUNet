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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sif_utils import plot_histogram, print_stats

parser = argparse.ArgumentParser()
parser.add_argument('-method', "--method", choices=["Ridge_Regression", "Gradient_Boosting_Regressor", "MLP"], type=str, help='Method type. MLP is the fully-connected artificial neural netwoprk.')
parser.add_argument('-multiplicative_noise', "--multiplicative_noise", action='store_true')
parser.add_argument('-mult_noise_std', "--mult_noise_std", default=0.2, type=float, help="If the 'multiplicative_noise' augmentation is used, multiply each channel by (1+eps), where eps ~ N(0, mult_noise_std)")
parser.add_argument('-mult_noise_repeats', "--mult_noise_repeats", default=10, type=int, help="How many times to repeat each example (with different multiplicative noise)")
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
DATA_DIR = "../data"

# Train files
CFIS_COARSE_METADATA_FILE = os.path.join(DATA_DIR, 'cfis_coarse_metadata.csv')
OCO2_METADATA_FILE = os.path.join(DATA_DIR, 'oco2_metadata.csv')
BAND_STATISTICS_FILE = os.path.join(DATA_DIR, 'cfis_band_statistics_train.csv')

# Only include CFIS tiles where at least this fraction of pixels have CFIS
# fine-resolution data
MIN_COARSE_FRACTION_VALID_PIXELS = 0.1

# Only EVALUATE on CFIS fine-resolution pixels with at least this number of soundings (measurements)
MIN_FINE_CFIS_SOUNDINGS = 30
eps = 1e-5

# For resolutions greater than 30m, only evaluate on grid cells where at least this fraction
# of 30m pixels have any CFIS data
MIN_FINE_FRACTION_VALID_PIXELS = 0.9-eps

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
RESULTS_DIR = "baseline_results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
CFIS_TRUE_VS_PREDICTED_PLOT = os.path.join(PLOTS_DIR, 'true_vs_predicted_sif_cfis_' + args.method)
RESULTS_SUMMARY_FILE = os.path.join(RESULTS_DIR, "results_summary_BASELINE.csv")
results_rows = {s: [args.method, s, MIN_FINE_CFIS_SOUNDINGS, MIN_FINE_FRACTION_VALID_PIXELS, args.mult_noise_std] for s in TRAIN_RANDOM_STATES}

# Input feature names
INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
                    'grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity', 'missing_reflectance']
ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']
REFLECTANCE_BANDS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11']
OUTPUT_COLUMN = ['SIF']

# Which columns to standardize. We do not standardize columns that come from averaging binary masks,
# since those are already going to be in the [0, 1] range.
COLUMNS_TO_STANDARDIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']

# Crop types to look at when analyzing results
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean'] 



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
    CFIS_FINE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_metadata_' + str(resolution) + 'm.csv')
    cfis_fine_metadata = pd.read_csv(CFIS_FINE_METADATA_FILE)
    cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['SIF'] >= MIN_SIF_CLIP) &
                                    (cfis_fine_metadata['tile_file'].isin(set(cfis_coarse_metadata['tile_file'])))]
    cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['num_soundings'] >= MIN_FINE_CFIS_SOUNDINGS) &
                                            (cfis_fine_metadata['fraction_valid'] >= MIN_FINE_FRACTION_VALID_PIXELS)]  # Avoid roundoff errors

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
        train_set = pd.concat([train_set] * args.mult_noise_repeats)
        train_set = train_set.reset_index(drop=True)
        for idx in range(len(train_set)):
            noise = 1 + np.random.normal(loc=0, scale=args.mult_noise_std)
            train_set.loc[idx, REFLECTANCE_BANDS] = train_set.loc[idx, REFLECTANCE_BANDS] * noise

    # Standardize data
    for idx, column in enumerate(COLUMNS_TO_STANDARDIZE):
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
                regression_model = Ridge(alpha=alpha, random_state=random_state).fit(X_train, Y_train)
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
        hidden_layer_sizes = [(100,), (20, 20), (100, 100), (100, 100, 100)]
        learning_rate_inits =  [1e-2, 1e-3, 1e-4]
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


    # Record performances for this setting
    all_r2 = {'all_coarse_val': [], 'all_fine_train': [], 'all_fine_val': [], 'all_fine_test': [], 'grassland_pasture': [], 'corn': [], 'soybean': []}
    all_nrmse = {'all_coarse_val': [], 'all_fine_train': [], 'all_fine_val': [], 'all_fine_test': [], 'grassland_pasture': [], 'corn': [], 'soybean': []}
    all_corr = {'all_coarse_val': [], 'all_fine_train': [], 'all_fine_val': [], 'all_fine_test': [], 'grassland_pasture': [], 'corn': [], 'soybean': []}

    print('========================================= FILTER ======================================================')
    print('*** Resolution', resolution)
    print('*** Min coarse fraction valid pixels', min_coarse_fraction_valid)
    print('*** Min fine soundings', min_fine_cfis_soundings)
    print('*** Min fine fraction valid pixels', min_fraction_valid_pixels)
    print('===================================================================================================')

    # Loop through trained models
    for idx, model in enumerate(regression_models[best_params]):
        # Only plot graphs for best model
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

        PLOT_PREFIX = CFIS_TRUE_VS_PREDICTED_PLOT + '_res' + str(resolution) + '_coarsefractionvalid' + str(min_coarse_fraction_valid) + '_finesoundings' + str(min_fine_cfis_soundings) + '_finefractionvalid' + str(min_fraction_valid_pixels)

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
        predictions_fine_train_filtered = model.predict(X_fine_train_filtered)
        predictions_fine_val_filtered = model.predict(X_fine_val_filtered)
        predictions_fine_test_filtered = model.predict(X_fine_test_filtered)
        predictions_fine_train_filtered = np.clip(predictions_fine_train_filtered, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
        predictions_fine_val_filtered = np.clip(predictions_fine_val_filtered, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
        predictions_fine_test_filtered = np.clip(predictions_fine_test_filtered, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)

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
        for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
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
            for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
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
    print('==================== SUMMARY (method: ' + METHOD_READABLE + ', resolution: ' + str(resolution) + 'm,) ===================')
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
            results_rows[seed].extend([all_nrmse["all_fine_train"][i],
                                        all_r2["all_fine_train"][i],
                                        all_corr["all_fine_train"][i],
                                        all_nrmse["all_fine_test"][i],
                                        all_r2["all_fine_test"][i],
                                        all_corr["all_fine_test"][i],
                                        all_nrmse["grassland_pasture"][i],
                                        all_nrmse["corn"][i],
                                        all_nrmse["soybean"][i]])
        else:
            results_rows[seed].append(all_nrmse["all_fine_train"][i])




header = ["method", "seed", "min_eval_cfis_soundings", "min_fraction_valid", 'mult_noise_std',
            "30m_train_nrmse", "30m_train_r2", "30m_train_corr", "30m_test_nrmse", "30m_test_r2", "30m_test_corr",
            "30m_grassland_nrmse", "30m_corn_nrmse", "30m_soybean_nrmse",
            "90m_nrmse", "150m_nrmse", "300m_nrmse", "600m_nrmse"]

if not os.path.isfile(RESULTS_SUMMARY_FILE):
    with open(RESULTS_SUMMARY_FILE, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
with open(RESULTS_SUMMARY_FILE, mode='a+') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for seed, row in results_rows.items():
        csv_writer.writerow(row)