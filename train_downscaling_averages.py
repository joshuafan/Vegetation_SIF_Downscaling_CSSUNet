"""
Runs pre-built ML methods over the channel averages of each tile (e.g. linear regression or gradient boosted tree)
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sif_utils import plot_histogram, print_stats

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
OCO2_DIR = os.path.join(DATA_DIR, "OCO2")
PLOTS_DIR = os.path.join(DATA_DIR, "exploratory_plots")

DATES = ["2016-06-15", "2016-08-01"]
# METHOD = "9a_Ridge_Regression_cfis" #_5soundings"
# METHOD = "9b_Gradient_Boosting_Regressor_cfis" #_5soundings"
# METHOD = "9c_MLP" #_10soundings"
# METHOD = "10a_Ridge_Regression_both"
# METHOD = "10b_Gradient_Boosting_Regressor_both"
# METHOD = "10c_MLP_both"
# METHOD = "11a_Ridge_Regression_oco2"
# METHOD = "11b_Gradient_Boosting_Regressor_oco2"
METHOD = "11c_MLP_oco2"
# TRAIN_SOURCES = ['CFIS', 'OCO2']
TRAIN_SOURCES = ['OCO2']
print("METHOD:", METHOD, "- SOURCES:", TRAIN_SOURCES)

PURE_THRESHOLD = 0.7
MIN_OCO2_SOUNDINGS = 5
MAX_OCO2_CLOUD_COVER = 0.2
MIN_FINE_CFIS_SOUNDINGS = 5

MIN_INPUT = -3
MAX_INPUT = 3
MIN_SIF = None
MAX_SIF = None
MIN_SIF_CLIP = 0.2
MAX_SIF_CLIP = 1.5
MIN_SIF_PLOT = -0.2
MAX_SIF_PLOT = 1.6
NUM_RUNS = 3

# Train files
FINE_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_train.csv')
FINE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_val.csv')
FINE_AVERAGES_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_test.csv')
COARSE_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_train.csv')
COARSE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_val.csv')
COARSE_AVERAGES_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_test.csv')
OCO2_METADATA_TRAIN_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_train.csv')
OCO2_METADATA_VAL_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_val.csv')
OCO2_METADATA_TEST_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_test.csv')
BAND_STATISTICS_CSV_FILE = os.path.join(CFIS_DIR, 'cfis_band_statistics_train.csv')

# True vs predicted plot
CFIS_TRUE_VS_PREDICTED_PLOT = os.path.join(PLOTS_DIR, 'true_vs_predicted_sif_cfis_' + METHOD)

# Read datasets
fine_train_set = pd.read_csv(FINE_AVERAGES_TRAIN_FILE)
fine_val_set = pd.read_csv(FINE_AVERAGES_VAL_FILE)
fine_test_set = pd.read_csv(FINE_AVERAGES_TEST_FILE)
coarse_train_set = pd.read_csv(COARSE_AVERAGES_TRAIN_FILE)
coarse_val_set = pd.read_csv(COARSE_AVERAGES_VAL_FILE)
coarse_test_set = pd.read_csv(COARSE_AVERAGES_TEST_FILE)
oco2_train_set = pd.read_csv(OCO2_METADATA_TRAIN_FILE)
oco2_val_set = pd.read_csv(OCO2_METADATA_VAL_FILE)
oco2_test_set = pd.read_csv(OCO2_METADATA_TEST_FILE)

# Filter fine CFIS sets
fine_train_set = fine_train_set[(fine_train_set['num_soundings'] >= MIN_FINE_CFIS_SOUNDINGS)] # & (fine_train_set['SIF'] >= MIN_SIF_CLIP)]
fine_val_set = fine_val_set[(fine_val_set['num_soundings'] >= MIN_FINE_CFIS_SOUNDINGS)] # & (fine_val_set['SIF'] >= MIN_SIF_CLIP)]
fine_test_set = fine_test_set[(fine_test_set['num_soundings'] >= MIN_FINE_CFIS_SOUNDINGS)] # & (fine_test_set['SIF'] >= MIN_SIF_CLIP)]

# Filter OCO2 sets
oco2_train_set = oco2_train_set[(oco2_train_set['num_soundings'] >= MIN_OCO2_SOUNDINGS) & (oco2_train_set['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER)]
oco2_val_set = oco2_val_set[(oco2_val_set['num_soundings'] >= MIN_OCO2_SOUNDINGS) & (oco2_val_set['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER)]
oco2_test_set = oco2_test_set[(oco2_test_set['num_soundings'] >= MIN_OCO2_SOUNDINGS) & (oco2_test_set['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER)]

# Construct combined train set
print('CFIS Coarse train set:', len(coarse_train_set))
print('OCO2 train set:', len(oco2_train_set))

# Filter train set to only include desired sources
if 'OCO2' in TRAIN_SOURCES and 'CFIS' in TRAIN_SOURCES:
    print('Using both OCO2 and CFIS')
    # Repeat OCO2 so that there's roughly the same number of OCO2 and TROPOMI points
    # train_oco2_repeated = pd.concat([train_oco2_set] * NUM_OCO2_REPEATS)
    train_set = pd.concat([oco2_train_set, coarse_train_set])
elif 'OCO2' in TRAIN_SOURCES:
    print('ONLY using OCO2')
    train_set = oco2_train_set
elif 'CFIS' in TRAIN_SOURCES:
    print('ONLY using CFIS')
    train_set = coarse_train_set
else:
    print("Didn't specify valid sources :(")
    exit(0)

# Shuffle train set
train_set = train_set.sample(frac=1).reset_index(drop=True)
print('Train set samples', len(train_set))

# Read band statistics
train_statistics = pd.read_csv(BAND_STATISTICS_CSV_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

# Print dataset info
print('Fine train samples', len(fine_train_set))
print('Fine val samples', len(fine_val_set))
print('Fine test samples', len(fine_test_set))
print('Coarse train samples', len(coarse_train_set))
print('Coarse val samples', len(coarse_val_set))
print('Coarse test samples', len(coarse_test_set))
print('OCO2 train samples', len(oco2_train_set))
print('OCO2 val samples', len(oco2_val_set))
print('OCO2 test samples', len(oco2_test_set))
print('Average SIF (fine, train)', fine_train_set['SIF'].mean())
print('Average SIF (fine, val)', fine_val_set['SIF'].mean())
print('Average SIF (fine, test)', fine_test_set['SIF'].mean())
print('Average SIF (coarse, train)', coarse_train_set['SIF'].mean())
print('Average SIF (coarse, val)', coarse_val_set['SIF'].mean())
print('Average SIF (coarse, test)', coarse_test_set['SIF'].mean())
print('Average SIF (OCO2, train)', oco2_train_set['SIF'].mean())
print('Average SIF (OCO2, val)', oco2_val_set['SIF'].mean())
print('Average SIF (OCO2, test)', oco2_test_set['SIF'].mean())
print('Average SIf from band statistics file', sif_mean)

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
#                     'ref_10', 'ref_11',
#                     'grassland_pasture', 'corn', 'soybean', 'shrubland',
#                     'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
#                     'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
#                     'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
#                     'developed_low_intensity', 'missing_reflectance']

COLUMNS_TO_STANDARDIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
OUTPUT_COLUMN = ['SIF']

# COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
#                     'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
#                     'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
#                     'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
#                     'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
#                     'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
#                     'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
#                     'lentils']

# Crop types to look at when analyzing results
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest'] #, 'evergreen_forest', 'spring_wheat']


# Standardize data
for idx, column in enumerate(COLUMNS_TO_STANDARDIZE):
    fine_train_set[column] = np.clip((fine_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    fine_val_set[column] = np.clip((fine_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    fine_test_set[column] = np.clip((fine_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    train_set[column] = np.clip((train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    coarse_val_set[column] = np.clip((coarse_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    # coarse_train_set[column] = np.clip((coarse_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    # coarse_test_set[column] = np.clip((coarse_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    # oco2_train_set[column] = np.clip((oco2_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    # oco2_val_set[column] = np.clip((oco2_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    # oco2_test_set[column] = np.clip((oco2_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)


X_fine_train = fine_train_set[INPUT_COLUMNS]
Y_fine_train = fine_train_set[OUTPUT_COLUMN].values.ravel()
X_fine_val = fine_val_set[INPUT_COLUMNS]
Y_fine_val = fine_val_set[OUTPUT_COLUMN].values.ravel()
X_fine_test = fine_test_set[INPUT_COLUMNS]
Y_fine_test = fine_test_set[OUTPUT_COLUMN].values.ravel()
X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
X_coarse_val = coarse_val_set[INPUT_COLUMNS]
Y_coarse_val = coarse_val_set[OUTPUT_COLUMN].values.ravel()
# X_coarse_test = coarse_test_set[INPUT_COLUMNS]
# Y_coarse_test = coarse_test_set[OUTPUT_COLUMN].values.ravel()
# X_coarse_train = coarse_train_set[INPUT_COLUMNS]
# Y_coarse_train = coarse_train_set[OUTPUT_COLUMN].values.ravel()
# X_coarse_val = coarse_val_set[INPUT_COLUMNS]
# Y_coarse_val = coarse_val_set[OUTPUT_COLUMN].values.ravel()
# X_coarse_test = coarse_test_set[INPUT_COLUMNS]
# Y_coarse_test = coarse_test_set[OUTPUT_COLUMN].values.ravel()

# Fit models on band averages (with various hyperparam settings)
regression_models = dict()
if 'Linear_Regression' in METHOD:
    regression_model = LinearRegression().fit(X_train, Y_train) #X_train, Y_train)
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
        for run in range(1):
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
            for run in range(NUM_RUNS):
                regression_model = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=max_depth, learning_rate=0.1).fit(X_train, Y_train)
                models.append(regression_model)
            param_string = 'max_iter=' + str(max_iter) + ', max_depth=' + str(max_depth)
            print(param_string)
            regression_models[param_string] = models
elif "MLP" in METHOD:
    hidden_layer_sizes = [(10,), (20,), (50,), (100,), (20, 20), (100, 100), (100, 100, 100)] #[(100, 100)] # 
    learning_rate_inits =  [1e-2, 1e-3, 1e-4]  # [1e-3] #
    max_iter = 1000
    for hidden_layer_size in hidden_layer_sizes:
        for learning_rate_init in learning_rate_inits:
            models = []
            for run in range(NUM_RUNS):
                regression_model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, learning_rate_init=learning_rate_init, max_iter=max_iter).fit(X_train, Y_train)
                models.append(regression_model)
            param_string = 'hidden_layer_sizes=' + str(hidden_layer_size) + ', learning_rate_init=' + str(learning_rate_init)
            print(param_string)
            regression_models[param_string] = models
else:
    print("Unsupported method")
    exit(1)

# print('Coefficients', regression_model.coef_)
best_loss = float('inf')
best_params = 'N/A'

# Loop through all hyperparameter settings we trained models for, and compute
# loss on the validation set
average_losses_val = []
for params, models in regression_models.items():
    losses_val = []
    for model in models:
        predictions_val = model.predict(X_coarse_val)
        loss_val = math.sqrt(mean_squared_error(Y_coarse_val, predictions_val)) / sif_mean  
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params
            best_model = model
        losses_val.append(loss_val)
    average_loss_val = sum(losses_val) / len(losses_val)
    print(params + ': avg val loss', round(average_loss_val, 4))
    average_losses_val.append(average_loss_val)

print('Best params:', best_params)
# print(best_model.coef_)

# Use the best model to make predictions
predictions_train = best_model.predict(X_train)
predictions_coarse_val = best_model.predict(X_coarse_val)
# predictions_coarse_test = best_model.predict(X_coarse_test)
predictions_fine_train = best_model.predict(X_fine_train)
predictions_fine_val = best_model.predict(X_fine_val)
predictions_fine_test = best_model.predict(X_fine_test)

# predictions_train = np.clip(predictions_train, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
# predictions_coarse_val = np.clip(predictions_coarse_val, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
# # predictions_coarse_test = np.clip(predictions_coarse_test, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
# predictions_fine_train = np.clip(predictions_fine_train, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
# predictions_fine_val = np.clip(predictions_fine_val, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)
# predictions_fine_test = np.clip(predictions_fine_test, a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)


# Print NRMSE, correlation, R2 on train/validation set
print('============== Train set stats =====================')
print_stats(Y_train, predictions_train, sif_mean)

print('============== Coarse val set stats =====================')
print_stats(Y_coarse_val, predictions_coarse_val, sif_mean, ax=plt.gca())
plt.title('Coarse val set: true vs predicted SIF (' + METHOD + ')')
plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_coarse_val.png')
plt.close()

# print('============== Coarse test set stats =====================')
# print_stats(Y_coarse_test, predictions_coarse_test, sif_mean)

print('============== Fine train set stats =====================')
print_stats(Y_fine_train, predictions_fine_train, sif_mean)

print('============== Fine val set stats =====================')
print_stats(Y_fine_val, predictions_fine_val, sif_mean, ax=plt.gca())
plt.title('Fine val set: true vs predicted SIF (' + METHOD + ')')
plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_fine_val.png')
plt.close()

print('============== Fine test set stats =====================')
print_stats(Y_fine_test, predictions_fine_test, sif_mean)

print('============= Fine train set stats: just use surrounding coarse tile =============')
print_stats(Y_fine_train, fine_train_set['coarse_sif'].to_numpy(), sif_mean, ax=plt.gca())
plt.title('Fine train set: true vs predicted SIF (predict coarse SIF)')
plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_fine_vs_coarse_train.png')
plt.close()

print('============= Fine val set stats: just use surrounding coarse tile =============')
print_stats(Y_fine_val, fine_val_set['coarse_sif'].to_numpy(), sif_mean, ax=plt.gca())
plt.title('Fine val set: true vs predicted SIF (predict coarse SIF)')
plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_fine_vs_coarse_val.png')
plt.close()

# Plot true vs. predicted for each crop on CFIS fine (for each crop)
fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('True vs predicted SIF by crop: ' + METHOD)
for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    predicted = predictions_fine_val[fine_val_set[crop_type] > PURE_THRESHOLD]
    true = Y_fine_val[fine_val_set[crop_type] > PURE_THRESHOLD]
    ax = axeslist.ravel()[idx]
    print('======================= (CFIS fine) CROP: ', crop_type, '==============================')
    print(len(predicted), 'pixels that are pure', crop_type)
    if len(predicted) >= 2:
        print(' ----- All crop regression ------')
        print_stats(true, predicted, sif_mean, ax=ax)
        ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
        ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
        ax.set_title(crop_type)
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

plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()





# Print statistics and plot by date
fig, axeslist = plt.subplots(ncols=1, nrows=len(DATES), figsize=(6, 6*len(DATES)))
fig.suptitle('True vs predicted SIF, by date: ' + METHOD)
idx = 0
for date in DATES:
    # Obtain global model's predictions for data points with this date
    predicted = predictions_fine_val[fine_val_set['date'] == date]
    true = Y_fine_val[fine_val_set['date'] == date]
    print('=================== Date ' + date + ' ======================')
    print('Number of rows', len(predicted))
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

    # Scatter plot of true vs predicted
    # axeslist.ravel()[idx].scatter(true, predicted)
    # axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    # axeslist.ravel()[idx].set_xlim(left=MIN_SIF, right=MAX_SIF_CLIP)
    # axeslist.ravel()[idx].set_ylim(bottom=MIN_SIF, top=MAX_SIF_CLIP)
    # axeslist.ravel()[idx].set_title(date + ', ' + source)

    # Fit linear model on just this date
    # X_train_date = X_train.loc[(train_set['date'] == date)] # & (train_set['source'] == source)]
    # Y_train_date = Y_train[(train_set['date'] == date)] # & (train_set['source'] == source)]
    # X_test_date = X_test.loc[(test_set['date'] == date) & (test_set['source'] == source)]
    # Y_test_date = Y_test[(test_set['date'] == date) & (test_set['source'] == source)]
    # date_regression = LinearRegression().fit(X_train_date, Y_train_date)
    # predicted_test_date = date_regression.predict(X_test_date)
    # print('----- Date and source-specific linear regression -----')
    # print_stats(Y_test_date, predicted_test_date, sif_mean)

plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_dates.png')
plt.close()
