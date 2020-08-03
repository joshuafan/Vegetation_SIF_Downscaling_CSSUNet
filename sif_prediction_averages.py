"""
Runs pre-built ML methods over the channel averages of each tile (e.g. linear regression or gradient boosted tree)
"""
import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Lasso, Ridge, LinearRegression, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt
from sif_utils import plot_histogram, print_stats

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"

# Train files
# TRAIN_DATE = "2018-08-01" # "2018-07-16"
# TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + TRAIN_DATE)
PROCESSED_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_all_2") #_tropomi_train")
TILE_AVERAGE_TRAIN_FILE = os.path.join(PROCESSED_DATASET_DIR, "tile_info_train.csv")
TILE_AVERAGE_VAL_FILE = os.path.join(PROCESSED_DATASET_DIR, "tile_info_val.csv")
TILE_AVERAGE_TEST_FILE = os.path.join(PROCESSED_DATASET_DIR, "tile_info_test.csv")

BAND_STATISTICS_FILE = os.path.join(PROCESSED_DATASET_DIR, "band_statistics_train.csv")

# CFIS eval files
# EVAL_DATE = "2016-08-01" #"2016-07-16"
# EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + EVAL_DATE)
CFIS_AVERAGE_FILE = os.path.join(PROCESSED_DATASET_DIR, "cfis_subtiles_filtered.csv")


DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
         "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
         "2018-09-16"]
TRAIN_SOURCES = ["TROPOMI", "OCO2"]
VAL_SOURCES = ["OCO2"]
MIN_CFIS_SOUNDINGS = 200

# METHOD = "1a_tropomi_Ridge_Regression"
# METHOD = "1b_tropomi_Gradient_Boosting_Regressor"
# METHOD = "1c_tropomi_MLP"
# METHOD = "2a_both_Ridge_Regression_467samples" 
# METHOD = "2b_both_Gradient_Boosting_Regressor_467samples" # + str(MIN_CFIS_SOUNDINGS) + "_soundings"
METHOD = "2c_both_MLP_467samples"
# METHOD = "3a_oco2_Ridge_Regression"
# METHOD = "3b_oco2_Gradient_Boosting_Regressor"
# METHOD = "3c_oco2_MLP_100samples"
# METHOD = "3_cfis_Linear_Regression"
CFIS_TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_cfis_' + METHOD
OCO2_TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_oco2_' + METHOD

PURE_THRESHOLD = 0.6
MIN_SOUNDINGS = 3
MIN_INPUT = -3
MAX_INPUT = 3
MIN_SIF = 0.2
MAX_SIF = 1.7
MAX_PRED = 1.7
MAX_CFIS_SIF = 2.7
NUM_RUNS = 3
NUM_OCO2_SAMPLES = 467
NUM_OCO2_REPEATS = 2 #round(0.5 * 51840 / NUM_OCO2_SAMPLES)
NUM_TROPOMI_SAMPLES = 1000
print('Num repeats:', NUM_OCO2_REPEATS)

# Read datasets
train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE)
val_set = pd.read_csv(TILE_AVERAGE_VAL_FILE)
test_set = pd.read_csv(TILE_AVERAGE_TEST_FILE)
eval_cfis_set = pd.read_csv(CFIS_AVERAGE_FILE)

# train_set = train_set[train_set['date'] == '2018-08-05']
# val_set = val_set[val_set['date'] == '2018-08-05']

# Filter number of CFIS soundings
# print('Before: CFIS tiles', len(eval_cfis_set))
eval_cfis_set = eval_cfis_set[eval_cfis_set['num_soundings'] >= MIN_CFIS_SOUNDINGS]

# Filter
train_tropomi_set = train_set[train_set['source'] == 'TROPOMI'].copy().iloc[0:NUM_TROPOMI_SAMPLES]
train_oco2_set = train_set[train_set['source'] == 'OCO2'].copy().iloc[0:NUM_OCO2_SAMPLES]
val_tropomi_set = val_set[val_set['source'] == 'TROPOMI'].copy()
val_oco2_set = val_set[val_set['source'] == 'OCO2'].copy()

# Filter train set to only include desired sources
if 'TROPOMI' in TRAIN_SOURCES and 'OCO2' in TRAIN_SOURCES:
    print('Using both TROPOMI and OCO2')
    # Repeat OCO2 so that there's roughly the same number of OCO2 and TROPOMI points
    train_oco2_repeated = pd.concat([train_oco2_set] * NUM_OCO2_REPEATS)
    train_set = pd.concat([train_tropomi_set, train_oco2_repeated])
elif 'TROPOMI' in TRAIN_SOURCES:
    print('ONLY using TROPOMI')
    train_set = train_tropomi_set
elif 'OCO2' in TRAIN_SOURCES:
    print('ONLY using OCO2')
    train_set = train_oco2_set
else:
    print("Didn't specify valid sources :(")
    exit(0)


# Read band statistics
band_statistics = pd.read_csv(BAND_STATISTICS_FILE)
average_sif = band_statistics['mean'].iloc[-1]
band_means = band_statistics['mean'].values[:-1]
band_stds = band_statistics['std'].values[:-1]

# Print dataset info
print('Train samples:', len(train_set))
print('    TROPOMI:', len(train_tropomi_set))
print('    OCO-2:', len(train_oco2_set))
print('Val samples (TROPOMI):', len(val_tropomi_set))
print('Val samples (OCO2):', len(val_oco2_set))
print('Eval CFIS samples:', len(eval_cfis_set))
print('average sif (train, according to band statistics file)', average_sif)
print('average sif (train, large tiles)', train_set['SIF'].mean())
print('average sif (train, TROPOMI)', train_tropomi_set['SIF'].mean())
print('average sif (train, OCO2)', train_oco2_set['SIF'].mean())
print('average sif (val, TROPOMI)', val_tropomi_set['SIF'].mean())
print('average sif (val, OCO2)', val_oco2_set['SIF'].mean())
print('average sif (test, OCO2)', test_set['SIF'].mean())
print('average sif (CFIS subtiles)', eval_cfis_set['SIF'].mean())

# Input feature names
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
for column in COLUMNS_TO_STANDARDIZE:
    column_mean = train_set[column].mean()
    column_std = train_set[column].std()
    train_set[column] = np.clip((train_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
    val_set[column] = np.clip((val_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
    val_tropomi_set[column] = np.clip((val_tropomi_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
    val_oco2_set[column] = np.clip((val_oco2_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
    test_set[column] = np.clip((test_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
    eval_cfis_set[column] = np.clip((eval_cfis_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)

    # # Histograms of standardized columns - MOVE (also plot by date)
    # plot_histogram(train_set[column].to_numpy(), "histogram_clipped_std_" + column + "_train.png")
    # plot_histogram(val_tropomi_set[column].to_numpy(), "histogram_clipped_std_" + column + "_val_tropomi.png")
    # plot_histogram(val_oco2_set[column].to_numpy(), "histogram_clipped_std_" + column + "_val_oco2.png")

    # train_set[column] = (train_set[column] - column_mean) / column_std
    # val_set[column] = (val_set[column] - column_mean) / column_std
    # val_tropomi_set[column] = (val_tropomi_set.loc[:, column] - column_mean) / column_std
    # val_oco2_set[column] = (val_oco2_set.loc[:, column] - column_mean) / column_std
    # eval_cfis_set[column] = (eval_cfis_set[column] - column_mean) / column_std

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

print('============= train set ==============')
pd.set_option('display.max_columns', 500)
print(train_set.tail())
print('============= cfis set ==============')
print(eval_cfis_set.head())
exit(0)
X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
X_val_tropomi = val_tropomi_set[INPUT_COLUMNS]
Y_val_tropomi = val_tropomi_set[OUTPUT_COLUMN].values.ravel()
X_val_oco2 = val_oco2_set[INPUT_COLUMNS]
Y_val_oco2 = val_oco2_set[OUTPUT_COLUMN].values.ravel()
X_test = test_set[INPUT_COLUMNS]
Y_test = test_set[OUTPUT_COLUMN].values.ravel()
X_cfis = eval_cfis_set[INPUT_COLUMNS]
Y_cfis = eval_cfis_set[OUTPUT_COLUMN].values.ravel()


# Fit models on band averages (with various hyperparam settings)
regression_models = dict()
if 'Linear_Regression' in METHOD:
    regression_model = LinearRegression().fit(X_cfis, Y_cfis) #X_train, Y_train)
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
    hidden_layer_sizes = [(20,), (100,), (20, 20), (100, 100), (100, 100, 100)]
    learning_rate_inits = [1e-2, 1e-3, 1e-4] 
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
for params, models in regression_models.items():
    losses_val = []
    for model in models:
        predictions_val = model.predict(X_val_oco2)
        loss_val = math.sqrt(mean_squared_error(Y_val_oco2, predictions_val)) / average_sif        
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params
            best_model = model
        losses_val.append(loss_val)
    average_loss_val = sum(losses_val) / len(losses_val)
    print(params + ': avg val loss', round(average_loss_val, 4))


print('Best params:', best_params)
# print(best_model.coef_)

# Use the best model to make predictions
predictions_train = best_model.predict(X_train)
predictions_val_tropomi = best_model.predict(X_val_tropomi)
predictions_val_oco2 = best_model.predict(X_val_oco2)
predictions_test = best_model.predict(X_test)
predictions_cfis = best_model.predict(X_cfis)

predictions_train = np.clip(predictions_train, a_min=MIN_SIF, a_max=MAX_SIF)
predictions_val_tropomi = np.clip(predictions_val_tropomi, a_min=MIN_SIF, a_max=MAX_SIF)
predictions_val_oco2 = np.clip(predictions_val_oco2, a_min=MIN_SIF, a_max=MAX_SIF)
predictions_test = np.clip(predictions_test, a_min=MIN_SIF, a_max=MAX_SIF)
predictions_cfis = np.clip(predictions_cfis, a_min=MIN_SIF, a_max=MAX_SIF)

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


# Print NRMSE, correlation, R2 on train/validation set
print('============== Train set stats =====================')
print_stats(Y_train, predictions_train, average_sif)

print('============== Val set stats (TROPOMI) =====================')
print_stats(Y_val_tropomi, predictions_val_tropomi, average_sif)

print('============== Val set stats (OCO2) =====================')
print_stats(Y_val_oco2, predictions_val_oco2, average_sif)

print('============== Test set stats (OCO2) =====================')
print_stats(Y_test, predictions_test, average_sif, ax=plt.gca())
plt.xlim(left=0, right=MAX_SIF)
plt.ylim(bottom=0, top=MAX_SIF)
plt.title('OCO2 test set: true vs predicted SIF (' + METHOD + ')')
plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '.png')
plt.close()

# Print stats for CFIS subtiles
print('========== CFIS Eval subtile stats ===========')
print_stats(Y_cfis, predictions_cfis, average_sif, ax=plt.gca())  #eval_cfis_set['SIF'].mean())  #average_sif)
plt.title('CFIS: true vs predicted SIF (' + METHOD + ')')
plt.xlim(left=0, right=MAX_PRED)
plt.ylim(bottom=0, top=MAX_CFIS_SIF)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '.png')
plt.close()


# Scatter plot of true vs predicted on TROPOMI val tiles
# plt.scatter(Y_val_tropomi, predictions_val_tropomi)
# plt.xlabel('True')
# plt.ylabel('Predicted')
# plt.xlim(left=MIN_SIF, right=MAX_SIF)
# plt.ylim(bottom=MIN_SIF, top=MAX_SIF)
# plt.title('TROPOMI val set: predicted vs true SIF (' + METHOD + ')')
# plt.savefig('exploratory_plots/true_vs_predicted_sif_val_tropomi_' + METHOD + '.png')
# plt.close()



# Plot true vs. predicted for each crop on OCO-2 (for each crop)
fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('True vs predicted SIF (OCO-2) by crop: ' + METHOD)
for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    predicted = predictions_test[test_set[crop_type] > PURE_THRESHOLD]
    true = Y_test[test_set[crop_type] > PURE_THRESHOLD]
    ax = axeslist.ravel()[idx]
    print('======================= (OCO2) CROP: ', crop_type, '==============================')
    print(len(predicted), 'subtiles that are pure', crop_type)
    if len(predicted) >= 2:
        print(' ----- All crop regression ------')
        print_stats(true, predicted, average_sif, ax=ax)
        ax.set_xlim(left=0, right=MAX_SIF)
        ax.set_ylim(bottom=0, top=MAX_SIF)
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
        # print_stats(Y_oco2_crop, predicted_oco2_crop, average_sif)
 
    # Plot true vs. predicted for that specific crop
    # axeslist.ravel()[idx].scatter(true, predicted)
    # axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')

plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()

# # Scatter plot of true vs predicted, entire val set
# plt.scatter(Y_val, predictions_val)
# plt.xlabel('True')
# plt.ylabel('Predicted')
# plt.xlim(left=MIN_SIF, right=MAX_SIF)
# plt.ylim(bottom=MIN_SIF, top=MAX_SIF)
# plt.title('Val set: predicted vs true SIF (' + METHOD + ')')
# plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '.png')
# plt.close()

# # Scatter plot of true vs. predicted on CFIS (all crops combined)
# plt.scatter(Y_cfis, predictions_cfis)
# plt.xlabel('True')
# plt.ylabel('Predicted')
# # plt.xlim(left=MIN_SIF, right=MAX_SIF)
# # plt.ylim(bottom=MIN_SIF, top=MAX_SIF)
# plt.title('CFIS subtile set: predicted vs true SIF (' + METHOD + ')')
# plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '.png')
# plt.close()

# Plot true vs. predicted for each crop on CFIS (for each crop)
fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('True vs predicted SIF (CFIS): ' + METHOD)
for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    predicted = predictions_cfis[eval_cfis_set[crop_type] > PURE_THRESHOLD]  # Find CFIS tiles which are "purely" this crop type
    true = Y_cfis[eval_cfis_set[crop_type] > PURE_THRESHOLD]
    ax = axeslist.ravel()[idx]
    print('======================= (CFIS) CROP: ', crop_type, '==============================')
    print(len(predicted), 'subtiles that are pure', crop_type)
    if len(predicted) >= 2:
        print(' ----- All crop regression ------')
        print_stats(true, predicted, average_sif, ax=ax)
        ax.set_xlim(left=0, right=MAX_PRED)
        ax.set_ylim(bottom=0, top=MAX_CFIS_SIF)
        ax.set_title(crop_type)

        # # Fit linear model on just this crop, to see how strong the relationship is
        # X_train_crop = X_train.loc[X_train[crop_type] > 0.5]
        # Y_train_crop = Y_train[X_train[crop_type] > 0.5]
        # X_cfis_crop = X_cfis.loc[X_cfis[crop_type] > PURE_THRESHOLD]
        # Y_cfis_crop = Y_cfis[X_cfis[crop_type] > PURE_THRESHOLD]
        # crop_regression = LinearRegression().fit(X_cfis_crop, Y_cfis_crop)
        # predicted_cfis_crop = crop_regression.predict(X_cfis_crop)
        # print(' ----- CFIS Crop specific regression -----')
        # print('Coefficients:', crop_regression.coef_)
        # print_stats(Y_cfis_crop, predicted_cfis_crop, average_sif)
 
    # Plot true vs. predicted for that specific crop
    # axeslist.ravel()[idx].scatter(true, predicted)
    # axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    # axeslist.ravel()[idx].set_xlim(left=0, right=2)
    # axeslist.ravel()[idx].set_ylim(bottom=0, top=2)
    # axeslist.ravel()[idx].set_title(crop_type)

plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()

# Scatter plot of true vs. predicted on OCO-2 (all crops combined)
# plt.scatter(Y_oco2, predictions_oco2)
# plt.xlabel('True')
# plt.ylabel('Predicted')
# plt.xlim(left=MIN_SIF, right=MAX_SIF)
# plt.ylim(bottom=MIN_SIF, top=MAX_SIF)
# plt.title('True vs predicted SIF (OCO-2, n > 5 only): ' + METHOD)
# plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '.png')
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
fig, axeslist = plt.subplots(ncols=len(VAL_SOURCES), nrows=len(DATES), figsize=(6*len(VAL_SOURCES), 6*len(DATES)))
fig.suptitle('True vs predicted SIF, by date/source: ' + METHOD)
idx = 0
for date in DATES:
    for source in VAL_SOURCES:
        # Obtain global model's predictions for data points with this date/source 
        predicted = predictions_test[(test_set['date'] == date) & (test_set['source'] == source)]
        true = Y_test[(test_set['date'] == date) & (test_set['source'] == source)]
        print('=================== Date ' + date + ', ' + source + ' ======================')
        print('Number of rows', len(predicted))
        assert(len(predicted) == len(true))
        if len(predicted) < 2:
            idx += 1
            continue

        # Print stats (true vs predicted)
        ax = axeslist.ravel()[idx]
        print_stats(true, predicted, average_sif, ax=ax)

        ax.set_xlim(left=0, right=MAX_SIF)
        ax.set_ylim(bottom=0, top=MAX_SIF)
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
plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_dates.png')
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


