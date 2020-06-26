"""
Runs pre-built ML methods over the channel averages of each tile (e.g. linear regression or gradient boosted tree)
"""
import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt
from sif_utils import plot_histogram, print_stats

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"

# Train files
# TRAIN_DATE = "2018-08-01" # "2018-07-16"
# TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + TRAIN_DATE)
PROCESSED_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset")
TILE_AVERAGE_TRAIN_FILE = os.path.join(PROCESSED_DATASET_DIR, "tile_info_train.csv")
TILE_AVERAGE_VAL_FILE = os.path.join(PROCESSED_DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(PROCESSED_DATASET_DIR, "band_statistics_train.csv")

DATES = ["2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02"]
SOURCES = ["TROPOMI", "OCO2"]

# OCO-2 eval file
# OCO2_AVERAGE_FILE = os.path.join(TRAIN_DATASET_DIR, "oco2_eval_subtiles.csv")
# print("OCO2 file", OCO2_AVERAGE_FILE)

# CFIS eval files
EVAL_DATE = "2016-08-01" #"2016-07-16"
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + EVAL_DATE)
CFIS_AVERAGE_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv")

METHOD = "1a_Linear_Regression"
#METHOD = "1d_MLP" #"1c_Gradient_Boosting_Regressor" # #"Gradient_Boosting_Regressor"
CFIS_TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_new_data_eval_subtile_' + METHOD
OCO2_TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_new_data_OCO2_' + METHOD

PURE_THRESHOLD = 0.6
MIN_SOUNDINGS = 3

# Read datasets
train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE)
val_set = pd.read_csv(TILE_AVERAGE_VAL_FILE)
eval_cfis_set = pd.read_csv(CFIS_AVERAGE_FILE)

# Split val set into TROPOMI and OCO2
val_tropomi_set = val_set[val_set['source'] == 'TROPOMI'].copy()
val_oco2_set = val_set[val_set['source'] == 'OCO2'].copy()

# Read band statistics
band_statistics = pd.read_csv(BAND_STATISTICS_FILE)
average_sif = band_statistics['mean'].iloc[-1]
band_means = band_statistics['mean'].values[:-1]
band_stds = band_statistics['std'].values[:-1]

# Print dataset info
print('Train samples:', len(train_set))
print('Val samples:', len(val_set))
print('OCO2 samples:', len(val_oco2_set))
print('average sif (train, according to band statistics file)', average_sif)
print('average sif (train, large tiles)', train_set['SIF'].mean())
print('average sif (val, TROPOMI)', val_tropomi_set['SIF'].mean())
print('average sif (val, OCO2)', val_oco2_set['SIF'].mean())
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
    train_set[column] = np.clip((train_set[column] - column_mean) / column_std, a_min=-3, a_max=3)
    val_set[column] = np.clip((val_set[column] - column_mean) / column_std, a_min=-3, a_max=3)
    val_tropomi_set[column] = np.clip((val_tropomi_set.loc[:, column] - column_mean) / column_std, a_min=-3, a_max=3)
    val_oco2_set[column] = np.clip((val_oco2_set.loc[:, column] - column_mean) / column_std, a_min=-3, a_max=3)
    eval_cfis_set[column] = np.clip((eval_cfis_set[column] - column_mean) / column_std, a_min=-3, a_max=3)
    # Histograms of standardized columns
    # plot_histogram(train_set[column].to_numpy(), "histogram_std_" + column + "_train.png")
    # plot_histogram(val_tropomi_set[column].to_numpy(), "histogram_std_" + column + "_val_tropomi.png")
    # plot_histogram(val_oco2_set[column].to_numpy(), "histogram_std_" + column + "_val_oco2.png")

    # train_set[column] = train_set[column] + np.random.normal(loc=0, scale=0.1, size=len(train_set[column]))
    # plot_histogram(eval_cfis_set[column].to_numpy(), "histogram_std_" + column + "_cfis.png")

print('train_set', train_set[INPUT_COLUMNS].head())
print('Val OCO2', val_oco2_set[INPUT_COLUMNS].head())
X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
X_val = val_set[INPUT_COLUMNS]
Y_val = val_set[OUTPUT_COLUMN].values.ravel()
X_val_tropomi = val_tropomi_set[INPUT_COLUMNS]
Y_val_tropomi = val_tropomi_set[OUTPUT_COLUMN].values.ravel()
X_val_oco2 = val_oco2_set[INPUT_COLUMNS]
Y_val_oco2 = val_oco2_set[OUTPUT_COLUMN].values.ravel()
X_cfis = eval_cfis_set[INPUT_COLUMNS]
Y_cfis = eval_cfis_set[OUTPUT_COLUMN].values.ravel()

# Print percentage of each crop type
print("************************************************************")
print('FEATURE AVERAGES')
for column_name in INPUT_COLUMNS:
    print('================== Feature:', column_name, '=================')
    print('Train:', round(np.mean(X_train[column_name]), 3))
    print('Val TROPOMI:', round(np.mean(X_val_tropomi[column_name]), 3))
    print('Val OCO2:', round(np.mean(X_val_oco2[column_name]), 3))
    print('CFIS:', round(np.mean(X_cfis[column_name]), 3))
print("************************************************************")

# Fit model on band averages
if METHOD == "1a_Linear_Regression":
    regression_model = LinearRegression().fit(X_train, Y_train)
elif METHOD == "1b_Ridge_Regression":
    regression_model = Ridge().fit(X_train, Y_train)
elif METHOD == "1c_Gradient_Boosting_Regressor":
    regression_model = GradientBoostingRegressor().fit(X_train, Y_train)
elif METHOD == "1d_MLP":
    regression_model = MLPRegressor(hidden_layer_sizes=(100, 100)).fit(X_train, Y_train)
else:
    print("Unsupported method")
    exit(1)

# print('Coefficients', regression_model.coef_)

predictions_train = regression_model.predict(X_train)
predictions_val = regression_model.predict(X_val)
predictions_val_tropomi = regression_model.predict(X_val_tropomi)
predictions_val_oco2 = regression_model.predict(X_val_oco2)
predictions_cfis = regression_model.predict(X_cfis)

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

print('============== Val set stats (all) =====================')
print_stats(Y_val, predictions_val, average_sif)

print('============== Val set stats (TROPOMI only) =====================')
print_stats(Y_val_tropomi, predictions_val_tropomi, average_sif)

print('============== Val set stats (OCO2 only) =====================')
print_stats(Y_val_oco2, predictions_val_oco2, average_sif)

# Print stats for CFIS subtiles
print('========== CFIS Eval subtile stats ===========')
print_stats(Y_cfis, predictions_cfis, average_sif)  #eval_cfis_set['SIF'].mean())  #average_sif)

# nrmse_train = math.sqrt(mean_squared_error(predictions_train, Y_train)) / average_sif
# nrmse_val = math.sqrt(mean_squared_error(predictions_val, Y_val)) / average_sif
# print(METHOD + ": train NRMSE", round(nrmse_train, 3))
# print(METHOD + ": val NRMSE", round(nrmse_val, 3))
# corr_train, _ = pearsonr(Y_train, predictions_train)
# corr_val, _ = pearsonr(Y_val, predictions_val)
# print("Train corr:", round(corr_train, 3))
# print("Val corr:", round(corr_val, 3))
# r2_train = r2_score(Y_train, predictions_train)
# r2_val = r2_score(Y_val, predictions_val)
# print("Train R2:", round(r2_train, 3))
# print("Val R2:", round(r2_val, 3))

# Scatter plot of true vs predicted on TROPOMI val tiles
plt.scatter(Y_val_tropomi, predictions_val_tropomi)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(left=0, right=1.5)
plt.ylim(bottom=0, top=1.5)
plt.title('TROPOMI val set: predicted vs true SIF (' + METHOD + ')')
plt.savefig('exploratory_plots/true_vs_predicted_sif_val_tropomi_' + METHOD + '.png')
plt.close()

# Scatter plot of true vs predicted on OCO2 val tiles
plt.scatter(Y_val_oco2, predictions_val_oco2)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(left=0, right=1.5)
plt.ylim(bottom=0, top=1.5)
plt.title('OCO2 val set: predicted vs true SIF (' + METHOD + ')')
plt.savefig('exploratory_plots/true_vs_predicted_sif_val_oco2_' + METHOD + '.png')
plt.close()

# # Scatter plot of true vs. predicted on CFIS (all crops combined)
# plt.scatter(Y_cfis, predictions_cfis)
# plt.xlabel('True')
# plt.ylabel('Predicted')
# plt.xlim(left=0, right=1.5)
# plt.ylim(bottom=0, top=1.5)
# plt.title('CFIS subtile set: predicted vs true SIF (' + METHOD + ')')
# plt.savefig('exploratory_plots/true_vs_predicted_sif_eval_subtile_' + METHOD + '.png')
# plt.close()

# # Plot true vs. predicted for each crop on CFIS (for each crop)
# fig, axeslist = plt.subplots(ncols=3, nrows=10, figsize=(15, 50))
# fig.suptitle('True vs predicted SIF (CFIS): ' + METHOD)
# for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
#     predicted = predictions_cfis[eval_cfis_set[crop_type] > PURE_THRESHOLD]  # Find CFIS tiles which are "purely" this crop type
#     true = Y_cfis[eval_cfis_set[crop_type] > PURE_THRESHOLD]
#     print('======================= CROP: ', crop_type, '==============================')
#     print(len(predicted), 'subtiles that are pure', crop_type)
#     if len(predicted) >= 2:
#         print(' ----- All crop regression ------')
#         print_stats(true, predicted, average_sif)

#         # Fit linear model on just this crop, to see how strong the relationship is
#         X_train_crop = X_train.loc[X_train[crop_type] > 0.5]
#         Y_train_crop = Y_train[X_train[crop_type] > 0.5]
#         X_cfis_crop = X_cfis.loc[X_cfis[crop_type] > PURE_THRESHOLD]
#         Y_cfis_crop = Y_cfis[X_cfis[crop_type] > PURE_THRESHOLD]
#         crop_regression = LinearRegression().fit(X_cfis_crop, Y_cfis_crop)
#         predicted_cfis_crop = crop_regression.predict(X_cfis_crop)
#         print(' ----- CFIS Crop specific regression -----')
#         print('Coefficients:', crop_regression.coef_)
#         print_stats(Y_cfis_crop, predicted_cfis_crop, average_sif)
 
#     # Plot true vs. predicted for that specific crop
#     axeslist.ravel()[idx].scatter(true, predicted)
#     axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
#     axeslist.ravel()[idx].set_xlim(left=0, right=2)
#     axeslist.ravel()[idx].set_ylim(bottom=0, top=2)
#     axeslist.ravel()[idx].set_title(crop_type)
# plt.tight_layout()
# fig.subplots_adjust(top=0.96)
# plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
# plt.close()

# Scatter plot of true vs. predicted on OCO-2 (all crops combined)
# plt.scatter(Y_oco2, predictions_oco2)
# plt.xlabel('True')
# plt.ylabel('Predicted')
# plt.xlim(left=0, right=1.5)
# plt.ylim(bottom=0, top=1.5)
# plt.title('True vs predicted SIF (OCO-2, n > 5 only): ' + METHOD)
# plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '.png')
# plt.close()

# Plot true vs. predicted for each crop on OCO-2 (for each crop)
fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('True vs predicted SIF (OCO-2) by crop: ' + METHOD)
for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    predicted = predictions_val_oco2[val_oco2_set[crop_type] > PURE_THRESHOLD]
    true = Y_val_oco2[val_oco2_set[crop_type] > PURE_THRESHOLD]
    print('======================= CROP: ', crop_type, '==============================')
    print(len(predicted), 'subtiles that are pure', crop_type)
    if len(predicted) >= 2:
        print(' ----- All crop regression ------')
        print_stats(true, predicted, average_sif)

        # Fit linear model on just this crop, to see how strong the relationship is
        X_train_crop = X_train.loc[train_set[crop_type] > PURE_THRESHOLD]
        Y_train_crop = Y_train[train_set[crop_type] > PURE_THRESHOLD]
        X_oco2_crop = X_val_oco2.loc[val_oco2_set[crop_type] > PURE_THRESHOLD]
        Y_oco2_crop = Y_val_oco2[val_oco2_set[crop_type] > PURE_THRESHOLD]
        crop_regression = LinearRegression().fit(X_train_crop, Y_train_crop)
        predicted_oco2_crop = crop_regression.predict(X_oco2_crop)
        print(' ----- Crop specific regression -----')
        #print('Coefficients:', crop_regression.coef_)
        print_stats(Y_oco2_crop, predicted_oco2_crop, average_sif)
 
    # Plot true vs. predicted for that specific crop
    axeslist.ravel()[idx].scatter(true, predicted)
    axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    axeslist.ravel()[idx].set_xlim(left=0, right=1.5)
    axeslist.ravel()[idx].set_ylim(bottom=0, top=1.5)
    axeslist.ravel()[idx].set_title(crop_type)
plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()

# Print summary statistics for each source
fig, axeslist = plt.subplots(ncols=len(SOURCES), nrows=1, figsize=(12, 6))
fig.suptitle('True vs predicted SIF, by source: ' + METHOD)

for idx, source in enumerate(SOURCES):
    # Obtain global model's predictions for data points with this source 
    print('=================== ALL DATES: ' + source + ' ======================')
    predicted = predictions_val[val_set['source'] == source]
    true = Y_val[val_set['source'] == source]
    print('Number of rows', len(predicted))
    if len(predicted) < 2:
        idx += 1
        continue

    print_stats(true, predicted, average_sif)

    # Scatter plot of true vs predicted
    axeslist.ravel()[idx].scatter(true, predicted)
    axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    axeslist.ravel()[idx].set_xlim(left=0, right=1.5)
    axeslist.ravel()[idx].set_ylim(bottom=0, top=1.5)
    axeslist.ravel()[idx].set_title(source)

    # Fit linear model on just this source
    X_train_source = X_train.loc[train_set['source'] == source]
    Y_train_source = Y_train[train_set['source'] == source]
    X_val_source = X_val.loc[val_set['source'] == source]
    Y_val_source = Y_val[val_set['source'] == source]
    source_regression = LinearRegression().fit(X_train_source, Y_train_source)
    predicted_val_source = source_regression.predict(X_val_source)
    print(' ----- Source-specific regression -----')
    print('Coefficients:', source_regression.coef_)
    print_stats(Y_val_source, predicted_val_source, average_sif)

plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_sources.png')
plt.close()

# Print statistics and plot by date and source
fig, axeslist = plt.subplots(ncols=len(SOURCES), nrows=len(DATES), figsize=(12, 30))
fig.suptitle('True vs predicted SIF, by date/source: ' + METHOD)
idx = 0
for date in DATES:
    for source in SOURCES:
        # Obtain global model's predictions for data points with this date/source 
        print('=================== Date ' + date + ', ' + source + ' ======================')
        predicted = predictions_val[(val_set['date'] == date) & (val_set['source'] == source)]
        true = Y_val[(val_set['date'] == date) & (val_set['source'] == source)]
        print('Number of rows', len(predicted))
        assert(len(predicted) == len(true))
        if len(predicted) < 2:
            idx += 1
            continue

        print_stats(true, predicted, average_sif)

        # Scatter plot of true vs predicted
        axeslist.ravel()[idx].scatter(true, predicted)
        axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
        axeslist.ravel()[idx].set_xlim(left=0, right=1.5)
        axeslist.ravel()[idx].set_ylim(bottom=0, top=1.5)
        axeslist.ravel()[idx].set_title(date + ', ' + source)
        idx += 1

        # Fit linear model on just this date
        X_train_date = X_train.loc[(train_set['date'] == date) & (train_set['source'] == source)]
        Y_train_date = Y_train[(train_set['date'] == date) & (train_set['source'] == source)]
        X_val_date = X_val.loc[(val_set['date'] == date) & (val_set['source'] == source)]
        Y_val_date = Y_val[(val_set['date'] == date) & (val_set['source'] == source)]
        date_regression = LinearRegression().fit(X_train_date, Y_train_date)
        predicted_val_date = date_regression.predict(X_val_date)
        print(' ----- Date and source-specific regression -----')
        print('Coefficients:', date_regression.coef_)
        print_stats(Y_val_date, predicted_val_date, average_sif)
plt.tight_layout()
fig.subplots_adjust(top=0.96)
plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_dates_and_sources.png')
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


