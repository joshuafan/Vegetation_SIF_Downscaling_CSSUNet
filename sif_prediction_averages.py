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
TRAIN_DATE = "2018-08-01" # "2018-07-16"
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + TRAIN_DATE)
TILE_AVERAGE_TRAIN_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_info_train.csv")
TILE_AVERAGE_VAL_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_info_val.csv")
OCO2_AVERAGE_FILE = os.path.join(TRAIN_DATASET_DIR, "oco2_eval_subtiles.csv")
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")

EVAL_DATE = "2016-08-01" #"2016-07-16"
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + EVAL_DATE)
EVAL_SUBTILE_AVERAGE_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtile_averages.csv")
METHOD = "1a_Linear_Regression" #"1b_Ridge_regression" #"Gradient_Boosting_Regressor"
CFIS_TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_eval_subtile_' + METHOD
OCO2_TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_OCO2_' + METHOD

# Read datasets
train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE).dropna()
val_set = pd.read_csv(TILE_AVERAGE_VAL_FILE).dropna()
eval_oco2_set = pd.read_csv(OCO2_AVERAGE_FILE).dropna()
eval_cfis_set = pd.read_csv(EVAL_SUBTILE_AVERAGE_FILE).dropna()
band_statistics = pd.read_csv(BAND_STATISTICS_FILE)
average_sif = band_statistics['mean'].iloc[-1]

print('Train samples:', len(train_set))
print('Val samples;', len(val_set))
print('average sif (train, according to band statistics file)', average_sif)
print('average sif (train, large tiles)', train_set['SIF'].mean())
print('average sif (val, large_tiles)', val_set['SIF'].mean())
print('average sif (OCO2)', eval_oco2_set['SIF'].mean())
print('average sif (eval subtiles)', eval_cfis_set['SIF'].mean())

# Columns to exclude from input
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
OUTPUT_COLUMN = ['SIF']
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']
X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
X_val = val_set[INPUT_COLUMNS]
Y_val = val_set[OUTPUT_COLUMN].values.ravel()
X_oco2 = eval_oco2_set[INPUT_COLUMNS]
Y_oco2 = eval_oco2_set[OUTPUT_COLUMN].values.ravel()
X_cfis = eval_cfis_set[INPUT_COLUMNS]
Y_cfis = eval_cfis_set[OUTPUT_COLUMN].values.ravel()

# Print percentage of each crop type
print('Train set: feature averages')
for column_name in INPUT_COLUMNS:
    print(column_name, round(np.mean(X_train[column_name]), 3))

#plot_histogram(Y_train, "train_large_tile_sif.png")
#plot_histogram(Y_val, "val_large_tile_sif.png")
#plot_histogram(Y_cfis, "eval_subtile_sif.png")


# Fit model on band averages
regression_model = LinearRegression().fit(X_train, Y_train)
predictions_train = regression_model.predict(X_train)
predictions_val = regression_model.predict(X_val)
predictions_oco2 = regression_model.predict(X_oco2)
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
nrmse_train = math.sqrt(mean_squared_error(predictions_train, Y_train)) / average_sif
nrmse_val = math.sqrt(mean_squared_error(predictions_val, Y_val)) / average_sif
print(METHOD + ": train NRMSE", round(nrmse_train, 3))
print(METHOD + ": val NRMSE", round(nrmse_val, 3))
corr_train, _ = pearsonr(Y_train, predictions_train)
corr_val, _ = pearsonr(Y_val, predictions_val)
print("Train corr:", round(corr_train, 3))
print("Val corr:", round(corr_val, 3))
r2_train = r2_score(Y_train, predictions_train)
r2_val = r2_score(Y_val, predictions_val)
print("Train R2:", round(r2_train, 3))
print("Val R2:", round(r2_val, 3))

# Print stats for eval subtiles
print('========== CFIS Eval subtile stats ===========')
print_stats(Y_cfis, predictions_cfis, average_sif)  #eval_cfis_set['SIF'].mean())  #average_sif)

print('========== OCO-2 Eval subtile stats ===========')
print_stats(Y_oco2, predictions_oco2, average_sif)  #eval_cfis_set['SIF'].mean())  #average_sif)

# Scatter plot of true vs predicted
plt.scatter(Y_val, predictions_val)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(left=0, right=2)
plt.ylim(bottom=0, top=2)
plt.title('Large tile val set: predicted vs true SIF (' + METHOD + ')')
plt.savefig('exploratory_plots/true_vs_predicted_sif_large_tile_' + METHOD + '.png')
plt.close()

# Scatter plot of true vs. predicted on CFIS (all crops combined)
plt.scatter(Y_cfis, predictions_cfis)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(left=0, right=2)
plt.ylim(bottom=0, top=2)
plt.title('CFIS subtile set: predicted vs true SIF (' + METHOD + ')')
plt.savefig('exploratory_plots/true_vs_predicted_sif_eval_subtile_' + METHOD + '.png')
plt.close()

# Plot true vs. predicted for each crop on CFIS (for each crop)
fig, axeslist = plt.subplots(ncols=3, nrows=10, figsize=(15, 50))
fig.suptitle('True vs predicted SIF (CFIS): ' + METHOD)
for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    predicted = predictions_cfis[eval_cfis_set[crop_type] > 0.7]
    true = Y_cfis[eval_cfis_set[crop_type] > 0.7]
    print('======================= CROP: ', crop_type, '==============================')
    print(len(predicted), 'subtiles that are pure', crop_type)
    if len(predicted) >= 2:
        print(' ----- All crop regression ------')
        print_stats(true, predicted, average_sif)

        # Fit linear model on just this crop, to see how strong the relationship is
        X_crop = X_cfis.loc[eval_cfis_set[crop_type] > 0.5]
        Y_crop = Y_cfis[eval_cfis_set[crop_type] > 0.5]
        crop_regression = LinearRegression().fit(X_crop, Y_crop)
        predicted_crop = crop_regression.predict(X_crop)
        print(' ----- Crop specific regression -----')
        print_stats(Y_crop, predicted_crop, average_sif)
 
    # Plot true vs. predicted for that specific crop
    axeslist.ravel()[idx].scatter(true, predicted)
    axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    axeslist.ravel()[idx].set_xlim(left=0, right=2)
    axeslist.ravel()[idx].set_ylim(bottom=0, top=2)
    axeslist.ravel()[idx].set_title(crop_type)
plt.tight_layout()
fig.subplots_adjust(top=0.96)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()

# Scatter plot of true vs. predicted on OCO-2 (all crops combined)
plt.scatter(Y_oco2, predictions_oco2)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(left=0, right=2)
plt.ylim(bottom=0, top=2)
plt.title('OCO-2 set: predicted vs true SIF (' + METHOD + ')')
plt.savefig('exploratory_plots/true_vs_predicted_sif_OCO2_' + METHOD + '.png')
plt.close()

# Plot true vs. predicted for each crop on OCO-2 (for each crop)
fig, axeslist = plt.subplots(ncols=3, nrows=10, figsize=(15, 50))
fig.suptitle('True vs predicted SIF (OCO-2): ' + METHOD)
for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    predicted = predictions_oco2[eval_oco2_set[crop_type] > 0.7]
    true = Y_oco2[eval_oco2_set[crop_type] > 0.7]
    print('======================= CROP: ', crop_type, '==============================')
    print(len(predicted), 'subtiles that are pure', crop_type)
    if len(predicted) >= 2:
        print(' ----- All crop regression ------')
        print_stats(true, predicted, average_sif)

        # Fit linear model on just this crop, to see how strong the relationship is
        X_crop = X_oco2.loc[eval_oco2_set[crop_type] > 0.5]
        Y_crop = Y_oco2[eval_oco2_set[crop_type] > 0.5]
        crop_regression = LinearRegression().fit(X_crop, Y_crop)
        predicted_crop = crop_regression.predict(X_crop)
        print(' ----- Crop specific regression -----')
        print_stats(Y_crop, predicted_crop, average_sif)
 
    # Plot true vs. predicted for that specific crop
    axeslist.ravel()[idx].scatter(true, predicted)
    axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    axeslist.ravel()[idx].set_xlim(left=0, right=2)
    axeslist.ravel()[idx].set_ylim(bottom=0, top=2)
    axeslist.ravel()[idx].set_title(crop_type)
plt.tight_layout()
fig.subplots_adjust(top=0.96)
plt.savefig(OCO2_TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()


