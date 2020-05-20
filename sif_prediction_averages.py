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
TILE_AVERAGE_TRAIN_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_averages_train.csv")
TILE_AVERAGE_VAL_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_averages_val.csv")
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")

EVAL_DATE = "2016-08-01" #"2016-07-16"
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + EVAL_DATE)
EVAL_SUBTILE_AVERAGE_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtile_averages.csv")
METHOD = "1c_Gradient_Boosting_Regressor" #"1b_Ridge_regression" #"Gradient_Boosting_Regressor"
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_eval_subtile_' + METHOD

# Read datasets
train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE).dropna()
val_set = pd.read_csv(TILE_AVERAGE_VAL_FILE).dropna()
eval_subtile_set = pd.read_csv(EVAL_SUBTILE_AVERAGE_FILE).dropna()
band_statistics = pd.read_csv(BAND_STATISTICS_FILE)
average_sif = band_statistics['mean'].iloc[-1]

print('Train samples:', len(train_set))
print('Val samples;', len(val_set))
print('average sif (train, large tiles)', average_sif)
print('average sif (val, large_tiles)', val_set['SIF'].mean())
print('average sif (eval subtiles)', eval_subtile_set['SIF'].mean())

# Columns to exclude from input
EXCLUDE_FROM_INPUT = ['date', 'tile_file', 'lat', 'lon', 'SIF']
INPUT_COLUMNS = list(train_set.columns.difference(EXCLUDE_FROM_INPUT))
print('input columns', INPUT_COLUMNS)
OUTPUT_COLUMN = ['SIF']

X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
X_val = val_set[INPUT_COLUMNS]
Y_val = val_set[OUTPUT_COLUMN].values.ravel()
X_eval_subtile = eval_subtile_set[INPUT_COLUMNS]
Y_eval_subtile = eval_subtile_set[OUTPUT_COLUMN].values.ravel() #1.52

# Print percentage of each crop type
print('In train set, average crop cover')
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']
for column_name in COVER_COLUMN_NAMES:
    print(column_name, round(np.mean(X_train[column_name]), 3))

#plot_histogram(Y_train, "train_large_tile_sif.png")
#plot_histogram(Y_val, "val_large_tile_sif.png")
#plot_histogram(Y_eval_subtile, "eval_subtile_sif.png")


# Fit model on band averages
linear_regression = GradientBoostingRegressor().fit(X_train, Y_train)
linear_predictions_train = linear_regression.predict(X_train)
linear_predictions_val = linear_regression.predict(X_val)
linear_predictions_eval_subtile = linear_regression.predict(X_eval_subtile)

#scale_factor = np.mean(linear_predictions_eval_subtile) / np.mean(Y_eval_subtile)
#print('Scale factor', scale_factor)
#Y_eval_subtile *= scale_factor
#print('Coef', linear_regression.coef_)

# Quantile analysis
#squared_errors = (Y_eval_subtile - linear_predictions_eval_subtile) ** 2
#indices = squared_errors.argsort() #Ascending order of squared error

#percentiles = [0, 0.05, 0.1, 0.2]
#for percentile in percentiles:
#    cutoff_idx = int((1 - percentile) * len(Y_eval_subtile))
#    indices_to_include = indices[:cutoff_idx]
#    nrmse = math.sqrt(np.mean(squared_errors[indices_to_include])) / average_sif
#    corr, _ = pearsonr(Y_eval_subtile[indices_to_include], linear_predictions_eval_subtile[indices_to_include])
#    print('Excluding ' + str(int(percentile*100)) + '% worst predictions')
#    print('NRMSE', round(nrmse, 3))
#    print('Corr', round(corr, 3))


# Print NRMSE, correlation, R2 on train/validation set
linear_nrmse_train = math.sqrt(mean_squared_error(linear_predictions_train, Y_train)) / average_sif
linear_nrmse_val = math.sqrt(mean_squared_error(linear_predictions_val, Y_val)) / average_sif
print(METHOD + ": train NRMSE", round(linear_nrmse_train, 3))
print(METHOD + ": val NRMSE", round(linear_nrmse_val, 3))
linear_corr_train, _ = pearsonr(Y_train, linear_predictions_train)
linear_corr_val, _ = pearsonr(Y_val, linear_predictions_val)
print("Train corr:", round(linear_corr_train, 3))
print("Val corr:", round(linear_corr_val, 3))
linear_r2_train = r2_score(Y_train, linear_predictions_train)
linear_r2_val = r2_score(Y_val, linear_predictions_val)
print("Train R2:", round(linear_r2_train, 3))
print("Val R2:", round(linear_r2_val, 3))

# Print stats for eval subtiles
print('========== Eval subtile stats ===========')
print_stats(Y_eval_subtile, linear_predictions_eval_subtile, average_sif)  #eval_subtile_set['SIF'].mean())  #average_sif)

# Scatter plot of true vs predicted
plt.scatter(Y_val, linear_predictions_val)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(left=0, right=2)
plt.ylim(bottom=0, top=2)
plt.title('Large tile val set: predicted vs true SIF (' + METHOD + ')')
plt.savefig('exploratory_plots/true_vs_predicted_sif_large_tile_' + METHOD + '.png')
plt.close()

# Scatter plot of true vs. predicted (all crops combined)
plt.scatter(Y_eval_subtile, linear_predictions_eval_subtile)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(left=0, right=2)
plt.ylim(bottom=0, top=2)
plt.title('Eval subtile set: predicted vs true SIF (' + METHOD + ')')
plt.savefig('exploratory_plots/true_vs_predicted_sif_eval_subtile_' + METHOD + '.png')
plt.close()

# Plot true vs. predicted for each crop
fig, axeslist = plt.subplots(ncols=3, nrows=10, figsize=(15, 50))
fig.suptitle('True vs predicted SIF (CFIS): ' + METHOD)
for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    predicted = linear_predictions_eval_subtile[eval_subtile_set[crop_type] > 0.5]
    true = Y_eval_subtile[eval_subtile_set[crop_type] > 0.5]
    print('======================= CROP: ', crop_type, '==============================')
    print(len(predicted), 'subtiles that are majority', crop_type)
    if len(predicted) >= 2:
        print(' ----- All crop regression ------')
        print_stats(true, predicted, average_sif)

        # Fit linear model on just this crop, to see how strong the relationship is
        X_crop = X_eval_subtile.loc[eval_subtile_set[crop_type] > 0.5]
        Y_crop = Y_eval_subtile[eval_subtile_set[crop_type] > 0.5]
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
plt.savefig(TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()


