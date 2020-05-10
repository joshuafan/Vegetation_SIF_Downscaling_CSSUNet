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
TRAIN_DATE = "2018-07-16"
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + TRAIN_DATE)
TILE_AVERAGE_TRAIN_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_averages_train.csv")
TILE_AVERAGE_VAL_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_averages_val.csv")
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")

EVAL_DATE = "2016-07-16"
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + EVAL_DATE)
EVAL_SUBTILE_AVERAGE_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtile_averages.csv")
METHOD = "Gradient_Boosting_Regressor"

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

# Plot histogram of each band
print('In train set, average crop cover')
column_names = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']
for column_name in column_names:
    print(column_name, round(np.mean(X_eval_subtile[column_name]), 3))


#plot_histogram(Y_train, "train_large_tile_sif.png")
#plot_histogram(Y_val, "val_large_tile_sif.png")
#plot_histogram(Y_eval_subtile, "eval_subtile_sif.png")



linear_regression = GradientBoostingRegressor().fit(X_train, Y_train)  #X_train, Y_train)
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


linear_nrmse_train = math.sqrt(mean_squared_error(linear_predictions_train, Y_train)) / average_sif
linear_nrmse_val = math.sqrt(mean_squared_error(linear_predictions_val, Y_val)) / average_sif
#linear_nrmse_eval_subtile = math.sqrt(mean_squared_error(linear_predictions_eval_subtile, Y_eval_subtile)) / average_sif
print(METHOD + ": train NRMSE", round(linear_nrmse_train, 3))
print(METHOD + ": val NRMSE", round(linear_nrmse_val, 3))
#print(METHOD + ": eval subtile NRMSE", round(linear_nrmse_eval_subtile, 3))

linear_corr_train, _ = pearsonr(Y_train, linear_predictions_train)
linear_corr_val, _ = pearsonr(Y_val, linear_predictions_val)
#linear_corr_eval_subtile, _ = pearsonr(Y_eval_subtile, linear_predictions_eval_subtile)
print("Train corr:", round(linear_corr_train, 3))
print("Val corr:", round(linear_corr_val, 3))
#print("Eval_subtile corr:", round(linear_corr_eval_subtile, 3))

linear_r2_train = r2_score(Y_train, linear_predictions_train)
linear_r2_val = r2_score(Y_val, linear_predictions_val)
#linear_r2_eval_subtile = r2_score(Y_eval_subtile, linear_predictions_eval_subtile)
print("Train R2:", round(linear_r2_train, 3))
print("Val R2:", round(linear_r2_val, 3))
#print("Eval_subtile R2:", round(linear_r2_eval_subtile, 3))

print('========== Eval subtile stats ===========')
print_stats(Y_eval_subtile, linear_predictions_eval_subtile, average_sif)  #eval_subtile_set['SIF'].mean())  #average_sif)

# Scatter plot of true vs predicted
plt.scatter(Y_val, linear_predictions_val)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Large tile val set: predicted vs true SIF (' + METHOD + ')')
plt.savefig('exploratory_plots/true_vs_predicted_sif_large_tile_' + METHOD + '.png')
plt.close()

# Scatter plot of true vs predicted
plt.scatter(Y_eval_subtile, linear_predictions_eval_subtile)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Eval subtile set: predicted vs true SIF (' + METHOD + ')')
plt.savefig('exploratory_plots/true_vs_predicted_sif_eval_subtile_' + METHOD + '.png')
plt.close()
