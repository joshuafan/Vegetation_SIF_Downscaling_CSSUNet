import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt
from sif_utils import plot_histogram

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
TRAIN_DATE = "2018-08-01"
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + TRAIN_DATE)
TILE_AVERAGE_TRAIN_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_averages_train.csv")
TILE_AVERAGE_VAL_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_averages_val.csv")

EVAL_DATE = "2016-08-01"
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + EVAL_DATE)
EVAL_SUBTILE_AVERAGE_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtile_averages.csv")

METHOD = "Gradient_Boosting_Regressor"

train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE).dropna()
val_set = pd.read_csv(TILE_AVERAGE_VAL_FILE).dropna()
eval_subtile_set = pd.read_csv(EVAL_SUBTILE_AVERAGE_FILE).dropna()
average_sif = train_set['SIF'].mean()

print('Train samples:', len(train_set))
print('Val samples;', len(val_set))
print('average sif (train, large tiles)', average_sif)
print('average sif (val, large_tiles)', val_set['SIF'].mean())
print('average sif (eval subtiles)', eval_subtile_set['SIF'].mean())

# Columns to exclude from input
EXCLUDE_FROM_INPUT = ['lat', 'lon', 'SIF']
INPUT_COLUMNS = list(train_set.columns.difference(EXCLUDE_FROM_INPUT))
print('input columns', INPUT_COLUMNS)
OUTPUT_COLUMN = ['SIF']

X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
X_val = val_set[INPUT_COLUMNS]
Y_val = val_set[OUTPUT_COLUMN].values.ravel()
X_eval_subtile = eval_subtile_set[INPUT_COLUMNS]
Y_eval_subtile = eval_subtile_set[OUTPUT_COLUMN].values.ravel()

#plot_histogram(Y_train, "train_large_tile_sif.png")
#plot_histogram(Y_val, "val_large_tile_sif.png")
#plot_histogram(Y_eval_subtile, "eval_subtile_sif.png")



linear_regression = GradientBoostingRegressor().fit(X_train, Y_train)
linear_predictions_train = linear_regression.predict(X_train)
linear_predictions_val = linear_regression.predict(X_val)
linear_predictions_eval_subtile = linear_regression.predict(X_eval_subtile)

print('Predicted val', linear_predictions_val)
print('True val', Y_val)

linear_nrmse_train = math.sqrt(mean_squared_error(linear_predictions_train, Y_train)) / average_sif
linear_nrmse_val = math.sqrt(mean_squared_error(linear_predictions_val, Y_val)) / average_sif
linear_nrmse_eval_subtile = math.sqrt(mean_squared_error(linear_predictions_eval_subtile, Y_eval_subtile)) / average_sif
print(METHOD + ": train NRMSE", round(linear_nrmse_train, 3))
print(METHOD + ": val NRMSE", round(linear_nrmse_val, 3))
print(METHOD + ": eval subtile NRMSE", round(linear_nrmse_eval_subtile, 3))

linear_corr_train, _ = pearsonr(linear_predictions_train, Y_train)
linear_corr_val, _ = pearsonr(linear_predictions_val, Y_val)
linear_corr_eval_subtile, _ = pearsonr(linear_predictions_eval_subtile, Y_eval_subtile)
print("Train corr:", round(linear_corr_train, 3))
print("Val corr:", round(linear_corr_val, 3))
print("Eval_subtile corr", round(linear_corr_eval_subtile, 3))

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
