import numpy as np
import os
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt
from sif_utils import plot_histogram

TRAIN_DATE = "2018-08-01"
TRAIN_DATASET_DIR = "datasets/dataset_" + TRAIN_DATE
TILE_AVERAGE_TRAIN_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_averages_train.csv")
TILE_AVERAGE_VAL_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_averages_val.csv")

EVAL_DATE = "2016-08-01"
EVAL_DATASET_DIR = "datasets/dataset_" + EVAL_DATE
EVAL_SUBTILE_AVERAGE_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtile_averages.csv")

train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE).dropna()
val_set = pd.read_csv(TILE_AVERAGE_VAL_FILE).dropna()
eval_subtile_set = pd.read_csv(EVAL_SUBTILE_AVERAGE_FILE).dropna()
average_sif = train_set['SIF'].mean()
print('Train samples:', len(train_set))
print('Val samples;', len(val_set))
print('average sif (train, large tiles)', average_sif)
print('average sif (val, large_tiles)', val_set['SIF'].mean())
print('average sif (eval subtiles)', eval_subtile_set['SIF'].mean())

INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7', 'ref_10',
                 'ref_11', 'corn', 'soybean', 'grassland', 'deciduous_forest', 'percent_missing']
OUTPUT_COLUMN = ['SIF']

X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
X_val = val_set[INPUT_COLUMNS]
Y_val = val_set[OUTPUT_COLUMN].values.ravel()
X_eval_subtile = eval_subtile_set[INPUT_COLUMNS]
Y_eval_subtile = eval_subtile_set[OUTPUT_COLUMN].values.ravel()

plot_histogram(Y_train, "train_large_tile_sif.png")
plot_histogram(Y_val, "val_large_tile_sif.png")
plot_histogram(Y_eval_subtile, "eval_subtile_sif.png")



linear_regression = LinearRegression().fit(X_train, Y_train)
linear_predictions_train = linear_regression.predict(X_train)
linear_predictions_val = linear_regression.predict(X_val)
linear_predictions_eval_subtile = linear_regression.predict(X_eval_subtile)

print('Linear Predictions val', linear_predictions_val)
print('True val', Y_val)

linear_nrmse_train = math.sqrt(mean_squared_error(linear_predictions_train, Y_train)) / average_sif
linear_nrmse_val = math.sqrt(mean_squared_error(linear_predictions_val, Y_val)) / average_sif
linear_nrmse_eval_subtile = math.sqrt(mean_squared_error(linear_predictions_eval_subtile, Y_eval_subtile)) / average_sif
print("Linear Regression: train NRMSE", linear_nrmse_train)
print("Linear Regression: val NRMSE", linear_nrmse_val)
print("Linear Regression: eval subtile NRMSE", linear_nrmse_eval_subtile)

linear_r2_train = r2_score(linear_predictions_train, Y_train)
linear_r2_val = r2_score(linear_predictions_val, Y_val)
linear_r2_eval_subtile = r2_score(linear_predictions_eval_subtile, Y_eval_subtile)
print("Train R2:", linear_r2_train)
print("Val R2:", linear_r2_val)
print("Eval_subtile R2", linear_r2_eval_subtile)

# Scatter plot of true vs predicted
plt.scatter(Y_val, linear_predictions_val)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Large tile val set: predicted vs true SIF')
plt.savefig('exploratory_plots/true_vs_predicted_sif_large_tile_val.png')
plt.close()

# Scatter plot of true vs predicted
plt.scatter(Y_eval_subtile, linear_predictions_eval_subtile)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Eval subtile set: predicted vs true SIF')
plt.savefig('exploratory_plots/true_vs_predicted_sif_eval_subtile.png')
plt.close()
