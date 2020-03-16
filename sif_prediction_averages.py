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
from sif_utils import plot_histogram

DATASET_DIR = "datasets/dataset_2018-08-01"
TILE_AVERAGE_TRAIN_FILE = os.path.join(DATASET_DIR, "tile_averages_train.csv")
TILE_AVERAGE_VAL_FILE = os.path.join(DATASET_DIR, "tile_averages_val.csv")

 #"datasets/generated_subtiles/eval_subtile_averages.csv" #
train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE).dropna()
val_set = pd.read_csv(TILE_AVERAGE_VAL_FILE).dropna()
average_sif = train_set['SIF'].mean()
print('Train samples:', len(train_set))
print('Val samples;', len(val_set))
print('average sif', average_sif)

INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7', 'ref_10',
                 'ref_11', 'corn', 'soybean', 'grassland', 'deciduous_forest', 'percent_missing']
OUTPUT_COLUMN = ['SIF']

X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
X_val = val_set[INPUT_COLUMNS]
Y_val = val_set[OUTPUT_COLUMN].values.ravel()


plot_histogram(Y_val, "val_large_tile_sif.png")

linear_regression = Ridge().fit(X_train, Y_train)
linear_predictions_train = linear_regression.predict(X_train)
linear_predictions_val = linear_regression.predict(X_val)
print('Linear Predictions val', linear_predictions_val)
print('True val', Y_val)
linear_nrmse_train = math.sqrt(mean_squared_error(linear_predictions_train, Y_train)) / average_sif
linear_nrmse_val = math.sqrt(mean_squared_error(linear_predictions_val, Y_val)) / average_sif
print("Linear Regression: train NRMSE", linear_nrmse_train)
print("Linear Regression: val NRMSE", linear_nrmse_val)

