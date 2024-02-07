"""
Try to compute the optimal scaling factor between OCO-2 and CFIS, by training a model on each dataset,
computing each model's predictions on CFIS, and running a linear regression to find the best scaling
factor between the two model's predictions
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

# Set random seed
np.random.seed(0)

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
OCO2_DIR = os.path.join(DATA_DIR, "OCO2")
PLOTS_DIR = os.path.join(DATA_DIR, "exploratory_plots")

DATES = ["2016-06-15", "2016-08-01"]
MIN_OCO2_SOUNDINGS = 3
MAX_OCO2_CLOUD_COVER = 0.5
MIN_COARSE_FRACTION_VALID_PIXELS = 0.1
MIN_FINE_CFIS_SOUNDINGS = 10
MIN_INPUT = -3
MAX_INPUT = 3
MIN_SIF_CLIP = 0.1
MAX_SIF_CLIP = None
MIN_SIF_PLOT = 0
MAX_SIF_PLOT = 2

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


# Read datasets
cfis_fine_train_set = pd.read_csv(FINE_AVERAGES_TRAIN_FILE)
cfis_fine_val_set = pd.read_csv(FINE_AVERAGES_VAL_FILE)
cfis_fine_test_set = pd.read_csv(FINE_AVERAGES_TEST_FILE)
cfis_coarse_train_set = pd.read_csv(COARSE_AVERAGES_TRAIN_FILE)
cfis_coarse_val_set = pd.read_csv(COARSE_AVERAGES_VAL_FILE)
cfis_coarse_test_set = pd.read_csv(COARSE_AVERAGES_TEST_FILE)
oco2_train_set = pd.read_csv(OCO2_METADATA_TRAIN_FILE)
oco2_val_set = pd.read_csv(OCO2_METADATA_VAL_FILE)
oco2_test_set = pd.read_csv(OCO2_METADATA_TEST_FILE)

# Only include CFIS tiles with enough valid pixels
cfis_coarse_train_set = cfis_coarse_train_set[(cfis_coarse_train_set['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                              (cfis_coarse_train_set['SIF'] >= MIN_SIF_CLIP)]
cfis_coarse_val_set = cfis_coarse_val_set[(cfis_coarse_val_set['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                          (cfis_coarse_val_set['SIF'] >= MIN_SIF_CLIP)]
cfis_coarse_test_set = cfis_coarse_test_set[(cfis_coarse_test_set['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                            (cfis_coarse_test_set['SIF'] >= MIN_SIF_CLIP)]

# Filter OCO2 sets
oco2_train_set = oco2_train_set[(oco2_train_set['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                                (oco2_train_set['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                                (oco2_train_set['SIF'] >= MIN_SIF_CLIP)]
oco2_val_set = oco2_val_set[(oco2_val_set['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                            (oco2_val_set['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                            (oco2_val_set['SIF'] >= MIN_SIF_CLIP)]
oco2_test_set = oco2_test_set[(oco2_test_set['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                            (oco2_test_set['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                            (oco2_test_set['SIF'] >= MIN_SIF_CLIP)]

cfis_coarse_train_set = cfis_coarse_train_set.sample(frac=1).reset_index(drop=True)
oco2_train_set = oco2_train_set.sample(frac=1).reset_index(drop=True)
print('CFIS coarse train samples', len(cfis_coarse_train_set))
print('OCO2 train samples', len(oco2_train_set))

# Read band statistics
train_statistics = pd.read_csv(BAND_STATISTICS_CSV_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

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

# Crop types to look at when analyzing results
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest'] #, 'evergreen_forest', 'spring_wheat']

# Standardize data
for idx, column in enumerate(COLUMNS_TO_STANDARDIZE):
    cfis_coarse_train_set[column] = np.clip((cfis_coarse_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    cfis_coarse_val_set[column] = np.clip((cfis_coarse_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    cfis_coarse_test_set[column] = np.clip((cfis_coarse_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    cfis_fine_train_set[column] = np.clip((cfis_fine_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    cfis_fine_val_set[column] = np.clip((cfis_fine_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    cfis_fine_test_set[column] = np.clip((cfis_fine_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    oco2_train_set[column] = np.clip((oco2_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    oco2_val_set[column] = np.clip((oco2_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    oco2_test_set[column] = np.clip((oco2_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)

X_cfis_coarse_train = cfis_coarse_train_set[INPUT_COLUMNS]
Y_cfis_coarse_train = cfis_coarse_train_set[OUTPUT_COLUMN].values.ravel()
X_oco2_train = oco2_train_set[INPUT_COLUMNS]
Y_oco2_train = oco2_train_set[OUTPUT_COLUMN].values.ravel()
X_cfis_coarse_val = cfis_coarse_val_set[INPUT_COLUMNS]
Y_cfis_coarse_val = cfis_coarse_val_set[OUTPUT_COLUMN].values.ravel()


cfis_model = MLPRegressor().fit(X_cfis_coarse_train, Y_cfis_coarse_train)
oco2_model = MLPRegressor().fit(X_oco2_train, Y_oco2_train)
cfis_model_predictions = cfis_model.predict(X_cfis_coarse_val)
oco2_model_predictions = oco2_model.predict(X_cfis_coarse_val)

print('=================== CFIS train vs OCO-2 train predictions ================')
print_stats(cfis_model_predictions, oco2_model_predictions, sif_mean, ax=plt.gca(), fit_intercept=False)
plt.title('Predictions: model trained on CFIS vs model trained on OCO-2')
plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
plt.savefig(os.path.join(PLOTS_DIR, 'model_trained_on_cfis_vs_oco2.png'))
plt.close()
