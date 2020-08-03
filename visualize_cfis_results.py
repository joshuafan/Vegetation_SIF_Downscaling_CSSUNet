import math
from matplotlib import path
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
import torch
import torchvision.transforms as transforms
import xarray as xr
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from numpy import poly1d

from sif_utils import lat_long_to_index, plot_histogram, get_subtiles_list, print_stats
import sif_utils
import cdl_utils
import simple_cnn
import tile_transforms
import resnet
from SAN import SAN
from unet.unet_model import UNet, UNetSmall

import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet
from embedding_to_sif_model import EmbeddingToSIFModel
from embedding_to_sif_nonlinear_model import EmbeddingToSIFNonlinearModel


# Taken from https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plot_figures(output_file, figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    #fig = plt.figure(figsize=(8, 20))
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 20))
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title])  #, cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.savefig(output_file)
    plt.close()

def plot_images(image_rows, image_filename_column, output_file):
    images = {}
    for idx, image_row in image_rows.iterrows():
        subtile = np.load(image_row[image_filename_column]).transpose((1, 2, 0))
        title = 'Lat' + str(round(image_row['lat'], 6)) + ', Lon' + str(round(image_row['lon'], 6)) + ' (SIF = ' + str(round(image_row['SIF'], 3)) + ')'
        #print('BLUE: max', np.max(subtile[:, :, 1]), 'min', np.min(subtile[:, :, 1]))
        #print('GREEN: max', np.max(subtile[:, :, 2]), 'min', np.min(subtile[:, :, 2]))
        #print('RED: max', np.max(subtile[:, :, 3]), 'min', np.min(subtile[:, :, 3]))
        images[title] = subtile[:, :, RGB_BANDS] / 1000

    plot_figures(output_file, images, nrows=math.ceil(len(images) / 5), ncols=5)
 

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATE = "2016-08-01"
TILES_DIR = os.path.join(DATA_DIR, "tiles_" + DATE)
PROCESSED_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_all_2")
TILE_AVERAGE_TRAIN_FILE = os.path.join(PROCESSED_DATASET_DIR, "tile_info_train.csv")
BAND_STATISTICS_FILE = os.path.join(PROCESSED_DATASET_DIR, "band_statistics_train.csv")
# METHOD = "7_unet_small_both_1000samples_random_output_0.03"
METHOD = "7_unet_small_both_1000samples"
# METHOD = "7_unet_small_both_random_output_0.01_decay_1e-3"
CFIS_UNET_RESULTS_FILE = os.path.join(PROCESSED_DATASET_DIR, "cfis_results_" + METHOD + ".csv")
UNET_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)

RES = (0.00026949458523585647, 0.00026949458523585647)
SUBTILE_PIXELS = 10
SUBTILE_DEGREES = RES[0] * SUBTILE_PIXELS

# LAT_LON = 'lat_' + str(LAT) + '_lon_' + str(LON)
# TILE_DESCRIPTION = LAT_LON + '_' + DATE
# IMAGE_FILE = os.path.join(TILES_DIR, "reflectance_" + LAT_LON + ".npy")

TILE_DEGREES = 0.1
eps = TILE_DEGREES / 2

RGB_BANDS = [3, 2, 1]
CDL_BANDS = list(range(12, 42))
UNET_BANDS = list(range(0, 43)) # list(range(0, 12)) + list(range(12, 27)) + [28] + [42] #list(range(0, 43)) #
INPUT_CHANNELS_UNET = len(UNET_BANDS)
REDUCED_CHANNELS = 15 
INPUT_SIZE = 371

MIN_INPUT = -3
MAX_INPUT = 3
MAX_CFIS_SIF = 2.7
MIN_SIF = None
MAX_SIF = None


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
sif_cmap = plt.get_cmap('viridis')
sif_cmap.set_bad(color='red')

# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]
if MIN_SIF is not None and MAX_SIF is not None:
    min_output = (MIN_SIF - sif_mean) / sif_std
    max_output = (MAX_SIF - sif_mean) / sif_std
else:
    min_output = None
    max_output = None

# Load "band average" dataset
train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE)

# Read CFIS results
unet_results_cfis = pd.read_csv(CFIS_UNET_RESULTS_FILE)
print('Read CFIS')

# Standardize train data (note: results file is already standardized)
for column in COLUMNS_TO_STANDARDIZE:
    column_mean = train_set[column].mean()
    column_std = train_set[column].std()
    train_set[column] = np.clip((train_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)

    # Also standardize CFIS averages
    unet_results_cfis[column] = np.clip((unet_results_cfis[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)

print('standardized train')

# Train linear regression and MLP to predict SIF given (standardized) band averages
# print('=============== train set =============')
# pd.set_option('display.max_columns', 500)
# print(train_set.head())
X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
linear_regression = LinearRegression().fit(X_train, Y_train)
mlp_regression = MLPRegressor(hidden_layer_sizes=(100, 100)).fit(X_train, Y_train)

# Load U-Net model
unet_model = UNetSmall(n_channels=INPUT_CHANNELS_UNET, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
unet_model.load_state_dict(torch.load(UNET_MODEL_FILE, map_location=device))
unet_model.eval()

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds, min_input=MIN_INPUT, max_input=MAX_INPUT))
transform = transforms.Compose(transform_list)



# Group points by which large tile file they come from
large_tile_files = unet_results_cfis['large_tile_file'].value_counts(sort=True)
for large_tile_file, count in large_tile_files.iteritems(): # large_tile_rows in unet_results_cfis.groupby('large_tile_file', as_index=False):
    print('Large tile file', large_tile_file, 'count', count)
    large_tile_rows = unet_results_cfis[unet_results_cfis['large_tile_file'] == large_tile_file]
    # print('head', large_tile_rows.head())
    
    # Read an input tile
    tile = np.load(large_tile_file)

    # Large tile description
    LAT = large_tile_rows.iloc[0]['large_tile_lat']
    LON = large_tile_rows.iloc[0]['large_tile_lon']
    DATE = large_tile_rows.iloc[0]['date']
    tile_description = 'lat_' + str(LAT) + '_lon_' + str(LON) + '_' + DATE

    # Standardize input tile
    input_tile_standardized = transform(tile)
    # print('Input tile dim', input_tile_standardized.shape)
    # print('Random pixel', input_tile_standardized[:, 8, 8])

    pixels = np.moveaxis(input_tile_standardized, 0, -1)
    pixels = pixels.reshape((-1, pixels.shape[2]))

    # Visualize the input tile
    rgb_tile = sif_utils.plot_tile(input_tile_standardized, 'cfis_' + tile_description, title=tile_description)

    # Obtain pixel predictions using linear/MLP regressor
    predicted_sifs_linear = linear_regression.predict(pixels).reshape((371, 371))
    predicted_sifs_mlp = mlp_regression.predict(pixels).reshape((371, 371))

    # Obtain U-Net model predictions
    unet_input = torch.tensor(input_tile_standardized[UNET_BANDS], dtype=torch.float).unsqueeze(0)  # Should be [1 x bands x 371 x 371]
    # print('UNet input shape', unet_input.shape)
    predicted_sifs_unet_standardized = unet_model(unet_input).detach().numpy()
    # print('UNet prediction shape', predicted_sifs_unet_standardized.shape)
    predicted_sifs_unet_non_standardized = (predicted_sifs_unet_standardized * sif_std + sif_mean).reshape((371, 371))


    # Compute tile-average based predictions for all CFIS sub-tiles in this large tile
    print('============= Large tile rows ==============')
    cfis_averages = large_tile_rows[INPUT_COLUMNS].to_numpy()
    cfis_predicted_sifs_linear = linear_regression.predict(cfis_averages).reshape(-1, 1)  # Need to reshape into 2d array #.reshape((4, 4))
    cfis_predicted_sifs_mlp = mlp_regression.predict(cfis_averages).reshape(-1, 1)  #.reshape((4, 371))
    cfis_predicted_sifs_unet = large_tile_rows['predicted'].to_numpy().reshape(-1, 1)

    # Rescale predictions linearly to match CFIS 
    # fit_intercept=False
    linear_to_cfis = LinearRegression().fit(cfis_predicted_sifs_linear, large_tile_rows['true'])
    mlp_to_cfis = LinearRegression().fit(cfis_predicted_sifs_mlp, large_tile_rows['true'])
    unet_to_cfis = LinearRegression().fit(cfis_predicted_sifs_unet, large_tile_rows['true'])
    # cfis_predicted_sifs_linear_rescaled = linear_to_cfis.predict(cfis_predicted_sifs_linear)
    # cfis_predicted_sifs_mlp_rescaled = mlp_to_cfis.predict(cfis_predicted_sifs_mlp)
    # cfis_predicted_sifs_unet_rescaled = unet_to_cfis.predict(cfis_predicted_sifs_unet)

    print('================== True vs linear predictions =================')
    print_stats(large_tile_rows['true'], cfis_predicted_sifs_linear, sif_mean, ax=plt.gca())
    # plt.xlim(left=0, right=MAX_CFIS_SIF)
    # plt.ylim(bottom=0, top=MAX_CFIS_SIF)
    plt.savefig('exploratory_plots/cfis_' + tile_description + '_true_vs_linear.png')
    plt.close()

    print('================== True vs MLP predictions ====================')
    print_stats(large_tile_rows['true'], cfis_predicted_sifs_mlp, sif_mean)

    print('================== True vs U-Net predictions ==================')
    print_stats(large_tile_rows['true'], cfis_predicted_sifs_unet, sif_mean, ax=plt.gca())
    plt.title('CFIS: true vs predicted SIF (' + METHOD + ', ' + tile_description + ')')
    # plt.xlim(left=0, right=MAX_CFIS_SIF)
    # plt.ylim(bottom=0, top=MAX_CFIS_SIF)
    plt.savefig('exploratory_plots/cfis_' + tile_description + '_true_vs_unet.png')
    plt.close()

    # Plot the highest error sub-tiles
    unet_errors = np.abs(large_tile_rows['true'].to_numpy() - cfis_predicted_sifs_unet) #large_tile_rows['predicted'])
    sorted_indices = unet_errors.argsort()  # Ascending order of distance
    high_error_indices = sorted_indices[-5:][::-1]
    for high_error_idx in high_error_indices:
        row = large_tile_rows.iloc[high_error_idx]
        subtile_lat = row['lat']
        subtile_lon = row['lon']
        subtile_true_sif = row['true']
        subtile_predicted_sif = row['predicted']
        print('subtile lat', subtile_lat)
        top_bound = sif_utils.get_top_bound(subtile_lat)
        left_bound = sif_utils.get_left_bound(subtile_lon)
        subtile_height_idx, subtile_width_idx = sif_utils.lat_long_to_index(subtile_lat, subtile_lon, top_bound, left_bound, RES)
        eps = int(SUBTILE_PIXELS / 2)
        height_start = max(subtile_height_idx-eps, 0)
        width_start = max(subtile_width_idx-eps, 0)
        height_end = min(subtile_height_idx+eps, input_tile_standardized.shape[1])
        width_end = min(subtile_width_idx+eps, input_tile_standardized.shape[2])

        high_error_subtile = input_tile_standardized[:, height_start:height_end, width_start:width_end]
        subtile_predictions = predicted_sifs_unet_non_standardized[height_start:height_end, width_start:width_end]
        subtile_description = 'cfis_high_error_lat_' + str(subtile_lat) + '_lon_' + str(subtile_lon)
        title = 'Lat ' + str(round(subtile_lat, 4)) + ', Lon ' + str(round(subtile_lon, 4))
        title += ('\n(True SIF: ' + str(round(subtile_true_sif, 3)) + ', Predicted SIF: ' + str(round(subtile_predicted_sif, 3)) + ')')
        print('Tile:', subtile_description)
        print('header:', title)
        sif_utils.plot_tile(high_error_subtile, subtile_description, title=title)

        # Plot U-Net's pixel-level predictions for this sub-tile
        plt.imshow(subtile_predictions, cmap=sif_cmap, vmin=0.2, vmax=1.7)
        plt.colorbar()
        plt.title('U-Net pixel predictions (avg ' + str(round(np.mean(subtile_predictions), 3)) + ') ' + title)
        plt.savefig('exploratory_plots/' + subtile_description + '_unet_predictions.png')
        plt.close()

    # Construct collection of patches
    patches = []
    for index, row in large_tile_rows.iterrows():
        # vertices = [[row['lon_0'], row['lat_0']],
        #             [row['lon_1'], row['lat_1']],
        #             [row['lon_2'], row['lat_2']],
        #             [row['lon_3'], row['lat_3']],
        #             [row['lon_0'], row['lat_0']]]
        subtile_eps = SUBTILE_DEGREES / 2
        vertices = [[row['lon'] - subtile_eps, row['lat'] + subtile_eps],
                    [row['lon'] + subtile_eps, row['lat'] + subtile_eps],
                    [row['lon'] + subtile_eps, row['lat'] - subtile_eps],
                    [row['lon'] - subtile_eps, row['lat'] - subtile_eps],
                    [row['lon'] - subtile_eps, row['lat'] + subtile_eps]]
        polygon = Polygon(vertices, True)
        patches.append(polygon)

    true_patches = PatchCollection(patches, alpha=1, cmap=sif_cmap)
    true_patches.set_clim(0.2, 1.7)
    true_patches.set_array(large_tile_rows['true'])

    # Construct linear patches (sorry, redundant code)
    linear_predicted_patches = PatchCollection(patches, alpha=1, cmap=sif_cmap)
    linear_predicted_patches.set_clim(0.2, 1.7)
    linear_predicted_patches.set_array(cfis_predicted_sifs_linear_rescaled)

    # Construct MLP patches
    mlp_predicted_patches = PatchCollection(patches, alpha=1, cmap=sif_cmap)
    mlp_predicted_patches.set_clim(0.2, 1.7)
    mlp_predicted_patches.set_array(cfis_predicted_sifs_mlp_rescaled)

    # Construct U-Net patches
    unet_predicted_patches = PatchCollection(patches, alpha=1, cmap=sif_cmap)
    unet_predicted_patches.set_clim(0.2, 1.7)
    unet_predicted_patches.set_array(cfis_predicted_sifs_unet_rescaled)

    # Plot different method's predictions
    fig, axeslist = plt.subplots(ncols=4, nrows=2, figsize=(40, 20))
    axeslist[0, 0].imshow(rgb_tile)
    axeslist[0, 0].set_title('RGB Bands')
    axeslist[0, 1].imshow(predicted_sifs_linear, cmap=sif_cmap, vmin=0.2, vmax=1.7)
    axeslist[0, 1].set_title('Linear Regression: predicted SIF')
    axeslist[0, 2].imshow(predicted_sifs_mlp, cmap=sif_cmap, vmin=0.2, vmax=1.7)
    axeslist[0, 2].set_title('MLP: predicted SIF')
    axeslist[0, 3].imshow(predicted_sifs_unet_non_standardized, cmap=sif_cmap, vmin=0.2, vmax=1.7)
    axeslist[0, 3].set_title('Unet: predicted SIF')

    axeslist[1, 0].add_collection(true_patches)
    axeslist[1, 0].set_xlim(LON-eps, LON+eps)
    axeslist[1, 0].set_ylim(LAT-eps, LAT+eps)
    axeslist[1, 0].set_title('True CFIS SIF')

    axeslist[1, 1].add_collection(linear_predicted_patches)
    axeslist[1, 1].set_xlim(LON-eps, LON+eps)
    axeslist[1, 1].set_ylim(LAT-eps, LAT+eps)
    axeslist[1, 1].set_title('Predicted OCO2 SIF (linear regression)')

    axeslist[1, 2].add_collection(mlp_predicted_patches)
    axeslist[1, 2].set_xlim(LON-eps, LON+eps)
    axeslist[1, 2].set_ylim(LAT-eps, LAT+eps)
    axeslist[1, 2].set_title('Predicted OCO2 SIF (MLP)')

    axeslist[1, 3].add_collection(unet_predicted_patches)
    axeslist[1, 3].set_xlim(LON-eps, LON+eps)
    axeslist[1, 3].set_ylim(LAT-eps, LAT+eps)
    axeslist[1, 3].set_title('Predicted OCO2 SIF (U-Net)')

    fig.colorbar(true_patches, ax=axeslist.ravel().tolist())
    plt.savefig('exploratory_plots/cfis_' + tile_description + '_compare_results.png')
    plt.close()



