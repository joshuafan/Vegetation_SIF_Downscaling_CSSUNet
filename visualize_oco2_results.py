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

from sif_utils import lat_long_to_index, plot_histogram, get_subtiles_list
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
DATE = "2018-06-24"
TILES_DIR = os.path.join(DATA_DIR, "tiles_" + DATE)
PROCESSED_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_all_2")
TILE_AVERAGE_TRAIN_FILE = os.path.join(PROCESSED_DATASET_DIR, "tile_info_train.csv")
#TILE_AVERAGE_VAL_FILE = os.path.join(DATASET_DIR, "tile_averages_val.csv")
BAND_STATISTICS_FILE = os.path.join(PROCESSED_DATASET_DIR, "band_statistics_train.csv")
OCO2_SUBTILE_CNN_RESULTS_FILE = os.path.join(PROCESSED_DATASET_DIR, "OCO2_results_all_1d_train_tropomi_subtile_resnet.csv")
OCO2_UNET_RESULTS_FILE = os.path.join(PROCESSED_DATASET_DIR, "OCO2_results_7_unet_small.csv")
TROPOMI_SIF_FILE = os.path.join(DATA_DIR, "TROPOMI_SIF/TROPO-SIF_01deg_biweekly_Apr18-Jan20.nc")
TROPOMI_DATE_RANGE = slice("2018-07-08", "2018-07-21")

RES = (0.00026949458523585647, 0.00026949458523585647)
SUBTILE_PIXELS = 100
SUBTILE_DEGREES = RES[0] * SUBTILE_PIXELS

#LAT = 41.15
#LON = -89.35
#LAT = 48.65
#LON = -84.45
#LAT = 42.55
#LON = -93.55 #-101.35  #-93.35
#LAT = 42.65
#LON = -93.35
# LAT = 42.55
# LON = -93.35
#LAT = 38.25 #41.25
#LON = -98.95
# LAT = 39.85
# LON = -96.35
# LAT = 39.95
# LON = -96.35
LAT = 38.15
LON = -97.25
LAT_LON = 'lat_' + str(LAT) + '_lon_' + str(LON)
TILE_DESCRIPTION = LAT_LON + '_' + DATE

TILE_DEGREES = 0.1
eps = TILE_DEGREES / 2
OCO2_IMAGE_FILE = os.path.join(TILES_DIR, "reflectance_" + LAT_LON + ".npy")

SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/all_1d_train_tropomi_subtile_resnet")
UNET_MODEL_FILE = os.path.join(DATA_DIR, "models/7_unet_small")

RGB_BANDS = [3, 2, 1]
CDL_BANDS = list(range(12, 42))
BANDS = list(range(0, 43)) # list(range(0, 12)) + list(range(12, 27)) + [28] + [42]  #
INPUT_CHANNELS = len(BANDS)
UNET_BANDS = list(range(0, 43)) # list(range(0, 12)) + list(range(12, 27)) + [28] + [42] #list(range(0, 43)) #
INPUT_CHANNELS_UNET = len(UNET_BANDS)
REDUCED_CHANNELS = 15 
SUBTILE_DIM = 100
TILE_SIZE_DEGREES = 0.1
INPUT_SIZE = 371
OUTPUT_SIZE = round(INPUT_SIZE / SUBTILE_DIM)

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

MIN_INPUT = -3
MAX_INPUT = 3
MIN_SIF = 0
MAX_SIF = 1.7
min_output = (MIN_SIF - sif_mean) / sif_std
max_output = (MAX_SIF - sif_mean) / sif_std

# Load subtile SIF model
# subtile_sif_model = simple_cnn.SimpleCNNSmall4(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
subtile_sif_model = resnet.resnet18(input_channels=INPUT_CHANNELS, num_classes=1,
                                        min_output=min_output, max_output=max_output).to(device)
subtile_sif_model.load_state_dict(torch.load(SUBTILE_SIF_MODEL_FILE, map_location=device))
subtile_sif_model.eval()

unet_model = UNetSmall(n_channels=INPUT_CHANNELS_UNET, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
unet_model.load_state_dict(torch.load(UNET_MODEL_FILE, map_location=device))
unet_model.eval()

# Read OCO-2 results
oco2_subtile_cnn_results = pd.read_csv(OCO2_SUBTILE_CNN_RESULTS_FILE)
oco2_subtile_cnn_results = oco2_subtile_cnn_results[(oco2_subtile_cnn_results['source'] == 'OCO2') & (oco2_subtile_cnn_results['date'] == DATE)]
oco2_unet_results = pd.read_csv(OCO2_UNET_RESULTS_FILE)
oco2_unet_results = oco2_unet_results[(oco2_unet_results['source'] == 'OCO2') & (oco2_unet_results['date'] == DATE)]

bin_size = 0.2
bin_start = 0.2
while bin_start < 1.2:
    bin_end = round(bin_start + bin_size, 1)
    bin_points = oco2_subtile_cnn_results[(oco2_subtile_cnn_results['true'] > bin_start) & (oco2_subtile_cnn_results['true'] < bin_end)]
    print(len(bin_points), 'points with true SIF between', bin_start, 'and', bin_end)
    plot_histogram(bin_points['predicted'].to_numpy() - bin_points['true'].to_numpy(), 'oco2_sif_errors_' + str(bin_start) + '_to_' + str(bin_end) + '.png',
                   title='Predicted - True SIF (OCO-2 sub-tiles where true SIF between ' + str(bin_start) + ' and ' + str(bin_end) + ')')
    bin_start = bin_end

oco2_subtile_cnn_region_points = oco2_subtile_cnn_results.loc[(oco2_subtile_cnn_results['lon'] >= LON-eps) & (oco2_subtile_cnn_results['lon'] <= LON+eps) & (oco2_subtile_cnn_results['lat'] >= LAT-eps) & (oco2_subtile_cnn_results['lat'] <= LAT+eps)]
print("OCO-2 region points (sub-tile CNN):", oco2_subtile_cnn_region_points.head())
oco2_unet_region_points = oco2_unet_results.loc[(oco2_unet_results['lon'] >= LON-eps) & (oco2_unet_results['lon'] <= LON+eps) & (oco2_unet_results['lat'] >= LAT-eps) & (oco2_unet_results['lat'] <= LAT+eps)]
print("OCO-2 region points (U-Net):", oco2_unet_region_points.head())

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds, min_input=MIN_INPUT, max_input=MAX_INPUT))
transform = transforms.Compose(transform_list)

# Read an input tile
tile = np.load(OCO2_IMAGE_FILE)
input_tile_non_standardized = torch.tensor(tile, dtype=torch.float).to(device)


# Load "band average" dataset
train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE)
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
sif_cmap = plt.get_cmap('YlGn')
sif_cmap.set_bad(color='red')

# Standardize data
for column in COLUMNS_TO_STANDARDIZE:
    column_mean = train_set[column].mean()
    column_std = train_set[column].std()
    train_set[column] = np.clip((train_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)

# Standardize input tile
input_tile_standardized = transform(tile)
print('Input tile dim', input_tile_standardized.shape)
print('Random pixel', input_tile_standardized[:, 8, 8])
subtiles_standardized = get_subtiles_list(input_tile_standardized, SUBTILE_DIM)  # (batch x num subtiles x bands x subtile_dim x subtile_dim)
# subtile_averages = np.mean(subtiles_standardized, axis=(2,3))
pixels = np.moveaxis(input_tile_standardized, 0, -1)
pixels = pixels.reshape((-1, pixels.shape[2]))

# Visualize the input tile
array = input_tile_standardized.transpose((1, 2, 0))
rgb_tile = (array[:, :, RGB_BANDS] + 3) / 6
plt.imshow(rgb_tile)
plt.savefig("exploratory_plots/" + TILE_DESCRIPTION + "_rgb.png")
plt.close()

# Visualize CDL
cdl_utils.plot_cdl_layers(tile[CDL_BANDS, :, :], "exploratory_plots/" + TILE_DESCRIPTION + "_cdl.png")

fig, axeslist = plt.subplots(ncols=6, nrows=8, figsize=(24, 24))
for band in range(0, 43):
    layer = array[:, :, band]
    axeslist.ravel()[band].imshow(layer, cmap='Greens', vmin=np.min(layer), vmax=np.max(layer))
    axeslist.ravel()[band].set_title('Band ' + str(band))
    axeslist.ravel()[band].set_axis_off()
plt.tight_layout() # optional
plt.savefig('exploratory_plots/' + TILE_DESCRIPTION + '_all_bands.png')
plt.close()


# Train linear regression and MLP to predict SIF given (standardized) band averages
X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
linear_regression = LinearRegression().fit(X_train, Y_train)
predicted_sifs_linear = linear_regression.predict(pixels).reshape((371, 371))
print('Predicted sifs linear', predicted_sifs_linear.shape)
mlp_regression = MLPRegressor(hidden_layer_sizes=(100, 100)).fit(X_train, Y_train)
predicted_sifs_mlp = mlp_regression.predict(pixels).reshape((371, 371))

# Obtain simple CNN model's subtile SIF predictions
print('Subtile shape', subtiles_standardized.shape)
with torch.set_grad_enabled(False):
    subtiles_standardized_tensor = torch.tensor(subtiles_standardized[:, BANDS], dtype=torch.float).to(device)
    predicted_sifs_simple_cnn_standardized = subtile_sif_model(subtiles_standardized_tensor).detach().numpy()
print('Predicted SIFs standardized', predicted_sifs_simple_cnn_standardized.shape)
predicted_sifs_simple_cnn_non_standardized = (predicted_sifs_simple_cnn_standardized * sif_std + sif_mean).reshape((4, 4))

# Obtain U-Net model predictions
unet_input = torch.tensor(input_tile_standardized[UNET_BANDS], dtype=torch.float).unsqueeze(0)  # Should be [1 x bands x 371 x 371]
print('UNet input shape', unet_input.shape)
predicted_sifs_unet_standardized = unet_model(unet_input).detach().numpy()
print('UNet prediction shape', predicted_sifs_unet_standardized.shape)
predicted_sifs_unet_non_standardized = (predicted_sifs_unet_standardized * sif_std + sif_mean).reshape((371, 371))

# Construct collection of patches
patches = []
for index, row in oco2_subtile_cnn_region_points.iterrows():
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

true_patches = PatchCollection(patches, alpha=1, cmap="YlGn")
true_patches.set_clim(0.2, 1.7)
true_patches.set_array(oco2_subtile_cnn_region_points['true'])

subtile_cnn_predicted_patches = PatchCollection(patches, alpha=1, cmap="YlGn")
subtile_cnn_predicted_patches.set_clim(0.2, 1.7)
subtile_cnn_predicted_patches.set_array(oco2_subtile_cnn_region_points['predicted'])

unet_predicted_patches = PatchCollection(patches, alpha=1, cmap="YlGn")
unet_predicted_patches.set_clim(0.2, 1.7)
unet_predicted_patches.set_array(oco2_unet_region_points['predicted'])

# Construct linear patches (sorry, redundant code)
oco2_averages = oco2_subtile_cnn_region_points[INPUT_COLUMNS].to_numpy()
oco2_predicted_sifs_linear = linear_regression.predict(oco2_averages) #.reshape((4, 4))
linear_predicted_patches = PatchCollection(patches, alpha=1, cmap="YlGn")
linear_predicted_patches.set_clim(0.2, 1.7)
linear_predicted_patches.set_array(oco2_predicted_sifs_linear)

# Construct MLP patches
oco2_predicted_sifs_mlp = mlp_regression.predict(oco2_averages) #.reshape((4, 371))
mlp_predicted_patches = PatchCollection(patches, alpha=1, cmap="YlGn")
mlp_predicted_patches.set_clim(0.2, 1.7)
mlp_predicted_patches.set_array(oco2_predicted_sifs_linear)

# Plot different method's predictions
fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(50, 20))
axeslist[0 ,0].imshow(rgb_tile)
axeslist[0, 0].set_title('RGB Bands')
axeslist[0, 1].imshow(predicted_sifs_linear, cmap=sif_cmap, vmin=0.2, vmax=1.7)
axeslist[0, 1].set_title('Linear Regression: predicted SIF')
axeslist[0, 2].imshow(predicted_sifs_mlp, cmap=sif_cmap, vmin=0.2, vmax=1.7)
axeslist[0, 2].set_title('MLP: predicted SIF')
axeslist[0, 3].imshow(predicted_sifs_simple_cnn_non_standardized, cmap=sif_cmap, vmin=0.2, vmax=1.7)
axeslist[0, 3].set_title('Subtile CNN: predicted SIF')
axeslist[0, 4].imshow(predicted_sifs_unet_non_standardized, cmap=sif_cmap, vmin=0.2, vmax=1.7)
axeslist[0, 4].set_title('Unet: predicted SIF')


axeslist[1, 0].add_collection(true_patches)
axeslist[1, 0].set_xlim(LON-eps, LON+eps)
axeslist[1, 0].set_ylim(LAT-eps, LAT+eps)
axeslist[1, 0].set_title('True OCO2 SIF')

axeslist[1, 1].add_collection(linear_predicted_patches)
axeslist[1, 1].set_xlim(LON-eps, LON+eps)
axeslist[1, 1].set_ylim(LAT-eps, LAT+eps)
axeslist[1, 1].set_title('Predicted OCO2 SIF (linear regression)')

axeslist[1, 2].add_collection(mlp_predicted_patches)
axeslist[1, 2].set_xlim(LON-eps, LON+eps)
axeslist[1, 2].set_ylim(LAT-eps, LAT+eps)
axeslist[1, 2].set_title('Predicted OCO2 SIF (MLP)')

axeslist[1, 3].add_collection(subtile_cnn_predicted_patches)
axeslist[1, 3].set_xlim(LON-eps, LON+eps)
axeslist[1, 3].set_ylim(LAT-eps, LAT+eps)
axeslist[1, 3].set_title('Predicted OCO2 SIF (subtile CNN)')

axeslist[1, 4].add_collection(unet_predicted_patches)
axeslist[1, 4].set_xlim(LON-eps, LON+eps)
axeslist[1, 4].set_ylim(LAT-eps, LAT+eps)
axeslist[1, 4].set_title('Predicted OCO2 SIF (U-Net)')

fig.colorbar(true_patches, ax=axeslist.ravel().tolist())
plt.savefig('exploratory_plots/' + TILE_DESCRIPTION + '_oco2_compare_predictions.png')
plt.close()


# Open TROPOMI SIF dataset
tropomi_dataset = xr.open_dataset(TROPOMI_SIF_FILE)
tropomi_array = tropomi_dataset.sif_dc.sel(time=TROPOMI_DATE_RANGE).mean(dim='time')

# Compare stats
predicted_sifs_simple_cnn_non_standardized = np.clip(predicted_sifs_simple_cnn_non_standardized, a_min=0.2, a_max=1.7)
print('===================== Comparing stats ======================')
print('CNN predictions for this tile (grid): mean', round(np.mean(predicted_sifs_simple_cnn_non_standardized), 3), 'std', round(np.std(predicted_sifs_simple_cnn_non_standardized), 3)) #'min', np.min(predicted_sifs_simple_cnn_non_standardized), 'max', np.max(predicted_sifs_simple_cnn_non_standardized))
print('Linear predictions for this tile (grid): mean', round(np.mean(predicted_sifs_linear), 3), 'std', round(np.std(predicted_sifs_linear), 3)) #, 'min', np.min(predicted_sifs_linear), 'max', np.max(predicted_sifs_linear))
print('Ground-truth OCO-2 SIF for this tile: mean', round(np.mean(oco2_subtile_cnn_results['true']), 3), 'std', round(np.std(oco2_subtile_cnn_results['true']), 3)) # 'min', np.min(predicted_sifs_tile2vec_fixed_non_standardized), 'max', np.max(predicted_sifs_tile2vec_fixed_non_standardized))
print('Predicted OCO-2 SIF for this tile (CNN): mean', round(np.mean(oco2_subtile_cnn_results['predicted']), 3), 'std', round(np.std(oco2_subtile_cnn_results['predicted']), 3)) # 'min', np.min(predicted_sifs_tile2vec_fixed_non_standardized), 'max', np.max(predicted_sifs_tile2vec_fixed_non_standardized))
print('Predicted OCO-2 SIF for this tile (U-Net): mean', round(np.mean(oco2_unet_results['predicted']), 3), 'std', round(np.std(oco2_unet_results['true']), 3)) # 'min', np.min(predicted_sifs_tile2vec_fixed_non_standardized), 'max', np.max(predicted_sifs_tile2vec_fixed_non_standardized))
print('TROPOMI SIF for this tile', tropomi_array.sel(lat=LAT, lon=LON, method='nearest'))
print('============================================================')


