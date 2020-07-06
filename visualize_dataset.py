import math
import matplotlib
import matplotlib
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
from unet.unet_model import UNet

import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet
from embedding_to_sif_model import EmbeddingToSIFModel
from embedding_to_sif_nonlinear_model import EmbeddingToSIFNonlinearModel


# Taken from https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plot_figures(output_file, figures, nrows = 1, ncols=1, cmap=None):
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
        axeslist.ravel()[ind].imshow(figures[title], vmin=0, vmax=1, cmap=cmap)  #, cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        #axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.savefig(output_file)
    plt.close()

def plot_rgb_images(image_rows, image_filename_column, output_file):
    images = {}
    for idx, image_row in image_rows.iterrows():
        subtile = np.load(image_row[image_filename_column]).transpose((1, 2, 0))
        title = 'Lat' + str(round(image_row['lat'], 6)) + ', Lon' + str(round(image_row['lon'], 6)) + ' (SIF = ' + str(round(image_row['SIF'], 3)) + ')'
        #print('BLUE: max', np.max(subtile[:, :, 1]), 'min', np.min(subtile[:, :, 1]))
        #print('GREEN: max', np.max(subtile[:, :, 2]), 'min', np.min(subtile[:, :, 2]))
        #print('RED: max', np.max(subtile[:, :, 3]), 'min', np.min(subtile[:, :, 3]))
        images[title] = subtile[:, :, RGB_BANDS] / 1000

    plot_figures(output_file, images, nrows=math.ceil(len(images) / 5), ncols=5)
 

def plot_band_images(image_rows, image_filename_column, output_file_prefix):
    band_to_max = {4: 5000, 5: 3000, 6: 2000}
    band_images = {}
    for band in band_to_max:
        band_images[band] = {}

    for idx, image_row in image_rows.iterrows():
        subtile = np.load(image_row[image_filename_column]).transpose((1, 2, 0))
        title = 'Lat' + str(round(image_row['lat'], 6)) + ', Lon' + str(round(image_row['lon'], 6)) + ' (SIF = ' + str(round(image_row['SIF'], 3)) + ')'
        for band, max_value in band_to_max.items():
            band_images[band][title] = subtile[:, :, band] / max_value

    for band, images in band_images.items():
        plot_figures(output_file_prefix + '_band_' + str(band) + '.png', images, nrows=math.ceil(len(images) / 5), ncols=5, cmap=plt.get_cmap('YlGn'))


# Note there's a lot of redundant code here
def plot_cdl_layers(image_rows, image_filename_column, output_file):
    # Load all tiles and store the CDL bands
    images = {}
    for idx, image_row in image_rows.iterrows():
        cdl_layers = np.load(image_row[image_filename_column])[CDL_BANDS, :, :]
        title = 'Lat' + str(round(image_row['lat'], 6)) + ', Lon' + str(round(image_row['lon'], 6)) + ' (SIF = ' + str(round(image_row['SIF'], 3)) + ')'
        images[title] = cdl_layers

    # Set up plots
    fig, axeslist = plt.subplots(ncols=5, nrows=math.ceil(len(images) / 5), figsize=(20, 20))
    # Custom CDL colormap
    cmap = matplotlib.colors.ListedColormap(cdl_utils.CDL_COLORS)

    for ind, title in enumerate(images):
        # Convert CDL bands into a single layer (each pixel has one number representing the crop type)
        cover_bands = images[title]
        cover_tile = np.zeros((cover_bands.shape[1], cover_bands.shape[2]))
        for i in range(cover_bands.shape[0]):
            # Reserving 0 for no cover, so add 1
            cover_tile[cover_bands[i, :, :] == 1] = i + 1
        img = axeslist.ravel()[ind].imshow(cover_tile, interpolation='nearest',
                     cmap=cmap, vmin=-0.5, vmax=len(cdl_utils.CDL_COLORS)-0.5)
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()

    ticks_loc = np.arange(0, len(cdl_utils.CDL_COLORS), 1) #len(COVERS_TO_MASK) / len(CDL_COLORS))
    cb = plt.colorbar(img, cmap=cmap)
    cb.set_ticks(ticks_loc)
    cb.set_ticklabels(cdl_utils.COVER_NAMES)
    cb.ax.tick_params(labelsize='small')
    plt.tight_layout() # optional
    plt.savefig(output_file)
    plt.close()      


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset")
TRAIN_TILE_DATASET = os.path.join(DATASET_DIR, "tile_info_train.csv")
VAL_TILE_DATASET = os.path.join(DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")

DATES = ["2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02"]

# TRAIN_DATE = "2018-08-01" 
# TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + TRAIN_DATE)
# TRAIN_TILE_DATASET = os.path.join(TRAIN_DATASET_DIR, "tile_info_train.csv")
# ALL_TILE_DATASET = os.path.join(TRAIN_DATASET_DIR, "reflectance_cover_to_sif.csv")
# OCO2_SUBTILE_DATASET = os.path.join(TRAIN_DATASET_DIR, "oco2_eval_subtiles.csv")

#LAT = 41.15
#LON = -89.35
#LAT = 48.65
#LON = -84.45
#LAT = 42.55
#LON = -93.55 #-101.35  #-93.35
#LAT = 42.65
#LON = -93.35
#LAT = 42.55
#LON = -93.35
# LAT = 41.15
# LON = -96.45
LAT = 47.55
LON = -101.35
LAT_LON = 'lat_' + str(LAT) + '_lon_' + str(LON)
SELECTED_DATE = '2018-08-05'
SELECTED_TILE_DESCRIPTION = LAT_LON + '_' + SELECTED_DATE

TILE_DEGREES = 0.1
eps = TILE_DEGREES / 2

CFIS_SIF_FILE = os.path.join(DATA_DIR, "CFIS/CFIS_201608a_300m_soundings.npy")
TROPOMI_SIF_FILE = os.path.join(DATA_DIR, "TROPOMI_SIF/TROPO-SIF_01deg_biweekly_Apr18-Jan20.nc")
OCO2_SIF_FILE = os.path.join(DATA_DIR, "OCO2/oco2_20180708_20180915_14day_3km.nc")

CFIS_DATE = '2016-08-01'
CFIS_SUBTILE_DATASET = os.path.join(DATA_DIR, "dataset_" + CFIS_DATE + "/eval_subtiles.csv")
CFIS_DATE_RANGE = pd.date_range(start="2016-08-01", end="2016-08-16")

RGB_BANDS = [3, 2, 1]
CDL_BANDS = list(range(12, 42))
SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn_v4_2")
TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_recon_5/TileNet.ckpt") #finetuned_tile2vec.ckpt") #TileNet.ckpt")
EMBEDDING_TO_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_embedding_to_sif") #finetuned_tile2vec_embedding_to_sif.ckpt") #tile2vec_embedding_to_sif")
EMBEDDING_TYPE = 'tile2vec'
Z_DIM = 256
HIDDEN_DIM = 1024

# SAN_MODEL_FILE_37 = os.path.join(DATA_DIR, "models/SAN_feat37")
# SAN_MODEL_FILE_74 = os.path.join(DATA_DIR, "models/SAN_feat74")
# SAN_MODEL_FILE_111 = os.path.join(DATA_DIR, "models/SAN_feat111_3")
UNET_MODEL_FILE = os.path.join(DATA_DIR, "models/unet")

INPUT_CHANNELS = 43
CROP_TYPE_START_IDX = 12
BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42] # list(range(0, 43))
INPUT_CHANNELS_SIMPLE = len(BANDS)
UNET_BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42]
INPUT_CHANNELS_UNET = len(UNET_BANDS)
REDUCED_CHANNELS = 15
SUBTILE_DIM = 10
TILE_SIZE_DEGREES = 0.1
INPUT_SIZE = 371
MIN_INPUT = -2
MAX_INPUT = 2
OUTPUT_SIZE = int(INPUT_SIZE / SUBTILE_DIM)


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

MIN_SIF = 0.2
MAX_SIF = 1.7
min_output = (MIN_SIF - sif_mean) / sif_std
max_output = (MAX_SIF - sif_mean) / sif_std


# Load subtile SIF model
# subtile_sif_model = simple_cnn.SimpleCNNSmall3(input_channels=INPUT_CHANNELS_SIMPLE, reduced_channels=REDUCED_CHANNELS, crop_type_start_idx=CROP_TYPE_START_IDX, output_dim=1, min_output=min_output, max_output=max_output).to(device)
subtile_sif_model = simple_cnn.SimpleCNNSmall3(input_channels=INPUT_CHANNELS_SIMPLE, output_dim=1, min_output=min_output, max_output=max_output).to(device)
subtile_sif_model.load_state_dict(torch.load(SUBTILE_SIF_MODEL_FILE, map_location=device))
subtile_sif_model.eval()

# Load trained models from file
if EMBEDDING_TYPE == 'tile2vec':
    tile2vec_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=Z_DIM).to(device)
    tile2vec_model.load_state_dict(torch.load(TILE2VEC_MODEL_FILE, map_location=device))
    tile2vec_model.eval()
else:
    tile2vec_model = None
embedding_to_sif_model = EmbeddingToSIFNonlinearModel(embedding_size=Z_DIM, hidden_size=HIDDEN_DIM, min_output=min_output, max_output=max_output).to(device)
embedding_to_sif_model.load_state_dict(torch.load(EMBEDDING_TO_SIF_MODEL_FILE, map_location=device))
embedding_to_sif_model.eval()

# Load SAN model
# resnet_model = resnet.resnet18(input_channels=INPUT_CHANNELS) 
# san_model_37 = SAN(resnet_model, min_output=min_output, max_output=max_output,
#                 input_height=INPUT_SIZE, input_width=INPUT_SIZE,
#                 output_height=OUTPUT_SIZE, output_width=OUTPUT_SIZE,
#                 feat_width=OUTPUT_SIZE, feat_height=OUTPUT_SIZE,
#                 in_channels=INPUT_CHANNELS).to(device)
# san_model_37.load_state_dict(torch.load(SAN_MODEL_FILE_37, map_location=device))
# san_model_37.eval()
# resnet_model = resnet.resnet18(input_channels=INPUT_CHANNELS) 
# san_model_74 = SAN(resnet_model, min_output=min_output, max_output=max_output,
#                 input_height=INPUT_SIZE, input_width=INPUT_SIZE,
#                 output_height=OUTPUT_SIZE, output_width=OUTPUT_SIZE,
#                 feat_width=2*OUTPUT_SIZE, feat_height=2*OUTPUT_SIZE,
#                 in_channels=INPUT_CHANNELS).to(device)
# san_model_74.load_state_dict(torch.load(SAN_MODEL_FILE_74, map_location=device))
# san_model_74.eval()
# resnet_model = resnet.resnet18(input_channels=INPUT_CHANNELS) 
# san_model_111 = SAN(resnet_model, min_output=min_output, max_output=max_output,
#                 input_height=INPUT_SIZE, input_width=INPUT_SIZE,
#                 output_height=OUTPUT_SIZE, output_width=OUTPUT_SIZE,
#                 feat_width=3*OUTPUT_SIZE, feat_height=3*OUTPUT_SIZE,
#                 in_channels=INPUT_CHANNELS).to(device)
# san_model_111.load_state_dict(torch.load(SAN_MODEL_FILE_111, map_location=device))
# san_model_111.eval()

unet_model = UNet(n_channels=INPUT_CHANNELS_UNET, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
unet_model.load_state_dict(torch.load(UNET_MODEL_FILE, map_location=device))
unet_model.eval()

# Colormap for SIF
sif_cmap = plt.get_cmap('YlGn')
sif_cmap.set_bad(color='red')

# Load datasets
train_metadata = pd.read_csv(TRAIN_TILE_DATASET)
train_tropomi_metadata = train_metadata[train_metadata['source'] == 'TROPOMI'].copy()
train_oco2_metadata = train_metadata[train_metadata['source'] == 'OCO2'].copy()
val_metadata = pd.read_csv(VAL_TILE_DATASET)
cfis_metadata = pd.read_csv(CFIS_SUBTILE_DATASET)

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
CROP_TYPES = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']

print('input columns', INPUT_COLUMNS)
OUTPUT_COLUMN = ['SIF']

band_maxs = train_metadata.max(axis=0)
band_mins = train_metadata.min(axis=0)

# Plot histogram of each band (TROPOMI and OCO2, all time periods)
# tropomi_stats = []
# oco2_stats = []
# for column in INPUT_COLUMNS + OUTPUT_COLUMN:
#     print('============== ALL DATES - Column:', column, '==================')
#     tropomi_column = np.array(train_tropomi_metadata[column])
#     tropomi_stats.append([round(np.mean(tropomi_column), 3), round(np.std(tropomi_column), 3)])
#     print('-------------- TROPOMI train tiles -----------------')
#     plot_histogram(tropomi_column, "histogram_unstd_all_dates_" + column + "_tropomi.png", range=(band_mins[column], band_maxs[column]), title="All dates (TROPOMI): "+column)

#     # print('-------------- CFIS sub-tiles ----------------------')
#     # plot_histogram(np.array(cfis_metadata[column]), "histogram_cfis_" + column + ".png")

#     oco2_column = np.array(train_oco2_metadata[column])
#     oco2_stats.append([round(np.mean(oco2_column), 3), round(np.std(oco2_column), 3)])
#     print('-------------- OCO-2 train tiles ----------------------')
#     plot_histogram(oco2_column, "histogram_unstd_all_dates_" + column + "_oco2.png", range=(band_mins[column], band_maxs[column]), title="All dates (OCO2): "+column)

# Standardize data
for column in COLUMNS_TO_STANDARDIZE:
    column_mean = train_metadata[column].mean()
    column_std = train_metadata[column].std()
    train_metadata[column] = np.clip((train_metadata[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
    val_metadata[column] = np.clip((val_metadata[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)

    # Histograms of standardized columns
    plot_histogram(train_metadata[column].to_numpy(), "histogram_std_" + column + "_train.png")


# # Print band means/stds across all dates, for both TROPOMI and OCO-2
# print('============== ALL DATES - TROPOMI Summary stats =====================')
# for i in range(len(tropomi_stats)):
#     print(tropomi_stats[i][0]) # print(', '.join(map(str, tropomi_stats[i])))  # Need to convert numbers to strings
# print('============== ALL DATES - OCO2 Summary stats =====================')
# for i in range(len(oco2_stats)):
#     print(oco2_stats[i][0]) # print(', '.join(map(str, oco2_stats[i])))
# print('===================================================================')

# # For each date range, plot distribution of each band 
# for date in DATES:
#     tropomi_metadata_date = train_tropomi_metadata[train_tropomi_metadata['date'] == date]
#     oco2_metadata_date = train_oco2_metadata[train_oco2_metadata['date'] == date]
#     tropomi_stats = []
#     oco2_stats = []

#     # Plot histogram of each band (TROPOMI)
#     for column in INPUT_COLUMNS + OUTPUT_COLUMN:
#         print('============== DATE', date, '- Column:', column, '==================')
#         tropomi_column = np.array(tropomi_metadata_date[column])
#         tropomi_stats.append([round(np.mean(tropomi_column), 2), round(np.std(tropomi_column), 2)])
#         print('-------------- TROPOMI train tiles -----------------')
#         plot_histogram(tropomi_column, "histogram_unstd_" + date + "_" + column + "_tropomi.png", range=(band_mins[column], band_maxs[column]), title=date+" (TROPOMI): "+column)

#         oco2_column = np.array(oco2_metadata_date[column])
#         oco2_stats.append([round(np.mean(oco2_column), 2), round(np.std(oco2_column), 2)])
#         print('-------------- OCO-2 train tiles ----------------------')
#         plot_histogram(oco2_column, "histogram_unstd_" + date + "_" + column + "_oco2.png", range=(band_mins[column], band_maxs[column]), title=date+" (OCO2): "+column)

#     # Print band means/stds for this date
#     print('============== DATE', date, '- TROPOMI Summary stats =====================')
#     for i in range(len(tropomi_stats)):
#         print(tropomi_stats[i][0]) # print(', '.join(map(str, tropomi_stats[i])))
#     print('============== DATE', date, '- OCO2 Summary stats =====================')
#     for i in range(len(oco2_stats)):
#         print(oco2_stats[i][0]) # print(', '.join(map(str, oco2_stats[i])))
#     print('===================================================================')

#     # Plot "map" of TROPOMI data-points for this date
#     plt.figure(figsize=(24, 10))
#     scatterplot = plt.scatter(tropomi_metadata_date['lon'], tropomi_metadata_date['lat'], c=tropomi_metadata_date['SIF'], cmap=sif_cmap, vmin=0.2, vmax=1.5)
#     plt.colorbar(scatterplot)
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title('TROPOMI points, date: ' + date)
#     plt.savefig('exploratory_plots/tropomi_points_' + date + '.png')
#     plt.close()

#     # Plot "map" of OCO-2 data-points for this date
#     plt.figure(figsize=(24, 10))
#     scatterplot = plt.scatter(oco2_metadata_date['lon'], oco2_metadata_date['lat'], c=oco2_metadata_date['SIF'], cmap=sif_cmap, vmin=0.2, vmax=1.5)
#     plt.colorbar(scatterplot)
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title('OCO-2 points, date: ' + date)
#     plt.savefig('exploratory_plots/oco2_points_' + date + '.png')
#     plt.close()

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds, min_input=MIN_INPUT, max_input=MAX_INPUT))
transform = transforms.Compose(transform_list)

# Get selected tile
tile_dir = os.path.join(DATA_DIR, "tiles_" + SELECTED_DATE)
image_file = os.path.join(tile_dir, "reflectance_" + LAT_LON + ".npy")
selected_tile = np.load(image_file)
input_tile_non_standardized = torch.tensor(selected_tile, dtype=torch.float).unsqueeze(0).to(device)

# Visualize the input tile
array = selected_tile.transpose((1, 2, 0))
selected_rgb_tile = array[:, :, RGB_BANDS] / 1000
print('Array shape', array.shape)
plt.imshow(selected_rgb_tile)
plt.savefig("exploratory_plots/" + SELECTED_TILE_DESCRIPTION + "_rgb.png")
plt.close()

# Loop through each date
# for date in DATES:
#     # Read an input tile
#     tile_dir = os.path.join(DATA_DIR, "tiles_" + date)
#     image_file = os.path.join(tile_dir, "reflectance_" + LAT_LON + ".npy")
#     tile = np.load(image_file)
#     input_tile_non_standardized = torch.tensor(tile, dtype=torch.float).unsqueeze(0).to(device)
#     subtiles_non_standardized = get_subtiles_list(input_tile_non_standardized[0], SUBTILE_DIM, device, max_subtile_cloud_cover=None)
#     subtile_averages = torch.mean(subtiles_non_standardized, dim=(2,3))

#     # String to identify the tile (lat, lon, date)
#     tile_description = LAT_LON + '_' + date

#     # Visualize the input tile
#     array = tile.transpose((1, 2, 0))
#     rgb_tile = array[:, :, RGB_BANDS] / 1000
#     print('Array shape', array.shape)
#     plt.imshow(rgb_tile)
#     plt.savefig("exploratory_plots/" + tile_description + "_rgb.png")
#     plt.close()

#     # Visualize CDL
#     cdl_utils.plot_cdl_layers(tile[CDL_BANDS, :, :], "exploratory_plots/" + tile_description + "_cdl.png")

#     fig, axeslist = plt.subplots(ncols=6, nrows=8, figsize=(24, 24))
#     for band in range(0, 43):
#         layer = array[:, :, band]
#         axeslist.ravel()[band].imshow(layer, cmap='Greens', vmin=band_mins[INPUT_COLUMNS[band]], vmax=band_maxs[INPUT_COLUMNS[band]])
#         axeslist.ravel()[band].set_title('Band ' + str(band))
#         axeslist.ravel()[band].set_axis_off()
#     plt.tight_layout() # optional
#     plt.savefig('exploratory_plots/' + tile_description + '_all_bands.png')
#     plt.close()






# for crop_type in CROP_TYPES:
#     print('========= SIF for crop type:', crop_type, '==========')

#     #print('-------------- Large tiles (unfiltered)-------------')
#     #crop_type_sif_all = all_metadata[all_metadata[crop_type] > 0.5]
#     #if len(crop_type_sif_all) >= 5:
#     #    plot_histogram(np.array(crop_type_sif_all['SIF']), "sif_distribution_all_" + column + ".png")

#     print('-------------- Large tiles (train) -----------------')
#     crop_type_sif_train = train_metadata[train_metadata[crop_type] > 0.5]
#     if len(crop_type_sif_train) >= 5:
#         plot_histogram(np.array(crop_type_sif_train['SIF']), "sif_distribution_train_" + crop_type + ".png")

#     # print('-------------- CFIS sub-tiles ----------------------')
#     # crop_type_sif_cfis = cfis_metadata[cfis_metadata[crop_type] > 0.7]
#     # if len(crop_type_sif_cfis) >= 5:
#     #     plot_histogram(np.array(crop_type_sif_cfis['SIF']), "sif_distribution_cfis_" + crop_type + ".png")

#     print('-------------- OCO2 sub-tiles ----------------------')
#     crop_type_sif_oco2 = oco2_metadata[oco2_metadata[crop_type] > 0.5]
#     if len(crop_type_sif_oco2) >= 5:
#         plot_histogram(np.array(crop_type_sif_oco2['SIF']), "sif_distribution_oco2_" + crop_type + ".png")


# Display tiles with largest/smallest TROPOMI SIFs
highest_tropomi_sifs = train_tropomi_metadata.nlargest(25, 'SIF')
plot_rgb_images(highest_tropomi_sifs, 'tile_file', 'exploratory_plots/tropomi_sif_high_subtiles.png')
plot_band_images(highest_tropomi_sifs, 'tile_file', 'exploratory_plots/tropomi_sif_high_subtiles')
lowest_tropomi_sifs = train_tropomi_metadata.nsmallest(25, 'SIF')
plot_rgb_images(lowest_tropomi_sifs, 'tile_file', 'exploratory_plots/tropomi_sif_low_subtiles.png')
plot_band_images(lowest_tropomi_sifs, 'tile_file', 'exploratory_plots/tropomi_sif_low_subtiles')

# # Display tiles with largest/smallest CFIS SIFs
# highest_cfis_sifs = cfis_metadata.nlargest(25, 'SIF')
# plot_rgb_images(highest_cfis_sifs, 'subtile_file', 'exploratory_plots/cfis_sif_high_subtiles.png')
# plot_rgb_images(highest_cfis_sifs, 'subtile_file', 'exploratory_plots/cfis_sif_high_subtiles.png')
# lowest_cfis_sifs = cfis_metadata.nsmallest(25, 'SIF')
# plot_rgb_images(lowest_cfis_sifs, 'subtile_file', 'exploratory_plots/cfis_sif_low_subtiles.png')
# plot_rgb_images(lowest_cfis_sifs, 'subtile_file', 'exploratory_plots/cfis_sif_low_subtiles')
# print('Most common regions in CFIS:', cfis_metadata['tile_file'].value_counts())

# Display tiles with largest/smallest OCO-2 SIFs
highest_oco2_sifs = train_oco2_metadata.nlargest(25, 'SIF')
plot_rgb_images(highest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_high_subtiles.png')
plot_band_images(highest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_high_subtiles')
plot_cdl_layers(highest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_high_subtiles_cdl.png')
lowest_oco2_sifs = train_oco2_metadata.nsmallest(25, 'SIF')
print("========================= Lowest OCO2 SIF files =====================")
for filename in lowest_oco2_sifs['tile_file']:
    print(filename)

# pd.options.display.max_columns = None
plot_rgb_images(lowest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_low_subtiles.png')
plot_band_images(lowest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_low_subtiles')
plot_cdl_layers(lowest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_low_subtiles_cdl.png')




# Plot map of linear_regression's subtile SIF predictions
#plt.imshow(predicted_sifs_linear, cmap=sif_cmap, vmin=0.2, vmax=1.5)
#plt.savefig("exploratory_plots/" + LAT_LON + "_subtile_sif_map_linear.png")
#plt.close()

# Standardize input tile
input_tile_standardized = torch.tensor(transform(selected_tile), dtype=torch.float).to(device)
subtiles_standardized = get_subtiles_list(input_tile_standardized, SUBTILE_DIM, device, max_subtile_cloud_cover=None)
subtile_averages = torch.mean(subtiles_standardized, dim=(2,3))
print('Input tile dim', input_tile_standardized.shape)
print('Random pixel', input_tile_standardized[:, 8, 8])

# Train linear regression to predict SIF given band averages
X_train = train_metadata[INPUT_COLUMNS]
Y_train = train_metadata[OUTPUT_COLUMN].values.ravel()
linear_regression = LinearRegression().fit(X_train, Y_train)
predicted_sifs_linear = linear_regression.predict(subtile_averages.cpu().numpy()).reshape((37, 37))
mlp_regression = MLPRegressor(hidden_layer_sizes=(100, 100)).fit(X_train, Y_train)
predicted_sifs_mlp = mlp_regression.predict(subtile_averages.cpu().numpy()).reshape((37, 37))

# Obtain simple CNN model's subtile SIF predictions
subtiles_standardized = get_subtiles_list(input_tile_standardized, SUBTILE_DIM, device, max_subtile_cloud_cover=None)  # (batch x num subtiles x bands x subtile_dim x subtile_dim)
print('Subtile shape', subtiles_standardized.shape)
with torch.set_grad_enabled(False):
    predicted_sifs_simple_cnn_standardized = subtile_sif_model(subtiles_standardized[:, BANDS, :, :]).detach().numpy()
print('Subtile CNN: Predicted SIFs standardized', predicted_sifs_simple_cnn_standardized.shape)
predicted_sifs_simple_cnn_non_standardized = (predicted_sifs_simple_cnn_standardized * sif_std + sif_mean).reshape((37, 37))

# Plot map of CNN's subtile SIF predictions
#plt.imshow(predicted_sifs_simple_cnn_non_standardized, cmap=sif_cmap, vmin=0.2, vmax=1.5)
#plt.savefig("exploratory_plots/" + LAT_LON + "_subtile_sif_map_9.png")
#plt.close()

# Obtain tile2vec model's subtile SIF predictions
with torch.set_grad_enabled(False):
    embeddings = tile2vec_model(subtiles_standardized)
    predicted_sifs_tile2vec_fixed_standardized = embedding_to_sif_model(embeddings).detach().numpy()
print('Tile2Vec: Predicted SIFs standardized', predicted_sifs_tile2vec_fixed_standardized.shape)
predicted_sifs_tile2vec_fixed_non_standardized = (predicted_sifs_tile2vec_fixed_standardized * sif_std + sif_mean).reshape((37, 37))

# Plot map of CNN's subtile SIF predictions
#plt.imshow(predicted_sifs_tile2vec_fixed_non_standardized, cmap=sif_cmap, vmin=0.2, vmax=1.5)
#plt.savefig("exploratory_plots/" + LAT_LON + "_subtile_sif_tile2vec_fixed.png")
#plt.close()


# Obtain SAN model predictions
# _, _, _, _, predicted_sifs_san_37_standardized = san_model_37(input_tile_standardized.unsqueeze(0))
# predicted_sifs_san_37_standardized = predicted_sifs_san_37_standardized.detach().numpy()
# predicted_sifs_san_37 = (predicted_sifs_san_37_standardized * sif_std + sif_mean).reshape((37, 37))
# _, _, _, _, predicted_sifs_san_111_standardized = san_model_111(input_tile_standardized.unsqueeze(0))
# predicted_sifs_san_111_standardized = predicted_sifs_san_111_standardized.detach().numpy()
# predicted_sifs_san_111 = (predicted_sifs_san_37_standardized * sif_std + sif_mean).reshape((37, 37))

# _, _, _, _, predicted_sifs_san_74_standardized = san_model_74(input_tile_standardized.unsqueeze(0))
# predicted_sifs_san_74_standardized = predicted_sifs_san_74_standardized.detach().numpy()
# predicted_sifs_san_74 = (predicted_sifs_san_74_standardized * sif_std + sif_mean).reshape((37, 37))

# Obtain U-Net model predictions
unet_input = input_tile_standardized[UNET_BANDS].unsqueeze(0)  # Should be [1 x bands x 371 x 371]
print('UNet input shape', unet_input.shape)
predicted_sifs_unet_standardized = unet_model(unet_input).detach().numpy()
print('UNet prediction shape', predicted_sifs_unet_standardized.shape)
predicted_sifs_unet_non_standardized = (predicted_sifs_unet_standardized * sif_std + sif_mean).reshape((371, 371))

# Plot map of SAN's subtile SIF predictions
#plt.imshow(predicted_sifs_san, cmap=sif_cmap, vmin=0.2, vmax=1.5)
#plt.savefig("exploratory_plots/" + LAT_LON + "_subtile_sif_SAN.png")
#plt.close()


# Open CFIS SIF evaluation dataset
all_cfis_points = np.load(CFIS_SIF_FILE)
print("CFIS points total", all_cfis_points.shape[0])
print('CFIS points with reflectance data', len(cfis_metadata))

# Open TROPOMI SIF dataset
tropomi_dataset = xr.open_dataset(TROPOMI_SIF_FILE)
tropomi_array = tropomi_dataset.sif_dc.sel(time=slice(CFIS_DATE_RANGE[0], CFIS_DATE_RANGE[-1])).mean(dim='time')

# For each CFIS SIF point, find TROPOMI SIF of surrounding tile
tropomi_sifs_filtered_cfis = []  # TROPOMI SIF corresponding to each CFIS point
for i in range(len(cfis_metadata)):  # range(cfis_points.shape[0]):
    point_lon = cfis_metadata['lon'][i]  # cfis_points[i, 1]
    point_lat = cfis_metadata['lat'][i]  # cfis_points[i, 2]
    tropomi_sif = tropomi_array.sel(lat=point_lat, lon=point_lon, method='nearest')
    tropomi_sifs_filtered_cfis.append(tropomi_sif)

# For each CFIS SIF point, find TROPOMI SIF of surrounding tile
tropomi_sifs_all_cfis = []  # TROPOMI SIF corresponding to each CFIS point
for i in range(len(cfis_metadata)):  # range(cfis_points.shape[0]):
    point_lon = all_cfis_points[i, 1]  # cfis_points[i, 1]
    point_lat = all_cfis_points[i, 2]  # cfis_points[i, 2]
    tropomi_sif = tropomi_array.sel(lat=point_lat, lon=point_lon, method='nearest')
    tropomi_sifs_all_cfis.append(tropomi_sif)

# Plot histogram of CFIS and TROPOMI SIFs
plot_histogram(np.array(all_cfis_points[:, 0]), "sif_distribution_cfis_all.png", title="CFIS SIF distribution (all)")
plot_histogram(np.array(cfis_metadata['SIF']), "sif_distribution_cfis_filtered.png", title="CFIS SIF distribution (filtered)") #  cfis_points[:, 0])
plot_histogram(np.array(train_oco2_metadata['SIF']), "sif_distribution_oco2.png", title="OCO2 SIF distribution (filtered)") #  cfis_points[:, 0])
plot_histogram(np.array(tropomi_sifs_all_cfis), "sif_distribution_tropomi_all_cfis_area.png", title="TROPOMI SIF distribution (regions overlapping with all CFIS)")
plot_histogram(np.array(tropomi_sifs_filtered_cfis), "sif_distribution_tropomi_eval_area.png", title="TROPOMI SIF distribution (regions overlapping with filtered CFIS)")
plot_histogram(np.array(train_metadata['SIF']), "sif_distribution_tropomi_train.png", title="TROPOMI SIF distribution (train set)")
#plot_histogram(np.array(all_metadata['SIF']), "sif_distribution_tropomi_all.png", title="TROPOMI SIF distribution (longitude: -108 to -82, latitude: 38 to 48.7)")

# sif_mean = np.mean(train_metadata['SIF'])
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
sif_mean = train_statistics['mean'].values[-1]
print('SIF mean (TROPOMI, train set)', sif_mean)

# Scatterplot of CFIS points (all)
plt.figure(figsize=(24, 24))
scatterplot = plt.scatter(all_cfis_points[:, 1], all_cfis_points[:, 2], c=all_cfis_points[:, 0], cmap=sif_cmap, vmin=0.2, vmax=1.5)
plt.colorbar(scatterplot)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('CFIS points (all)')
plt.savefig('exploratory_plots/cfis_points_all.png')
plt.close()

# Scatterplot of CFIS points (eval)
plt.figure(figsize=(24, 24))
scatterplot = plt.scatter(cfis_metadata['lon'], cfis_metadata['lat'], c=cfis_metadata['SIF'], cmap=sif_cmap, vmin=0.2, vmax=1.5)
plt.colorbar(scatterplot)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('CFIS points (reflectance data available, eval set)')
plt.savefig('exploratory_plots/cfis_points_filtered.png')
plt.close()

# Scatterplot of CFIS points in the dense area
plt.figure(figsize=(24, 24))
cfis_dense_area = all_cfis_points[all_cfis_points[:, 1] > -94]
plt.scatter(cfis_dense_area[:, 1], cfis_dense_area[:, 2], c=cfis_dense_area[:, 0], cmap=sif_cmap, vmin=0.2, vmax=1.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('CFIS points (area)')
plt.savefig('exploratory_plots/cfis_points_dense.png')
plt.close()

# Scatterplot of CFIS points in the particular area
plt.figure(figsize=(24, 24))
#cfis_area = all_cfis_points[(all_cfis_points[:, 1] > LON-eps) & (all_cfis_points[:, 1] < LON+eps) & (all_cfis_points[:, 2] > LAT-eps) & (all_cfis_points[:, 2] < LAT+eps)]
cfis_area = cfis_metadata.loc[(cfis_metadata['lon'] >= LON-eps) & (cfis_metadata['lon'] <= LON+eps) & (cfis_metadata['lat'] >= LAT-eps) & (cfis_metadata['lat'] <= LAT+eps)]
#plt.scatter(cfis_area['lon'], cfis_area['lat'], c=cfis_area['SIF'], cmap=sif_cmap, vmin=0.2, vmax=1.5)
# plt.scatter(cfis_area[:, 1], cfis_area[:, 2], c=cfis_area[:, 0], cmap=green_cmap, vmin=0, vmax=1.5)
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.title('CFIS points (area)')
#plt.savefig('exploratory_plots/' + LAT_LON + '_cfis_points.png')
#plt.close()

# Convert CFIS into matrix
cfis_tile = np.empty(predicted_sifs_simple_cnn_non_standardized.shape)
cfis_tile[:] = np.NaN
print('CFIS tile shape', cfis_tile.shape)
top_bound = LAT + eps
left_bound = LON - eps
for index, row in cfis_area.iterrows():
    res = (TILE_DEGREES / cfis_tile.shape[0], TILE_DEGREES / cfis_tile.shape[1])
    height_idx, width_idx = lat_long_to_index(row['lat'], row['lon'], top_bound, left_bound, res)
    # height_idx, width_idx = lat_long_to_index(cfis_area[p, 2], cfis_area[p, 1], top_bound, left_bound, res)
    cfis_tile[height_idx, width_idx] = row['SIF']  #p, 0] # * 1.52

#plt.imshow(cfis_tile, cmap=sif_cmap, vmin=0.2, vmax=1.5)
#plt.savefig("exploratory_plots/" + LAT_LON + "_cfis_sifs.png")
#plt.close()

# Scatterplot of TROPOMI points: train, val
plt.figure(figsize=(24, 24))
plt.scatter(train_metadata['lon'], train_metadata['lat'], c=train_metadata['SIF'], cmap=sif_cmap, vmin=0.2, vmax=1.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('TROPOMI points (train)')
plt.savefig('exploratory_plots/tropomi_points_train.png')
plt.close()
plt.figure(figsize=(24, 24))
plt.scatter(val_metadata['lon'], val_metadata['lat'], c=val_metadata['SIF'], cmap=sif_cmap, vmin=0.2, vmax=1.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('TROPOMI points (val)')
plt.savefig('exploratory_plots/tropomi_points_val.png')
plt.close()

# # Plot different versions of SAN
# fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(16, 16))
# axeslist[0 ,0].imshow(rgb_tile)
# axeslist[0, 0].set_title('RGB Bands')
# axeslist[0, 1].imshow(predicted_sifs_san_37, cmap=sif_cmap, vmin=0.2, vmax=1.7)
# axeslist[0, 1].set_title('SAN (37 x 37)')
# axeslist[1, 0].imshow(predicted_sifs_san_74, cmap=sif_cmap, vmin=0.2, vmax=1.7)
# axeslist[1, 0].set_title('SAN (74 x 74)')
# pcm = axeslist[1, 1].imshow(predicted_sifs_san_111, cmap=sif_cmap, vmin=0.2, vmax=1.7)
# axeslist[1, 1].set_title('SAN (111 x 111)')
# fig.colorbar(pcm, ax=axeslist.ravel().tolist()) #[:, 2]) #, shrink=0.6)
# plt.savefig('exploratory_plots/SAN_versions.png')
# plt.close()


# Plot different method's predictions
fig, axeslist = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))
axeslist[0 ,0].imshow(selected_rgb_tile)
axeslist[0, 0].set_title('RGB Bands')
axeslist[0, 1].imshow(predicted_sifs_linear, cmap=sif_cmap, vmin=0.2, vmax=1.7)
axeslist[0, 1].set_title('Linear Regression: predicted SIF')
axeslist[0, 2].imshow(predicted_sifs_mlp, cmap=sif_cmap, vmin=0.2, vmax=1.7)
axeslist[0, 2].set_title('Average ANN: predicted SIF')
axeslist[1, 0].imshow(cfis_tile, cmap=sif_cmap, vmin=0.2, vmax=1.7)
axeslist[1, 0].set_title('Ground-truth CFIS SIF')
axeslist[1, 1].imshow(predicted_sifs_simple_cnn_non_standardized, cmap=sif_cmap, vmin=0.2, vmax=1.7)
axeslist[1, 1].set_title('Sub-tile CNN: predicted SIF')
pcm = axeslist[1, 2].imshow(predicted_sifs_unet_non_standardized, cmap=sif_cmap, vmin=0.2, vmax=1.7)
axeslist[1, 2].set_title('U-Net: predicted SIF')

fig.colorbar(pcm, ax=axeslist.ravel().tolist()) #[:, 2]) #, shrink=0.6)
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(pcm, cax=cbar_ax)

#plt.tight_layout() # optional
plt.savefig('exploratory_plots/' + SELECTED_TILE_DESCRIPTION + '_compare_predictions.png')
plt.close()
# Compare stats
# predicted_sifs_simple_cnn_non_standardized = np.clip(predicted_sifs_simple_cnn_non_standardized, a_min=0.2, a_max=1.7)
# print('===================== Comparing stats ======================')
# print('Linear predictions for this tile: mean', round(np.mean(predicted_sifs_linear), 3), 'std', round(np.std(predicted_sifs_linear), 3)) #, 'min', np.min(predicted_sifs_linear), 'max', np.max(predicted_sifs_linear))
# print('CNN predictions for this tile: mean', round(np.mean(predicted_sifs_simple_cnn_non_standardized), 3), 'std', round(np.std(predicted_sifs_simple_cnn_non_standardized), 3)) #'min', np.min(predicted_sifs_simple_cnn_non_standardized), 'max', np.max(predicted_sifs_simple_cnn_non_standardized))
# print('Tile2Vec Fixed predictions for this tile: mean', round(np.mean(predicted_sifs_tile2vec_fixed_non_standardized), 3), 'std', round(np.std(predicted_sifs_tile2vec_fixed_non_standardized), 3)) # 'min', np.min(predicted_sifs_tile2vec_fixed_non_standardized), 'max', np.max(predicted_sifs_tile2vec_fixed_non_standardized))
# print('SAN predictions for this tile: mean', round(np.mean(predicted_sifs_san_111), 3), 'std', round(np.std(predicted_sifs_san_111), 3)) # 'min', np.min(predicted_sifs_san), 'max', np.max(predicted_sifs_san))
# print('Ground-truth CFIS SIF for this tile: mean', round(np.mean(cfis_area['SIF']), 3), 'std', round(np.std(cfis_area['SIF']), 3)) # 'min', np.min(cfis_area['SIF']), 'max', np.max(cfis_area['SIF']))
# print('Linear / Ground-truth', round(np.mean(predicted_sifs_linear) / np.mean(cfis_area['SIF']), 3))
# print('TROPOMI SIF for this tile', tropomi_array.sel(lat=LAT, lon=LON, method='nearest'))
# print('============================================================')

# Plot TROPOMI vs SIF (and linear regression)
# x = cfis_metadata['SIF']  # cfis_points[:, 0]
# y = tropomi_sifs
# coef = np.polyfit(x, y, 1)
# print('Linear regression: x=CFIS, y=TROPOMI', coef)
# poly1d_fn = np.poly1d(coef) 
# plt.plot(x, y, 'bo', x, poly1d_fn(x), '--k')
# plt.xlabel('CFIS SIF (small tile, 2016)')
# plt.ylabel('TROPOMI SIF (surrounding large tile, 2018)')
# plt.title('TROPOMI vs CFIS SIF')
# plt.savefig('exploratory_plots/TROPOMI_vs_CFIS_SIF')
# plt.close()

# # Calculate NRMSE and correlation
# nrmse = math.sqrt(mean_squared_error(y, x)) / sif_mean
# corr, _ = pearsonr(y, x)
# print('NRMSE', round(nrmse, 3))
# print('Correlation', round(corr, 3))

