import math
import matplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torchvision.transforms as transforms
import xarray as xr
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from numpy import poly1d

from sif_utils import lat_long_to_index, plot_histogram, get_subtiles_list, plot_tile
import cdl_utils
import simple_cnn
import tile_transforms
import resnet
from SAN import SAN
from unet.unet_model import UNet, UNetSmall, UNet2_NoBilinear

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
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_all_2")
TRAIN_TILE_DATASET = os.path.join(DATASET_DIR, "tile_info_train.csv")
VAL_TILE_DATASET = os.path.join(DATASET_DIR, "tile_info_val.csv")
CFIS_SUBTILE_DATASET = os.path.join(DATASET_DIR, "cfis_subtiles_filtered.csv")

BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")

DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
         "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
         "2018-09-16"]

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
TILE_DEGREES = 0.1
eps = TILE_DEGREES / 2

CFIS_SIF_FILE = os.path.join(DATA_DIR, "CFIS/CFIS_201608a_300m_soundings.npy")
TROPOMI_SIF_FILE = os.path.join(DATA_DIR, "TROPOMI_SIF/TROPO-SIF_01deg_biweekly_Apr18-Jan20.nc")
OCO2_SIF_FILE = os.path.join(DATA_DIR, "OCO2/oco2_20180708_20180915_14day_3km.nc")


RGB_BANDS = [3, 2, 1]
CDL_BANDS = list(range(12, 42))
SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/2d_train_both_subtile_resnet") # "models/AUG_subtile_simple_cnn_v5")
UNET_MODEL_FILE = os.path.join(DATA_DIR, "models/7_unet2_nobilinear")

EMBEDDING_TYPE = 'tile2vec'
Z_DIM = 256
HIDDEN_DIM = 1024

# BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42] #
BANDS = list(range(0, 43))
INPUT_CHANNELS = len(BANDS)
UNET_BANDS = list(range(0, 43)) #list(range(0, 12)) + list(range(12, 27)) + [28] + [42]
INPUT_CHANNELS_UNET = len(UNET_BANDS)
REDUCED_CHANNELS = 15
SUBTILE_DIM = 100
TILE_SIZE_DEGREES = 0.1
INPUT_SIZE = 371
MIN_INPUT = -1000
MAX_INPUT = 1000
NUM_OCO2_SAMPLES = 500
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

# Compute bounds on standardized SIF
MIN_SIF = 0
MAX_SIF = 1.7
min_output = (MIN_SIF - sif_mean) / sif_std
max_output = (MAX_SIF - sif_mean) / sif_std


# Load subtile SIF model
subtile_sif_model = resnet.resnet18(input_channels=INPUT_CHANNELS, num_classes=1,
                                    min_output=min_output, max_output=max_output).to(device)
subtile_sif_model.load_state_dict(torch.load(SUBTILE_SIF_MODEL_FILE, map_location=device))
subtile_sif_model.eval()

# Load UNet model
unet_model = UNet2_NoBilinear(n_channels=INPUT_CHANNELS_UNET, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
unet_model.load_state_dict(torch.load(UNET_MODEL_FILE, map_location=device))
unet_model.eval()


# Colormap for SIF
sif_cmap = plt.get_cmap('YlGn')
sif_cmap.set_bad(color='red')

# Load datasets
train_set = pd.read_csv(TRAIN_TILE_DATASET)
train_tropomi_set = train_set[train_set['source'] == 'TROPOMI'].copy()
train_oco2_set = train_set[train_set['source'] == 'OCO2'].copy().iloc[0:NUM_OCO2_SAMPLES]
val_set = pd.read_csv(VAL_TILE_DATASET)
val_tropomi_set = val_set[val_set['source'] == 'TROPOMI'].copy()
val_oco2_set = val_set[val_set['source'] == 'OCO2'].copy()
cfis_set = pd.read_csv(CFIS_SUBTILE_DATASET)

# Only using TROPOMI to train
train_set = train_tropomi_set

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




# Standardize data
for column in COLUMNS_TO_STANDARDIZE:
    column_mean = train_set[column].mean()
    column_std = train_set[column].std()
    train_set[column] = np.clip((train_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
    val_set[column] = np.clip((val_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
    val_tropomi_set[column] = np.clip((val_tropomi_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
    val_oco2_set[column] = np.clip((val_oco2_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)
    cfis_set[column] = np.clip((cfis_set[column] - column_mean) / column_std, a_min=MIN_INPUT, a_max=MAX_INPUT)

    # train_set[column] = (train_set[column] - column_mean) / column_std
    # val_tropomi_set[column] = (val_tropomi_set[column] - column_mean) / column_std
    # val_oco2_set[column] = (val_oco2_set[column] - column_mean) / column_std
    # cfis_set[column] = (cfis_set[column] - column_mean) / column_std

band_maxs = train_set.max(axis=0)
band_mins = train_set.min(axis=0)
train_set_august = train_set[train_set['date'] == '2018-08-05'].copy()
val_oco2_set_august = val_oco2_set[val_oco2_set['date'] == '2018-08-05'].copy()

# Plot histogram of each band (TROPOMI and OCO2, all time periods)
for column in COLUMNS_TO_STANDARDIZE + OUTPUT_COLUMN:
    print('============== ALL DATES - Column:', column, '==================')
    train_tropomi_column = np.array(train_set_august[column])
    print('-------------- Train TROPOMI ----------------------')
    plot_histogram(train_tropomi_column, "histogram_aug_" + column + "_train_tropomi.png", title=column + ' (TROPOMI, 2018-08-05)') #, range=(band_mins[column], band_maxs[column]), title="All dates (train TROPOMI): "+column)

    val_oco2_column = np.array(val_oco2_set_august[column])
    print('-------------- Val OCO-2 ----------------------')
    plot_histogram(val_oco2_column, "histogram_aug_" + column + "_val_oco2.png", title=column + ' (OCO-2, 2018-08-05)') #, range=(band_mins[column], band_maxs[column]), title="All dates (val OCO2): "+column)

    print('-------------- CFIS sub-tiles ----------------------')
    cfis_column = np.array(cfis_set[column])
    plot_histogram(cfis_column, "histogram_aug_" + column + "_cfis.png", title=column + ' (CFIS, 2016-08-01)')

exit(0)

# # Plot histogram of each band (TROPOMI and OCO2, all time periods)
# train_tropomi_stats = []
# val_tropomi_stats = []
# val_oco2_stats = []
# for column in INPUT_COLUMNS + OUTPUT_COLUMN:
#     print('============== ALL DATES - Column:', column, '==================')
#     train_tropomi_column = np.array(train_set[column])
#     train_tropomi_stats.append([round(np.mean(train_tropomi_column), 3), round(np.std(train_tropomi_column), 3)])
#     print('-------------- Train TROPOMI ----------------------')
#     plot_histogram(train_tropomi_column, "histogram_all_dates_" + column + "_train_tropomi.png", range=(band_mins[column], band_maxs[column]), title="All dates (train TROPOMI): "+column)

#     val_tropomi_column = np.array(val_tropomi_set[column])
#     val_tropomi_stats.append([round(np.mean(val_tropomi_column), 3), round(np.std(val_tropomi_column), 3)])
#     print('-------------- Val TROPOMI ----------------------')
#     plot_histogram(val_tropomi_column, "histogram_all_dates_" + column + "_val_tropomi.png", range=(band_mins[column], band_maxs[column]), title="All dates (val TROPOMI): "+column)

#     val_oco2_column = np.array(val_oco2_set[column])
#     val_oco2_stats.append([round(np.mean(val_oco2_column), 3), round(np.std(val_oco2_column), 3)])
#     print('-------------- Val OCO-2 ----------------------')
#     plot_histogram(val_oco2_column, "histogram_all_dates_" + column + "_val_oco2.png", range=(band_mins[column], band_maxs[column]), title="All dates (val OCO2): "+column)

#     # print('-------------- CFIS sub-tiles ----------------------')
#     # plot_histogram(np.array(cfis_set[column]), "histogram_cfis_" + column + ".png")

# # Print band means/stds across all dates, for both TROPOMI and OCO-2
# print('============== ALL DATES - Train TROPOMI summary =====================')
# for i in range(len(train_tropomi_stats)):
#     print(train_tropomi_stats[i][0])
# print('============== ALL DATES - Val TROPOMI summary =====================')
# for i in range(len(val_tropomi_stats)):
#     print(val_tropomi_stats[i][0])
# print('============== ALL DATES - Val OCO2 summary =====================')
# for i in range(len(val_oco2_stats)):
#     print(val_oco2_stats[i][0])
# print('===================================================================')


# For each date range, plot histogram of distribution of each band
column_averages = dict()
for column in COLUMNS_TO_STANDARDIZE + OUTPUT_COLUMN:
    column_averages[column] = []

for date in DATES:
    train_tropomi_date = train_set[train_set['date'] == date]
    val_tropomi_date = val_tropomi_set[val_tropomi_set['date'] == date]
    val_oco2_date = val_oco2_set[val_oco2_set['date'] == date]
    train_tropomi_stats = []
    val_tropomi_stats = []
    val_oco2_stats = []

    for column in COLUMNS_TO_STANDARDIZE + OUTPUT_COLUMN:
        print('==============  DATE', date, '- Column:', column, '==================')
        train_tropomi_column = np.array(train_tropomi_date[column])
        column_averages[column].append(round(np.mean(train_tropomi_column), 3))

        # train_tropomi_stats.append([round(np.mean(train_tropomi_column), 3), round(np.std(train_tropomi_column), 3)])
        print('-------------- Train TROPOMI ----------------------')
        plot_histogram(train_tropomi_column, "histogram_" + column + "_train_tropomi_" + date + ".png", range=(band_mins[column], band_maxs[column]), title=date + " (train TROPOMI): "+column)

        val_tropomi_column = np.array(val_tropomi_date[column])
        # val_tropomi_stats.append([round(np.mean(val_tropomi_column), 3), round(np.std(val_tropomi_column), 3)])
        print('-------------- Val TROPOMI ----------------------')
        plot_histogram(val_tropomi_column, "histogram_" + column + "_val_tropomi_" + date + ".png", range=(band_mins[column], band_maxs[column]), title=date + " (val TROPOMI): "+column)

        val_oco2_column = np.array(val_oco2_date[column])
        # val_oco2_stats.append([round(np.mean(val_oco2_column), 3), round(np.std(val_oco2_column), 3)])
        print('-------------- Val OCO-2 ----------------------')
        plot_histogram(val_oco2_column, "histogram_" + column + "_val_oco2_" + date + ".png", range=(band_mins[column], band_maxs[column]), title=date + " (val OCO2): " +column)

    # Plot "map" of TROPOMI data-points for this date
    plt.figure(figsize=(30, 10))
    scatterplot = plt.scatter(train_tropomi_date['lon'], train_tropomi_date['lat'], c=train_tropomi_date['SIF'], cmap=sif_cmap, vmin=0.2, vmax=1.5)
    plt.colorbar(scatterplot)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Train TROPOMI points, date: ' + date)
    plt.savefig('exploratory_plots/train_tropomi_points_sif_' + date + '.png')
    plt.close()

    # Plot "map" of OCO-2 data-points for this date
    plt.figure(figsize=(30, 10))
    scatterplot = plt.scatter(val_oco2_date['lon'], val_oco2_date['lat'], c=val_oco2_date['SIF'], cmap=sif_cmap, vmin=0.2, vmax=1.5)
    plt.colorbar(scatterplot)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Val OCO-2 points, date: ' + date)
    plt.savefig('exploratory_plots/val_oco2_points_sif_' + date + '.png')
    plt.close()

    # Plot temperatures of TROPOMI data-points for this date
    plt.figure(figsize=(30, 10))
    scatterplot = plt.scatter(train_tropomi_date['lon'], train_tropomi_date['lat'], c=train_tropomi_date['Tair_f_tavg'], cmap='Reds', vmin=-3, vmax=3)
    plt.colorbar(scatterplot)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Train TROPOMI points, date: ' + date)
    plt.savefig('exploratory_plots/train_tropomi_points_temp_' + date + '.png')
    plt.close()

    # Plot temperatures of OCO-2 data-points for this date
    plt.figure(figsize=(30, 10))
    scatterplot = plt.scatter(val_oco2_date['lon'], val_oco2_date['lat'], c=val_oco2_date['Tair_f_tavg'], cmap='Reds', vmin=-3, vmax=3)
    plt.colorbar(scatterplot)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Val OCO-2 points, date: ' + date)
    plt.savefig('exploratory_plots/val_oco2_points_temp_' + date + '.png')
    plt.close()
# For each feature, track progression over time (train TROPOMI)
print('================= Feature averages over time =====================')
for column, averages in column_averages.items():
    print('Column ' + column + ': ' + str(averages))
print('==================================================================')

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds, min_input=MIN_INPUT, max_input=MAX_INPUT))
transform = transforms.Compose(transform_list)


# Future: SIF per crop type / date combined?
# for crop_type in CROP_TYPES:
#     print('========= SIF for crop type:', crop_type, '==========')

#     #print('-------------- Large tiles (unfiltered)-------------')
#     #crop_type_sif_all = all_metadata[all_metadata[crop_type] > 0.5]
#     #if len(crop_type_sif_all) >= 5:
#     #    plot_histogram(np.array(crop_type_sif_all['SIF']), "sif_distribution_all_" + column + ".png")

#     print('-------------- Large tiles (train) -----------------')
#     crop_type_sif_train = train_set[train_set[crop_type] > 0.5]
#     if len(crop_type_sif_train) >= 5:
#         plot_histogram(np.array(crop_type_sif_train['SIF']), "sif_distribution_train_" + crop_type + ".png")

#     # print('-------------- CFIS sub-tiles ----------------------')
#     # crop_type_sif_cfis = cfis_set[cfis_set[crop_type] > 0.7]
#     # if len(crop_type_sif_cfis) >= 5:
#     #     plot_histogram(np.array(crop_type_sif_cfis['SIF']), "sif_distribution_cfis_" + crop_type + ".png")

#     print('-------------- OCO2 sub-tiles ----------------------')
#     crop_type_sif_oco2 = oco2_metadata[oco2_metadata[crop_type] > 0.5]
#     if len(crop_type_sif_oco2) >= 5:
#         plot_histogram(np.array(crop_type_sif_oco2['SIF']), "sif_distribution_oco2_" + crop_type + ".png")

# Loop through each date
for date in DATES:
    # Read an input tile
    tile_dir = os.path.join(DATA_DIR, "tiles_" + date)
    image_file = os.path.join(tile_dir, "reflectance_" + LAT_LON + ".npy")
    tile = np.load(image_file)
    input_tile_standardized = transform(tile)

    # String to identify the tile (lat, lon, date)
    tile_description = LAT_LON + '_' + date

    # Plot tile
    plot_tile(input_tile_standardized, tile_description)

    # Plot shrunk tile
    shrink_transform = transforms.Compose([tile_transforms.ShrinkTile(target_dim=100)])
    shrunk_tile = shrink_transform(input_tile_standardized)
    plot_tile(shrunk_tile, tile_description + '_shrunk')

    # Compute RGB tile again
    array = input_tile_standardized.transpose((1, 2, 0))
    rgb_tile = (array[:, :, RGB_BANDS] + 3) / 6

    # Get subtiles and subtile averages
    subtiles_standardized = get_subtiles_list(input_tile_standardized, SUBTILE_DIM)  # (num subtiles x bands x subtile_dim x subtile_dim)
    print('subtiles shape', subtiles_standardized.shape)
    subtile_averages = np.mean(subtiles_standardized, axis=(2,3))

    # Get pixels
    pixels = np.moveaxis(input_tile_standardized, 0, -1)
    pixels = pixels.reshape((-1, pixels.shape[2]))
    print('pixels shape', pixels.shape)

    # Train linear regression / GB to predict SIF given band averages
    X_train = train_set[INPUT_COLUMNS]
    Y_train = train_set[OUTPUT_COLUMN].values.ravel()
    linear_regression = LinearRegression().fit(X_train, Y_train)
    predicted_sifs_linear = linear_regression.predict(subtile_averages).reshape((4, 4))
    predicted_pixel_sifs_linear = linear_regression.predict(pixels).reshape((371, 371))
    gradient_boosting_regressor = HistGradientBoostingRegressor(max_iter=1000).fit(X_train, Y_train)
    predicted_sifs_gb = gradient_boosting_regressor.predict(subtile_averages).reshape((4, 4))
    predicted_pixel_sifs_gb = gradient_boosting_regressor.predict(pixels).reshape((371, 371))

    # Obtain simple CNN model's subtile SIF predictions
    print('Input tile dim', input_tile_standardized.shape)
    print('Random pixel', input_tile_standardized[:, 8, 8])
    print('Subtile shape', subtiles_standardized.shape)
    with torch.set_grad_enabled(False):
        subtiles_tensor = torch.tensor(subtiles_standardized[:, BANDS, :, :], dtype=torch.float)
        predicted_sifs_simple_cnn_standardized = subtile_sif_model(subtiles_tensor).detach().numpy()
    print('Subtile CNN: Predicted SIFs standardized', predicted_sifs_simple_cnn_standardized.shape)
    predicted_sifs_simple_cnn_non_standardized = (predicted_sifs_simple_cnn_standardized * sif_std + sif_mean).reshape((4, 4))

    # Obtain U-Net model predictions
    unet_input = torch.tensor(input_tile_standardized[UNET_BANDS], dtype=torch.float).unsqueeze(0)  # Should be [1 x bands x 371 x 371]
    print('UNet input shape', unet_input.shape)
    predicted_sifs_unet_standardized = unet_model(unet_input).detach().numpy()
    print('UNet prediction shape', predicted_sifs_unet_standardized.shape)
    predicted_sifs_unet_non_standardized = (predicted_sifs_unet_standardized * sif_std + sif_mean).reshape((371, 371))


    # TODO Compute model predictions...
    # Plot different method's predictions
    fig, axeslist = plt.subplots(ncols=4, nrows=2, figsize=(24, 12))
    axeslist[0, 0].imshow(rgb_tile)
    axeslist[0, 0].set_title('RGB Bands')
    axeslist[0, 1].imshow(predicted_sifs_linear, cmap=sif_cmap, vmin=0.2, vmax=1.7)
    axeslist[0, 1].set_title('Linear Regression: predicted SIF (sub-tile)')
    axeslist[0, 2].imshow(predicted_sifs_gb, cmap=sif_cmap, vmin=0.2, vmax=1.7)
    axeslist[0, 2].set_title('Gradient Boosting Regressor: predicted SIF (sub-tile)')
    axeslist[0, 3].imshow(predicted_sifs_simple_cnn_non_standardized, cmap=sif_cmap, vmin=0.2, vmax=1.7)
    axeslist[0, 3].set_title('Sub-tile CNN: predicted SIF')
    axeslist[1, 0].imshow(predicted_sifs_unet_non_standardized, cmap=sif_cmap, vmin=0.2, vmax=1.7)
    axeslist[1, 0].set_title('U-Net predicted SIF')
    axeslist[1, 1].imshow(predicted_pixel_sifs_linear, cmap=sif_cmap, vmin=0.2, vmax=1.7)
    axeslist[1, 1].set_title('Linear regression: predicted pixel SIFs')
    pcm = axeslist[1, 2].imshow(predicted_pixel_sifs_gb, cmap=sif_cmap, vmin=0.2, vmax=1.7)
    axeslist[1, 2].set_title('Gradient Boosting Regressor: predicted pixel SIFs')
    fig.colorbar(pcm, ax=axeslist.ravel().tolist())
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(pcm, cax=cbar_ax)

    #plt.tight_layout() # optional
    plt.savefig('exploratory_plots/' + tile_description + '_compare_predictions.png')
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




# Display tiles with largest/smallest TROPOMI SIFs
highest_tropomi_sifs = train_tropomi_set.nlargest(25, 'SIF')
plot_rgb_images(highest_tropomi_sifs, 'tile_file', 'exploratory_plots/tropomi_sif_high_subtiles.png')
plot_band_images(highest_tropomi_sifs, 'tile_file', 'exploratory_plots/tropomi_sif_high_subtiles')
lowest_tropomi_sifs = train_tropomi_set.nsmallest(25, 'SIF')
plot_rgb_images(lowest_tropomi_sifs, 'tile_file', 'exploratory_plots/tropomi_sif_low_subtiles.png')
plot_band_images(lowest_tropomi_sifs, 'tile_file', 'exploratory_plots/tropomi_sif_low_subtiles')

# Display tiles with largest/smallest OCO-2 SIFs
highest_oco2_sifs = train_oco2_set.nlargest(25, 'SIF')
plot_rgb_images(highest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_high_subtiles.png')
plot_band_images(highest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_high_subtiles')
plot_cdl_layers(highest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_high_subtiles_cdl.png')
lowest_oco2_sifs = train_oco2_set.nsmallest(25, 'SIF')
plot_rgb_images(lowest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_low_subtiles.png')
plot_band_images(lowest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_low_subtiles')
plot_cdl_layers(lowest_oco2_sifs, 'tile_file', 'exploratory_plots/oco2_sif_low_subtiles_cdl.png')

exit(0)
# print("========================= Lowest OCO2 SIF files =====================")
# for filename in lowest_oco2_sifs['tile_file']:
#     print(filename)

# pd.options.display.max_columns = None

# # Display tiles with largest/smallest CFIS SIFs
# highest_cfis_sifs = cfis_set.nlargest(25, 'SIF')
# plot_rgb_images(highest_cfis_sifs, 'subtile_file', 'exploratory_plots/cfis_sif_high_subtiles.png')
# plot_rgb_images(highest_cfis_sifs, 'subtile_file', 'exploratory_plots/cfis_sif_high_subtiles.png')
# lowest_cfis_sifs = cfis_set.nsmallest(25, 'SIF')
# plot_rgb_images(lowest_cfis_sifs, 'subtile_file', 'exploratory_plots/cfis_sif_low_subtiles.png')
# plot_rgb_images(lowest_cfis_sifs, 'subtile_file', 'exploratory_plots/cfis_sif_low_subtiles')
# print('Most common regions in CFIS:', cfis_set['tile_file'].value_counts())




# Open CFIS SIF evaluation dataset
all_cfis_points = np.load(CFIS_SIF_FILE)
print("CFIS points total", all_cfis_points.shape[0])
print('CFIS points with reflectance data', len(cfis_set))

# Open TROPOMI SIF dataset
tropomi_dataset = xr.open_dataset(TROPOMI_SIF_FILE)
tropomi_array = tropomi_dataset.sif_dc.sel(time=slice(CFIS_DATE_RANGE[0], CFIS_DATE_RANGE[-1])).mean(dim='time')

# For each CFIS SIF point, find TROPOMI SIF of surrounding tile
tropomi_sifs_filtered_cfis = []  # TROPOMI SIF corresponding to each CFIS point
for i in range(len(cfis_set)):  # range(cfis_points.shape[0]):
    point_lon = cfis_set['lon'][i]  # cfis_points[i, 1]
    point_lat = cfis_set['lat'][i]  # cfis_points[i, 2]
    tropomi_sif = tropomi_array.sel(lat=point_lat, lon=point_lon, method='nearest')
    tropomi_sifs_filtered_cfis.append(tropomi_sif)

# For each CFIS SIF point, find TROPOMI SIF of surrounding tile
tropomi_sifs_all_cfis = []  # TROPOMI SIF corresponding to each CFIS point
for i in range(len(cfis_set)):  # range(cfis_points.shape[0]):
    point_lon = all_cfis_points[i, 1]  # cfis_points[i, 1]
    point_lat = all_cfis_points[i, 2]  # cfis_points[i, 2]
    tropomi_sif = tropomi_array.sel(lat=point_lat, lon=point_lon, method='nearest')
    tropomi_sifs_all_cfis.append(tropomi_sif)

# Plot histogram of CFIS and TROPOMI SIFs
plot_histogram(np.array(all_cfis_points[:, 0]), "sif_distribution_cfis_all.png", title="CFIS SIF distribution (all)")
plot_histogram(np.array(cfis_set['SIF']), "sif_distribution_cfis_filtered.png", title="CFIS SIF distribution (filtered)") #  cfis_points[:, 0])
plot_histogram(np.array(train_oco2_metadata['SIF']), "sif_distribution_oco2.png", title="OCO2 SIF distribution (filtered)") #  cfis_points[:, 0])
plot_histogram(np.array(tropomi_sifs_all_cfis), "sif_distribution_tropomi_all_cfis_area.png", title="TROPOMI SIF distribution (regions overlapping with all CFIS)")
plot_histogram(np.array(tropomi_sifs_filtered_cfis), "sif_distribution_tropomi_eval_area.png", title="TROPOMI SIF distribution (regions overlapping with filtered CFIS)")
plot_histogram(np.array(train_set['SIF']), "sif_distribution_tropomi_train.png", title="TROPOMI SIF distribution (train set)")
#plot_histogram(np.array(all_metadata['SIF']), "sif_distribution_tropomi_all.png", title="TROPOMI SIF distribution (longitude: -108 to -82, latitude: 38 to 48.7)")

# sif_mean = np.mean(train_set['SIF'])
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
scatterplot = plt.scatter(cfis_set['lon'], cfis_set['lat'], c=cfis_set['SIF'], cmap=sif_cmap, vmin=0.2, vmax=1.5)
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
cfis_area = cfis_set.loc[(cfis_set['lon'] >= LON-eps) & (cfis_set['lon'] <= LON+eps) & (cfis_set['lat'] >= LAT-eps) & (cfis_set['lat'] <= LAT+eps)]
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
plt.scatter(train_set['lon'], train_set['lat'], c=train_set['SIF'], cmap=sif_cmap, vmin=0.2, vmax=1.5)
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





# Plot TROPOMI vs SIF (and linear regression)
# x = cfis_set['SIF']  # cfis_points[:, 0]
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

