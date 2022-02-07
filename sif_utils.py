import copy
import math
import numpy as np
import os
import random
import subprocess
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.colors import Normalize 

from datetime import datetime
import visualization_utils

from scipy.interpolate import interpn
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


# Encourages predictions to be similar within a single crop type, and
# different across crop types.
def crop_type_loss(predicted_fine_sifs, tiles, valid_masks, crop_type_indices=list(range(12, 42)),
                   min_fraction=0.01):
    loss = 0
    for i in range(predicted_fine_sifs.shape[0]):
        predicted = predicted_fine_sifs[i]
        tile = tiles[i]
        valid_mask = valid_masks[i]
        crop_type_sif_means = [] #torch.empty((len(crop_type_indices)))
        crop_type_sif_stds = [] #torch.empty((len(crop_type_indices)))
        total_pixels = predicted_fine_sifs.shape[1] * predicted_fine_sifs.shape[2]
        # print('Total pixels', total_pixels)
        for i, idx in enumerate(crop_type_indices):
            valid_crop_type_pixels = tile[idx].bool() & valid_mask
            # print('Crop type', i, '# pixels', torch.count_nonzero(valid_crop_type_pixels))
            if torch.count_nonzero(valid_crop_type_pixels) < min_fraction * total_pixels:
                continue
            predicted_sif_crop_type = predicted[valid_crop_type_pixels].flatten()
            loss += torch.std(predicted_sif_crop_type)
        #     print('Predicted sif crop type shape', predicted_sif_crop_type.shape)
        #     crop_type_sif_means.append(torch.mean(predicted_sif_crop_type).item())
        #     crop_type_sif_stds.append(torch.std(predicted_sif_crop_type))
        # print('Crop type sif means', crop_type_sif_means)
        # print('Crop type sif stds', crop_type_sif_stds)
        # loss += (np.std(crop_type_sif_means) - np.mean(crop_type_sif_stds))
    return loss




def remove_pure_tiles(df, threshold=0.5):
    CROP_TYPES = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']
    mixed_pixels = df[CROP_TYPES[0]] < threshold
    # print('Total tiles', len(df))
    for idx in range(1, len(CROP_TYPES)):
        # print('Num tiles with a lot of ', CROP_TYPES[idx], len(df[df[CROP_TYPES[idx]] > threshold]))
        mixed_pixels = mixed_pixels & (df[CROP_TYPES[idx]] < threshold)
    df = df[mixed_pixels]
    # print('Mixed (non-pure) tiles', len(df))
    return df
    
# Returns the upper-left corner of the large (1x1 degree) grid area
def get_large_grid_area_coordinates(lon, lat, grid_area_degrees, decimal_places=1):
    return (round(round_down(lon, grid_area_degrees), decimal_places),
            round(round_down(lat, grid_area_degrees) + grid_area_degrees, decimal_places))


# Returns the upper-left corner of the large (1x1 degree) grid area, but with latitude specified first
def get_large_grid_area_coordinates_lat_first(lat, lon, grid_area_degrees, decimal_places=1):
    return (round(round_down(lat, grid_area_degrees) + grid_area_degrees, decimal_places),
            round(round_down(lon, grid_area_degrees), decimal_places))

# Round down to the next-lower multiple of "divisor".
# Code from https://stackoverflow.com/questions/13082698/rounding-down-integers-to-nearest-multiple
def round_down(num, divisor):
    return math.floor(num / divisor) * divisor  # int(num - (num%divisor))

def determine_split(large_grid_areas, row, grid_area_degrees):
    grid_area_coordinates = get_large_grid_area_coordinates(row['lon'], row['lat'], grid_area_degrees)
    if grid_area_coordinates in large_grid_areas:
        return large_grid_areas[grid_area_coordinates]
    else:
        print('Row was outside the range', row)
        exit(1)

def determine_split_random(row, fraction_val, fraction_test):
    random_number = random.random()
    if random_number < 1 - fraction_test - fraction_val:
        return 'train'
    elif random_number < 1 - fraction_test:
        return 'val'
    else:
        return 'test'

# Given a date string (e.g. "20150608"), returns the year, month, and day-of-year (e.g. June 8 is day #159 of the year)
def parse_date_string(date_string):
    dt = datetime.strptime(date_string, '%Y-%m-%d')
    day_of_year = dt.timetuple().tm_yday
    month = dt.timetuple().tm_mon
    year = dt.timetuple().tm_year
    return year, month, day_of_year


def lat_long_to_index(lat, lon, dataset_top_bound, dataset_left_bound, resolution):
    height_idx = (dataset_top_bound - lat) / resolution[0]
    width_idx = (lon - dataset_left_bound) / resolution[1]
    eps = 1e-6
    return int(height_idx+eps), int(width_idx+eps)


# Compute average input features for this subregion, over valid pixels.
# "input_tile" should be of shape (# channels x height x width).
# "invalid_mask" should be of shape (height x width).
def compute_band_averages(input_tile, invalid_mask, missing_reflectance_idx=-1):
    assert input_tile.shape[1] == invalid_mask.shape[0] and input_tile.shape[2] == invalid_mask.shape[1]

    input_pixels = np.moveaxis(input_tile, 0, -1)
    input_pixels = input_pixels.reshape((-1, input_pixels.shape[2]))
    invalid_pixels = invalid_mask.flatten()

    if invalid_pixels.shape[0] > 1:
        valid_mask = (invalid_pixels == 0)
        pixels_with_data = input_pixels[valid_mask, :]
    else:
        # If there is only 1 pixel, something goes wrong with the Boolean indexing,
        # so we treat this case separately.
        pixels_with_data = input_pixels

    average_input_features = np.mean(pixels_with_data, axis=0)

    # Only change the "missing reflectance" feature to be the average across all pixels
    # (not just non-missing ones)
    average_input_features[missing_reflectance_idx] = np.mean(input_tile[missing_reflectance_idx, :, :])
    return average_input_features


def extract_input_subtile(min_lon, max_lon, min_lat, max_lat, input_tiles_dir, subtile_size_pixels,
                          res, input_channels=43, reflectance_tile_pixels=371):
    # Figure out which reflectance files to open. For each edge of the bounding box,
    # find the left/top bound of the surrounding reflectance large tile.
    min_lon_tile_left = (math.floor(min_lon * 10) / 10)
    max_lon_tile_left = (math.floor(max_lon * 10) / 10)
    min_lat_tile_top = (math.ceil(min_lat * 10) / 10)
    max_lat_tile_top = (math.ceil(max_lat * 10) / 10)
    num_tiles_lon = round((max_lon_tile_left - min_lon_tile_left) * 10) + 1
    num_tiles_lat = round((max_lat_tile_top - min_lat_tile_top) * 10) + 1
    file_left_lons = np.linspace(min_lon_tile_left, max_lon_tile_left, num_tiles_lon,
                                 endpoint=True)

    # Go through lats from top to bottom, because indices are numbered from top to bottom
    file_top_lats = np.linspace(min_lat_tile_top, max_lat_tile_top, num_tiles_lat,
                                endpoint=True)[::-1]
    # print("File left lons", file_left_lons)
    # print("File top lats", file_top_lats)


    # Because a sub-tile could span multiple files, patch together all of the files that
    # contain any portion of the sub-tile
    columns = []
    FILE_EXISTS = False  # Set to True if at least one file exists
    for file_left_lon in file_left_lons:
        rows = []
        for file_top_lat in file_top_lats:
            # Find what reflectance file to read from
            file_center_lon = round(file_left_lon + 0.05, 2)
            file_center_lat = round(file_top_lat - 0.05, 2)
            large_tile_filename = input_tiles_dir + "/reflectance_lat_" + str(file_center_lat) +  \
                                  "_lon_" + str(file_center_lon) + ".npy"
            if not os.path.exists(large_tile_filename):
                print('Needed data file', large_tile_filename, 'does not exist!')
                # For now, consider the data for this section as missing
                missing_tile = np.zeros((input_channels, reflectance_tile_pixels,
                                         reflectance_tile_pixels))
                missing_tile[-1, :, :] = 1
                rows.append(missing_tile)
            else:
                # print('Large tile filename', large_tile_filename)
                large_tile = np.load(large_tile_filename)
                rows.append(large_tile)
                FILE_EXISTS = True

        column = np.concatenate(rows, axis=1)
        columns.append(column)

    # If no input files exist, ignore this tile
    if not FILE_EXISTS:
        return None

    combined_large_tiles = np.concatenate(columns, axis=2)
    # print('All large tiles shape', combined_large_tiles.shape)

    # Find indices of bounding box within this combined large tile
    top_idx, left_idx = lat_long_to_index(max_lat, min_lon, max_lat_tile_top,
                                          min_lon_tile_left, res)
    bottom_idx = top_idx + subtile_size_pixels
    right_idx = left_idx + subtile_size_pixels
    # print('From combined large tile: Top', top_idx, 'Bottom', bottom_idx, 'Left', left_idx, 'Right', right_idx)

    # If the selected region (box) goes outside the range of the cover or reflectance dataset, that's a bug!
    if top_idx < 0 or left_idx < 0:
        print("Index was negative!")
        exit(1)
    if (bottom_idx > combined_large_tiles.shape[1] or right_idx > combined_large_tiles.shape[2]):
        print("Reflectance index went beyond edge of array!")
        exit(1)

    input_tile = combined_large_tiles[:, top_idx:bottom_idx, left_idx:right_idx]
    return input_tile



def downsample_sif(sif_array, valid_sif_mask, soundings_array, resolution_pixels):
    # Zero out SIFs for invalid pixels (pixels with no valid SIF label, or cloudy pixels).
    # Now, only VALID pixels contain a non-zero SIF.
    sif_array[valid_sif_mask == 0] = 0
    soundings_array[valid_sif_mask == 0] = 0
    # print('Sif array', sif_array.shape)
    # print('valid sif mask', valid_sif_mask.shape)
    # print('soundings array', soundings_array.shape)

    # For each coarse-SIF sub-region, compute the fraction of valid pixels.
    # Each square of "fraction_valid" is: (# valid fine pixels) / (# total fine pixels)
    avg_pool = nn.AvgPool2d(kernel_size=resolution_pixels)
    fraction_valid = avg_pool(valid_sif_mask.float())

    # Average together fine SIF predictions for each coarse SIF area.
    # Each square of "coarse_sifs" is: (sum SIF over valid fine pixels) / (# total fine pixels)
    coarse_sifs = avg_pool(sif_array)
    # print('Coarse sifs', coarse_sifs.shape)
    # print('Fraction valid', fraction_valid.shape)

    # Instead of dividing by the total number of fine pixels, we want to divide by the number of VALID fine pixels.
    # Each square is now: (sum SIF over valid fine pixels) / (# valid fine pixels), which is what we want.
    coarse_sifs = coarse_sifs / fraction_valid

    # Compute number of soundings in coarse array: (average * # total fine pixels)
    coarse_soundings = avg_pool(soundings_array) * (resolution_pixels ** 2)

    return coarse_sifs, fraction_valid, coarse_soundings


# Inefficient method, used to double-check correctness of downsampling
def downsample_sif_for_loop(sif_array, valid_sif_mask, soundings_array, resolution_pixels):
    # Zero out SIFs for invalid pixels (pixels with no valid SIF label, or cloudy pixels).
    # Now, only VALID pixels contain a non-zero SIF.
    sif_array[valid_sif_mask == 0] = 0
    soundings_array[valid_sif_mask == 0] = 0

    coarse_shape = (int(sif_array.shape[0] / resolution_pixels), int(sif_array.shape[1] / resolution_pixels))
    coarse_sifs = np.zeros(coarse_shape)
    fraction_valid = np.zeros(coarse_shape)
    coarse_soundings = np.zeros(coarse_shape)

    for i in range(coarse_shape[0]):
        for j in range(coarse_shape[1]):
            # For each coarse area, grab the corresponding subregion from input tile and SIF
            valid_sif_mask_subregion = valid_sif_mask[i*resolution_pixels:(i+1)*resolution_pixels, 
                                                      j*resolution_pixels:(j+1)*resolution_pixels].numpy()
            if np.count_nonzero(valid_sif_mask_subregion) == 0:
                continue
            sif_subregion = sif_array[i*resolution_pixels:(i+1)*resolution_pixels, 
                                      j*resolution_pixels:(j+1)*resolution_pixels].numpy()
            soundings_subregion = soundings_array[i*resolution_pixels:(i+1)*resolution_pixels, 
                                                  j*resolution_pixels:(j+1)*resolution_pixels].numpy()
            coarse_sifs[i, j] = np.sum(sif_subregion) / np.count_nonzero(valid_sif_mask_subregion)
            fraction_valid[i, j] = np.count_nonzero(valid_sif_mask_subregion) / valid_sif_mask_subregion.size
            coarse_soundings[i, j] = np.sum(soundings_subregion)
    return coarse_sifs, fraction_valid, coarse_soundings




# For each datapoint, returns the average of the pixels in "array" where "mask" is 1.
# "array" and "mask" are assumed to have the same shape.
# "mask" MUST be a binary 1/0 mask over the pixels.
# "dim_to_average" are the dimensions to average over (usually, the height/width dimensions if you're averaging across an image)
def masked_average(array, mask, dims_to_average):
    assert(len(array.shape) == len(mask.shape))

    # Zero out unwanted pixels
    wanted_values = array * mask

    # Sum over wanted pixels, then divide by the number of wanted pixels
    return torch.sum(wanted_values, dim=dims_to_average) / torch.sum(mask, dim=dims_to_average)

def masked_average_numpy(array, mask, dims_to_average):
    assert(len(array.shape) == len(mask.shape))

    # Zero out unwanted pixels
    wanted_values = array * mask

    # Sum over wanted pixels, then divide by the number of wanted pixels
    return np.sum(wanted_values, axis=dims_to_average) / np.sum(mask, axis=dims_to_average)


def plot_histogram(column, plot_filename, plot_dir="/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/exploratory_plots", title=None, weights=None):
    column = column.flatten()
    column = column[~np.isnan(column)]
    print(plot_filename)
    # print('Number of datapoints:', len(column))
    if weights is not None:
        print('Weighted mean', round(np.average(column, weights=weights), 4))
    else:
        print('Mean:', round(np.mean(column), 4))
        print('Std:', round(np.std(column), 4))
    # print('Max:', round(np.max(column), 4))
    # print('Min:', round(np.min(column), 4))
    n, bins, patches = plt.hist(column, 40, facecolor='blue', alpha=0.5, weights=weights)
    if title is None:
        title = ""
    else:
        title += "\n"
    title += ("(Mean = " + str(round(np.mean(column), 3)) + ", Std = " + str(round(np.std(column), 3)) + ", Num datapoints = " + str(len(column)) + ")")
    plt.title(title)
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.close()


# Code from
# https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig, ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )  # cmap=plt.get_cmap("Greys"),

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    # cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    # cbar.ax.set_ylabel('Density')

    return ax


def print_stats(true, predicted, average_sif, print_report=True, ax=None, fit_intercept=False):
    if isinstance(true, list):
        true = np.array(true)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    # true = true.reshape(-1, 1)
    # true_to_predicted = LinearRegression().fit(true, predicted)
    # slope = true_to_predicted.coef_[0]
    # intercept = true_to_predicted.intercept_
    # true_rescaled = true_to_predicted.predict(true)
    # r2 = r2_score(true_rescaled, predicted)

    predicted = predicted.reshape(-1, 1)
    predicted_to_true = LinearRegression(fit_intercept=fit_intercept).fit(predicted, true)
    slope = predicted_to_true.coef_[0]
    if fit_intercept:
        intercept = predicted_to_true.intercept_
    predicted_rescaled = predicted_to_true.predict(predicted)
    predicted = predicted.flatten()
    r2_scaled = r2_score(true, predicted_rescaled)
    r2_unscaled = r2_score(true, predicted)
    corr, _ = pearsonr(true, predicted)

    # Note NRMSE is not scaled linearly?
    nrmse_unscaled = math.sqrt(mean_squared_error(true, predicted)) / average_sif
    nrmse_scaled = math.sqrt(mean_squared_error(true, predicted_rescaled)) / average_sif
    mse_scaled = mean_squared_error(true, predicted_rescaled)
    if fit_intercept:
        line = slope * predicted + intercept
        equation_string = 'y={:.2f}x+{:.2f}'.format(slope, intercept)
    else:
        line = slope * predicted
        equation_string = 'y={:.2f}x'.format(slope)

    if ax is not None:
        predicted = predicted.ravel()
        true = true.ravel()
        if predicted.size > 500:
            ax = density_scatter(predicted, true, bins=[40, 40], ax=ax, s=5)
            # # Calculate the point density
            # xy = np.vstack([predicted, true])
            # z = gaussian_kde(xy)(xy)

            # # Sort the points by density, so that the densest points are plotted last
            # idx = z.argsort()
            # x, y, z = predicted[idx], true[idx], z[idx]

            # fig, ax = plt.subplots()
            # ax.scatter(x, y, c=z, s=50, edgecolor='')
        else:
            ax.scatter(predicted, true, color="k", s=5)

        ax.plot(predicted, line, 'r', label=equation_string + ' (R^2={:.3f}, unscaled NRMSE={:.3f})'.format(r2_unscaled, nrmse_unscaled))
        # ax.plot(predicted, line, 'r', label='y={:.2f}x+{:.2f} (R^2={:.3f}, unscaled NRMSE={:.3f})'.format(slope, intercept, r2_scaled, nrmse_unscaled))
        ax.set(xlabel='Predicted', ylabel='True')
        if predicted.size > 1000:
            loc = 'lower right'
        else:
            loc = 'best'
        ax.legend(fontsize=9, loc=loc)

    if print_report:
        print('(num datapoints)', true.size)
        # print('True vs predicted regression:', equation_string)
        # print('R2:', round(r2_scaled, 3))
        print('NRMSE (unscaled):', round(nrmse_unscaled, 3))
        print('R2 (unscaled):', round(r2_unscaled, 3))
        print('Corr:', round(corr, 3))
        # print('NRMSE (scaled):', round(nrmse_scaled, 3))

    return r2_unscaled, nrmse_unscaled, corr

    # corr, _ = pearsonr(true, predicted_rescaled)
    # nrmse_unstd = math.sqrt(mean_squared_error(true, predicted)) / average_sif
    # spearman_rank_corr, _ = spearmanr(true, predicted_rescaled)
    #print('NRMSE (unstandardized):', round(nrmse_unstd, 3))
    #print('Pearson correlation:', round(corr, 3))
    #print('Pearson (unstandardized):', round(pearsonr(true, predicted)[0], 3))
    #print('Spearman rank corr:', round(spearman_rank_corr, 3))


# For the given tile, returns a list of subtiles.
# Given a tile Tensor with shape (C x H x W), returns a Tensor of
# shape (SUBTILE x C x subtile_dim x subtile_dim).
# NOTE: some sub-tiles will be removed if its fraction covered by clouds
# exceeds "max_subtile_cloud_cover"
def get_subtiles_list(tile, subtile_dim): #, max_subtile_cloud_cover=None):
    bands, height, width = tile.shape
    num_subtiles_along_height = round(height / subtile_dim)
    num_subtiles_along_width = round(width / subtile_dim)
    num_subtiles = num_subtiles_along_height * num_subtiles_along_width
    subtiles = []
    skipped = 0

    # Loop through all sub-tile positions
    for i in range(num_subtiles_along_height):
        for j in range(num_subtiles_along_width):
            # If this sub-tile would extend beyond the edge of the large tile,
            # push it back inside. (That's what the "min" does.)
            subtile_height_start = min(subtile_dim * i, height - subtile_dim)
            subtile_height_end = min(subtile_dim * (i+1), height)
            subtile_width_start = min(subtile_dim * j, width - subtile_dim)
            subtile_width_end = min(subtile_dim * (j+1), width)

            # Extract sub-tile
            subtile = tile[:, subtile_height_start:subtile_height_end, subtile_width_start:subtile_width_end]

            # If too much data is missing, don't include this sub-tile. (Not inlcuding this for now)
            # fraction_missing = 1 - np.mean(subtile[-1, :, :])
            # if (max_subtile_cloud_cover is not None) and fraction_missing > max_subtile_cloud_cover:
            #     skipped += 1
            #     continue

            subtiles.append(subtile)
    subtiles = np.stack(subtiles)
    #print('Subtile tensor shape', subtiles.shape, 'num skipped:', skipped)
    return subtiles


# If any of the specified crop types (listed in "crop_indices") makes up
# more than "pure_threshold" fraction of the sub-tile, return that crop type.
# Otherwise, if none of the specified crop types dominates, return -1.
def get_crop_type(band_means, crop_indices, pure_threshold):
    assert(pure_threshold >= 0.5)
    for crop_idx in crop_indices:
        if band_means[crop_idx] >= pure_threshold:
            return crop_idx
    return -1 # Return -1 if sub-tile is not pure


# For the given tile, returns a dictionary mapping (crop type -> list of subtiles)
# The key is the index of the crop type channel in the tile. -1 is the key for
# mixed sub-tiles that are not dominated by any crop type.
# Each list of subtiles is a  Tensor with shape (C x H x W), returns a Tensor of
# shape (SUBTILE x C x subtile_dim x subtile_dim).
# NOTE: sub-tiles can be returned in any order
# NOTE: sub-tiles will be removed if its fraction covered by clouds
# exceeds "max_subtile_cloud_cover"
def get_subtiles_list_by_crop(tile, subtile_dim, device, crop_indices, pure_threshold, max_subtile_cloud_cover=None):
    assert(pure_threshold >= 0.5)
    bands, height, width = tile.shape
    num_subtiles_along_height = int(height / subtile_dim)
    num_subtiles_along_width = int(width / subtile_dim)
    num_subtiles = num_subtiles_along_height * num_subtiles_along_width
    subtiles = {}
    skipped = 0
    num_subtiles = 0

    # Loop through all sub-tiles
    for i in range(num_subtiles_along_height):
        for j in range(num_subtiles_along_width):
            # Extract subtile, compute mean of each band (channel)
            subtile = tile[:, subtile_dim*i:subtile_dim*(i+1), subtile_dim*j:subtile_dim*(j+1)].to(device)
            band_means = torch.mean(subtile, dim=(1, 2))
            #print('=====================================')
            #print('Band means:', band_means)

            # Remove subtile if it's covered by clouds
            if (max_subtile_cloud_cover is not None) and 1 - band_means[-1] > max_subtile_cloud_cover:
                skipped += 1
                continue

            # Find which crop type dominates this subtile (if any)
            num_subtiles += 1
            crop = get_crop_type(band_means, crop_indices, pure_threshold)
            #print('Crop type:', crop)
            if crop not in subtiles:
                subtiles[crop] = []
            subtiles[crop].append(subtile)

    for crop, subtile_list in subtiles.items():
        subtiles[crop] = torch.stack(subtile_list)
        #print('Crop type', crop, 'Shape', subtiles[crop].shape)
    return subtiles, num_subtiles


def get_top_bound(point_lat):
    return math.ceil(point_lat * 10) / 10

def get_left_bound(point_lon):
    return math.floor(point_lon * 10) / 10


# Train CNN to predict total SIF of tile.
# "model" should take in a (standardized) tile (with dimensions CxWxH), and output standardized SIF.
# "dataloader" should return, for each training example: 'tile' (standardized CxWxH tile), and 'SIF' (non-standardized SIF) 
def train_single_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, MODEL_FILE, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            j = 0
            for sample in dataloaders[phase]: 
                # Standardized input tile, (batch x C x W x H)
                input_tile_standardized = sample['tile'].to(device)
                print('=========================')
                print('Input tile - random pixel', input_tile_standardized[0, :, 8, 8])
                print('Input band means')
                print(torch.mean(input_tile_standardized[0], dim=(1,2)))

                # Real SIF value (non-standardized)
                true_sif_non_standardized = sample['SIF'].to(device)

                # Standardize SIF to have distribution with mean 0, standard deviation 1
                true_sif_standardized = ((true_sif_non_standardized - sif_mean) / sif_std).to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input_tile_standardized)
                    #print('pred shape', pred.shape)
                    # if type(output) is tuple:
                    #     #subtile_pred = output[4] * sif_std + sif_mean
                    #     output = output[0]
                    predicted_sif_standardized = output.flatten()
                    print('pred sif std', predicted_sif_standardized.shape)
                    loss = criterion(predicted_sif_standardized, true_sif_standardized)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()  #retain_graph=True)
                        optimizer.step()

                # statistics
                with torch.set_grad_enabled(False):
                    predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                    non_standardized_loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)
                    j += 1
                    if j % 100 == 1:
                        print('========================')
                        #print('Subtile SIF pred:', subtile_pred)
                        print('> Predicted', predicted_sif_non_standardized[0:20])
                        print('> True', true_sif_non_standardized[0:20])
                        print('> batch loss', (math.sqrt(non_standardized_loss.item()) / sif_mean).item())
                    running_loss += non_standardized_loss.item() * len(sample['SIF'])

            epoch_loss = math.sqrt(running_loss / dataset_sizes[phase]) / sif_mean
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
 
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                # save model in case
                torch.save(model.state_dict(), MODEL_FILE)


            # Record loss
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
                epoch_time = time.time() - epoch_start
                print('Epoch time:', epoch_time)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, best_loss

