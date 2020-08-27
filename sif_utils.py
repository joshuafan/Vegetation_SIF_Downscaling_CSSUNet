import copy
import math
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.colors import Normalize 

from datetime import datetime
import cdl_utils

from scipy.interpolate import interpn
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Returns the upper-left corner of the large (1x1 degree) grid area
def get_large_grid_area_coordinates(lon, lat):
    return (math.floor(lon), math.ceil(lat))

# Round down to the next-lower multiple of "divisor".
# Code from https://stackoverflow.com/questions/13082698/rounding-down-integers-to-nearest-multiple
def round_down(num, divisor):
    return num - (num%divisor)

def determine_split(large_grid_areas, row):
    grid_area_coordinates = get_large_grid_area_coordinates(row['lon'], row['lat'])
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



def plot_histogram(column, plot_filename, title=None):
    column = column.flatten()
    column = column[~np.isnan(column)]
    print(plot_filename)
    # print('Number of datapoints:', len(column))
    print('Mean:', round(np.mean(column), 4))
    print('Std:', round(np.std(column), 4))
    # print('Max:', round(np.max(column), 4))
    # print('Min:', round(np.min(column), 4))
    n, bins, patches = plt.hist(column, 40, facecolor='blue', alpha=0.5)
    if title is not None:
        plt.title(title)
    plt.savefig('exploratory_plots/' + plot_filename)
    plt.close()


# Code from
# https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig, ax = plt.subplots()
    print('x/y shape', x.shape, y.shape)
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    # cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    # cbar.ax.set_ylabel('Density')

    return ax


def print_stats(true, predicted, average_sif, print_report=True, ax=None):
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
    predicted_to_true = LinearRegression().fit(predicted, true)
    slope = predicted_to_true.coef_[0]
    intercept = predicted_to_true.intercept_
    predicted_rescaled = predicted_to_true.predict(predicted)
    r2 = r2_score(true, predicted_rescaled)

    # Note NRMSE is not scaled linearly?
    nrmse_unscaled = math.sqrt(mean_squared_error(true, predicted)) / average_sif
    nrmse_scaled = math.sqrt(mean_squared_error(true, predicted_rescaled)) / average_sif
    mse_scaled = mean_squared_error(true, predicted_rescaled)
    line = slope * predicted + intercept

    if ax is not None:
        predicted = predicted.ravel()
        true = true.ravel()
        print('Predicted/true shape', predicted.shape, true.shape)
        if predicted.size > 1000:
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

        ax.plot(predicted, line, 'r', label='y={:.2f}x+{:.2f} (R^2={:.3f}, NRMSE={:.3f})'.format(slope, intercept, r2, nrmse_scaled))
        ax.set(xlabel='Predicted', ylabel='True')
        ax.legend(fontsize=9)

    if print_report:
        print('True vs predicted regression: y = ' + str(round(slope, 3)) + 'x + ' + str(round(intercept, 3)))
        print('R2:', round(r2, 3))
        print('NRMSE (unscaled):', round(nrmse_unscaled, 3))
        print('NRMSE (scaled):', round(nrmse_scaled, 3))

    return nrmse_scaled, mse_scaled, r2

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

