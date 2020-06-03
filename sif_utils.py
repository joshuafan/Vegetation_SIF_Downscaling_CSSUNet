import copy
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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


def plot_histogram(column, plot_filename, title=None):
    column = column.flatten()
    #print('Column', column)
    column = column[~np.isnan(column)]
    print(plot_filename)
    print('Number of datapoints:', len(column))
    print('Mean:', round(np.mean(column), 4))
    print('Std:', round(np.std(column), 4))
    print('Max:', round(np.max(column), 4))
    print('Min:', round(np.min(column), 4))
    n, bins, patches = plt.hist(column, 20, facecolor='blue', alpha=0.5)
    if title is not None:
        plt.title(title)
    plt.savefig('exploratory_plots/' + plot_filename)
    plt.close()


def print_stats(true, predicted, average_sif):
    if isinstance(true, list): 
        true = np.array(true)
    if isinstance(predicted, list):
        predicted = np.array(predicted)
    #print('True', true[:50])
    #print('Predicted', predicted[:50])
    #print('Sif mean:', average_sif)
    predicted_to_true = LinearRegression().fit(predicted.reshape(-1, 1), true)
    #print('True vs predicted regression', predicted_to_true.coef_, 'intercept', predicted_to_true.intercept_)
    predicted_rescaled = predicted_to_true.predict(predicted.reshape(-1, 1))
    r2 = r2_score(true, predicted_rescaled)
    corr, _ = pearsonr(true, predicted_rescaled)
    nrmse = math.sqrt(mean_squared_error(true, predicted_rescaled)) / average_sif
    nrmse_unstd = math.sqrt(mean_squared_error(true, predicted)) / average_sif
    spearman_rank_corr, _ = spearmanr(true, predicted_rescaled)
    print('R2:', round(r2, 3))
    print('NRMSE:', round(nrmse, 3))
    #print('NRMSE (unstandardized):', round(nrmse_unstd, 3))
    print('Pearson correlation:', round(corr, 3))
    #print('Pearson (unstandardized):', round(pearsonr(true, predicted)[0], 3))
    #print('Spearman rank corr:', round(spearman_rank_corr, 3))


# For the given tile, returns a list of subtiles.
# Given a tile Tensor with shape (C x H x W), returns a Tensor of
# shape (SUBTILE x C x subtile_dim x subtile_dim).
# NOTE: some sub-tiles will be removed if its fraction covered by clouds
# exceeds "max_subtile_cloud_cover"
def get_subtiles_list(tile, subtile_dim, device, max_subtile_cloud_cover=None):
    bands, height, width = tile.shape
    num_subtiles_along_height = int(height / subtile_dim)
    num_subtiles_along_width = int(width / subtile_dim)
    num_subtiles = num_subtiles_along_height * num_subtiles_along_width
    subtiles = []
    skipped = 0
    for i in range(num_subtiles_along_height):
        for j in range(num_subtiles_along_width):
            subtile = tile[:, subtile_dim*i:subtile_dim*(i+1), subtile_dim*j:subtile_dim*(j+1)].to(device)
            fraction_missing = 1 - torch.mean(subtile[-1, :, :])
            #print("Missing", fraction_missing)
            if (max_subtile_cloud_cover is not None) and fraction_missing > max_subtile_cloud_cover:
                skipped += 1
                continue
            subtiles.append(subtile)
    subtiles = torch.stack(subtiles)
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
                #print(input_tile_standardized.shape)
                #print('=========================')
                #print('Input tile - random pixel', input_tile_standardized[0, :, 200, 370])
                #print('Input band means')
                #print(torch.mean(input_tile_standardized[0], dim=(1,2)))

                # Real SIF value (non-standardized)
                true_sif_non_standardized = sample['SIF'].to(device)

                # Standardize SIF to have distribution with mean 0, standard deviation 1
                true_sif_standardized = ((true_sif_non_standardized - sif_mean) / sif_std).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input_tile_standardized)
                    #print('pred sif std', predicted_sif_standardized.shape)
                    #print('pred shape', pred.shape)
                    if type(output) is tuple:
                        #subtile_pred = output[4] * sif_std + sif_mean
                        output = output[0]
                    predicted_sif_standardized = output.flatten()
                    loss = criterion(predicted_sif_standardized, true_sif_standardized)

                    # backward + optimize only if in training phase
                    if phase == 'train':
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
                        print('> Predicted', predicted_sif_non_standardized)
                        print('> True', true_sif_non_standardized)
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
