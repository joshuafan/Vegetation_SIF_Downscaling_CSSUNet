import copy
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime


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


def plot_histogram(column, plot_filename):
    column = column.flatten()
    print('Column', column)
    column = column[~np.isnan(column)]
    print(plot_filename)
    print('Number of datapoints:', len(column))
    print('Mean:', round(np.mean(column), 4))
    print('Std:', round(np.std(column), 4))
    print('Max:', round(np.max(column), 4))
    print('Min:', round(np.min(column), 4))
    n, bins, patches = plt.hist(column, 20, facecolor='blue', alpha=0.5)
    plt.savefig('exploratory_plots/' + plot_filename)
    plt.close()


# For each tile in the batch, returns a list of subtiles.
# Given a Tensor of tiles, with shape (batch x C x W x H), returns a Tensor of
# shape (batch x SUBTILE x C x subtile_dim x subtile_dim)
def get_subtiles_list(tile, subtile_dim, device):
    batch_size, bands, width, height = tile.shape
    num_subtiles_along_width = int(width / subtile_dim)
    num_subtiles_along_height = int(height / subtile_dim)
    num_subtiles = num_subtiles_along_width * num_subtiles_along_height
    assert(num_subtiles_along_width == 37)
    assert(num_subtiles_along_height == 37)
    subtiles = torch.empty((batch_size, num_subtiles, bands, subtile_dim, subtile_dim), device=device)
    for b in range(batch_size):
        subtile_idx = 0
        for i in range(num_subtiles_along_width):
            for j in range(num_subtiles_along_height):
                subtile = tile[b, :, subtile_dim*i:subtile_dim*(i+1), subtile_dim*j:subtile_dim*(j+1)].to(device)
                subtiles[b, subtile_idx, :, :, :] = subtile
                subtile_idx += 1
    return subtiles


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
                #print('=========================')
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
                    predicted_sif_standardized = model(input_tile_standardized)[0].flatten()
                    loss = criterion(predicted_sif_standardized, true_sif_standardized)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                with torch.set_grad_enabled(False):
                    predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                    non_standardized_loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)
                    j += 1
                    if j % 1 == 0:
                        print('========================')
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
                torch.save(model.state_dict(), TRAINED_MODEL_FILE)


            # Record loss
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, best_loss




