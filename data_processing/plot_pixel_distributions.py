"""
Plot pixel-level distributions for TROPOMI, and write the mean/std of each
channel to a file
"""
import copy
import csv
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler

import time
import torch
import torchvision
import torchvision.transforms as transforms
import resnet
import torch.nn as nn
import torch.optim as optim

from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
from subtile_embedding_dataset import SubtileEmbeddingDataset

import sif_utils
import tile_transforms


# TODO this is a hack
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet
from embedding_to_sif_model import EmbeddingToSIFModel


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_2degree_random0")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
PIXEL_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_pixels.csv")
MISSING_REFLECTANCE_IDX = -1

# INFO_FILE_CFIS = os.path.join(DATASET_DIR, "cfis_subtiles_filtered.csv")

COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7', 'ref_10', 'ref_11']

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)

# Read tile metadata for train and CFIS 
train_metadata = pd.read_csv(INFO_FILE_TRAIN)
train_metadata = train_metadata[(train_metadata['source'] == 'TROPOMI')]
# cfis_metadata = pd.read_csv(INFO_FILE_CFIS)

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
print('Tile-level statistics', train_statistics)
train_means = train_statistics['mean'].copy().to_numpy()
train_stds = train_statistics['std'].copy().to_numpy()
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

# Read all large tile files, extract pixel values
pixel_arrays = []
sum_squared_differences = np.zeros_like(band_means)  # For each band, keep track of the sum of squared differences from mean
i = 0
total_pixels = 0
for large_tile_file in train_metadata['tile_file']:
    tile = np.load(large_tile_file)

    # Compute averages of each band (over non-cloudy pixels)
    # Reshape tile into a list of pixels (pixels x channels)
    pixels = np.moveaxis(tile, 0, -1)
    pixels = pixels.reshape((-1, pixels.shape[2]))

    # NOTE: Exclude missing (cloudy) pixels
    pixels_with_data = pixels[pixels[:, MISSING_REFLECTANCE_IDX] == 0]

    # Remove tiles where no pixels have data (it's completely covered by clouds)
    num_pixels_with_data = pixels_with_data.shape[0]
    if num_pixels_with_data == 0:
        continue
    total_pixels += num_pixels_with_data

    # Compute squared differences from mean
    squared_differences = (pixels_with_data - band_means) ** 2
    sum_squared_differences += np.sum(squared_differences, axis=0)

    # Sample some pixels for plotting purposes
    num_pixels_to_sample = math.ceil(num_pixels_with_data / 1000)
    random_indices = np.random.choice(num_pixels_with_data, num_pixels_to_sample)
    pixel_arrays.append(pixels_with_data[random_indices, :])

    # Compute average of the reflectance, over the non-cloudy pixels
    # reflectance_averages = np.mean(pixels_with_data[:, REFLECTANCE_BANDS], axis=0)
    # tile_averages[REFLECTANCE_BANDS] = reflectance_averages
    # pixel_arrays.append(pixels)
    i += 1
    if i % 100 == 0:
        print('large tile', i)

# Compute standard deviation of pixels for each band, and write those statistics to a file
pixels_stds = np.sqrt(sum_squared_differences / total_pixels)
pixels_stds = np.append(pixels_stds, sif_std)
train_statistics['std'] = pixels_stds
print('New pixel statistics', train_statistics)
train_statistics.to_csv(PIXEL_STATISTICS_FILE)

# Combine all sampled pixels into an array, for plotting
pixel_values = np.concatenate(pixel_arrays, axis=0)

# Plot histograms
for idx, column in enumerate(COLUMNS):
    sif_utils.plot_histogram(pixel_values[:, idx], "pixels_histogram_" + column + "_train_tropomi.png", title=column + ' (TROPOMI pixels)')
    standardized_pixel_values = (pixel_values[:, idx] - band_means[idx]) / band_stds[idx]
    sif_utils.plot_histogram(standardized_pixel_values, "pixels_histogram_" + column + "_train_tropomi_std.png", title=column + ' (TROPOMI pixels, std. by tile std dev)')
    standardized_pixel_values_by_pixel_std = (pixel_values[:, idx] - band_means[idx]) / pixels_stds[idx]
    sif_utils.plot_histogram(standardized_pixel_values_by_pixel_std, "pixels_histogram_" + column + "_train_tropomi_pixel_std.png", title=column + ' (TROPOMI pixels, std. by pixel std dev)')


# # Read through CFIS subtiles, extract pixel values
# cfis_pixel_arrays = []
# i = 0
# for subtile_file in cfis_metadata['tile_file']:
#     subtile = np.load(subtile_file)
#     pixels = np.moveaxis(subtile, 0, -1)
#     pixels = pixels.reshape((-1, pixels.shape[2]))
#     cfis_pixel_arrays.append(pixels)
#     i += 1
#     if i % 1000 == 0:
#         print('subtile', i)
# cfis_pixel_values = np.concatenate(cfis_pixel_arrays, axis=0)

# # Plot histograms
# for idx, column in enumerate(COLUMNS):
#     sif_utils.plot_histogram(cfis_pixel_values[:, idx], "pixels_histogram_aug_" + column + "_cfis.png", title=column + ' (CFIS pixels, 2016-08-01)')
#     standardized_pixel_values = (cfis_pixel_values[:, idx] - band_means[idx]) / band_stds[idx]
#     sif_utils.plot_histogram(standardized_pixel_values, "pixels_histogram_aug_" + column + "_cfis_std.png", title=column + ' (CFIS pixels, std. by band-average std dev)')
#     standardized_pixel_values_by_pixel_std = (cfis_pixel_values[:, idx] - band_means[idx]) / pixels_stds[idx]
#     sif_utils.plot_histogram(standardized_pixel_values_by_pixel_std, "pixels_histogram_aug_" + column + "_cfis_pixel_std.png", title=column + ' (CFIS pixels, std. by pixel std dev)')


