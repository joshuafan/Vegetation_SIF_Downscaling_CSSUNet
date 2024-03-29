"""
Evaluates performance of a pretrained Coarsely-Supervised Smooth U-Net model
on fine-resolution (pixel-level) labels.
"""
import argparse
import copy
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import simple_cnn
from datasets import FineSIFDataset
from unet.unet_model import UNetContrastive, UNet2Contrastive, UNet, UNet2, PixelNN, UNet2Spectral
import visualization_utils
import sif_utils
import tile_transforms
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import segmentation_models_pytorch as smp
sys.path.append('UNetPlusPlus/pytorch')
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.generic_UNetPlusPlus import Generic_UNetPlusPlus


# Data directories
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF"
METADATA_DIR = os.path.join(DATA_DIR, "metadata/CFIS_OCO2_dataset")
CFIS_COARSE_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_coarse_metadata.csv')
OCO2_METADATA_FILE = os.path.join(METADATA_DIR, 'oco2_metadata.csv')

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-model', "--model", default='unet2', choices=['unet2', 'unet2_spectral', 'unet2_contrastive',
                                                                   'unet', 'unet_contrastive', 'pixel_nn',
                                                                   'smp_unet', 'smp_unet_plus_plus',
                                                                   'mrg_unet', 'mrg_unet_plus_plus'], type=str, help='model type')
parser.add_argument('-model_path', "--model_path", type=str)
parser.add_argument('-scale_predictions_by', "--scale_predictions_by", type=float, default=1)

# Whether to use batchnorm and dropout
parser.add_argument('-batch_norm', "--batch_norm",  default=False, action='store_true', help='Whether to use BatchNorm')
parser.add_argument('-dropout_prob', "--dropout_prob", type=float, default=0, help="Dropout probability (set to 0 to not use dropout)")

parser.add_argument('-use_precomputed_results', "--use_precomputed_results", default=False, action='store_true', help='Whether to use pre-computed results (instead of running dataset through model)')
parser.add_argument('-plot_examples', "--plot_examples", default=False, action='store_true', help='Whether to plot example tiles')
parser.add_argument('-seed', "--seed", default=0, type=int)
parser.add_argument('-test_set', "--test_set", choices=["train", "val", "test"])
parser.add_argument('-normalize', "--normalize", action='store_true', help='Not used currently. Whether to normalize the reflectance bands to have norm 1. If this is enabled, the reflectance bands are NOT standardized.')
parser.add_argument('-compute_vi', "--compute_vi", action='store_true', help="Not used currently. Whether to compute vegetation indices per pixel")
parser.add_argument('-match_coarse', "--match_coarse", action='store_true', help='Whether to adjust fine train predictions to match the coarse average')
parser.add_argument('-match_coarse_1sounding', "--match_coarse_1sounding", action='store_true', help='Whether to adjust fine train predictions to match the coarse average, where coarse prediction is defined over 1 sounding')

parser.add_argument('-min_sif_clip', "--min_sif_clip", default=0.1, type=float, help="Before computing loss, clip outputs below this to this value.")
parser.add_argument('-min_input', "--min_input", default=-3, type=float, help="Clip extreme input values to this many standard deviations below mean")
parser.add_argument('-max_input', "--max_input", default=3, type=float, help="Clip extreme input values to this many standard deviations above mean")


args = parser.parse_args()
print("Model:", args.model_path)

# Folds
TRAIN_FOLDS = [0, 1, 2]
VAL_FOLDS = [3]
if args.test_set == "train":
    TEST_FOLDS = [0, 1, 2]
elif args.test_set == "val":
    TEST_FOLDS = [3]
elif args.test_set == "test":
    TEST_FOLDS = [4]
else:
    raise ValueError("invalid argument for --test_set", args.test_set)

# Set random seeds
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Band statistics (mean/std) for standardizing each channel. Depends on whether we're first normalizing reflectance vectors to norm 1 or not.
if args.normalize:
    BAND_STATISTICS_FILE = os.path.join(METADATA_DIR, 'normalized_cfis_band_statistics_train.csv')
else:
    BAND_STATISTICS_FILE = os.path.join(METADATA_DIR, 'cfis_band_statistics_train.csv')

# Resolution parameters
DEGREES_PER_PIXEL = (0.00026949458523585647, 0.00026949458523585647)
METERS_PER_PIXEL = 30
RESOLUTIONS = [30, 90, 150, 300, 600]
TILE_PIXELS = 100
TILE_SIZE_DEGREES = DEGREES_PER_PIXEL[0] * TILE_PIXELS


# CFIS filtering
eps = 1e-5
MIN_EVAL_CFIS_SOUNDINGS_EXPERIMENT = [30]  # [1, 5, 10, 20, 30]
MIN_EVAL_FRACTION_VALID = 0.9-eps
MIN_SIF_CLIP = 0.1
MAX_SIF_CLIP = None
MIN_COARSE_FRACTION_VALID_PIXELS = 0.1

# Dates
TRAIN_DATES = ['2016-06-15', '2016-08-01']
TEST_DATES = ['2016-06-15', '2016-08-01']

# Directory to write visualizations/plots to
RESULTS_DIR = os.path.join(os.path.dirname(args.model_path), "final_results_" + args.test_set)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
CFIS_TRUE_VS_PREDICTED_PLOT = os.path.join(RESULTS_DIR, "true_vs_predicted_sif_cfis_" + args.model)
RESULTS_SUMMARY_FILE = os.path.join(DATA_DIR, "unet_results", "results_summary_EVAL_" + args.test_set + ".csv")

# Other params
BATCH_SIZE = 128
NUM_WORKERS = 8
MIN_SIF = None
MAX_SIF = None
MIN_INPUT = -3
MAX_INPUT = 3
MIN_SIF_CLIP = 0.1
MIN_SIF_PLOT = 0
MAX_SIF_PLOT = 1.5

# BANDS = list(range(0, 12)) + [12, 13, 14, 16, 17, 19, 23, 24, 25, 28, 34] + [42]
# BANDS = list(range(0, 9)) + list(range(12, 27)) + [28] + [42] 
# BANDS = list(range(0, 43))
BANDS = list(range(0, 7)) + list(range(9, 12)) + [12, 13, 14, 16, 17, 19, 23, 24, 25, 28, 34] + [42]
CONTINUOUS_INDICES = list(range(0, 12))  # Indices of continuous bands to standardize/clip
CROP_TYPE_INDICES = list(range(10, 21))  # Indices of crop type bands (within BANDS)
INPUT_CHANNELS = len(BANDS)
OUTPUT_CHANNELS = 1
MISSING_REFLECTANCE_IDX = len(BANDS) - 1

# Filtering
PURE_THRESHOLD = 0.7  # Minimum percent land cover for "pure pixels"
MAX_CFIS_CLOUD_COVER = 0.5
MIN_OCO2_SOUNDINGS = 3
MAX_OCO2_CLOUD_COVER = 0.5
ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']  # Filter out tiles where less than 50% is any of these

# Columns for "prediction results" dataframe
COLUMN_NAMES = ['true_sif', 'predicted_sif_linear', 'predicted_sif_mlp', 'predicted_sif_unet',
                'lon', 'lat', 'source', 'date', 'large_tile_file', 'large_tile_lon', 'large_tile_lat',
                'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                'grassland_pasture', 'corn', 'soybean',
                'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                'woody_wetlands', 'open_water', 'alfalfa',
                'developed_low_intensity', 'developed_med_intensity', 'missing_reflectance',
                'num_soundings', 'fraction_valid',
                'true_coarse_sif']
PREDICTION_COLUMNS = ['predicted_sif_ridge', 'predicted_sif_mlp', 'predicted_sif_unet']

# Input and output column names
INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                 'grassland_pasture', 'corn', 'soybean',
                 'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                 'woody_wetlands', 'open_water', 'alfalfa',
                 'developed_low_intensity', 'developed_med_intensity', 'missing_reflectance']
OUTPUT_COLUMN = ['SIF']

# Order of column in band averages file
BAND_AVERAGES_COLUMN_ORDER =  ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
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
                          'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']

# Crop types to look at when analyzing results
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest']
DATES = ["2016-06-15", "2016-08-01"]



def eval_unet_fast(args, unet_model, dataloader, criterion, device, sif_mean, sif_std, resolution_meters, min_eval_cfis_soundings, min_eval_fraction_valid):
    """Quickly computes true vs predicted loss of U-Net on fine CFIS dataset, without
    recording predictions for each pixel or comparing with other methods.
    """
    fine_pixels_per_eval = int(resolution_meters / METERS_PER_PIXEL)
    running_coarse_loss = 0
    running_eval_loss = 0
    num_coarse_datapoints = 0
    num_eval_datapoints = 0
    all_true_coarse_sifs = []
    all_true_eval_sifs = []
    all_predicted_coarse_sifs = []
    all_predicted_eval_sifs = []

    # Set model to eval mode
    unet_model.eval()

    # Iterate over data.
    for sample in dataloader:
        with torch.set_grad_enabled(False):
            # Read input tile
            input_tiles_std = sample['input_tile'][:, BANDS, :, :].to(device)

            # Read coarse-resolution SIF label
            true_coarse_sifs = sample['coarse_sif'].to(device)

            # Read fine-resolution SIF labels
            true_fine_sifs = sample['fine_sif'].to(device)
            valid_fine_sif_mask = torch.logical_not(sample['fine_sif_mask']).to(device)
            fine_soundings = sample['fine_soundings'].to(device)

            # Pass tile through model to obtain fine-resolution SIF predictions
            if "smp" in args.model or "mrg" in args.model:
                # If using these pre-built U-Net models, need to pad images to 128x128,
                # then extract predictions from non-padded pixels.
                # TODO - this is adhoc and won't be robust to different image sizes
                batch, channels, height, width = input_tiles_std.shape
                padded_tiles = torch.zeros((batch, channels, 128, 128)).to(device)  # pad to [batch, channel, 128, 128]
                padded_tiles[:, :, 0:height, 0:width] = input_tiles_std
                predicted_fine_sifs_std = unet_model(padded_tiles)  # [batch, 1, 128, 128]
                predicted_fine_sifs_std = predicted_fine_sifs_std[:, :, 0:height, 0:width]  # [batch, 1, H, W]
            else:
                # Otherwise using our implementation, no padding is required
                outputs = unet_model(input_tiles_std)
                if type(outputs) == tuple:
                    predicted_fine_sifs_std = torch.squeeze(outputs[0], dim=1)  # predicted_fine_sifs_std: (batch, 1, H, W)
                else:
                    predicted_fine_sifs_std = torch.squeeze(outputs, dim=1)  # predicted_fine_sifs_std: (batch, 1, H, W)

            predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean
            predicted_fine_sifs *= args.scale_predictions_by

            # For each tile, take the average SIF over all valid pixels
            predicted_coarse_sifs = sif_utils.masked_average(predicted_fine_sifs, valid_fine_sif_mask, dims_to_average=(1, 2)) # (batch size)

            # If train, multiply predictions by "ratio" to ensure predicted coarse SIF matches ground-truth exactly
            if args.test_set == "train" and args.match_coarse:
                ratios = true_coarse_sifs / predicted_coarse_sifs
                predicted_fine_sifs = predicted_fine_sifs * ratios[:, None, None]

            # Compute loss (predicted vs true coarse SIF)
            coarse_loss = criterion(true_coarse_sifs, predicted_coarse_sifs)

            # Scale predicted/true to desired eval resolution
            predicted_eval_sifs, _, _ = sif_utils.downsample_sif(predicted_fine_sifs, valid_fine_sif_mask, fine_soundings, fine_pixels_per_eval)
            true_eval_sifs, eval_fraction_valid, eval_soundings = sif_utils.downsample_sif(true_fine_sifs, valid_fine_sif_mask, fine_soundings, fine_pixels_per_eval)

            # Filter noisy coarse tiles
            non_noisy_mask = (eval_soundings >= min_eval_cfis_soundings) & (true_eval_sifs >= MIN_SIF_CLIP) & (eval_fraction_valid >= min_eval_fraction_valid)
            non_noisy_mask_flat = non_noisy_mask.flatten()
            true_eval_sifs_filtered = true_eval_sifs.flatten()[non_noisy_mask_flat]
            predicted_eval_sifs_filtered = predicted_eval_sifs.flatten()[non_noisy_mask_flat]
            predicted_eval_sifs_filtered = torch.clamp(predicted_eval_sifs_filtered, min=MIN_SIF_CLIP)  # Clip low SIF to 0.1
            eval_loss = criterion(true_eval_sifs_filtered, predicted_eval_sifs_filtered)

            # Record stats
            running_coarse_loss += coarse_loss.item() * len(true_coarse_sifs)
            num_coarse_datapoints += len(true_coarse_sifs)
            running_eval_loss += eval_loss.item() * len(true_eval_sifs_filtered)
            num_eval_datapoints += len(true_eval_sifs_filtered)
            all_true_coarse_sifs.append(true_coarse_sifs.cpu().detach().numpy())
            all_true_eval_sifs.append(true_eval_sifs_filtered.cpu().detach().numpy())
            all_predicted_coarse_sifs.append(predicted_coarse_sifs.cpu().detach().numpy())
            all_predicted_eval_sifs.append(predicted_eval_sifs_filtered.cpu().detach().numpy())

    true_coarse = np.concatenate(all_true_coarse_sifs)
    true_eval = np.concatenate(all_true_eval_sifs)
    predicted_coarse = np.concatenate(all_predicted_coarse_sifs)
    predicted_eval = np.concatenate(all_predicted_eval_sifs)
    print('================== (Quick computation) Coarse CFIS stats ======================')
    sif_utils.print_stats(true_coarse, predicted_coarse, sif_mean, fit_intercept=False, ax=None)
    print('================== (Quick computation) Eval CFIS stats ======================')
    sif_utils.print_stats(true_eval, predicted_eval, sif_mean, fit_intercept=False, ax=None)

    return true_coarse, predicted_coarse, true_eval, predicted_eval


def compare_unet_to_others(args, unet_model, dataloader, device, sif_mean, sif_std, resolution_meters, min_eval_cfis_soundings, min_eval_fraction_valid,
                           coarse_set, eval_set, linear_model, mlp_model):
    """Compute detailed results. For each fine pixel, record metadata and compute predictions from
    U-Net and baseline methods. If "args.plot_examples" is True, also plots visualizations.
    """
    fine_pixels_per_eval = int(resolution_meters / METERS_PER_PIXEL)
    eval_resolution_degrees = (fine_pixels_per_eval * DEGREES_PER_PIXEL[0], fine_pixels_per_eval * DEGREES_PER_PIXEL[1])

    plot_counter = 0 # Number of plots made so far

    # Store results
    coarse_results = []
    eval_results = []

    # Set model to eval mode
    unet_model.eval()

    # Iterate over data.
    for sample in dataloader:
        with torch.set_grad_enabled(False):
            # Read input tile
            input_tiles_std = sample['input_tile'].to(device)

            # Read coarse-resolution SIF label
            true_coarse_sifs = sample['coarse_sif'].to(device)

            # Read fine-resolution SIF labels
            true_fine_sifs = sample['fine_sif'].to(device)
            valid_fine_sif_mask = torch.logical_not(sample['fine_sif_mask']).to(device)  # torch.ones_like(true_fine_sifs, dtype=bool)
            fine_soundings = sample['fine_soundings'].to(device)

            # Predict fine-resolution SIF using model
            outputs = unet_model(input_tiles_std[:, BANDS, :, :])  # predicted_fine_sifs_std: (batch size, 1, H, W)
            if type(outputs) == tuple:
                outputs = outputs[0]
            predicted_fine_sifs_std = torch.squeeze(outputs, dim=1)  # outputs[:, 0, :, :]
            predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean
            predicted_fine_sifs = predicted_fine_sifs * args.scale_predictions_by

            # For each tile, compute coarse SIF as the average SIF over all valid pixels
            predicted_coarse_sifs = sif_utils.masked_average(predicted_fine_sifs, valid_fine_sif_mask, dims_to_average=(1, 2)) # (batch size)

            # Scale predicted/true to desired eval resolution
            predicted_eval_sifs, _, _ = sif_utils.downsample_sif(predicted_fine_sifs, valid_fine_sif_mask, fine_soundings, fine_pixels_per_eval)
            true_eval_sifs, eval_fraction_valid, eval_soundings = sif_utils.downsample_sif(true_fine_sifs, valid_fine_sif_mask, fine_soundings, fine_pixels_per_eval)
            if resolution_meters > 30:
                valid_fine_sif_mask = eval_fraction_valid > 0.9

            # Filter noisy coarse tiles
            non_noisy_mask = (eval_soundings >= min_eval_cfis_soundings) & (true_eval_sifs >= MIN_SIF_CLIP) & (eval_fraction_valid >= min_eval_fraction_valid)
            non_noisy_mask_flat = non_noisy_mask.flatten()
            true_eval_sifs_filtered = true_eval_sifs.flatten()[non_noisy_mask_flat]
            predicted_eval_sifs_filtered = predicted_eval_sifs.flatten()[non_noisy_mask_flat]
            predicted_eval_sifs_filtered = torch.clamp(predicted_eval_sifs_filtered, min=MIN_SIF_CLIP)

            # Iterate through all examples in batch
            for i in range(input_tiles_std.shape[0]):
                large_tile_lat = sample['lat'][i].item()
                large_tile_lon = sample['lon'][i].item()
                large_tile_file = sample['tile_file'][i]
                date = sample['date'][i]
                input_tile = input_tiles_std[i].cpu().detach().numpy()
                valid_eval_sif_mask_tile = valid_fine_sif_mask[i].cpu().detach().numpy()
                non_noisy_mask_tile = non_noisy_mask[i].cpu().detach().numpy()
                soundings_tile = eval_soundings[i].cpu().detach().numpy()
                true_coarse_sif_tile = true_coarse_sifs[i].cpu().detach().numpy()
                true_eval_sifs_tile = true_eval_sifs[i].cpu().detach().numpy()
                fraction_valid = np.count_nonzero(valid_eval_sif_mask_tile) / true_eval_sifs_tile.size

                # Initialize tensors for Linear/MLP/U-Net pixel predictions
                predicted_eval_sifs_linear = np.zeros(valid_eval_sif_mask_tile.shape)
                predicted_eval_sifs_mlp = np.zeros(valid_eval_sif_mask_tile.shape)
                predicted_eval_sifs_unet = predicted_eval_sifs[i].cpu().detach().numpy()

                # Get (pre-computed) averages of valid eval (fine) regions, and this coarse tile
                tile_eval_averages = eval_set[eval_set['tile_file'] == large_tile_file]
                coarse_averages_pandas = coarse_set[coarse_set['tile_file'] == large_tile_file]
                assert len(coarse_averages_pandas) == 1
                # assert len(tile_eval_averages) == np.count_nonzero(valid_eval_sif_mask_tile) 
                tile_pandas_row = coarse_averages_pandas.iloc[0]
                coarse_averages = tile_pandas_row[INPUT_COLUMNS].to_numpy(copy=True).reshape(1, -1)
                true_sif = tile_pandas_row['SIF']
                tile_description = 'lat_' + str(round(large_tile_lat, 5)) + '_lon_' + str(round(large_tile_lon, 5)) + '_' + date

                # Obtain predicted COARSE SIF using different methods
                linear_predicted_sif = linear_model.predict(coarse_averages)[0]
                mlp_predicted_sif = mlp_model.predict(coarse_averages)[0]
                unet_predicted_sif = predicted_coarse_sifs[i].item()

                # Record COARSE result row
                result_row = [true_sif, linear_predicted_sif, mlp_predicted_sif, unet_predicted_sif,
                            tile_pandas_row['lon'], tile_pandas_row['lat'], 'CFIS', tile_pandas_row['date'],
                            large_tile_file, large_tile_lon, large_tile_lat] + coarse_averages.flatten().tolist() + \
                            [sample['coarse_soundings'][i].item(), fraction_valid, 'same as true_sif column']  # ignore the final column ('coarse_sif') - this is only used for finer-resolution results
                coarse_results.append(result_row)

                # # For each pixel, compute linear/MLP predictions
                # for height_idx in range(input_tile.shape[1]):
                #     for width_idx in range(input_tile.shape[2]):
                #         fine_averages = input_tile[:, height_idx, width_idx].reshape(1, -1)
                #         linear_predicted_sif = linear_model.predict(fine_averages)[0]
                #         mlp_predicted_sif = mlp_model.predict(fine_averages)[0]
                #         predicted_fine_sifs_linear[height_idx, width_idx] = linear_predicted_sif
                #         predicted_fine_sifs_mlp[height_idx, width_idx] = mlp_predicted_sif                    

                # Loop through all *FINE* pixels in this tile, compute linear/MLP predictions, store results.
                # (This is not computationally efficient)
                for idx, row in tile_eval_averages.iterrows():
                    eval_averages = row[INPUT_COLUMNS].to_numpy(copy=True).reshape(1, -1)
                    true_sif = row['SIF']

                    # Index of fine pixel within this tile
                    height_idx, width_idx = sif_utils.lat_long_to_index(row['lat'], row['lon'], # - EVAL_RES[0]/2, row['lon'] + EVAL_RES[0]/2,
                                                                        large_tile_lat + TILE_SIZE_DEGREES / 2,
                                                                        large_tile_lon - TILE_SIZE_DEGREES / 2,
                                                                        eval_resolution_degrees)

                    # # Verify that feature values in Pandas match the feature values in the input tile
                    # fine_averages_npy = input_tile[:, height_idx, width_idx]
                    # assert fine_averages_npy.size == 43
                    # assert fine_averages.size == 43
                    # # print('Fine averages npy', fine_averages_npy)
                    # # print('from pandas', fine_averages)
                    # for j in range(fine_averages_npy.size):
                    #     assert abs(fine_averages_npy[j] - fine_averages[0, j]) < 1e-6

                    # Obtain SIF predictions for this pixel
                    linear_predicted_sif = linear_model.predict(eval_averages)[0]
                    mlp_predicted_sif = mlp_model.predict(eval_averages)[0]
                    unet_predicted_sif = predicted_eval_sifs_unet[height_idx, width_idx]

                    predicted_eval_sifs_linear[height_idx, width_idx] = linear_predicted_sif
                    predicted_eval_sifs_mlp[height_idx, width_idx] = mlp_predicted_sif

                    # print('Top left lat/lon', large_tile_lat + TILE_SIZE_DEGREES / 2, large_tile_lon - TILE_SIZE_DEGREES / 2)
                    # print('Lat/lon', row['lat'] - RES[0] / 2, row['lon'] + RES[0]/2)
                    # print('Indices', height_idx, width_idx)
                    # assert valid_fine_sif_mask_tile[height_idx, width_idx]
                    # assert abs(true_sif - true_fine_sifs_tile[height_idx, width_idx]) < 1e-6
                    # assert abs(linear_predicted_sif - predicted_fine_sifs_linear[height_idx, width_idx]) < 1e-6
                    # assert abs(mlp_predicted_sif - predicted_fine_sifs_mlp[height_idx, width_idx]) < 1e-6

                    # Only record the actual result if there are enough soundings
                    # if row['num_soundings'] >= min_eval_cfis_soundings:
                    result_row = [true_sif, linear_predicted_sif, mlp_predicted_sif, unet_predicted_sif,
                                row['lon'], row['lat'], 'CFIS', row['date'],
                                large_tile_file, large_tile_lon, large_tile_lat] + eval_averages.flatten().tolist() + \
                                [eval_soundings[i, height_idx, width_idx].item(), eval_fraction_valid[i, height_idx, width_idx].item(), row['coarse_sif']]
                    eval_results.append(result_row)


                # Plot selected tiles
                if args.plot_examples and sample['fraction_valid'][i] > 0.6:
                    print('Plotting')
                    # Plot example tile
                    true_eval_sifs_tile[valid_eval_sif_mask_tile == 0] = 0
                    predicted_eval_sifs_linear[valid_eval_sif_mask_tile == 0] = 0
                    predicted_eval_sifs_mlp[valid_eval_sif_mask_tile == 0] = 0
                    predicted_eval_sifs_unet[valid_eval_sif_mask_tile == 0] = 0

                    # # If train, multiply predictions by "ratio" to ensure predicted coarse SIF matches ground-truth exactly
                    # if args.test_set == "train" and args.match_coarse_1sounding:
                    #     linear_ratio = true_sif / sif_utils.masked_average_numpy(predicted_eval_sifs_linear, valid_eval_sif_mask_tile, dims_to_average=(0, 1))
                    #     predicted_eval_sifs_linear = predicted_eval_sifs_linear * linear_ratio
                    #     mlp_ratio = true_sif / sif_utils.masked_average_numpy(predicted_eval_sifs_mlp, valid_eval_sif_mask_tile, dims_to_average=(0, 1))
                    #     predicted_eval_sifs_mlp = predicted_eval_sifs_mlp * mlp_ratio
                    #     unet_ratio = true_sif / sif_utils.masked_average_numpy(predicted_eval_sifs_unet, valid_eval_sif_mask_tile, dims_to_average=(0, 1))
                    #     predicted_eval_sifs_unet = predicted_eval_sifs_unet * unet_ratio

                    predicted_sif_tiles = [predicted_eval_sifs_linear,
                                        predicted_eval_sifs_mlp,
                                        predicted_eval_sifs_unet]
                    prediction_methods = ['Ridge', 'ANN', 'CS-SUNet']
                    average_sifs = []
                    tile_description = 'lat_' + str(round(large_tile_lat, 4)) + '_lon_' + str(round(large_tile_lon, 4)) + '_' \
                                        + date + '_res' + str(resolution_meters) + 'm_best_fine'  # _soundings' + str(min_eval_cfis_soundings) + '_fractionvalid' + str(MIN_EVAL_FRACTION_VALID)
                    tile_description_basic = 'lat_' + str(round(large_tile_lat, 4)) + '_lon_' + str(round(large_tile_lon, 4)) + '_' + date 
                    visualization_utils.plot_tile_predictions(input_tiles_std[i].cpu().detach().numpy(),
                                                tile_description,
                                                true_eval_sifs_tile, predicted_sif_tiles,
                                                valid_eval_sif_mask_tile, non_noisy_mask_tile,
                                                prediction_methods,
                                                large_tile_lon, large_tile_lat, date, TILE_SIZE_DEGREES,
                                                resolution_meters, #  soundings_tile=soundings_tile,  -- now don't include soundings tile
                                                cdl_bands=CROP_TYPE_INDICES,
                                                plot_dir=os.path.join(RESULTS_DIR, tile_description_basic))
                    plot_counter += 1
                    # if plot_counter > 5:
                    #     return None, None

    coarse_results_df = pd.DataFrame(coarse_results, columns=COLUMN_NAMES)
    eval_results_df = pd.DataFrame(eval_results, columns=COLUMN_NAMES)

    # If desired, adjust predictions on fine train set so that the average prediction over each tile matches the true coarse SIF
    if args.match_coarse and args.test_set == "train":
        eval_results_df["predicted_coarse_sif_unet"] = eval_results_df.groupby('large_tile_file')['predicted_sif_unet'].transform('mean')
        eval_results_df["predicted_sif_unet"] *= (eval_results_df["true_coarse_sif"] / eval_results_df["predicted_coarse_sif_unet"])
        eval_results_df["predicted_coarse_sif_linear"] = eval_results_df.groupby('large_tile_file')['predicted_sif_linear'].transform('mean')
        eval_results_df["predicted_sif_linear"] *= (eval_results_df["true_coarse_sif"] / eval_results_df["predicted_coarse_sif_linear"])
        eval_results_df["predicted_coarse_sif_mlp"] = eval_results_df.groupby('large_tile_file')['predicted_sif_mlp'].transform('mean')
        eval_results_df["predicted_sif_mlp"] *= (eval_results_df["true_coarse_sif"] / eval_results_df["predicted_coarse_sif_mlp"])

    # Clip low predictions to "MIN_SIF_CLIP"
    for result_column in ['predicted_sif_linear', 'predicted_sif_mlp', 'predicted_sif_unet']:
        eval_results_df[result_column] = np.clip(eval_results_df[result_column], a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)

    return coarse_results_df, eval_results_df

def main():
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
    print("Means", train_means)
    print("Stds", train_stds)
    band_means = train_means[:-1]
    sif_mean = train_means[-1]
    band_stds = train_stds[:-1]
    sif_std = train_stds[-1]

    # Constrain predicted SIF to be between certain values (unstandardized), if desired
    # Don't forget to standardize
    if MIN_SIF is not None and MAX_SIF is not None:
        min_output = (MIN_SIF - sif_mean) / sif_std
        max_output = (MAX_SIF - sif_mean) / sif_std
    else:
        min_output = None
        max_output = None


    # Read CFIS coarse metadata
    cfis_coarse_metadata = pd.read_csv(CFIS_COARSE_METADATA_FILE)

    # Only include CFIS tiles with enough valid pixels
    cfis_coarse_metadata = cfis_coarse_metadata[(cfis_coarse_metadata['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                        (cfis_coarse_metadata['SIF'] >= MIN_SIF_CLIP) &
                                        (cfis_coarse_metadata['missing_reflectance'] <= MAX_CFIS_CLOUD_COVER)]
    cfis_coarse_metadata = cfis_coarse_metadata[cfis_coarse_metadata[ALL_COVER_COLUMNS].sum(axis=1) >= 0.5]
    print('After filtering - CFIS coarse', len(cfis_coarse_metadata))

    # Record results
    for MIN_EVAL_CFIS_SOUNDINGS in MIN_EVAL_CFIS_SOUNDINGS_EXPERIMENT:
        results_row = [args.model_path, MIN_EVAL_CFIS_SOUNDINGS, MIN_EVAL_FRACTION_VALID]

        # Iterate through resolutions
        for RESOLUTION_METERS in RESOLUTIONS:
            CFIS_EVAL_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_metadata_' + str(RESOLUTION_METERS) + 'm.csv')
            COARSE_CFIS_RESULTS_CSV_FILE = os.path.join(RESULTS_DIR, 'cfis_results_' + args.model + '_coarse_' + args.test_set + '.csv')
            EVAL_CFIS_RESULTS_CSV_FILE = os.path.join(RESULTS_DIR, 'cfis_results_' + args.model + '_' + str(RESOLUTION_METERS) + 'm_' + args.test_set + '.csv')

            # Read fine metadata at particular resolution
            cfis_eval_metadata = pd.read_csv(CFIS_EVAL_METADATA_FILE)
            cfis_eval_metadata = cfis_eval_metadata[(cfis_eval_metadata['fraction_valid'] >= 0.5) &  #(cfis_eval_metadata['SIF'] >= MIN_SIF_CLIP) &
                                                    # (cfis_eval_metadata['num_soundings'] >= MIN_EVAL_CFIS_SOUNDINGS) &  # Remove this condition for plotting purposes 
                                                    (cfis_eval_metadata['tile_file'].isin(set(cfis_coarse_metadata['tile_file'])))]
            # cfis_eval_metadata = cfis_eval_metadata[cfis_eval_metadata[ALL_COVER_COLUMNS].sum(axis=1) >= 0.5]
            # if not args.plot_examples:
            #     cfis_eval_metadata = cfis_eval_metadata[(cfis_eval_metadata['num_soundings'] >= MIN_EVAL_CFIS_SOUNDINGS)]  # If not plotting, just remove the pixels with few soundings

            # Read dataset splits
            coarse_train_set = cfis_coarse_metadata[(cfis_coarse_metadata['fold'].isin(TRAIN_FOLDS)) &
                                                    (cfis_coarse_metadata['date'].isin(TRAIN_DATES))].copy()
            coarse_test_set = cfis_coarse_metadata[(cfis_coarse_metadata['fold'].isin(TEST_FOLDS)) &
                                                    (cfis_coarse_metadata['date'].isin(TEST_DATES))].copy()
            eval_test_set = cfis_eval_metadata[(cfis_eval_metadata['fold'].isin(TEST_FOLDS)) &
                                                    (cfis_eval_metadata['date'].isin(TEST_DATES))].copy()
            print('Eval metadata', len(eval_test_set))

            # Read OCO2 metadata
            oco2_metadata = pd.read_csv(OCO2_METADATA_FILE)
            oco2_metadata = oco2_metadata[(oco2_metadata['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                                            (oco2_metadata['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                                            (oco2_metadata['SIF'] >= MIN_SIF_CLIP)]
            oco2_metadata = oco2_metadata[oco2_metadata[ALL_COVER_COLUMNS].sum(axis=1) >= 0.5]

            oco2_train_set = oco2_metadata[(oco2_metadata['fold'].isin(TRAIN_FOLDS)) &
                                            (oco2_metadata['date'].isin(TRAIN_DATES))].copy()

            train_set = pd.concat([oco2_train_set, coarse_train_set])

            # # Read coarse/fine pixel averages
            # coarse_train_set = pd.read_csv(COARSE_AVERAGES_TRAIN_FILE)
            # eval_train_set = pd.read_csv(EVAL_AVERAGES_TRAIN_FILE)
            # coarse_test_set = pd.read_csv(COARSE_AVERAGES_TEST_FILE)
            # eval_test_set = pd.read_csv(EVAL_AVERAGES_TEST_FILE)

            # # Filter coarse CFIS
            # coarse_train_set = coarse_train_set[(coarse_train_set['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
            #                                     (coarse_train_set['SIF'] >= MIN_SIF_CLIP) &
            #                                     (coarse_train_set['date'].isin(TRAIN_DATES))]
            # coarse_test_set = coarse_test_set[(coarse_test_set['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
            #                                   (coarse_test_set['SIF'] >= MIN_SIF_CLIP) &
            #                                   (coarse_test_set['date'].isin(TEST_DATES))]

            # # Filter fine CFIS (note: more filtering happens with eval_results_df)
            # eval_train_set = eval_train_set[(eval_train_set['SIF'] >= MIN_SIF_CLIP) &
            #                                 (eval_train_set['date'].isin(TRAIN_DATES)) &
            #                                 (eval_train_set['num_soundings'] >= MIN_EVAL_CFIS_SOUNDINGS) &
            #                                 (eval_train_set['fraction_valid'] >= MIN_EVAL_FRACTION_VALID) &
            #                                 (eval_train_set['tile_file'].isin(set(coarse_train_set['tile_file'])))]
            # eval_test_set = eval_test_set[(eval_test_set['SIF'] >= MIN_SIF_CLIP) &
            #                                 (eval_test_set['date'].isin(TEST_DATES)) &
            #                                 (eval_test_set['num_soundings'] >= MIN_EVAL_CFIS_SOUNDINGS) &
            #                                 (eval_test_set['fraction_valid'] >= MIN_EVAL_FRACTION_VALID) &
            #                                 (eval_test_set['tile_file'].isin(set(coarse_test_set['tile_file'])))]

            # Standardize data
            for column in COLUMNS_TO_STANDARDIZE:
                idx = BAND_AVERAGES_COLUMN_ORDER.index(column)
                train_set[column] = np.clip((train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
                coarse_test_set[column] = np.clip((coarse_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
                eval_test_set[column] = np.clip((eval_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)

            # Set up image transforms / augmentations
            standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds, bands_to_transform=CONTINUOUS_INDICES)
            clip_transform = tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT, bands_to_transform=CONTINUOUS_INDICES)

            transform_list = [standardize_transform, clip_transform]
            transform = transforms.Compose(transform_list)

            # Create dataset/dataloader
            dataset = FineSIFDataset(coarse_test_set, transform, None)  # CombinedCfisOco2Dataset(coarse_train_set, None, transform, MIN_EVAL_CFIS_SOUNDINGS)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                    shuffle=False, num_workers=NUM_WORKERS)

            X_coarse_train = train_set[INPUT_COLUMNS]
            Y_coarse_train = train_set[OUTPUT_COLUMN].values.ravel()
            X_coarse_test = coarse_test_set[INPUT_COLUMNS]
            Y_coarse_test = coarse_test_set[OUTPUT_COLUMN].values.ravel()
            X_eval_test = eval_test_set[INPUT_COLUMNS]
            Y_eval_test = eval_test_set[OUTPUT_COLUMN].values.ravel()

            if not args.use_precomputed_results:
                # Train averages models, based on best parameters from "./run_baseline.sh"
                linear_model = Ridge(alpha=100).fit(X_coarse_train, Y_coarse_train)
                mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), learning_rate_init=1e-3, max_iter=10000).fit(X_coarse_train, Y_coarse_train) 

                # Initialize model
                if args.batch_norm:
                    norm_op = nn.BatchNorm2d
                else:
                    norm_op = None # nn.Identity
                if args.dropout_prob != 0:
                    dropout_op = nn.Dropout2d
                else:
                    dropout_op = None #  nn.Identity
                norm_op_kwargs = {'eps': 1e-5, 'affine': True}
                dropout_op_kwargs = {'p': args.dropout_prob, 'inplace': True}

                if args.model == 'unet2':
                    unet_model = UNet2(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS,
                                       dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                       norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                       min_output=min_output, max_output=max_output).to(device)
                elif args.model == 'unet2_contrastive':
                    unet_model = UNet2Contrastive(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS,
                                                  dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                                  norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                                  min_output=min_output, max_output=max_output).to(device)
                elif args.model == 'unet2_spectral':
                    unet_model = UNet2Spectral(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS,
                                               dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                               norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                               min_output=min_output, max_output=max_output).to(device)
                elif args.model == 'pixel_nn':
                    unet_model = PixelNN(input_channels=INPUT_CHANNELS, output_dim=OUTPUT_CHANNELS,
                                         min_output=min_output, max_output=max_output).to(device)
                elif args.model == 'unet':
                    unet_model = UNet(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS,
                                    dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                    min_output=min_output, max_output=max_output).to(device)   
                elif args.model == 'unet_contrastive':
                    unet_model = UNetContrastive(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS,
                                                dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                                min_output=min_output, max_output=max_output).to(device)
                elif args.model == 'mrg_unet' or args.model == 'mrg_unet_plus_plus':  # Experimenting with other existing U-Net and U-Net++ implementations
                    conv_op = nn.Conv2d
                    net_nonlin = nn.LeakyReLU
                    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
                    base_num_features = 30
                    num_classes = 1
                    num_pool = 5
                    conv_per_stage = 2
                    feat_map_mul_on_downscale = 2
                    if args.model == 'mrg_unet':
                        unet_model = Generic_UNet(INPUT_CHANNELS, base_num_features, num_classes, num_pool, conv_per_stage,
                                                feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                                dropout_op_kwargs, net_nonlin, net_nonlin_kwargs, False, # deep supervision
                                                False, lambda x: x, InitWeights_He(1e-2),
                                                None, None, False, True, True).to(device)
                    elif args.model == 'mrg_unet_plus_plus':
                        unet_model = Generic_UNetPlusPlus(INPUT_CHANNELS, base_num_features, num_classes, num_pool, conv_per_stage,
                                                        feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                                        dropout_op_kwargs, net_nonlin, net_nonlin_kwargs, False, # deep supervision
                                                        False, lambda x: x, InitWeights_He(1e-2),
                                                        None, None, False, True, True).to(device)
                elif args.model == 'smp_unet':
                    if args.encoder_depth == 3:
                        decoder_channels = (64, 32, 16)
                    elif args.encoder_depth == 4:
                        decoder_channels = (128, 64, 32, 16)
                    elif args.encoder_depth == 5:
                        decoder_channels = (256, 128, 64, 32, 16)
                    unet_model = smp.Unet(
                        encoder_name="resnet34",        # Encoder architecture
                        encoder_depth=args.encoder_depth,   # How many encoder blocks (between 3 and 5)
                        encoder_weights=None,           # Since our images have more than 3 channels, we cannot use pretrained weights
                        decoder_use_batchnorm=args.decoder_use_batchnorm,     # Whether decoder should use batchnorm
                        decoder_channels=decoder_channels,
                        in_channels=INPUT_CHANNELS,     # model input channels
                        classes=1,                      # model output channels
                    ).to(device)
                elif args.model == 'smp_unet_plus_plus':
                    if args.encoder_depth == 3:
                        decoder_channels = (128, 64, 32)
                    elif args.encoder_depth == 4:
                        decoder_channels = (128, 64, 32, 16)
                    elif args.encoder_depth == 5:
                        decoder_channels = (256, 128, 64, 32, 16)
                    unet_model = smp.UnetPlusPlus(
                        encoder_name="resnet34",        # Encoder architecture
                        encoder_depth=args.encoder_depth,   # How many encoder blocks (between 3 and 5)
                        encoder_weights=None,           # Since our images have more than 3 channels, we cannot use pretrained weights
                        decoder_use_batchnorm=args.decoder_use_batchnorm,   # Whether decoder should use batchnorm
                        decoder_channels=decoder_channels,
                        in_channels=INPUT_CHANNELS,     # model input channels
                        classes=1                       # model output channels
                    ).to(device)
                else:
                    print('Model type not supported', args.model)
                    exit(1)
                unet_model.load_state_dict(torch.load(args.model_path, map_location=device))

                # Initialize loss and optimizer
                criterion = nn.MSELoss(reduction='mean')

                # Quickly get summary statistics
                eval_unet_fast(args, unet_model, dataloader, criterion, device, sif_mean, sif_std, RESOLUTION_METERS, MIN_EVAL_CFIS_SOUNDINGS, MIN_EVAL_FRACTION_VALID)

                # Get detailed results (including metadata)
                coarse_results_df, eval_results_df = compare_unet_to_others(args, unet_model, dataloader, device, sif_mean, sif_std,
                                                                            RESOLUTION_METERS, MIN_EVAL_CFIS_SOUNDINGS,
                                                                            MIN_EVAL_FRACTION_VALID,
                                                                            coarse_test_set, eval_test_set, linear_model,
                                                                            mlp_model)
                coarse_results_df.to_csv(COARSE_CFIS_RESULTS_CSV_FILE)
                eval_results_df.to_csv(EVAL_CFIS_RESULTS_CSV_FILE)
            else:
                coarse_results_df = pd.read_csv(COARSE_CFIS_RESULTS_CSV_FILE)
                eval_results_df = pd.read_csv(EVAL_CFIS_RESULTS_CSV_FILE)


            # for min_eval_cfis_soundings in MIN_EVAL_CFIS_SOUNDINGS_EXPERIMENT:
            #     for min_fraction_valid_pixels in MIN_EVAL_FRACTION_VALID_EXPERIMENT:
            print('========================================== FILTER ================================================')
            print('*** Resolution', RESOLUTION_METERS)
            print('*** Min fine soundings', MIN_EVAL_CFIS_SOUNDINGS)
            print('*** Min fine fraction valid pixels', MIN_EVAL_FRACTION_VALID)
            print('==================================================================================================')
            PLOT_PREFIX = CFIS_TRUE_VS_PREDICTED_PLOT + '_res' + str(RESOLUTION_METERS) #+ '_finesoundings' + str(MIN_EVAL_CFIS_SOUNDINGS) + '_finefractionvalid' + str(MIN_EVAL_FRACTION_VALID)

            eval_results_df_filtered = eval_results_df[(eval_results_df['num_soundings'] >= MIN_EVAL_CFIS_SOUNDINGS) &
                                                        (eval_results_df['fraction_valid'] >= MIN_EVAL_FRACTION_VALID) &
                                                        (eval_results_df["true_sif"] >= MIN_SIF_CLIP)]

            print('========= Fine pixels: True vs Linear predictions ==================')
            sif_utils.print_stats(eval_results_df_filtered['true_sif'].values.ravel(), eval_results_df_filtered['predicted_sif_linear'].values.ravel(), sif_mean, ax=None) #plt.gca())
            plt.title('True vs predicted (Ridge regression)')
            plt.xlim(left=0, right=MAX_SIF_CLIP)
            plt.ylim(bottom=0, top=MAX_SIF_CLIP)
            plt.savefig(PLOT_PREFIX + '_linear.png')
            plt.close()

            print('========= Fine pixels: True vs MLP predictions ==================')
            sif_utils.print_stats(eval_results_df_filtered['true_sif'].values.ravel(), eval_results_df_filtered['predicted_sif_mlp'].values.ravel(), sif_mean, ax=None) #plt.gca())
            plt.title('True vs predicted (ANN)')
            plt.xlim(left=0, right=MAX_SIF_CLIP)
            plt.ylim(bottom=0, top=MAX_SIF_CLIP)
            plt.savefig(PLOT_PREFIX + '_mlp.png')
            plt.close()

            print('========= Fine pixels: True vs U-Net predictions ==================')
            r2, nrmse, corr = sif_utils.print_stats(eval_results_df_filtered['true_sif'].values.ravel(), eval_results_df_filtered['predicted_sif_unet'].values.ravel(), sif_mean, ax=plt.gca())
            plt.title('True vs predicted SIF (CSR-U-Net): ' + str(int(RESOLUTION_METERS)) + 'm pixels, ' + args.test_set + ' tiles')
            plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
            plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
            plt.savefig(PLOT_PREFIX + '_unet.png')
            plt.close()

            results_row.extend([nrmse, r2, corr])


            # Plot true vs. predicted for each crop on CFIS fine (for each crop)
            predictions_fine_eval = eval_results_df_filtered['predicted_sif_unet'].values.ravel()
            Y_fine_eval = eval_results_df_filtered['true_sif'].values.ravel()
            fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
            fig.suptitle('True vs predicted SIF by crop: ' + args.model)
            for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
                predicted = predictions_fine_eval[eval_results_df_filtered[crop_type] > PURE_THRESHOLD]
                true = Y_fine_eval[eval_results_df_filtered[crop_type] > PURE_THRESHOLD]
                ax = axeslist.ravel()[idx]
                print('======================= (CFIS fine) CROP: ', crop_type, '==============================')
                print(len(predicted), 'pixels that are pure', crop_type)
                # if len(predicted) >= 2:
                    # print(' ----- All crop regression ------')
                crop_r2, crop_nrmse, crop_corr = sif_utils.print_stats(true, predicted, sif_mean, ax=ax)
                ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                ax.set_title(crop_type)
                if RESOLUTION_METERS == 30:
                    results_row.append(crop_nrmse)

            plt.tight_layout()
            fig.subplots_adjust(top=0.92)
            plt.savefig(PLOT_PREFIX + '_crop_types.png')
            plt.close()

            # Print statistics and plot by date
            fig, axeslist = plt.subplots(ncols=1, nrows=len(DATES), figsize=(6, 6*len(DATES)))
            fig.suptitle('True vs predicted SIF, by date: ' + args.model)
            idx = 0
            for date in DATES:
                # Obtain global model's predictions for data points with this date
                predicted = predictions_fine_eval[eval_results_df_filtered['date'] == date]
                true = Y_fine_eval[eval_results_df_filtered['date'] == date]
                print('=================== Date ' + date + ' ======================')
                print('Number of rows', len(predicted))
                assert(len(predicted) == len(true))
                if len(predicted) < 2:
                    idx += 1
                    continue

                # Print stats (true vs predicted)
                ax = axeslist.ravel()[idx]
                sif_utils.print_stats(true, predicted, sif_mean, ax=ax)

                ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                ax.set_title(date)
                idx += 1

            plt.tight_layout()
            fig.subplots_adjust(top=0.92)
            plt.savefig(PLOT_PREFIX + '_dates.png')
            plt.close()

        if args.match_coarse and args.test_set == "train":
            results_row.append("match_coarse")

        header = ["model_path", "min_eval_cfis_soundings", "min_fraction_valid", 
                "30m_nrmse", "30m_r2", "30m_corr", "30m_grassland_nrmse", "30m_corn_nrmse", "30m_soybean_nrmse", "30m_deciduous_forest_nrmse",
                "90m_nrmse", "90m_r2", "90m_corr",
                "150m_nrmse", "150m_r2", "150m_corr",
                "300m_nrmse", "300m_r2", "300m_corr",
                "600m_nrmse", "600m_r2", "600m_corr"]
        if not os.path.isfile(RESULTS_SUMMARY_FILE):
            with open(RESULTS_SUMMARY_FILE, mode='w') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(header)
        with open(RESULTS_SUMMARY_FILE, mode='a+') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(results_row)

    


if __name__ == "__main__":
    main()

# ================================================ OLD CODE =================================================
               # exit(0)

                # sif_tiles = [true_fine_sifs_tile[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS],
                #         predicted_fine_sifs_linear[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS],
                #         predicted_fine_sifs_mlp[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS],
                #         predicted_fine_sifs_unet[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS]]
                # cdl_utils.plot_tile(input_tiles_std[i, :, height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS].cpu().detach().numpy(), 
                #         sif_tiles, plot_names, subtile_lon, subtile_lat, date, TILE_SIZE_DEGREES)




            # # Get averages of valid coarse and fine pixels (which have been pre-computed)
            # print('Large tile file', large_tile_file)
            # tile_coarse_averages = coarse_val_set.loc[coarse_val_set['tile_file'] == large_tile_file]
            # print('Tile coarse avg', len(tile_coarse_averages))
            # tile_fine_averages = fine_val_set.loc[fine_val_set['tile_file'] == large_tile_file]
            # print('Tile fine averages', len(tile_fine_averages))
            # tile_description = 'lat_' + str(round(large_tile_lat, 5)) + '_lon_' + str(round(large_tile_lon, 5)) + '_' + date

            # # Loop through all coarse pixels, compute linear/MLP predictions, store results
            # for idx, row in tile_coarse_averages.iterrows():
            #     coarse_averages = row[INPUT_COLUMNS].to_numpy(copy=True).reshape(1, -1)
            #     true_sif = row['SIF']
            #     linear_predicted_sif = linear_model.predict(coarse_averages)[0]
            #     mlp_predicted_sif = mlp_model.predict(coarse_averages)[0]

            #     # Index of coarse pixel within this tile
            #     height_idx, width_idx = sif_utils.lat_long_to_index(row['lat'], row['lon'],
            #                                                         large_tile_lat + TILE_SIZE_DEGREES / 2,
            #                                                         large_tile_lon - TILE_SIZE_DEGREES / 2,
            #                                                         COARSE_RES)

            #     unet_predicted_sif = predicted_coarse_sifs_unet[height_idx, width_idx]
            #     assert valid_coarse_sif_mask_tile[height_idx, width_idx]
            #     assert abs(true_sif - true_coarse_sifs_tile[height_idx, width_idx]) < 1e-3

            #     # Update linear/MLP prediction array
            #     predicted_coarse_sifs_linear[height_idx, width_idx] = linear_predicted_sif
            #     predicted_coarse_sifs_mlp[height_idx, width_idx] = mlp_predicted_sif

            #     # Create csv row
            #     result_row = [true_sif, linear_predicted_sif, mlp_predicted_sif, unet_predicted_sif,
            #                 row['lon'], row['lat'], 'CFIS', row['date'],
            #                 large_tile_file, large_tile_lon, large_tile_lat] + coarse_averages.flatten().tolist()
            #     coarse_results.append(result_row)

            #     # # Safety check
            #     # valid_fine_sif_mask = valid_fine_sif_mask.unsqueeze(1).expand(-1, 43, -1, -1)
            #     # fraction_valid = fraction_valid.unsqueeze(1).expand(-1, 43, -1, -1)
            #     # print('Valid fine sif mask', valid_fine_sif_mask.shape)
            #     # input_tiles_std[valid_fine_sif_mask == 0] = 0
            #     # # avg_pooled_coarse_sifs = avg_pool(input_tiles_std[i:i+1]) / fraction_valid[i:i+1]

            #     # print('Coarse averages', coarse_averages)
            #     # print('Avg pooled', avg_pooled_coarse_sifs[0, height_idx, width_idx])
            #     # # print('Averages in tile', torch.mean(input_tiles_std[i, :, COARSE_SIF_PIXELS*height_idx:COARSE_SIF_PIXELS*(height_idx+1),
            #     # #                                                      COARSE_SIF_PIXELS*width_idx:COARSE_SIF_PIXELS*(width_idx+1)], dim=(1, 2)))
            #     # exit(0)

            # # Loop through all fine pixels, compute linear/MLP predictions, store results
            # for idx, row in tile_fine_averages.iterrows():
            #     fine_averages = row[INPUT_COLUMNS].to_numpy(copy=True).reshape(1, -1)

            #     # Index of fine pixel within this tile
            #     height_idx, width_idx = sif_utils.lat_long_to_index(row['lat'] - RES[0]/2, row['lon'] + RES[0]/2,
            #                                                         large_tile_lat + TILE_SIZE_DEGREES / 2,
            #                                                         large_tile_lon - TILE_SIZE_DEGREES / 2,
            #                                                         RES)
            #     true_sif = row['SIF']
            #     linear_predicted_sif = linear_model.predict(fine_averages)[0]
            #     mlp_predicted_sif = mlp_model.predict(fine_averages)[0]
            #     unet_predicted_sif = predicted_fine_sifs_unet[height_idx, width_idx]
            #     # print('Top left lat/lon', large_tile_lat + TILE_SIZE_DEGREES / 2, large_tile_lon - TILE_SIZE_DEGREES / 2)
            #     # print('Lat/lon', row['lat'] - RES[0] / 2, row['lon'] + RES[0]/2)
            #     # print('Indices', height_idx, width_idx)
            #     # Update linear/MLP prediction array
            #     predicted_fine_sifs_linear[height_idx, width_idx] = linear_predicted_sif
            #     predicted_fine_sifs_mlp[height_idx, width_idx] = mlp_predicted_sif
            #     assert valid_fine_sif_mask_tile[height_idx, width_idx]
            #     assert abs(true_sif - true_fine_sifs_tile[height_idx, width_idx]) < 1e-6
            #     result_row = [true_sif, linear_predicted_sif, mlp_predicted_sif, unet_predicted_sif,
            #                 row['lon'], row['lat'], 'CFIS', row['date'],
            #                 large_tile_file, large_tile_lon, large_tile_lat] + fine_averages.flatten().tolist()
            #     fine_results.append(result_row)

            # # Plot selected coarse tiles
            # if i == 0:
            #     # Find index of first nonzero coarse pixel
            #     valid_coarse = np.nonzero(valid_coarse_sif_mask_tile)
            #     coarse_height_idx = valid_coarse[0][0]
            #     coarse_width_idx = valid_coarse[1][0]
            #     plot_names = ['True SIF', 'Predicted SIF (Linear)', 'Predicted SIF (ANN)', 'Predicted SIF (U-Net)']
            #     height_idx = coarse_height_idx * COARSE_SIF_PIXELS
            #     width_idx = coarse_width_idx * COARSE_SIF_PIXELS
            #     assert valid_coarse_sif_mask_tile[coarse_height_idx, coarse_width_idx]
            #     large_tile_upper_lat = large_tile_lat + TILE_SIZE_DEGREES / 2
            #     large_tile_left_lon = large_tile_lon + TILE_SIZE_DEGREES / 2
            #     subtile_lat = large_tile_upper_lat - RES[0] * (height_idx + COARSE_SIF_PIXELS / 2)
            #     subtile_lon = large_tile_left_lon + RES[1] * (width_idx + COARSE_SIF_PIXELS / 2)

            #     # Plot example tile 
            #     true_fine_sifs_tile[valid_fine_sif_mask_tile == 0] = 0
            #     sif_tiles = [true_fine_sifs_tile,
            #             predicted_fine_sifs_linear,
            #             predicted_fine_sifs_mlp,
            #             predicted_fine_sifs_unet]
            #     visualization_utils.plot_tile(input_tiles_std[i].cpu().detach().numpy(), 
            #             sif_tiles, plot_names, large_tile_lon, large_tile_lat, date, TILE_SIZE_DEGREES)

            #     # sif_tiles = [true_fine_sifs_tile[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS],
            #     #         predicted_fine_sifs_linear[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS],
            #     #         predicted_fine_sifs_mlp[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS],
            #     #         predicted_fine_sifs_unet[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS]]
            #     # cdl_utils.plot_tile(input_tiles_std[i, :, height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS].cpu().detach().numpy(), 
            #     #         sif_tiles, plot_names, subtile_lon, subtile_lat, date, TILE_SIZE_DEGREES)
                

            #     # cdl_utils.plot_tile(input_tiles_std[i].cpu().detach().numpy(), 
            #     #                     true_coarse_sifs[i].cpu().detach().numpy(),
            #     #                     true_fine_sifs[i].cpu().detach().numpy(),
            #     #                     predicted_coarse_sifs_list,
            #     #                     predicted_fine_sifs_list,
            #     #                     method_names, large_tile_lon, large_tile_lat, date,
            #     #                     TILE_SIZE_DEGREES)
