import argparse
import copy
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import simple_cnn
from datasets import FineSIFDataset
from unet.unet_model import UNet, UNetSmall, UNet2, UNet2PixelEmbedding, UNet2Larger
import visualization_utils
import sif_utils
import tile_transforms
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor



DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF"
METADATA_DIR = os.path.join(DATA_DIR, "metadata/CFIS_OCO2_dataset")
CFIS_COARSE_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_coarse_metadata.csv')
OCO2_METADATA_FILE = os.path.join(METADATA_DIR, 'oco2_metadata.csv')
BAND_STATISTICS_FILE = os.path.join(METADATA_DIR, 'cfis_band_statistics_train.csv')

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-model', "--model", default='unet2', type=str, help='model type')
parser.add_argument('-model_path', "--model_path", type=str)
parser.add_argument('-scale_predictions_by', "--scale_predictions_by", type=float, default=1)
parser.add_argument('-use_precomputed_results', "--use_precomputed_results", default=False, action='store_true', help='Whether to use pre-computed results (instead of running dataset through model)')
parser.add_argument('-plot_examples', "--plot_examples", default=False, action='store_true', help='Whether to plot example tiles')
parser.add_argument('-seed', "--seed", default=0, type=int)
parser.add_argument('-test_set', "--test_set", choices=["train", "val", "test"])
args = parser.parse_args()

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

# METHOD = "10e_unet2_contrastive"
# MODEL_TYPE = "unet2_pixel_embedding"
# METHOD = "9e_unet2_contrastive"
# METHOD = "9e_unet2_contrastive"
# METHOD = "9d_unet2_pixel_embedding"
# MODEL_TYPE = "unet2_pixel_embedding"
# METHOD = "10d_unet2"
# MODEL_TYPE = "unet2"
# METHOD = "10d_unet2_larger_dropout"
# MODEL_TYPE = "unet2_larger"
# METHOD = "11d_unet2"
# MODEL_TYPE = "unet2"
# METHOD = "10d_unet2_9_best_val_fine"
# MODEL_TYPE = "unet2"


# METHOD = "10d_pixel_nn"
# MODEL_TYPE = "pixel_nn"
# METHOD = "2e_unet2"
# MODEL_TYPE = "unet2"
# METHOD = "9d_unet2_contrastive"
# MODEL_TYPE = "unet2_pixel_embedding"
# METHOD = "11d_unet2"
# MODEL_TYPE = "unet2"

# CFIS filtering
eps = 1e-5
MIN_EVAL_CFIS_SOUNDINGS = 30 # 10
MIN_EVAL_CFIS_SOUNDINGS_EXPERIMENT = [30] # [10, 20, 25, 30, 40, 50] #[1, 5, 10, 20, 30] #[100, 300, 1000, 3000]
MIN_EVAL_FRACTION_VALID = 0.9-eps
MIN_EVAL_FRACTION_VALID_EXPERIMENT = [0.9-eps] # [0.1, 0.3, 0.5, 0.7]
MIN_SIF_CLIP = 0.1
MAX_SIF_CLIP = None
MIN_COARSE_FRACTION_VALID_PIXELS = 0.1

# Dates
TRAIN_DATES = ['2016-06-15', '2016-08-01']
TEST_DATES = ['2016-06-15', '2016-08-01'] #['2016-06-15', '2016-08-01']

RESULTS_DIR = os.path.dirname(args.model_path)  # + "_BEST_VAL_COARSE"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
CFIS_TRUE_VS_PREDICTED_PLOT = os.path.join(RESULTS_DIR, "true_vs_predicted_sif_cfis_" + args.model)
RESULTS_SUMMARY_FILE = os.path.join("unet_results", "results_summary_EVAL_" + args.test_set + ".csv")
BATCH_SIZE = 100
NUM_WORKERS = 8
MIN_SIF = None
MAX_SIF = None
MIN_INPUT = -3
MAX_INPUT = 3
MIN_SIF_CLIP = 0.1
MIN_SIF_PLOT = 0
MAX_SIF_PLOT = 1.5
BANDS = list(range(0, 12)) + [12, 13, 14, 16, 17, 19, 23, 24, 25, 28, 34] + [42]
# BANDS = list(range(0, 9)) + list(range(12, 27)) + [28] + [42] 
# BANDS = list(range(0, 43))
RECONSTRUCTION_BANDS = list(range(0, 9)) + [12, 13, 14, 16, 17, 19, 23, 24, 25, 28, 34]
INPUT_CHANNELS = len(BANDS)
OUTPUT_CHANNELS = 1 + len(RECONSTRUCTION_BANDS)
MISSING_REFLECTANCE_IDX = -1
REDUCED_CHANNELS = None
DEGREES_PER_PIXEL = (0.00026949458523585647, 0.00026949458523585647)
METERS_PER_PIXEL = 30
RESOLUTIONS = [30, 90, 150, 300, 600]
TILE_PIXELS = 100
TILE_SIZE_DEGREES = DEGREES_PER_PIXEL[0] * TILE_PIXELS
PURE_THRESHOLD = 0.7
MAX_CFIS_CLOUD_COVER = 0.5
MIN_OCO2_SOUNDINGS = 3
MAX_OCO2_CLOUD_COVER = 0.5

ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']

COLUMN_NAMES = ['true_sif', 'predicted_sif_linear', 'predicted_sif_mlp', 'predicted_sif_unet',
                    'lon', 'lat', 'source', 'date', 'large_tile_file', 'large_tile_lon', 'large_tile_lat',
                    'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity', 'missing_reflectance',
                    'num_soundings', 'fraction_valid']
                    # 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    # 'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    # 'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    # 'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    # 'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    # 'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    # 'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    # 'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    # 'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    # 'lentils', 'missing_reflectance', 'num_soundings', 'fraction_valid']

INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity', 'missing_reflectance']
                    # 'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    # 'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    # 'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    # 'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    # 'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    # 'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    # 'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    # 'lentils', 'missing_reflectance']
OUTPUT_COLUMN = ['SIF']
COLUMNS_TO_STANDARDIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
PREDICTION_COLUMNS = ['predicted_sif_linear', 'predicted_sif_mlp', 'predicted_sif_unet']

# Crop types to look at when analyzing results
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest'] #, 'evergreen_forest', 'spring_wheat']

DATES = ["2016-06-15", "2016-08-01"]


"""
Quickly computes true vs predicted loss of U-Net on fine CFIS dataset
"""
def eval_unet_fast(args, model, dataloader, criterion, device, sif_mean, sif_std, resolution_meters, min_eval_cfis_soundings, min_eval_fraction_valid):
    fine_pixels_per_eval = int(resolution_meters / METERS_PER_PIXEL)
    print('Fine pixels per eval', fine_pixels_per_eval)

    running_coarse_loss = 0
    running_eval_loss = 0
    num_coarse_datapoints = 0
    num_eval_datapoints = 0
    all_true_coarse_sifs = []
    all_true_eval_sifs = []
    all_predicted_coarse_sifs = []
    all_predicted_eval_sifs = []

    # Set model to eval mode
    model.eval()

    # Iterate over data. TODO this is redundant with compare_unet_to_others()
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

            # Predict fine-resolution SIF using model
            outputs = model(input_tiles_std)  # predicted_fine_sifs_std: (batch size, 1, H, W)
            if type(outputs) == tuple:
                outputs = outputs[0]
            predicted_fine_sifs_std = outputs[:, 0, :, :] # torch.squeeze(predicted_fine_sifs_std, dim=1)
            predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean
            predicted_fine_sifs *= args.scale_predictions_by

            # For each tile, take the average SIF over all valid pixels
            predicted_coarse_sifs = sif_utils.masked_average(predicted_fine_sifs, valid_fine_sif_mask, dims_to_average=(1, 2)) # (batch size)

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
            predicted_eval_sifs_filtered = torch.clamp(predicted_eval_sifs_filtered, min=MIN_SIF_CLIP)
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
    print("True eval (after fast)", true_eval.shape)
    print('================== Coarse CFIS stats ======================')
    sif_utils.print_stats(true_coarse, predicted_coarse, sif_mean, fit_intercept=False, ax=None)
    print('================== Eval CFIS stats ======================')
    sif_utils.print_stats(true_eval, predicted_eval, sif_mean, fit_intercept=False, ax=None)

    return true_coarse, predicted_coarse, true_eval, predicted_eval


"""
Compute detailed results. For each fine pixel, record metadata and compute predictions from
U-Net and baseline methods. If "plot" is True, also plots visualizations.
"""
def compare_unet_to_others(args, model, dataloader, device, sif_mean, sif_std, resolution_meters, min_eval_cfis_soundings, min_eval_fraction_valid,
                           coarse_set, eval_set, linear_model, mlp_model):
    fine_pixels_per_eval = int(resolution_meters / METERS_PER_PIXEL)
    eval_resolution_degrees = (fine_pixels_per_eval * DEGREES_PER_PIXEL[0], fine_pixels_per_eval * DEGREES_PER_PIXEL[1])

    plot_counter = 0 # Number of plots made so far
    
    # Store results
    coarse_results = []
    eval_results = []

    # Set model to eval mode
    model.eval()

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
            outputs = model(input_tiles_std[:, BANDS, :, :])  # predicted_fine_sifs_std: (batch size, 1, H, W)
            if type(outputs) == tuple:
                outputs = outputs[0]
            predicted_fine_sifs_std = outputs[:, 0, :, :] # torch.squeeze(predicted_fine_sifs_std, dim=1)
            predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean
            predicted_fine_sifs = predicted_fine_sifs * args.scale_predictions_by

            # For each tile, compute coarse SIF as the average SIF over all valid pixels
            predicted_coarse_sifs = sif_utils.masked_average(predicted_fine_sifs, valid_fine_sif_mask, dims_to_average=(1, 2)) # (batch size)

            # Scale predicted/true to desired eval resolution
            predicted_eval_sifs, _, _ = sif_utils.downsample_sif(predicted_fine_sifs, valid_fine_sif_mask, fine_soundings, fine_pixels_per_eval)
            # before = time.time()
            true_eval_sifs, eval_fraction_valid, eval_soundings = sif_utils.downsample_sif(true_fine_sifs, valid_fine_sif_mask, fine_soundings, fine_pixels_per_eval)
            
            # print('Avgpool time', time.time() - before)
            # before = time.time()
            # true_eval_sifs, eval_fraction_valid, eval_soundings = sif_utils.downsample_sif_for_loop(true_fine_sifs, valid_fine_sif_mask, fine_soundings, FINE_PIXELS_PER_EVAL)
            # print('for loop time', time.time() - before)
            # print('Predicted eval sifs', predicted_eval_sifs.shape)

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
                predicted_eval_sifs_linear = np.zeros(valid_eval_sif_mask_tile.shape)
                predicted_eval_sifs_mlp = np.zeros(valid_eval_sif_mask_tile.shape)
                predicted_eval_sifs_unet = predicted_eval_sifs[i].cpu().detach().numpy()
                fraction_valid = np.count_nonzero(valid_eval_sif_mask_tile) / true_eval_sifs_tile.size

                # Get (pre-computed) averages of valid coarse/fine regions
                tile_eval_averages = eval_set[eval_set['tile_file'] == large_tile_file]
                # print('Tile fine averages', len(tile_eval_averages))
                coarse_averages_pandas = coarse_set[coarse_set['tile_file'] == large_tile_file]

                # print('Large tile file', large_tile_file)
                # print('Coarse avg pandas', coarse_averages_pandas)
                assert len(coarse_averages_pandas) == 1
                # assert len(tile_eval_averages) == np.count_nonzero(valid_eval_sif_mask_tile) 
                pandas_row = coarse_averages_pandas.iloc[0]
                coarse_averages = pandas_row[INPUT_COLUMNS].to_numpy(copy=True).reshape(1, -1)
                true_sif = pandas_row['SIF']
                # print('==========================')
                # print('Tile averages: npy', coarse_averages_npy)
                # print('Pandas', coarse_averages)
                # for idx, name in enumerate(INPUT_COLUMNS):
                #     assert abs(coarse_averages[0, idx] - coarse_averages_npy[idx]) < 1e-3

                tile_description = 'lat_' + str(round(large_tile_lat, 5)) + '_lon_' + str(round(large_tile_lon, 5)) + '_' + date

                # Obtain predicted coarse SIF using different methods
                linear_predicted_sif = linear_model.predict(coarse_averages)[0]
                mlp_predicted_sif = mlp_model.predict(coarse_averages)[0]
                unet_predicted_sif = predicted_coarse_sifs[i].item()
                result_row = [true_sif, linear_predicted_sif, mlp_predicted_sif, unet_predicted_sif,
                            pandas_row['lon'], pandas_row['lat'], 'CFIS', pandas_row['date'],
                            large_tile_file, large_tile_lon, large_tile_lat] + coarse_averages.flatten().tolist() + \
                            [sample['coarse_soundings'][i].item(), fraction_valid]
                coarse_results.append(result_row)

                # # For each pixel, compute linear/MLP predictions
                # for height_idx in range(input_tile.shape[1]):
                #     for width_idx in range(input_tile.shape[2]):
                #         fine_averages = input_tile[:, height_idx, width_idx].reshape(1, -1)
                #         linear_predicted_sif = linear_model.predict(fine_averages)[0]
                #         mlp_predicted_sif = mlp_model.predict(fine_averages)[0]
                #         predicted_fine_sifs_linear[height_idx, width_idx] = linear_predicted_sif
                #         predicted_fine_sifs_mlp[height_idx, width_idx] = mlp_predicted_sif                    

                # Loop through all *FINE* pixels, compute linear/MLP predictions, store results
                for idx, row in tile_eval_averages.iterrows():
                    eval_averages = row[INPUT_COLUMNS].to_numpy(copy=True).reshape(1, -1)

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

                    true_sif = row['SIF']
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
                    if row['num_soundings'] >= min_eval_cfis_soundings:
                        result_row = [true_sif, linear_predicted_sif, mlp_predicted_sif, unet_predicted_sif,
                                    row['lon'], row['lat'], 'CFIS', row['date'],
                                    large_tile_file, large_tile_lon, large_tile_lat] + eval_averages.flatten().tolist() + \
                                    [eval_soundings[i, height_idx, width_idx].item(), eval_fraction_valid[i, height_idx, width_idx].item()]
                        eval_results.append(result_row)

                # Plot selected tiles
                if args.plot_examples and sample['fraction_valid'][i] > 0.5:
                    print('Plotting')
                    # Plot example tile
                    true_eval_sifs_tile[valid_eval_sif_mask_tile == 0] = 0
                    predicted_eval_sifs_linear[valid_eval_sif_mask_tile == 0] = 0
                    predicted_eval_sifs_mlp[valid_eval_sif_mask_tile == 0] = 0
                    predicted_eval_sifs_unet[valid_eval_sif_mask_tile == 0] = 0
                    predicted_sif_tiles = [predicted_eval_sifs_linear,
                                        predicted_eval_sifs_mlp,
                                        predicted_eval_sifs_unet]
                    prediction_methods = ['Ridge', 'ANN', 'CSR-U-Net']
                    average_sifs = []
                    tile_description = 'lat_' + str(round(large_tile_lat, 4)) + '_lon_' + str(round(large_tile_lon, 4)) + '_' \
                                        + date + '_' + str(resolution_meters) + 'm_soundings' + str(min_eval_cfis_soundings) + '_fractionvalid' + str(MIN_EVAL_FRACTION_VALID) + '_best_fine'

                    visualization_utils.plot_tile_predictions(input_tiles_std[i].cpu().detach().numpy(),
                                                tile_description,
                                                true_eval_sifs_tile, predicted_sif_tiles,
                                                valid_eval_sif_mask_tile, non_noisy_mask_tile,
                                                prediction_methods,
                                                large_tile_lon, large_tile_lat, date, TILE_SIZE_DEGREES,
                                                resolution_meters, #  soundings_tile=soundings_tile,  -- now don't include soundings tile
                                                plot_dir=RESULTS_DIR)
                    plot_counter += 1
                    # if plot_counter > 5:
                    #     return None, None

    coarse_results_df = pd.DataFrame(coarse_results, columns=COLUMN_NAMES)
    eval_results_df = pd.DataFrame(eval_results, columns=COLUMN_NAMES)

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
    results_row = [args.model_path, MIN_EVAL_CFIS_SOUNDINGS, MIN_EVAL_FRACTION_VALID]

    # Iterate through resolutions
    for RESOLUTION_METERS in RESOLUTIONS:
        CFIS_EVAL_METADATA_FILE = os.path.join(METADATA_DIR, 'cfis_metadata_' + str(RESOLUTION_METERS) + 'm.csv')
        COARSE_CFIS_RESULTS_CSV_FILE = os.path.join(RESULTS_DIR, 'cfis_results_' + args.model + '_coarse_' + args.test_set + '.csv')
        EVAL_CFIS_RESULTS_CSV_FILE = os.path.join(RESULTS_DIR, 'cfis_results_' + args.model + '_' + str(RESOLUTION_METERS) + 'm_' + args.test_set + '.csv')

        # Read fine metadata at particular resolution
        cfis_eval_metadata = pd.read_csv(CFIS_EVAL_METADATA_FILE)
        cfis_eval_metadata = cfis_eval_metadata[(cfis_eval_metadata['fraction_valid'] >= MIN_EVAL_FRACTION_VALID) &  #(cfis_eval_metadata['SIF'] >= MIN_SIF_CLIP) &
                                                # (cfis_eval_metadata['num_soundings'] >= MIN_EVAL_CFIS_SOUNDINGS) &  # Remove this condition for plotting purposes 
                                                (cfis_eval_metadata['tile_file'].isin(set(cfis_coarse_metadata['tile_file'])))]
        # cfis_eval_metadata = cfis_eval_metadata[cfis_eval_metadata[ALL_COVER_COLUMNS].sum(axis=1) >= 0.5]
        if not args.plot_examples:
            cfis_eval_metadata = cfis_eval_metadata[(cfis_eval_metadata['num_soundings'] >= MIN_EVAL_CFIS_SOUNDINGS)]  # If not plotting, just remove the pixels with few soundings

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
        for idx, column in enumerate(COLUMNS_TO_STANDARDIZE):
            train_set[column] = np.clip((train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
            coarse_test_set[column] = np.clip((coarse_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
            eval_test_set[column] = np.clip((eval_test_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)

        # Set up image transforms / augmentations
        standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds)
        clip_transform = tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT)
        transform_list = [standardize_transform, clip_transform]
        transform = transforms.Compose(transform_list)

        # Create dataset/dataloader
        dataset = FineSIFDataset(coarse_test_set, transform)  # CombinedCfisOco2Dataset(coarse_train_set, None, transform, MIN_EVAL_CFIS_SOUNDINGS)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=NUM_WORKERS)

        X_coarse_train = train_set[INPUT_COLUMNS]
        Y_coarse_train = train_set[OUTPUT_COLUMN].values.ravel()
        X_coarse_test = coarse_test_set[INPUT_COLUMNS]
        Y_coarse_test = coarse_test_set[OUTPUT_COLUMN].values.ravel()
        X_eval_test = eval_test_set[INPUT_COLUMNS]
        Y_eval_test = eval_test_set[OUTPUT_COLUMN].values.ravel()

        if not args.use_precomputed_results:
            # Train averages models - TODO change parameters
            linear_model = Ridge(alpha=100).fit(X_coarse_train, Y_coarse_train)
            mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100), learning_rate_init=1e-3, max_iter=10000).fit(X_coarse_train, Y_coarse_train) 

            # Initialize model
            if args.model == 'unet_small':
                model = UNetSmall(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
            elif args.model == 'pixel_nn':
                model = simple_cnn.PixelNN(input_channels=INPUT_CHANNELS, output_dim=OUTPUT_CHANNELS, min_output=min_output, max_output=max_output).to(device)
            elif args.model == 'unet2':
                model = UNet2(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
            elif args.model == 'unet2_larger':
                model = UNet2Larger(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
            elif args.model == 'unet2_pixel_embedding':
                model = UNet2PixelEmbedding(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
            elif args.model == 'unet':
                model = UNet(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)   
            else:
                print('Model type not supported')
                exit(1)

            model.load_state_dict(torch.load(args.model_path, map_location=device))

            # Initialize loss and optimizer
            criterion = nn.MSELoss(reduction='mean')

            # Quickly get summary statistics
            eval_unet_fast(args, model, dataloader, criterion, device, sif_mean, sif_std, RESOLUTION_METERS, MIN_EVAL_CFIS_SOUNDINGS, MIN_EVAL_FRACTION_VALID)

            # Get detailed results (including metadata)
            coarse_results_df, eval_results_df = compare_unet_to_others(args, model, dataloader, device, sif_mean, sif_std,
                                                                        RESOLUTION_METERS, MIN_EVAL_CFIS_SOUNDINGS,
                                                                        MIN_EVAL_FRACTION_VALID,
                                                                        coarse_test_set, eval_test_set, linear_model,
                                                                        mlp_model)
            coarse_results_df.to_csv(COARSE_CFIS_RESULTS_CSV_FILE)
            eval_results_df.to_csv(EVAL_CFIS_RESULTS_CSV_FILE)
        else:
            coarse_results_df = pd.read_csv(COARSE_CFIS_RESULTS_CSV_FILE)
            eval_results_df = pd.read_csv(EVAL_CFIS_RESULTS_CSV_FILE)


        for min_eval_cfis_soundings in MIN_EVAL_CFIS_SOUNDINGS_EXPERIMENT:
            for min_fraction_valid_pixels in MIN_EVAL_FRACTION_VALID_EXPERIMENT:
                print('========================================== FILTER ================================================')
                print('*** Resolution', RESOLUTION_METERS)
                print('*** Min fine soundings', min_eval_cfis_soundings)
                print('*** Min fine fraction valid pixels', min_fraction_valid_pixels)
                print('==================================================================================================')
                PLOT_PREFIX = CFIS_TRUE_VS_PREDICTED_PLOT + '_res' + str(RESOLUTION_METERS) + '_finesoundings' + str(min_eval_cfis_soundings) + '_finefractionvalid' + str(min_fraction_valid_pixels)

                eval_results_df_filtered = eval_results_df[(eval_results_df['num_soundings'] >= min_eval_cfis_soundings) &
                                                           (eval_results_df['fraction_valid'] >= min_fraction_valid_pixels) &
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

                if RESOLUTION_METERS == 30:
                    results_row.extend([nrmse, r2, corr])
                else:
                    results_row.append(nrmse)

                # Plot true vs. predicted for each crop on CFIS fine (for each crop)
                predictions_fine_val = eval_results_df_filtered['predicted_sif_unet'].values.ravel()
                Y_fine_val = eval_results_df_filtered['true_sif'].values.ravel()
                fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
                fig.suptitle('True vs predicted SIF by crop: ' + args.model)
                for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
                    predicted = predictions_fine_val[eval_results_df_filtered[crop_type] > PURE_THRESHOLD]
                    true = Y_fine_val[eval_results_df_filtered[crop_type] > PURE_THRESHOLD]
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
                    predicted = predictions_fine_val[eval_results_df_filtered['date'] == date]
                    true = Y_fine_val[eval_results_df_filtered['date'] == date]
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

    header = ["model_path", "min_eval_cfis_soundings", "min_fraction_valid", "30m_nrmse", "30m_r2", "30m_corr", "30m_grassland_nrmse", "30m_corn_nrmse", "30m_soybean_nrmse", "30m_deciduous_forest_nrmse",
              "90m_nrmse", "150m_nrmse", "300m_nrmse", "600m_nrmse"]
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
