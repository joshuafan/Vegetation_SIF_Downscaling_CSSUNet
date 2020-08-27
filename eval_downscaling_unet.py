import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import simple_cnn
from reflectance_cover_sif_dataset import CFISDataset
from unet.unet_model import UNet, UNetSmall, UNet2
import cdl_utils
import sif_utils
import tile_transforms
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

# Set random seed
torch.manual_seed(0)

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "CFIS")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "cfis_tile_metadata_val_4soundings.csv")
FINE_AVERAGES_TRAIN_FILE = os.path.join(DATASET_DIR, 'cfis_fine_averages_train_4soundings.csv')
COARSE_AVERAGES_TRAIN_FILE = os.path.join(DATASET_DIR, 'cfis_coarse_averages_train_4soundings.csv')
FINE_AVERAGES_VAL_FILE = os.path.join(DATASET_DIR, 'cfis_fine_averages_val_4soundings.csv')
COARSE_AVERAGES_VAL_FILE = os.path.join(DATASET_DIR, 'cfis_coarse_averages_val_4soundings.csv')

BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "cfis_band_statistics_train_4soundings.csv")
METHOD = "8_downscaling_unet_no_batchnorm_no_augment_no_decay_4soundings"
MODEL_TYPE = "unet"
COARSE_CFIS_RESULTS_CSV_FILE = os.path.join(DATASET_DIR, 'cfis_results_' + METHOD + '_coarse.csv')
FINE_CFIS_RESULTS_CSV_FILE = os.path.join(DATASET_DIR, 'cfis_results_' + METHOD + '_fine.csv')

CFIS_TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_cfis_' + METHOD
MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)
BATCH_SIZE = 8
NUM_WORKERS = 8
MIN_SIF = None
MAX_SIF = None
MIN_SIF_CLIP = 0.2
MAX_SIF_CLIP = 1.5
MIN_INPUT = -2
MAX_INPUT = 2
MAX_PRED = 2
MAX_CFIS_SIF = 3
BANDS = list(range(0, 43))
INPUT_CHANNELS = len(BANDS)
MISSING_REFLECTANCE_IDX = -1
REDUCED_CHANNELS = 15
COARSE_SIF_PIXELS = 25
RES = (0.00026949458523585647, 0.00026949458523585647)
COARSE_RES = (RES[0] * COARSE_SIF_PIXELS, RES[1] * COARSE_SIF_PIXELS)
TILE_PIXELS = 200
TILE_SIZE_DEGREES = RES[0] * TILE_PIXELS

PURE_THRESHOLD = 0.7

COLUMN_NAMES = ['true_sif', 'predicted_sif_linear', 'predicted_sif_mlp', 'predicted_sif_unet',
                    'lon', 'lat', 'source', 'date', 'large_tile_file', 'large_tile_lon', 'large_tile_lat',
                    'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils', 'missing_reflectance']

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
PREDICTION_COLUMNS = ['predicted_sif_linear', 'predicted_sif_mlp', 'predicted_sif_unet']

# Crop types to look at when analyzing results
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest'] #, 'evergreen_forest', 'spring_wheat']

DATES = ["2016-06-15", "2016-08-01"]

# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)


# Read train/val tile metadata
val_metadata = pd.read_csv(INFO_FILE_VAL)
print('Number of val tiles', len(val_metadata))

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

# Set up image transforms / augmentations
standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds)
clip_transform = tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT)
transform_list_val = [standardize_transform, clip_transform]
val_transform = transforms.Compose(transform_list_val)

# Create dataset/dataloader
dataset = CFISDataset(val_metadata, val_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS)

# Read coarse/fine pixel averages
coarse_train_set = pd.read_csv(COARSE_AVERAGES_TRAIN_FILE)
fine_train_set = pd.read_csv(FINE_AVERAGES_TRAIN_FILE)
coarse_val_set = pd.read_csv(COARSE_AVERAGES_VAL_FILE)
fine_val_set = pd.read_csv(FINE_AVERAGES_VAL_FILE)


# Standardize data
for idx, column in enumerate(COLUMNS_TO_STANDARDIZE):
    coarse_train_set[column] = np.clip((coarse_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    fine_train_set[column] = np.clip((fine_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    coarse_val_set[column] = np.clip((coarse_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    fine_val_set[column] = np.clip((fine_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)

X_fine_train = fine_train_set[INPUT_COLUMNS]
Y_fine_train = fine_train_set[OUTPUT_COLUMN].values.ravel()
X_fine_val = fine_val_set[INPUT_COLUMNS]
Y_fine_val = fine_val_set[OUTPUT_COLUMN].values.ravel()
X_coarse_train = coarse_train_set[INPUT_COLUMNS]
Y_coarse_train = coarse_train_set[OUTPUT_COLUMN].values.ravel()
X_coarse_val = coarse_val_set[INPUT_COLUMNS]
Y_coarse_val = coarse_val_set[OUTPUT_COLUMN].values.ravel()

# Train averages models - TODO change parameters
linear_model = Ridge(alpha=0.1).fit(X_coarse_train, Y_coarse_train)
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), learning_rate_init=1e-3, max_iter=1000).fit(X_coarse_train, Y_coarse_train) 

# Initialize model
if MODEL_TYPE == 'unet_small':
    model = UNetSmall(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'pixel_nn':
    model = simple_cnn.PixelNN(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'unet2':
    model = UNet2(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'unet':
    model = UNet(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)   
else:
    print('Model type not supported')
    exit(1)

model.load_state_dict(torch.load(MODEL_FILE, map_location=device))

# Initialize loss and optimizer
criterion = nn.MSELoss(reduction='mean')

# Store results
coarse_results = []
fine_results = []
running_coarse_loss = 0
num_coarse_datapoints = 0
running_fine_loss = 0
num_fine_datapoints = 0

# Set model to eval mode
model.eval()

# Iterate over data.
for sample in dataloader:
    # Read input tile
    input_tiles_std = sample['input_tile'][:, BANDS, :, :].to(device)

    # Read coarse-resolution SIF labels
    true_coarse_sifs = sample['coarse_sif'].to(device)
    valid_coarse_sif_mask = torch.logical_not(sample['coarse_sif_mask']).to(device)  # Flatten coarse SIF mask, and flip so that valid points are True

    # Read fine-resolution SIF labels
    true_fine_sifs = sample['fine_sif'].to(device)
    valid_fine_sif_mask = torch.logical_not(sample['fine_sif_mask']).to(device)

    # Predict fine-resolution SIF using model
    predicted_fine_sifs_std = model(input_tiles_std)  # predicted_fine_sifs_std: (batch size, 1, H, W)
    predicted_fine_sifs_std = torch.squeeze(predicted_fine_sifs_std, dim=1)
    predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean

    # Zero out predicted SIFs for invalid pixels (pixels with no valid SIF label, or cloudy pixels).
    # Now, only VALID pixels contain a non-zero predicted SIF.
    predicted_fine_sifs[valid_fine_sif_mask == 0] = 0

    # For each coarse-SIF sub-region, compute the fraction of valid pixels.
    # Each square is: (# valid fine pixels) / (# total fine pixels)
    avg_pool = nn.AvgPool2d(kernel_size=COARSE_SIF_PIXELS)
    fraction_valid = avg_pool(valid_fine_sif_mask.float())
    # Average together fine SIF predictions for each coarse SIF area.
    # Each square is: (sum predicted SIF over valid fine pixels) / (# total fine pixels)
    predicted_coarse_sifs = avg_pool(predicted_fine_sifs)

    # Instead of dividing by the total number of fine pixels, divide by the number of VALID fine pixels.
    # Each square is now: (sum predicted SIF over valid fine pixels) / (# valid fine pixels), which is what we want.
    predicted_coarse_sifs = predicted_coarse_sifs / fraction_valid
    predicted_coarse_sifs[valid_coarse_sif_mask == 0] = 0

    # Extract the coarse SIF data points where we have labels, and compute loss
    valid_coarse_sif_mask_flat = valid_coarse_sif_mask.flatten()
    true_coarse_sifs_filtered = true_coarse_sifs.flatten()[valid_coarse_sif_mask_flat]
    predicted_coarse_sifs_filtered = predicted_coarse_sifs.flatten()[valid_coarse_sif_mask_flat]
    coarse_loss = criterion(true_coarse_sifs_filtered, predicted_coarse_sifs_filtered)

    # Extract the fine SIF data points where we have labels, and compute loss
    valid_fine_sif_mask_flat = valid_fine_sif_mask.flatten()
    true_fine_sifs_filtered = true_fine_sifs.flatten()[valid_fine_sif_mask_flat]
    predicted_fine_sifs_filtered = predicted_fine_sifs.flatten()[valid_fine_sif_mask_flat]
    fine_loss = criterion(true_fine_sifs_filtered, predicted_fine_sifs_filtered)

    running_coarse_loss += coarse_loss.item() * len(true_coarse_sifs_filtered)
    num_coarse_datapoints += len(true_coarse_sifs_filtered)
    running_fine_loss += fine_loss.item() * len(true_fine_sifs_filtered)
    num_fine_datapoints += len(true_fine_sifs_filtered)

    # Iterate through all examples in batch
    for i in range(input_tiles_std.shape[0]):
        large_tile_lat = sample['lat'][i].item()
        large_tile_lon = sample['lon'][i].item()
        large_tile_file = sample['tile_file'][i]
        date = sample['date'][i]
        valid_coarse_sif_mask_tile = valid_coarse_sif_mask[i].cpu().detach().numpy()
        valid_fine_sif_mask_tile = valid_fine_sif_mask[i].cpu().detach().numpy()
        true_coarse_sifs_tile = true_coarse_sifs[i].cpu().detach().numpy()
        true_fine_sifs_tile = true_fine_sifs[i].cpu().detach().numpy()
        predicted_coarse_sifs_linear = np.zeros(valid_coarse_sif_mask_tile.shape)
        predicted_coarse_sifs_mlp = np.zeros(valid_coarse_sif_mask_tile.shape)
        predicted_coarse_sifs_unet = predicted_coarse_sifs[i].cpu().detach().numpy()
        predicted_fine_sifs_linear = np.zeros(valid_fine_sif_mask_tile.shape)
        predicted_fine_sifs_mlp = np.zeros(valid_fine_sif_mask_tile.shape)
        predicted_fine_sifs_unet = predicted_fine_sifs[i].cpu().detach().numpy()

        # Get averages of valid coarse and fine pixels (which have been pre-computed)
        tile_coarse_averages = coarse_val_set.loc[coarse_val_set['tile_file'] == large_tile_file]
        tile_fine_averages = fine_val_set.loc[fine_val_set['tile_file'] == large_tile_file]
        tile_description = 'lat_' + str(round(large_tile_lat, 5)) + '_lon_' + str(round(large_tile_lon, 5)) + '_' + sample['date'][i]

        # Loop through all coarse pixels, compute linear/MLP predictions, store results
        for idx, row in tile_coarse_averages.iterrows():
            coarse_averages = row[INPUT_COLUMNS].to_numpy(copy=True).reshape(1, -1)
            true_sif = row['SIF']
            linear_predicted_sif = linear_model.predict(coarse_averages)[0]
            mlp_predicted_sif = mlp_model.predict(coarse_averages)[0]

            # Index of coarse pixel within this tile
            height_idx, width_idx = sif_utils.lat_long_to_index(row['lat'], row['lon'],
                                                                large_tile_lat + TILE_SIZE_DEGREES / 2,
                                                                large_tile_lon - TILE_SIZE_DEGREES / 2,
                                                                COARSE_RES)

            unet_predicted_sif = predicted_coarse_sifs_unet[height_idx, width_idx]
            assert valid_coarse_sif_mask_tile[height_idx, width_idx]
            assert abs(true_sif - true_coarse_sifs_tile[height_idx, width_idx]) < 1e-3

            # Update linear/MLP prediction array
            predicted_coarse_sifs_linear[height_idx, width_idx] = linear_predicted_sif
            predicted_coarse_sifs_mlp[height_idx, width_idx] = mlp_predicted_sif

            # Create csv row
            result_row = [true_sif, linear_predicted_sif, mlp_predicted_sif, unet_predicted_sif,
                          row['lon'], row['lat'], 'CFIS', row['date'],
                          large_tile_file, large_tile_lon, large_tile_lat] + coarse_averages.flatten().tolist()
            coarse_results.append(result_row)

            # # Safety check
            # valid_fine_sif_mask = valid_fine_sif_mask.unsqueeze(1).expand(-1, 43, -1, -1)
            # fraction_valid = fraction_valid.unsqueeze(1).expand(-1, 43, -1, -1)
            # print('Valid fine sif mask', valid_fine_sif_mask.shape)
            # input_tiles_std[valid_fine_sif_mask == 0] = 0
            # # avg_pooled_coarse_sifs = avg_pool(input_tiles_std[i:i+1]) / fraction_valid[i:i+1]

            # print('Coarse averages', coarse_averages)
            # print('Avg pooled', avg_pooled_coarse_sifs[0, height_idx, width_idx])
            # # print('Averages in tile', torch.mean(input_tiles_std[i, :, COARSE_SIF_PIXELS*height_idx:COARSE_SIF_PIXELS*(height_idx+1),
            # #                                                      COARSE_SIF_PIXELS*width_idx:COARSE_SIF_PIXELS*(width_idx+1)], dim=(1, 2)))
            # exit(0)

        # Loop through all fine pixels, compute linear/MLP predictions, store results
        for idx, row in tile_fine_averages.iterrows():
            fine_averages = row[INPUT_COLUMNS].to_numpy(copy=True).reshape(1, -1)

            # Index of fine pixel within this tile
            height_idx, width_idx = sif_utils.lat_long_to_index(row['lat'] - RES[0]/2, row['lon'] + RES[0]/2,
                                                                large_tile_lat + TILE_SIZE_DEGREES / 2,
                                                                large_tile_lon - TILE_SIZE_DEGREES / 2,
                                                                RES)
            true_sif = row['SIF']
            linear_predicted_sif = linear_model.predict(fine_averages)[0]
            mlp_predicted_sif = mlp_model.predict(fine_averages)[0]
            unet_predicted_sif = predicted_fine_sifs_unet[height_idx, width_idx]
            # print('Top left lat/lon', large_tile_lat + TILE_SIZE_DEGREES / 2, large_tile_lon - TILE_SIZE_DEGREES / 2)
            # print('Lat/lon', row['lat'] - RES[0] / 2, row['lon'] + RES[0]/2)
            # print('Indices', height_idx, width_idx)
            # Update linear/MLP prediction array
            predicted_fine_sifs_linear[height_idx, width_idx] = linear_predicted_sif
            predicted_fine_sifs_mlp[height_idx, width_idx] = mlp_predicted_sif
            assert valid_fine_sif_mask_tile[height_idx, width_idx]
            assert abs(true_sif - true_fine_sifs_tile[height_idx, width_idx]) < 1e-6
            result_row = [true_sif, linear_predicted_sif, mlp_predicted_sif, unet_predicted_sif,
                          row['lon'], row['lat'], 'CFIS', row['date'],
                          large_tile_file, large_tile_lon, large_tile_lat] + fine_averages.flatten().tolist()
            fine_results.append(result_row)

        # Plot selected coarse tiles
        if i == 0:
            # Find index of first nonzero coarse pixel
            valid_coarse = np.nonzero(valid_coarse_sif_mask_tile)
            coarse_height_idx = valid_coarse[0][0]
            coarse_width_idx = valid_coarse[1][0]
            plot_names = ['True SIF', 'Predicted SIF (Linear)', 'Predicted SIF (ANN)', 'Predicted SIF (U-Net)']
            height_idx = coarse_height_idx * COARSE_SIF_PIXELS
            width_idx = coarse_width_idx * COARSE_SIF_PIXELS
            assert valid_coarse_sif_mask_tile[coarse_height_idx, coarse_width_idx]
            large_tile_upper_lat = large_tile_lat + TILE_SIZE_DEGREES / 2
            large_tile_left_lon = large_tile_lon + TILE_SIZE_DEGREES / 2
            subtile_lat = large_tile_upper_lat - RES[0] * (height_idx + COARSE_SIF_PIXELS / 2)
            subtile_lon = large_tile_left_lon + RES[1] * (width_idx + COARSE_SIF_PIXELS / 2)

            true_fine_sifs_tile[valid_fine_sif_mask_tile == 0] = 0
            sif_tiles = [true_fine_sifs_tile,
                    predicted_fine_sifs_linear,
                    predicted_fine_sifs_mlp,
                    predicted_fine_sifs_unet]
            cdl_utils.plot_tile(input_tiles_std[i].cpu().detach().numpy(), 
                    sif_tiles, plot_names, large_tile_lon, large_tile_lat, date, TILE_SIZE_DEGREES)

            # sif_tiles = [true_fine_sifs_tile[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS],
            #         predicted_fine_sifs_linear[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS],
            #         predicted_fine_sifs_mlp[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS],
            #         predicted_fine_sifs_unet[height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS]]
            # cdl_utils.plot_tile(input_tiles_std[i, :, height_idx:height_idx+COARSE_SIF_PIXELS, width_idx:width_idx+COARSE_SIF_PIXELS].cpu().detach().numpy(), 
            #         sif_tiles, plot_names, subtile_lon, subtile_lat, date, TILE_SIZE_DEGREES)
            

            # cdl_utils.plot_tile(input_tiles_std[i].cpu().detach().numpy(), 
            #                     true_coarse_sifs[i].cpu().detach().numpy(),
            #                     true_fine_sifs[i].cpu().detach().numpy(),
            #                     predicted_coarse_sifs_list,
            #                     predicted_fine_sifs_list,
            #                     method_names, large_tile_lon, large_tile_lat, date,
            #                     TILE_SIZE_DEGREES)

coarse_nrmse = math.sqrt(running_coarse_loss / num_coarse_datapoints) / sif_mean
fine_nrmse = math.sqrt(running_fine_loss / num_fine_datapoints) / sif_mean
print('Coarse NRMSE (calculated from running loss)', coarse_nrmse)
print('Fine NRMSE (calculated from running loss)', fine_nrmse)

# TODO add crop-specific scatters
coarse_results_df = pd.DataFrame(coarse_results, columns=COLUMN_NAMES)
coarse_results_df[PREDICTION_COLUMNS] = coarse_results_df[PREDICTION_COLUMNS].clip(lower=MIN_SIF, upper=MAX_SIF)
fine_results_df = pd.DataFrame(fine_results, columns=COLUMN_NAMES)
fine_results_df[PREDICTION_COLUMNS] = fine_results_df[PREDICTION_COLUMNS].clip(lower=MIN_SIF, upper=MAX_SIF)
coarse_results_df.to_csv(COARSE_CFIS_RESULTS_CSV_FILE)
fine_results_df.to_csv(FINE_CFIS_RESULTS_CSV_FILE)

print('Fine results df', fine_results_df.head())
print('Len Fine results df', len(fine_results_df))

print('========= Fine pixels: True vs Linear predictions ==================')
sif_utils.print_stats(fine_results_df['true_sif'].values.ravel(), fine_results_df['predicted_sif_linear'].values.ravel(), sif_mean, ax=None) #plt.gca())
# plt.title('True vs predicted (Ridge regression)')
# plt.xlim(left=0, right=MAX_SIF_CLIP)
# plt.ylim(bottom=0, top=MAX_SIF_CLIP)
# plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_linear.png')
# plt.close()

print('========= Fine pixels: True vs MLP predictions ==================')
sif_utils.print_stats(fine_results_df['true_sif'].values.ravel(), fine_results_df['predicted_sif_mlp'].values.ravel(), sif_mean, ax=None) #plt.gca())
# plt.title('True vs predicted (ANN)')
# plt.xlim(left=0, right=MAX_SIF_CLIP)
# plt.ylim(bottom=0, top=MAX_SIF_CLIP)
# plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_mlp.png')
# plt.close()

print('========= Fine pixels: True vs U-Net predictions ==================')
sif_utils.print_stats(fine_results_df['true_sif'].values.ravel(), fine_results_df['predicted_sif_unet'].values.ravel(), sif_mean, ax=plt.gca())
plt.title('True vs predicted (U-Net)')
plt.xlim(left=0, right=MAX_SIF_CLIP)
plt.ylim(bottom=0, top=MAX_SIF_CLIP)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '.png')
plt.close()

# Plot true vs. predicted for each crop on CFIS fine (for each crop)
predictions_fine_val = fine_results_df['predicted_sif_unet'].values.ravel()
Y_fine_val = fine_results_df['true_sif'].values.ravel()
fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('True vs predicted SIF by crop: ' + METHOD)
for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
    predicted = predictions_fine_val[fine_results_df[crop_type] > PURE_THRESHOLD]
    true = Y_fine_val[fine_results_df[crop_type] > PURE_THRESHOLD]
    ax = axeslist.ravel()[idx]
    print('======================= (CFIS fine) CROP: ', crop_type, '==============================')
    print(len(predicted), 'pixels that are pure', crop_type)
    if len(predicted) >= 2:
        print(' ----- All crop regression ------')
        sif_utils.print_stats(true, predicted, sif_mean, ax=ax)
        ax.set_xlim(left=0, right=MAX_SIF_CLIP)
        ax.set_ylim(bottom=0, top=MAX_SIF_CLIP)
        ax.set_title(crop_type)

plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()

# Print statistics and plot by date
fig, axeslist = plt.subplots(ncols=1, nrows=len(DATES), figsize=(6, 6*len(DATES)))
fig.suptitle('True vs predicted SIF, by date: ' + METHOD)
idx = 0
for date in DATES:
    # Obtain global model's predictions for data points with this date
    predicted = predictions_fine_val[fine_results_df['date'] == date]
    true = Y_fine_val[fine_results_df['date'] == date]
    print('=================== Date ' + date + ' ======================')
    print('Number of rows', len(predicted))
    assert(len(predicted) == len(true))
    if len(predicted) < 2:
        idx += 1
        continue

    # Print stats (true vs predicted)
    ax = axeslist.ravel()[idx]
    sif_utils.print_stats(true, predicted, sif_mean, ax=ax)

    ax.set_xlim(left=0, right=MAX_SIF_CLIP)
    ax.set_ylim(bottom=0, top=MAX_SIF_CLIP)
    ax.set_title(date)
    idx += 1

plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_dates.png')
plt.close()




