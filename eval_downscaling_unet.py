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
from reflectance_cover_sif_dataset import CFISDataset, CombinedCfisOco2Dataset
from unet.unet_model import UNet, UNetSmall, UNet2, UNet2PixelEmbedding
import visualization_utils
import sif_utils
import tile_transforms
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
OCO2_DIR = os.path.join(DATA_DIR, "OCO2")
RESOLUTION_METERS = 30
FINE_PIXELS_PER_EVAL = int(RESOLUTION_METERS / 30)
OCO2_METADATA_TRAIN_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_train.csv')
EVAL_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_averages_' + str(RESOLUTION_METERS) + 'm_train.csv')
EVAL_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_averages_' + str(RESOLUTION_METERS) + 'm_train.csv')
# EVAL_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_averages_' + str(RESOLUTION_METERS) + 'm_val.csv')
COARSE_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_train.csv')
COARSE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_train.csv')
# COARSE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_val.csv')

BAND_STATISTICS_FILE = os.path.join(CFIS_DIR, 'cfis_band_statistics_train.csv')


# METHOD = "10e_unet2_contrastive"
# MODEL_TYPE = "unet2_pixel_embedding"
METHOD = "9e_unet2_contrastive"
MODEL_TYPE = "unet2_pixel_embedding"
# METHOD = "9d_unet2_fully_supervised"
# MODEL_TYPE = "unet2"
COMPUTE_RESULTS = True
PLOT = True
COARSE_CFIS_RESULTS_CSV_FILE = os.path.join(CFIS_DIR, 'cfis_results_' + METHOD + '_train_coarse.csv')
EVAL_CFIS_RESULTS_CSV_FILE = os.path.join(CFIS_DIR, 'cfis_results_' + METHOD + '_' + str(RESOLUTION_METERS) + 'm_train_fine.csv')

# CFIS filtering
MIN_EVAL_CFIS_SOUNDINGS = 10
MIN_EVAL_CFIS_SOUNDINGS_EXPERIMENT = [10] #[100, 300, 1000, 3000]
MIN_EVAL_FRACTION_VALID = 0.5
MIN_EVAL_FRACTION_VALID_EXPERIMENT = [0.5] # [0.1, 0.3, 0.5, 0.7]
MIN_SIF_CLIP = 0.1
MAX_SIF_CLIP = None
MIN_COARSE_FRACTION_VALID_PIXELS = 0.1

# Dates
TRAIN_DATES = ['2016-06-15', '2016-08-01']
VAL_DATES = ['2016-06-15', '2016-08-01']

CFIS_TRUE_VS_PREDICTED_PLOT = os.path.join(DATA_DIR, "exploratory_plots/true_vs_predicted_sif_cfis_" + METHOD)
MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)
BATCH_SIZE = 128
NUM_WORKERS = 8
MIN_SIF = None
MAX_SIF = None
MIN_INPUT = -3 #-3
MAX_INPUT = 3 #3

MIN_SIF_PLOT = 0
MAX_SIF_PLOT = 2
BANDS = list(range(0, 43))
INPUT_CHANNELS = len(BANDS)
MISSING_REFLECTANCE_IDX = -1
REDUCED_CHANNELS = 10
RES = (0.00026949458523585647, 0.00026949458523585647)
EVAL_RES = (RES[0] * FINE_PIXELS_PER_EVAL, RES[1] * FINE_PIXELS_PER_EVAL)
TILE_PIXELS = 100
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
                    'lentils', 'missing_reflectance', 'num_soundings', 'fraction_valid']

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

# Read coarse/fine pixel averages
coarse_train_set = pd.read_csv(COARSE_AVERAGES_TRAIN_FILE)
eval_train_set = pd.read_csv(EVAL_AVERAGES_TRAIN_FILE)
coarse_val_set = pd.read_csv(COARSE_AVERAGES_VAL_FILE)
eval_val_set = pd.read_csv(EVAL_AVERAGES_VAL_FILE)

# Filter coarse CFIS
coarse_train_set = coarse_train_set[(coarse_train_set['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                    (coarse_train_set['SIF'] >= MIN_SIF_CLIP) &
                                    (coarse_train_set['date'].isin(TRAIN_DATES))]
coarse_val_set = coarse_val_set[(coarse_val_set['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                (coarse_val_set['SIF'] >= MIN_SIF_CLIP) &
                                (coarse_val_set['date'].isin(VAL_DATES))]
print('Coarse val set', len(coarse_val_set))

# Filter fine CFIS (note: more filtering happens with eval_results_df)
eval_train_set = eval_train_set[(eval_train_set['SIF'] >= MIN_SIF_CLIP) &
                                (eval_train_set['date'].isin(VAL_DATES)) &
                                (eval_train_set['tile_file'].isin(set(coarse_train_set['tile_file'])))]
eval_val_set = eval_val_set[(eval_val_set['SIF'] >= MIN_SIF_CLIP) &
                                (eval_val_set['date'].isin(VAL_DATES)) &
                                (eval_val_set['tile_file'].isin(set(coarse_val_set['tile_file'])))]
# Standardize data
for idx, column in enumerate(COLUMNS_TO_STANDARDIZE):
    coarse_train_set[column] = np.clip((coarse_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    eval_train_set[column] = np.clip((eval_train_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    coarse_val_set[column] = np.clip((coarse_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)
    eval_val_set[column] = np.clip((eval_val_set[column] - band_means[idx]) / band_stds[idx], a_min=MIN_INPUT, a_max=MAX_INPUT)



# Set up image transforms / augmentations
standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds)
clip_transform = tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT)
transform_list_val = [standardize_transform, clip_transform]
val_transform = transforms.Compose(transform_list_val)

# Create dataset/dataloader
dataset = CombinedCfisOco2Dataset(coarse_val_set, None, val_transform, MIN_EVAL_CFIS_SOUNDINGS)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS)


X_eval_train = eval_train_set[INPUT_COLUMNS]
Y_eval_train = eval_train_set[OUTPUT_COLUMN].values.ravel()
X_eval_val = eval_val_set[INPUT_COLUMNS]
Y_eval_val = eval_val_set[OUTPUT_COLUMN].values.ravel()
X_coarse_train = coarse_train_set[INPUT_COLUMNS]
Y_coarse_train = coarse_train_set[OUTPUT_COLUMN].values.ravel()
X_coarse_val = coarse_val_set[INPUT_COLUMNS]
Y_coarse_val = coarse_val_set[OUTPUT_COLUMN].values.ravel()


if COMPUTE_RESULTS:

    # Train averages models - TODO change parameters
    linear_model = Ridge(alpha=0.01).fit(X_coarse_train, Y_coarse_train)
    mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), learning_rate_init=1e-2, max_iter=10000).fit(X_coarse_train, Y_coarse_train) 

    # Initialize model
    if MODEL_TYPE == 'unet_small':
        model = UNetSmall(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
    elif MODEL_TYPE == 'pixel_nn':
        model = simple_cnn.PixelNN(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    elif MODEL_TYPE == 'unet2':
        model = UNet2(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
    elif MODEL_TYPE == 'unet2_pixel_embedding':
        model = UNet2PixelEmbedding(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
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
    eval_results = []
    running_coarse_loss = 0
    num_coarse_datapoints = 0
    running_eval_loss = 0
    num_eval_datapoints = 0
    all_true_eval_sifs = []
    all_true_coarse_sifs = []
    all_predicted_eval_sifs = []
    all_predicted_coarse_sifs = []

    # Set model to eval mode
    model.eval()

    # Iterate over data.
    for sample in dataloader:
        with torch.set_grad_enabled(False):
            # Read input tile
            input_tiles_std = sample['cfis_input_tile'][:, BANDS, :, :].to(device)

            # Read coarse-resolution SIF label
            true_coarse_sifs = sample['cfis_coarse_sif'].to(device)

            # Read fine-resolution SIF labels
            true_fine_sifs = sample['cfis_fine_sif'].to(device)
            valid_fine_sif_mask = torch.logical_not(sample['cfis_fine_sif_mask']).to(device)
            fine_soundings = sample['cfis_fine_soundings'].to(device)

            # Predict fine-resolution SIF using model
            predicted_fine_sifs_std, _ = model(input_tiles_std)  # predicted_fine_sifs_std: (batch size, 1, H, W)
            predicted_fine_sifs_std = torch.squeeze(predicted_fine_sifs_std, dim=1)
            predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean

            # For each tile, take the average SIF over all valid pixels
            predicted_coarse_sifs = sif_utils.masked_average(predicted_fine_sifs, valid_fine_sif_mask, dims_to_average=(1, 2)) # (batch size)

            # Compute loss (predicted vs true coarse SIF)
            coarse_loss = criterion(true_coarse_sifs, predicted_coarse_sifs)

            # Scale predicted/true to desired eval resolution
            predicted_eval_sifs, _, _ = sif_utils.downsample_sif(predicted_fine_sifs, valid_fine_sif_mask, fine_soundings, FINE_PIXELS_PER_EVAL)
            # before = time.time()
            true_eval_sifs, eval_fraction_valid, eval_soundings = sif_utils.downsample_sif(true_fine_sifs, valid_fine_sif_mask, fine_soundings, FINE_PIXELS_PER_EVAL)
            # print('Avgpool time', time.time() - before)
            # before = time.time()
            # true_eval_sifs, eval_fraction_valid, eval_soundings = sif_utils.downsample_sif_for_loop(true_fine_sifs, valid_fine_sif_mask, fine_soundings, FINE_PIXELS_PER_EVAL)
            # print('for loop time', time.time() - before)
            # print('Predicted eval sifs', predicted_eval_sifs.shape)

            # Filter noisy coarse tiles
            
            non_noisy_mask = (eval_soundings >= MIN_EVAL_CFIS_SOUNDINGS) & (true_eval_sifs >= MIN_SIF_CLIP) & (eval_fraction_valid >= MIN_EVAL_FRACTION_VALID)
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
            all_true_eval_sifs.append(true_eval_sifs_filtered.cpu().detach().numpy())
            all_true_coarse_sifs.append(true_coarse_sifs.cpu().detach().numpy())
            all_predicted_eval_sifs.append(predicted_eval_sifs_filtered.cpu().detach().numpy())
            all_predicted_coarse_sifs.append(predicted_coarse_sifs.cpu().detach().numpy())

            # Iterate through all examples in batch
            for i in range(input_tiles_std.shape[0]):
                large_tile_lat = sample['cfis_lat'][i].item()
                large_tile_lon = sample['cfis_lon'][i].item()
                large_tile_file = sample['cfis_tile_file'][i]
                date = sample['cfis_date'][i]
                input_tile = input_tiles_std[i].cpu().detach().numpy()
                valid_eval_sif_mask_tile = non_noisy_mask[i].cpu().detach().numpy()
                true_coarse_sif_tile = true_coarse_sifs[i].cpu().detach().numpy()
                true_eval_sifs_tile = true_eval_sifs[i].cpu().detach().numpy()
                predicted_eval_sifs_linear = np.zeros(valid_eval_sif_mask_tile.shape)
                predicted_eval_sifs_mlp = np.zeros(valid_eval_sif_mask_tile.shape)
                predicted_eval_sifs_unet = predicted_eval_sifs[i].cpu().detach().numpy()
                fraction_valid = np.count_nonzero(valid_eval_sif_mask_tile) / true_eval_sifs_tile.size


                # Get (pre-computed) averages of valid coarse/fine regions
                tile_eval_averages = eval_val_set[eval_val_set['tile_file'] == large_tile_file]
                # print('Tile fine averages', len(tile_eval_averages))
                coarse_averages_pandas = coarse_val_set[coarse_val_set['tile_file'] == large_tile_file]
                # print('Large tile file', large_tile_file)
                # print('Coarse avg pandas', coarse_averages_pandas)
                assert len(coarse_averages_pandas) == 1
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
                            [sample['cfis_coarse_soundings'][i].item(), fraction_valid]
                coarse_results.append(result_row)

                # # For each pixel, compute linear/MLP predictions
                # for height_idx in range(input_tile.shape[1]):
                #     for width_idx in range(input_tile.shape[2]):
                #         fine_averages = input_tile[:, height_idx, width_idx].reshape(1, -1)
                #         linear_predicted_sif = linear_model.predict(fine_averages)[0]
                #         mlp_predicted_sif = mlp_model.predict(fine_averages)[0]
                #         predicted_fine_sifs_linear[height_idx, width_idx] = linear_predicted_sif
                #         predicted_fine_sifs_mlp[height_idx, width_idx] = mlp_predicted_sif                    

                # Loop through all fine pixels, compute linear/MLP predictions, store results
                for idx, row in tile_eval_averages.iterrows():
                    eval_averages = row[INPUT_COLUMNS].to_numpy(copy=True).reshape(1, -1)

                    # Index of fine pixel within this tile
                    height_idx, width_idx = sif_utils.lat_long_to_index(row['lat'], row['lon'], # - EVAL_RES[0]/2, row['lon'] + EVAL_RES[0]/2,
                                                                        large_tile_lat + TILE_SIZE_DEGREES / 2,
                                                                        large_tile_lon - TILE_SIZE_DEGREES / 2,
                                                                        EVAL_RES)
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
                    result_row = [true_sif, linear_predicted_sif, mlp_predicted_sif, unet_predicted_sif,
                                row['lon'], row['lat'], 'CFIS', row['date'],
                                large_tile_file, large_tile_lon, large_tile_lat] + eval_averages.flatten().tolist() + \
                                [eval_soundings[i, height_idx, width_idx].item(), eval_fraction_valid[i, height_idx, width_idx].item()]
                    eval_results.append(result_row)

                # Plot selected tiles
                if PLOT: #and i % 10 == 0:
                    # Plot example tile
                    true_eval_sifs_tile[valid_eval_sif_mask_tile == 0] = 0
                    predicted_eval_sifs_linear[valid_eval_sif_mask_tile == 0] = 0
                    predicted_eval_sifs_mlp[valid_eval_sif_mask_tile == 0] = 0
                    predicted_eval_sifs_unet[valid_eval_sif_mask_tile == 0] = 0
                    predicted_sif_tiles = [predicted_eval_sifs_linear,
                                        predicted_eval_sifs_mlp,
                                        predicted_eval_sifs_unet]
                    prediction_methods = ['Linear', 'ANN', 'U-Net']
                    average_sifs = []
                    visualization_utils.plot_tile(input_tiles_std[i].cpu().detach().numpy(),
                                                true_eval_sifs_tile, predicted_sif_tiles,
                                                valid_eval_sif_mask_tile, prediction_methods, 
                                                large_tile_lon, large_tile_lat, date, TILE_SIZE_DEGREES,
                                                RESOLUTION_METERS)
 
    # coarse_nrmse = math.sqrt(running_coarse_loss / num_coarse_datapoints) / sif_mean
    # fine_nrmse = math.sqevaleval) / sif_mean
    # print('Coarse NRMSE (calculated from running loss)', coarse_nrmse)
    # print('Fine NRMSE (calculated from running loss)', fine_nrmse)

    coarse_results_df = pd.DataFrame(coarse_results, columns=COLUMN_NAMES)
    # coarse_results_df[PREDICTION_COLUMNS] = coarse_results_df[PREDICTION_COLUMNS].clip(lower=MIN_SIF, upper=MAX_SIF)
    eval_results_df = pd.DataFrame(eval_results, columns=COLUMN_NAMES)
    # fine_results_df[PREDICTION_COLUMNS] = fine_results_df[PREDICTION_COLUMNS].clip(lower=MIN_SIF, upper=MAX_SIF)    

    for result_column in ['predicted_sif_linear', 'predicted_sif_mlp', 'predicted_sif_unet']:
        eval_results_df[result_column] = np.clip(eval_results_df[result_column], a_min=MIN_SIF_CLIP, a_max=MAX_SIF_CLIP)

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
                                                   (eval_results_df['fraction_valid'] >= min_fraction_valid_pixels)]

        print('========= Fine pixels: True vs Linear predictions ==================')
        sif_utils.print_stats(eval_results_df_filtered['true_sif'].values.ravel(), eval_results_df_filtered['predicted_sif_linear'].values.ravel(), sif_mean, ax=None) #plt.gca())
        # plt.title('True vs predicted (Ridge regression)')
        # plt.xlim(left=0, right=MAX_SIF_CLIP)
        # plt.ylim(bottom=0, top=MAX_SIF_CLIP)
        # plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_linear.png')
        # plt.close()

        print('========= Fine pixels: True vs MLP predictions ==================')
        sif_utils.print_stats(eval_results_df_filtered['true_sif'].values.ravel(), eval_results_df_filtered['predicted_sif_mlp'].values.ravel(), sif_mean, ax=None) #plt.gca())
        # plt.title('True vs predicted (ANN)')
        # plt.xlim(left=0, right=MAX_SIF_CLIP)
        # plt.ylim(bottom=0, top=MAX_SIF_CLIP)
        # plt.savefig(CFIS_TRUE_VS_PREDICTED_PLOT + '_mlp.png')
        # plt.close()

        print('========= Fine pixels: True vs U-Net predictions ==================')
        sif_utils.print_stats(eval_results_df_filtered['true_sif'].values.ravel(), eval_results_df_filtered['predicted_sif_unet'].values.ravel(), sif_mean, ax=plt.gca())
        plt.title('True vs predicted (U-Net)')
        plt.xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
        plt.ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
        plt.savefig(PLOT_PREFIX + '.png')
        plt.close()

        # Plot true vs. predicted for each crop on CFIS fine (for each crop)
        predictions_fine_val = eval_results_df_filtered['predicted_sif_unet'].values.ravel()
        Y_fine_val = eval_results_df_filtered['true_sif'].values.ravel()
        fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
        fig.suptitle('True vs predicted SIF by crop: ' + METHOD)
        for idx, crop_type in enumerate(COVER_COLUMN_NAMES):
            predicted = predictions_fine_val[eval_results_df_filtered[crop_type] > PURE_THRESHOLD]
            true = Y_fine_val[eval_results_df_filtered[crop_type] > PURE_THRESHOLD]
            ax = axeslist.ravel()[idx]
            print('======================= (CFIS fine) CROP: ', crop_type, '==============================')
            print(len(predicted), 'pixels that are pure', crop_type)
            if len(predicted) >= 2:
                print(' ----- All crop regression ------')
                sif_utils.print_stats(true, predicted, sif_mean, ax=ax)
                ax.set_xlim(left=MIN_SIF_PLOT, right=MAX_SIF_PLOT)
                ax.set_ylim(bottom=MIN_SIF_PLOT, top=MAX_SIF_PLOT)
                ax.set_title(crop_type)

        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        plt.savefig(PLOT_PREFIX + '_crop_types.png')
        plt.close()

        # Print statistics and plot by date
        fig, axeslist = plt.subplots(ncols=1, nrows=len(DATES), figsize=(6, 6*len(DATES)))
        fig.suptitle('True vs predicted SIF, by date: ' + METHOD)
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
