import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from itertools import chain
import random
import time
import torch
import torchvision
import torchvision.transforms as transforms
import resnet
import torch.nn as nn
import torch.optim as optim

import simple_cnn
from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset, CombinedDataset, CombinedCfisOco2Dataset
from sklearn.metrics import mean_squared_error, r2_score

from unet.unet_model import UNet, UNetSmall, UNet2
import eval_downscaling_unet
import visualization_utils
import sif_utils
import tile_transforms

# Set random seed
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Data files
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_2degree_random0")
# INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "standardized_tiles_train.csv")
# INFO_FILE_VAL = os.path.join(DATASET_DIR, "standardized_tiles_val.csv")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
INFO_FILE_TEST = os.path.join(DATASET_DIR, "tile_info_test.csv")
# BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_pixels.csv") #"band_statistics_train.csv")

# TROPOMI settings
TILE_SIZE_DEGREES = 0.1
RES_METERS = 30 # (0.00026949458523585647, 0.00026949458523585647)

# CFIS settings
CFIS_RESOLUTION = 30
FINE_PIXELS_PER_EVAL = int(CFIS_RESOLUTION / 30)
MIN_CFIS_SOUNDINGS = 10
CFIS_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_test.csv')

# Method/model type
# METHOD = "7_unet_small_both_10000samples" #_random_output_0.01"
# METHOD = "7_unet2" #_1000samples"
# METHOD = "7_unet2_reflectance_only_aug"
# METHOD = "2e_unet_jigsaw"
# MODEL_TYPE = "unet"
# METHOD = "2f_pixel_nn"
# MODEL_TYPE = "pixel_nn"
METHOD = "2e_unet"
MODEL_TYPE = "unet"
# METHOD = "1e_unet"
# MODEL_TYPE = "unet"
# METHOD = "1e_unet2"
# MODEL_TYPE = "unet2"
# METHOD = "1f_pixel_nn"
# MODEL_TYPE = "pixel_nn"
# # METHOD = "1f_resnet18"
# MODEL_TYPE = "resnet18"
# Which sources to train on
TRAIN_SOURCES = ['TROPOMI', 'OCO2'] #, 'OCO2']
VAL_SOURCES = ['OCO2']
NUM_TROPOMI_SAMPLES_TRAIN = 1000
NUM_OCO2_SAMPLES_TRAIN = 1000
OCO2_UPDATES_PER_TROPOMI = 0.1

# Model files
PRETRAINED_UNET_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)
UNET_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)

# Results files/plots
CFIS_RESULTS_CSV_FILE = os.path.join(DATASET_DIR, 'cfis_results_' + METHOD + '.csv')
CFIS_TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_cfis_' + METHOD
LOSS_PLOT = 'loss_plots/losses_' + METHOD

# Parameters
OPTIMIZER_TYPE = "Adam"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 16
NUM_WORKERS = 4
FROM_PRETRAINED = False
MIN_SIF = None # 0 #None
MAX_SIF = None #1.5 #None
MIN_SIF_CLIP = 0.1
MIN_INPUT = -3
MAX_INPUT = 3
REMOVE_PURE_TRAIN = False #True
PURE_THRESHOLD_TRAIN = 0.6
REDUCED_CHANNELS = 10

# Which bands
# BANDS = list(range(0, 43))
#BANDS = list(range(0, 12)) + [12, 13, 14, 16] + [42]
BANDS = list(range(0, 9)) + list(range(12, 27)) + [28] + [42] 
INPUT_CHANNELS = len(BANDS)
MISSING_REFLECTANCE_IDX = -1

# Augmentations
AUGMENTATIONS = ['flip_and_rotate', 'gaussian_noise', 'jigsaw']
RESIZE_DIM = 100
NOISE = 0.01
FRACTION_OUTPUTS_TO_AVERAGE = 0.1

# Filtering
MIN_SOUNDINGS = 5
MAX_CLOUD_COVER = 0.2

# Dates
# TRAIN_TROPOMI_DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
#                        "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
#                        "2018-09-16"]
# TRAIN_OCO2_DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
#                     "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
#                     "2018-09-16"]
# TEST_DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
#               "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
#               "2018-09-16"]
TRAIN_TROPOMI_DATES = ["2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19"]
TRAIN_OCO2_DATES = ["2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19"]
TEST_DATES = ["2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19"]






# TODO should there be 2 separate models?
def train_model(model, dataloaders, cfis_dataloader, criterion, optimizer, device, sif_mean, sif_std, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_tropomi_loss = float('inf')
    best_oco2_loss = float('inf')
    best_cfis_loss = float('inf')
    train_tropomi_losses = []
    train_oco2_losses = []
    val_tropomi_losses = []
    val_oco2_losses = []
    cfis_losses = []

    print('SIF mean', sif_mean)
    print('SIF std', sif_std)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_tropomi_loss = 0.0
            running_oco2_loss = 0.0
            num_tropomi_points = 0
            num_oco2_points = 0
            all_true_tropomi_sifs = []
            all_predicted_tropomi_sifs = []
            all_true_oco2_sifs = []
            all_predicted_oco2_sifs = []

            # Iterate over data.
            # j = 0
            for sample in dataloaders[phase]:
                if 'tropomi_tile' not in sample and 'oco2_tile' not in sample:
                    print('Dataloader sample contained neither tropomi_subtiles nor oco2_subtiles :(')
                    print(sample)
                    exit(1)

                if 'tropomi_tile' in sample:
                    # Read TROPOMI tile and SIF from dataloader
                    tropomi_tiles_std = sample['tropomi_tile'][:, BANDS, :, :].to(device)  # (batch size, num channels, H, W)
                    tropomi_true_sifs = sample['tropomi_sif'].to(device)  # (batch_size)
                    assert(tropomi_tiles_std.shape[0] == tropomi_true_sifs.shape[0])

                    # tile_description = sample['tropomi_description'][1]
                    # title = tile_description + ' (SIF = ' + str(round(sample['tropomi_sif'][1].item(), 3)) + ')'
                    # sif_utils.plot_tile(tropomi_tiles_std[1].detach().numpy(), 'unet_input_' + tile_description + '.png', title=title)
                    # exit(0)

                    with torch.set_grad_enabled(phase == 'train'):
                        # print('TROPOMI tile', tropomi_tiles_std[0, :, 6, 6])
                        # print('TROPOMI tile', tropomi_tiles_std[0, :, 20, 90])
                        # print('TROPOMI tile', tropomi_tiles_std[0, :, 50, 78])
                        optimizer.zero_grad()

                        if MODEL_TYPE == 'resnet18':
                            tropomi_predicted_sifs_std = model(tropomi_tiles_std)
                            tropomi_predicted_sifs = tropomi_predicted_sifs_std * sif_std + sif_mean
                            # print('Predicted', tropomi_predicted_sifs)
                            # print('True', tropomi_true_sifs)
                        else:
                            tropomi_predicted_pixel_sifs_std = model(tropomi_tiles_std)  # tropomi_predicted_sifs_std: (batch size, 1, H, W)
                            if type(tropomi_predicted_pixel_sifs_std) == tuple:
                                tropomi_predicted_pixel_sifs_std = tropomi_predicted_pixel_sifs_std[0]
                            tropomi_predicted_pixel_sifs_std = torch.squeeze(tropomi_predicted_pixel_sifs_std, dim=1)
                            tropomi_predicted_pixel_sifs = tropomi_predicted_pixel_sifs_std * sif_std + sif_mean
                            # print('TROPOMI predicted sifs shape', tropomi_predicted_sifs.shape)
                            # print('TROPOMI predicted sifs', tropomi_predicted_sifs)

                            # Binary mask for non-cloudy pixels. (Previously, MISSING_REFLECTANCE_IDX was 1 if the pixel
                            # was cloudy, and 0 otherwise; we flip it so that it is 1 if the pixel is valid/non-cloudy.)
                            non_cloudy_pixels = torch.logical_not(tropomi_tiles_std[:, MISSING_REFLECTANCE_IDX, :, :])  # (batch size, H, W)

                            # As a regularization technique, randomly choose more pixels to ignore.
                            if phase == 'train':
                                pixels_to_include = torch.rand(non_cloudy_pixels.shape, device=device) > (1 - FRACTION_OUTPUTS_TO_AVERAGE)
                                non_cloudy_pixels = non_cloudy_pixels * pixels_to_include

                            # For each tile, take the average SIF over all valid pixels
                            tropomi_predicted_sifs = sif_utils.masked_average(tropomi_predicted_pixel_sifs, non_cloudy_pixels, dims_to_average=(1, 2)) # (batch size)
                            # tropomi_predicted_sifs = torch.clamp(tropomi_predicted_sifs, min=MIN_SIF_CLIP)

                        # Compute loss: predicted vs true SIF (standardized)
                        loss = criterion(tropomi_predicted_sifs, tropomi_true_sifs)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # visualization_utils.plot_tile(tropomi_tiles_std[0].cpu().detach().numpy(),
                    #                               non_cloudy_pixels[0].cpu().detach().numpy(),
                    #                               sample['tropomi_lon'][0].item(), sample['tropomi_lat'][0].item(),
                    #                               sample['tropomi_date'][0], TILE_SIZE_DEGREES)
                    # exit(0)

                    with torch.set_grad_enabled(False):
                        # statistics
                        running_tropomi_loss += loss.item() * len(sample['tropomi_sif'])
                        num_tropomi_points += len(sample['tropomi_sif'])
                        all_true_tropomi_sifs.append(tropomi_true_sifs.cpu().detach().numpy())
                        all_predicted_tropomi_sifs.append(tropomi_predicted_sifs.cpu().detach().numpy())
                    del(tropomi_tiles_std, tropomi_true_sifs, tropomi_predicted_sifs)

                if 'oco2_tile' in sample:
                    if (phase == 'val') or (phase == 'train' and random.random() < OCO2_UPDATES_PER_TROPOMI):
                        # Read OCO-2 input tile and SIF label
                        oco2_tiles_std = sample['oco2_tile'][:, BANDS, :, :].to(device)  # (batch size, num channels, H, W)
                        # print('OCO2 tile', oco2_tiles_std[0, :, 6, 6])
                        # print('OCO2 tile', oco2_tiles_std[0, :, 20, 90])
                        # print('OCO2 tile', oco2_tiles_std[0, :, 50, 78])
                        oco2_true_sifs = sample['oco2_sif'].to(device)  # (batch_size)
                        assert(oco2_tiles_std.shape[0] == oco2_true_sifs.shape[0])

                        with torch.set_grad_enabled(phase == 'train'):
                            optimizer.zero_grad()

                            if MODEL_TYPE == 'resnet18':
                                oco2_predicted_sifs_std = model(oco2_tiles_std)
                                oco2_predicted_sifs = oco2_predicted_sifs_std * sif_std + sif_mean
                            
                            else:
                                oco2_predicted_pixel_sifs_std = model(oco2_tiles_std)  # (batch size, 1, H, W)
                                if type(oco2_predicted_pixel_sifs_std) == tuple:
                                    oco2_predicted_pixel_sifs_std = oco2_predicted_pixel_sifs_std[0]
                                oco2_predicted_pixel_sifs_std = torch.squeeze(oco2_predicted_pixel_sifs_std, dim=1)
                                oco2_predicted_pixel_sifs = oco2_predicted_pixel_sifs_std * sif_std + sif_mean

                                # Binary mask for non-cloudy pixels
                                non_cloudy_pixels = torch.logical_not(oco2_tiles_std[:, MISSING_REFLECTANCE_IDX, :, :])  # (batch size, H, W)
                                # print('Fraction non-cloudy', torch.sum(non_cloudy_pixels).item() / torch.numel(non_cloudy_pixels))

                                # As a regularization technique, randomly choose more pixels to ignore.
                                if phase == 'train':
                                    pixels_to_include = torch.rand(non_cloudy_pixels.shape, device=device) > (1 - FRACTION_OUTPUTS_TO_AVERAGE)
                                    non_cloudy_pixels = non_cloudy_pixels * pixels_to_include
                                # print('Non cloudy pixels', torch.sum(non_cloudy_pixels))
                                oco2_predicted_sifs = sif_utils.masked_average(oco2_predicted_pixel_sifs, non_cloudy_pixels, dims_to_average=(1, 2)) # (batch size)
                                # oco2_predicted_sifs = torch.clamp(oco2_predicted_sifs, min=MIN_SIF_CLIP)
                                # print('OCO2 predicted', oco2_predicted_sifs)
                                # print('OCO2 true', oco2_true_sifs)

                            # Compute loss: predicted vs true SIF
                            loss = criterion(oco2_predicted_sifs, oco2_true_sifs)
                            if phase == 'train':
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                        with torch.set_grad_enabled(False):
                            # statistics
                            running_oco2_loss += loss.item() * len(sample['oco2_sif'])
                            num_oco2_points += len(sample['oco2_sif'])
                            all_true_oco2_sifs.append(oco2_true_sifs.cpu().detach().numpy())
                            all_predicted_oco2_sifs.append(oco2_predicted_sifs.cpu().detach().numpy())
                        del(oco2_tiles_std, oco2_true_sifs, oco2_predicted_sifs)


            # Compute NRMSE on entire TROPOMI and OCO-2 datasets
            if num_tropomi_points > 0:
                epoch_tropomi_nrmse = (math.sqrt(running_tropomi_loss / num_tropomi_points) / sif_mean).item()            
                true_tropomi = np.concatenate(all_true_tropomi_sifs)
                predicted_tropomi = np.concatenate(all_predicted_tropomi_sifs)
                print('======', phase, 'TROPOMI stats ======')
                sif_utils.print_stats(true_tropomi, predicted_tropomi, sif_mean) 
                if phase == 'train':
                    train_tropomi_losses.append(epoch_tropomi_nrmse)
                else:
                    val_tropomi_losses.append(epoch_tropomi_nrmse)

            if num_oco2_points > 0:
                epoch_oco2_nrmse = (math.sqrt(running_oco2_loss / num_oco2_points) / sif_mean).item()
                true_oco2 = np.concatenate(all_true_oco2_sifs)
                predicted_oco2 = np.concatenate(all_predicted_oco2_sifs)
                print('======', phase, 'OCO-2 stats ======')
                sif_utils.print_stats(true_oco2, predicted_oco2, sif_mean)
                if phase == 'train':
                    train_oco2_losses.append(epoch_oco2_nrmse)
                else:
                    val_oco2_losses.append(epoch_oco2_nrmse)

            # deep copy the model
            # if phase == 'val' and epoch_tropomi_nrmse < best_tropomi_loss:
            #     best_tropomi_loss = epoch_tropomi_nrmse
            if phase == 'val' and epoch_oco2_nrmse < best_oco2_loss:
                best_oco2_loss = epoch_oco2_nrmse
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), UNET_MODEL_FILE)
                # torch.save(model.state_dict(), UNET_MODEL_FILE + "_epoch" + str(epoch))
            
        # # Evaluate on CFIS
        # true_coarse, predicted_coarse, true_eval, predicted_eval = eval_downscaling_unet.eval_unet_fast(model, cfis_dataloader, criterion, device, sif_mean, sif_std, FINE_PIXELS_PER_EVAL, MIN_CFIS_SOUNDINGS)
        # cfis_nrmse = math.sqrt(mean_squared_error(true_eval, predicted_eval)) / sif_mean
        # cfis_losses.append(cfis_nrmse)
        # if cfis_nrmse < best_cfis_loss:
        #     best_cfis_loss = cfis_nrmse
            # cfis_results_df.to_csv(CFIS_RESULTS_CSV_FILE)
            # eval_unet_cfis.plot_cfis_scatters(cfis_results_df, METHOD, CFIS_TRUE_VS_PREDICTED_PLOT, MAX_PRED, MAX_CFIS_SIF, sif_mean)

        # Print elapsed time per epoch
        epoch_time = time.time() - epoch_start
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            epoch_time // 60, epoch_time % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best TROPOMI val loss: {:4f}'.format(best_tropomi_loss))
    print('Best OCO-2 val loss: {:4f}'.format(best_oco2_loss))
    print('Best CFIS loss: {:4f}'.format(best_cfis_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_tropomi_losses, train_oco2_losses, val_tropomi_losses, val_oco2_losses, cfis_losses, best_oco2_loss


# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)

# Read train/val tile metadata
train_set = pd.read_csv(INFO_FILE_TRAIN)
val_set = pd.read_csv(INFO_FILE_VAL)
test_set = pd.read_csv(INFO_FILE_TEST)
cfis_metadata = pd.read_csv(CFIS_FILE)

# Filter. Note that most of these filters are redundant with create_filtered_dataset.py
train_tropomi_set = train_set[(train_set['source'] == 'TROPOMI') &
                              (train_set['num_soundings'] >= MIN_SOUNDINGS) &
                              (train_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                              (train_set['SIF'] >= MIN_SIF_CLIP) &
                              (train_set['date'].isin(TRAIN_TROPOMI_DATES))].copy()
train_oco2_set = train_set[(train_set['source'] == 'OCO2') &
                              (train_set['num_soundings'] >= MIN_SOUNDINGS) &
                              (train_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                              (train_set['SIF'] >= MIN_SIF_CLIP) &
                              (train_set['date'].isin(TRAIN_OCO2_DATES))].copy()
val_tropomi_set = val_set[(val_set['source'] == 'TROPOMI') &
                              (val_set['num_soundings'] >= MIN_SOUNDINGS) &
                              (val_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                              (val_set['SIF'] >= MIN_SIF_CLIP) &
                              (val_set['date'].isin(TRAIN_TROPOMI_DATES))].copy()
val_oco2_set = val_set[(val_set['source'] == 'OCO2') &
                              (val_set['num_soundings'] >= MIN_SOUNDINGS) &
                              (val_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                              (val_set['SIF'] >= MIN_SIF_CLIP) &
                              (val_set['date'].isin(TRAIN_OCO2_DATES))].copy()
test_tropomi_set = test_set[(test_set['source'] == 'TROPOMI') &
                              (test_set['num_soundings'] >= MIN_SOUNDINGS) &
                              (test_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                              (test_set['SIF'] >= MIN_SIF_CLIP) &
                              (test_set['date'].isin(TRAIN_TROPOMI_DATES))].copy()
test_oco2_set = test_set[(test_set['source'] == 'OCO2') &
                              (test_set['num_soundings'] >= MIN_SOUNDINGS) &
                              (test_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
                              (test_set['SIF'] >= MIN_SIF_CLIP) &
                              (test_set['date'].isin(TEST_DATES))].copy()
train_oco2_set['SIF'] /= 1.04
val_oco2_set['SIF'] /= 1.04
test_oco2_set['SIF'] /= 1.04

# Artificially remove pure tiles
if REMOVE_PURE_TRAIN:
    train_tropomi_set = sif_utils.remove_pure_tiles(train_tropomi_set, threshold=PURE_THRESHOLD_TRAIN)
    train_oco2_set = sif_utils.remove_pure_tiles(train_oco2_set, threshold=PURE_THRESHOLD_TRAIN)

# Create shuffled train sets
# combined_tropomi_set = pd.concat([train_tropomi_set, val_tropomi_set, test_tropomi_set])
shuffled_tropomi_set = train_tropomi_set.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) #.iloc[0:NUM_TROPOMI_SAMPLES_TRAIN]
shuffled_oco2_set = train_oco2_set.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) #.iloc[0:NUM_OCO2_SAMPLES_TRAIN]

# Extract TROPOMI and OCO-2 rows, if applicable
if 'TROPOMI' in TRAIN_SOURCES:
    train_tropomi_metadata = shuffled_tropomi_set
    print('Train TROPOMI samples', len(train_tropomi_metadata))
else:
    train_tropomi_metadata = None

if 'OCO2' in TRAIN_SOURCES:
    train_oco2_metadata = shuffled_oco2_set
    print('Train OCO2 samples', len(train_oco2_metadata))
else:
    train_oco2_metadata = None



# Print params for reference
print("=========================== PARAMS ===========================")
PARAM_STRING = ''
PARAM_STRING += '============= DATASET PARAMS =============\n'
PARAM_STRING += ('Dataset dir: ' + DATASET_DIR + '\n')
PARAM_STRING += ('Train sources: ' + str(TRAIN_SOURCES) + '\n')
if 'TROPOMI' in TRAIN_SOURCES:
    PARAM_STRING += ('Train TROPOMI dates: ' + str(TRAIN_TROPOMI_DATES) + '\n')
    PARAM_STRING += ('Num TROPOMI samples: ' + str(len(train_tropomi_metadata)) + '\n')
if 'OCO2' in TRAIN_SOURCES:
    PARAM_STRING += ('Train OCO-2 dates: ' + str(TRAIN_OCO2_DATES) + '\n')
    PARAM_STRING += ('Num OCO-2 samples: ' + str(len(train_oco2_metadata)) + '; OCO-2 updates per TROPOMI: ' + str(OCO2_UPDATES_PER_TROPOMI) + '\n')
PARAM_STRING += ('Test dates: ' + str(TEST_DATES) + '\n')
PARAM_STRING += ('Min soundings: ' + str(MIN_SOUNDINGS) + '\n')
PARAM_STRING += ('Min SIF clip: ' + str(MIN_SIF_CLIP) + '\n')
PARAM_STRING += ('Max cloud cover: ' + str(MAX_CLOUD_COVER) + '\n')
PARAM_STRING += ('Train features: ' + str(BANDS) + '\n')
PARAM_STRING += ("Clip input features: " + str(MIN_INPUT) + " to " + str(MAX_INPUT) + " standard deviations from mean\n")
if REMOVE_PURE_TRAIN:
    PARAM_STRING += ('Removing pure train tiles above ' + str(PURE_THRESHOLD_TRAIN) + '\n')
PARAM_STRING += ('================= METHOD ===============\n')
if FROM_PRETRAINED:
    PARAM_STRING += ('From pretrained model: ' + os.path.basename(PRETRAINED_UNET_MODEL_FILE) + '\n')
else:
    PARAM_STRING += ("Training from scratch\n")
PARAM_STRING += ("Model name: " + os.path.basename(UNET_MODEL_FILE) + '\n')
PARAM_STRING += ("Model type: " + MODEL_TYPE + '\n')
PARAM_STRING += ("Optimizer: " + OPTIMIZER_TYPE + '\n')
PARAM_STRING += ("Learning rate: " + str(LEARNING_RATE) + '\n')
PARAM_STRING += ("Weight decay: " + str(WEIGHT_DECAY) + '\n')
PARAM_STRING += ("Num workers: " + str(NUM_WORKERS) + '\n')
PARAM_STRING += ("Batch size: " + str(BATCH_SIZE) + '\n')
PARAM_STRING += ("Num epochs: " + str(NUM_EPOCHS) + '\n')
PARAM_STRING += ("Augmentations: " + str(AUGMENTATIONS) + '\n')
if 'resize' in AUGMENTATIONS:
    PARAM_STRING += ('Resize images to: ' + str(RESIZE_DIM) + '\n')
if 'gaussian_noise' in AUGMENTATIONS:
    PARAM_STRING += ("Gaussian noise (std deviation): " + str(NOISE) + '\n')
PARAM_STRING += ("Fraction outputs to average: " + str(FRACTION_OUTPUTS_TO_AVERAGE) + '\n')
PARAM_STRING += ("SIF range: " + str(MIN_SIF) + " to " + str(MAX_SIF) + '\n')
PARAM_STRING += ("==============================================================\n")
print(PARAM_STRING)

# if 'OCO2' in VAL_SOURCES:
#     val_oco2_metadata = oco2_set[(oco2_set['source'] == 'OCO2') &
#                                 (oco2_set['num_soundings'] >= MIN_SOUNDINGS) &
#                                 (oco2_set['missing_reflectance'] <= MAX_CLOUD_COVER) &
#                                 (oco2_set['SIF'] >= MIN_SIF_CLIP) &
#                                 (oco2_set['date'].isin(TRAIN_OCO2_DATES))].copy()
#     print('Val OCO2 samples', len(val_oco2_metadata))
# else:
#     val_oco2_metadata = None

# # NOTE For technical reasons, ensure that TROPOMI and OCO2 validation data have the same length
# # (so that OCO2 points are not repeated)
# if val_tropomi_metadata is not None:
#     val_tropomi_metadata = val_tropomi_metadata.iloc[0:len(val_oco2_metadata)]

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

# Constrain predicted SIF to be between 0.2 and 1.7 (unstandardized)
# Don't forget to standardize
if MIN_SIF is not None and MAX_SIF is not None:
    min_output = (MIN_SIF - sif_mean) / sif_std
    max_output = (MAX_SIF - sif_mean) / sif_std
else:
    min_output = None
    max_output = None


# Set up image transforms / augmentations
standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds) #, min_input=MIN_INPUT, max_input=MAX_INPUT)
clip_transform = tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT)
noise_transform = tile_transforms.GaussianNoise(continuous_bands=list(range(0, 9)), standard_deviation=NOISE)
flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()
jigsaw_transform = tile_transforms.RandomJigsaw()
resize_transform = tile_transforms.ResizeTile(target_dim=[RESIZE_DIM, RESIZE_DIM])

transform_list_train = [standardize_transform, clip_transform] # [standardize_transform, noise_transform]
transform_list_val = [standardize_transform, clip_transform] #[standardize_transform]
transform_list_cfis = [standardize_transform, clip_transform]
if 'resize' in AUGMENTATIONS:
    transform_list_train.append(resize_transform)
if 'flip_and_rotate' in AUGMENTATIONS:
    transform_list_train.append(flip_and_rotate_transform)
if 'gaussian_noise' in AUGMENTATIONS:
    transform_list_train.append(noise_transform)
if 'jigsaw' in AUGMENTATIONS:
    transform_list_train.append(jigsaw_transform)
train_transform = transforms.Compose(transform_list_train)
val_transform = transforms.Compose(transform_list_val)
cfis_transform = transforms.Compose(transform_list_cfis)

# Create dataset/dataloaders
datasets = {'train': CombinedDataset(train_tropomi_metadata, train_oco2_metadata, train_transform, return_subtiles=False,
                                     subtile_dim=None, tile_file_column='tile_file'), #'standardized_tile_file'),
            'val': CombinedDataset(None, val_oco2_set, val_transform, return_subtiles=False,
                                   subtile_dim=None, tile_file_column='tile_file')} #standardized_tile_file')}  # Only validate on OCO-2
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS, timeout=100)
            for x in ['train', 'val']}
cfis_dataset = CombinedCfisOco2Dataset(cfis_metadata, None, cfis_transform, MIN_CFIS_SOUNDINGS)
cfis_dataloader = torch.utils.data.DataLoader(cfis_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS)

# Initialize model
if MODEL_TYPE == 'unet_small':
    model = UNetSmall(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'pixel_nn':
    model = simple_cnn.PixelNN(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'unet':
    model = UNet(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)   
elif MODEL_TYPE == 'unet2':
    model = UNet2(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'resnet18':
    model = resnet.resnet18(input_channels=INPUT_CHANNELS, num_classes=1,
                            min_output=min_output, max_output=max_output).to(device)
else:
    print('Model type not supported')
    exit(1)

# If we're loading a pre-trained model, read model params from file
if FROM_PRETRAINED:
    model.load_state_dict(torch.load(PRETRAINED_UNET_MODEL_FILE, map_location=device))

# Initialize loss and optimizer
criterion = nn.MSELoss(reduction='mean')
if OPTIMIZER_TYPE == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    print("Optimizer not supported")
    exit(1)

model, train_tropomi_losses, train_oco2_losses, val_tropomi_losses, val_oco2_losses, cfis_losses, best_loss = train_model(model, dataloaders, cfis_dataloader, criterion, optimizer, device, sif_mean, sif_std, num_epochs=NUM_EPOCHS)
torch.save(model.state_dict(), UNET_MODEL_FILE)
print('Best OCO2 loss', best_loss)

# Plot loss curves: NRMSE
epoch_list = range(NUM_EPOCHS)
plots = []
if 'TROPOMI' in TRAIN_SOURCES:
    print("Train TROPOMI losses:", train_tropomi_losses)
    train_tropomi_plot, = plt.plot(epoch_list, train_tropomi_losses, color='blue', label='Train TROPOMI NRMSE')
    plots.append(train_tropomi_plot)
if 'OCO2' in TRAIN_SOURCES:
    print("Train OCO2 losses:", train_oco2_losses)
    train_oco2_plot, = plt.plot(epoch_list, train_oco2_losses, color='red', label='Train OCO-2 NRMSE')
    plots.append(train_oco2_plot)
if 'TROPOMI' in VAL_SOURCES:
    print("Val TROPOMI losses:", val_tropomi_losses)
    val_tropomi_plot, = plt.plot(epoch_list, val_tropomi_losses, color='green', label='Val TROPOMI NRMSE')
    plots.append(val_tropomi_plot)
if 'OCO2' in VAL_SOURCES:
    print("Val OCO2 losses:", val_oco2_losses)
    val_oco2_plot, = plt.plot(epoch_list, val_oco2_losses, color='orange', label='Val OCO-2 NRMSE')
    plots.append(val_oco2_plot)
# print('CFIS losses', cfis_losses)
# cfis_plot, = plt.plot(epoch_list, cfis_losses, color='black', label='CFIS NRMSE (scaled)')
# plots.append(cfis_plot)

# Add legend and axis labels
plt.legend(handles=plots)
plt.xlabel('Epoch #')
plt.ylabel('NRMSE')
plt.savefig(LOSS_PLOT + '_nrmse.png') 
plt.close()

# # Plot loss curves: NRMSE
# epoch_list = range(NUM_EPOCHS)
# plots = []
# if 'OCO2' in VAL_SOURCES:
#     val_oco2_plot, = plt.plot(epoch_list, val_oco2_losses, color='orange', label='Val OCO-2 NRMSE')
#     plots.append(val_oco2_plot)
# # cfis_plot, = plt.plot(epoch_list, cfis_losses, color='black', label='CFIS NRMSE (scaled)')
# # plots.append(cfis_plot)

# # Add legend and axis labels
# plt.legend(handles=plots)
# plt.xlabel('Epoch #')
# plt.ylabel('NRMSE')
# plt.savefig(LOSS_PLOT + '_nrmse_val_oco2.png') 
# plt.close()

# # Plot val TROPOMI vs OCO-2 losses
# print('============== CFIS vs val OCO-2 losses ===============')
# sif_utils.print_stats(cfis_losses, val_oco2_losses, sif_mean, ax=plt.gca())
# plt.xlabel('Val OCO2 losses')
# plt.ylabel('CFIS losses')
# plt.title('CFIS vs val OCO-2 losses' + METHOD)
# # plt.xlim(left=0, right=0.5)
# # plt.ylim(bottom=0, top=0.5)
# plt.savefig(LOSS_PLOT + '_scatter_cfis_vs_val_oco2.png')
# plt.close()
