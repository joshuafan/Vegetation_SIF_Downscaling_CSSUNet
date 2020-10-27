import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler

import pdb
import random
import time
import torch
import torchvision
import torchvision.transforms as transforms
import resnet
import torch.nn as nn
import torch.optim as optim

from reflectance_cover_sif_dataset import CombinedDataset, SubtileListDataset, ReflectanceCoverSIFDataset
from sif_utils import get_subtiles_list
import sif_utils
import simple_cnn
import tile_transforms

# import sys
# sys.path.append('../')
# from tile2vec.src.tilenet import make_tilenet
# from embedding_to_sif_model import EmbeddingToSIFModel

# Set random seed
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Data directories
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset")
# INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "standardized_tiles_train.csv")
# INFO_FILE_VAL = os.path.join(DATASET_DIR, "standardized_tiles_val.csv")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
INFO_FILE_TEST = os.path.join(DATASET_DIR, "tile_info_test.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_pixels.csv")

# Method/model type
# METHOD = "1d_train_tropomi_subtile_resnet" #1d_train_tropomi_subtile_resnet_no_dimred"
METHOD = "2d_train_both_subtile_resnet"
# METHOD = "3d_train_oco2_subtile_resnet_500samples"
MODEL_TYPE = "resnet18"

# Which sources to train on
TRAIN_SOURCES = ["TROPOMI", "OCO2"]
VAL_SOURCES = ["OCO2"]
NUM_TROPOMI_SAMPLES_TRAIN = 1000
NUM_OCO2_SAMPLES_TRAIN = 1000
OCO2_UPDATES_PER_TROPOMI = 1

# Model files
PRETRAINED_SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)
SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)

# Loss plot file
LOSS_PLOTS_DIR = 'loss_plots'
if not os.path.exists(LOSS_PLOTS_DIR):
    os.makedirs(LOSS_PLOTS_DIR)
TRAINING_PLOT_FILE = os.path.join(LOSS_PLOTS_DIR, 'losses_' + METHOD + '.png')

# Hyperparameters
OPTIMIZER_TYPE = "Adam"
SUBTILE_DIM = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 50
BATCH_SIZE = 64  # Change
NUM_WORKERS = 8  # Change
AUGMENT = True  # Change
FROM_PRETRAINED = False #True
MIN_SIF = None
MAX_SIF = None
MIN_SIF_CLIP = 0.1
MIN_INPUT = -3
MAX_INPUT = 3
NOISE = 0.05
# If we do dimensionality reduction
# REDUCED_CHANNELS = 15
# CROP_TYPE_EMBEDDING_DIM = 10
# CROP_TYPE_START_IDX = 12

# Which bands to use
#BANDS = list(range(0, 9)) + [42] #  12)) + list(range(12, 27)) + [28] + [42] 
# BANDS = list(range(0, 12)) + [12, 13, 14, 16] + [42]
# BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42]  #list(range(0, 43))
BANDS = list(range(0, 43))
INPUT_CHANNELS = len(BANDS)

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
MIN_SOUNDINGS = 5
MAX_CLOUD_COVER = 0.2

# Print params for reference
print("=========================== PARAMS ===========================")
print("Train sources:", TRAIN_SOURCES)
print("Method:", METHOD)
print("Dataset: ", os.path.basename(DATASET_DIR))
if 'OCO2' in TRAIN_SOURCES:
    print('Num OCO2 samples:', NUM_OCO2_SAMPLES_TRAIN)
    print('OCO2 updates per TROPOMI:', OCO2_UPDATES_PER_TROPOMI)
if FROM_PRETRAINED:
    print("From pretrained model", os.path.basename(PRETRAINED_SUBTILE_SIF_MODEL_FILE))
else:
    print("Training from scratch")
print("Output model:", os.path.basename(SUBTILE_SIF_MODEL_FILE))
print("Bands:", BANDS)
print("---------------------------------")
print("Model:", MODEL_TYPE)
print("Optimizer:", OPTIMIZER_TYPE)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Num workers:", NUM_WORKERS)
print("Batch size:", BATCH_SIZE)
print("Num epochs:", NUM_EPOCHS)
print("Augment:", AUGMENT)
if AUGMENT:
    print("Gaussian noise (std deviation):", NOISE)
# print("Crop type embedding dim:", CROP_TYPE_EMBEDDING_DIM)
# print("Reduced channels:", REDUCED_CHANNELS)
print("Subtile dim:", SUBTILE_DIM)
print("Input features clipped to", MIN_INPUT, "to", MAX_INPUT, "standard deviations from mean")
print("SIF range:", MIN_SIF, "to", MAX_SIF)
print("==============================================================")



# TODO should there be 2 separate models?
def train_model(model, dataloaders, criterion, optimizer, device, 
                sif_mean, sif_std, subtile_dim, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_tropomi_loss = float('inf')
    best_oco2_loss = float('inf')
    train_tropomi_losses = []
    train_oco2_losses = []
    val_tropomi_losses = []
    val_oco2_losses = []

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
            for sample in dataloaders[phase]:
                if 'tropomi_subtiles' not in sample and 'oco2_subtiles' not in sample:
                    print('Dataloader sample contained neither tropomi_subtiles nor oco2_subtiles :(')
                    print(sample)
                    exit(1)

                if 'tropomi_subtiles' in sample:
                    # Read TROPOMI tile and SIF from dataloader
                    tropomi_subtiles_std = sample['tropomi_subtiles'].to(device)  # (batch size, num sub-tiles, num channels, H, W)
                    tropomi_true_sifs = sample['tropomi_sif'].to(device)  # (batch_size)
                    # print('TROPOMI subtiles shape', tropomi_subtiles_std.shape)
                    # exit(0)
                    assert(tropomi_subtiles_std.shape[0] == tropomi_true_sifs.shape[0])
                    # print('TROPOMI Subtiles shape', tropomi_subtiles_std.shape)
                    # Reshape into a batch of "sub-tiles"
                    input_shape = tropomi_subtiles_std.shape
                    total_num_subtiles = input_shape[0] * input_shape[1]
                    input_subtiles = tropomi_subtiles_std.view((total_num_subtiles, input_shape[2], input_shape[3], input_shape[4]))
                    # print('Subtiles input shape', input_subtiles.shape)
                    # tile_description = sample['tropomi_description'][1]
                    # title = tile_description + ' subtile #5, (SIF = ' + str(round(sample['tropomi_sif'][1].item(), 3)) + ')'
                    # sif_utils.plot_tile(input_subtiles[32+5].detach().numpy(), tile_description + '_subtile_5.png', title=title)

                    with torch.set_grad_enabled(phase == 'train'):
                        predicted_subtile_sifs_std = model(input_subtiles)

                        # Re-group subtiles into corresponding large tiles
                        # print('Subtile SIFs shape', predicted_subtile_sifs.shape)
                        predicted_subtile_sifs_std = predicted_subtile_sifs_std.view((input_shape[0], input_shape[1]))
                        # print('Reshaped subtile SIFs', predicted_subtile_sifs.shape)
                        tropomi_predicted_sifs_std = torch.mean(predicted_subtile_sifs_std, dim=1)
                        tropomi_predicted_sifs = tropomi_predicted_sifs_std * sif_std + sif_mean
                        # tropomi_predicted_sifs = torch.clamp(tropomi_predicted_sifs, min=MIN_SIF_CLIP)

                        # print('Predicted SIFs (mean)', tropomi_predicted_sifs_std.shape)

                        # Compute loss: predicted vs true SIF (standardized)
                        loss = criterion(tropomi_predicted_sifs, tropomi_true_sifs)
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # print('======= Model timing =======')
                    # print('Get batch', after_get_sample - before_start)
                    # print('Forward pass', after_forward_pass - before_model)
                    # print('Backward pass', after_step - after_forward_pass)

                    with torch.set_grad_enabled(False):
                        # statistics
                        running_tropomi_loss += loss.item() * len(sample['tropomi_sif'])
                        num_tropomi_points += len(sample['tropomi_sif'])
                        all_true_tropomi_sifs.append(tropomi_true_sifs.cpu().detach().numpy())
                        all_predicted_tropomi_sifs.append(tropomi_predicted_sifs.cpu().detach().numpy())

                if 'oco2_subtiles' in sample:
                    if (phase == 'val') or (phase == 'train' and random.random() < OCO2_UPDATES_PER_TROPOMI):
                        # Read TROPOMI tile and SIF from dataloader
                        oco2_subtiles_std = sample['oco2_subtiles'].to(device)  # (batch size, num sub-tiles, num channels, H, W)
                        oco2_true_sifs = sample['oco2_sif'].to(device)  # (batch_size)
                        assert(oco2_subtiles_std.shape[0] == oco2_true_sifs.shape[0])
 
                        # Reshape into a batch of "sub-tiles"
                        input_shape = oco2_subtiles_std.shape
                        assert(input_shape[1] == 1)
                        total_num_subtiles = input_shape[0] * input_shape[1]
                        input_subtiles = oco2_subtiles_std.view((total_num_subtiles, input_shape[2], input_shape[3], input_shape[4]))

                        with torch.set_grad_enabled(phase == 'train'):
                            oco2_predicted_sifs_std = model(input_subtiles).flatten()
                            oco2_predicted_sifs = oco2_predicted_sifs_std * sif_std + sif_mean
                            # oco2_predicted_sifs = torch.clamp(oco2_predicted_sifs, min=MIN_SIF_CLIP)

                            # Compute loss: predicted vs true SIF (standardized)
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


                # # Read batch from dataloader
                # batch_loss = 0.0
                # batch_size = len(sample['SIF'])
                # input_tiles_standardized = sample['tile'].to(device)
                # # print('Input shape', input_tiles_standardized.shape)
                # true_sifs_non_standardized = sample['SIF'].to(device)

                # # Standardize true SIF
                # true_sifs_standardized = ((true_sifs_non_standardized - sif_mean) / sif_std).to(device)

                # # For each large tile, predict SIF by passing each of its sub-tiles through the model
                # with torch.set_grad_enabled(phase == 'train'):
                    

                    # input_shape = input_tiles_standardized.shape
                    # total_num_subtiles = input_shape[0] * input_shape[1]
                    # input_subtiles = input_tiles_standardized.view((total_num_subtiles, input_shape[2], input_shape[3], input_shape[4]))
                    # # print('Subtiles input shape', input_subtiles.shape)
                    # predicted_subtile_sifs = subtile_sif_model(input_subtiles)
                    # # print('Subtile SIFs shape', predicted_subtile_sifs.shape)
                    # predicted_subtile_sifs = predicted_subtile_sifs.view((input_shape[0], input_shape[1]))
                    # # print('Reshaped subtile SIFs', predicted_subtile_sifs.shape)
                    # predicted_sifs_standardized = torch.mean(predicted_subtile_sifs, dim=1)
                    # # print('Predicted SIFs (mean)', predicted_sifs_standardized.shape)

                    # # Compute loss: predicted vs true SIF (standardized)
                    # loss = criterion(predicted_sifs_standardized, true_sifs_standardized)
                    
                    # if phase == 'train':
                    #     optimizer.zero_grad()
                    #     loss.backward()
                    #     optimizer.step()  



                    # for i in range(batch_size):
                    #     # Get list of sub-tiles in this large tile
                    #     # subtiles = get_subtiles_list(input_tiles_standardized[i, BANDS, :, :], subtile_dim, device, MAX_SUBTILE_CLOUD_COVER)
                    #     # print(' ====== Random pixels (0th sub-tile) =======')
                    #     # print(subtiles[0, :, 8, 8])
                    #     # print(subtiles[0, :, 8, 9])
                    #     # print(subtiles[0, :, 9, 9])
                    #     subtiles = input_tiles_standardized[i]
                    #     # print('Subtiles shape', subtiles.shape)
                    #     predicted_subtile_sifs = subtile_sif_model(subtiles)
                    #     #print('predicted_subtile_sifs requires grad', predicted_subtile_sifs.requires_grad)
                    #     # print('Predicted subtile SIFs', predicted_subtile_sifs[100:120] * sif_std + sif_mean)
                    #     # print(predicted_subtile_sifs.shape)
                    #     predicted_sifs_standardized[i] = torch.mean(predicted_subtile_sifs)
                    # # print('predicted_sifs_standardized requires grad', predicted_sifs_standardized.requires_grad)
                    # # print('Shape of predicted total SIFs', predicted_sifs_standardized.shape)
                    # # print('Shape of true total SIFs', true_sifs_standardized.shape)

                    # # Compute loss: predicted vs true SIF (standardized)
                    # loss = criterion(predicted_sifs_standardized, true_sifs_standardized)
                    # batch_loss += loss
                    # # backward + optimize only if in training phase, and if we've reached the end of a batch
                    # if phase == 'train' and j % BATCH_SIZE == 0:
                    #     average_loss = batch_loss / BATCH_SIZE
                    #     optimizer.zero_grad()
                    #     average_loss.backward()
                    #     optimizer.step()
                    #     batch_loss = 0.0
                    #     #print("Conv1 grad after backward", subtile_sif_model.conv1.bias.grad)   

                # with torch.set_grad_enabled(False):
                #     # statistics
                #     predicted_sifs_non_standardized = torch.tensor(predicted_sifs_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                #     non_standardized_loss = criterion(predicted_sifs_non_standardized, true_sifs_non_standardized)
                #     j += 1
                #     if j % 2000 == 0:
                #         print('========================')
                #         print('*** >>> predicted subtile sifs for 0-4th', predicted_subtile_sifs.flatten()[0:5] * sif_std + sif_mean)
                #         print('*** Predicted', predicted_sifs_non_standardized[0:20])
                #         print('*** True', true_sifs_non_standardized[0:20])
                #         print('*** batch loss', (math.sqrt(non_standardized_loss.item()) / sif_mean).item())
                #     running_loss += non_standardized_loss.item() * len(sample['SIF'])
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
            if phase == 'val' and epoch_tropomi_nrmse < best_tropomi_loss:
                best_tropomi_loss = epoch_tropomi_nrmse
            if phase == 'val' and epoch_oco2_nrmse < best_oco2_loss:
                best_oco2_loss = epoch_oco2_nrmse
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), SUBTILE_SIF_MODEL_FILE)


        # Print elapsed time per epoch
        epoch_time = time.time() - epoch_start
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            epoch_time // 60, epoch_time % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best OCO-2 val loss: {:3f}'.format(best_oco2_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_tropomi_losses, train_oco2_losses, val_tropomi_losses, val_oco2_losses, best_oco2_loss


def main():
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
    train_oco2_set['SIF'] /= 1.03
    val_oco2_set['SIF'] /= 1.03
    test_oco2_set['SIF'] /= 1.03

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

    transform_list_train = [standardize_transform, clip_transform]
    transform_list_val = [standardize_transform, clip_transform]
    if AUGMENT:
        transform_list_train += [flip_and_rotate_transform, noise_transform]
    train_transform = transforms.Compose(transform_list_train)
    val_transform = transforms.Compose(transform_list_val)

    datasets = {'train': CombinedDataset(train_tropomi_metadata, train_oco2_metadata, train_transform, return_subtiles=True,
                                         subtile_dim=SUBTILE_DIM, tile_file_column='tile_file'), #'standardized_subtiles_file'),
                                         #tile_file_column='tile_file'),
                'val': CombinedDataset(None, val_oco2_set, val_transform, return_subtiles=True,
                                       subtile_dim=SUBTILE_DIM, tile_file_column='tile_file')} #standardized_subtiles_file')}  
                                       #tile_file_column='tile_file')}  # Only validate on OCO-2
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=NUM_WORKERS)
                for x in ['train', 'val']}


    # Set up Datasets and Dataloaders
    # datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, transform_with_noise),
    #             'val': ReflectanceCoverSIFDataset(val_metadata, transform)}
    # datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, transform=transform, tile_file_column='standardized_subtiles_file'),
    #             'val': ReflectanceCoverSIFDataset(val_metadata, transform=None, tile_file_column='standardized_subtiles_file')}
    # datasets = {'train': SubtileListDataset(train_metadata, transform=None, tile_file_column='standardized_subtiles_file', num_subtiles=NUM_SUBTILES),
    #             'val': SubtileListDataset(val_metadata, transform=None, tile_file_column='standardized_subtiles_file', num_subtiles=NUM_SUBTILES)}


    print("Dataloaders")

    if MODEL_TYPE == 'resnet18':
        subtile_sif_model = resnet.resnet18(input_channels=INPUT_CHANNELS, num_classes=1,
                                            min_output=min_output, max_output=max_output).to(device)
        #reduced_channels=REDUCED_CHANNELS, crop_type_embedding_dim=CROP_TYPE_EMBEDDING_DIM,
        #subtile_sif_model = small_resnet.resnet18(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
    # elif MODEL_TYPE == 'simple_cnn':
    #     subtile_sif_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    # elif MODEL_TYPE == 'simple_cnn_small':
    #     subtile_sif_model = simple_cnn.SimpleCNNSmall(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, crop_type_start_idx=CROP_TYPE_START_IDX, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    # elif MODEL_TYPE == 'simple_cnn_small_v2':
    #     subtile_sif_model = simple_cnn.SimpleCNNSmall2(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, crop_type_start_idx=CROP_TYPE_START_IDX, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    # elif MODEL_TYPE == 'simple_cnn_small_3':
    #     subtile_sif_model = simple_cnn.SimpleCNNSmall3(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    # elif MODEL_TYPE == 'simple_cnn_small_v4':
    #     subtile_sif_model = simple_cnn.SimpleCNNSmall4(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    # elif MODEL_TYPE == 'simple_cnn_small_v5':
    #     subtile_sif_model = simple_cnn.SimpleCNNSmall5(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    else:
        print('Model type not supported')
        exit(1)

    if FROM_PRETRAINED:
        subtile_sif_model.load_state_dict(torch.load(PRETRAINED_SUBTILE_SIF_MODEL_FILE, map_location=device))

    criterion = nn.MSELoss(reduction='mean')

    if OPTIMIZER_TYPE == "Adam":
        optimizer = optim.Adam(subtile_sif_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        print("Optimizer not supported")
        exit(1)

    subtile_sif_model, train_tropomi_losses, train_oco2_losses, val_tropomi_losses, val_oco2_losses, best_loss = train_model(subtile_sif_model, dataloaders, criterion, optimizer, device, sif_mean, sif_std, subtile_dim=SUBTILE_DIM, num_epochs=NUM_EPOCHS)
    torch.save(subtile_sif_model.state_dict(), SUBTILE_SIF_MODEL_FILE)
    print('Best OCO2 loss', best_loss)

    # Plot loss curves
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

    # Add legend and axis labels
    plt.legend(handles=plots)
    plt.xlabel('Epoch #')
    plt.ylabel('Normalized Root Mean Squared Error')
    plt.savefig(TRAINING_PLOT_FILE) 
    plt.close()

if __name__ == '__main__':
    main()

    # import cProfile
    # import pstats
    # filename = 'profile_stats'

    # cProfile.run('main()', filename)
    # stats = pstats.Stats(filename)

    # # Clean up filenames for the report
    # stats.strip_dirs()

    # # Sort the statistics by the cumulative time spent
    # # in the function
    # stats.sort_stats('cumulative')
    # stats.print_stats(100)
