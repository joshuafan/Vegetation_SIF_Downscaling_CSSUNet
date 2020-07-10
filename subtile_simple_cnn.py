import copy
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

from reflectance_cover_sif_dataset import CombinedDataset, SubtileListDataset, ReflectanceCoverSIFDataset
from sif_utils import get_subtiles_list
import simple_cnn
import small_resnet
import tile_transforms

# import sys
# sys.path.append('../')
# from tile2vec.src.tilenet import make_tilenet
# from embedding_to_sif_model import EmbeddingToSIFModel

# Data directories
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_tropomi_train") #_landsat_0.1")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
# INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "standardized_tiles_train.csv")
# INFO_FILE_VAL = os.path.join(DATASET_DIR, "standardized_tiles_val.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")

# Method, optimizer, model type
METHOD = "1d_train_tropomi_subtile_resnet"
OPTIMIZER_TYPE = "Adam"
MODEL_TYPE = "resnet18"

# Which sources to train on
TRAIN_SOURCES = ["OCO2", "TROPOMI"]
# Model files
PRETRAINED_SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)
SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)

# Loss plot file
TRAINING_PLOT_FILE = 'result_plots/losses_' + METHOD + '.png'

# Hyperparameters
SUBTILE_DIM = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 128
NUM_WORKERS = 4
AUGMENT = False
FROM_PRETRAINED = False
MIN_SIF = 0
MAX_SIF = 1.7
MIN_INPUT = -2
MAX_INPUT = 2
MAX_SUBTILE_CLOUD_COVER = 0.5
NOISE = 0.05

# Which bands to use
#BANDS = list(range(0, 9)) + [42] #  12)) + list(range(12, 27)) + [28] + [42] 
# BANDS = list(range(0, 12)) + [12, 13, 14, 16] + [42]
# BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42]  #list(range(0, 43))
BANDS = list(range(0, 43))
INPUT_CHANNELS = len(BANDS)

# If we do dimensionality reduction
REDUCED_CHANNELS = 15
CROP_TYPE_START_IDX = 12

# Print params for reference
print("=========================== PARAMS ===========================")
print("Method:", METHOD)
print("Dataset: ", os.path.basename(DATASET_DIR))
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
print("Batch size:", BATCH_SIZE)
print("Num epochs:", NUM_EPOCHS)
print("Augment:", AUGMENT)
print("Gaussian noise (std deviation):", NOISE)
print("Reduced channels:", REDUCED_CHANNELS)
print("Subtile dim:", SUBTILE_DIM)
print("Num subtiles to sample:", NUM_SUBTILES)
print("Max sub-tile cloud cover:", MAX_SUBTILE_CLOUD_COVER)
print("Input features clipped to", MIN_INPUT, "to", MAX_INPUT, "standard deviations from mean")
print("SIF range:", MIN_SIF, "to", MAX_SIF)
print("==============================================================")



# TODO should there be 2 separate models?
def train_model(subtile_sif_model, dataloaders, dataset_sizes, criterion, optimizer, device, 
                sif_mean, sif_std, subtile_dim, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(subtile_sif_model.state_dict())
    best_loss = float('inf')
    train_tropomi_losses = []
    train_oco2_losses = []
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
                subtile_sif_model.train()
            else:
                subtile_sif_model.eval()

            running_tropomi_loss = 0.0
            running_oco2_loss = 0.0
            batch_loss = 0.0

            # Iterate over data.
            tropomi_idx = 0
            oco2_idx = 0
            for sample in dataloaders[phase]:
                if 'tropomi_subtiles' in sample:
                    # Read TROPOMI tile and SIF from dataloader
                    tropomi_subtiles_std = sample['tropomi_subtiles'].to(device)  # (batch size, num sub-tiles, num channels, H, W)
                    tropomi_true_sifs = sample['tropomi_sif'].to(device)  # (batch_size)
                    print('TROPOMI tile shape', tropomi_true_sifs.shape)
                    assert(tropomi_subtiles_std.shape[0] == tropomi_true_sifs.shape[0] * 2)

                    # Standardize SIF
                    tropomi_true_sifs_std = ((tropomi_true_sifs - sif_mean) / sif_std).to(device)

                    # Reshape into a batch of "sub-tiles"
                    input_shape = tropomi_subtiles_std.shape
                    total_num_subtiles = input_shape[0] * input_shape[1]
                    input_subtiles = tropomi_subtiles_std.view((total_num_subtiles, input_shape[2], input_shape[3], input_shape[4]))
                    print('Subtiles input shape', input_subtiles.shape)

                    # Zero out gradients
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        predicted_subtile_sifs = subtile_sif_model(input_subtiles)
                        print('Subtile SIFs shape', predicted_subtile_sifs.shape)
                        predicted_subtile_sifs = predicted_subtile_sifs.view((input_shape[0], input_shape[1]))
                        print('Reshaped subtile SIFs', predicted_subtile_sifs.shape)
                        tropomi_predicted_sifs_std = torch.mean(predicted_subtile_sifs, dim=1)
                        print('Predicted SIFs (mean)', predicted_sifs_standardized.shape)

                        # Compute loss: predicted vs true SIF (standardized)
                        loss = criterion(tropomi_predicted_sifs_std, tropomi_true_sifs_std)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step() 

                    with torch.set_grad_enabled(False):
                        # statistics
                        tropomi_predicted_sifs = torch.tensor(tropomi_predicted_sifs_std * sif_std + sif_mean, dtype=torch.float).to(device)
                        non_standardized_loss = criterion(tropomi_predicted_sifs, tropomi_true_sifs)
                        tropomi_idx += 1
                        if tropomi_idx % 2 == 0:
                            print('========================')
                            print('*** >>> (TROPOMI) predicted subtile sifs for 0-4th', predicted_subtile_sifs.flatten()[0:5] * sif_std + sif_mean)
                            print('*** Predicted', tropomi_predicted_sifs[0:20])
                            print('*** True', tropomi_true_sifs[0:20])
                            print('*** batch loss', (math.sqrt(non_standardized_loss.item()) / sif_mean).item())
                        running_tropomi_loss += non_standardized_loss.item() * len(sample['tropomi_sif'])

                if 'oco2_tile' in sample:
                    # Read TROPOMI tile and SIF from dataloader
                    oco2_tiles_std = sample['oco2_tile'].to(device)  # (batch size, num channels, H, W)
                    oco2_true_sifs = sample['oco2_sif'].to(device)  # (batch_size)
                    print('TROPOMI tile shape', oco2_true_sifs.shape)
                    assert(oco2_tiles_std.shape[0] == oco2_true_sifs.shape[0] * 2)

                    # Standardize SIF
                    oco2_true_sifs_std = ((oco2_true_sifs - sif_mean) / sif_std).to(device)

                    # Zero out gradients
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        oco2_predicted_sifs_std = subtile_sif_model(oco2_tiles_std)
                        print('Subtile SIFs shape', predicted_sifs.shape)

                        # Compute loss: predicted vs true SIF (standardized)
                        loss = criterion(oco2_predicted_sifs_std, oco2_true_sifs_std)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step() 

                    with torch.set_grad_enabled(False):
                        # statistics
                        oco2_predicted_sifs = torch.tensor(oco2_predicted_sifs_std * sif_std + sif_mean, dtype=torch.float).to(device)
                        non_standardized_loss = criterion(oco2_predicted_sifs, oco2_true_sifs)
                        oco2_idx += 1
                        if oco2_idx % 2 == 0:
                            print('========================')
                            print('*** Predicted', oco2_predicted_sifs[0:20])
                            print('*** True', oco2_true_sifs[0:20])
                            print('*** batch loss', (math.sqrt(non_standardized_loss.item()) / sif_mean).item())
                        running_oco2_loss += non_standardized_loss.item() * len(sample['oco2_SIF'])
                

                # # Read batch from dataloader
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

            # Note: the way the Dataset is set up, we loop through the same number of TROPOMI and OCO-2 points per epoch
            epoch_tropomi_loss = math.sqrt(running_tropomi_loss / dataset_sizes[phase]) / sif_mean
            epoch_oco2_loss = math.sqrt(running_oco2_loss / dataset_sizes[phase]) / sif_mean

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(subtile_sif_model.state_dict())
                torch.save(subtile_sif_model.state_dict(), SUBTILE_SIF_MODEL_FILE)
                # torch.save(subtile_sif_model.state_dict(), SUBTILE_SIF_MODEL_FILE + "_epoch" + str(epoch))

            # Record loss
            if phase == 'train':
                if epoch_tropomi_loss > 0:
                    train_tropomi_losses.append(epoch_tropomi_loss)
                if epoch_oco2_loss > 0:
                    train_oco2_losses.append(epoch_oco2_loss)
            else:
                val_losses.append(epoch_oco2_loss)
        
        # Print elapsed time per epoch
        epoch_time = time.time() - epoch_start
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            epoch_time // 60, epoch_time % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    subtile_sif_model.load_state_dict(best_model_wts)
    return subtile_sif_model, train_losses, val_losses, best_loss


# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)

# Read train/val tile metadata
train_metadata = pd.read_csv(INFO_FILE_TRAIN)
val_metadata = pd.read_csv(INFO_FILE_VAL)

# Filter
train_oco2_metadata = train_metadata[train_metadata['source'] == 'OCO2'] 
val_oco2_metadata = val_metadata[val_metadata['source'] == 'OCO2'] 

# Add copies of OCO-2 tiles
# train_oco2_repeated = pd.concat([train_oco2_metadata]*4)
# train_metadata = pd.concat([train_metadata, train_oco2_repeated])
# val_oco2_repeated = pd.concat([val_oco2_metadata]*4)
# val_metadata = pd.concat([val_metadata, val_oco2_repeated])

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

# Read train/val tile metadata
train_metadata = pd.read_csv(INFO_FILE_TRAIN)
val_metadata = pd.read_csv(INFO_FILE_VAL)

# Extract TROPOMI and OCO-2 rows, if applicable
if 'TROPOMI' in TRAIN_SOURCES:
    train_tropomi_metadata = train_metadata[train_metadata['source'] == 'TROPOMI']
    print('Train OCO2 samples', len(train_tropomi_metadata))
else:
    train_tropomi_metadata = None

if 'OCO2' in TRAIN_SOURCES:
    train_oco2_metadata = train_metadata[train_metadata['source'] == 'TROPOMI']
    print('Train OCO2 sample', len(train_oco2_metadata))
else:
    train_oco2_metadata = None


# Set up image transforms
standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds, min_input=MIN_INPUT, max_input=MAX_INPUT)
noise_transform = tile_transforms.GaussianNoise(continuous_bands=list(range(0, 12)), standard_deviation=NOISE)
flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()
transform_list_train = [standardize_transform, noise_transform]
transform_list_val = [standardize_transform]
if AUGMENT:
    transform_list_train.append(flip_and_rotate_transform)
transform_train = transforms.Compose(transform_list_train)
transform_val = transforms.Compose(transform_list_val)

datasets = {'train': CombinedDataset(train_tropomi_metadata, train_oco2_metadata, transform_train, SUBTILE_DIM, tile_file_column='tile_file'),
            'val': CombinedDataset(None, val_metadata, transform_val, SUBTILE_DIM, tile_file_column='tile_file')}  # Only validate on OCO-2
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
    subtile_sif_model = small_resnet.resnet18(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn':
    subtile_sif_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small':
    subtile_sif_model = simple_cnn.SimpleCNNSmall(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, crop_type_start_idx=CROP_TYPE_START_IDX, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small_v2':
    subtile_sif_model = simple_cnn.SimpleCNNSmall2(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, crop_type_start_idx=CROP_TYPE_START_IDX, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small_3':
    subtile_sif_model = simple_cnn.SimpleCNNSmall3(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small_v4':
    subtile_sif_model = simple_cnn.SimpleCNNSmall4(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small_v5':
    subtile_sif_model = simple_cnn.SimpleCNNSmall5(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
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

dataset_sizes = {'train': len(train_metadata),
                 'val': len(val_metadata)}

subtile_sif_model, train_losses, val_losses, best_loss = train_model(subtile_sif_model, dataloaders, dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, subtile_dim=SUBTILE_DIM, num_epochs=NUM_EPOCHS)

torch.save(subtile_sif_model.state_dict(), SUBTILE_SIF_MODEL_FILE)

# Plot loss curves
print("Train losses:", train_losses)
print("Validation losses:", val_losses)
epoch_list = range(NUM_EPOCHS)
train_plot, = plt.plot(epoch_list, train_losses, color='blue', label='Train NRMSE')
val_plot, = plt.plot(epoch_list, val_losses, color='red', label='Validation NRMSE')
plt.legend(handles=[train_plot, val_plot])
plt.xlabel('Epoch #')
plt.ylabel('Normalized Root Mean Squared Error')
plt.savefig(TRAINING_PLOT_FILE) 
plt.close()
