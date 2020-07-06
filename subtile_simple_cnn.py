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

from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
from sif_utils import get_subtiles_list
import simple_cnn
import small_resnet
import tile_transforms


# TODO this is a hack
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet
from embedding_to_sif_model import EmbeddingToSIFModel

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
# DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
# DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_tropomi_train") #_landsat_0.1")
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
METHOD = "4a_subtile_simple_cnn_new_data"
TRAINING_PLOT_FILE = 'exploratory_plots/losses_' + METHOD + '.png'

PRETRAINED_SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn_v5")
SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn_v5")
SUBTILE_DIM = 10
OPTIMIZER_TYPE = "Adam"
MODEL_TYPE = "simple_cnn_small_v5"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
TRUE_BATCH_SIZE = 16
NUM_WORKERS = 4
AUGMENT = True
FROM_PRETRAINED = False
MIN_SIF = 0
MAX_SIF = 1.7
MIN_INPUT = -2
MAX_INPUT = 2
MAX_SUBTILE_CLOUD_COVER = 0.2

#BANDS = list(range(0, 9)) + [42] #  12)) + list(range(12, 27)) + [28] + [42] 
# BANDS = list(range(0, 12)) + [12, 13, 14, 16] + [42]
# BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42]  #list(range(0, 43))
BANDS = list(range(0, 43))
INPUT_CHANNELS = len(BANDS)
REDUCED_CHANNELS = 15
CROP_TYPE_START_IDX = 12
NOISE = 0

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
print("Batch size:", TRUE_BATCH_SIZE)
print("Num epochs:", NUM_EPOCHS)
print("Augment:", AUGMENT)
print("Gaussian noise (std deviation):", NOISE)
print("Reduced channels:", REDUCED_CHANNELS)
print("Subtile dim:", SUBTILE_DIM)
print("Max sub-tile cloud cover:", MAX_SUBTILE_CLOUD_COVER)
print("Input features clipped to", MIN_INPUT, "to", MAX_INPUT, "standard deviations from mean")
print("SIF range:", MIN_SIF, "to", MAX_SIF)
print("==============================================================")


# TODO should there be 2 separate models?
def train_model(subtile_sif_model, dataloaders, dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, subtile_dim, sif_min=0.2, sif_max=1.7, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(subtile_sif_model.state_dict())
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
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

            running_loss = 0.0
            batch_loss = 0.0

            # Iterate over data.
            j = 1
            for sample in dataloaders[phase]:
                # Read batch from dataloader
                batch_size = len(sample['SIF'])
                input_tiles_standardized = sample['tile'].to(device)
                true_sifs_non_standardized = sample['SIF'].to(device)

                # Standardize true SIF
                true_sifs_standardized = ((true_sifs_non_standardized - sif_mean) / sif_std).to(device)

                # For each large tile, predict SIF by passing each of its sub-tiles through the model
                predicted_sifs_standardized = torch.empty((batch_size)).to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    #optimizer.zero_grad()  # Clear gradients
                    for i in range(batch_size):
                        # Get list of sub-tiles in this large tile
                        subtiles = get_subtiles_list(input_tiles_standardized[i, BANDS, :, :], subtile_dim, device, MAX_SUBTILE_CLOUD_COVER)
                        # print(' ====== Random pixels (0th sub-tile) =======')
                        # print(subtiles[0, :, 8, 8])
                        # print(subtiles[0, :, 8, 9])
                        # print(subtiles[0, :, 9, 9])
                        predicted_subtile_sifs = subtile_sif_model(subtiles)
                        #print('predicted_subtile_sifs requires grad', predicted_subtile_sifs.requires_grad)
                        # print('Predicted subtile SIFs', predicted_subtile_sifs[100:120] * sif_std + sif_mean)
                        # print(predicted_subtile_sifs.shape)
                        predicted_sifs_standardized[i] = torch.mean(predicted_subtile_sifs)
                    #print('predicted_sifs_standardized requires grad', predicted_sifs_standardized.requires_grad)
                    #print('Shape of predicted total SIFs', predicted_sifs_standardized.shape)
                    #print('Shape of true total SIFs', true_sifs_standardized.shape)

                    # Compute loss: predicted vs true SIF (standardized)
                    loss = criterion(predicted_sifs_standardized, true_sifs_standardized)
                    batch_loss += loss

                    # backward + optimize only if in training phase, and if we've reached the end of a batch
                    if phase == 'train' and j % TRUE_BATCH_SIZE == 0:
                        average_loss = batch_loss / TRUE_BATCH_SIZE
                        optimizer.zero_grad()
                        average_loss.backward()
                        optimizer.step()
                        batch_loss = 0.0
                        #print("Conv1 grad after backward", subtile_sif_model.conv1.bias.grad)   

                    # statistics
                    predicted_sifs_non_standardized = torch.tensor(predicted_sifs_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                    non_standardized_loss = criterion(predicted_sifs_non_standardized, true_sifs_non_standardized)
                    j += batch_size
                    if j % 5000 == 0:
                        print('========================')
                        print('*** >>> predicted subtile sifs for 0-4th', predicted_subtile_sifs[0:5].flatten() * sif_std + sif_mean)
                        print('*** Predicted', predicted_sifs_non_standardized)
                        print('*** True', true_sifs_non_standardized)
                        print('*** batch loss', (math.sqrt(non_standardized_loss.item()) / sif_mean).item())
                    running_loss += non_standardized_loss.item() * len(sample['SIF'])

            epoch_loss = math.sqrt(running_loss / dataset_sizes[phase]) / sif_mean

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))


            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(subtile_sif_model.state_dict())
                torch.save(subtile_sif_model.state_dict(), SUBTILE_SIF_MODEL_FILE)
                torch.save(subtile_sif_model.state_dict(), SUBTILE_SIF_MODEL_FILE + "_epoch" + str(epoch))

            # Record loss
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
        
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
print("Train samples", len(train_metadata))
print("Validation samples", len(val_metadata))
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


# Set up image transforms
standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds, min_input=MIN_INPUT, max_input=MAX_INPUT)
noise_transform = tile_transforms.GaussianNoise(continuous_bands=list(range(0, 12)), standard_deviation=NOISE)
flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()
transform_list = [standardize_transform]
transform_list_with_noise = [standardize_transform, noise_transform]
if AUGMENT:
    transform_list.append(flip_and_rotate_transform)
    transform_list_with_noise.append(flip_and_rotate_transform)
transform = transforms.Compose(transform_list)
transform_with_noise = transforms.Compose(transform_list_with_noise)

# Set up Datasets and Dataloaders
# resize_transform = torchvision.transforms.Resize((224, 224))
datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, transform_with_noise),
            'val': ReflectanceCoverSIFDataset(val_metadata, transform)}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1,
                                              shuffle=True, num_workers=NUM_WORKERS)
              for x in ['train', 'val']}

print("Dataloaders")

if MODEL_TYPE == 'resnet18':
    subtile_sif_model = small_resnet.resnet18(input_channels=INPUT_CHANNELS).to(device)
elif MODEL_TYPE == 'simple_cnn':
    subtile_sif_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small':
    subtile_sif_model = simple_cnn.SimpleCNNSmall(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, crop_type_start_idx=CROP_TYPE_START_IDX, output_dim=1, min_output=min_output, max_output=max_output).to(device)
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
