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
from sif_utils import get_subtiles_list_by_crop
import simple_cnn
import small_resnet
import tile_transforms


# TODO this is a hack
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet
from embedding_to_sif_model import EmbeddingToSIFModel

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
METHOD = "4a_subtile_simple_cnn_small"
TRAINING_PLOT_FILE = 'exploratory_plots/losses_' + METHOD + '.png'

PRETRAINED_SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn_4crop_small")
SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn_croptype_16crop") #aug_2")
SUBTILE_DIM = 10
OPTIMIZER_TYPE = "Adam"
MODEL_TYPE = "simple_cnn_small"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 50
BATCH_SIZE = 64 
NUM_WORKERS = 4
AUGMENT = True
FROM_PRETRAINED = False
MIN_SIF = 0.2
MAX_SIF = 1.7
MAX_SUBTILE_CLOUD_COVER = 0.1

# CROP_TYPE_INDICES = [12, 13, 14, 16]
CROP_TYPE_INDICES = list(range(12, 27)) + [28]  
#CROP_TYPE_INDICES = [12, 13, 14, 16] # list(range(12, 42))
# BANDS = list(range(0, 12)) + CROP_TYPE_INDICES + [42] # Don't include crop types?
FEATURE_BANDS = list(range(0, 12)) + [42]
PURE_THRESHOLD = 0.5

INPUT_CHANNELS = len(FEATURE_BANDS)
REDUCED_CHANNELS = 12

# Print params for reference
print("=========================== PARAMS ===========================")
print("Method:", METHOD)
print("Dataset: ", os.path.basename(DATASET_DIR))
if FROM_PRETRAINED:
    print("From pretrained model", os.path.basename(PRETRAINED_SUBTILE_SIF_MODEL_FILE))
else:
    print("Training from scratch")
print("Output model:", os.path.basename(SUBTILE_SIF_MODEL_FILE))
print("Crop types:", CROP_TYPE_INDICES)
print("Bands:", FEATURE_BANDS)
print("---------------------------------")
print("Model:", MODEL_TYPE)
print("Optimizer:", OPTIMIZER_TYPE)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Batch size:", BATCH_SIZE)
print("Num epochs:", NUM_EPOCHS)
print("Augment:", AUGMENT)
print("Reduced channels:", REDUCED_CHANNELS)
print("Subtile dim:", SUBTILE_DIM)
print("SIF range:", MIN_SIF, "to", MAX_SIF)
print("==============================================================")


# TODO should there be 2 separate models?
def train_model(models, dataloaders, dataset_sizes, criterion, optimizers, device, sif_mean, sif_std, subtile_dim, sif_min=0.2, sif_max=1.7, num_epochs=25):
    since = time.time()

    # Store weights for the best model for each crop type
    best_model_wts = {crop:copy.deepcopy(model.state_dict()) for (crop, model) in models.items()}
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

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            for crop, model in models.items():
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

            # Count number of subtiles per crop
            num_subtiles_per_crop = {}
            for crop in models:
                num_subtiles_per_crop[crop] = 0

            # Iterate over data.
            running_loss = 0.0
            j = 0
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
                    # Clear gradients
                    for crop, optimizer in optimizers.items():
                        optimizer.zero_grad()
                    
                    # seen_crops = set()
                    for i in range(batch_size):
                        # Get list of sub-tiles in this large tile, by crop type
                        subtiles_by_crop, num_subtiles = get_subtiles_list_by_crop(input_tiles_standardized[i], subtile_dim, device,
                                                                               CROP_TYPE_INDICES, PURE_THRESHOLD, MAX_SUBTILE_CLOUD_COVER)

                        # For each crop type: pass the subtiles dominated by the crop through that
                        # crop type's network.
                        predicted_subtile_sifs = torch.empty((num_subtiles))
                        subtile_idx = 0
                        for crop, subtiles in subtiles_by_crop.items():
                            num_subtiles_this_crop = subtiles.shape[0]
                            num_subtiles_per_crop[crop] += num_subtiles_this_crop
                            subtiles = subtiles[:, FEATURE_BANDS, :, :]
                            predicted_subtile_sifs[subtile_idx:subtile_idx+num_subtiles_this_crop] = models[crop](subtiles).flatten()
                            subtile_idx += num_subtiles_this_crop
                            # if crop not in seen_crops:
                            #     seen_crops.append(crop)

                        predicted_sifs_standardized[i] = torch.mean(predicted_subtile_sifs)
                    #print('predicted_sifs_standardized requires grad', predicted_sifs_standardized.requires_grad)
                    #print('Shape of predicted total SIFs', predicted_sifs_standardized.shape)
                    #print('Shape of true total SIFs', true_sifs_standardized.shape)

                    # Compute loss: predicted vs true SIF (standardized)
                    loss = criterion(predicted_sifs_standardized, true_sifs_standardized)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        #print("Conv1 grad after backward", models[12].conv1.bias.grad)
                        for crop, optimizer in optimizers.items():
                            optimizer.step()

                        #print("Seen crops", seen_crops)
                        #for crop in seen_crops:
                        #    optimizers[crop].step()

                    # statistics
                    predicted_sifs_non_standardized = torch.tensor(predicted_sifs_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                    non_standardized_loss = criterion(predicted_sifs_non_standardized, true_sifs_non_standardized)
                    j += 1
                    if j % 50 == 0:
                        print('========================')
                        print('*** >>> predicted subtile sifs for last', predicted_subtile_sifs * sif_std + sif_mean)
                        print('*** Predicted', predicted_sifs_non_standardized)
                        print('*** True', true_sifs_non_standardized)
                        print('*** batch loss', (math.sqrt(non_standardized_loss.item()) / sif_mean).item())
                    running_loss += non_standardized_loss.item() * len(sample['SIF'])

            epoch_loss = math.sqrt(running_loss / dataset_sizes[phase]) / sif_mean

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            print('Num subtiles per crop', num_subtiles_per_crop)

            # deep copy the models
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = {crop:copy.deepcopy(model.state_dict()) for (crop, model) in models.items()}
                for crop, model in models.items():
                    torch.save(model.state_dict(), SUBTILE_SIF_MODEL_FILE + "_crop_" + str(crop))


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
    for crop, model_wts in best_model_wts:
        models[crop].load_state_dict(model_wts)
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
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
if AUGMENT:
    transform_list.append(tile_transforms.RandomFlipAndRotate())
transform = transforms.Compose(transform_list)

# Set up Datasets and Dataloaders
# resize_transform = torchvision.transforms.Resize((224, 224))
datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, transform),
            'val': ReflectanceCoverSIFDataset(val_metadata, transform)}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS)
              for x in ['train', 'val']}

print("Dataloaders")

# Initialize models for each crop type
models = dict()
optimizers = dict()
for crop in [-1] + CROP_TYPE_INDICES:
    if MODEL_TYPE == 'resnet18':
        model = small_resnet.resnet18(input_channels=INPUT_CHANNELS).to(device)
    elif MODEL_TYPE == 'simple_cnn':
        model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    elif MODEL_TYPE == 'simple_cnn_small':
        model = simple_cnn.SimpleCNNSmall(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    else:
        print('Model type not supported')
        exit(1)
    models[crop] = model

    # Create optimizer for the model
    if OPTIMIZER_TYPE == "Adam":
        optimizers[crop] = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        print("Optimizer not supported")
        exit(1)

# If "FROM_PRETRAINED" is true, load pre-trained models
if FROM_PRETRAINED:
    for crop, model in models.items():
        model.load_state_dict(torch.load(PRETRAINED_SUBTILE_SIF_MODEL_FILE + "_crop_" + str(crop), map_location=device))

# Loss function
criterion = nn.MSELoss(reduction='mean')

dataset_sizes = {'train': len(train_metadata),
                 'val': len(val_metadata)}

models, train_losses, val_losses, best_loss = train_model(models, dataloaders, dataset_sizes, criterion, optimizers, device, sif_mean, sif_std, subtile_dim=SUBTILE_DIM, num_epochs=NUM_EPOCHS)


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
