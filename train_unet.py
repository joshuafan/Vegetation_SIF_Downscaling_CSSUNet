import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from itertools import chain
import time
import torch
import torchvision
import torchvision.transforms as transforms
import resnet
import torch.nn as nn
import torch.optim as optim

from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
from unet.unet_model import UNet
import tile_transforms


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
METHOD = "7_unet"
TRAINING_PLOT_FILE = 'exploratory_plots/losses_' + METHOD + '.png'

PRETRAINED_UNET_MODEL_FILE = os.path.join(DATA_DIR, "models/unet")
UNET_MODEL_FILE = os.path.join(DATA_DIR, "models/unet_2") #aug_2")
MODEL_TYPE = "UNet"
OPTIMIZER_TYPE = "Adam"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
TRUE_BATCH_SIZE = 8
NUM_WORKERS = 2
AUGMENT = True
FROM_PRETRAINED = True
MIN_SIF = 0
MAX_SIF = 1.7
MIN_INPUT = -2
MAX_INPUT = 2

#BANDS = list(range(0, 9)) + [42] #  12)) + list(range(12, 27)) + [28] + [42] 
#BANDS = list(range(0, 12)) + [12, 13, 14, 16] + [42]
BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42]  #list(range(0, 43))
INPUT_CHANNELS = len(BANDS)
REDUCED_CHANNELS = 15
NOISE = 0.1

# Print params for reference
print("=========================== PARAMS ===========================")
print("Method:", METHOD)
print("Dataset: ", os.path.basename(DATASET_DIR))
if FROM_PRETRAINED:
    print("From pretrained model", os.path.basename(PRETRAINED_UNET_MODEL_FILE))
else:
    print("Training from scratch")
print("Output model:", os.path.basename(UNET_MODEL_FILE))
print("Bands:", BANDS)
print("---------------------------------")
print("Model:", MODEL_TYPE)
print("Optimizer:", OPTIMIZER_TYPE)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Num workers:", NUM_WORKERS)
print("Batch size:", TRUE_BATCH_SIZE)
print("Num epochs:", NUM_EPOCHS)
print("Augment:", AUGMENT)
print("Gaussian noise (std deviation):", NOISE)
print("Input features clipped to", MIN_INPUT, "to", MAX_INPUT, "standard deviations from mean")
print("SIF range:", MIN_SIF, "to", MAX_SIF)
print("==============================================================")


# TODO should there be 2 separate models?
def train_model(model, tropomi_dataloaders, oco2_dataloaders, dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, sif_min=0.2, sif_max=1.7, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
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
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            batch_loss = 0.0

            # Iterate over data.
            j = 0
            for sample in chain(tropomi_dataloaders[phase], oco2_dataloaders[phase]):
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
                    predictions = model(input_tiles_standardized[:, BANDS, :, :])
                    # print('Predictions', predictions * sif_std + sif_mean)
                    predicted_sifs_standardized = torch.mean(predictions, dim=(1, 2, 3))
                    # print('Mean prediction', predicted_sifs_standardized)

                    # Compute loss: predicted vs true SIF (standardized)
                    loss = criterion(predicted_sifs_standardized, true_sifs_standardized)
                    # batch_loss += loss

                    # backward + optimize only if in training phase, and if we've reached the end of a batch
                    if (phase == 'train'): # and (j % TRUE_BATCH_SIZE == 0):
                        # average_loss = batch_loss / TRUE_BATCH_SIZE
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_value_(model.parameters(), 0.1)
                        # print("Conv1 grad after backward", predicted_sifs_standardized.grad)   
                        optimizer.step()
                        # batch_loss = 0.0

                    # statistics
                    predicted_sifs_non_standardized = torch.tensor(predicted_sifs_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                    non_standardized_loss = criterion(predicted_sifs_non_standardized, true_sifs_non_standardized)
                    j += batch_size
                    if j % 1000 == 0:
                        print('========================')
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
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), UNET_MODEL_FILE)
                torch.save(model.state_dict(), UNET_MODEL_FILE + "_epoch" + str(epoch))

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
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, best_loss


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
train_tropomi_metadata = train_metadata[train_metadata['source'] == 'TROPOMI'] 
val_tropomi_metadata = val_metadata[val_metadata['source'] == 'TROPOMI'] 
train_oco2_metadata = train_metadata[train_metadata['source'] == 'OCO2'] 
val_oco2_metadata = val_metadata[val_metadata['source'] == 'OCO2'] 

# # Add copies of OCO-2 tiles
# train_oco2_repeated = pd.concat([train_oco2_metadata]*4)
# val_oco2_repeated = pd.concat([val_oco2_metadata]*4)
# train_metadata = pd.concat([train_metadata, train_oco2_repeated])
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
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds, min_input=MIN_INPUT, max_input=MAX_INPUT))
transform_list.append(tile_transforms.GaussianNoise(continuous_bands=list(range(0, 12)), standard_deviation=NOISE))
if AUGMENT:
    transform_list.append(tile_transforms.RandomFlipAndRotate())
transform = transforms.Compose(transform_list)

# Set up Datasets and Dataloaders
# resize_transform = torchvision.transforms.Resize((224, 224))
tropomi_datasets = {'train': ReflectanceCoverSIFDataset(train_tropomi_metadata, transform),
            'val': ReflectanceCoverSIFDataset(val_tropomi_metadata, transform)}
oco2_datasets = {'train': ReflectanceCoverSIFDataset(train_oco2_metadata, transform),
            'val': ReflectanceCoverSIFDataset(val_oco2_metadata, transform)}
tropomi_dataloaders = {x: torch.utils.data.DataLoader(tropomi_datasets[x], batch_size=TRUE_BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS)
              for x in ['train', 'val']}
oco2_dataloaders = {x: torch.utils.data.DataLoader(tropomi_datasets[x], batch_size=TRUE_BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS)
              for x in ['train', 'val']}

print("Dataloaders")

if MODEL_TYPE == 'UNet':
    model = UNet(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
else:
    print('Model type not supported')
    exit(1)

if FROM_PRETRAINED:
    model.load_state_dict(torch.load(PRETRAINED_UNET_MODEL_FILE, map_location=device))

criterion = nn.MSELoss(reduction='mean')

if OPTIMIZER_TYPE == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    print("Optimizer not supported")
    exit(1)

dataset_sizes = {'train': len(train_metadata),
                 'val': len(val_metadata)}

model, train_losses, val_losses, best_loss = train_model(model, tropomi_dataloaders, oco2_dataloaders, dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, num_epochs=NUM_EPOCHS)

torch.save(model.state_dict(), UNET_MODEL_FILE)

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
