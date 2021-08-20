"""
Trains a ResNet (with full supervision) to predict the total SIF of a small sub-tile.
"""

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
import small_resnet
import simple_cnn
import torch.nn as nn
import torch.optim as optim

from eval_subtile_dataset import EvalSubtileDataset
from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
from sif_utils import train_single_model
import tile_transforms
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01") #08-01") #07-16")
CFIS_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-08-01") #07-16") #-01") #07-16")
CFIS_INFO_FILE = os.path.join(CFIS_DATASET_DIR, "eval_subtiles.csv")
CFIS_INFO_FILE_TRAIN = os.path.join(CFIS_DATASET_DIR, "eval_subtiles_train.csv")
CFIS_INFO_FILE_VAL = os.path.join(CFIS_DATASET_DIR, "eval_subtiles_val.csv")
TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/cfis_sif_aug") #jul")
METHOD = "0_cheating_cfis_sif"
LOSS_PLOT_FILE = "exploratory_plots/losses_" + METHOD + ".png"
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
FROM_PRETRAINED = False
SHRINK = False # True
AUGMENT = False #True
NUM_EPOCHS = 50
INPUT_CHANNELS = 43
OPTIMIZER_TYPE = "Adam"
LEARNING_RATE = 1e-3 # 0.01 # 1e-5 # 0.00001  # 1e-3
WEIGHT_DECAY = 0 #1e-4
BATCH_SIZE = 32
NUM_WORKERS = 4
RGB_BANDS = [1, 2, 3]
MIN_SIF = 0.2
MAX_SIF = 1.7

# Print params for reference
print("=========================== PARAMS ===========================")
print("Method:", METHOD)
print("Dataset (band statistics):", os.path.basename(DATASET_DIR))
print("CFIS dataset (band statistics):", os.path.basename(CFIS_DATASET_DIR))
if FROM_PRETRAINED:
    print("From pretrained model")
else:
    print("Training from scratch")
print("Output model:", os.path.basename(TRAINED_MODEL_FILE))
print("---------------------------------")
print("Optimizer:", OPTIMIZER_TYPE)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Batch size:", BATCH_SIZE)
print("Num epochs:", NUM_EPOCHS)
print("Augment:", AUGMENT)
print("Shrink:", SHRINK)
print("SIF range:", MIN_SIF, "to", MAX_SIF)
print("==============================================================")

# Visualize images (RGB bands only)
# Image is assumed to be standardized. You need to pass in band_means and band_stds
# so it can be un-standardized.
# Tile is assumed to be in Pytorch format: CxWxH
def imshow(tile, band_means, band_stds):
    tile = (tile * band_stds) + band_means

    print("================= Per-band averages: =====================")
    for i in range(tile.shape[0]):
        print("Band", i, ":", np.mean(tile[i].flatten()))
    print("==========================================================")
    img = tile[RGB_BANDS, :, :]
    print("Image shape", img.shape)

    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()



# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)

# Read CFIS tile metadata, and randomly split into train/val sets
all_metadata = pd.read_csv(CFIS_INFO_FILE)
train_metadata, val_metadata = train_test_split(all_metadata, test_size=0.3)
train_metadata.to_csv(CFIS_INFO_FILE_TRAIN)
val_metadata.to_csv(CFIS_INFO_FILE_VAL)

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

min_output = (MIN_SIF - sif_mean) / sif_std
max_output = (MAX_SIF - sif_mean) / sif_std


# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
if SHRINK:
    transform_list.append(tile_transforms.ShrinkTile())
if AUGMENT:
    transform_list.append(tile_transforms.RandomFlipAndRotate())
transform = transforms.Compose(transform_list)

# Set up Datasets and Dataloaders
datasets = {'train': EvalSubtileDataset(train_metadata, transform),
            'val': EvalSubtileDataset(val_metadata, transform)}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS)
              for x in ['train', 'val']}

print("Dataloaders")
resnet_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
# resnet_model = resnet.resnet18(input_channels=INPUT_CHANNELS).to(device)
# resnet_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=1)  #.to(device)
if FROM_PRETRAINED:
    resnet_model.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=device))
print("Loaded model")


criterion = nn.MSELoss(reduction='mean')
#optimizer = optim.SGD(resnet_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = optim.Adam(resnet_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
dataset_sizes = {'train': len(train_metadata),
                 'val': len(val_metadata)}

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Train model
print("Starting to train")
resnet_model, train_losses, val_losses, best_loss = train_single_model(resnet_model, dataloaders,
    dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, TRAINED_MODEL_FILE,
    num_epochs=NUM_EPOCHS)

# Save model to file
torch.save(resnet_model.state_dict(), TRAINED_MODEL_FILE)

# Plot loss curves
print("Train losses:", train_losses)
print("Validation losses:", val_losses)
epoch_list = range(NUM_EPOCHS)
train_plot, = plt.plot(epoch_list, train_losses, color='blue', label='Train NRMSE')
val_plot, = plt.plot(epoch_list, val_losses, color='red', label='Validation NRMSE')
plt.legend(handles=[train_plot, val_plot])
plt.xlabel('Epoch #')
plt.ylabel('Normalized Root Mean Squared Error')
plt.savefig('plots/losses_small_tile_sif_prediction.png')
plt.close()
