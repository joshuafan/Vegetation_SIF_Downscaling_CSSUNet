"""
Trains a ResNet (with full supervision) to predict the total SIF of a large tile (0.1 x 0.1 degree)
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

from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
from sif_utils import train_single_model
import tile_transforms
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-07-16")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/large_tile_resnet18_sgd") #small_tile_simple")  # "models/large_tile_resnet50")
LOSS_PLOT_FILE = "exploratory_plots/losses_large_tile_resnet18" # small_tile_simple.png"  #losses_large_tile_resnet50.png"
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
FROM_PRETRAINED = False  #True # False  # Truei
SHRINK = False #True # True
AUGMENT = True
NUM_EPOCHS = 50
INPUT_CHANNELS = 43
LEARNING_RATE = 1e-3 # 0.01 # 1e-5 # 0.00001  # 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 4
RGB_BANDS = [1, 2, 3]

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

# Read train/val tile metadata
train_metadata = pd.read_csv(INFO_FILE_TRAIN) #.iloc[0:100]
val_metadata = pd.read_csv(INFO_FILE_VAL) # .iloc[0:100]

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

# Set up image transforms
transform_list = []
if SHRINK:
    transform_list.append(tile_transforms.ShrinkTile())
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
#resnet_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, output_dim=1).to(device)
resnet_model = resnet.resnet18(input_channels=INPUT_CHANNELS).to(device)
# resnet_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=1)  #.to(device)
if FROM_PRETRAINED:
    resnet_model.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=device))
print("Loaded model")


criterion = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(resnet_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
#optimizer = optim.Adam(resnet_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
