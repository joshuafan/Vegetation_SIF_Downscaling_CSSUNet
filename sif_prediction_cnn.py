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

# Set random seed
torch.manual_seed(0)

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset")
# DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-07-16") #08-01")
# INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
# INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "resized_tiles_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "resized_tiles_val.csv")
TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/small_tile_simple_v2")  # "models/AUG_large_tile_resnet") #small_tile_simple") #small_tile_simple")  # "models/large_tile_resnet50")
METHOD = "3_small_tile_simple" # "3_small_tile_resnet" # "2_large_tile_resnet" # "3_small_tile_simple"
LOSS_PLOT_FILE = "exploratory_plots/losses_" + METHOD + ".png" # small_tile_simple.png"  #losses_large_tile_resnet50.png"
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
MODEL_TYPE = "simple_cnn_small_v2" #"resnet18" 
OPTIMIZER_TYPE = "Adam"
FROM_PRETRAINED = True
SHRINK = False #True
AUGMENT = True
NUM_EPOCHS = 50
INPUT_CHANNELS = 43
REDUCED_CHANNELS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
NUM_WORKERS = 1
CROP_TYPE_START_IDX = 12
RGB_BANDS = [1, 2, 3]
MIN_SIF = 0 # 0.2
MAX_SIF = 1.7 # 1.7

# Print params for reference
print("=========================== PARAMS ===========================")
print("Method:", METHOD)
print("Dataset:", os.path.basename(DATASET_DIR))
if FROM_PRETRAINED:
    print("From pretrained model")
else:
    print("Training from scratch")
print("Output model:", os.path.basename(TRAINED_MODEL_FILE))
print("---------------------------------")
print("Model type:", MODEL_TYPE)
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
if MIN_SIF is not None and MAX_SIF is not None:
    min_output = (MIN_SIF - sif_mean) / sif_std
    max_output = (MAX_SIF - sif_mean) / sif_std
else:
    min_output = None
    max_output = None

# Set up image transforms
transform_list = []
# transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
if SHRINK:
    transform_list.append(tile_transforms.ShrinkTile())
if AUGMENT:
    transform_list.append(tile_transforms.RandomFlipAndRotate())
transform = transforms.Compose(transform_list)

# Set up Datasets and Dataloaders
# resize_transform = torchvision.transforms.Resize((224, 224))
datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, transform=transform, tile_file_column='resized_tile_file'),
            'val': ReflectanceCoverSIFDataset(val_metadata, transform=None, tile_file_column='resized_tile_file')}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS)
              for x in ['train', 'val']}

print("Dataloaders")
#resnet_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, output_dim=1,
#                                    min_output=min_output, max_output=max_output).to(device)
if MODEL_TYPE == 'simple_cnn_small_v2':
    resnet_model = simple_cnn.SimpleCNNSmall2(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, 
                                              crop_type_start_idx=CROP_TYPE_START_IDX, output_dim=1, 
                                              min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'resnet18':
    resnet_model = small_resnet.resnet18(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS,
                                         min_output=min_output, max_output=max_output).to(device)
else:
    print('Model type not supported:', MODEL_TYPE)
    exit(1)

# resnet_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=1)  #.to(device)
if FROM_PRETRAINED:
    resnet_model.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=device))
elif os.path.exists(TRAINED_MODEL_FILE):
    response = input("Warning about to overwrite existing model, but youre not pretraining! Is that ok? (y/n)")
    if response != 'y' and response != 'Y':
        exit(1)
print("Loaded model")


criterion = nn.MSELoss(reduction='mean')
#optimizer = optim.SGD(resnet_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
if OPTIMIZER_TYPE == "Adam":
    optimizer = optim.Adam(resnet_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    print("Optimizer not supported.")
    exit(1)

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
