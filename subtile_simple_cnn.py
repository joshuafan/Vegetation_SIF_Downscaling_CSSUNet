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
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-07-16")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
TRAINING_PLOT_FILE = 'exploratory_plots/losses_subtile_simple_cnn_11.png'

PRETRAINED_SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/small_tile_simple")
SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/subtile_sif_simple_cnn_12")
INPUT_CHANNELS = 43
LEARNING_RATE = 1e-3  # 0.001  #1e-4i
WEIGHT_DECAY = 1e-3 #0 #1e-6
NUM_EPOCHS = 50
SUBTILE_DIM = 10
BATCH_SIZE = 32 
NUM_WORKERS = 4
AUGMENT = True
FROM_PRETRAINED = False #True  #False  # True
MIN_SIF = 0.2
MAX_SIF = 1.7

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

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                subtile_sif_model.train()
            else:
                subtile_sif_model.eval()

            running_loss = 0.0

            # Iterate over data.
            j = 0
            for sample in dataloaders[phase]:
                batch_size = len(sample['SIF'])
                input_tile_standardized = sample['tile'].to(device)
                true_sif_non_standardized = sample['SIF'].to(device)
                true_sif_standardized = ((true_sif_non_standardized - sif_mean) / sif_std).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Obtain subtiles (NOTE Pay attention to standardization :( )
                subtiles = get_subtiles_list(input_tile_standardized, subtile_dim, device)  # (batch x num subtiles x bands x subtile_dim x subtile_dim)
                #print('subtiles returned by get_subtiles_list', subtiles.shape)
                #print('0th example:', subtiles[0].shape)
                predicted_subtile_sifs = torch.empty((batch_size, subtiles.shape[1]), device=device)
                #print('Predicted subtile SIFs', predicted_subtile_sifs.shape)

                # Forward pass: feed subtiles through embedding model and then the
                # embedding -> SIF model
                with torch.set_grad_enabled(phase == 'train'):
                    #print(' ====== Random pixels =======')
                    #print(subtiles[0, 0, :, 8, 8])
                    #print(subtiles[0, 0, :, 8, 9])
                    #print(subtiles[0, 0, :, 9, 9])
 
                    for i in range(batch_size):
                        #print('Embedding shape', embeddings.shape)
                        predicted_sifs = subtile_sif_model(subtiles[i])
                        predicted_subtile_sifs[i] = predicted_sifs.flatten()
                    
                    # Predicted SIF for full tile
                    predicted_sif_standardized = torch.mean(predicted_subtile_sifs, axis=1)
                    #print('Shape of predicted total SIFs', predicted_sif_standardized.shape)
                    #print('Shape of true total SIFs', true_sif_standardized.shape)
                    loss = criterion(predicted_sif_standardized, true_sif_standardized)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                    non_standardized_loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)
                    j += 1
                    if j % 100 == 0:
                        print('========================')
                        print('*** >>> predicted subtile sifs for 0th', predicted_subtile_sifs[0] * sif_std + sif_mean)
                        print('*** Predicted', predicted_sif_non_standardized)
                        print('*** True', true_sif_non_standardized)
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
train_metadata = pd.read_csv(INFO_FILE_TRAIN) #.iloc[0:200]
val_metadata = pd.read_csv(INFO_FILE_VAL) #.iloc[0:200]

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
min_output = None #(MIN_SIF - sif_mean) / sif_std
max_output = None #(MAX_SIF - sif_mean) / sif_std


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

# subtile_sif_model = small_resnet.resnet18(input_channels=INPUT_CHANNELS)
subtile_sif_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=43, output_dim=1, min_output=min_output, max_output=max_output).to(device)
if FROM_PRETRAINED:
    subtile_sif_model.load_state_dict(torch.load(PRETRAINED_SUBTILE_SIF_MODEL_FILE, map_location=device))

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(subtile_sif_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
