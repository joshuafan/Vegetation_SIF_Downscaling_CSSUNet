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
import tile_transforms


# TODO this is a hack
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet
from embedding_to_sif_model import EmbeddingToSIFModel


DATASET_DIR = "datasets/dataset_2018-08-01"
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
TILE2VEC_MODEL_FILE = "models/tile2vec/TileNet_epoch50.ckpt"
EMBEDDING_TO_SIF_MODEL_FILE = "models/embedding_to_sif"
FREEZE_TILE2VEC = True
Z_DIM = 32
INPUT_CHANNELS= 14
LEARNING_RATE = 5e-3
NUM_EPOCHS = 20
SUBTILE_DIM = 10

# For each tile in the batch, returns a list of subtiles.
# Given a Tensor of tiles, with shape (batch x C x W x H), returns a Tensor of
# shape (batch x SUBTILE x C x subtile_dim x subtile_dim)
def get_subtiles_list(tile, subtile_dim, device):
    print('Tile shape', tile.shape)
    batch_size, bands, width, height = tile.shape
    num_subtiles_along_width = int(width / subtile_dim)
    num_subtiles_along_height = int(height / subtile_dim)
    num_subtiles = num_subtiles_along_width * num_subtiles_along_height
    assert(num_subtiles_along_width == 37)
    assert(num_subtiles_along_height == 37)
    subtiles = torch.empty((batch_size, num_subtiles, bands, subtile_dim, subtile_dim), device=device)
    for b in range(batch_size):
        subtile_idx = 0
        for i in range(num_subtiles_along_width):
            for j in range(num_subtiles_along_height):
                subtile = tile[b, :, subtile_dim*i:subtile_dim*(i+1), subtile_dim*j:subtile_dim*(j+1)].to(device)
                subtiles[b, subtile_idx, :, :, :] = subtile
                subtile_idx += 1
    return subtiles


# TODO precompute embedding
def create_subtile_embeddings_to_sif_dataset(tile2vec_model, dataloader, subtile_dim, z_dim, output_file):
    tile2vec_model.eval()
    csv_rows = []
    for sample in dataloader:
        input_tile_standardized = sample['tile'].to(device)
        true_sif_non_standardized = sample['SIF'].to(device)
        subtiles = get_subtiles_list(input_tile_standardized, subtile_dim)
        subtile_embeddings = np.empty((len(subtiles), z_dim))
        for i in range(len(subtiles)):
            subtile_embeddings[i] = tile2vec_model(subtiles[i])
        


# "means" is a list of band averages (+ average SIF at end)
# TODO should there be 2 separate models?
def train_model(tile2vec_model, embedding_to_sif_model, freeze_tile2vec, dataloaders, dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, subtile_dim, num_epochs=25):
    since = time.time()

    best_model_wts = [copy.deepcopy(tile2vec_model.state_dict()),
                      copy.deepcopy(embedding_to_sif_model.state_dict())]
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
                if freeze_tile2vec:
                    tile2vec_model.eval()
                else:
                    tile2vec_model.train()
                embedding_to_sif_model.train()
            else:
                tile2vec_model.eval()
                embedding_to_sif_model.eval()

            running_loss = 0.0

            # Iterate over data.
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
                    for i in range(batch_size):
                        # embeddings = torch.mean(subtiles[i], dim=(2,3))  # (num subtiles x embedding size)
                        embeddings = tile2vec_model(subtiles[i])
                        #print('Embedding shape', embeddings.shape)
                        predicted_sifs = embedding_to_sif_model(embeddings)
                        #print('predicted_sif shape', predicted_sifs.shape)
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
                    print('========================')
                    print('Predicted', predicted_sif_non_standardized)
                    print('True', true_sif_non_standardized)
                    non_standardized_loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)
                    running_loss += non_standardized_loss.item()

            epoch_loss = math.sqrt(running_loss / dataset_sizes[phase]) / sif_mean

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = [copy.deepcopy(tile2vec_model.state_dict()),
                                  copy.deepcopy(embedding_to_sif_model.state_dict())]

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
    tile2vec_model.load_state_dict(best_model_wts[0])
    embedding_to_sif_model.load_state_dict(best_model_wts[1])
    return tile2vec_model, embedding_to_sif_model, train_losses, val_losses, best_loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
transform = transforms.Compose(transform_list)

# Set up Datasets and Dataloaders
# resize_transform = torchvision.transforms.Resize((224, 224))
datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, transform),
            'val': ReflectanceCoverSIFDataset(val_metadata, transform)}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=2,
                                              shuffle=True, num_workers=1)
              for x in ['train', 'val']}

print("Dataloaders")

tile2vec_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=Z_DIM).to(device)
tile2vec_model.load_state_dict(torch.load(TILE2VEC_MODEL_FILE))
embedding_to_sif_model = EmbeddingToSIFModel(embedding_size=Z_DIM).to(device)  # TODO
criterion = nn.MSELoss(reduction='mean')
#optimizer = optim.SGD(resnet_model.parameters(), lr=1e-4, momentum=0.9)

# Don't optimize Tile2vec model; just use pre-trained version
if FREEZE_TILE2VEC:
    params_to_optimize = embedding_to_sif_model.parameters()
else:
    params_to_optimize = list(tile2vec_model.parameters()) + list(embedding_to_sif_model.parameters())
optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE)
dataset_sizes = {'train': len(train_metadata),
                 'val': len(val_metadata)}


tile2vec_model, embedding_to_sif_model, train_losses, val_losses, best_loss = train_model(tile2vec_model, embedding_to_sif_model, FREEZE_TILE2VEC, dataloaders, dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, subtile_dim=SUBTILE_DIM, num_epochs=NUM_EPOCHS)

torch.save(embedding_to_sif_model.state_dict(), EMBEDDING_TO_SIF_MODEL_FILE)

# Plot loss curves
print("Train losses:", train_losses)
print("Validation losses:", val_losses)
epoch_list = range(NUM_EPOCHS)
train_plot, = plt.plot(epoch_list, train_losses, color='blue', label='Train NRMSE')
val_plot, = plt.plot(epoch_list, val_losses, color='red', label='Validation NRMSE')
plt.legend(handles=[train_plot, val_plot])
plt.xlabel('Epoch #')
plt.ylabel('Normalized Root Mean Squared Error')
plt.savefig('plots/resnet_sif_prediction.png')
plt.close()
