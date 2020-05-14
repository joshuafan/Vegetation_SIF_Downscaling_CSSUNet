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
import tile_transforms


# TODO this is a hack
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet
from embedding_to_sif_model import EmbeddingToSIFModel
from embedding_to_sif_nonlinear_model import EmbeddingToSIFNonlinearModel

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-07-16")
INFO_FILE_TRAIN = os.path.join(TRAIN_DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(TRAIN_DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")

PRETRAINED_TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_recon_5/TileNet.ckpt")
#ILE2VEC_MODEL_FILE = "models/tile2vec_dim10_v2/TileNet_epoch50.ckpt"
FINETUNED_TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_recon_5/finetuned_tile2vec.ckpt")
PRETRAINED_EMBEDDING_TO_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_embedding_to_sif")
EMBEDDING_TO_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/finetuned_tile2vec_embedding_to_sif.ckpt")
#PRETRAINED_EMBEDDING_TO_SIF_MODEL_FILE = EMBEDDING_TO_SIF_MODEL_FILE
TRAINING_PLOT_FILE = 'exploratory_plots/losses_finetuned_tile2vec.png'
EMBEDDING_TYPE = 'tile2vec'

FROM_PRETRAINED_EMBEDDING_TO_SIF = True #False #False # False
FREEZE_TILE2VEC = False # True # False
Z_DIM = 256
HIDDEN_DIM = 1024
INPUT_CHANNELS = 43
LEARNING_RATE_TILE2VEC = 1e-6  #1e-4 #4
LEARNING_RATE_EMBEDDING_TO_SIF = 1e-6 #0.1  #1e-2  # 0.1 #1e-2  # 0.01  #1e-3
WEIGHT_DECAY = 0 # 1e-6
NUM_EPOCHS = 20
SUBTILE_DIM = 10
BATCH_SIZE = 16
NUM_WORKERS = 4
AUGMENT = False #True
MIN_SIF = 0.2
MAX_SIF = 1.7

# TODO should there be 2 separate models?
def train_model(tile2vec_model, embedding_to_sif_model, freeze_tile2vec, dataloaders, dataset_sizes, criterion, tile2vec_optimizer, embedding_to_sif_optimizer, device, sif_mean, sif_std, subtile_dim, num_epochs=25):
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
                    #tile2vec_model.train()
                    tile2vec_model.eval()
                embedding_to_sif_model.train()
            else:
                tile2vec_model.eval()
                embedding_to_sif_model.eval()

            running_loss = 0.0

            # Iterate over data.
            count = 0
            for sample in dataloaders[phase]:
                batch_size = len(sample['SIF'])
                input_tile_standardized = sample['tile'].to(device)
                true_sif_non_standardized = sample['SIF'].to(device)
                true_sif_standardized = ((true_sif_non_standardized - sif_mean) / sif_std).to(device)

                # Obtain subtiles (NOTE Pay attention to standardization :( )
                subtiles = get_subtiles_list(input_tile_standardized, subtile_dim, device)  # (batch x num subtiles x bands x subtile_dim x subtile_dim)
                #print('subtiles returned by get_subtiles_list', subtiles.shape)
                #print('0th example:', subtiles[0].shape)
                predicted_subtile_sifs = torch.empty((batch_size, subtiles.shape[1]), device=device)
                #print('Predicted subtile SIFs', predicted_subtile_sifs.shape)

                # zero the parameter gradients
                if tile2vec_optimizer:
                    tile2vec_optimizer.zero_grad()
                embedding_to_sif_optimizer.zero_grad()


                # Forward pass: feed subtiles through embedding model and then the
                # embedding -> SIF model
                with torch.set_grad_enabled(phase == 'train'):
                    for i in range(batch_size):
                        if EMBEDDING_TYPE == 'average':
                            embeddings = torch.mean(subtiles[i], dim=(2,3))  # (num subtiles x embedding size)
                        elif EMBEDDING_TYPE == 'tile2vec':
                            embeddings = tile2vec_model(subtiles[i])
                        else:
                            print('Unsupported embedding type', EMBEDDING_TYPE)
                            exit(1)
                        #print('Embedding shape', embeddings.shape)
                        predicted_sifs = embedding_to_sif_model(embeddings)
                        predicted_subtile_sifs[i] = predicted_sifs.flatten()
                    
                    # Predicted SIF for full tile
                    predicted_sif_standardized = torch.mean(predicted_subtile_sifs, axis=1)
                    #print('Shape of predicted total SIFs', predicted_sif_standardized.shape)
                    #print('Shape of true total SIFs', true_sif_standardized.shape)
                    loss = criterion(predicted_sif_standardized, true_sif_standardized)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        embedding_to_sif_optimizer.step()
                        #print(tile2vec_model.conv1.weight.grad)
                        if tile2vec_optimizer:
                            tile2vec_optimizer.step()

                    # statistics
                    predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                    non_standardized_loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)
                    running_loss += non_standardized_loss.item() * len(sample['SIF'])
                    if count % 50 == 1:
                        print('========================')
                        #print('***** Band means', torch.mean(input_tile_standardized, dim=(2, 3)))
                        print('***** Predicted', predicted_sif_non_standardized)
                        print('***** True', true_sif_non_standardized)
                        print('***** batch loss', (math.sqrt(non_standardized_loss.item()) / sif_mean).item())
                    count += 1

            epoch_loss = math.sqrt(running_loss / dataset_sizes[phase]) / sif_mean

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = [copy.deepcopy(tile2vec_model.state_dict()),
                                  copy.deepcopy(embedding_to_sif_model.state_dict())]
                torch.save(tile2vec_model.state_dict(), FINETUNED_TILE2VEC_MODEL_FILE)
                torch.save(embedding_to_sif_model.state_dict(), EMBEDDING_TO_SIF_MODEL_FILE)


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
min_output = (MIN_SIF - sif_mean) / sif_std
max_output = (MAX_SIF - sif_mean) / sif_std
print('minoutput', min_output)
print('maxoutput', max_output)

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

tile2vec_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=Z_DIM).to(device)
#tile2vec_model.load_state_dict(torch.load(FINETUNED_TILE2VEC_MODEL_FILE, map_location=device))
tile2vec_model.load_state_dict(torch.load(PRETRAINED_TILE2VEC_MODEL_FILE, map_location=device))
print('Loaded tile2vec from', PRETRAINED_TILE2VEC_MODEL_FILE)

embedding_to_sif_model = EmbeddingToSIFNonlinearModel(embedding_size=Z_DIM, hidden_size=HIDDEN_DIM, min_output=min_output, max_output=max_output).to(device)  # TODO
if FROM_PRETRAINED_EMBEDDING_TO_SIF:
    embedding_to_sif_model.load_state_dict(torch.load(PRETRAINED_EMBEDDING_TO_SIF_MODEL_FILE, map_location=device))
    print('Loaded embedding->SIF model frmo', PRETRAINED_EMBEDDING_TO_SIF_MODEL_FILE)
criterion = nn.MSELoss(reduction='mean')

if FREEZE_TILE2VEC:
    # Don't optimize Tile2vec model; just use pre-trained version
    tile2vec_optimizer = None
else:
    tile2vec_optimizer = optim.Adam(tile2vec_model.parameters(), lr=LEARNING_RATE_TILE2VEC, weight_decay=WEIGHT_DECAY)

embedding_to_sif_optimizer = optim.Adam(embedding_to_sif_model.parameters(), lr=LEARNING_RATE_EMBEDDING_TO_SIF, weight_decay=WEIGHT_DECAY)
#embedding_to_sif_optimizer = optim.Adam(list(embedding_to_sif_model.parameters()) + list(tile2vec_model.parameters()), lr=LEARNING_RATE_EMBEDDING_TO_SIF)

dataset_sizes = {'train': len(train_metadata),
                 'val': len(val_metadata)}


tile2vec_model, embedding_to_sif_model, train_losses, val_losses, best_loss = train_model(tile2vec_model, embedding_to_sif_model, FREEZE_TILE2VEC, dataloaders, dataset_sizes, criterion, tile2vec_optimizer, embedding_to_sif_optimizer, device, sif_mean, sif_std, subtile_dim=SUBTILE_DIM, num_epochs=NUM_EPOCHS)

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
