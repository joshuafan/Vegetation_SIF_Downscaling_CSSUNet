import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
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


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_dim256_neighborhood100/TileNet.ckpt")
# "models/tile2vec_dim10_neighborhood500/TileNet_epoch50.ckpt"
EMBEDDING_TO_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/avg_embedding_to_sif")  # "models/tile2vec_dim10_embedding_to_sif"
EMBEDDING_TYPE = "average"
Z_DIM = 256
INPUT_CHANNELS = 29
SUBTILE_DIM = 10


# Run model on all large tiles, return predicted and true total SIF for large tile
def eval_model(tile2vec_model, embedding_to_sif_model, dataloader, device, sif_mean, sif_std, subtile_dim):
    since = time.time()

    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)

    tile2vec_model.eval()
    embedding_to_sif_model.eval()
    predicted = []
    true = []

    # Iterate over data.
    for sample in dataloader:
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

        # Forward pass: feed subtiles through embedding model and then the
        # embedding -> SIF model
        with torch.set_grad_enabled(False):
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
                #print('predicted_sif shape', predicted_sifs.shape)
                predicted_subtile_sifs[i] = predicted_sifs.flatten()
            
            # Predicted SIF for full tile
            predicted_sif_standardized = torch.mean(predicted_subtile_sifs, axis=1)

            # statistics
            predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float)
            #print('========================')
            #print('Predicted', predicted_sif_non_standardized)
            #print('True', true_sif_non_standardized)
        predicted += predicted_sif_non_standardized.tolist()
        true += true_sif_non_standardized.tolist()
   
    return predicted, true


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

if EMBEDDING_TYPE == 'average':
    embedding_size = INPUT_CHANNELS
else:
    embedding_size = Z_DIM
embedding_to_sif_model = EmbeddingToSIFModel(embedding_size=embedding_size).to(device)  # TODO
embedding_to_sif_model.load_state_dict(torch.load(EMBEDDING_TO_SIF_MODEL_FILE))


predicted_train, true_train = eval_model(tile2vec_model, embedding_to_sif_model, dataloaders['train'], device, sif_mean, sif_std, subtile_dim=SUBTILE_DIM)
predicted_val, true_val = eval_model(tile2vec_model, embedding_to_sif_model, dataloaders['val'], device, sif_mean, sif_std, subtile_dim=SUBTILE_DIM)

train_nrmse = math.sqrt(mean_squared_error(predicted_train, true_train)) / sif_mean
train_corr, _ = pearsonr(predicted_train, true_train)
val_nrmse = math.sqrt(mean_squared_error(predicted_val, true_val)) / sif_mean
val_corr, _ = pearsonr(predicted_val, true_val)

print('Train NRMSE', round(train_nrmse, 3))
print('Val NRMSE', round(val_nrmse, 3))
print('Train correlation', round(train_corr, 3))
print('Val correlation', round(val_corr, 3))

