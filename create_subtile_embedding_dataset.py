"""
Learns embedding-to-SIF model, given pre-computed (fixed) embeddings.

If LOAD_EMBEDDINGS is true, embeddings will be precomputed from SUBTILE_EMBEDDING_DATASET_FILE.
Otherwise, they will be computed at the start, and will be saved to SUBTILE_EMBEDDING_DATASET_FILE.
"""
# TODO
import copy
import csv
import pickle
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
from subtile_embedding_dataset import SubtileEmbeddingDataset

from sif_utils import get_subtiles_list
import tile_transforms


# TODO this is a hack
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet
from embedding_to_sif_model import EmbeddingToSIFModel


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_pixels.csv")
# SUBTILE_EMBEDDING_FILE_TRAIN = os.path.join(DATASET_DIR, "avg_embeddings_train.csv")
# SUBTILE_EMBEDDING_FILE_VAL = os.path.join(DATASET_DIR, "avg_embeddings_val.csv")
SUBTILE_EMBEDDING_FILE_TRAIN = os.path.join(DATASET_DIR, "standardized_tiles_train.csv")
SUBTILE_EMBEDDING_FILE_VAL = os.path.join(DATASET_DIR, "standardized_tiles_val.csv")
# SUBTILE_EMBEDDING_FILE_TRAIN = os.path.join(DATASET_DIR, "resized_tiles_train.csv")
# SUBTILE_EMBEDDING_FILE_VAL = os.path.join(DATASET_DIR, "resized_tiles_val.csv")
EMBEDDING_FILE_SUFFIX = '_avg_embeddings.npy'
STANDARDIZED_TILE_FILE_SUFFIX = '_standardized.npy'
STANDARDIZED_SUBTILES_FILE_SUFFIX = '_standardized_subtiles.npy'
RESIZED_TILE_FILE_SUFFIX = '_resized_tile.npy'

# TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_august/TileNet.ckpt")

# If EMBEDDING_TYPE is 'average', the embedding is just the average of each band.
# If it is 'tile2vec', we use the Tile2Vec model 
EMBEDDING_TYPE = 'average' #tile2vec'  #average' # 'tile2vec'
# TRAINING_PLOT_FILE = 'exploratory_plots/tile2vec_subtile_sif_prediction.png'
SUBTILE_DIM = 100
Z_DIM = 256
INPUT_CHANNELS = 43
NUM_WORKERS = 8


# For each tile returned by the dataloader, obtain a list of embeddings for each subtile. Save
# it to a .npy file.
def compute_subtile_embeddings_to_sif_dataset(tile2vec_model, dataloader, subtile_dim, device):
    if tile2vec_model is not None:
        tile2vec_model.eval()
    tile_rows = [['embedding_file', 'sif']]
    for sample in dataloader:
        batch_size = len(sample['SIF'])
        input_tile_standardized = sample['tile']
        true_sif_non_standardized = sample['SIF']
        filenames = sample['tile_file']
        for i in range(batch_size):
            #print('Random pixel', subtiles[i, 0, :, 5, 5])
            subtiles = get_subtiles_list(input_tile_standardized[i], subtile_dim) #, device) #, max_subtile_cloud_cover=MAX_SUBTILE_CLOUD_COVER)
            print('Subtiles shape', subtiles.shape)
            if EMBEDDING_TYPE == 'tile2vec':
                with torch.set_grad_enabled(False):
                    embeddings = tile2vec_model(subtiles)
            elif EMBEDDING_TYPE == 'average':
                embeddings = torch.mean(subtiles, dim=(2,3))
            else:
                print('Unsupported embedding type', EMBEDDING_TYPE)
                exit(1)
            embedding_filename = filenames[i] + EMBEDDING_FILE_SUFFIX
            print('Embedding filename', embedding_filename)
            np.save(embedding_filename, embeddings.cpu().numpy())
            tile_rows.append([embedding_filename, true_sif_non_standardized[i].item()])
    return tile_rows


# For each tile returned by the dataloader, compute the standardized tile, and standardized
# sub-tiles. Save them to .npy files.
def compute_standardized_tiles_to_sif_dataset(dataloader, subtile_dim, device):
    tile_rows = [['lon', 'lat', 'standardized_tile_file', 'standardized_subtiles_file', 'source', 'date', 'SIF']]
    for sample in dataloader:
        batch_size = len(sample['SIF'])
        input_tiles_standardized = sample['tile']
        true_sifs_non_standardized = sample['SIF']
        filenames = sample['tile_file']
        for i in range(batch_size):
            standardized_tile_filename = filenames[i] + STANDARDIZED_TILE_FILE_SUFFIX
            np.save(standardized_tile_filename, input_tiles_standardized[i])
            subtiles = get_subtiles_list(input_tiles_standardized[i], subtile_dim) #, device, max_subtile_cloud_cover=MAX_SUBTILE_CLOUD_COVER)
            subtiles_filename = filenames[i] + STANDARDIZED_SUBTILES_FILE_SUFFIX
            np.save(subtiles_filename, subtiles)
            tile_rows.append([sample['lon'][i].item(), sample['lat'][i].item(), standardized_tile_filename,
                              subtiles_filename, sample['source'][i], sample['date'][i],
                              true_sifs_non_standardized[i].item()])
            print(tile_rows[-1])
    return tile_rows


# For each tile returned by the dataloader, compute the standardized tile, and standardized
# sub-tiles. Save them to .npy files.
def compute_resized_tile_dataset(dataloader, device):
    tile_rows = [['lon', 'lat', 'resized_tile_file', 'source', 'date', 'SIF']]
    for sample in dataloader:
        batch_size = len(sample['SIF'])
        input_tiles_standardized = sample['tile']
        if np.isnan(input_tiles_standardized).any():
            print('Nan found :(')
            print(sample['tile_file'])
            exit(0)
        assert(input_tiles_standardized.shape[2] == SUBTILE_DIM)
        true_sifs_non_standardized = sample['SIF']
        filenames = sample['tile_file']
        for i in range(batch_size):
            resized_tile_filename = filenames[i] + RESIZED_TILE_FILE_SUFFIX
            np.save(resized_tile_filename, input_tiles_standardized[i])
            tile_rows.append([sample['lon'][i].item(), sample['lat'][i].item(), resized_tile_filename,
                              sample['source'][i], sample['date'][i],
                              true_sifs_non_standardized[i].item()])
            print(tile_rows[-1])
    return tile_rows

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
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
# transform_list.append(tile_transforms.ShrinkTile(target_dim=SUBTILE_DIM))

transform = transforms.Compose(transform_list)

# Set up Datasets and Dataloaders
datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, transform),
            'val': ReflectanceCoverSIFDataset(val_metadata, transform)}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1,
                                                  shuffle=True, num_workers=NUM_WORKERS)
                   for x in ['train', 'val']}

# # Load pre-trained Tile2Vec embedding model
# tile2vec_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=Z_DIM).to(device)
# tile2vec_model.load_state_dict(torch.load(TILE2VEC_MODEL_FILE, map_location=device))
# #tile2vec_model = None
# print('loaded tile2vec')

# Obtain embeddings for all subtiles
# train_tile_rows = compute_subtile_embeddings_to_sif_dataset(tile2vec_model, dataloaders['train'], SUBTILE_DIM, device) 
# val_tile_rows = compute_subtile_embeddings_to_sif_dataset(tile2vec_model, dataloaders['val'], SUBTILE_DIM, device)
train_tile_rows = compute_standardized_tiles_to_sif_dataset(dataloaders['train'], SUBTILE_DIM, device) 
val_tile_rows = compute_standardized_tiles_to_sif_dataset(dataloaders['val'], SUBTILE_DIM, device)
# train_tile_rows = compute_resized_tile_dataset(dataloaders['train'], device) 
# val_tile_rows = compute_resized_tile_dataset(dataloaders['val'], device)
   
# Write subtile embeddings to file
with open(SUBTILE_EMBEDDING_FILE_TRAIN, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in train_tile_rows:
        csv_writer.writerow(row)

with open(SUBTILE_EMBEDDING_FILE_VAL, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in val_tile_rows:
        csv_writer.writerow(row)

    print('Dumped precomputed embeddings.')



