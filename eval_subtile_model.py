import copy
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from eval_subtile_dataset import EvalSubtileDataset
import time
import torch
import torchvision
import torchvision.transforms as transforms
import resnet
import torch.nn as nn
import torch.optim as optim

# Don't know how to properly import from Tile2Vec
# TODO this is a hack
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet
from embedding_to_sif_model import EmbeddingToSIFModel
import tile_transforms



EVAL_DATASET_DIR = "datasets/dataset_2016-08-01"
EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv")  #"datasets/generated_subtiles/eval_subtiles.csv" 

TRAIN_DATASET_DIR = "datasets/dataset_2018-08-01"
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
TILE2VEC_MODEL_FILE = "models/tile2vec/TileNet_epoch50.ckpt"
EMBEDDING_TO_SIF_MODEL_FILE = "models/avg_reflectance_to_sif"
Z_DIM = 32
INPUT_CHANNELS= 14

eval_points = pd.read_csv(EVAL_FILE)


def eval_model(tile2vec_model, embedding_to_sif_model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    #tile2vec_model.eval()   # Set model to evaluate mode
    embedding_to_sif_model.eval()
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)


    running_loss = 0.0

    # Iterate over data.
    for sample in dataloader:
        input_tile_standardized = sample['subtile'].to(device)
        true_sif_non_standardized = sample['SIF'].to(device)
        # print('Input tile shape', input_tile_standardized.shape)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # embedding = tile2vec_model(input_tile_standardized).flatten()
            embedding = torch.mean(input_tile_standardized, dim=(2,3))
            #print('Embedding', embedding)
            # print('Embedding shape', embedding.shape)
            predicted_sif_standardized = embedding_to_sif_model(embedding).flatten()
        predicted_sif_non_standardized = predicted_sif_standardized * sif_std + sif_mean
        #print('Predicted:', predicted_sif_non_standardized)
        #print('True:', true_sif_non_standardized)
        loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)

        # statistics
        running_loss += loss.item() * len(sample['SIF'])

    loss = math.sqrt(running_loss / dataset_size) / sif_mean
    return loss



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", device)

# Read train/val tile metadata
eval_metadata = pd.read_csv(EVAL_FILE)
average_sif = eval_metadata['SIF'].mean()
print("Average sif", average_sif)
print("Eval samples", len(eval_metadata))

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
print("Validation samples", len(eval_metadata))
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

# Set up Dataset and Dataloader
dataset_size = len(eval_metadata)
dataset = EvalSubtileDataset(eval_metadata, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                         shuffle=True, num_workers=4)

# Load trained models from file
#tile2vec_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=Z_DIM).to(device)
#tile2vec_model.load_state_dict(torch.load(TILE2VEC_MODEL_FILE))
tile2vec_model = None
embedding_to_sif_model = EmbeddingToSIFModel(embedding_size=INPUT_CHANNELS)
embedding_to_sif_model.load_state_dict(torch.load(EMBEDDING_TO_SIF_MODEL_FILE))
embedding_to_sif_model = embedding_to_sif_model.to(device)

criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
loss = eval_model(tile2vec_model, embedding_to_sif_model, dataloader, dataset_size, criterion, device, sif_mean, sif_std)
print("Eval Loss", loss)


