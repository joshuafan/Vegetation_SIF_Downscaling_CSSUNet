"""
Evaluates models that first compute an embedding from each subtile, then predict SIF.
"""

import copy
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
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
from embedding_to_sif_nonlinear_model import EmbeddingToSIFNonlinearModel
import tile_transforms
from sif_utils import print_stats


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-08-01") # "dataset_2016-07-16")
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01") #"dataset_2018-07-16")
EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv") 
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_recon_5/TileNet.ckpt") #finetuned_tile2vec.ckpt")
# TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_dim512_neighborhood100/finetuned_tile2vec.ckpt"

EMBEDDING_TO_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/avg_embedding_to_sif")
#EMBEDDING_TO_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_embedding_to_sif")
#EMBEDDING_TO_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/finetuned_tile2vec_embedding_to_sif.ckpt")
# EMBEDDING_TO_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/finetuned_embedding_to_sif.ckpt")
METHOD = "4b_subtile_avg"
#METHOD = "4c_tile2vec_fixed"
#METHOD = "4d_tile2vec_finetuned" #4b_subtile_avg" #"tile2vec_finetuned"
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_AUG_eval_subtile_' + METHOD
EMBEDDING_TYPE = 'average'  # 'tile2vec'  # average'  # 'tile2vec'


COLUMN_NAMES = ['true', 'predicted',
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']
RESULTS_CSV_FILE = os.path.join(EVAL_DATASET_DIR, 'results_' + METHOD + '.csv')
Z_DIM = 43 # 256 # 43
HIDDEN_DIM = 1024
INPUT_CHANNELS = 43
COVER_INDICES = list(range(12, 42))
MIN_SIF = 0.2
MAX_SIF = 1.7

def eval_model(tile2vec_model, embedding_to_sif_model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    if EMBEDDING_TYPE == 'tile2vec':
        tile2vec_model.eval()   # Set model to evaluate mode
    embedding_to_sif_model.eval()
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    results = np.zeros((dataset_size, 2+len(COVER_INDICES)))
    j = 0

    # Iterate over data.
    for sample in dataloader:
        input_tile_standardized = sample['subtile'].to(device)
        true_sif_non_standardized = sample['SIF'].to(device)
        #print('Band means', torch.mean(input_tile_standardized, dim=(2,3)))# print('Input tile shape', input_tile_standardized.shape)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # embedding = tile2vec_model(input_tile_standardized)
            if EMBEDDING_TYPE == 'average':
                embedding = torch.mean(input_tile_standardized, dim=(2,3))
            elif EMBEDDING_TYPE == 'tile2vec':
                embedding = tile2vec_model(input_tile_standardized)
            else:
                print('Unsupported embedding type', EMBEDDING_TYPE)
                exit(1)

            #print('Embedding', embedding)
            # print('Embedding shape', embedding.shape)
            predicted_sif_standardized = embedding_to_sif_model(embedding).flatten()
        predicted_sif_non_standardized = predicted_sif_standardized * sif_std + sif_mean
        print('PRedicted', predicted_sif_non_standardized)
        print('True', true_sif_non_standardized)
        loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)

        # statistics
        batch_size = len(sample['SIF'])
        band_means = torch.mean(input_tile_standardized[:, COVER_INDICES, :, :], dim=(2,3))
        results[j:j+batch_size, 0] = true_sif_non_standardized.cpu().numpy()
        results[j:j+batch_size, 1] = predicted_sif_non_standardized.cpu().numpy()
        results[j:j+batch_size, 2:] = band_means.cpu().numpy()
        j += batch_size
        #if j > 50:
        #    break
    return results


# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)

# Read train/val tile metadata
eval_metadata = pd.read_csv(EVAL_FILE)
average_sif = eval_metadata['SIF'].mean()
print("(Eval subtile) Average sif", average_sif)
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

# Constrain predicted SIF to be between 0.2 and 1.7 (unstandardized)
# Don't forget to standardize
min_output = (MIN_SIF - sif_mean) / sif_std
max_output = (MAX_SIF - sif_mean) / sif_std


# Set up Dataset and Dataloader
dataset_size = len(eval_metadata)
dataset = EvalSubtileDataset(eval_metadata, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                         shuffle=True, num_workers=4)

# Load trained models from file
if EMBEDDING_TYPE == 'tile2vec':
    tile2vec_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=Z_DIM).to(device)
    tile2vec_model.load_state_dict(torch.load(TILE2VEC_MODEL_FILE, map_location=device))
else:
    tile2vec_model = None
embedding_to_sif_model = EmbeddingToSIFNonlinearModel(embedding_size=Z_DIM, hidden_size=HIDDEN_DIM, min_output=min_output, max_output=max_output).to(device)
embedding_to_sif_model.load_state_dict(torch.load(EMBEDDING_TO_SIF_MODEL_FILE, map_location=device))

criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
results_numpy = eval_model(tile2vec_model, embedding_to_sif_model, dataloader, dataset_size, criterion, device, sif_mean, sif_std)
results_df = pd.DataFrame(results_numpy, columns=COLUMN_NAMES)
results_df.to_csv(RESULTS_CSV_FILE)

true = results_df['true'].tolist()
predicted = results_df['predicted'].tolist()

# Print statistics
print_stats(true, predicted, sif_mean)

# Scatter plot of true vs predicted
plt.scatter(true, predicted)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('True vs predicted SIF (CFIS):' + METHOD)
plt.xlim(left=0, right=2)
plt.ylim(bottom=0, top=2)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '.png')
plt.close()

# Plot true vs predicted for each crop
fig, axeslist = plt.subplots(ncols=3, nrows=10, figsize=(15, 50))
fig.suptitle('True vs predicted SIF (CFIS): ' + METHOD)
for idx, crop_type in enumerate(COLUMN_NAMES[2:]):
    crop_rows = results_df.loc[results_df[crop_type] > 0.5]
    print(len(crop_rows), 'subtiles that are majority', crop_type)
    axeslist.ravel()[idx].scatter(crop_rows['true'], crop_rows['predicted'])
    axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    axeslist.ravel()[idx].set_xlim(left=0, right=2)
    axeslist.ravel()[idx].set_ylim(bottom=0, top=2)
    axeslist.ravel()[idx].set_title(crop_type)

plt.tight_layout()
fig.subplots_adjust(top=0.96)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()


