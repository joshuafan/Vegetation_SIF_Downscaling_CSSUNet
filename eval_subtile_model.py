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



DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-07-17")
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-07-17")
EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv") 
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_recon/TileNet.ckpt")
# TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_dim512_neighborhood100/finetuned_tile2vec.ckpt"

EMBEDDING_TO_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/avg_embedding_to_sif")
# EMBEDDING_TO_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/finetuned_embedding_to_sif.ckpt")

METHOD = "avg_embedding"
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_eval_subtile_' + METHOD
# TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_eval_subtile_finetuned_tile2vec.ping'

COLUMN_NAMES = ['predicted','true',
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']
RESULTS_CSV_FILE = os.path.join(EVAL_DATASET_DIR, 'results_' + METHOD + '.csv')
Z_DIM = 43
HIDDEN_DIM = 1024
INPUT_CHANNELS = 43
COVER_INDICES = list(range(12, 42))
EMBEDDING_TYPE = 'average'  # average'  # 'tile2vec'  # average'  # 'tile2vec'

eval_points = pd.read_csv(EVAL_FILE)


def eval_model(tile2vec_model, embedding_to_sif_model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    if EMBEDDING_TYPE == 'tile2vec':
        tile2vec_model.eval()   # Set model to evaluate mode
    embedding_to_sif_model.eval()
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    results = np.zeros((dataset_size, 2+len(COVER_INDICES)))
    running_loss = 0.0
    j = 0

    # Iterate over data.
    for sample in dataloader:
        input_tile_standardized = sample['subtile'].to(device)
        true_sif_non_standardized = sample['SIF'].to(device)
        # print('Input tile shape', input_tile_standardized.shape)

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
        loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)

        # statistics
        batch_size = len(sample['SIF'])
        running_loss += loss.item() * batch_size
        band_means = torch.mean(input_tile_standardized[:, COVER_INDICES, :, :], dim=(2,3))
        results[j:j+batch_size, 0] = predicted_sif_non_standardized.numpy()
        results[j:j+batch_size, 1] = true_sif_non_standardized.numpy()
        results[j:j+batch_size, 2:] = band_means
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
if EMBEDDING_TYPE == 'tile2vec':
    tile2vec_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=Z_DIM).to(device)
    tile2vec_model.load_state_dict(torch.load(TILE2VEC_MODEL_FILE, map_location=device))
else:
    tile2vec_model = None
embedding_to_sif_model = EmbeddingToSIFNonlinearModel(embedding_size=Z_DIM, hidden_size=HIDDEN_DIM).to(device)
embedding_to_sif_model.load_state_dict(torch.load(EMBEDDING_TO_SIF_MODEL_FILE, map_location=device))

criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
results_numpy = eval_model(tile2vec_model, embedding_to_sif_model, dataloader, dataset_size, criterion, device, sif_mean, sif_std)
results_df = pd.DataFrame(results_numpy, columns=COLUMN_NAMES)

results_df.to_csv(RESULTS_CSV_FILE)

predicted = results_df['predicted'].tolist()
true = results_df['true'].tolist()
print('Predicted', predicted[0:50])
print('True', true[0:50])

# Compare predicted vs true: calculate NRMSE, R2, scatter plot
nrmse = math.sqrt(mean_squared_error(predicted, true)) / sif_mean
corr, _ = pearsonr(predicted, true)
print('NRMSE:', round(nrmse, 3))
print("Pearson's correlation coefficient:", round(corr, 3))

# Scatter plot of true vs predicted
plt.scatter(true, predicted)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('True vs predicted SIF (CFIS):' + METHOD)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '.png')
plt.close()

for crop_type in COLUMN_NAMES[2:]:
    crop_rows = results_df.loc[results_df[crop_type] > 0.5]
    print(len(crop_rows), 'subtiles that are majority', crop_type)

    # Scatter plot of true vs predicted
    plt.scatter(crop_rows['true'], crop_rows['predicted'])
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('True vs predicted SIF (CFIS, pure ' + crop_type + '): ' + METHOD)
    plt.savefig(TRUE_VS_PREDICTED_PLOT + '_' + crop_type + '.png')
    plt.close()


