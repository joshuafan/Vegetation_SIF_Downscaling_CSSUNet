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
from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
from eval_subtile_dataset import EvalSubtileDataset
import tile_transforms
import time
import torch
import torchvision
import torchvision.transforms as transforms
import simple_cnn
import small_resnet
import resnet
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-08-01")
EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv")
# EVAL_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_info_val.csv")

TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/small_tile_sif_prediction")
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_eval_subtile_small_tile_cnn.png' 
PLOT_TITLE = 'Small tile CNN (trained by resizing large tiles, eval subtile)'
INPUT_CHANNELS = 29
eval_points = pd.read_csv(EVAL_FILE)

def eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    model.eval()   # Set model to evaluate mode
    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    predicted = []
    true = []
    running_loss = 0.0

    # Iterate over data.
    for sample in dataloader:
        input_tile = sample['subtile'].to(device)
        #print('=========================')
        #print('Input band means')
        #print(torch.mean(input_tile, dim=(2,3)))
        true_sif_non_standardized = sample['SIF'].to(device)

        # forward
        # track history if only in train
        # with torch.set_grad_enabled(False):
        predicted_sif_standardized = model(input_tile).flatten()
        predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
        loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)

        # statistics
        running_loss += loss.item() * len(sample['SIF'])
        predicted += predicted_sif_non_standardized.tolist()
        true += true_sif_non_standardized.tolist()
    loss = math.sqrt(running_loss / dataset_size) / sif_mean
    return loss, predicted, true


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
print("Eval samples", len(eval_metadata))

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
print("Train Means", train_means)
print("Train stds", train_stds)
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.ShrinkTile())
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
transform = transforms.Compose(transform_list)

# Set up Dataset and Dataloader
dataset_size = len(eval_metadata)
# dataset = ReflectanceCoverSIFDataset(eval_metadata, transform)
dataset = EvalSubtileDataset(eval_metadata, transform)  # ReflectanceCoverSIFDataset(eval_metadata, transform) #  EvalSubtileDataset(eval_metadata, transform)  #    ReflectanceCoverSIFDataset(eval_metadata, transform)  # ReflectanceCoverSIFDataset(eval_metadata, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                         shuffle=True, num_workers=4)

# Load trained model from file
resnet_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, output_dim=1)  # small_resnet.resnet18(input_channels=INPUT_CHANNELS)
# resnet_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=1)  #.to(device)
resnet_model.load_state_dict(torch.load(TRAINED_MODEL_FILE))
resnet_model = resnet_model.to(device)
criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
loss, predicted, true = eval_model(resnet_model, dataloader, dataset_size, criterion, device, sif_mean, sif_std)
print('Predicted', predicted[0:50])
print('True', true[0:50])
print("Eval Loss", loss)

# Compare predicted vs true: calculate NRMSE, R2, scatter plot
nrmse = math.sqrt(mean_squared_error(predicted, true)) / sif_mean
corr, _ = pearsonr(predicted, true)
print('NRMSE:', round(nrmse, 3))
print("Pearson's correlation:", round(corr, 3))

# Scatter plot of true vs predicted
plt.scatter(true, predicted)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(PLOT_TITLE)
plt.savefig(TRUE_VS_PREDICTED_PLOT)
plt.close()
