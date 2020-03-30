import copy
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim import lr_scheduler
from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
import tile_transforms
import time
import torch
import torchvision
import torchvision.transforms as transforms
import resnet
import torch.nn as nn
import torch.optim as optim


EVAL_DATASET_DIR = "datasets/dataset_2016-08-01"
EVAL_FILE = "datasets/dataset_2018-08-01/tile_info_train.csv"  # os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv")  #"datasets/generated_subtiles/eval_subtiles.csv" 
TRAINED_MODEL_FILE = "models/large_tile_sif_prediction"
TRAIN_DATASET_DIR = "datasets/dataset_2018-08-01"
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")

eval_points = pd.read_csv(EVAL_FILE)

def eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    model.eval()   # Set model to evaluate mode
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    predicted = []
    true = []
    running_loss = 0.0

    # Iterate over data.
    for sample in dataloader:
        input_tile = sample['tile'].to(device)
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
transform = transforms.Compose(transform_list)

# Set up Dataset and Dataloader
dataset_size = len(eval_metadata)
dataset = ReflectanceCoverSIFDataset(eval_metadata, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                         shuffle=True, num_workers=4)

# Load trained model from file
resnet_model = resnet.resnet18(input_channels=14)
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
r2 = r2_score(predicted, true)
print('NRMSE:', nrmse)
print('R2:', r2)

# Scatter plot of true vs predicted
plt.scatter(true, predicted)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Predict SIF for small tile = large tile CNN prediction')
plt.savefig('exploratory_plots/true_vs_predicted_trivial_large_tile_cnn.png')
plt.close()
