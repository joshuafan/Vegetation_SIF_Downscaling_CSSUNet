import copy
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from torch.optim import lr_scheduler
from eval_subtile_dataset import EvalSubtileDataset
import time
import torch
import torchvision
import torchvision.transforms as transforms
import resnet
import torch.nn as nn
import torch.optim as optim
from sif_utils import print_stats

# Don't know how to properly import from Tile2Vec
# TODO this is a hack
import sys
sys.path.append('../')
import simple_cnn
import tile_transforms


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-08-01") #07-16")
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01") #07-16")
EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles_val.csv") 
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/cfis_sif") #subtile_sif_simple_cnn_9")  # "models/subtile_sif_simple_cnn_4")
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_eval_subtile_cheating_cfis' #simple_cnn.png'

INPUT_CHANNELS = 43
MIN_SIF = 0.2
MAX_SIF = 1.7

eval_points = pd.read_csv(EVAL_FILE)


def eval_model(subtile_sif_model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    subtile_sif_model.eval()
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    true = []
    predicted = []
    running_loss = 0.0

    # Iterate over data.
    for sample in dataloader:
        input_tile_standardized = sample['subtile'].to(device)
        true_sif_non_standardized = 1.52 * sample['SIF'].to(device)
        #print('Sample tile', input_tile_standardized[0, :, 8, 8])

        # forward
        with torch.set_grad_enabled(False):
            predicted_sif_standardized = subtile_sif_model(input_tile_standardized).flatten()
        predicted_sif_non_standardized = predicted_sif_standardized * sif_std + sif_mean
        predicted_sif_non_standardized = torch.clamp(predicted_sif_non_standardized, min=0.2, max=1.7)
        loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)

        # statistics
        running_loss += loss.item() * len(sample['SIF'])
        true += true_sif_non_standardized.tolist()
        predicted += predicted_sif_non_standardized.tolist()
    return true, predicted


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
print("Validation samples", len(eval_metadata))
print("Means", train_means)
print("Stds", train_stds)
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]
min_output = (MIN_SIF - sif_mean) / sif_std
max_output = (MAX_SIF - sif_mean) / sif_std


# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
transform = transforms.Compose(transform_list)

# Set up Dataset and Dataloader
dataset_size = len(eval_metadata)
dataset = EvalSubtileDataset(eval_metadata, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                         shuffle=True, num_workers=4)

subtile_sif_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=43, output_dim=1, min_output=min_output, max_output=max_output).to(device)
subtile_sif_model.load_state_dict(torch.load(SUBTILE_SIF_MODEL_FILE, map_location=device))

criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
true, predicted = eval_model(subtile_sif_model, dataloader, dataset_size, criterion, device, sif_mean, sif_std)

print_stats(true, predicted, sif_mean)

# Scatter plot of true vs predicted
plt.scatter(true, predicted)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Subtile prediction with simple CNN')
plt.savefig(TRUE_VS_PREDICTED_PLOT)
plt.close()

# Quantile analysis
true = np.array(true)
predicted = np.array(predicted)
squared_errors = (true - predicted) ** 2
indices = squared_errors.argsort() #Ascending order of squared error

percentiles = [0, 0.05, 0.1, 0.2]
for percentile in percentiles:
    cutoff_idx = int((1 - percentile) * len(true))
    indices_to_include = indices[:cutoff_idx]
    nrmse = math.sqrt(np.mean(squared_errors[indices_to_include])) / sif_mean
    corr, _ = pearsonr(true[indices_to_include], predicted[indices_to_include])
    print('Excluding ' + str(int(percentile*100)) + '% worst predictions')
    print('NRMSE', round(nrmse, 3))
    print('Corr', round(corr, 3))


