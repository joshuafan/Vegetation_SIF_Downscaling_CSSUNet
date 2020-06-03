"""
Evaluate a "subtile -> SIF" model on the OCO-2 dataset
"""
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
from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
from eval_subtile_dataset import EvalSubtileDataset #EvalOCO2Dataset
from sif_utils import print_stats, get_subtiles_list_by_crop
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
#EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-08-01")
#OCO2_EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv")
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
OCO2_EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "oco2_eval_subtiles.csv")
TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn_croptype_4crop")

BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
METHOD = '4a_subtile_simple_cnn_small'  #'3_small_tile_simple' # '2_large_tile_resnet18'
MODEL_TYPE = 'simple_cnn_small'
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_OCO2_eval_subtile_' + METHOD 
MODEL_SUBTILE_DIM = 10
MAX_SUBTILE_CLOUD_COVER = 0.1
COLUMN_NAMES = ['true', 'predicted',
                    'lon', 'lat',
                    'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils', 'missing_reflectance']
CROP_TYPES = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']
RESULTS_CSV_FILE = os.path.join(EVAL_DATASET_DIR, 'OCO2_results_' + METHOD + '.csv')

# CROP_TYPE_INDICES = list(range(12, 42))
# BANDS = list(range(0, 12)) + CROP_TYPES + [42] # Don't include crop types?
CROP_TYPE_INDICES = [12, 13, 14, 16]
FEATURE_BANDS = list(range(0, 12)) + [42]
INPUT_CHANNELS = len(FEATURE_BANDS)
REDUCED_CHANNELS = 8
PURE_THRESHOLD = 0.5
RESIZE = False
MIN_SIF = 0.2
MAX_SIF = 1.7
BATCH_SIZE = 1
assert(BATCH_SIZE == 1)


def eval_model(models, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    for crop, model in models.items():
        model.eval()  # Set model to evaluate mode

    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    results = np.zeros((dataset_size, len(COLUMN_NAMES)))
    j = 0

    # Iterate over data.
    for sample in dataloader:
        for i in range(len(sample['SIF'])):
            input_tile_standardized = sample['subtile'][i].to(device)
            true_sif_non_standardized = sample['SIF'][i].to(device)

            with torch.set_grad_enabled(False):
                # Get list of sub-tiles for each crop type
                subtiles_by_crop, num_subtiles = get_subtiles_list_by_crop(input_tile_standardized, MODEL_SUBTILE_DIM, device,
                                                                           CROP_TYPE_INDICES, PURE_THRESHOLD, MAX_SUBTILE_CLOUD_COVER)
                predicted_subtile_sifs = torch.empty((num_subtiles))
                subtile_idx = 0

                # Loop through all crop types. For each crop type, pass their sub-tiles through model to predict sub-tile SIF
                for crop, subtiles in subtiles_by_crop.items():
                    num_subtiles_this_crop = subtiles.shape[0]
                    subtiles = subtiles[:, FEATURE_BANDS, :, :]
                    predicted_subtile_sifs[subtile_idx:subtile_idx+num_subtiles_this_crop] = models[crop](subtiles).flatten()
                    subtile_idx += num_subtiles_this_crop

                # Take the average predicted SIF over all sub-tiles
                predicted_sif_standardized = torch.mean(predicted_subtile_sifs)
                predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)
                #print('Predicted', predicted_sif_non_standardized, 'True', true_sif_non_standardized)

            # statistics
            band_means = torch.mean(input_tile_standardized, dim=(1, 2))
            results[j, 0] = true_sif_non_standardized.item()
            results[j, 1] = predicted_sif_non_standardized.item()
            results[j, 2] = sample['lon'][i].item()
            results[j, 3] = sample['lat'][i].item()
            results[j, 4:] = band_means.cpu().numpy()
            j += 1
 
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
eval_metadata = pd.read_csv(OCO2_EVAL_FILE)
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

# Constrain predicted SIF to be between 0.2 and 1.7 (unstandardized)
# Don't forget to standardize
if MIN_SIF is not None and MAX_SIF is not None:
    min_output = (MIN_SIF - sif_mean) / sif_std
    max_output = (MAX_SIF - sif_mean) / sif_std
else:
    min_output = None
    max_output = None

# Set up image transforms
transform_list = []
# transform_list.append(tile_transforms.ShrinkTile())
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
if RESIZE:
    transform_list.append(tile_transforms.ResizeTile(target_dim=RESIZED_DIM, discrete_bands=DISCRETE_BANDS))
transform = transforms.Compose(transform_list)

# Set up Dataset and Dataloader
dataset_size = len(eval_metadata)
# dataset = ReflectanceCoverSIFDataset(eval_metadata, transform)
dataset = EvalSubtileDataset(eval_metadata, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

# Load trained models from file
models = dict()
for crop in [-1] + CROP_TYPE_INDICES:
    if MODEL_TYPE == 'resnet18':
        model = small_resnet.resnet18(input_channels=INPUT_CHANNELS).to(device)
    elif MODEL_TYPE == 'simple_cnn':
        model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    elif MODEL_TYPE == 'simple_cnn_small':
        model = simple_cnn.SimpleCNNSmall(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    else:
        print('Model type not supported')
        exit(1)
    model.load_state_dict(torch.load(TRAINED_MODEL_FILE + "_crop_" + str(crop), map_location=device))
    models[crop] = model

criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
results_numpy = eval_model(models, dataloader, dataset_size, criterion, device, sif_mean, sif_std)

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
plt.title('True vs predicted SIF (OCO2):' + METHOD)
plt.xlim(left=0, right=1.5)
plt.ylim(bottom=0, top=1.5)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '.png')
plt.close()

PURE_THRESHOLD = 0.7

fig, axeslist = plt.subplots(ncols=3, nrows=10, figsize=(15, 50))
fig.suptitle('True vs predicted SIF (OCO2): ' + METHOD)
for idx, crop_type in enumerate(CROP_TYPES):
    crop_rows = results_df.loc[results_df[crop_type] > PURE_THRESHOLD]
    print(len(crop_rows), 'subtiles that are majority', crop_type)

    # Scatter plot of true vs predicted
    axeslist.ravel()[idx].scatter(crop_rows['true'], crop_rows['predicted'])
    axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    axeslist.ravel()[idx].set_xlim(left=0, right=1.5)
    axeslist.ravel()[idx].set_ylim(bottom=0, top=1.5)
    axeslist.ravel()[idx].set_title(crop_type)

plt.tight_layout()
fig.subplots_adjust(top=0.96)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()

