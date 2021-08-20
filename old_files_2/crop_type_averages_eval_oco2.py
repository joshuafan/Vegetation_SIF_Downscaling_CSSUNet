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
#from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
#from eval_subtile_dataset import EvalSubtileDataset #EvalOCO2Dataset
from crop_type_averages_dataset import CropTypeAveragesFromTileDataset
from embedding_to_sif_nonlinear_model import EmbeddingToSIFNonlinearModel

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
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
OCO2_EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "oco2_eval_subtiles.csv")
TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/crop_type_averages_nn")

BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
METHOD = '6_crop_type_averages_nn'  #'3_small_tile_simple' # '2_large_tile_resnet18'
MODEL_TYPE = 'embedding_to_sif_nonlinear'
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_OCO2_eval_subtile_' + METHOD 


CROP_TYPES = {'grassland_pasture': 12,
              'corn': 13,
              'soybean': 14,
              'shrubland': 15,
              'deciduous_forest': 16,
              'evergreen_forest': 17,
              'spring_wheat': 18,
              'developed_open_space': 19,
              'other_hay_non_alfalfa': 20,
              'winter_wheat': 21,
              'herbaceous_wetlands': 22,
              'woody_wetlands': 23,
              'open_water': 24,
              'alfalfa': 25,
              'fallow_idle_cropland': 26,
              'sorghum': 27,
              'developed_low_intensity': 28,
              'barren': 29,
              'durum_wheat': 30,
              'canola': 31,
              'sunflower': 32,
              'dry_beans': 33,
              'developed_med_intensity': 34,
              'millet': 35,
              'sugarbeets': 36,
              'oats': 37,
              'mixed_forest': 38,
              'peas': 39,
              'barley': 40,
              'lentils': 41}

CROP_TYPES_TO_PLOT = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest']

FEATURES = {'ref_1': 0,
            'ref_2': 1,
            'ref_3': 2,
            'ref_4': 3,
            'ref_5': 4,
            'ref_6': 5,
            'ref_7': 6,
            'ref_10': 7,
            'ref_11': 8,
            'Rainf_f_tavg': 9,
            'SWdown_f_tavg': 10,
            'Tair_f_tavg': 11}

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
# CROP_TYPES = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
#                     'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
#                     'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
#                     'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
#                     'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
#                     'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
#                     'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
#                     'lentils']
RESULTS_CSV_FILE = os.path.join(EVAL_DATASET_DIR, 'OCO2_results_' + METHOD + '.csv')

# CROP_TYPE_INDICES = list(range(12, 42))
# BANDS = list(range(0, 12)) + CROP_TYPES + [42] # Don't include crop types?
# CROP_TYPE_INDICES = [12, 13, 14, 16]
# FEATURE_BANDS = list(range(0, 12)) + [42]
# INPUT_CHANNELS = len(FEATURE_BANDS)
# REDUCED_CHANNELS = 8
INPUT_DIM = len(FEATURES)
HIDDEN_DIM = 64
PURE_THRESHOLD = 0.6
RESIZE = False
MIN_SIF = 0.
MAX_SIF = 1.7
BATCH_SIZE = 1
MIN_SOUNDINGS = 5
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
        # Read batch from dataloader
        batch_size = len(sample['SIF'])
        crop_to_features = sample['features']
        area_fractions = sample['cover_fractions']

        # For each large tile, predict SIF by passing each of its sub-tiles through the model
        predicted_sifs_standardized = torch.zeros((batch_size)).to(device)
        with torch.set_grad_enabled(False):
            for i in range(batch_size):
                input_tile_standardized = sample['tile'][i].to(device)
                true_sif_non_standardized = sample['SIF'][i].to(device)

                predicted_sif_standardized = 0
                for crop, crop_features in crop_to_features.items():
                    crop_area_fraction = area_fractions[crop][i]
                    crop_model = models[crop]
                    predicted_crop_sif = crop_model(crop_features[i])
                    predicted_sif_standardized += (predicted_crop_sif * crop_area_fraction)
                    #if crop_area_fraction > 0.01:
                    #    print('Predicted SIF for ', crop, '(fraction:', crop_area_fraction.item(), '):', (predicted_crop_sif * sif_std + sif_mean).item())

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
eval_metadata = eval_metadata.loc[eval_metadata['num_soundings'] >= MIN_SOUNDINGS]
eval_metadata = eval_metadata.loc[eval_metadata['SIF'] >= 0.2]
print("Eval samples with more than", MIN_SOUNDINGS, "soundings:", len(eval_metadata))

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
dataset = CropTypeAveragesFromTileDataset(eval_metadata, CROP_TYPES, FEATURES, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

# Load trained models from file
models = dict()
for crop in list(CROP_TYPES.keys()) + ['other']:
    if MODEL_TYPE == 'embedding_to_sif_nonlinear':
        model = EmbeddingToSIFNonlinearModel(embedding_size=INPUT_DIM, hidden_size=HIDDEN_DIM, min_output=min_output, max_output=max_output).to(device)
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

fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('True vs predicted SIF (OCO2): ' + METHOD)
for idx, crop_type in enumerate(CROP_TYPES_TO_PLOT):
    crop_rows = results_df.loc[results_df[crop_type] > PURE_THRESHOLD]
    print(len(crop_rows), 'subtiles that are majority', crop_type)
    if len(crop_rows) < 2:
        continue
    print('======================== Crop type', crop_type, '==========================')
    print_stats(crop_rows['true'].to_numpy(), crop_rows['predicted'].to_numpy(), sif_mean)

    # Scatter plot of true vs predicted
    axeslist.ravel()[idx].scatter(crop_rows['true'], crop_rows['predicted'])
    axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    axeslist.ravel()[idx].set_xlim(left=0, right=1.5)
    axeslist.ravel()[idx].set_ylim(bottom=0, top=1.5)
    axeslist.ravel()[idx].set_title(crop_type)

plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()

