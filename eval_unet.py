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
from embedding_to_sif_nonlinear_model import EmbeddingToSIFNonlinearModel
from eval_subtile_dataset import EvalSubtileDataset #EvalOCO2Dataset

from sif_utils import print_stats, get_subtiles_list
import sif_utils
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
from unet.unet_model import UNet, UNetSmall, UNet2


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_all_2")
EVAL_FILE = os.path.join(DATASET_DIR, "standardized_tiles_test.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
METHOD = "7_unet2_clip_-6_8_batchnorm_dimred"
# METHOD = "7_unet2_reflectance_only_aug"
MODEL_TYPE = "unet2"
TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)
print('trained model file', os.path.basename(TRAINED_MODEL_FILE))
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_OCO2_' + METHOD 
RESULTS_CSV_FILE = os.path.join(DATASET_DIR, 'OCO2_results_' + METHOD + '.csv')

COLUMN_NAMES = ['true', 'predicted',
                    'lon', 'lat', 'source', 'date',
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
CROP_TYPES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest']
BANDS = list(range(0, 43))
# BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42]  #list(range(0, 43))
DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
         "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
         "2018-09-16"]
SOURCES = ["OCO2"]

INPUT_CHANNELS = len(BANDS)
REDUCED_CHANNELS = 15
DISCRETE_BANDS = list(range(12, 43))
COVER_INDICES = list(range(12, 42))
MISSING_REFLECTANCE_IDX = -1
RESIZE = False
MIN_SIF = None #0
MAX_SIF = None #1.7
MIN_INPUT = -6
MAX_INPUT = 8
BATCH_SIZE = 16
NUM_WORKERS = 8
PURE_THRESHOLD = 0.6
MIN_SOUNDINGS = 5

# assert(BATCH_SIZE == 1)


def eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    model.train()   # Set model to evaluate mode
    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    results = [] #np.zeros((dataset_size, len(COLUMN_NAMES)))
    j = 0

    # Iterate over data.
    for sample in dataloader:
        input_tiles_standardized = sample['tile'].to(device)
        true_sifs_non_standardized = sample['SIF'].to(device)

        # Binary mask for non-cloudy pixels. Since the tiles are passed through the
        # StandardizeTile transform, 1 now represents non-cloudy (data present) and 0 represents
        # cloudy (data missing).
        non_cloudy_pixels = input_tiles_standardized[:, MISSING_REFLECTANCE_IDX, :, :]  # (batch size, H, W)

        # print('Input tile shape', input_tiles_standardized.shape)
        #print('=========================')
        #print('Input band means')
        #print(torch.mean(input_tiles_standardized, dim=(2,3)))

        # Pass sub-tiles through network
        with torch.set_grad_enabled(False):
            predictions = model(input_tiles_standardized[:, BANDS, :, :]) # predictions: (batch size, 1, H, W)
            predictions = torch.squeeze(predictions, dim=1)  # (batch size, H, W)
            predicted_sifs_standardized = sif_utils.masked_average(predictions, non_cloudy_pixels, dims_to_average=(1, 2)) # (batch size)
            predicted_sifs_non_standardized = torch.tensor(predicted_sifs_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
            loss = criterion(predicted_sifs_non_standardized, true_sifs_non_standardized)

        # statistics
        band_means = torch.mean(input_tiles_standardized, dim=(2, 3))
        # print('band means shape', band_means.shape)
        for i in range(true_sifs_non_standardized.shape[0]):
            row = [true_sifs_non_standardized[i].item(), predicted_sifs_non_standardized[i].item(), sample['lon'][i].item(),
                    sample['lat'][i].item(), sample['source'][i], sample['date'][i]] + band_means[i].cpu().tolist()
            results.append(row)
            j += 1
        print(j)
 
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

# Only look at OCO-2 this time
eval_metadata = eval_metadata[eval_metadata['source'] == 'OCO2'] 
print("Eval samples:", len(eval_metadata))

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
#transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
transform_list.append(tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT))
if RESIZE:
    transform_list.append(tile_transforms.ResizeTile(target_dim=RESIZED_DIM, discrete_bands=DISCRETE_BANDS))
transform = transforms.Compose(transform_list)

# Set up Dataset and Dataloader
dataset_size = len(eval_metadata)
dataset = ReflectanceCoverSIFDataset(eval_metadata, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS)

# Load trained model from file
if MODEL_TYPE == 'unet_small':
    model = UNetSmall(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'unet2':
    model = UNet2(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)

else:
    print('Model type not supported')
    exit(1)
model.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=device))

criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
results = eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std)
results_df = pd.DataFrame(results, columns=COLUMN_NAMES)
print("Result example", results_df.head())
results_df.to_csv(RESULTS_CSV_FILE)

true = results_df['true'].tolist()
predicted = results_df['predicted'].tolist()
print('========== before clipping =========')
print_stats(true, predicted, sif_mean)

# Clip to be between 0.2 and 1.7
predicted = np.clip(predicted, a_min=0.2, a_max=None)

# Print statistics
print('========== after clipping =========')
print_stats(true, predicted, sif_mean, ax=plt.gca())

# Scatter plot of true vs predicted (ALL POINTS)
plt.title('True vs predicted SIF (OCO2):' + METHOD)
plt.xlim(left=0, right=1.7)
plt.ylim(bottom=0, top=1.7)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '.png')
plt.close()

# Plot scatterplots by source/date
fig, axeslist = plt.subplots(ncols=len(SOURCES), nrows=len(DATES), figsize=(6*len(SOURCES), 6*len(DATES)))
fig.suptitle('True vs predicted SIF, by date/source: ' + METHOD)

# Statistics by date and source
idx = 0
for date in DATES:
    for source in SOURCES:
        print('=================== Date ' + date + ', ' + source + ' ======================')
        rows = results_df.loc[(results_df['date'] == date) & (results_df['source'] == source)]
        print('Number of rows', len(rows))
        if len(rows) < 2:
            idx += 1
            continue

        # Scatter plot of true vs predicted
        ax = axeslist.ravel()[idx]
        print_stats(rows['true'].to_numpy(), rows['predicted'].to_numpy(), sif_mean, ax=ax)
        ax.set_xlim(left=0, right=1.7)
        ax.set_ylim(bottom=0, top=1.7)
        ax.set_title(date + ', ' + source)
        idx += 1

plt.tight_layout()
fig.subplots_adjust(top=0.96)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '_dates_and_sources.png')
plt.close()


# # Plot scatterplot by source
# fig, axeslist = plt.subplots(ncols=len(SOURCES), nrows=1, figsize=(12, 6))
# fig.suptitle('True vs predicted SIF, by source: ' + METHOD)

# for idx, source in enumerate(SOURCES):
#     print('=================== All dates: ' + source + ' ======================')
#     rows = results_df.loc[results_df['source'] == source]
#     if len(rows) < 2:
#         continue
#     print_stats(rows['true'].to_numpy(), rows['predicted'].to_numpy(), sif_mean)

#     # Scatter plot of true vs predicted
#     axeslist.ravel()[idx].scatter(rows['true'].to_numpy(), rows['predicted'].to_numpy())
#     axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
#     axeslist.ravel()[idx].set_xlim(left=MIN_SIF, right=MAX_SIF)
#     axeslist.ravel()[idx].set_ylim(bottom=MIN_SIF, top=MAX_SIF)
#     axeslist.ravel()[idx].set_title(source)

# plt.tight_layout()
# fig.subplots_adjust(top=0.92)
# plt.savefig(TRUE_VS_PREDICTED_PLOT + '_sources.png')
# plt.close()

fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('True vs predicted SIF (OCO2) by crop type: ' + METHOD)
for idx, crop_type in enumerate(CROP_TYPES):
    crop_rows = results_df.loc[(results_df[crop_type] > PURE_THRESHOLD) & (results_df['source'] == 'OCO2')]
    print(len(crop_rows), 'subtiles that are majority', crop_type)
    if len(crop_rows) < 2:
        continue
    print('======================== Crop type', crop_type, '==========================')
    ax = axeslist.ravel()[idx]
    print_stats(crop_rows['true'].to_numpy(), crop_rows['predicted'].to_numpy(), sif_mean, ax=ax)
    ax.set_xlim(left=0, right=1.7)
    ax.set_ylim(bottom=0, top=1.7)
    ax.set_title(crop_type)


plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()

