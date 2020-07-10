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
# DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_tropomi_train")
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset")
EVAL_FILE = os.path.join(DATASET_DIR, "tile_info_val.csv")
# EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-08-01")
# EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv") 

# TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/avg_embedding_to_sif")
# TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn_new_data_tiny")
TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn_v5")
# TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_pretrained_simple_cnn_small_v2")
print('trained model file', os.path.basename(TRAINED_MODEL_FILE))

#DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01") #"dataset_2018-07-16") #7-16") #07-16") #07-16")
#OCO2_EVAL_FILE = os.path.join(DATASET_DIR, "oco2_eval_subtiles.csv")
#TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn_16crop_embedding")
#TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/avg_embedding_to_sif")

BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
METHOD = '4a_subtile_simple_cnn_small_v5' #'4a_subtile_simple_cnn_small_v2_pretrained'
MODEL_TYPE = 'simple_cnn_small_v5'
# METHOD = '4a_subtile_simple_cnn_small_v5' # oco2_only'  #'3_small_tile_simple' # '2_large_tile_resnet18'
# MODEL_TYPE = 'simple_cnn_small_v5'
# MODEL_TYPE = 'simple_cnn_small_3'
# METHOD = '4b_avg_embedding'
# MODEL_TYPE = 'avg_embedding'
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_new_data_OCO2_' + METHOD 
MODEL_SUBTILE_DIM = 10
MAX_SUBTILE_CLOUD_COVER = 0.2
CROP_TYPE_START_IDX = 12
COLUMN_NAMES = ['true', 'predicted', 'tile_file',
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
RESULTS_CSV_FILE = os.path.join(DATASET_DIR, 'OCO2_results_' + METHOD + '.csv')

# BANDS = list(range(0, 9)) + [42]
# BANDS =  list(range(0, 12)) + list(range(12, 27)) + [28] + [42]
# BANDS = list(range(0, 12)) + [12, 13, 14, 16] + [42]
BANDS = list(range(0, 43))
DATES = ["2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02"]
SOURCES = ["TROPOMI", "OCO2"]

INPUT_CHANNELS = len(BANDS)
REDUCED_CHANNELS = 15
DISCRETE_BANDS = list(range(12, 43))
COVER_INDICES = list(range(12, 42))
RESIZE = False
MIN_SIF = 0
MAX_SIF = 1.7
MIN_INPUT = -2
MAX_INPUT = 2
BATCH_SIZE = 1
PURE_THRESHOLD = 0.6
# MIN_SOUNDINGS = 5

assert(BATCH_SIZE == 1)


def eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    model.eval()   # Set model to evaluate mode
    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    results = [] #np.zeros((dataset_size, len(COLUMN_NAMES)))
    j = 0

    # Iterate over data.
    for sample in dataloader:
        for i in range(len(sample['SIF'])):
            input_tile_standardized = sample['tile'][i].to(device)
            true_sif_non_standardized = sample['SIF'][i].to(device)

            #print('Input tile shape', input_tile_standardized.shape)
            #print('=========================')
            # print('Input band means')
            # print(torch.mean(input_tile_standardized, dim=(1, 2)))

            # Pass sub-tiles through network
            with torch.set_grad_enabled(False):
                subtiles = get_subtiles_list(input_tile_standardized[BANDS, :, :], MODEL_SUBTILE_DIM, device, MAX_SUBTILE_CLOUD_COVER)
                # print('subtiles shape', subtiles.shape)
                if MODEL_TYPE == 'avg_embedding':
                    subtile_averages = torch.mean(subtiles, dim=(2, 3))
                    print('Subtile averages shape', subtile_averages.shape)
                    predicted_subtile_sifs = model(subtile_averages)
                else:
                    predicted_subtile_sifs = model(subtiles)
                predicted_sif_standardized = torch.mean(predicted_subtile_sifs)
                predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)
                print('Predicted', predicted_sif_non_standardized, 'True', true_sif_non_standardized)

            # statistics
            band_means = torch.mean(input_tile_standardized, dim=(1, 2))
            row = [true_sif_non_standardized.item(), predicted_sif_non_standardized.item(), sample['tile_file'][i], sample['lon'][i].item(), 
                   sample['lat'][i].item(), sample['source'][i], sample['date'][i]] + band_means.cpu().tolist()
            results.append(row)
            # results[j, 0] = true_sif_non_standardized.item()
            # results[j, 1] = predicted_sif_non_standardized.item()
            # results[j, 2] = sample['lon'][i].item()
            # results[j, 3] = sample['lat'][i].item()
            # results[j, 4] = sample['source'][i]
            # results[j, 5] = sample['date'][i]    
            # results[j, 6:] = band_means.cpu().numpy()
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
eval_metadata = pd.read_csv(EVAL_FILE)
# eval_metadata = eval_metadata.loc[eval_metadata['source'] == 'OCO2']
print("Eval samples:", len(eval_metadata))

# eval_metadata = eval_metadata.loc[eval_metadata['num_soundings'] >= MIN_SOUNDINGS]
# eval_metadata = eval_metadata.loc[eval_metadata['SIF'] >= 0.2]
# print("Eval samples with more than", MIN_SOUNDINGS, "soundings:", len(eval_metadata))

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
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds, min_input=MIN_INPUT, max_input=MAX_INPUT))
if RESIZE:
    transform_list.append(tile_transforms.ResizeTile(target_dim=RESIZED_DIM, discrete_bands=DISCRETE_BANDS))
transform = transforms.Compose(transform_list)

# Set up Dataset and Dataloader
dataset_size = len(eval_metadata)
dataset = ReflectanceCoverSIFDataset(eval_metadata, transform)
# dataset = EvalSubtileDataset(eval_metadata, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

# Load trained model from file
if MODEL_TYPE == 'resnet18':
    model = small_resnet.resnet18(input_channels=INPUT_CHANNELS).to(device)
elif MODEL_TYPE == 'simple_cnn':
    model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small':
    model = simple_cnn.SimpleCNNSmall(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, crop_type_start_idx=CROP_TYPE_START_IDX, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small_v2':
    model = simple_cnn.SimpleCNNSmall2(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, crop_type_start_idx=CROP_TYPE_START_IDX, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small_3':
    model = simple_cnn.SimpleCNNSmall3(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small_v4':
    model = simple_cnn.SimpleCNNSmall4(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'simple_cnn_small_v5':
    model = simple_cnn.SimpleCNNSmall5(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'avg_embedding':
    model = EmbeddingToSIFNonlinearModel(embedding_size=INPUT_CHANNELS, hidden_size=HIDDEN_DIM, min_output=min_output, max_output=max_output).to(device)

else:
    print('Model type not supported')
    exit(1)

#resnet_model = resnet.resnet18(input_channels=INPUT_CHANNELS).to(device)
# resnet_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=1).to(device)
model.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=device))

criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
# results = eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std)
# results_df = pd.DataFrame(results, columns=COLUMN_NAMES)
# print("Result example", results_df.head())
# results_df.to_csv(RESULTS_CSV_FILE)
results_df = pd.read_csv(RESULTS_CSV_FILE)

true = results_df['true'].tolist()
predicted = results_df['predicted'].tolist()

# Print statistics
print_stats(true, predicted, sif_mean)

# Scatter plot of true vs predicted (ALL POINTS)
plt.scatter(true, predicted)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('True vs predicted SIF (OCO2):' + METHOD)
plt.xlim(left=0, right=1.5)
plt.ylim(bottom=0, top=1.5)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '.png')
plt.close()

# Plot scatterplots by source/date
fig, axeslist = plt.subplots(ncols=len(SOURCES), nrows=len(DATES), figsize=(12, 30))
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

        print_stats(rows['true'].to_numpy(), rows['predicted'].to_numpy(), sif_mean)

        # Scatter plot of true vs predicted
        axeslist.ravel()[idx].scatter(rows['true'], rows['predicted'])
        axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
        axeslist.ravel()[idx].set_xlim(left=0, right=1.5)
        axeslist.ravel()[idx].set_ylim(bottom=0, top=1.5)
        axeslist.ravel()[idx].set_title(date + ', ' + source)
        idx += 1

plt.tight_layout()
fig.subplots_adjust(top=0.96)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '_dates_and_sources.png')
plt.close()


# Plot scatterplot by source
fig, axeslist = plt.subplots(ncols=len(SOURCES), nrows=1, figsize=(12, 6))
fig.suptitle('True vs predicted SIF, by source: ' + METHOD)

for idx, source in enumerate(SOURCES):
    print('=================== All dates: ' + source + ' ======================')
    rows = results_df.loc[results_df['source'] == source]
    if len(rows) < 2:
        continue
    print_stats(rows['true'].to_numpy(), rows['predicted'].to_numpy(), sif_mean)

    # Scatter plot of true vs predicted
    axeslist.ravel()[idx].scatter(rows['true'].to_numpy(), rows['predicted'].to_numpy())
    axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    axeslist.ravel()[idx].set_xlim(left=0, right=1.5)
    axeslist.ravel()[idx].set_ylim(bottom=0, top=1.5)
    axeslist.ravel()[idx].set_title(source)

plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '_sources.png')
plt.close()

fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
fig.suptitle('True vs predicted SIF (OCO2) by crop type: ' + METHOD)
for idx, crop_type in enumerate(CROP_TYPES):
    crop_rows = results_df.loc[(results_df[crop_type] > PURE_THRESHOLD) & (results_df['source'] == 'OCO2')]
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

