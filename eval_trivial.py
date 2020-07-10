"""
Evaluate a "subtile -> SIF" model
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
from eval_subtile_dataset import EvalSubtileDataset
from sif_utils import print_stats
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
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset") #"dataset_2018-07-16")
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-08-01") #"dataset_2016-07-16")
EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv") 
TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn_v5")
# TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/small_tile_simple_v2")
#TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/cfis_sif_aug")
#RAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/subtile_sif_simple_cnn_aug") #cfis_sif")
#TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/small_tile_simple") #large_tile_resnet18")  #test_large_tile_simple")  # small_tile_sif_prediction")
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
METHOD = '4a_subtile_simple_cnn_new_data' # '2_large_tile_resnet' # '3_small_tile_simple' #'0_cfis_cheating' # #'3_small_tile_simple' # '2_large_tile_resnet18'
MODEL_TYPE = 'simple_cnn_small_v5'
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_cfis_eval_subtile_' + METHOD 

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
CROP_TYPES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest']

RESULTS_CSV_FILE = os.path.join(EVAL_DATASET_DIR, 'results_' + METHOD + '.csv')

BANDS = list(range(0, 43))
# BANDS =  list(range(0, 12)) + list(range(12, 27)) + [28] + [42]
INPUT_CHANNELS = len(BANDS)
REDUCED_CHANNELS = 15
RESIZE = False
RESIZED_DIM = [371, 371]
DISCRETE_BANDS = list(range(12, 43))
COVER_INDICES = list(range(12, 42))
PURE_THRESHOLD = 0.7
CROP_TYPE_START_IDX = 12
MIN_SIF = 0
MAX_SIF = 1.7
MIN_INPUT = -2
MAX_INPUT = 2

def eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    model.eval()   # Set model to evaluate mode
    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    results = np.zeros((dataset_size, len(COLUMN_NAMES)))
    file_names = []
    j = 0

    # Iterate over data.
    for sample in dataloader:
        input_tile_standardized = sample['tile'].to(device)
        # print('=========================')
        # print('Input band means')
        # print(torch.mean(input_tile_standardized, dim=(2,3)))
        print('Tile shape', input_tile_standardized.shape)
        true_sif_non_standardized = sample['SIF'].to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            predicted_sif_standardized = model(input_tile_standardized[:, BANDS, :, :]).flatten()
            predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
            loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)

        # statistics
        batch_size = len(sample['SIF'])
        band_means = torch.mean(input_tile_standardized, dim=(2,3))
        results[j:j+batch_size, 0] = true_sif_non_standardized.cpu().numpy()
        results[j:j+batch_size, 1] = predicted_sif_non_standardized.cpu().numpy()
        results[j:j+batch_size, 2] = sample['lon']
        results[j:j+batch_size, 3] = sample['lat']
        results[j:j+batch_size, 4:] = band_means.cpu().numpy()
        file_names.extend(sample['tile_file'])
        j += batch_size
        #if j > 50:
        #    break
 
    return results, file_names


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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=True, num_workers=4)

# Load trained model from file
if MODEL_TYPE == 'simple_cnn_small_v2':
    model = simple_cnn.SimpleCNNSmall2(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1, crop_type_start_idx=CROP_TYPE_START_IDX, min_output=min_output, max_output=max_output).to(device)  
elif MODEL_TYPE == 'simple_cnn_small_v5':
    model = simple_cnn.SimpleCNNSmall5(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)  
elif MODEL_TYPE == 'resnet18':
    model = resnet.resnet18(input_channels=INPUT_CHANNELS).to(device)
else:
    print('Unsupported model type', MODEL_TYPE)
    exit(1)

# model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=1).to(device)
model.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=device))

criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
results_numpy, file_names = eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std)

results_df = pd.DataFrame(results_numpy, columns=COLUMN_NAMES)
results_df['tile_file'] = file_names
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

fig, axeslist = plt.subplots(ncols=3, nrows=10, figsize=(15, 50))
fig.suptitle('True vs predicted SIF (CFIS): ' + METHOD)
for idx, crop_type in enumerate(CROP_TYPES):
    crop_rows = results_df.loc[results_df[crop_type] > PURE_THRESHOLD]
    print(len(crop_rows), 'subtiles that are majority', crop_type)

    print('======================== Crop type', crop_type, '==========================')
    print_stats(crop_rows['true'].to_numpy(), crop_rows['predicted'].to_numpy(), sif_mean)

    # Scatter plot of true vs predicted
    axeslist.ravel()[idx].scatter(crop_rows['true'], crop_rows['predicted'])
    axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    axeslist.ravel()[idx].set_xlim(left=0, right=2)
    axeslist.ravel()[idx].set_ylim(bottom=0, top=2)
    axeslist.ravel()[idx].set_title(crop_type)

plt.tight_layout()
fig.subplots_adjust(top=0.96)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()

