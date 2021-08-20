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
from eval_subtile_dataset import EvalOCO2Dataset
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
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01") #"dataset_2018-07-16") #7-16") #07-16") #07-16")
OCO2_EVAL_FILE = os.path.join(DATASET_DIR, "oco2_eval_subtiles.csv")
TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_subtile_simple_cnn")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
METHOD = '4a_subtile_simple_cnn'  #'3_small_tile_simple' # '2_large_tile_resnet18'
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_OCO2_eval_subtile_' + METHOD 

COLUMN_NAMES = ['true', 'predicted',
                    'center_lon', 'center_lat',
                    'lon_0', 'lat_0', 'lon_1', 'lat_1', 'lon_2', 'lat_2', 'lon_3', 'lat_3',
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
RESULTS_CSV_FILE = os.path.join(DATASET_DIR, 'OCO2_results_' + METHOD + '.csv')

INPUT_CHANNELS = 43
REDUCED_CHANNELS = 43
DISCRETE_BANDS = list(range(12, 43))
COVER_INDICES = list(range(12, 42))
RESIZE = False
MIN_SIF = 0.2
MAX_SIF = 1.7
BATCH_SIZE = 1
assert(BATCH_SIZE == 1)

def eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    model.eval()   # Set model to evaluate mode
    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    results = np.zeros((dataset_size, len(COLUMN_NAMES)))
    j = 0

    # Iterate over data.
    for sample in dataloader:
        for i in range(len(sample['SIF'])):
            input_tile_standardized = sample['subtiles'][i].to(device)
            print('Input tile shape', input_tile_standardized.shape)
            #print('=========================')
            #print('Input band means')
            #print(torch.mean(input_tile_standardized, dim=(2,3)))
            true_sif_non_standardized = sample['SIF'][i].to(device)

            # forward
            # track history if only in train
            # with torch.set_grad_enabled(False):
            predicted_sif_standardized = torch.mean(model(input_tile_standardized))
            predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
            loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)
            print('Predicted', predicted_sif_non_standardized, 'True', true_sif_non_standardized)
            # statistics
            band_means = torch.mean(input_tile_standardized, dim=(0,2,3))
            results[j, 0] = true_sif_non_standardized.item()
            results[j, 1] = predicted_sif_non_standardized.item()
            results[j, 2] = sample['center_lon'][i].item()
            results[j, 3] = sample['center_lat'][i].item()
            results[j, 4] = sample['lon_0'][i].item()
            results[j, 5] = sample['lat_0'][i].item()
            results[j, 6] = sample['lon_1'][i].item()
            results[j, 7] = sample['lat_1'][i].item()
            results[j, 8] = sample['lon_2'][i].item()
            results[j, 9] = sample['lat_2'][i].item()
            results[j, 10] = sample['lon_3'][i].item()
            results[j, 11] = sample['lat_3'][i].item()
            results[j, 12:] = band_means.cpu().numpy()
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
dataset = EvalOCO2Dataset(eval_metadata, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

# Load trained model from file
resnet_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)  
#resnet_model = resnet.resnet18(input_channels=INPUT_CHANNELS).to(device)
# resnet_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=1).to(device)
resnet_model.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=device))

criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
results_numpy = eval_model(resnet_model, dataloader, dataset_size, criterion, device, sif_mean, sif_std)

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

fig, axeslist = plt.subplots(ncols=3, nrows=10, figsize=(15, 50))
fig.suptitle('True vs predicted SIF (CFIS): ' + METHOD)
for idx, crop_type in enumerate(CROP_TYPES):
    crop_rows = results_df.loc[results_df[crop_type] > 0.5]
    print(len(crop_rows), 'subtiles that are majority', crop_type)

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

