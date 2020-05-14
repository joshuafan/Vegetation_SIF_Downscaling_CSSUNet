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
from sif_utils import get_top_bound, get_left_bound, lat_long_to_index, print_stats
from SAN import SAN

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-07-16")
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-07-16")
EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv")
# EVAL_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_info_val.csv")

TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/SAN_feat111")  # small_tile_sif_prediction")
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
METHOD = "5_SAN" #"tile2vec_finetuned"
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_eval_subtile_' + METHOD

COLUMN_NAMES = ['true', 'predicted',
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']
RESULTS_CSV_FILE = os.path.join(EVAL_DATASET_DIR, 'results_' + METHOD + '.csv')

TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_eval_subtile_' + METHOD
PLOT_TITLE = 'Structured Attention Network'
INPUT_CHANNELS = 43
BATCH_SIZE = 4
eval_points = pd.read_csv(EVAL_FILE)
TILE_SIZE_DEGREES = 0.1
INPUT_SIZE = 371
SUBTILE_DIM = 10
OUTPUT_SIZE = int(INPUT_SIZE / SUBTILE_DIM)
COVER_INDICES = list(range(12, 42))
MIN_SIF = 0.2
MAX_SIF = 1.7


# How many degrees each SUBTILE is
RES = (TILE_SIZE_DEGREES / OUTPUT_SIZE, TILE_SIZE_DEGREES / OUTPUT_SIZE)

def eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std):
    model.eval()   # Set model to evaluate mode
    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    results = np.empty((dataset_size, 2+len(COVER_INDICES)))
    j = 0

    # Iterate over data.
    for sample in dataloader:
        # Get surrounding large tile 
        input_large_tiles = sample['large_tile'].to(device)
        subtiles = sample['subtile'].to(device)
        #print('=========================')
        #print('Input band means')
        #print(torch.mean(input_tile, dim=(2,3)))
        true_sifs = sample['SIF'].to(device)

        # forward
        # track history if only in train
        # with torch.set_grad_enabled(False):
        # Obtain model predictions for each subtile
        _, _, _, _, predictions = model(input_large_tiles)
        #print('Prdictions shape', predictions.shape)

        # Loop through each subtile in the batch to obtain the correct prediction
        for i in range(sample['SIF'].shape[0]):
            subtile_lat = sample['lat'][i]
            subtile_lon = sample['lon'][i]

            # TODO Batch needs to be 1?
            top_bound = get_top_bound(subtile_lat)
            left_bound = get_left_bound(subtile_lon)
            subtile_height_idx, subtile_width_idx = lat_long_to_index(subtile_lat, subtile_lon, top_bound, left_bound, RES)
            predicted_sif_standardized = predictions[i, 0, subtile_height_idx, subtile_width_idx]
            predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
            #predicted_sif_non_standardized = torch.clamp(predicted_sif_non_standardized, min=0.2, max=1.7)
            band_means = torch.mean(subtiles[i:i+1, COVER_INDICES, :, :], dim=(2, 3))
            #print('Band means shape', band_means.shape)
            results[j, 0] = true_sifs[i].cpu().item() #numpy()
            results[j, 1] = predicted_sif_non_standardized.cpu().item() #numpy()
            results[j, 2:] = band_means.cpu().numpy()
            #print('Results j', results[j])
            j += 1
        print(j)
        #if j > 1000: #10:
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
# transform_list.append(tile_transforms.ShrinkTile())
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
transform = transforms.Compose(transform_list)

# Set up Dataset and Dataloader
dataset_size = len(eval_metadata)
#dataset = ReflectanceCoverSIFDataset(eval_metadata, transform)
dataset = EvalSubtileDataset(eval_metadata, transform, load_large_tile=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

# Constrain predicted SIF to be between 0.2 and 1.7 (unstandardized)
# Don't forget to standardize
min_output = (MIN_SIF - sif_mean) / sif_std
max_output = (MAX_SIF - sif_mean) / sif_std



# Load trained model from file
# resnet_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=1)  #.to(device)
resnet_model = resnet.resnet18(input_channels=INPUT_CHANNELS) 
model = SAN(resnet_model, input_height=INPUT_SIZE, input_width=INPUT_SIZE,
                output_height=OUTPUT_SIZE, output_width=OUTPUT_SIZE,
                feat_width=3*OUTPUT_SIZE, feat_height=3*OUTPUT_SIZE,
                min_output=min_output, max_output=max_output,
                in_channels=INPUT_CHANNELS).to(device)
model.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=device)) 

#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])
#print('Number of params', params)
#exit(0)

criterion = nn.MSELoss(reduction='mean')

# Evaluate the model
results_numpy = eval_model(model, dataloader, dataset_size, criterion, device, sif_mean, sif_std)

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
plt.xlim(left=0, right=2)
plt.ylim(bottom=0, top=2)
plt.title('True vs predicted SIF (CFIS):' + METHOD)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '.png')
plt.close()

# Plot true vs predicted for each crop
fig, axeslist = plt.subplots(ncols=3, nrows=10, figsize=(15, 50))
fig.suptitle('True vs predicted SIF (CFIS): ' + METHOD)
for idx, crop_type in enumerate(COLUMN_NAMES[2:]):
    crop_rows = results_df.loc[results_df[crop_type] > 0.5]
    print(len(crop_rows), 'subtiles that are majority', crop_type)
    axeslist.ravel()[idx].scatter(crop_rows['true'], crop_rows['predicted'])
    axeslist.ravel()[idx].set(xlabel='True', ylabel='Predicted')
    axeslist.ravel()[idx].set_xlim(left=0, right=2)
    axeslist.ravel()[idx].set_ylim(bottom=0, top=2)
    axeslist.ravel()[idx].set_title(crop_type)

plt.tight_layout()
fig.subplots_adjust(top=0.96)
plt.savefig(TRUE_VS_PREDICTED_PLOT + '_crop_types.png')
plt.close()



