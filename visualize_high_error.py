import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import sys
import resnet
import simple_cnn
import sif_utils
import tile_transforms
import torchvision
import torchvision.transforms as transforms
from eval_subtile_dataset import EvalSubtileDataset
from sif_utils import plot_histogram

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"

TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_all_2") #"dataset_2018-07-16")

# RESULTS_FILE = os.path.join(TRAIN_DATASET_DIR, "OCO2_results_2d_train_both_subtile_resnet_100samples_3.csv")
# TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/2d_train_both_subtile_resnet_100samples_3")

# EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-08-01") #"dataset_2016-07-16")
# RESULTS_FILE = os.path.join(EVAL_DATASET_DIR, "results_4a_subtile_simple_cnn_new_data.csv")
# EVAL_FILE = os.path.join(EVAL_DATASET_DIR, "eval_subtiles.csv") 
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
MIN_INPUT = -3
MAX_INPUT = 3
NUM_TILES = 10
SUBTILE_DIM = 10
BANDS = list(range(0, 43))
INPUT_CHANNELS = len(BANDS)
REDUCED_CHANNELS = 15
CROP_TYPE_EMBEDDING_DIM = 10
MIN_SIF = 0
MAX_SIF = 1.7
sif_cmap = plt.get_cmap('YlGn')


# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

# Standardized bounds on SIF
min_output = (MIN_SIF - sif_mean) / sif_std
max_output = (MAX_SIF - sif_mean) / sif_std

# Load sub-tile model
# subtile_sif_model = simple_cnn.SimpleCNNSmall5(input_channels=INPUT_CHANNELS_SIMPLE, output_dim=1, min_output=min_output, max_output=max_output).to(device)
subtile_sif_model = resnet.resnet18(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS,
                                    crop_type_embedding_dim=CROP_TYPE_EMBEDDING_DIM, num_classes=1,
                                    min_output=min_output, max_output=max_output).to(device)

subtile_sif_model.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=device))
subtile_sif_model.eval()

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds, min_input=MIN_INPUT, max_input=MAX_INPUT))
transform = transforms.Compose(transform_list)

results = pd.read_csv(RESULTS_FILE)
errors = np.abs(results['true'] - results['predicted'])
print('Error shape', errors.shape)
print(errors)
sorted_indices = errors.argsort()  # Ascending order of distance
high_error_indices = sorted_indices[-NUM_TILES:][::-1]

for high_error_idx in high_error_indices:
    row = results.iloc[high_error_idx]
    high_error_tile = transform(np.load(row['tile_file']))
    tile_description = 'oco2_high_error_' + os.path.basename(row['tile_file'])
    title = 'Lat ' + str(round(row['lat'], 4)) + ', Lon ' + str(round(row['lon'], 4))
    if 'date' in row:
        title += (', Date ' + row['date'])
    title += ('\n(True SIF: ' + str(round(row['true'], 3)) + ', Predicted SIF: ' + str(round(row['predicted'], 3)) + ')')
    print('Tile:', tile_description)
    print('header:', title)
    sif_utils.plot_tile(high_error_tile, tile_description, title=title)

    # # Feed each sub-tile through the model
    # input_tile_standardized = torch.tensor(high_error_tile, dtype=torch.float) 
    # subtiles_standardized = sif_utils.get_subtiles_list(input_tile_standardized, SUBTILE_DIM) #, device, max_subtile_cloud_cover=None)  # (batch x num subtiles x bands x subtile_dim x subtile_dim)
    # with torch.set_grad_enabled(False):
    #     predicted_sifs_simple_cnn_standardized = subtile_sif_model(subtiles_standardized[:, BANDS, :, :]).detach().numpy()
    # predicted_sifs_simple_cnn_non_standardized = (predicted_sifs_simple_cnn_standardized * sif_std + sif_mean).reshape((10, 10))

    # fig, axeslist = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # pcm = axeslist.imshow(predicted_sifs_simple_cnn_non_standardized, cmap=sif_cmap, vmin=0.2, vmax=1.7)
    # axeslist.set_title('Predicted SIF (sub-tile CNN)\n' + title)
    # fig.colorbar(pcm, ax=axeslist) #.ravel().tolist())
    # plt.savefig('exploratory_plots/' + tile_description + '_predicted_subtile_sifs.png')
    # plt.close()