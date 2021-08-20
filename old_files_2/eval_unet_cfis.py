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
import sif_utils
from unet.unet_model import UNet, UNetSmall, UNet2

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_all_2") #"dataset_2018-07-16")
# EVAL_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2016-08-01") #"dataset_2016-07-16")
EVAL_FILE = os.path.join(TRAIN_DATASET_DIR, "cfis_subtiles_filtered.csv") #_1000soundings.csv") 
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
# MODEL_TYPE = "unet_small"
# METHOD = "7_unet_small_both_random_output_0.01_decay_1e-3" #"7_unet_small_both_random_output_0.01_decay_1e-3" # "7_unet_small_both_1000samples_random_output_0.001" #"7_unet_small_both_1000samples_no_norm"
# METHOD = "7_unet_small_both_1000samples_no_norm"
# METHOD = "7_unet_small_both_1000samples_random_output_0.001
# METHOD = "7_unet_small_both_1000samples"
MODEL_TYPE = "unet2"
METHOD = "7_unet2_clip_-6_8_batchnorm_aug"
# MODEL_TYPE = "pixel_nn"
# METHOD = "7_pixel_nn_1000samples"

TRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)
TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_cfis_' + METHOD
RESULTS_CSV_FILE = os.path.join(TRAIN_DATASET_DIR, 'cfis_results_' + METHOD + '.csv')

COLUMN_NAMES = ['true', 'predicted',
                    'lon', 'lat', 'date', 'tile_file', 'large_tile_lon', 'large_tile_lat',
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
INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils', 'missing_reflectance']
CROP_TYPES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest']
BANDS = list(range(0, 43))
# BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42]  #list(range(0, 43))
DATES = ["2018-04-29", "2018-05-13", "2018-05-27", "2018-06-10", "2018-06-24", 
         "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19", "2018-09-02",
         "2018-09-16"]

INPUT_CHANNELS = len(BANDS)
REDUCED_CHANNELS = 15
# DISCRETE_BANDS = list(range(12, 43))
# COVER_INDICES = list(range(12, 42))
MISSING_REFLECTANCE_IDX = -1
# RESIZE = False
MIN_SIF = None #0
MAX_SIF = None # 1.7
MAX_CFIS_SIF = 2.7
MAX_PRED = 1.7
MIN_INPUT = -6 # -20
MAX_INPUT = 8  # 20
BATCH_SIZE = 64
NUM_WORKERS = 8
PURE_THRESHOLD = 0.7
MIN_SOUNDINGS = 5

INPUT_SIZE = 371
TILE_SIZE_DEGREES = 0.1
SUBTILE_PIXELS = 10
RES = (0.00026949458523585647, 0.00026949458523585647)  # Size of each Landsat/image pixel, in degrees


# Returns a Pandas dataframe with true vs predicted CFIS results. 
def compute_cfis_results(model, dataset, transform, bands, criterion, device, sif_mean, sif_std):
    model.eval()   # Set model to evaluate mode
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)
    results = []
    j = 0

    # Group points by which large tile file they come from
    for large_tile_file, rows in dataset.groupby('tile_file', as_index=False):
        # Read large tile, standardize it, convert to Torch tensor
        input_tile_standardized = transform(np.load(large_tile_file))
        title = os.path.basename(large_tile_file)
        # sif_utils.plot_tile(input_tile_standardized, 'unet_eval_input_' + title + '.png', title=title)

        input_tile_standardized = torch.tensor(input_tile_standardized, dtype=torch.float).to(device)
        cloud_mask = input_tile_standardized[MISSING_REFLECTANCE_IDX, :, :]  # (H, W)

        # Add a batch dimension (even though batch size is 1)
        input_tile_standardized = input_tile_standardized.unsqueeze(0)

        # Obtain model SIF predictions
        with torch.set_grad_enabled(False):
            predictions_std = model(input_tile_standardized[:, bands, :, :])  # predictions: (1, 1, H, W)
            predictions_std = torch.squeeze(predictions_std) # (H, W)
            predictions = torch.tensor(predictions_std * sif_std + sif_mean, dtype=torch.float).to(device)

        # Loop through each sub-tile in this large tile
        for idx in range(len(rows)):
            row = rows.iloc[idx]
            subtile_lat = row['lat']
            subtile_lon = row['lon']
            subtile_true_sif = row['SIF']

            top_bound = get_top_bound(subtile_lat)
            left_bound = get_left_bound(subtile_lon)
            large_tile_lat = round(top_bound - TILE_SIZE_DEGREES / 2, 2)
            large_tile_lon = round(left_bound + TILE_SIZE_DEGREES / 2, 2)

            subtile_height_idx, subtile_width_idx = lat_long_to_index(subtile_lat, subtile_lon, top_bound, left_bound, RES)
            eps = int(SUBTILE_PIXELS / 2)
            height_start = max(subtile_height_idx-eps, 0)
            width_start = max(subtile_width_idx-eps, 0)
            height_end = min(subtile_height_idx+eps, predictions.shape[0])
            width_end = min(subtile_width_idx+eps, predictions.shape[1])
            # Edge case: subtile is 1 pixel only
            if height_end == height_start:
                height_end += 1
            if width_end == width_start:
                width_end += 1
            subtile_predictions = predictions[height_start:height_end, width_start:width_end] #subtile_height_idx-eps:subtile_height_idx+eps, subtile_width_idx-eps:subtile_width_idx+eps]
            subtile_cloud_mask = cloud_mask[height_start:height_end, width_start:width_end] #[subtile_height_idx-eps:subtile_height_idx+eps, subtile_width_idx-eps:subtile_width_idx+eps]
            # print('large tile file', large_tile_file)
            # print('lat', subtile_lat, 'lon', subtile_lon)
            # print('Height indices', height_start, height_end)
            # print('Width indices', width_start, width_end)
            # print('Subtile predictions', subtile_predictions)
            # print('Subtile cloud mask', subtile_cloud_mask)
            subtile_predicted_sif = sif_utils.masked_average(subtile_predictions, subtile_cloud_mask, dims_to_average=(0, 1)).item()
            # print('True', subtile_true_sif, 'Predicted', subtile_predicted_sif)

            #print('Band means shape', band_means.shape)
            subtile = input_tile_standardized[:, :, height_start:height_end, width_start:width_end] # [:, :, subtile_height_idx-eps:subtile_height_idx+eps, subtile_width_idx-eps:subtile_width_idx+eps]
            subtile_averages = torch.mean(subtile, dim=(0, 2, 3))
            result_row = [subtile_true_sif, subtile_predicted_sif, row['lon'], row['lat'], row['date'],
                          large_tile_file, large_tile_lon, large_tile_lat] + row[INPUT_COLUMNS].tolist()
            results.append(result_row)
            j += 1
        #if j > 1000: #10:
        #    break

    results_df = pd.DataFrame(results, columns=COLUMN_NAMES)
    results_df['predicted'] = results_df['predicted'].clip(lower=0.2)
    return results_df


def plot_cfis_scatters(results_df, method, true_vs_predicted_plot, x_max, y_max, sif_mean):
    print('================== CFIS stats: all crops ========================')
    print_stats(results_df['true'].to_numpy(), results_df['predicted'].to_numpy(), sif_mean, ax=plt.gca())
    plt.title('CFIS: true vs predicted SIF (' + method + ')')
    plt.xlim(left=0, right=x_max)
    plt.ylim(bottom=0, top=y_max)
    plt.savefig(true_vs_predicted_plot + '.png')
    plt.close()

    fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
    fig.suptitle('True vs predicted SIF by crop type: ' + method)
    for idx, crop_type in enumerate(CROP_TYPES):
        crop_rows = results_df.loc[results_df[crop_type] > PURE_THRESHOLD]
        print('==================== CFIS stats: Crop type', crop_type, '-', len(crop_rows), 'subtiles ==========================')
        if len(crop_rows) < 2:
            continue
        ax = axeslist.ravel()[idx]
        print_stats(crop_rows['true'].to_numpy(), crop_rows['predicted'].to_numpy(), sif_mean, ax=ax)
        ax.set_xlim(left=0, right=x_max)
        ax.set_ylim(bottom=0, top=y_max)
        ax.set_title(crop_type)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(true_vs_predicted_plot + '_crop_types.png')
    plt.close()


def main():
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
    transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds)) #, min_input=MIN_INPUT, max_input=MAX_INPUT))
    transform_list.append(tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT))
    # transform_list.append(tile_transforms.TanhTile(tanh_stretch=3, bands_to_transform=list(range(0, 12))))
    # if RESIZE:
    #     transform_list.append(tile_transforms.ResizeTile(target_dim=RESIZED_DIM, discrete_bands=DISCRETE_BANDS))
    transform = transforms.Compose(transform_list)

    # # Set up Dataset and Dataloader
    # dataset_size = len(eval_metadata)
    # dataset = ReflectanceCoverSIFDataset(eval_metadata, transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
    #                                          shuffle=False, num_workers=NUM_WORKERS)

    if MODEL_TYPE == 'unet_small':
        model = UNetSmall(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
    elif MODEL_TYPE == 'pixel_nn':
        model = simple_cnn.PixelNN(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
    elif MODEL_TYPE == 'unet2':
        model = UNet2(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
    else:
        print('Model type not supported')
        exit(1)
    model.load_state_dict(torch.load(TRAINED_MODEL_FILE, map_location=device))

    criterion = nn.MSELoss(reduction='mean')

    # Evaluate the model
    results_df = compute_cfis_results(model, eval_metadata, transform, BANDS, criterion, device, sif_mean, sif_std)
    # print("Result example", results_df.head())
    results_df.to_csv(RESULTS_CSV_FILE)
    plot_cfis_scatters(results_df, METHOD, TRUE_VS_PREDICTED_PLOT, MAX_PRED, MAX_CFIS_SIF, sif_mean)

if __name__ == "__main__":
    main()