import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from itertools import chain
import random
import time
import torch
import torchvision
import torchvision.transforms as transforms
import resnet
import torch.nn as nn
import torch.optim as optim

import simple_cnn
from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset, CombinedDataset
from unet.unet_model import UNet, UNetSmall, UNet2
import eval_unet_cfis
import sif_utils
import tile_transforms

# Set random seed
torch.manual_seed(0)

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "processed_dataset_all_2")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "standardized_tiles_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "standardized_tiles_val.csv")
CFIS_FILE = os.path.join(DATASET_DIR, "cfis_subtiles_filtered.csv")
# INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
# INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
# METHOD = "7_unet_small_both_10000samples" #_random_output_0.01"
# METHOD = "7_unet2" #_1000samples"
METHOD = "7_unet2_clip_-6_8_batchnorm_dimred"
# METHOD = "7_unet2_reflectance_only_aug"
MODEL_TYPE = "unet2"
# MODEL_TYPE = "pixel_nn"
# METHOD = "7_pixel_nn_1000samples"
CFIS_RESULTS_CSV_FILE = os.path.join(DATASET_DIR, 'cfis_results_' + METHOD + '.csv')
CFIS_TRUE_VS_PREDICTED_PLOT = 'exploratory_plots/true_vs_predicted_sif_cfis_' + METHOD
LOSS_PLOT = 'loss_plots/losses_' + METHOD

PRETRAINED_UNET_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)
UNET_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD) #aug_2")
# MODEL_TYPE = "pixel_nn" # "unet_small"
# MODEL_TYPE = "unet2"
OPTIMIZER_TYPE = "Adam"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 16
NUM_WORKERS = 8
AUGMENT = True
FROM_PRETRAINED = True
MIN_SIF = None
MAX_SIF = None
MIN_INPUT = -6 #-3
MAX_INPUT = 8 #3

MAX_PRED = 2
MAX_CFIS_SIF = 3
TRAIN_SOURCES = ['TROPOMI', 'OCO2']
VAL_SOURCES = ['TROPOMI', 'OCO2']
NUM_OCO2_TRAIN_SAMPLES = 467
OCO2_UPDATES_PER_TROPOMI = 1 #0.1
#BANDS = list(range(0, 9)) + [42] #  12)) + list(range(12, 27)) + [28] + [42] 
#BANDS = list(range(0, 12)) + [12, 13, 14, 16] + [42]
# BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42] 
BANDS = list(range(0, 43))
# BANDS = list(range(0, 9)) + [42]
MISSING_REFLECTANCE_IDX = -1
INPUT_CHANNELS = len(BANDS)
REDUCED_CHANNELS = 15
NOISE = 0.1
FRACTION_OUTPUTS_TO_AVERAGE = 1 # 0.01 # 0.1 # 1

# Print params for reference
print("=========================== PARAMS ===========================")
print("Train sources:", TRAIN_SOURCES)
print("Method:", METHOD)
print("Dataset: ", os.path.basename(DATASET_DIR))
if 'OCO2' in TRAIN_SOURCES:
    print('Num OCO2 samples:', NUM_OCO2_TRAIN_SAMPLES)
    print('OCO2 updates per TROPOMI:', OCO2_UPDATES_PER_TROPOMI)
if FROM_PRETRAINED:
    print("From pretrained model", os.path.basename(PRETRAINED_UNET_MODEL_FILE))
else:
    print("Training from scratch")
print("Output model:", os.path.basename(UNET_MODEL_FILE))
print("Bands:", BANDS)
print("---------------------------------")
print("Model:", MODEL_TYPE)
print("Optimizer:", OPTIMIZER_TYPE)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Num workers:", NUM_WORKERS)
print("Batch size:", BATCH_SIZE)
print("Num epochs:", NUM_EPOCHS)
print("Augment:", AUGMENT)
if AUGMENT:
    print("Gaussian noise (std deviation):", NOISE)
print("Input features clipped to", MIN_INPUT, "to", MAX_INPUT, "standard deviations from mean")
print("SIF range:", MIN_SIF, "to", MAX_SIF)
print("Fraction outputs to average:", FRACTION_OUTPUTS_TO_AVERAGE)
print("==============================================================")


# TODO should there be 2 separate models?
def train_model(model, dataloaders, cfis_metadata, cfis_transform, criterion, optimizer, device, sif_mean, sif_std, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_tropomi_loss = float('inf')
    best_oco2_loss = float('inf')
    best_cfis_loss = float('inf')
    train_tropomi_losses = []
    train_tropomi_mse = []
    train_oco2_losses = []
    train_oco2_mse = []
    val_tropomi_losses = []
    val_tropomi_mse = []
    val_oco2_losses = []
    val_oco2_mse = []
    cfis_losses = []
    cfis_mse = []

    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
    # sif_mean = torch.tensor(sif_mean).to(device)
    # sif_std = torch.tensor(sif_std).to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_tropomi_loss = 0.0
            running_oco2_loss = 0.0
            num_tropomi_points = 0
            num_oco2_points = 0
            tropomi_batch_idx = 0
            oco2_batch_idx = 0

            # Iterate over data.
            j = 0
            for sample in dataloaders[phase]:
                if 'tropomi_tile' in sample:
                    # Read TROPOMI tile and SIF from dataloader
                    tropomi_tiles_std = sample['tropomi_tile'][:, BANDS, :, :].to(device)  # (batch size, num channels, H, W)
                    tropomi_true_sifs = sample['tropomi_sif'].to(device)  # (batch_size)
                    assert(tropomi_tiles_std.shape[0] == tropomi_true_sifs.shape[0])

                    # Binary mask for non-cloudy pixels. Since the tiles are passed through the
                    # StandardizeTile transform, 1 now represents non-cloudy (data present) and 0 represents
                    # cloudy (data missing).
                    non_cloudy_pixels = tropomi_tiles_std[:, MISSING_REFLECTANCE_IDX, :, :]  # (batch size, H, W)

                    # As a regularization technique, randomly choose more pixels to ignore.
                    # if phase == 'train':
                    #     pixels_to_include = torch.rand_like(non_cloudy_pixels, device=device) > (1 - FRACTION_OUTPUTS_TO_AVERAGE)
                    #     non_cloudy_pixels = non_cloudy_pixels * pixels_to_include

                    # Standardize SIF
                    tropomi_true_sifs_std = ((tropomi_true_sifs - sif_mean) / sif_std).to(device)

                    # tile_description = sample['tropomi_description'][1]
                    # title = tile_description + ' (SIF = ' + str(round(sample['tropomi_sif'][1].item(), 3)) + ')'
                    # sif_utils.plot_tile(tropomi_tiles_std[1].detach().numpy(), 'unet_input_' + tile_description + '.png', title=title)
                    # exit(0)

                    with torch.set_grad_enabled(phase == 'train'):
                        tropomi_predicted_sifs_std = model(tropomi_tiles_std)  # tropomi_predicted_sifs_std: (batch size, 1, H, W)
                        tropomi_predicted_sifs_std = torch.squeeze(tropomi_predicted_sifs_std, dim=1)
                        tropomi_predicted_sifs_std = sif_utils.masked_average(tropomi_predicted_sifs_std, non_cloudy_pixels, dims_to_average=(1, 2)) # (batch size)

                        # Compute loss: predicted vs true SIF (standardized)
                        loss = criterion(tropomi_predicted_sifs_std, tropomi_true_sifs_std)
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    with torch.set_grad_enabled(False):
                        # statistics
                        tropomi_predicted_sifs = torch.tensor(tropomi_predicted_sifs_std * sif_std + sif_mean, dtype=torch.float).to(device)
                        non_standardized_loss = criterion(tropomi_predicted_sifs, tropomi_true_sifs)
                        tropomi_batch_idx += 1
                        # if tropomi_batch_idx % 500 == 1:
                        #     print('========================')
                        #     print('*** (TROPOMI) Predicted', tropomi_predicted_sifs[0:20])
                        #     print('*** True', tropomi_true_sifs[0:20])
                        #     print('*** batch loss', (math.sqrt(non_standardized_loss.item()) / sif_mean).item())
                        running_tropomi_loss += non_standardized_loss.item() * len(sample['tropomi_sif'])
                        num_tropomi_points += len(sample['tropomi_sif'])

                if 'oco2_tile' in sample:
                    if (phase == 'val') or (phase == 'train' and random.random() < OCO2_UPDATES_PER_TROPOMI):
                        # Read TROPOMI tile and SIF from dataloader
                        oco2_tiles_std = sample['oco2_tile'][:, BANDS, :, :].to(device)  # (batch size, num channels, H, W)
                        oco2_true_sifs = sample['oco2_sif'].to(device)  # (batch_size)
                        assert(oco2_tiles_std.shape[0] == oco2_true_sifs.shape[0])

                        # Standardize SIF
                        oco2_true_sifs_std = ((oco2_true_sifs - sif_mean) / sif_std).to(device)

                        # Binary mask for non-cloudy pixels
                        non_cloudy_pixels = oco2_tiles_std[:, MISSING_REFLECTANCE_IDX, :, :]  # (batch size, H, W)
 
                        # As a regularization technique, randomly choose more pixels to ignore.
                        # if phase == 'train':
                        #     pixels_to_include = torch.rand_like(non_cloudy_pixels, device=device) > (1 - FRACTION_OUTPUTS_TO_AVERAGE)
                        #     non_cloudy_pixels = non_cloudy_pixels * pixels_to_include

                        with torch.set_grad_enabled(phase == 'train'):
                            oco2_predicted_sifs_std = model(oco2_tiles_std)  # (batch size, 1, H, W)
                            oco2_predicted_sifs_std = torch.squeeze(oco2_predicted_sifs_std, dim=1)
                            oco2_predicted_sifs_std = sif_utils.masked_average(oco2_predicted_sifs_std, non_cloudy_pixels, dims_to_average=(1, 2)) # (batch size)

                            # Compute loss: predicted vs true SIF (standardized)
                            loss = criterion(oco2_predicted_sifs_std, oco2_true_sifs_std)
                            if phase == 'train':
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step() 

                        with torch.set_grad_enabled(False):
                            # statistics
                            oco2_predicted_sifs = torch.tensor(oco2_predicted_sifs_std * sif_std + sif_mean, dtype=torch.float).to(device)
                            oco2_predicted_sifs = torch.clamp(oco2_predicted_sifs, min=0.2, max=1.7)
                            non_standardized_loss = criterion(oco2_predicted_sifs, oco2_true_sifs)
                            oco2_batch_idx += 1
                            # if oco2_batch_idx % 200 == 1:
                            #     print('========================')
                            #     print('*** (OCO-2) Predicted', oco2_predicted_sifs[0:20])
                            #     print('*** True', oco2_true_sifs[0:20])
                            #     print('*** batch loss', (math.sqrt(non_standardized_loss.item()) / sif_mean).item())
                            running_oco2_loss += non_standardized_loss.item() * len(sample['oco2_sif'])
                            num_oco2_points += len(sample['oco2_sif'])


            # Compute NRMSE and MSE on entire TROPOMI and OCO-2 datasets
            if num_tropomi_points > 0:
                epoch_tropomi_mse = running_tropomi_loss / num_tropomi_points
                epoch_tropomi_nrmse = (math.sqrt(epoch_tropomi_mse) / sif_mean).item()
                # print('{} MSE - TROPOMI: {:.3f}'.format(
                #     phase, epoch_tropomi_mse))                
                print('{} NRMSE - TROPOMI: {:.3f}'.format(
                    phase, epoch_tropomi_nrmse))
                if phase == 'train':
                    train_tropomi_losses.append(epoch_tropomi_nrmse)
                    train_tropomi_mse.append(epoch_tropomi_mse)
                else:
                    val_tropomi_losses.append(epoch_tropomi_nrmse)
                    val_tropomi_mse.append(epoch_tropomi_mse)

            if num_oco2_points > 0:
                epoch_oco2_mse = running_oco2_loss / num_oco2_points
                epoch_oco2_nrmse = (math.sqrt(epoch_oco2_mse) / sif_mean).item()
                # print('{} MSE - OCO-2: {:.3f}'.format(
                #     phase, epoch_oco2_mse))
                print('{} NRMSE - OCO-2: {:.3f}'.format(
                    phase, epoch_oco2_nrmse))
                if phase == 'train':
                    train_oco2_losses.append(epoch_oco2_nrmse)
                    train_oco2_mse.append(epoch_oco2_mse)
                else:
                    val_oco2_losses.append(epoch_oco2_nrmse)
                    val_oco2_mse.append(epoch_oco2_mse)

            # deep copy the model
            if phase == 'val' and epoch_tropomi_nrmse < best_tropomi_loss:
                best_tropomi_loss = epoch_tropomi_nrmse
            if phase == 'val' and epoch_oco2_nrmse < best_oco2_loss:
                best_oco2_loss = epoch_oco2_nrmse
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), UNET_MODEL_FILE)
                # torch.save(model.state_dict(), UNET_MODEL_FILE + "_epoch" + str(epoch))
            
        # Evaluate on CFIS
        cfis_results_df = eval_unet_cfis.compute_cfis_results(model, cfis_metadata, cfis_transform, BANDS, criterion, device, sif_mean, sif_std)
        cfis_nrmse_scaled, cfis_mse_scaled, cfis_r2 = sif_utils.print_stats(cfis_results_df['true'].to_numpy(), cfis_results_df['predicted'].to_numpy(), sif_mean, print_report=False, ax=None)
        print('NRMSE (scaled) - CFIS: {:.3f}'.format(cfis_nrmse_scaled))
        cfis_losses.append(cfis_nrmse_scaled)
        cfis_mse.append(cfis_mse_scaled)
        print('R2 - CFIS: {:.3f}'.format(cfis_r2))
        if cfis_nrmse_scaled < best_cfis_loss:
            best_cfis_loss = cfis_nrmse_scaled
            cfis_results_df.to_csv(CFIS_RESULTS_CSV_FILE)
            eval_unet_cfis.plot_cfis_scatters(cfis_results_df, METHOD, CFIS_TRUE_VS_PREDICTED_PLOT, MAX_PRED, MAX_CFIS_SIF, sif_mean)

        # Print elapsed time per epoch
        epoch_time = time.time() - epoch_start
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            epoch_time // 60, epoch_time % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best TROPOMI val loss: {:4f}'.format(best_tropomi_loss))
    print('Best OCO-2 val loss: {:4f}'.format(best_oco2_loss))
    print('Best CFIS loss: {:4f}'.format(best_cfis_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_tropomi_losses, train_oco2_losses, val_tropomi_losses, val_oco2_losses, cfis_losses, train_tropomi_mse, train_oco2_mse, val_tropomi_mse, val_oco2_mse, cfis_mse, best_oco2_loss


# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)

# Read train/val tile metadata
train_metadata = pd.read_csv(INFO_FILE_TRAIN)
val_metadata = pd.read_csv(INFO_FILE_VAL)
cfis_metadata = pd.read_csv(CFIS_FILE)

# Extract TROPOMI and OCO-2 rows, if applicable
if 'TROPOMI' in TRAIN_SOURCES:
    train_tropomi_metadata = train_metadata[(train_metadata['source'] == 'TROPOMI')].iloc[0:1000] # & (train_metadata['date'] == '2018-08-05')].iloc[0:1000]
    print('Train TROPOMI samples', len(train_tropomi_metadata))
else:
    train_tropomi_metadata = None

if 'OCO2' in TRAIN_SOURCES:
    train_oco2_metadata = train_metadata[train_metadata['source'] == 'OCO2'].iloc[0:NUM_OCO2_TRAIN_SAMPLES]
    print('Train OCO2 samples', len(train_oco2_metadata))
else:
    train_oco2_metadata = None

if 'TROPOMI' in VAL_SOURCES:
    val_tropomi_metadata = val_metadata[val_metadata['source'] == 'TROPOMI']
    print('Val TROPOMI samples', len(val_tropomi_metadata))
else:
    val_tropomi_metadata = None

if 'OCO2' in VAL_SOURCES:
    val_oco2_metadata = val_metadata[val_metadata['source'] == 'OCO2']
    print('Val OCO2 samples', len(val_oco2_metadata))
else:
    val_oco2_metadata = None

# NOTE For technical reasons, ensure that TROPOMI and OCO2 validation data have the same length
# (so that OCO2 points are not repeated)
val_tropomi_metadata = val_tropomi_metadata.iloc[0:len(val_oco2_metadata)]

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
print("Means", train_means)
print("Stds", train_stds)
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


# Set up image transforms / augmentations
standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds) #, min_input=MIN_INPUT, max_input=MAX_INPUT)
# tanh_transform = tile_transforms.TanhTile(tanh_stretch=3, bands_to_transform=list(range(0, 12)))
clip_transform = tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT)
noise_transform = tile_transforms.GaussianNoise(continuous_bands=list(range(0, 9)), standard_deviation=NOISE)
flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()

transform_list_train = [clip_transform] #noise_transform] # [standardize_transform, noise_transform]
transform_list_val = [clip_transform] #[standardize_transform]
transform_list_cfis = [standardize_transform, clip_transform]
if AUGMENT:
    transform_list_train += [flip_and_rotate_transform, noise_transform]
train_transform = transforms.Compose(transform_list_train)
val_transform = transforms.Compose(transform_list_val)
cfis_transform = transforms.Compose(transform_list_cfis)

# Create dataset/dataloaders
datasets = {'train': CombinedDataset(train_tropomi_metadata, train_oco2_metadata, train_transform, return_subtiles=False,
                                     subtile_dim=None, tile_file_column='standardized_tile_file'),
            'val': CombinedDataset(val_tropomi_metadata, val_oco2_metadata, val_transform, return_subtiles=False,
                                   subtile_dim=None, tile_file_column='standardized_tile_file')}  # Only validate on OCO-2
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS)
            for x in ['train', 'val']}

# Initialize model
if MODEL_TYPE == 'unet_small':
    model = UNetSmall(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'pixel_nn':
    model = simple_cnn.PixelNN(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'unet2':
    model = UNet2(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
else:
    print('Model type not supported')
    exit(1)

# If we're loading a pre-trained model, read model params from file
if FROM_PRETRAINED:
    model.load_state_dict(torch.load(PRETRAINED_UNET_MODEL_FILE, map_location=device))

# Initialize loss and optimizer
criterion = nn.MSELoss(reduction='mean')
if OPTIMIZER_TYPE == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    print("Optimizer not supported")
    exit(1)

model, train_tropomi_losses, train_oco2_losses, val_tropomi_losses, val_oco2_losses, cfis_losses, train_tropomi_mse, train_oco2_mse, val_tropomi_mse, val_oco2_mse, cfis_mse, best_loss = train_model(model, dataloaders, cfis_metadata, cfis_transform, criterion, optimizer, device, sif_mean, sif_std, num_epochs=NUM_EPOCHS)

torch.save(model.state_dict(), UNET_MODEL_FILE)

# Plot loss curves: MSE
epoch_list = range(NUM_EPOCHS)
plots = []
if 'TROPOMI' in TRAIN_SOURCES:
    train_tropomi_plot, = plt.plot(epoch_list, train_tropomi_mse, color='blue', label='Train TROPOMI L2 loss')
    plots.append(train_tropomi_plot)
if 'OCO2' in TRAIN_SOURCES:
    train_oco2_plot, = plt.plot(epoch_list, train_oco2_mse, color='red', label='Train OCO-2 L2 loss')
    plots.append(train_oco2_plot)
if 'TROPOMI' in VAL_SOURCES:
    val_tropomi_plot, = plt.plot(epoch_list, val_tropomi_mse, color='green', label='Val TROPOMI L2 loss')
    plots.append(val_tropomi_plot)
if 'OCO2' in VAL_SOURCES:
    val_oco2_plot, = plt.plot(epoch_list, val_oco2_mse, color='orange', label='Val OCO-2 L2 loss')
    plots.append(val_oco2_plot)
cfis_plot, = plt.plot(epoch_list, cfis_mse, color='black', label='CFIS L2 loss (scaled)')
plots.append(cfis_plot)

# Add legend and axis labels
plt.legend(handles=plots)
plt.xlabel('Epoch #')
plt.ylabel('L2 loss (mean squared error)')
plt.savefig(LOSS_PLOT + '_mse.png') 
plt.close()

# Plot loss curves: NRMSE
epoch_list = range(NUM_EPOCHS)
plots = []
if 'TROPOMI' in TRAIN_SOURCES:
    print("Train TROPOMI losses:", train_tropomi_losses)
    train_tropomi_plot, = plt.plot(epoch_list, train_tropomi_losses, color='blue', label='Train TROPOMI NRMSE')
    plots.append(train_tropomi_plot)
if 'OCO2' in TRAIN_SOURCES:
    print("Train OCO2 losses:", train_oco2_losses)
    train_oco2_plot, = plt.plot(epoch_list, train_oco2_losses, color='red', label='Train OCO-2 NRMSE')
    plots.append(train_oco2_plot)
if 'TROPOMI' in VAL_SOURCES:
    print("Val TROPOMI losses:", val_tropomi_losses)
    val_tropomi_plot, = plt.plot(epoch_list, val_tropomi_losses, color='green', label='Val TROPOMI NRMSE')
    plots.append(val_tropomi_plot)
if 'OCO2' in VAL_SOURCES:
    print("Val OCO2 losses:", val_oco2_losses)
    val_oco2_plot, = plt.plot(epoch_list, val_oco2_losses, color='orange', label='Val OCO-2 NRMSE')
    plots.append(val_oco2_plot)
print('CFIS losses', cfis_losses)
cfis_plot, = plt.plot(epoch_list, cfis_losses, color='black', label='CFIS NRMSE (scaled)')
plots.append(cfis_plot)

# Add legend and axis labels
plt.legend(handles=plots)
plt.xlabel('Epoch #')
plt.ylabel('NRMSE')
plt.savefig(LOSS_PLOT + '_nrmse.png') 
plt.close()

# Plot loss curves: NRMSE
epoch_list = range(NUM_EPOCHS)
plots = []
if 'OCO2' in VAL_SOURCES:
    val_oco2_plot, = plt.plot(epoch_list, val_oco2_losses, color='orange', label='Val OCO-2 NRMSE')
    plots.append(val_oco2_plot)
cfis_plot, = plt.plot(epoch_list, cfis_losses, color='black', label='CFIS NRMSE (scaled)')
plots.append(cfis_plot)

# Add legend and axis labels
plt.legend(handles=plots)
plt.xlabel('Epoch #')
plt.ylabel('NRMSE')
plt.savefig(LOSS_PLOT + '_nrmse_cfis_vs_val_oco2.png') 
plt.close()

# Plot val TROPOMI vs OCO-2 losses
print('============== CFIS vs val OCO-2 losses ===============')
sif_utils.print_stats(cfis_losses, val_oco2_losses, sif_mean, ax=plt.gca())
# plt.scatter(val_oco2_losses, cfis_losses)
plt.xlabel('Val OCO2 losses')
plt.ylabel('CFIS losses')
plt.title('CFIS vs val OCO-2 losses' + METHOD)
# plt.xlim(left=0, right=0.5)
# plt.ylim(bottom=0, top=0.5)
plt.savefig(LOSS_PLOT + '_scatter_cfis_vs_val_oco2.png')
plt.close()
