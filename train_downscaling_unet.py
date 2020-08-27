import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import simple_cnn
from reflectance_cover_sif_dataset import CFISDataset
from unet.unet_model import UNet, UNetSmall, UNet2
import cdl_utils
import sif_utils
import tile_transforms

# Set random seed
torch.manual_seed(0)

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "CFIS")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "cfis_tile_metadata_train_4soundings.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "cfis_tile_metadata_val_4soundings.csv")
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "cfis_band_statistics_train_4soundings.csv")
METHOD = "8_downscaling_unet_no_batchnorm_no_augment_no_decay_4soundings"
MODEL_TYPE = "unet"
CFIS_RESULTS_CSV_FILE = os.path.join(DATASET_DIR, 'cfis_results_' + METHOD + '.csv')
LOSS_PLOT = 'loss_plots/losses_' + METHOD
PRETRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD)
MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD) #aug_2")
OPTIMIZER_TYPE = "Adam"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
NUM_EPOCHS = 50
BATCH_SIZE = 8
NUM_WORKERS = 8
AUGMENT = False
FROM_PRETRAINED = False
MIN_SIF = None
MAX_SIF = None
MIN_INPUT = -2
MAX_INPUT = 2
MAX_PRED = 2
MAX_CFIS_SIF = 3
BANDS = list(range(0, 43))
INPUT_CHANNELS = len(BANDS)
MISSING_REFLECTANCE_IDX = -1
REDUCED_CHANNELS = 15
NOISE = 0 #.1
COARSE_SIF_PIXELS = 25
RES = (0.00026949458523585647, 0.00026949458523585647)
TILE_PIXELS = 200
TILE_SIZE_DEGREES = RES[0] * TILE_PIXELS

# Print params for reference
print("=========================== PARAMS ===========================")
print("Method:", METHOD)
print("Dataset: ", os.path.basename(DATASET_DIR))
if FROM_PRETRAINED:
    print("From pretrained model", os.path.basename(PRETRAINED_MODEL_FILE))
else:
    print("Training from scratch")
print("Output model:", os.path.basename(MODEL_FILE))
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
print("==============================================================")


def train_model(model, dataloaders, criterion, optimizer, device, sif_mean, sif_std, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_coarse_loss = float('inf')
    best_fine_loss = float('inf')
    train_coarse_losses = []
    val_coarse_losses = []
    train_fine_losses = []
    val_fine_losses = []
    print('SIF mean', sif_mean)
    print('SIF std', sif_std)

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

            running_coarse_loss = 0
            num_coarse_datapoints = 0
            running_fine_loss = 0
            num_fine_datapoints = 0
            batch_idx = 0
            all_true_fine_sifs = []
            all_true_coarse_sifs = []
            all_predicted_fine_sifs = []
            all_predicted_coarse_sifs = []            

            # Iterate over data.
            for sample in dataloaders[phase]:
                # Read input tile
                input_tiles_std = sample['input_tile'][:, BANDS, :, :].to(device)

                # Read coarse-resolution SIF labels
                true_coarse_sifs = sample['coarse_sif'].to(device)
                valid_coarse_sif_mask = torch.logical_not(sample['coarse_sif_mask']).to(device)  # Flatten coarse SIF mask, and flip so that valid points are True

                # Read fine-resolution SIF labels
                true_fine_sifs = sample['fine_sif'].to(device)
                valid_fine_sif_mask = torch.logical_not(sample['fine_sif_mask']).to(device)

                # Standardize SIF
                # true_coarse_sifs_std = ((true_coarse_sifs - sif_mean) / sif_std).to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    predicted_fine_sifs_std = model(input_tiles_std)  # predicted_fine_sifs_std: (batch size, 1, H, W)
                    predicted_fine_sifs_std = torch.squeeze(predicted_fine_sifs_std, dim=1)
                    predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean
                    # print('Predicted fine sifs', predicted_fine_sifs[0])

                    # Zero out predicted SIFs for invalid pixels (pixels with no valid SIF label, or cloudy pixels).
                    # Now, only VALID pixels contain a non-zero predicted SIF.
                    predicted_fine_sifs[valid_fine_sif_mask == 0] = 0

                    # TODO test if this computation is correct (compare against for-loop)

                    # For each coarse-SIF sub-region, compute the fraction of valid pixels.
                    # Each square is: (# valid fine pixels) / (# total fine pixels)
                    avg_pool = nn.AvgPool2d(kernel_size=COARSE_SIF_PIXELS)
                    fraction_valid = avg_pool(valid_fine_sif_mask.float())

                    # Average together fine SIF predictions for each coarse SIF area.
                    # Each square is: (sum predicted SIF over valid fine pixels) / (# total fine pixels)
                    predicted_coarse_sifs = avg_pool(predicted_fine_sifs)
                    # print('Predicted coarse sifs', predicted_coarse_sifs.shape)
                    # print('Fraction valid', fraction_valid.shape)

                    # Instead of dividing by the total number of fine pixels, divide by the number of VALID fine pixels.
                    # Each square is now: (sum predicted SIF over valid fine pixels) / (# valid fine pixels), which is what we want.
                    predicted_coarse_sifs = predicted_coarse_sifs / fraction_valid
                    predicted_coarse_sifs[valid_coarse_sif_mask == 0] = 0
                    # print('Predicted coarse sifs', predicted_coarse_sifs[0])
                    # print('True coarse sifs', true_coarse_sifs[0])

                    # large_tile_lat = sample['lat'][0].item()
                    # large_tile_lon = sample['lon'][0].item()
                    # date = sample['date'][0]
                    # cdl_utils.plot_tile(input_tiles_std[0].cpu().detach().numpy(), 
                    #                     true_coarse_sifs[0].cpu().detach().numpy(),
                    #                     true_fine_sifs[0].cpu().detach().numpy(),
                    #                     [predicted_coarse_sifs[0].cpu().detach().numpy()],
                    #                     [predicted_fine_sifs[0].cpu().detach().numpy()],
                    #                     ['UNet'], large_tile_lon, large_tile_lat, date,
                    #                     TILE_SIZE_DEGREES)
                    # exit(0)

                    # Extract the coarse SIF data points where we have labels, and compute loss
                    valid_coarse_sif_mask = valid_coarse_sif_mask.flatten()
                    true_coarse_sifs_filtered = true_coarse_sifs.flatten()[valid_coarse_sif_mask]
                    predicted_coarse_sifs_filtered = predicted_coarse_sifs.flatten()[valid_coarse_sif_mask]
                    coarse_loss = criterion(true_coarse_sifs_filtered, predicted_coarse_sifs_filtered)

                    # Extract the fine SIF data points where we have labels, and compute loss
                    valid_fine_sif_mask = valid_fine_sif_mask.flatten()
                    true_fine_sifs_filtered = true_fine_sifs.flatten()[valid_fine_sif_mask]
                    predicted_fine_sifs_filtered = predicted_fine_sifs.flatten()[valid_fine_sif_mask]
                    fine_loss = criterion(true_fine_sifs_filtered, predicted_fine_sifs_filtered)

                    # Backpropagate coarse loss
                    if phase == 'train':
                        optimizer.zero_grad()
                        coarse_loss.backward()
                        # print('Grad', model.down1.maxpool_conv[1].double_conv[0].weight.grad)
                        optimizer.step()

                # Record loss
                with torch.set_grad_enabled(False):
                    running_coarse_loss += coarse_loss.item() * len(true_coarse_sifs_filtered)
                    num_coarse_datapoints += len(true_coarse_sifs_filtered)
                    running_fine_loss += fine_loss.item() * len(true_fine_sifs_filtered)
                    num_fine_datapoints += len(true_fine_sifs_filtered)
                    all_true_fine_sifs.append(true_fine_sifs_filtered.cpu().detach().numpy())
                    all_true_coarse_sifs.append(true_coarse_sifs_filtered.cpu().detach().numpy())
                    all_predicted_coarse_sifs.append(predicted_coarse_sifs_filtered.cpu().detach().numpy())
                    all_predicted_fine_sifs.append(predicted_fine_sifs_filtered.cpu().detach().numpy())

            epoch_coarse_nrmse = math.sqrt(running_coarse_loss / num_coarse_datapoints) / sif_mean                
            epoch_fine_nrmse = math.sqrt(running_fine_loss / num_fine_datapoints) / sif_mean


            true_fine = np.concatenate(all_true_fine_sifs)
            true_coarse = np.concatenate(all_true_coarse_sifs)
            predicted_fine = np.concatenate(all_predicted_fine_sifs)
            predicted_coarse = np.concatenate(all_predicted_coarse_sifs)
            print('===== ', phase, 'Coarse stats ====')
            sif_utils.print_stats(true_coarse, predicted_coarse, sif_mean)
            print('===== ', phase, 'Fine stats ====')
            sif_utils.print_stats(true_fine, predicted_fine, sif_mean)

            if phase == 'train':
                train_coarse_losses.append(epoch_coarse_nrmse)
                train_fine_losses.append(epoch_fine_nrmse)
            else:
                val_coarse_losses.append(epoch_coarse_nrmse)
                val_fine_losses.append(epoch_fine_nrmse)

            # deep copy the model
            if phase == 'val' and epoch_coarse_nrmse < best_coarse_loss:
                best_coarse_loss = epoch_coarse_nrmse
                best_fine_loss = epoch_fine_nrmse
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_FILE)
                # torch.save(model.state_dict(), UNET_MODEL_FILE + "_epoch" + str(epoch))

        # Print elapsed time per epoch
        epoch_time = time.time() - epoch_start
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            epoch_time // 60, epoch_time % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best coarse loss: {:3f}'.format(best_coarse_loss))
    print('Best fine loss: {:3f}'.format(best_fine_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_coarse_losses, val_coarse_losses, train_fine_losses, val_fine_losses


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
print('Number of train tiles', len(train_metadata))
print('Number of val tiles', len(val_metadata))
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

# Constrain predicted SIF to be between certain values (unstandardized), if desired
# Don't forget to standardize
if MIN_SIF is not None and MAX_SIF is not None:
    min_output = (MIN_SIF - sif_mean) / sif_std
    max_output = (MAX_SIF - sif_mean) / sif_std
else:
    min_output = None
    max_output = None

# Set up image transforms / augmentations
standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds)
clip_transform = tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT)
noise_transform = tile_transforms.GaussianNoise(continuous_bands=list(range(0, 9)), standard_deviation=NOISE)
flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()


transform_list_train = [standardize_transform, clip_transform]
transform_list_val = [standardize_transform, clip_transform]
if AUGMENT:
    transform_list_train += [noise_transform]
train_transform = transforms.Compose(transform_list_train)
val_transform = transforms.Compose(transform_list_val)

# Create dataset/dataloaders
datasets = {'train': CFISDataset(train_metadata, train_transform),
            'val': CFISDataset(val_metadata, val_transform)}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS)
               for x in ['train', 'val']}

# Initialize model
if MODEL_TYPE == 'unet_small':
    unet_model = UNetSmall(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'pixel_nn':
    unet_model = simple_cnn.PixelNN(input_channels=INPUT_CHANNELS, output_dim=1, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'unet2':
    unet_model = UNet2(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif MODEL_TYPE == 'unet':
    unet_model = UNet(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)   
else:
    print('Model type not supported')
    exit(1)

# If we're loading a pre-trained model, read model params from file
if FROM_PRETRAINED:
    unet_model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE, map_location=device))

# Initialize loss and optimizer
criterion = nn.MSELoss(reduction='mean')
if OPTIMIZER_TYPE == "Adam":
    optimizer = optim.Adam(unet_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    print("Optimizer not supported")
    exit(1)

unet_model, train_coarse_losses, val_coarse_losses, train_fine_losses, val_fine_losses = train_model(unet_model, dataloaders, criterion, optimizer, device, sif_mean, sif_std, num_epochs=NUM_EPOCHS)
torch.save(unet_model.state_dict(), MODEL_FILE)

# Plot loss curves: NRMSE
epoch_list = range(NUM_EPOCHS)
plots = []
print("Coarse Train NRMSE:", train_coarse_losses)
train_coarse_plot, = plt.plot(epoch_list, train_coarse_losses, color='blue', label='Coarse Train NRMSE')
plots.append(train_coarse_plot)
print("Coarse Val NRMSE:", val_coarse_losses)
val_coarse_plot, = plt.plot(epoch_list, val_coarse_losses, color='green', label='Coarse Val NRMSE')
plots.append(val_coarse_plot)
print("Fine Train NRMSE:", train_fine_losses)
train_fine_plot, = plt.plot(epoch_list, train_fine_losses, color='red', label='Fine Train NRMSE')
plots.append(train_fine_plot)
print("Fine Val NRMSE:", val_fine_losses)
val_fine_plot, = plt.plot(epoch_list, val_fine_losses, color='orange', label='Fine Val NRMSE')
plots.append(val_fine_plot)

# Add legend and axis labels
plt.legend(handles=plots)
plt.xlabel('Epoch #')
plt.ylabel('NRMSE')
plt.savefig(LOSS_PLOT + '_nrmse.png')
plt.close()

# Plot train coarse vs train fine losses
print('============== Train: Fine vs Coarse Losses ===============')
sif_utils.print_stats(train_fine_losses, train_coarse_losses, sif_mean, ax=plt.gca())
plt.xlabel('Train Coarse Losses')
plt.ylabel('Train Fine Losses')
plt.title('Fine vs Coarse train losses' + METHOD)
plt.savefig(LOSS_PLOT + '_scatter_fine_vs_coarse.png')
plt.close()