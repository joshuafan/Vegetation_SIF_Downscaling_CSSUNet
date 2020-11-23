import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import simple_cnn
from reflectance_cover_sif_dataset import CFISDataset, CombinedCfisOco2Dataset
from unet.unet_model import UNet, UNetSmall, UNet2, UNet2PixelEmbedding
import visualization_utils
import nt_xent
import sif_utils
import tile_transforms
import tqdm

# Set random seed
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Folds
TRAIN_FOLDS = [1, 2, 3]
VAL_FOLDS = [4]
TEST_FOLDS = [5]

# Data files
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
OCO2_DIR = os.path.join(DATA_DIR, "OCO2")
# CFIS_TILE_METADATA_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_train.csv')
# CFIS_TILE_METADATA_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_val.csv')
# CFIS_TILE_METADATA_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_test.csv')
# OCO2_TILE_METADATA_TRAIN_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_train.csv')
CFIS_COARSE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_metadata.csv')
CFIS_FINE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_fine_metadata.csv')
BAND_STATISTICS_FILE = os.path.join(CFIS_DIR, 'cfis_band_statistics_train.csv')
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018")
OCO2_METADATA_FILE = os.path.join(DATASET_DIR, 'tropomi_metadata.csv')

# Dataset resolution/scale
RES = (0.00026949458523585647, 0.00026949458523585647)
TILE_PIXELS = 100
TILE_SIZE_DEGREES = RES[0] * TILE_PIXELS

# Method/model type
# METHOD = "9d_unet_contrastive"
# MODEL_TYPE = "unet"
# METHOD = "9d_unet2"
# MODEL_TYPE = "unet2"
# METHOD = "9d_unet2_contrastive"
# MODEL_TYPE = "unet2_pixel_embedding"
# METHOD = "9e_unet2_pixel_contrastive_3" #contrastive_2" #pixel_embedding"
# MODEL_TYPE = "unet2_pixel_embedding"
# METHOD = "9e_unet2_pixel_embedding"
# MODEL_TYPE = "unet2_pixel_embedding"
# METHOD = "9d_pixel_nn"
# MODEL_TYPE = "pixel_nn"
# METHOD = "10d_unet2"
# MODEL_TYPE = "unet2"
# METHOD = "10e_unet2_contrastive"
# MODEL_TYPE = "unet2_pixel_embedding"
# METHOD = "11d_unet2"
# MODEL_TYPE = "unet2"
# METHOD = "2d_unet2"
# MODEL_TYPE = "unet2"
METHOD = "tropomi_cfis_unet2"
MODEL_TYPE = "unet2"
TRAIN_SOURCES = ["OCO2", "CFIS"] #, "OCO2"]
OCO2_UPDATES_PER_CFIS = 1

# Model files
PRETRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD) #9e_unet2_contrastive")
MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD) #aug_2")

# Results files/plots
CFIS_RESULTS_CSV_FILE = os.path.join(CFIS_DIR, 'cfis_results_' + METHOD + '.csv')
LOSS_PLOT = os.path.join(DATA_DIR, 'loss_plots/losses_' + METHOD)


# Parameters
OPTIMIZER_TYPE = "Adam"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 50 # 100
BATCH_SIZE = 16
NUM_WORKERS = 8
FROM_PRETRAINED = False
PRETRAIN_CONTRASTIVE = False
FREEZE_PIXEL_ENCODER = False
MIN_SIF = None
MAX_SIF = None
MIN_SIF_CLIP = 0.1
MIN_INPUT = -3
MAX_INPUT = 3
REDUCED_CHANNELS = 10


# Which bands
BANDS = list(range(0, 43))
# BANDS = list(range(0, 12)) + [12, 13, 14, 16] + [42]
# BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42]
INPUT_CHANNELS = len(BANDS)
CROP_TYPE_INDICES = list(range(12, 42))
MISSING_REFLECTANCE_IDX = -1

# Augmentations
AUGMENTATIONS = ['flip_and_rotate', 'gaussian_noise', 'jigsaw']
RESIZE_DIM = 100
NOISE = 0.01
FRACTION_OUTPUTS_TO_AVERAGE = 0.01

# OCO-2 filtering
MIN_OCO2_SOUNDINGS = 3
MAX_OCO2_CLOUD_COVER = 0.5
OCO2_SCALING_FACTOR = 0.97

# CFIS filtering
MIN_FINE_CFIS_SOUNDINGS = 10
MIN_COARSE_FRACTION_VALID_PIXELS = 0.1

# Dates
TRAIN_DATES = ['2016-06-15', '2016-08-01', "2018-06-10", "2018-06-24", "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19"]
TEST_DATES = ['2016-06-15', '2016-08-01']

# Contrastive training settings
CONTRASTIVE_NUM_EPOCHS = 50
CONTRASTIVE_LEARNING_RATE = 1e-3
CONTRASTIVE_WEIGHT_DECAY = 1e-3
CONTRASTIVE_TEMP = 0.2
PIXEL_PAIRS_PER_IMAGE = 5
CONTRASTIVE_BATCH_SIZE = BATCH_SIZE * PIXEL_PAIRS_PER_IMAGE



# # Print params for reference
# print("=========================== PARAMS ===========================")
# print("Method:", METHOD)
# if FROM_PRETRAINED:
#     print("From pretrained model", os.path.basename(PRETRAINED_MODEL_FILE))
# else:
#     print("Training from scratch")
# print("Output model:", os.path.basename(MODEL_FILE))
# print("Bands:", BANDS)
# print("---------------------------------")
# print("Model:", MODEL_TYPE)
# print("Optimizer:", OPTIMIZER_TYPE)
# print("Learning rate:", LEARNING_RATE)
# print("Weight decay:", WEIGHT_DECAY)
# print("Num workers:", NUM_WORKERS)
# print("Batch size:", BATCH_SIZE)
# print("Num epochs:", NUM_EPOCHS)
# print("Augmentations:", AUGMENTATIONS)
# if 'gaussian_noise' in AUGMENTATIONS:
#     print("Gaussian noise (std deviation):", NOISE)
# print('Fraction of outputs averaged in training:', FRACTION_OUTPUTS_TO_AVERAGE)
# print("Input features clipped to", MIN_INPUT, "to", MAX_INPUT, "standard deviations from mean")
# print("SIF range:", MIN_SIF, "to", MAX_SIF)
# print("==============================================================")


"""
Selects indices of pixels that are within "radius" of each other. #, ideally with the same crop type.
"images" has shape [batch x channels x height x width]
Returns two Tensors, each of shape [batch x pixel_pairs_per_image x 2], where "2" is (height_idx, width_idx)

TODO "radius" is not quite the right terminology
"""
def get_neighbor_pixel_indices(images, radius=10, pixel_pairs_per_image=10, num_tries=3):
    indices1 = torch.zeros((images.shape[0], pixel_pairs_per_image, 2))
    indices2 = torch.zeros((images.shape[0], pixel_pairs_per_image, 2))
    for image_idx in range(images.shape[0]):
        for pair_idx in range(pixel_pairs_per_image):
            # Randomly choose anchor pixel
            for i in range(num_tries):
                anchor_height_idx = np.random.randint(0, images.shape[2])
                anchor_width_idx = np.random.randint(0, images.shape[3])
                if images[image_idx, MISSING_REFLECTANCE_IDX, anchor_height_idx, anchor_width_idx] == 0:
                    break

            # Randomly sample neighbor pixel (within "radius" pixels of anchor)
            min_height = max(0, anchor_height_idx - radius)
            max_height = min(images.shape[2], anchor_height_idx + radius + 1)  # exclusive
            min_width = max(0, anchor_width_idx - radius)
            max_width = min(images.shape[3], anchor_width_idx + radius + 1)  # exclusive
            anchor_crop_type = images[image_idx, CROP_TYPE_INDICES, anchor_height_idx, anchor_width_idx]
            for i in range(num_tries):
                neighbor_height_idx = np.random.randint(min_height, max_height)
                neighbor_width_idx = np.random.randint(min_width, max_width)
                neighbor_crop_type = images[image_idx, CROP_TYPE_INDICES, neighbor_height_idx, neighbor_width_idx]
                if torch.equal(neighbor_crop_type, anchor_crop_type) and images[image_idx, MISSING_REFLECTANCE_IDX, neighbor_height_idx, neighbor_width_idx] == 0:
                    # print('found neighbor')
                    break
                # else:
                    # print('neighbor invalid')

            # Record indices
            indices1[image_idx, pair_idx, 0] = anchor_height_idx
            indices1[image_idx, pair_idx, 1] = anchor_width_idx
            indices2[image_idx, pair_idx, 0] = neighbor_height_idx
            indices2[image_idx, pair_idx, 1] = neighbor_width_idx

    # indices1 = torch.empty((images.shape[0], pixel_pairs_per_image, 2))
    # indices1[:, :, 0] = torch.randint(0, images.shape[2], (indices1.shape[0], indices1.shape[1]))
    # indices1[:, :, 0] = torch.randint(0, images.shape[2], (indices1.shape[0], indices1.shape[1]))

    return indices1, indices2



def train_contrastive(model, dataloader, criterion, optimizer, device, pixel_pairs_per_image, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        train_loss, total = 0, 0
        pbar = tqdm.tqdm(dataloader, total=len(dataloader), desc='train epoch {}'.format(epoch))
        for sample in pbar:
            input_tiles = sample['cfis_input_tile'].to(device)
            _, pixel_embeddings = model(input_tiles)

            indices1, indices2 = get_neighbor_pixel_indices(input_tiles, pixel_pairs_per_image=pixel_pairs_per_image)
            # print(indices1.shape)
            batch_size = indices1.shape[0]
            assert batch_size == input_tiles.shape[0]
            assert batch_size == indices2.shape[0]
            num_points = indices1.shape[1]
            assert num_points == indices2.shape[1]
            indices1 = indices1.reshape((batch_size * num_points, 2)).long()
            indices2 = indices2.reshape((batch_size * num_points, 2)).long()
            # print('indices1 shape', indices1.shape)
            bselect = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, num_points).flatten()
            # print('Bselect shape', bselect.shape)

            embeddings1 = pixel_embeddings[bselect, :, indices1[:, 0], indices1[:, 1]]
            embeddings2 = pixel_embeddings[bselect, :, indices2[:, 0], indices2[:, 1]]
            # print('Embeddings shape', embeddings1.shape)  # [batch_size * num_points, reduced_dim]

            #normalize
            embeddings1 = F.normalize(embeddings1, dim=1)
            embeddings2 = F.normalize(embeddings2, dim=1)

            loss = criterion(embeddings1, embeddings2)
            # output_perm = torch.cat((output1[2:], output1[:2]))
            #loss, l_n, l_d = triplet_loss(output1, output2, output_perm, margin=args.triplet_margin, l2=args.triplet_l2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += batch_size
            pbar.set_postfix(avg_loss=train_loss/total)
    return model


def train_model(model, dataloaders, criterion, optimizer, device, sif_mean, sif_std, num_epochs=100):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_coarse_loss = float('inf')
    best_val_coarse_loss = float('inf')
    best_train_fine_loss = float('inf')
    best_val_fine_loss = float('inf')
    train_coarse_losses = []
    val_coarse_losses = []
    train_fine_losses = []
    val_fine_losses = []
    train_oco2_losses = []

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
            running_oco2_loss = 0
            num_oco2_datapoints = 0
            batch_idx = 0
            all_true_fine_sifs = []
            all_true_coarse_sifs = []
            all_true_oco2_sifs = []
            all_predicted_fine_sifs = []
            all_predicted_coarse_sifs = []
            all_predicted_oco2_sifs = []

            # Iterate over data.
            for sample in dataloaders[phase]:
                if 'cfis_input_tile' in sample:
                    # Read input tile
                    input_tiles_std = sample['cfis_input_tile'][:, BANDS, :, :].to(device)

                    # Read coarse-resolution SIF label
                    true_coarse_sifs = sample['cfis_coarse_sif'].to(device)              

                    # Read fine-resolution SIF labels
                    true_fine_sifs = sample['cfis_fine_sif'].to(device)
                    valid_fine_sif_mask = torch.logical_not(sample['cfis_fine_sif_mask']).to(device)
                    fine_soundings = sample['cfis_fine_soundings'].to(device)

                    # print('True fine SIFs', true_fine_sifs[0, 0:4, 0:4])
                    # print('Valid fine sif mask', valid_fine_sif_mask[0, 0:4, 0:4])
                    # print('Fine soundings', fine_soundings[0, 0:4, 0:4])


                    with torch.set_grad_enabled(phase == 'train'):
                        predicted_fine_sifs_std = model(input_tiles_std)  # predicted_fine_sifs_std: (batch size, 1, H, W)
                        if type(predicted_fine_sifs_std) == tuple:
                            predicted_fine_sifs_std = predicted_fine_sifs_std[0]
                        predicted_fine_sifs_std = torch.squeeze(predicted_fine_sifs_std, dim=1)
                        predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean
                        # print('Valid', torch.mean(valid_fine_sif_mask.float()))

                        # As a regularization technique, randomly choose more pixels to ignore.
                        if phase == 'train':
                            pixels_to_include = torch.rand(valid_fine_sif_mask.shape, device=device) > (1 - FRACTION_OUTPUTS_TO_AVERAGE)
                            valid_fine_sif_mask = valid_fine_sif_mask * pixels_to_include
                            # print('after random selection valid', torch.mean(valid_fine_sif_mask.float()))

                        # For each tile, take the average SIF over all valid pixels
                        predicted_coarse_sifs = sif_utils.masked_average(predicted_fine_sifs, valid_fine_sif_mask, dims_to_average=(1, 2)) # (batch size)

                        # Zero out predicted SIFs for invalid pixels (pixels with no valid SIF label, or cloudy pixels).
                        # Now, only VALID pixels contain a non-zero predicted SIF.
                        # predicted_fine_sifs[valid_fine_sif_mask == 0] = 0

                        # TODO test if this computation is correct (compare against for-loop)

                        # # For each coarse-SIF sub-region, compute the fraction of valid pixels.
                        # # Each square is: (# valid fine pixels) / (# total fine pixels)
                        # avg_pool = nn.AvgPool2d(kernel_size=COARSE_SIF_PIXELS)
                        # fraction_valid = avg_pool(valid_fine_sif_mask.float())

                        # # Average together fine SIF predictions for each coarse SIF area.
                        # # Each square is: (sum predicted SIF over valid fine pixels) / (# total fine pixels)
                        # predicted_coarse_sifs = avg_pool(predicted_fine_sifs)
                        # # print('Predicted coarse sifs', predicted_coarse_sifs.shape)
                        # # print('Fraction valid', fraction_valid.shape)

                        # # Instead of dividing by the total number of fine pixels, divide by the number of VALID fine pixels.
                        # # Each square is now: (sum predicted SIF over valid fine pixels) / (# valid fine pixels), which is what we want.
                        # predicted_coarse_sifs = predicted_coarse_sifs / fraction_valid
                        # predicted_coarse_sifs[valid_coarse_sif_mask == 0] = 0

                        # Plot example tile 
                        # large_tile_lat = sample['cfis_lat'][0].item()
                        # large_tile_lon = sample['cfis_lon'][0].item()
                        # date = sample['cfis_date'][0]
                        # sif_tiles = [true_coarse_sifs[0].cpu().detach().numpy(),
                        #              true_fine_sifs[0].cpu().detach().numpy(),
                        #              predicted_coarse_sifs[0].cpu().detach().numpy(),
                        #              predicted_fine_sifs[0].cpu().detach().numpy()]
                        # plot_names = ['True coarse SIF', 'True fine SIF', 'Predicted coarse SIF', 'Predicted fine SIF']
                        # cdl_utils.plot_tile(input_tiles_std[0].cpu().detach().numpy(),
                        #         sif_tiles, plot_names, large_tile_lon, large_tile_lat, date, TILE_SIZE_DEGREES)
                        # exit(0)

                        # cdl_utils.plot_tile(input_tiles_std[0].cpu().detach().numpy(), 
                        #                     true_coarse_sifs[0].cpu().detach().numpy(),
                        #                     true_fine_sifs[0].cpu().detach().numpy(),
                        #                     [predicted_coarse_sifs[0].cpu().detach().numpy()],
                        #                     [predicted_fine_sifs[0].cpu().detach().numpy()],
                        #                     ['UNet'], large_tile_lon, large_tile_lat, date,
                        #                     TILE_SIZE_DEGREES)

                        # Extract the coarse SIF data points where we have labels, and compute loss
                        # valid_coarse_sif_mask = valid_coarse_sif_mask.flatten()
                        # true_coarse_sifs_filtered = true_coarse_sifs.flatten()[valid_coarse_sif_mask]
                        # predicted_coarse_sifs_filtered = predicted_coarse_sifs.flatten()[valid_coarse_sif_mask]

                        # Compute loss (predicted vs true coarse SIF)
                        coarse_loss = criterion(true_coarse_sifs, predicted_coarse_sifs)

                        # Extract the fine SIF data points where we have labels, and compute loss
                        non_noisy_mask = (fine_soundings >= MIN_FINE_CFIS_SOUNDINGS) & (true_fine_sifs >= MIN_SIF_CLIP)
                        valid_fine_sif_mask_flat = valid_fine_sif_mask.flatten() & non_noisy_mask.flatten()
                        true_fine_sifs_filtered = true_fine_sifs.flatten()[valid_fine_sif_mask_flat]
                        predicted_fine_sifs_filtered = predicted_fine_sifs.flatten()[valid_fine_sif_mask_flat]
                        predicted_fine_sifs_filtered = torch.clamp(predicted_fine_sifs_filtered, min=MIN_SIF_CLIP)
                        fine_loss = criterion(true_fine_sifs_filtered, predicted_fine_sifs_filtered)

                        # Backpropagate coarse loss
                        if phase == 'train': # and not np.isnan(fine_loss.item()):
                            optimizer.zero_grad()
                            # fine_loss.backward()
                            coarse_loss.backward()
                            # print('Grad', model.down1.maxpool_conv[1].double_conv[0].weight.grad)
                            optimizer.step()

                    # Record loss
                    with torch.set_grad_enabled(False):
                        running_coarse_loss += coarse_loss.item() * len(true_coarse_sifs)
                        num_coarse_datapoints += len(true_coarse_sifs)
                        if not np.isnan(fine_loss.item()):
                            running_fine_loss += fine_loss.item() * len(true_fine_sifs_filtered)
                            num_fine_datapoints += len(true_fine_sifs_filtered)
                        all_true_fine_sifs.append(true_fine_sifs_filtered.cpu().detach().numpy())
                        all_true_coarse_sifs.append(true_coarse_sifs.cpu().detach().numpy())
                        all_predicted_fine_sifs.append(predicted_fine_sifs_filtered.cpu().detach().numpy())
                        all_predicted_coarse_sifs.append(predicted_coarse_sifs.cpu().detach().numpy())


                # Additional training with OCO-2 SIF
                if 'oco2_input_tile' in sample:
                    if (phase == 'val') or (phase == 'train' and random.random() < OCO2_UPDATES_PER_CFIS):
                        # Read OCO-2 input tile and SIF label
                        oco2_tiles_std = sample['oco2_input_tile'][:, BANDS, :, :].to(device)
                        oco2_true_sifs = sample['oco2_sif'].to(device)

                        with torch.set_grad_enabled(phase == 'train'):
                            predicted_fine_sifs_std = model(oco2_tiles_std)  # predicted_fine_sifs_std: (batch size, 1, H, W)
                            if type(predicted_fine_sifs_std) == tuple:
                                predicted_fine_sifs_std = predicted_fine_sifs_std[0]
                            predicted_fine_sifs_std = torch.squeeze(predicted_fine_sifs_std, dim=1)
                            predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean

                            # Binary mask for non-cloudy pixels
                            non_cloudy_pixels = torch.logical_not(oco2_tiles_std[:, MISSING_REFLECTANCE_IDX, :, :])  # (batch size, H, W)
                            # print('Percent noncloudy', torch.mean(non_cloudy_pixels, dim=(1, 2)))

                            # As a regularization technique, randomly choose more pixels to ignore.
                            if phase == 'train':
                                pixels_to_include = torch.rand(non_cloudy_pixels.shape, device=device) > (1 - FRACTION_OUTPUTS_TO_AVERAGE)
                                non_cloudy_pixels = non_cloudy_pixels * pixels_to_include

                            # oco2_predicted_sifs = torch.mean(predicted_fine_sifs, dim=(1,2))
                            oco2_predicted_sifs = sif_utils.masked_average(predicted_fine_sifs, non_cloudy_pixels, dims_to_average=(1, 2)) # (batch size)
                            # print('oco2 predicted sifs', oco2_predicted_sifs)
                            # print('oco2 true sifs', oco2_true_sifs)
                            # Compute loss: predicted vs true SIF (standardized)
                            oco2_predicted_sifs = torch.clamp(oco2_predicted_sifs, min=MIN_SIF_CLIP)

                            # Compute loss: predicted vs true SIF
                            oco2_loss = criterion(oco2_predicted_sifs, oco2_true_sifs)
                            if phase == 'train':
                                optimizer.zero_grad()
                                oco2_loss.backward()
                                optimizer.step()

                        with torch.set_grad_enabled(False):
                            running_oco2_loss += oco2_loss.item() * len(oco2_true_sifs)
                            num_oco2_datapoints += len(oco2_true_sifs)
                            all_true_oco2_sifs.append(oco2_true_sifs.cpu().detach().numpy())
                            all_predicted_oco2_sifs.append(oco2_predicted_sifs.cpu().detach().numpy())


            if num_coarse_datapoints > 0:
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
            else:
                print(phase, 'no CFIS datapoints!')

            if num_oco2_datapoints > 0:
                epoch_oco2_nrmse = math.sqrt(running_oco2_loss / num_oco2_datapoints) / sif_mean

            if phase == 'train' and num_oco2_datapoints > 0:
                true_oco2 = np.concatenate(all_true_oco2_sifs)
                predicted_oco2 = np.concatenate(all_predicted_oco2_sifs)
                print('======', phase, 'OCO2 stats =====')
                sif_utils.print_stats(true_oco2, predicted_oco2, sif_mean)


            if phase == 'train':
                if num_coarse_datapoints > 0:
                    train_coarse_losses.append(epoch_coarse_nrmse)
                    train_fine_losses.append(epoch_fine_nrmse)
                if num_oco2_datapoints > 0:
                    train_oco2_losses.append(epoch_oco2_nrmse)
            else:
                val_coarse_losses.append(epoch_coarse_nrmse)
                val_fine_losses.append(epoch_fine_nrmse)


            # deep copy the model
            if phase == 'val' and epoch_coarse_nrmse < best_val_coarse_loss:
                best_val_coarse_loss = epoch_coarse_nrmse
                best_val_fine_loss = epoch_fine_nrmse
                best_train_coarse_loss = train_coarse_losses[-1] if len(train_coarse_losses) > 0 else None
                best_train_fine_loss = train_fine_losses[-1] if len(train_fine_losses) > 0 else None

                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_FILE)
                # torch.save(model.state_dict(), UNET_MODEL_FILE + "_epoch" + str(epoch))
            
            # if phase == 'train' and epoch_coarse_nrmse < best_train_coarse_loss:
            #     best_train_coarse_loss = epoch_coarse_nrmse
            #     torch.save(model.state_dict(), MODEL_FILE + '_best_train')


        # Print elapsed time per epoch
        epoch_time = time.time() - epoch_start
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            epoch_time // 60, epoch_time % 60))
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best train coarse loss: {:.3f}'.format(best_train_coarse_loss))
    print('Best train fine loss: {:.3f}'.format(best_train_fine_loss))
    print('Best val coarse loss: {:.3f}'.format(best_val_coarse_loss))
    print('Best val fine loss: {:.3f}'.format(best_val_fine_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_coarse_losses, val_coarse_losses, train_fine_losses, val_fine_losses, train_oco2_losses


# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)


# Filter OCO2 tiles
oco2_metadata = pd.read_csv(OCO2_METADATA_FILE)
oco2_metadata = oco2_metadata[(oco2_metadata['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                                (oco2_metadata['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                                (oco2_metadata['SIF'] >= MIN_SIF_CLIP) &
                                (oco2_metadata['source'] == 'TROPOMI')]

cfis_coarse_metadata = pd.read_csv(CFIS_COARSE_METADATA_FILE)

# Only include CFIS tiles with enough valid pixels
cfis_coarse_metadata = cfis_coarse_metadata[(cfis_coarse_metadata['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                    (cfis_coarse_metadata['SIF'] >= MIN_SIF_CLIP)]

# Read fine metadata at particular resolution
cfis_fine_metadata = pd.read_csv(CFIS_FINE_METADATA_FILE)
cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['SIF'] >= MIN_SIF_CLIP) &
                                (cfis_fine_metadata['tile_file'].isin(set(cfis_coarse_metadata['tile_file'])))]

# Read dataset splits
oco2_train_set = oco2_metadata[(oco2_metadata['fold'].isin(TRAIN_FOLDS)) &
                                (oco2_metadata['date'].isin(TRAIN_DATES))].copy()
oco2_val_set = oco2_metadata[(oco2_metadata['fold'].isin(VAL_FOLDS)) &
                                (oco2_metadata['date'].isin(TRAIN_DATES))].copy()
oco2_test_set = oco2_metadata[(oco2_metadata['fold'].isin(TEST_FOLDS)) &
                                (oco2_metadata['date'].isin(TEST_DATES))].copy()
coarse_train_set = cfis_coarse_metadata[(cfis_coarse_metadata['fold'].isin(TRAIN_FOLDS)) &
                                        (cfis_coarse_metadata['date'].isin(TRAIN_DATES))].copy()
coarse_val_set = cfis_coarse_metadata[(cfis_coarse_metadata['fold'].isin(VAL_FOLDS)) &
                                        (cfis_coarse_metadata['date'].isin(TRAIN_DATES))].copy()
coarse_test_set = cfis_coarse_metadata[(cfis_coarse_metadata['fold'].isin(TEST_FOLDS)) &
                                        (cfis_coarse_metadata['date'].isin(TEST_DATES))].copy()
fine_train_set = cfis_fine_metadata[(cfis_fine_metadata['fold'].isin(TRAIN_FOLDS)) &
                                        (cfis_fine_metadata['date'].isin(TRAIN_DATES))].copy()
fine_val_set = cfis_fine_metadata[(cfis_fine_metadata['fold'].isin(VAL_FOLDS)) &
                                        (cfis_fine_metadata['date'].isin(TRAIN_DATES))].copy()
fine_test_set = cfis_fine_metadata[(cfis_fine_metadata['fold'].isin(TEST_FOLDS)) &
                                        (cfis_fine_metadata['date'].isin(TEST_DATES))].copy()

# Read train/val tile metadata
if 'CFIS' in TRAIN_SOURCES:
    cfis_train_metadata = coarse_train_set
else:
    cfis_train_metadata = None
if 'OCO2' in TRAIN_SOURCES:
    oco2_train_metadata = oco2_train_set
else:
    oco2_train_metadata = None
cfis_val_metadata = coarse_val_set


# if 'CFIS' in TRAIN_SOURCES:
#     cfis_train_metadata = pd.read_csv(CFIS_TILE_METADATA_TRAIN_FILE)
# else:
#     cfis_train_metadata = None

# if 'OCO2' in TRAIN_SOURCES:
#     oco2_train_metadata = pd.read_csv(OCO2_TILE_METADATA_TRAIN_FILE)
#     # Filter OCO2 sets
#     oco2_train_metadata = oco2_train_metadata[(oco2_train_metadata['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
#                                               (oco2_train_metadata['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
#                                               (oco2_train_metadata['SIF'] >= MIN_SIF_CLIP) &
#                                               (oco2_train_metadata['date'].isin(TRAIN_DATES))]
#     oco2_train_metadata['SIF'] = oco2_train_metadata['SIF'] * OCO2_SCALING_FACTOR
#     print('Number of OCO2 train tiles', len(oco2_train_metadata))
# else:
#     oco2_train_metadata = None

# cfis_val_metadata = pd.read_csv(CFIS_TILE_METADATA_VAL_FILE)

# # Filter CFIS sets
# cfis_train_metadata = cfis_train_metadata[(cfis_train_metadata['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
#                                           (cfis_train_metadata['SIF'] >= MIN_SIF_CLIP) &
#                                           (cfis_train_metadata['date'].isin(TRAIN_DATES))]
# cfis_val_metadata = cfis_val_metadata[(cfis_val_metadata['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
#                                       (cfis_val_metadata['SIF'] >= MIN_SIF_CLIP) &
#                                       (cfis_val_metadata['date'].isin(TRAIN_DATES))]
# print('Number of CFIS train tiles', len(cfis_train_metadata))
# print('Number of CFIS val tiles', len(cfis_val_metadata))

# Print params for reference
print("=========================== PARAMS ===========================")
PARAM_STRING = ''
PARAM_STRING += '============= DATASET PARAMS =============\n'
PARAM_STRING += ('CFIS Dataset dir: ' + CFIS_DIR + '\n')
PARAM_STRING += ('Train sources: ' + str(TRAIN_SOURCES) + '\n')
if 'CFIS' in TRAIN_SOURCES:
    PARAM_STRING += ('Num CFIS tiles (train): ' + str(len(cfis_train_metadata)) + '\n')
if 'OCO2' in TRAIN_SOURCES:
    PARAM_STRING += ('Num OCO-2 tiles: ' + str(len(oco2_train_metadata)) + '; OCO-2 updates per CFIS: ' + str(OCO2_UPDATES_PER_CFIS) + '\n')
PARAM_STRING += ('Train dates: ' + str(TRAIN_DATES) + '\n')
PARAM_STRING += ('Test dates: ' + str(TEST_DATES) + '\n')
PARAM_STRING += ('Min soundings (fine CFIS): ' + str(MIN_FINE_CFIS_SOUNDINGS) + '\n')
PARAM_STRING += ('Min SIF clip: ' + str(MIN_SIF_CLIP) + '\n')
PARAM_STRING += ('Min fraction valid pixels in CFIS tile: ' + str(MIN_COARSE_FRACTION_VALID_PIXELS) + '\n')
PARAM_STRING += ('Train features: ' + str(BANDS) + '\n')
PARAM_STRING += ("Clip input features: " + str(MIN_INPUT) + " to " + str(MAX_INPUT) + " standard deviations from mean\n")
# if REMOVE_PURE_TRAIN:
#     PARAM_STRING += ('Removing pure train tiles above ' + str(PURE_THRESHOLD_TRAIN) + '\n')
PARAM_STRING += ('================= METHOD ===============\n')
if FROM_PRETRAINED:
    PARAM_STRING += ('From pretrained model: ' + os.path.basename(PRETRAINED_MODEL_FILE) + '\n')
else:
    PARAM_STRING += ("Training from scratch\n")
PARAM_STRING += ("Model name: " + os.path.basename(MODEL_FILE) + '\n')
PARAM_STRING += ("Model type: " + MODEL_TYPE + '\n')
PARAM_STRING += ("Optimizer: " + OPTIMIZER_TYPE + '\n')
PARAM_STRING += ("Learning rate: " + str(LEARNING_RATE) + '\n')
PARAM_STRING += ("Weight decay: " + str(WEIGHT_DECAY) + '\n')
PARAM_STRING += ("Num workers: " + str(NUM_WORKERS) + '\n')
PARAM_STRING += ("Batch size: " + str(BATCH_SIZE) + '\n')
PARAM_STRING += ("Num epochs: " + str(NUM_EPOCHS) + '\n')
PARAM_STRING += ("Augmentations: " + str(AUGMENTATIONS) + '\n')
if 'resize' in AUGMENTATIONS:
    PARAM_STRING += ('Resize images to: ' + str(RESIZE_DIM) + '\n')
if 'gaussian_noise' in AUGMENTATIONS:
    PARAM_STRING += ("Gaussian noise (std deviation): " + str(NOISE) + '\n')
PARAM_STRING += ("Fraction outputs to average: " + str(FRACTION_OUTPUTS_TO_AVERAGE) + '\n')
PARAM_STRING += ("SIF range: " + str(MIN_SIF) + " to " + str(MAX_SIF) + '\n')
if PRETRAIN_CONTRASTIVE:
    PARAM_STRING += ('============ CONTRASTIVE PARAMS ===========\n')
    PARAM_STRING += ("Learning rate: " + str(CONTRASTIVE_LEARNING_RATE) + '\n')
    PARAM_STRING += ("Weight decay: " + str(CONTRASTIVE_WEIGHT_DECAY) + '\n')
    PARAM_STRING += ("Batch size: " + str(CONTRASTIVE_BATCH_SIZE) + '\n')
    PARAM_STRING += ("Num epochs: " + str(CONTRASTIVE_NUM_EPOCHS) + '\n')
    PARAM_STRING += ("Temperature: " + str(CONTRASTIVE_TEMP) + '\n')
PARAM_STRING += ("==============================================================\n")
print(PARAM_STRING)

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
standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds) #, min_input=MIN_INPUT, max_input=MAX_INPUT)
clip_transform = tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT)
color_distortion_transform = tile_transforms.ColorDistortion(continuous_bands=list(range(0, 12)), standard_deviation=NOISE)
noise_transform = tile_transforms.GaussianNoise(continuous_bands=list(range(0, 9)), standard_deviation=NOISE)
flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()
jigsaw_transform = tile_transforms.RandomJigsaw()
resize_transform = tile_transforms.ResizeTile(target_dim=[RESIZE_DIM, RESIZE_DIM])

transform_list_train = [standardize_transform, clip_transform] # [standardize_transform, noise_transform]
transform_list_val = [standardize_transform, clip_transform] #[standardize_transform]
transform_list_cfis = [standardize_transform, clip_transform]
if 'resize' in AUGMENTATIONS:
    transform_list_train.append(resize_transform)
if 'flip_and_rotate' in AUGMENTATIONS:
    transform_list_train.append(flip_and_rotate_transform)
if 'gaussian_noise' in AUGMENTATIONS:
    transform_list_train.append(noise_transform)
if 'jigsaw' in AUGMENTATIONS:
    transform_list_train.append(jigsaw_transform)
train_transform = transforms.Compose(transform_list_train)
val_transform = transforms.Compose(transform_list_val)

# # Set up image transforms / augmentations
# standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds)
# clip_transform = tile_transforms.ClipTile(min_input=MIN_INPUT, max_input=MAX_INPUT)
# noise_transform = tile_transforms.GaussianNoise(continuous_bands=list(range(0, 9)), standard_deviation=NOISE)
# flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()

# transform_list_train = [standardize_transform, clip_transform]
# transform_list_val = [standardize_transform, clip_transform]
# if AUGMENT:
#     transform_list_train += [flip_and_rotate_transform, noise_transform]
# train_transform = transforms.Compose(transform_list_train)
# val_transform = transforms.Compose(transform_list_val)

# Create dataset/dataloaders
datasets = {'train': CombinedCfisOco2Dataset(cfis_train_metadata, oco2_train_metadata, train_transform, MIN_FINE_CFIS_SOUNDINGS),
            'val': CombinedCfisOco2Dataset(cfis_val_metadata, None, val_transform, MIN_FINE_CFIS_SOUNDINGS)}
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
elif MODEL_TYPE == 'unet2_pixel_embedding':
    unet_model = UNet2PixelEmbedding(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
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

if PRETRAIN_CONTRASTIVE:
    contrastive_loss = nt_xent.NTXentLoss(device, CONTRASTIVE_BATCH_SIZE, CONTRASTIVE_TEMP, True)
    contrastive_optimizer = optim.Adam(unet_model.parameters(), lr=CONTRASTIVE_LEARNING_RATE, weight_decay=CONTRASTIVE_WEIGHT_DECAY)
    contrastive_dataloader = torch.utils.data.DataLoader(datasets['train'], batch_size=BATCH_SIZE, 
                                                         shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    print('Contrastive training!')
    # Train pixel embedding
    unet_model = train_contrastive(unet_model, contrastive_dataloader, contrastive_loss, contrastive_optimizer, device,
                                   pixel_pairs_per_image=PIXEL_PAIRS_PER_IMAGE, num_epochs=CONTRASTIVE_NUM_EPOCHS)

# Freeze pixel embedding layers
if FREEZE_PIXEL_ENCODER:
    unet_model.dimensionality_reduction.requires_grad = False
    unet_model.inc.requires_grad = False
    # unet_model.dimensionality_reduction_2.requires_grad = False

# Train model to predict SIF
unet_model, train_coarse_losses, val_coarse_losses, train_fine_losses, val_fine_losses, train_oco2_losses = train_model(unet_model, dataloaders, criterion, optimizer, device, sif_mean, sif_std, num_epochs=NUM_EPOCHS)
torch.save(unet_model.state_dict(), MODEL_FILE)

# Plot loss curves: NRMSE
epoch_list = range(NUM_EPOCHS)
plots = []
print("Coarse Train NRMSE:", train_coarse_losses)
train_coarse_plot, = plt.plot(epoch_list, train_coarse_losses, color='blue', label='Coarse Train NRMSE')
plots.append(train_coarse_plot)
if len(train_oco2_losses) > 0:
    print("OCO2 Train NRMSE:", train_oco2_losses)
    train_oco2_plot, = plt.plot(epoch_list, train_oco2_losses, color='purple', label='OCO2 Train NRMSE')
    plots.append(train_oco2_plot)
print("Coarse Val NRMSE:", val_coarse_losses)
val_coarse_plot, = plt.plot(epoch_list, val_coarse_losses, color='green', label='Coarse Val NRMSE')
plots.append(val_coarse_plot)
print("Fine Train NRMSE:", train_fine_losses)
train_fine_plot, = plt.plot(epoch_list, train_fine_losses, color='red', label='Fine Train (Interpolated) NRMSE')
plots.append(train_fine_plot)
print("Fine Val NRMSE:", val_fine_losses)
val_fine_plot, = plt.plot(epoch_list, val_fine_losses, color='orange', label='Fine Val (Interpolated) NRMSE')
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

