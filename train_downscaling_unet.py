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
from reflectance_cover_sif_dataset import CombinedDataset, CoarseSIFDataset, FineSIFDataset
from unet.unet_model import UNet, UNetSmall, UNet2, UNet2PixelEmbedding, UNet2Larger
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
TRAIN_FOLDS = [0, 1, 2]
VAL_FOLDS = [3]
TEST_FOLDS = [4]

# Data files
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_2016_DIR = os.path.join(DATA_DIR, "CFIS")
OCO2_2016_DIR = os.path.join(DATA_DIR, "OCO2")
DATASET_2018_DIR = os.path.join(DATA_DIR, "dataset_2018")
# CFIS_TILE_METADATA_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_train.csv')
# CFIS_TILE_METADATA_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_val.csv')
# CFIS_TILE_METADATA_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_test.csv')
# OCO2_TILE_METADATA_TRAIN_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_train.csv')
# CFIS_FINE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_fine_metadata.csv')

# CFIS_COARSE_METADATA_FILE = os.path.join(CFIS_2016_DIR, 'cfis_coarse_metadata.csv')
# OCO2_2016_METADATA_FILE = os.path.join(OCO2_2016_DIR, 'oco2_metadata.csv')
# OCO2_2018_METADATA_FILE = os.path.join(DATASET_2018_DIR, 'oco2_metadata.csv')
# TROPOMI_2018_METADATA_FILE = os.path.join(DATASET_2018_DIR, 'tropomi_metadata.csv')

DATASET_FILES = {'CFIS_2016': os.path.join(CFIS_2016_DIR, 'cfis_coarse_metadata.csv'),
                 'OCO2_2016': os.path.join(OCO2_2016_DIR, 'oco2_metadata_overlap.csv'),
                 'OCO2_2018': os.path.join(DATASET_2018_DIR, 'oco2_metadata.csv'),
                 'TROPOMI_2018': os.path.join(DATASET_2018_DIR, 'tropomi_metadata.csv')}
COARSE_SIF_DATASETS = {'train': ['CFIS_2016', 'OCO2_2016'], #, 'TROPOMI_2018'], #, 'TROPOMI_2018'], # 'OCO2_2016', 'OCO2_2018', 'TROPOMI_2018'],
                       'val': ['CFIS_2016']}
FINE_SIF_DATASETS = {'train': ['CFIS_2016'],
                     'val': ['CFIS_2016']}
MODEL_SELECTION_DATASET = 'CFIS_2016'
UPDATE_FRACTIONS = {'CFIS_2016': 1,
                    'OCO2_2016': 1,
                    'OCO2_2018': 0,
                    'TROPOMI_2018': 0}
BAND_STATISTICS_FILE = os.path.join(CFIS_2016_DIR, 'cfis_band_statistics_train.csv')


# Dataset resolution/scale
RES = (0.00026949458523585647, 0.00026949458523585647)
TILE_PIXELS = 100
TILE_SIZE_DEGREES = RES[0] * TILE_PIXELS

# Method/model type
# METHOD = "9d_unet" #_contrastive"
# MODEL_TYPE = "unet"
# METHOD = "9d_unet2_local"
# MODEL_TYPE = "unet2"
# METHOD = "9d_unet2_larger"
# MODEL_TYPE = "unet2_larger"
# METHOD = "9d_unet2_pixel_embedding"
# MODEL_TYPE = "unet2_pixel_embedding"
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
METHOD = "10d_unet2_larger_dropout"
MODEL_TYPE = "unet2_larger"
# METHOD = "10e_unet2_contrastive"
# MODEL_TYPE = "unet2_pixel_embedding"
# METHOD = "11d_unet2"
# MODEL_TYPE = "unet2"
# METHOD = "11d_unet2_pixel_embedding"
# MODEL_TYPE = "unet2_pixel_embedding"
# METHOD = "11d_pixel_nn"
# MODEL_TYPE = "pixel_nn"
# METHOD = "2d_unet2"
# MODEL_TYPE = "unet2"
# METHOD = "tropomi_cfis_unet2"
# MODEL_TYPE = "unet"


# Model files
PRETRAINED_MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD) #9e_unet2_contrastive")
MODEL_FILE = os.path.join(DATA_DIR, "models/" + METHOD) #aug_2")

# Results files/plots
CFIS_RESULTS_CSV_FILE = os.path.join(CFIS_2016_DIR, 'cfis_results_' + METHOD + '.csv')
LOSS_PLOT = os.path.join(DATA_DIR, 'loss_plots/losses_' + METHOD)


# Parameters
OPTIMIZER_TYPE = "Adam"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 50
NUM_WORKERS = 8
FROM_PRETRAINED = False
CROP_TYPE_LOSS = False
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
# BANDS = list(range(0, 9)) + list(range(12, 42))
# BANDS = list(range(0, 12)) + [12, 13, 14, 16] + [42]
# BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42]
INPUT_CHANNELS = len(BANDS)
CROP_TYPE_INDICES = list(range(12, 42))
MISSING_REFLECTANCE_IDX = len(BANDS) - 1

# Augmentations
AUGMENTATIONS = ['cutout', 'flip_and_rotate', 'gaussian_noise', 'jigsaw']
RESIZE_DIM = 100
CROP_DIM = 80
NOISE = 0.1
FRACTION_OUTPUTS_TO_AVERAGE = 0.5
CUTOUT_DIM = 25
CUTOUT_PROB = 1

# OCO-2 filtering
MIN_OCO2_SOUNDINGS = 3
MAX_OCO2_CLOUD_COVER = 0.5
OCO2_SCALING_FACTOR = 0.97

# CFIS filtering
MIN_FINE_CFIS_SOUNDINGS = 10
MIN_COARSE_FRACTION_VALID_PIXELS = 0.2
MAX_CFIS_CLOUD_COVER = 0.5

# Dates
TRAIN_DATES = ['2016-06-15', '2016-08-01'] #, '2016-08-01', "2018-06-10", "2018-06-24", "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19"]
TEST_DATES = ['2016-06-15', '2016-08-01'] #['2016-06-15', '2016-08-01']

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
        for combined_sample in pbar:
            for dataset_name, sample in combined_sample.items():

                input_tiles = sample['input_tile'].to(device)
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
    best_train_coarse_loss = {k: float('inf') for k in COARSE_SIF_DATASETS['train']}
    best_val_coarse_loss = {k: float('inf') for k in COARSE_SIF_DATASETS['val']}
    best_train_fine_loss = {k: float('inf') for k in FINE_SIF_DATASETS['train']}
    best_val_fine_loss = {k: float('inf') for k in FINE_SIF_DATASETS['val']}
    train_coarse_losses = {k: [] for k in COARSE_SIF_DATASETS['train']}
    val_coarse_losses = {k: [] for k in COARSE_SIF_DATASETS['val']}
    train_fine_losses = {k: [] for k in FINE_SIF_DATASETS['train']}
    val_fine_losses = {k: [] for k in FINE_SIF_DATASETS['val']}

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

            running_coarse_loss = {k: 0 for k in COARSE_SIF_DATASETS[phase]}
            num_coarse_datapoints = {k: 0 for k in COARSE_SIF_DATASETS[phase]}
            all_true_coarse_sifs = {k: [] for k in COARSE_SIF_DATASETS[phase]}
            all_predicted_coarse_sifs = {k: [] for k in COARSE_SIF_DATASETS[phase]}
            running_fine_loss = {k: 0 for k in FINE_SIF_DATASETS[phase]}
            num_fine_datapoints = {k: 0 for k in FINE_SIF_DATASETS[phase]}
            all_true_fine_sifs = {k: [] for k in FINE_SIF_DATASETS[phase]}
            all_predicted_fine_sifs = {k: [] for k in FINE_SIF_DATASETS[phase]}

            # Iterate over data.
            for combined_sample in dataloaders[phase]:
                # Loop through all datasets in this sample
                for dataset_name, sample in combined_sample.items():
                    # if (phase == 'val') or (phase == 'train' and random.random() < UPDATE_FRACTIONS[dataset_name]):
                    # Read input tile
                    input_tiles_std = sample['input_tile'][:, BANDS, :, :].to(device)

                    # Read coarse-resolution SIF label
                    true_coarse_sifs = sample['coarse_sif'].to(device)

                    # Read "mask" of which fine SIF pixels are valid and should be used in averaging.
                    # If the mask doesn't exist, use the Landsat quality mask.
                    if 'fine_sif_mask' in sample:
                        valid_fine_sif_mask = torch.logical_not(sample['fine_sif_mask']).to(device)
                    else:
                        valid_fine_sif_mask = torch.logical_not(input_tiles_std[:, MISSING_REFLECTANCE_IDX, :, :])
                    valid_mask_numpy = valid_fine_sif_mask.cpu().detach().numpy() 

                    with torch.set_grad_enabled(phase == 'train'):
                        # Pass tile through model to obtain fine-resolution SIF predictions
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

                        # In the extremely rare case where there are no valid pixels, choose
                        # the upper-left corner pixel to be valid just to avoid NaN
                        for i in range(valid_fine_sif_mask.shape[0]):
                            if torch.sum(valid_fine_sif_mask[i]) == 0:
                                valid_fine_sif_mask[i, 0, 0] = True

                        # For each tile, compute average predicted SIF over all valid pixels
                        predicted_coarse_sifs = sif_utils.masked_average(predicted_fine_sifs, valid_fine_sif_mask, dims_to_average=(1, 2)) # (batch size)

                        # if torch.isnan(predicted_coarse_sifs).any():
                            # predicted_coarse
                            # print('Predicted coarses SIFs', predicted_coarse_sifs)
                            # for i in range(valid_fine_sif_mask.shape[0]):
                            #     print('Tile', i, 'num valid', torch.sum(valid_fine_sif_mask[i]))
                            # print('Predicted nan')
                            # exit(1)

                        # Compute loss (predicted vs true coarse SIF)
                        coarse_loss = criterion(true_coarse_sifs, predicted_coarse_sifs)
                        if CROP_TYPE_LOSS:
                            coarse_loss += 0.0001 * sif_utils.crop_type_loss(predicted_fine_sifs, input_tiles_std, valid_fine_sif_mask)

                        # Backpropagate coarse loss
                        if phase == 'train' and random.random() < UPDATE_FRACTIONS[dataset_name]: # and not np.isnan(fine_loss.item()):
                            optimizer.zero_grad()
                            coarse_loss.backward()
                            # print('Grad', model.down1.maxpool_conv[1].double_conv[0].weight.grad)
                            optimizer.step()


                        # Compute/record losses
                        with torch.set_grad_enabled(False):
                            running_coarse_loss[dataset_name] += coarse_loss.item() * len(true_coarse_sifs)
                            num_coarse_datapoints[dataset_name] += len(true_coarse_sifs)
                            all_true_coarse_sifs[dataset_name].append(true_coarse_sifs.cpu().detach().numpy())
                            all_predicted_coarse_sifs[dataset_name].append(predicted_coarse_sifs.cpu().detach().numpy())

                            # Read fine-resolution SIF labels, if they exist
                            if 'fine_sif' in sample:
                                true_fine_sifs = sample['fine_sif'].to(device)
                                fine_soundings = sample['fine_soundings'].to(device)

                                # Extract the fine SIF data points where we have labels, and compute loss (just for information, not used during training)
                                non_noisy_mask = (fine_soundings >= MIN_FINE_CFIS_SOUNDINGS) & (true_fine_sifs >= MIN_SIF_CLIP)
                                valid_fine_sif_mask_flat = valid_fine_sif_mask.flatten() & non_noisy_mask.flatten()
                                true_fine_sifs_filtered = true_fine_sifs.flatten()[valid_fine_sif_mask_flat]
                                predicted_fine_sifs_filtered = predicted_fine_sifs.flatten()[valid_fine_sif_mask_flat]
                                predicted_fine_sifs_filtered = torch.clamp(predicted_fine_sifs_filtered, min=MIN_SIF_CLIP)
                                fine_loss = criterion(true_fine_sifs_filtered, predicted_fine_sifs_filtered)

                                # # Backpropagate fine loss (SHOULD NOT BE USED)
                                # if phase == 'train': # and not np.isnan(fine_loss.item()):
                                #     optimizer.zero_grad()
                                #     fine_loss.backward()
                                #     optimizer.step()

                                running_fine_loss[dataset_name] += fine_loss.item() * len(true_fine_sifs_filtered)
                                num_fine_datapoints[dataset_name] += len(true_fine_sifs_filtered)
                                all_true_fine_sifs[dataset_name].append(true_fine_sifs_filtered.cpu().detach().numpy())
                                all_predicted_fine_sifs[dataset_name].append(predicted_fine_sifs_filtered.cpu().detach().numpy())

                            # for i in range(0, 30, 2):
                            #     visualization_utils.plot_tile_predictions(input_tiles_std[i].cpu().detach().numpy(),
                            #                                 true_fine_sifs[i].cpu().detach().numpy(),
                            #                                 [predicted_fine_sifs[i].cpu().detach().numpy()],
                            #                                 valid_mask_numpy[i], ['U-Net'], 
                            #                                 sample['lon'][i].item(), sample['lat'][i].item(),
                            #                                 sample['date'][i],
                            #                                 TILE_SIZE_DEGREES, res=30)
                            # exit(0)

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

            # For each dataset, compute loss
            epoch_coarse_losses = dict()
            for coarse_dataset in COARSE_SIF_DATASETS[phase]:
                epoch_coarse_nrmse = math.sqrt(running_coarse_loss[coarse_dataset] / num_coarse_datapoints[coarse_dataset]) / sif_mean
                epoch_coarse_losses[coarse_dataset] = epoch_coarse_nrmse
                if phase == 'train':
                    train_coarse_losses[coarse_dataset].append(epoch_coarse_nrmse)
                else:
                    val_coarse_losses[coarse_dataset].append(epoch_coarse_nrmse)
                true_coarse = np.concatenate(all_true_coarse_sifs[coarse_dataset])
                predicted_coarse = np.concatenate(all_predicted_coarse_sifs[coarse_dataset])
                print('===== ', phase, coarse_dataset, 'Coarse stats ====')
                sif_utils.print_stats(true_coarse, predicted_coarse, sif_mean)

            for fine_dataset in FINE_SIF_DATASETS[phase]:
                epoch_fine_nrmse = math.sqrt(running_fine_loss[fine_dataset] / num_fine_datapoints[fine_dataset]) / sif_mean
                if phase == 'train':
                    train_fine_losses[fine_dataset].append(epoch_fine_nrmse)
                else:
                    val_fine_losses[fine_dataset].append(epoch_fine_nrmse)
                true_fine = np.concatenate(all_true_fine_sifs[fine_dataset])
                predicted_fine = np.concatenate(all_predicted_fine_sifs[fine_dataset])
                print('===== ', phase, fine_dataset, 'Fine stats ====')
                sif_utils.print_stats(true_fine, predicted_fine, sif_mean)



            # If the model performed better on "MODEL_SELECTION_DATASET" validation set than the
            # best previous model, record losses for this epoch, and save this model
            if phase == 'val' and epoch_coarse_losses[MODEL_SELECTION_DATASET] < best_val_coarse_loss[MODEL_SELECTION_DATASET]:
                for coarse_dataset in train_coarse_losses:
                    best_train_coarse_loss[coarse_dataset] = train_coarse_losses[coarse_dataset][-1]
                for coarse_dataset in val_coarse_losses:
                    best_val_coarse_loss[coarse_dataset] = val_coarse_losses[coarse_dataset][-1]
                for fine_dataset in train_fine_losses:
                    best_train_fine_loss[fine_dataset] = train_fine_losses[fine_dataset][-1]
                for fine_dataset in val_fine_losses:
                    best_val_fine_loss[fine_dataset] = val_fine_losses[fine_dataset][-1]
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
    print('Best train coarse losses:', best_train_coarse_loss)
    print('Best train fine loss:', best_train_fine_loss)
    print('Best val coarse loss:', best_val_coarse_loss)
    print('Best val fine loss:', best_val_fine_loss)

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
random_crop_transform = tile_transforms.RandomCrop(crop_dim=CROP_DIM)
cutout_transform = tile_transforms.Cutout(cutout_dim=CUTOUT_DIM, prob=CUTOUT_PROB)
transform_list_train = [standardize_transform, clip_transform] # [standardize_transform, noise_transform]
transform_list_val = [standardize_transform, clip_transform] #[standardize_transform]

# Apply cutout at beginning, since it is supposed to zero pixels out BEFORE standardizing
if 'cutout' in AUGMENTATIONS:
    transform_list_train.insert(0, cutout_transform)
if 'resize' in AUGMENTATIONS:
    transform_list_train.append(resize_transform)
if 'flip_and_rotate' in AUGMENTATIONS:
    transform_list_train.append(flip_and_rotate_transform)
if 'gaussian_noise' in AUGMENTATIONS:
    transform_list_train.append(noise_transform)
if 'jigsaw' in AUGMENTATIONS:
    transform_list_train.append(jigsaw_transform)
if 'random_crop' in AUGMENTATIONS:
    transform_list_train.append(random_crop_transform)

train_transform = transforms.Compose(transform_list_train)
val_transform = transforms.Compose(transform_list_val)

# Read dataset metadata files
train_datasets = dict()
val_datasets = dict()

for dataset_name, dataset_file in DATASET_FILES.items():
    metadata = pd.read_csv(dataset_file)

    # Filter tiles
    if 'CFIS' in dataset_name:
        # Only include CFIS tiles with enough valid pixels
        metadata = metadata[(metadata['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                            (metadata['SIF'] >= MIN_SIF_CLIP) &
                                            (metadata['missing_reflectance'] <= MAX_CFIS_CLOUD_COVER)]

    else:
        metadata = metadata[(metadata['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                                        (metadata['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                                        (metadata['SIF'] >= MIN_SIF_CLIP)]

    if '2018' in dataset_name:
        metadata['SIF'] /= 1.52
        print(metadata['SIF'].head())

    # Read dataset splits
    if dataset_name == 'OCO2_2016' or dataset_name == 'CFIS_2016':
        train_set = metadata[(metadata['grid_fold'].isin(TRAIN_FOLDS)) &
                            (metadata['date'].isin(TRAIN_DATES))].copy()
        val_set = metadata[(metadata['grid_fold'].isin(VAL_FOLDS)) &
                        (metadata['date'].isin(TRAIN_DATES))].copy()
        test_set = metadata[(metadata['grid_fold'].isin(TEST_FOLDS)) &
                            (metadata['date'].isin(TEST_DATES))].copy()
    else:
        train_set = metadata[(metadata['fold'].isin(TRAIN_FOLDS)) &
                            (metadata['date'].isin(TRAIN_DATES))].copy()
        val_set = metadata[(metadata['fold'].isin(VAL_FOLDS)) &
                        (metadata['date'].isin(TRAIN_DATES))].copy()
        test_set = metadata[(metadata['fold'].isin(TEST_FOLDS)) &
                            (metadata['date'].isin(TEST_DATES))].copy()

    # Create Dataset objects
    if dataset_name in COARSE_SIF_DATASETS['train'] or dataset_name in FINE_SIF_DATASETS['train']:
        if 'CFIS' in dataset_name:
            train_datasets[dataset_name] = FineSIFDataset(train_set, train_transform)
        else:
            train_datasets[dataset_name] = CoarseSIFDataset(train_set, train_transform)
    if dataset_name in COARSE_SIF_DATASETS['val'] or dataset_name in FINE_SIF_DATASETS['val']:
        if 'CFIS' in dataset_name:
            val_datasets[dataset_name] = FineSIFDataset(val_set, val_transform)
        else:
            val_datasets[dataset_name] = CoarseSIFDataset(val_set, val_transform)


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
PARAM_STRING += ('Train datasets:\n')
for dataset_name, dataset in train_datasets.items():
    PARAM_STRING += ('   *' + dataset_name + ': ' + str(len(dataset)) + ' tiles\n')
PARAM_STRING += ('Val datasets:\n')
for dataset_name, dataset in val_datasets.items():
    PARAM_STRING += ('   *' + dataset_name + ': ' + str(len(dataset)) + ' tiles\n')
PARAM_STRING += ('Train dates: ' + str(TRAIN_DATES) + '\n')
PARAM_STRING += ('Test dates: ' + str(TEST_DATES) + '\n')
PARAM_STRING += ('Min soundings (fine CFIS): ' + str(MIN_FINE_CFIS_SOUNDINGS) + '\n')
PARAM_STRING += ('Min SIF clip: ' + str(MIN_SIF_CLIP) + '\n')
PARAM_STRING += ('Max cloud cover (coarse CFIS): ' + str(MAX_CFIS_CLOUD_COVER) + '\n')
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
if 'random_crop' in AUGMENTATIONS:
    PARAM_STRING += ("Random crop size: " + str(CROP_DIM) + '\n')
if 'cutout' in AUGMENTATIONS:
    PARAM_STRING += ("Cutout: size " + str(CUTOUT_DIM) + ", prob " + str(CUTOUT_PROB) + '\n')
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


# Create dataset/dataloaders
datasets = {'train': CombinedDataset(train_datasets),
            'val': CombinedDataset(val_datasets)}
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
elif MODEL_TYPE == 'unet2_larger':
    unet_model = UNet2Larger(n_channels=INPUT_CHANNELS, n_classes=1, reduced_channels=REDUCED_CHANNELS, min_output=min_output, max_output=max_output).to(device)
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
# criterion = nn.SmoothL1Loss(reduction='mean')
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
    unet_model.dimensionality_reduction_1.requires_grad = False
    unet_model.inc.requires_grad = False
    # unet_model.dimensionality_reduction_2.requires_grad = False

# Train model to predict SIF
unet_model, train_coarse_losses, val_coarse_losses, train_fine_losses, val_fine_losses = train_model(unet_model, dataloaders, criterion, optimizer, device, sif_mean, sif_std, num_epochs=NUM_EPOCHS)
torch.save(unet_model.state_dict(), MODEL_FILE)

# Plot loss curves: NRMSE
epoch_list = range(NUM_EPOCHS)
plots = []
print("Coarse Train NRMSE:", train_coarse_losses['CFIS_2016'])
train_coarse_plot, = plt.plot(epoch_list, train_coarse_losses['CFIS_2016'], color='blue', label='Coarse Train NRMSE (CFIS)')
plots.append(train_coarse_plot)
if 'OCO2_2016' in COARSE_SIF_DATASETS:
    print("OCO2 Train NRMSE:", train_coarse_losses['OCO2_2016'])
    train_oco2_plot, = plt.plot(epoch_list, train_coarse_losses['OCO2_2016'], color='purple', label='Coarse Train NRMSE (OCO-2)')
    plots.append(train_oco2_plot)
print("Coarse Val NRMSE:", val_coarse_losses['CFIS_2016'])
val_coarse_plot, = plt.plot(epoch_list, val_coarse_losses['CFIS_2016'], color='green', label='Coarse Val NRMSE')
plots.append(val_coarse_plot)
print("Fine Train NRMSE:", train_fine_losses['CFIS_2016'])
train_fine_plot, = plt.plot(epoch_list, train_fine_losses['CFIS_2016'], color='red', label='Fine Train (Interpolated) NRMSE')
plots.append(train_fine_plot)
print("Fine Val NRMSE:", val_fine_losses['CFIS_2016'])
val_fine_plot, = plt.plot(epoch_list, val_fine_losses['CFIS_2016'], color='orange', label='Fine Val (Interpolated) NRMSE')
plots.append(val_fine_plot)


# Add legend and axis labels
plt.legend(handles=plots)
plt.xlabel('Epoch #')
plt.ylabel('NRMSE')
plt.savefig(LOSS_PLOT + '_nrmse.png')
plt.close()

# Plot train coarse vs train fine losses
print('============== Train: Fine vs Coarse Losses ===============')
sif_utils.print_stats(train_fine_losses['CFIS_2016'], train_coarse_losses['CFIS_2016'], sif_mean, ax=plt.gca())
plt.xlabel('Train Coarse Losses')
plt.ylabel('Train Fine Losses')
plt.title('Fine vs Coarse train losses' + METHOD)
plt.savefig(LOSS_PLOT + '_scatter_fine_vs_coarse.png')
plt.close()