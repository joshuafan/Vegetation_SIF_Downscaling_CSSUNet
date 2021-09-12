"""
Trains CSR-U-Net on SIF data. The model uses coarse-resolution labels for supervision, but can 
generate fine-resolution predictions.
"""
import argparse
import copy
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import grad

from reflectance_cover_sif_dataset import CombinedDataset, CoarseSIFDataset, FineSIFDataset
from unet.unet_model import UNet, UNetSmall, UNet2, UNet2PixelEmbedding, UNet2Larger
import visualization_utils
import sif_utils
import tile_transforms
import tqdm


# Folds
TRAIN_FOLDS = [0, 1, 2]
VAL_FOLDS = [3]
TEST_FOLDS = [4]

# Data files
DATA_DIR = "../data"

# Datasets tp ise/
DATASET_FILES = {'CFIS_2016': os.path.join(DATA_DIR, 'cfis_coarse_metadata.csv'),
                 'OCO2_2016': os.path.join(DATA_DIR, 'oco2_metadata.csv')}
COARSE_SIF_DATASETS = {'train': ['CFIS_2016', 'OCO2_2016'],
                       'val': ['CFIS_2016']}
FINE_SIF_DATASETS = {'train': ['CFIS_2016'],
                     'val': ['CFIS_2016']}
MODEL_SELECTION_DATASET = 'CFIS_2016'

# Ratio of how often each dataset is used. If both 1, this means that every
# batch of CFIS alternates with one batch of OCO2.
UPDATE_FRACTIONS = {'CFIS_2016': 1,
                    'OCO2_2016': 1}
BAND_STATISTICS_FILE = os.path.join(DATA_DIR, 'cfis_band_statistics_train.csv')

# Dataset resolution/scale
RES = (0.00026949458523585647, 0.00026949458523585647)
TILE_PIXELS = 100
TILE_SIZE_DEGREES = RES[0] * TILE_PIXELS


# Commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('-prefix', "--prefix", default='10d_', type=str, help='prefix')
parser.add_argument('-model', "--model", default='unet2', type=str, help='model type')

# Random seed
parser.add_argument('-seed', "--seed", default=0, type=int)

# Optimizer params
parser.add_argument('-optimizer', "--optimizer", default='AdamW', choices=["Adam", "AdamW"], type=str, help='optimizer type')
parser.add_argument('-lr', "--learning_rate", default=1e-4, type=float, help='initial learning rate')
parser.add_argument('-wd', "--weight_decay", default=1e-4, type=float, help='weight decay rate')
parser.add_argument('-sche', "--scheduler", default='cosine', choices=['cosine', 'step', 'plateau', 'exp', 'const'], help='lr scheduler')
parser.add_argument('-T0', "--T0", default=50, type=int, help='optimizer T0 (cosine only)')
parser.add_argument('-T_mult', "--T_mult", default=2, type=int, help='optimizer T_multi (cosine only)') 
parser.add_argument('-eta_min', "--eta_min", default=1e-5, type=float, help='minimum lr (cosine only)')
parser.add_argument('-gamma', "--gamma", default=0.5, type=float, help='StepLR decay (step only)')
parser.add_argument('-lrsteps', "--lrsteps", default=50, type=int, help='StepLR steps (step only)')


# Training params
parser.add_argument('-epoch', "--max_epoch", default=100, type=int, help='max epoch to train')
parser.add_argument('-patience', "--patience", default=5, type=int, help="Early stopping patience (stop if val loss doesn't improve for this many epochs)")
parser.add_argument('-bs', "--batch_size", default=100, type=int, help="Batch size")
parser.add_argument('-num_workers', "--num_workers", default=4, type=int, help="Number of dataloader workers")
parser.add_argument('-from_pretrained', "--from_pretrained", default=False, action='store_true', help='Whether to initialize from pre-trained model')

# Optional loss terms
parser.add_argument('-crop_type_loss', "--crop_type_loss", default=False, action='store_true', help='Whether to add a "crop type loss" term (encouraging predictions for same crop type to be similar)')
parser.add_argument('-recon_loss', "--recon_loss", default=False, action='store_true', help='Whether to add a reconstruction loss')

# Restricting output and input values
parser.add_argument('-min_sif', "--min_sif", default=None, type=float, help="If (min_sif, max_sif) are set, the model uses a tanh function to ensure the output is within that range.")
parser.add_argument('-max_sif', "--max_sif", default=None, type=float, help="If (min_sif, max_sif) are set, the model uses a tanh function to ensure the output is within that range.")
parser.add_argument('-min_sif_clip', "--min_sif_clip", default=0.1, type=float, help="Before computing loss, clip outputs below this to this value.")
parser.add_argument('-min_input', "--min_input", default=-3, type=float, help="Clip extreme input values to this many standard deviations below mean")
parser.add_argument('-max_input', "--max_input", default=3, type=float, help="Clip extreme input values to this many standard deviations above mean")
parser.add_argument('-reduced_channels', "--reduced_channels", default=None, type=int, help="If this is set, add a 'dimensionality reduction' layer to the front of the model, to reduce the number of channels to this.")

# Augmentations. None are enabled by default.
parser.add_argument('-fraction_outputs_to_average', "--fraction_outputs_to_average", default=0.2, type=float, help="Fraction of outputs to average when computing loss.")
parser.add_argument('-flip_and_rotate', "--flip_and_rotate", action='store_true')
parser.add_argument('-jigsaw', "--jigsaw", action='store_true')
parser.add_argument('-multiplicative_noise', "--multiplicative_noise", action='store_true')
parser.add_argument('-mult_noise_std', "--mult_noise_std", default=0.2, type=float, help="If the 'multiplicative_noise' augmentation is used, multiply each channel by (1+eps), where eps ~ N(0, mult_noise_std)")
parser.add_argument('-gaussian_noise', "--gaussian_noise", action='store_true')
parser.add_argument('-gaussian_noise_std', "--gaussian_noise_std", default=0.2, type=float, help="If the 'multiplicative_noise' augmentation is used, add eps to each pixel, where eps ~ N(0, gaussian_noise_std)")
parser.add_argument('-cutout', "--cutout", action='store_true')
parser.add_argument('-cutout_dim', "--cutout_dim", default=10, type=int, help="If the 'cutout' augmentation is used, this is the dimension of the square to be erased.")
parser.add_argument('-cutout_prob', "--cutout_prob", default=0.5, type=float, help="If the 'cutout' augmentation is used, this is the probability that a square will be erased.")
parser.add_argument('-resize', "--resize", action='store_true')
parser.add_argument('-resize_dim', "--resize_dim", default=100, type=int, help="If the 'resize' augmentation is used, the dimension to resize to.")
parser.add_argument('-random_crop', "--random_crop", action='store_true')
parser.add_argument('-crop_dim', "--crop_dim", default=80, type=int, help="If the 'random_crop' augmentation is used, the dimension to crop.")


args = parser.parse_args()

# Set random seeds
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Create param string used for model path
PARAM_SETTING = "{}_{}_optimizer-{}_bs-{}_lr-{}_wd-{}_maxepoch-{}_sche-{}_fractionoutputs-{}_seed-{}".format( 
    args.prefix, args.model, args.optimizer, args.batch_size, args.learning_rate, args.weight_decay, args.max_epoch, args.scheduler, args.fraction_outputs_to_average, args.seed
)
if args.flip_and_rotate:
    PARAM_SETTING += "_fliprotate"
if args.jigsaw:
    PARAM_SETTING += "_jigsaw"
if args.multiplicative_noise:
    PARAM_SETTING += "_multiplicativenoise-{}".format(args.mult_noise_std)
if args.gaussian_noise:
    PARAM_SETTING += "_gaussiannoise-{}".format(args.mult_noise_std)
if args.cutout:
    PARAM_SETTING += "_cutout-{}_prob-{}".format(args.cutout_dim, args.cutout_prob)
if args.resize:
    PARAM_SETTING += "_resize-{}".format(args.resize_dim)
if args.random_crop:
    PARAM_SETTING += "_randomcrop-{}".format(args.crop_dim)


# Model files
results_dir = os.path.join("unet_results", PARAM_SETTING)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
PRETRAINED_MODEL_FILE = os.path.join(results_dir, "model.ckpt")
MODEL_FILE = os.path.join(results_dir, "model.ckpt")

# Results files/plots
CFIS_RESULTS_CSV_FILE = os.path.join(results_dir, 'cfis_results_' + PARAM_SETTING + '.csv')
LOSS_PLOT = os.path.join(results_dir, 'losses')

# Summary csv file of all results. Create this if it doesn't exist
RESULTS_SUMMARY_FILE = "unet_results/results_summary.csv"
if not os.path.isfile(RESULTS_SUMMARY_FILE):
    with open(RESULTS_SUMMARY_FILE, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['model', 'git_commit', 'command', 'optimizer', 'lr', 'wd', 'fine_val_nrmse', 'path_to_model', 'best_val_fine_epoch'])
GIT_COMMIT = sif_utils.get_git_revision_hash()
COMMAND_STRING = " ".join(sys.argv)


# Dataset params
# OCO-2 filtering
MIN_OCO2_SOUNDINGS = 3
MAX_OCO2_CLOUD_COVER = 0.5
OCO2_SCALING_FACTOR = 0.97

# CFIS filtering
MIN_FINE_CFIS_SOUNDINGS = 30
MIN_COARSE_FRACTION_VALID_PIXELS = 0.1
MAX_CFIS_CLOUD_COVER = 0.5
MIN_CDL_COVERAGE = 0.5

# Dates
TRAIN_DATES = ['2016-06-15', '2016-08-01']
TEST_DATES = ['2016-06-15', '2016-08-01']

ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']

# Which bands
BANDS = list(range(0, 24))
RECONSTRUCTION_BANDS = list(range(0, 23))
INPUT_CHANNELS = len(BANDS)
OUTPUT_CHANNELS = 1 + len(RECONSTRUCTION_BANDS)
CROP_TYPE_INDICES = list(range(12, 23))
MISSING_REFLECTANCE_IDX = len(BANDS) - 1





def train_model(args, model, dataloaders, criterion, optimizer, device, sif_mean, sif_std):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    # Record losses at each epoch
    train_coarse_losses = {k: [] for k in COARSE_SIF_DATASETS['train']}
    val_coarse_losses = {k: [] for k in COARSE_SIF_DATASETS['val']}
    train_reconstruction_losses = {k: [] for k in COARSE_SIF_DATASETS['train']}
    val_reconstruction_losses = {k: [] for k in COARSE_SIF_DATASETS['val']}
    train_fine_losses = {k: [] for k in FINE_SIF_DATASETS['train']}
    val_fine_losses = {k: [] for k in FINE_SIF_DATASETS['val']}
    num_epochs_no_improvement = 0  # Record number of consecutive epochs in which fine validation loss did not improve

    # Record the losses at the epoch with lowest *fine validation loss*, and at the epoch with lowest *coarse validation loss*
    best_val_fine_epoch = -1
    best_val_coarse_epoch = -1
    train_coarse_loss_at_best_val_fine = {k: float('inf') for k in COARSE_SIF_DATASETS['train']}
    val_coarse_loss_at_best_val_fine = {k: float('inf') for k in COARSE_SIF_DATASETS['val']}
    train_coarse_loss_at_best_val_coarse = {k: float('inf') for k in COARSE_SIF_DATASETS['train']}
    val_coarse_loss_at_best_val_coarse = {k: float('inf') for k in COARSE_SIF_DATASETS['val']}
    train_fine_loss_at_best_val_fine = {k: float('inf') for k in FINE_SIF_DATASETS['train']}
    val_fine_loss_at_best_val_fine = {k: float('inf') for k in FINE_SIF_DATASETS['val']}    
    train_fine_loss_at_best_val_coarse = {k: float('inf') for k in FINE_SIF_DATASETS['train']}
    val_fine_loss_at_best_val_coarse = {k: float('inf') for k in FINE_SIF_DATASETS['val']}


    for epoch in range(args.max_epoch):
        print('Epoch {}/{}'.format(epoch, args.max_epoch - 1))
        print('-' * 10)
        epoch_start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_reconstruction_loss = {k: 0 for k in COARSE_SIF_DATASETS[phase]}
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
                    # Read input tile
                    input_tiles_std = sample['input_tile'][:, BANDS, :, :].to(device)

                    # Note - we do not use reconstruction loss in this paper. It did not affect results much.
                    reconstruction_target = sample['input_tile'][:, RECONSTRUCTION_BANDS, :, :].detach().clone().to(device)

                    # Read coarse-resolution SIF label
                    true_coarse_sifs = sample['coarse_sif'].to(device)

                    # Read "mask" of which fine SIF pixels are valid and should be used in averaging.
                    # If the mask doesn't exist, use the Landsat quality mask.
                    if 'fine_sif_mask' in sample:
                        valid_fine_sif_mask = torch.logical_not(sample['fine_sif_mask']).to(device)
                    else:
                        valid_fine_sif_mask = torch.logical_not(input_tiles_std[:, MISSING_REFLECTANCE_IDX, :, :])
                    valid_mask_numpy = valid_fine_sif_mask.cpu().detach().numpy() 

                    with torch.set_grad_enabled(phase == 'train' and dataset_name in COARSE_SIF_DATASETS[phase]):

                        # Pass tile through model to obtain fine-resolution SIF predictions
                        outputs = model(input_tiles_std)  # predicted_fine_sifs_std: (batch size, output dim, H, W)
                        if type(outputs) == tuple:
                            outputs = outputs[0]
                        predicted_fine_sifs_std = outputs[:, 0, :, :]  # torch.squeeze(predicted_fine_sifs_std, dim=1)
                        predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean


                        # As a regularization technique, randomly choose more pixels to ignore.
                        if phase == 'train':
                            pixels_to_include = torch.rand(valid_fine_sif_mask.shape, device=device) > (1 - args.fraction_outputs_to_average)
                            valid_fine_sif_mask = valid_fine_sif_mask * pixels_to_include

                        # In the extremely rare case where there are no valid pixels, choose
                        # the upper-left corner pixel to be valid just to avoid NaN
                        for i in range(valid_fine_sif_mask.shape[0]):
                            if torch.sum(valid_fine_sif_mask[i]) == 0:
                                valid_fine_sif_mask[i, 0, 0] = True
                                print("No valid pixels in ", sample['tile_file'][i])
                                exit(1)

                        # For each tile, compute average predicted SIF over all valid pixels
                        predicted_coarse_sifs = sif_utils.masked_average(predicted_fine_sifs, valid_fine_sif_mask, dims_to_average=(1, 2)) # (batch size)

                        # Compute loss (predicted vs true coarse SIF)
                        coarse_loss = criterion(true_coarse_sifs, predicted_coarse_sifs)

                        # OPTIONAL - NOT USED IN PAPER: Compute reconstruction loss
                        reconstruction_loss = criterion(outputs[:, 1:, :, :].flatten(start_dim=1),
                                                        reconstruction_target.flatten(start_dim=1))
                        if args.recon_loss:
                            loss = coarse_loss + reconstruction_loss
                        else:
                            loss = coarse_loss


                        # Backpropagate coarse loss
                        if phase == 'train' and random.random() < UPDATE_FRACTIONS[dataset_name] and dataset_name in COARSE_SIF_DATASETS[phase]:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        # Compute/record losses
                        with torch.set_grad_enabled(False):
                            if dataset_name in COARSE_SIF_DATASETS[phase]:
                                running_coarse_loss[dataset_name] += coarse_loss.item() * len(true_coarse_sifs)
                                running_reconstruction_loss[dataset_name] += reconstruction_loss.item() * len(true_coarse_sifs)
                                num_coarse_datapoints[dataset_name] += len(true_coarse_sifs)
                                all_true_coarse_sifs[dataset_name].append(true_coarse_sifs.cpu().detach().numpy())
                                all_predicted_coarse_sifs[dataset_name].append(predicted_coarse_sifs.cpu().detach().numpy())

                            # Read fine-resolution SIF labels, if they exist
                            if 'fine_sif' in sample and dataset_name in FINE_SIF_DATASETS[phase]:
                                true_fine_sifs = sample['fine_sif'].to(device)
                                fine_soundings = sample['fine_soundings'].to(device)

                                # Extract the fine SIF data points where we have labels, and compute loss (just for information, not used during training)
                                non_noisy_mask = (fine_soundings >= MIN_FINE_CFIS_SOUNDINGS) & (true_fine_sifs >= args.min_sif_clip)
                                valid_fine_sif_mask_flat = valid_fine_sif_mask.flatten() & non_noisy_mask.flatten()
                                true_fine_sifs_filtered = true_fine_sifs.flatten()[valid_fine_sif_mask_flat]
                                if true_fine_sifs_filtered.shape[0] == 0:  # If there are no non-noisy pixels in this batch, skip it
                                    continue
                                predicted_fine_sifs_filtered = predicted_fine_sifs.flatten()[valid_fine_sif_mask_flat]
                                predicted_fine_sifs_filtered = torch.clamp(predicted_fine_sifs_filtered, min=args.min_sif_clip)
                                fine_loss = criterion(true_fine_sifs_filtered, predicted_fine_sifs_filtered)

                                running_fine_loss[dataset_name] += fine_loss.item() * len(true_fine_sifs_filtered)
                                num_fine_datapoints[dataset_name] += len(true_fine_sifs_filtered)
                                all_true_fine_sifs[dataset_name].append(true_fine_sifs_filtered.cpu().detach().numpy())
                                all_predicted_fine_sifs[dataset_name].append(predicted_fine_sifs_filtered.cpu().detach().numpy())


            # For each dataset, compute loss
            for coarse_dataset in COARSE_SIF_DATASETS[phase]:
                # Record reconstruction loss
                epoch_reconstruction_loss = running_reconstruction_loss[coarse_dataset] / num_coarse_datapoints[coarse_dataset]

                # Compute and record coarse SIF loss
                epoch_coarse_nrmse = math.sqrt(running_coarse_loss[coarse_dataset] / num_coarse_datapoints[coarse_dataset]) / sif_mean
                if phase == 'train':
                    train_reconstruction_losses[coarse_dataset].append(epoch_reconstruction_loss)
                    train_coarse_losses[coarse_dataset].append(epoch_coarse_nrmse)
                else:
                    val_reconstruction_losses[coarse_dataset].append(epoch_reconstruction_loss)
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
                    if np.isnan(epoch_fine_nrmse):
                        print("EPOCH FINE NRMSE was NAN!\nRunning fine loss", running_fine_loss[fine_dataset])
                        print("Num fine datapoints", num_fine_datapoints[fine_dataset])
                        print("Sif mean", sif_mean)
                    val_fine_losses[fine_dataset].append(epoch_fine_nrmse)

                true_fine = np.concatenate(all_true_fine_sifs[fine_dataset])
                predicted_fine = np.concatenate(all_predicted_fine_sifs[fine_dataset])
                print('===== ', phase, fine_dataset, 'Fine stats ====')
                sif_utils.print_stats(true_fine, predicted_fine, sif_mean)


            # If this is the best-performing model on the COARSE-resolution validation set,
            # save the model under "best_val_coarse". But this actually doesn't work well, it's
            # for informational purposes only
            if phase == 'val' and val_coarse_losses[MODEL_SELECTION_DATASET][-1] < val_coarse_loss_at_best_val_coarse[MODEL_SELECTION_DATASET]:
                torch.save(model.state_dict(), MODEL_FILE + '_best_val_coarse')
                for coarse_dataset in train_coarse_losses:
                    train_coarse_loss_at_best_val_coarse[coarse_dataset] = train_coarse_losses[coarse_dataset][-1]
                for coarse_dataset in val_coarse_losses:
                    val_coarse_loss_at_best_val_coarse[coarse_dataset] = val_coarse_losses[coarse_dataset][-1]
                for fine_dataset in train_fine_losses:
                    train_fine_loss_at_best_val_coarse[fine_dataset] = train_fine_losses[fine_dataset][-1]
                for fine_dataset in val_fine_losses:
                    val_fine_loss_at_best_val_coarse[fine_dataset] = val_fine_losses[fine_dataset][-1]
                best_val_coarse_epoch = epoch

            # If this is the best-performing model on the FINE-resolution validation set,
            # save the model file and parameters. We select the model by its performance
            # on the fine-resolution validation set.
            if phase == 'val' and 'CFIS_2016' in val_fine_losses:
                if val_fine_losses[MODEL_SELECTION_DATASET][-1] < val_fine_loss_at_best_val_fine[MODEL_SELECTION_DATASET]:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), MODEL_FILE)
                    for coarse_dataset in train_coarse_losses:
                        train_coarse_loss_at_best_val_fine[coarse_dataset] = train_coarse_losses[coarse_dataset][-1]
                    for coarse_dataset in val_coarse_losses:
                        val_coarse_loss_at_best_val_fine[coarse_dataset] = val_coarse_losses[coarse_dataset][-1]
                    for fine_dataset in train_fine_losses:
                        train_fine_loss_at_best_val_fine[fine_dataset] = train_fine_losses[fine_dataset][-1]
                    for fine_dataset in val_fine_losses:
                        val_fine_loss_at_best_val_fine[fine_dataset] = val_fine_losses[fine_dataset][-1]
                    best_val_fine_epoch = epoch

                # Write results to file
                with open(os.path.join(results_dir, "csv_row.csv"), mode='w') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow([args.model, GIT_COMMIT, COMMAND_STRING, args.optimizer, args.learning_rate, args.weight_decay, val_fine_loss_at_best_val_fine['CFIS_2016'], MODEL_FILE])


        # Print elapsed time per epoch
        epoch_time = time.time() - epoch_start
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            epoch_time // 60, epoch_time % 60))
        print()

        # If fine validation loss did not improve for more than "patience" consecutive epochs, stop training.
        if epoch >= 1 and val_fine_losses[MODEL_SELECTION_DATASET][-1] < val_fine_losses[MODEL_SELECTION_DATASET][-2]:
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1
            if num_epochs_no_improvement > args.patience:
                break


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('=================== At best val fine (epoch ' + str(best_val_fine_epoch) + ') ========================')
    print('train coarse losses:', train_coarse_loss_at_best_val_fine)
    print('train fine loss:', train_fine_loss_at_best_val_fine)
    print('val coarse loss:', val_coarse_loss_at_best_val_fine)
    print('val fine loss:', val_fine_loss_at_best_val_fine)
    print('============= INFO ONLY: At best val coarse (epoch ' + str(best_val_coarse_epoch) + ') ===============')
    print('train coarse losses:', train_coarse_loss_at_best_val_coarse)
    print('train fine loss:', train_fine_loss_at_best_val_coarse)
    print('val coarse loss:', val_coarse_loss_at_best_val_coarse)
    print('val fine loss:', val_fine_loss_at_best_val_coarse)

    # Write results to file
    with open(RESULTS_SUMMARY_FILE, "a+") as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([args.model, GIT_COMMIT, COMMAND_STRING, args.optimizer, args.learning_rate, args.weight_decay, val_fine_loss_at_best_val_fine['CFIS_2016'], MODEL_FILE, best_val_fine_epoch])


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_coarse_losses, val_coarse_losses, train_fine_losses, val_fine_losses, train_reconstruction_losses, val_reconstruction_losses


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


# Constrain predicted SIF to be between certain values (unstandardized), if desired
# Don't forget to standardize
if args.min_sif is not None and args.max_sif is not None:
    min_output = (args.min_sif - sif_mean) / sif_std
    max_output = (args.max_sif - sif_mean) / sif_std
else:
    min_output = None
    max_output = None

# Set up image transforms / augmentations
standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds) #, min_input=MIN_INPUT, max_input=MAX_INPUT)
clip_transform = tile_transforms.ClipTile(min_input=args.min_input, max_input=args.max_input)
noise_transform = tile_transforms.GaussianNoise(continuous_bands=list(range(0, 9)), standard_deviation=args.gaussian_noise_std)
multiplicative_noise_transform = tile_transforms.MultiplicativeGaussianNoise(continuous_bands=list(range(0, 9)), standard_deviation=args.mult_noise_std)
flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()
jigsaw_transform = tile_transforms.RandomJigsaw()
resize_transform = tile_transforms.ResizeTile(target_dim=[args.resize_dim, args.resize_dim])
random_crop_transform = tile_transforms.RandomCrop(crop_dim=args.crop_dim)
cutout_transform = tile_transforms.Cutout(cutout_dim=args.cutout_dim, prob=args.cutout_prob)
transform_list_train = [standardize_transform, clip_transform]
transform_list_val = [standardize_transform, clip_transform]

# Apply cutout at beginning, since it is supposed to zero pixels out BEFORE standardizing
if args.cutout:
    transform_list_train.insert(0, cutout_transform)

# Add multiplicative noise at beginning (before standardization)
if args.multiplicative_noise:
    transform_list_train.insert(0, multiplicative_noise_transform)

# Other augmentations
if args.resize:
    transform_list_train.append(resize_transform)
if args.flip_and_rotate:
    transform_list_train.append(flip_and_rotate_transform)
if args.gaussian_noise:
    transform_list_train.append(noise_transform)
if args.jigsaw:
    transform_list_train.append(jigsaw_transform)
if args.random_crop:
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
                            (metadata['SIF'] >= args.min_sif_clip) &
                            (metadata['missing_reflectance'] <= MAX_CFIS_CLOUD_COVER)]

    else:
        metadata = metadata[(metadata['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                            (metadata['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                            (metadata['SIF'] >= args.min_sif_clip)]
    metadata = metadata[metadata[ALL_COVER_COLUMNS].sum(axis=1) >= MIN_CDL_COVERAGE]

    # Read dataset splits
    if dataset_name == 'OCO2_2016' or dataset_name == 'CFIS_2016':
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
PARAM_STRING += ('Min SIF clip: ' + str(args.min_sif_clip) + '\n')
PARAM_STRING += ('Max cloud cover (coarse CFIS): ' + str(MAX_CFIS_CLOUD_COVER) + '\n')
PARAM_STRING += ('Min fraction valid pixels in CFIS tile: ' + str(MIN_COARSE_FRACTION_VALID_PIXELS) + '\n')
PARAM_STRING += ('Train features: ' + str(BANDS) + '\n')
PARAM_STRING += ("Clip input features: " + str(args.min_input) + " to " + str(args.max_input) + " standard deviations from mean\n")
PARAM_STRING += ('================= METHOD ===============\n')
if args.from_pretrained:
    PARAM_STRING += ('From pretrained model: ' + os.path.basename(PRETRAINED_MODEL_FILE) + '\n')
else:
    PARAM_STRING += ("Training from scratch\n")
if args.recon_loss:
    PARAM_STRING += ("Using reconstruction loss!\n")
PARAM_STRING += ("Param string: " + PARAM_SETTING + '\n')
PARAM_STRING += ("Model type: " + args.model + '\n')
PARAM_STRING += ("Optimizer: " + args.optimizer + '\n')
PARAM_STRING += ("Learning rate: " + str(args.learning_rate) + '\n')
PARAM_STRING += ("Weight decay: " + str(args.weight_decay) + '\n')
PARAM_STRING += ("Num workers: " + str(args.num_workers) + '\n')
PARAM_STRING += ("Batch size: " + str(args.batch_size) + '\n')
PARAM_STRING += ("Num epochs: " + str(args.max_epoch) + '\n')
PARAM_STRING += ("Augmentations:\n")
if args.flip_and_rotate:
    PARAM_STRING += ('\tRandom flip and rotate\n')
if args.jigsaw:
    PARAM_STRING += ('\tJigsaw\n')
if args.resize:
    PARAM_STRING += ('\tResize images to: ' + str(args.resize_dim) + '\n')
if args.gaussian_noise:
    PARAM_STRING += ("\tGaussian noise (std deviation): " + str(args.gaussian_noise_std) + '\n')
if args.multiplicative_noise:
    PARAM_STRING += ("\tMultiplicative noise: Gaussian (std deviation): " + str(args.mult_noise_std) + '\n')
if args.random_crop:
    PARAM_STRING += ("\tRandom crop size: " + str(args.crop_dim) + '\n')
if args.cutout:
    PARAM_STRING += ("\tCutout: size " + str(args.cutout_dim) + ", prob " + str(args.cutout_prob) + '\n')
PARAM_STRING += ("Fraction outputs to average: " + str(args.fraction_outputs_to_average) + '\n')
PARAM_STRING += ("SIF range: " + str(args.min_sif) + " to " + str(args.max_sif) + '\n')
PARAM_STRING += ("SIF statistics: mean " + str(sif_mean) + ", std " + str(sif_std) + '\n')
print(PARAM_STRING)


# Create dataset/dataloaders
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)
datasets = {'train': CombinedDataset(train_datasets),
            'val': CombinedDataset(val_datasets)}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
               for x in ['train', 'val']}

# Initialize model
if args.model == 'unet_small':
    unet_model = UNetSmall(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, reduced_channels=args.reduced_channels, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'unet2':
    unet_model = UNet2(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, reduced_channels=args.reduced_channels, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'unet2_larger':
    unet_model = UNet2Larger(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, reduced_channels=args.reduced_channels, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'unet2_pixel_embedding':
    unet_model = UNet2PixelEmbedding(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, reduced_channels=args.reduced_channels, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'unet':
    unet_model = UNet(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, reduced_channels=args.reduced_channels, min_output=min_output, max_output=max_output).to(device)   
else:
    print('Model type not supported')
    exit(1)

# If we're loading a pre-trained model, read model params from file
if args.from_pretrained:
    unet_model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE, map_location=device))

# Initialize loss and optimizer
criterion = nn.MSELoss(reduction='mean')
if args.optimizer == "Adam":
    optimizer = optim.Adam(unet_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer == "AdamW":
    optimizer = optim.AdamW(unet_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
else:
    print("Optimizer not supported")
    exit(1)


# Train model to predict SIF
unet_model, train_coarse_losses, val_coarse_losses, train_fine_losses, val_fine_losses, train_reconstruction_losses, val_reconstruction_losses = train_model(args, unet_model, dataloaders, criterion, optimizer, device, sif_mean, sif_std)
torch.save(unet_model.state_dict(), MODEL_FILE)

print("Train coarse", train_coarse_losses)
# Plot loss curves: NRMSE
epoch_list = range(len(train_coarse_losses['CFIS_2016']))
plots = []
train_coarse_plot, = plt.plot(epoch_list, train_coarse_losses['CFIS_2016'], color='blue', label='Coarse Train NRMSE (CFIS)')
plots.append(train_coarse_plot)
if args.recon_loss:
    train_recon_plot, = plt.plot(epoch_list, train_reconstruction_losses['CFIS_2016'], color='black', label='Train reconstruction loss')
    plots.append(train_recon_plot)
# if 'OCO2_2016' in COARSE_SIF_DATASETS["train"]:
#     train_oco2_plot, = plt.plot(epoch_list, train_coarse_losses['OCO2_2016'], color='purple', label='Coarse Train NRMSE (OCO-2)')
#     plots.append(train_oco2_plot)
val_coarse_plot, = plt.plot(epoch_list, val_coarse_losses['CFIS_2016'], color='green', label='Coarse Val NRMSE')
plots.append(val_coarse_plot)
if args.recon_loss:
    val_recon_plot, = plt.plot(epoch_list, val_reconstruction_losses['CFIS_2016'], color='gray', label='Val reconstruction loss')
    plots.append(val_recon_plot)
train_fine_plot, = plt.plot(epoch_list, train_fine_losses['CFIS_2016'], color='red', label='Fine Train (Interpolated) NRMSE')
plots.append(train_fine_plot)
val_fine_plot, = plt.plot(epoch_list, val_fine_losses['CFIS_2016'], color='orange', label='Fine Val (Interpolated) NRMSE')
plots.append(val_fine_plot)

# Add legend and axis labels
plt.legend(handles=plots)
plt.xlabel('Epoch #')
plt.ylabel('NRMSE')
plt.ylim(0, 0.4)
plt.savefig(LOSS_PLOT + '_nrmse.png')
plt.close()

# Plot train coarse vs train fine losses
print('============== Train: Fine vs Coarse Losses ===============')
sif_utils.print_stats(train_fine_losses['CFIS_2016'], train_coarse_losses['CFIS_2016'], sif_mean, ax=plt.gca())
plt.xlabel('Train Coarse Losses')
plt.ylabel('Train Fine Losses')
plt.title('Fine vs Coarse train losses: ' + args.model)
plt.savefig(LOSS_PLOT + '_scatter_fine_vs_coarse.png')
plt.close()
