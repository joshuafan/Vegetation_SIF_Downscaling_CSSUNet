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
from unet.unet_model import UNet, UNetSmall, UNet2, UNet2PixelEmbedding, UNet2Larger, PixelNN, UNet2WithReconstruction
import visualization_utils
import sif_utils
import tile_transforms
import tqdm


# Folds
TRAIN_FOLDS = [0, 1, 2]
VAL_FOLDS = [3]
TEST_FOLDS = [4]

# Data files
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF"
METADATA_DIR = os.path.join(DATA_DIR, "metadata/CFIS_OCO2_dataset")
DATASET_2018_DIR = os.path.join(DATA_DIR, "metadata/dataset_2018")
# CFIS_TILE_METADATA_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_train.csv')
# CFIS_TILE_METADATA_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_val.csv')
# CFIS_TILE_METADATA_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_test.csv')
# OCO2_TILE_METADATA_TRAIN_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_train.csv')
# CFIS_FINE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_fine_metadata.csv')

# CFIS_COARSE_METADATA_FILE = os.path.join(CFIS_2016_DIR, 'cfis_coarse_metadata.csv')
# OCO2_2016_METADATA_FILE = os.path.join(OCO2_2016_DIR, 'oco2_metadata.csv')
# OCO2_2018_METADATA_FILE = os.path.join(DATASET_2018_DIR, 'oco2_metadata.csv')
# TROPOMI_2018_METADATA_FILE = os.path.join(DATASET_2018_DIR, 'tropomi_metadata.csv')

DATASET_FILES = {'CFIS_2016': os.path.join(METADATA_DIR, 'cfis_coarse_metadata.csv'),
                 'OCO2_2016': os.path.join(METADATA_DIR, 'oco2_metadata.csv'),
                 'OCO2_2018': os.path.join(DATASET_2018_DIR, 'oco2_metadata.csv'),
                 'TROPOMI_2018': os.path.join(DATASET_2018_DIR, 'tropomi_metadata.csv')}
COARSE_SIF_DATASETS = {'train': ['CFIS_2016', 'OCO2_2016'], #, 'TROPOMI_2018'], #, 'TROPOMI_2018'], # 'OCO2_2016', 'OCO2_2018', 'TROPOMI_2018'],
                       'val': ['CFIS_2016']} # ['CFIS_2016']}
FINE_SIF_DATASETS = {'train': ['CFIS_2016'],
                     'val': ['CFIS_2016']}
MODEL_SELECTION_DATASET = 'CFIS_2016' # 'OCO2_2016' #' 'CFIS_2016'
UPDATE_FRACTIONS = {'CFIS_2016': 1,
                    'OCO2_2016': 1,
                    'OCO2_2018': 0,
                    'TROPOMI_2018': 0}
BAND_STATISTICS_FILE = os.path.join(METADATA_DIR, 'cfis_band_statistics_train.csv')


# Dataset resolution/scale
RES = (0.00026949458523585647, 0.00026949458523585647)
TILE_PIXELS = 100
TILE_SIZE_DEGREES = RES[0] * TILE_PIXELS

# Method/model type
# METHOD = "9d_unet" #_contrastive"
# args.model = "unet"
# METHOD = "9d_unet2_local"
# args.model = "unet2"
# METHOD = "9d_unet2_larger"
# args.model = "unet2_larger"
# METHOD = "9d_unet2_pixel_embedding"
# args.model = "unet2_pixel_embedding"
# METHOD = "9d_unet2_contrastive"
# args.model = "unet2_pixel_embedding"
# METHOD = "9e_unet2_pixel_contrastive_3" #contrastive_2" #pixel_embedding"
# args.model = "unet2_pixel_embedding"
# METHOD = "9e_unet2_pixel_embedding"
# args.model = "unet2_pixel_embedding"
# METHOD = "9d_pixel_nn"
# args.model = "pixel_nn"
# METHOD = "9d_unet2_3"
# args.model = "unet2"
# METHOD = "10d_unet2_larger"
# args.model = "unet2_larger"
# METHOD = "10d_unet2_9"
# args.model = "unet2"
# METHOD = "10d_unet2_lr1e-4_weightdecay1e-4_multiplicative_fractionoutput0.2"
# args.model = "unet2"
# METHOD = "10d_pixel_nn"
# args.model = "pixel_nn"
# METHOD = "10d_unet2_larger"
# args.model = "unet2_larger"
# METHOD = "10e_unet2_contrastive"
# args.model = "unet2_pixel_embedding"
# METHOD = "9d_unet2"
# args.model = "unet2"
# METHOD = "11d_unet2_pixel_embedding"
# args.model = "unet2_pixel_embedding"
# METHOD = "11d_pixel_nn"
# args.model = "pixel_nn"
# METHOD = "2d_unet2"
# args.model = "unet2"
# METHOD = "tropomi_cfis_unet2"
# args.model = "unet"

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

# Contrastive pre-training
parser.add_argument('-pretrain_contrastive', "--pretrain_contrastive", default=False, action='store_true', help='Whether to pre-train initial layers in contrastive way (nearby/similar pixels should have similar representations)')
parser.add_argument('-freeze_pixel_encoder', "--freeze_pixel_encoder", default=False, action='store_true', help='If using pretrain_contrastive, whether to freeze the initial learned layers')

# Optional loss terms
parser.add_argument('-smoothness_loss', "--smoothness_loss", default=False, action='store_true', help='Whether to add a "smoothness loss" term (encouraging pixels with similar features to have similar SIF)')
parser.add_argument('-similarity_loss', "--similarity_loss", default=False, action='store_true', help='Whether to add a "similarity loss" term (encouraging pixels with similar features to have similar SIF)')
parser.add_argument('-similarity_temp', "--similarity_temp", default=0.2, type=float)
parser.add_argument('-crop_type_loss', "--crop_type_loss", default=False, action='store_true', help='Whether to add a "crop type loss" term (encouraging predictions for same crop type to be similar)')
parser.add_argument('-recon_loss', "--recon_loss", default=False, action='store_true', help='Whether to add a reconstruction loss')
parser.add_argument('-gradient_penalty', "--gradient_penalty", default=False, action='store_true', help="DOES NOT SEEM TO WORK. Whether to penalize gradient of SIF wrt input, to make function Lipschitz.")
parser.add_argument('-lambduh', "--lambduh", default=1, type=float, help="Gradient penalty lambda")

# Restricting output and input values
parser.add_argument('-min_sif', "--min_sif", default=None, type=float, help="If (min_sif, max_sif) are set, the model uses a tanh function to ensure the output is within that range.")
parser.add_argument('-max_sif', "--max_sif", default=None, type=float, help="If (min_sif, max_sif) are set, the model uses a tanh function to ensure the output is within that range.")
parser.add_argument('-min_sif_clip', "--min_sif_clip", default=0.1, type=float, help="Before computing loss, clip outputs below this to this value.")
parser.add_argument('-min_input', "--min_input", default=-3, type=float, help="Clip extreme input values to this many standard deviations below mean")
parser.add_argument('-max_input', "--max_input", default=3, type=float, help="Clip extreme input values to this many standard deviations above mean")
parser.add_argument('-reduced_channels', "--reduced_channels", default=None, type=int, help="If this is set, add a 'dimensionality reduction' layer to the front of the model, to reduce the number of channels to this.")

# Augmentations. None are enabled by default.
parser.add_argument('-normalize', "--normalize", action='store_true', help='Whether to normalize the reflectance bands to have norm 1. If this is enabled, the reflectance bands are NOT standardized.')
parser.add_argument('-fraction_outputs_to_average', "--fraction_outputs_to_average", default=0.2, type=float, help="Fraction of outputs to average when computing loss.")
parser.add_argument('-flip_and_rotate', "--flip_and_rotate", action='store_true')
parser.add_argument('-jigsaw', "--jigsaw", action='store_true')
parser.add_argument('-compute_vi', "--compute_vi", action='store_true', help="Whether to compute vegetation indices per pixel")
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
parser.add_argument('-label_noise', "--label_noise", default=0, type=float, help="Add random noise with this standard deviation to the label")

# Contrastive training settings
parser.add_argument('-contrastive_epochs', "--contrastive_epochs", default=50, type=int)
parser.add_argument('-contrastive_learning_rate', "--contrastive_learning_rate", default=1e-3, type=float)
parser.add_argument('-contrastive_weight_decay', "--contrastive_weight_decay", default=1e-3, type=float)
parser.add_argument('-contrastive_temp', "--contrastive_temp", default=0.2, type=float)
parser.add_argument('-pixel_pairs_per_image', "--pixel_pairs_per_image", default=5, type=int)

args = parser.parse_args()

# Set random seeds
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


CONTRASTIVE_BATCH_SIZE = args.batch_size * args.pixel_pairs_per_image

PARAM_SETTING = "{}_{}_optimizer-{}_bs-{}_lr-{}_wd-{}_maxepoch-{}_sche-{}_fractionoutputs-{}_seed-{}".format(  #_T0-{}_etamin-{}_step-{}_gamma-{}_
    args.prefix, args.model, args.optimizer, args.batch_size, args.learning_rate, args.weight_decay, args.max_epoch, args.scheduler, args.fraction_outputs_to_average, args.seed  #args.T0, args.eta_min, args.lrsteps, args.gamma,
)
if args.normalize:
    PARAM_SETTING += "_normalize"
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

if args.smoothness_loss:
    PARAM_SETTING += "_smoothnessloss-{}".format(args.lambduh)

# Model files
results_dir = os.path.join(DATA_DIR, "unet_results", PARAM_SETTING)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

PRETRAINED_MODEL_FILE = os.path.join(results_dir, "model.ckpt") #9e_unet2_contrastive")
MODEL_FILE = os.path.join(results_dir, "model.ckpt")

# Results files/plots
CFIS_RESULTS_CSV_FILE = os.path.join(results_dir, 'cfis_results_' + PARAM_SETTING + '.csv')
LOSS_PLOT = os.path.join(results_dir, 'losses')

# Summary csv file of all results. Create this if it doesn't exist
RESULTS_SUMMARY_FILE = os.path.join(DATA_DIR, "results_summary.csv")
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
TRAIN_DATES = ['2016-06-15', '2016-08-01'] #, '2016-08-01', "2018-06-10", "2018-06-24", "2018-07-08", "2018-07-22", "2018-08-05", "2018-08-19"]
TEST_DATES = ['2016-06-15', '2016-08-01'] #['2016-06-15', '2016-08-01']

ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']

# Which bands
# Just for the VI experiment
# BANDS = list(range(0, 48))
# RECONSTRUCTION_BANDS = list(range(0, 47))
# BANDS = list(range(0, 12)) + [17, 18, 19, 21, 22, 24, 28, 29, 30, 33, 39] + [47]
# RECONSTRUCTION_BANDS = list(range(0, 9)) + [17, 18, 19, 21, 22, 24, 28, 29, 30, 33, 39]  # list(range(12, 17)) 
# CROP_TYPE_INDICES = list(range(17, 47))

BANDS = list(range(0, 43))
RECONSTRUCTION_BANDS = list(range(0, 43))
# BANDS = list(range(0, 12)) + [12, 13, 14, 16, 17, 19, 23, 24, 25, 28, 34] + [42]
# RECONSTRUCTION_BANDS = list(range(0, 9)) + [12, 13, 14, 16, 17, 19, 23, 24, 25, 28, 34]
# BANDS = list(range(0, 12)) + list(range(12, 27)) + [28] + [42]
SIMILARITY_INDICES = list(range(0, 9))
CROP_TYPE_INDICES = list(range(12, 42))
MISSING_REFLECTANCE_IDX = len(BANDS) - 1
INPUT_CHANNELS = len(BANDS)
OUTPUT_CHANNELS = 1 
RECON_CHANNELS = len(RECONSTRUCTION_BANDS)



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
# print("Model:", args.model)
# print("Optimizer:", args.optimizer)
# print("Learning rate:", LEARNING_RATE)
# print("Weight decay:", WEIGHT_DECAY)
# print("Num workers:", NUM_WORKERS)
# print("Batch size:", BATCH_SIZE)
# print("Num epochs:", args.max_epoch)
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



def train_contrastive(args, model, dataloader, criterion, optimizer, device, pixel_pairs_per_image):
    model.train()
    for epoch in range(args.contrastive_epochs):
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

def calc_gradient_penalty(args, model, predicted_coarse_sifs, input_tiles_std):
    # # Get predicted pixel SIFs
    # input_tiles_std.requires_grad_(True)
    # outputs = model(input_tiles_std)  # predicted_fine_sifs_std: (batch size, output dim, H, W)
    # if type(outputs) == tuple:
    #     outputs = outputs[0]
    # predicted_fine_sifs_std = outputs[:, 0, :, :]  # torch.squeeze(predicted_fine_sifs_std, dim=1)
    # predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean

    # print("Predicted coarse sifs", predicted_coarse_sifs.shape, "Input tiles std", input_tiles_std.shape)
    gradients = autograd.grad(outputs=predicted_coarse_sifs, inputs=input_tiles_std,
                              grad_outputs=torch.ones(predicted_coarse_sifs.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    # print("Gradients shape", gradients.shape)
    gradients = gradients[:, 0:17, :, :] * 10000 #* args.batch_size
    # print("Gradients", gradients[0, 0, 50:55, 50:55])

    gradients = gradients.view(gradients.size(0), -1)
    # print("After view", gradients.shape)
    gradient_norms = gradients.norm(2, dim=1)
    gradient_penalty = (torch.clamp(gradient_norms - 1, min=0) ** 2).mean() # (torch.max(torch.zeros_like(gradient_norms), gradient_norms - 1) ** 2).mean() * lambduh
    return gradient_penalty



def similarity_loss(args, input_tiles_std, predicted_sif, device):
    # print("SIMILARITY LOSS")
    # print("Input tiles std shape", input_tiles_std.shape) # [batch x 24 x 100 x 100]
    # print("predicted sif shape", predicted_sif.shape)  # [batch x 100 x 100]
    height_idx = np.random.randint(low=0, high=input_tiles_std.shape[2])
    width_idx = np.random.randint(low=0, high=input_tiles_std.shape[3])

    # "Features" are actually predicted SIF. We want similar pixels to have similar predicted SIF.
    features = predicted_sif[:, height_idx, width_idx]
    # print("Debug: predicted pixel SIFs", features[0:5])
    difference_matrix = (features.reshape(-1, 1) - features) ** 2  # Representation similarity is just 1 minus absolute difference in SIF
    # print("Difference matrix:", difference_matrix[0:5, 0:5])

    # Compute which pixels have most similar input features
    pixel_inputs = input_tiles_std[:, :, height_idx, width_idx]
    # print("Pixel inputs", pixel_inputs[0:2])
    # print("Pixel inputs shape originally", pixel_inputs.shape, pixel_inputs[:, None].shape)
    pixel_differences = torch.norm(pixel_inputs[:, None] - pixel_inputs, dim=2)
    # print("Pixel differences", pixel_differences[0:2, 0:2])
    # sif_utils.plot_histogram(pixel_differences.cpu().detach().numpy().flatten(), "pixel_differences_euclidean.png", title="Euclidean distances between pixels")

    mask = (pixel_differences < 4).to(device)
    # print("Mask", mask[0:2, 0:2])
    # exit(0)
    # logits = difference_matrix # / args.similarity_temp

    logits_mask = torch.ones_like(mask, device=device)
    logits_mask.fill_diagonal_(0)
    mask = mask * logits_mask
    loss = difference_matrix[mask].mean()
    return loss



    # contrast_count = features.shape[1]
    # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    # if self.contrast_mode == 'one':
    #     anchor_feature = features[:, 0]
    #     anchor_count = 1
    # elif self.contrast_mode == 'all':
    #     anchor_feature = contrast_feature
    #     anchor_count = contrast_count
    # else:
    #     raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # # compute logits
    # anchor_dot_contrast = torch.div(
    #     torch.matmul(anchor_feature, contrast_feature.T),
    #     args.cont)
    # # for numerical stability
    # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    # logits = anchor_dot_contrast - logits_max.detach()

    # # tile mask
    # mask = mask.repeat(anchor_count, contrast_count)

    # # mask-out self-contrast cases
    # logits_mask = torch.scatter(
    #     torch.ones_like(mask),
    #     1,
    #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
    #     0
    # )
    # logits_mask = torch.ones_like(difference_matrix, device=device)
    # logits_mask.fill_diagonal_(0)
    # # print("Check: logits_mask", logits_mask[0:5, 0:5])
    # mask = mask * logits_mask
    # # print("Check: mask", mask[0:5, 0:5])
    # valid_rows = (mask == 1).any(dim=1)
    # mask = mask[valid_rows]
    # print("mask shape after removing invalid", mask.shape)
    # if mask.shape[0] == 0:
    #     return 0

    # logits_mask = logits_mask[valid_rows]
    # logits = logits[valid_rows]

    # # compute log_prob
    # exp_logits = torch.exp(logits) * logits_mask
    # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # # compute mean of log-likelihood over positive
    # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # # loss
    # loss = - mean_log_prob_pos #[~torch.isnan(mean_log_prob_pos)]
    # loss = loss.mean()

    # return loss
    
    # # labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
    # # labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # # labels = labels.to(self.args.device)

    # # features = F.normalize(features, dim=1)

    # # similarity_matrix = torch.matmul(features, features.T)
    # # assert similarity_matrix.shape == (
    # #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # # assert similarity_matrix.shape == labels.shape

    # # discard the main diagonal from both: labels and similarities matrix
    # mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
    # labels = labels[~mask].view(labels.shape[0], -1)
    # similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # # assert similarity_matrix.shape == labels.shape

    # # select and combine multiple positives
    # positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # # select only the negatives the negatives
    # negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # logits = torch.cat([positives, negatives], dim=1)
    # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    # logits = logits / self.args.temperature
    # return logits, labels


# Computes similarity between 0 and 1 between the two given pixel feature vectors.
# If the pixels are not of the same crop type, or if one of them is missing, assign
# a similarity of 0 (so that they are ignored in the smoothness loss). Otherwise,
# the similarity is a function of the distance between the feature vectors.
def pixel_similarity(pixel1, pixel2):
    if not torch.equal(pixel1[CROP_TYPE_INDICES], pixel2[CROP_TYPE_INDICES]):
        return 0
    if pixel1[MISSING_REFLECTANCE_IDX] == 1 or pixel2[MISSING_REFLECTANCE_IDX] == 1:
        return 0
    similarity = torch.exp(-0.1 * (torch.linalg.norm(pixel1[SIMILARITY_INDICES] - pixel2[SIMILARITY_INDICES]) ** 2))
    return similarity
    # similarity = torch.exp((-1 / len(SIMILARITY_INDICES)) * (torch.linalg.norm(pixel1[SIMILARITY_INDICES] - pixel2[SIMILARITY_INDICES]) ** 2))
    # return similarity  
    # if similarity > -1: # > 0.5:
    #     return similarity
    # else:
    #     return 0


def smoothness_loss(args, input_tiles_std, predicted_sif, device):
    assert input_tiles_std.shape[0] == predicted_sif.shape[0]
    assert input_tiles_std.shape[2] == predicted_sif.shape[1]
    assert input_tiles_std.shape[3] == predicted_sif.shape[2]
    assert input_tiles_std.shape[1] == 43
    assert input_tiles_std.shape[2] == 100
    assert input_tiles_std.shape[3] == 100
    num_features = input_tiles_std.shape[1]
    input_pixels = input_tiles_std.permute(0, 2, 3, 1)
    input_pixels = input_pixels.reshape(-1, num_features)
    sif_pixels = predicted_sif.flatten()
    indices = torch.randint(low=0, high=input_pixels.shape[0], size=(10000, 2))
    total_loss = 0
    num_points = 0
    num_zero_similarity = 0

    # TODO: Potentially vectorize this using "cdist" to improve efficiency, instead of for loop
    for i in range(indices.shape[0]):
        idx1 = indices[i, 0]
        idx2 = indices[i, 1]
        input_similarity = pixel_similarity(input_pixels[idx1], input_pixels[idx2])
        sif_difference = torch.abs(sif_pixels[idx1] - sif_pixels[idx2])
        total_loss += input_similarity * sif_difference
        if input_similarity != 0:
            num_points += 1
            if i % 100 == 0:
                print("Input 1", input_pixels[idx1][SIMILARITY_INDICES])
                print("Input 2", input_pixels[idx2][SIMILARITY_INDICES])
                print("Input similarity", input_similarity, "SIF difference", sif_difference.item())
        else:
            num_zero_similarity += 1
    # print("Num points with nonzero similarity", num_points, "with zero similarity", num_zero_similarity)
    return total_loss / num_points

    

def train_model(args, model, dataloaders, criterion, optimizer, device, sif_mean, sif_std):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    # Record losses at each epoch
    train_coarse_losses = {k: [] for k in COARSE_SIF_DATASETS['train']}
    val_coarse_losses = {k: [] for k in COARSE_SIF_DATASETS['val']}
    train_other_losses = {k: [] for k in COARSE_SIF_DATASETS['train']}
    val_other_losses = {k: [] for k in COARSE_SIF_DATASETS['val']}
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

    # Record the best-seen loss on the fine validation set (this is an alternate way to choose
    # the best model, if we are allowed to peek at the fine-resolution dataset for validation)
    # best_val_coarse_loss = float('inf')
    # best_val_fine_loss = float('inf')

    # Similarity loss criterion
    # similarity_criterion = torch.nn.CrossEntropyLoss().to(device)

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

            running_other_loss = {k: 0 for k in COARSE_SIF_DATASETS[phase]}
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
                    # print("input tiles shape", input_tiles_std.shape)
                    # input_tiles_std = input_tiles_std[:, BANDS, :, :]
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
                        # If penalizing gradient of pixel SIF w.r.t inputs, require gradient for input
                        if args.gradient_penalty:
                            input_tiles_std.requires_grad_(True)

                        # TODO check why extreme values occur
                        if torch.isnan(input_tiles_std).any():
                            print("input_tiles_std contained nan")
                            print(input_tiles_std)
                            exit(0)

                        # Pass tile through model to obtain fine-resolution SIF predictions
                        predicted_fine_sifs_std, predicted_reconstruction = model(input_tiles_std)  # predicted_fine_sifs_std: (batch size, output dim, H, W)
                        if (predicted_reconstruction is None) and args.recon_loss:
                            raise ValueError("The model you chose does not output a reconstruction, but you set --recon_loss" +
                                             "to True. Remove the --recon_loss flag to train the model without reconstruction loss.")
                        # print("Predicted fine sifs", predicted_fine_sifs_std.shape)
                        # print("Pred recon", predicted_reconstruction.shape)
                        # if type(outputs) == tuple:
                        #     outputs = outputs[0]
                        if torch.isnan(predicted_fine_sifs_std).any():
                            print("Predicted fine sifs contained nan")
                            print(predicted_fine_sifs_std)
                            exit(0)

                        predicted_fine_sifs_std = torch.squeeze(predicted_fine_sifs_std, dim=1)
                        predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean
                        # print("Predicted fine sifs", predicted_fine_sifs[0, 0:10, 0:10])
                        # print('Valid', torch.mean(valid_fine_sif_mask.float()))

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

                        if torch.isnan(predicted_coarse_sifs).any():
                            print('Predicted coarse SIFs', predicted_coarse_sifs)
                            for i in range(valid_fine_sif_mask.shape[0]):
                                print('Tile', i, 'num valid', torch.sum(valid_fine_sif_mask[i]))
                            print('Predicted nan')
                            exit(1)

                        # During training, inject noise into the true coarse labels to make model more robust
                        if phase == 'train' and args.label_noise != 0:
                            true_coarse_sifs = true_coarse_sifs + torch.randn(true_coarse_sifs.shape, device=device) * args.label_noise

                        # Compute loss (predicted vs true coarse SIF)
                        coarse_loss = criterion(true_coarse_sifs, predicted_coarse_sifs)
                        # if CROP_TYPE_LOSS:
                        #     coarse_loss += 0.0001 * sif_utils.crop_type_loss(predicted_fine_sifs, input_tiles_std, valid_fine_sif_mask)

                        # Compute reconstruction loss
                        # print("Reconstructed", outputs[:, 1:, :, :].shape)
                        # print("Reconstructed targets", reconstruction_target.shape)
                        # print("Flattened", reconstruction_target.flatten(start_dim=1).shape)
                        if args.recon_loss:
                            other_loss = criterion(predicted_reconstruction.flatten(start_dim=1),
                                                   reconstruction_target.flatten(start_dim=1))
                            loss = coarse_loss + 0.1*other_loss
                        elif args.smoothness_loss:
                            other_loss = smoothness_loss(args, input_tiles_std, predicted_fine_sifs, device)
                            loss = coarse_loss + other_loss * args.lambduh
                        elif args.gradient_penalty and phase == 'train':
                            other_loss = calc_gradient_penalty(args, model, predicted_coarse_sifs, input_tiles_std)
                            # print("Coarse loss", coarse_loss.item(), "--- Gradient penalty", other_loss.item())
                            loss = coarse_loss + other_loss * args.lambduh
                        else:
                            loss = coarse_loss

                        # Collect gradients w.r.t input (for visualization of training process)
                        if phase == 'train' and args.gradient_penalty:
                            gradients = autograd.grad(outputs=predicted_coarse_sifs, inputs=input_tiles_std,
                                grad_outputs=torch.ones(predicted_coarse_sifs.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                            # gradients = gradients[:, 0:17, :, :]
                            # print("Gradients shape", gradients.shape)
                            # print("Input tile grad", input_tiles_std.grad)

                        # Backpropagate coarse loss
                        if phase == 'train' and random.random() < UPDATE_FRACTIONS[dataset_name] and dataset_name in COARSE_SIF_DATASETS[phase]: # and not np.isnan(fine_loss.item()):
                            optimizer.zero_grad()
                            loss.backward()
                            # print('Grad', model.down1.maxpool_conv[1].double_conv[0].weight.grad)
                            optimizer.step()


                        # Compute/record losses
                        with torch.set_grad_enabled(False):
                            if dataset_name in COARSE_SIF_DATASETS[phase]:
                                running_coarse_loss[dataset_name] += coarse_loss.item() * len(true_coarse_sifs)
                                if args.recon_loss or args.smoothness_loss or args.gradient_penalty:
                                    running_other_loss[dataset_name] += other_loss.item() * len(true_coarse_sifs)
                                else:
                                    running_other_loss[dataset_name] += 0.
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

                                # # Backpropagate fine loss (SHOULD NOT BE USED EXCEPT FOR FULLY-SUPERVISED TRAINING)
                                # if phase == 'train': # and not np.isnan(fine_loss.item()):
                                #     optimizer.zero_grad()
                                #     fine_loss.backward()
                                #     optimizer.step()

                                running_fine_loss[dataset_name] += fine_loss.item() * len(true_fine_sifs_filtered)
                                num_fine_datapoints[dataset_name] += len(true_fine_sifs_filtered)
                                all_true_fine_sifs[dataset_name].append(true_fine_sifs_filtered.cpu().detach().numpy())
                                all_predicted_fine_sifs[dataset_name].append(predicted_fine_sifs_filtered.cpu().detach().numpy())

                                # VISUALIZATIONS
                                if epoch % 10 == 5 and len(all_true_fine_sifs[dataset_name]) == 1 and phase == 'train' and 'CFIS' in dataset_name:  # Only do first train batch
                                    base_plot_dir = os.path.join(results_dir, "example_predictions_epoch_" + str(epoch))
                                    for i in range(0, 30, 5):
                                        center_lat = sample['lat'][i].item()
                                        center_lon = sample['lon'][i].item()
                                        date = sample['date'][i]
                                        tile_description = 'lat_' + str(round(center_lat, 4)) + '_lon_' + str(round(center_lon, 4)) + '_' + date
                                        title = 'Lon ' + str(round(center_lon, 4)) + ', Lat ' + str(round(center_lat, 4)) + ', ' + date
                                        plot_dir = os.path.join(base_plot_dir, tile_description)
                                        if not os.path.exists(plot_dir):
                                            os.makedirs(plot_dir)

                                        # Plot input tile
                                        visualization_utils.plot_tile(input_tiles_std[i].cpu().detach().numpy(),
                                                            center_lon, center_lat, date, TILE_SIZE_DEGREES, 
                                                            tile_description=tile_description, title=title, plot_dir=plot_dir)

                                        # Plot predictions against ground truth
                                        visualization_utils.plot_tile_predictions(input_tiles_std[i].cpu().detach().numpy(),
                                                                    tile_description,
                                                                    true_fine_sifs[i].cpu().detach().numpy(),
                                                                    [predicted_fine_sifs[i].cpu().detach().numpy()],
                                                                    valid_mask_numpy[i],
                                                                    non_noisy_mask[i].cpu().detach().numpy(),
                                                                    ['U-Net'], 
                                                                    center_lon, center_lat, date,
                                                                    TILE_SIZE_DEGREES, res=30,
                                                                    plot_dir=plot_dir)

                                        # # Plot gradients wrt input
                                        # visualization_utils.plot_individual_bands(gradients[i].cpu().detach().numpy(),
                                        #                                           title, center_lon, center_lat, TILE_SIZE_DEGREES, 
                                        #                                           os.path.join(plot_dir, "input_gradients.png"),
                                        #                                           crop_type_start_idx=12, num_grid_squares=4,
                                        #                                           decimal_places=3, min_feature=None, max_feature=None)
                                        
                                        # # Print norm of gradients in model
                                        # total_norm = 0
                                        # for p in model.parameters():
                                        #     param_norm = p.grad.detach().data.norm(2)
                                        #     total_norm += param_norm.item() ** 2
                                        # total_norm = total_norm ** 0.5
                                        # print("Overall gradient norm", total_norm)

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
            for coarse_dataset in COARSE_SIF_DATASETS[phase]:
                # Record reconstruction loss
                epoch_other_loss = running_other_loss[coarse_dataset] / num_coarse_datapoints[coarse_dataset]

                # Compute and record coarse SIF loss
                epoch_coarse_nrmse = math.sqrt(running_coarse_loss[coarse_dataset] / num_coarse_datapoints[coarse_dataset]) / sif_mean
                if phase == 'train':
                    train_other_losses[coarse_dataset].append(epoch_other_loss)
                    train_coarse_losses[coarse_dataset].append(epoch_coarse_nrmse)
                else:
                    val_other_losses[coarse_dataset].append(epoch_other_loss)
                    val_coarse_losses[coarse_dataset].append(epoch_coarse_nrmse)
                true_coarse = np.concatenate(all_true_coarse_sifs[coarse_dataset])
                predicted_coarse = np.concatenate(all_predicted_coarse_sifs[coarse_dataset])
                print('===== ', phase, coarse_dataset, 'Coarse stats ====')
                if args.recon_loss:
                    print("Reconstruction loss", epoch_other_loss)
                if args.smoothness_loss:
                    print("Smoothness loss", epoch_other_loss)
                if args.gradient_penalty:
                    print("Gradient penalty", epoch_other_loss)
                # print("True coarse", true_coarse[-20:])
                # print("Predicted coarse", predicted_coarse[-20:])
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



            # If the model performed better on "MODEL_SELECTION_DATASET" validation set than the
            # best previous model, record losses for this epoch, and save this model
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

            # if phase == 'train' and epoch_coarse_nrmse < best_train_coarse_loss:
            #     best_train_coarse_loss = epoch_coarse_nrmse
            #     torch.save(model.state_dict(), MODEL_FILE + '_best_train')


        # Print elapsed time per epoch
        epoch_time = time.time() - epoch_start
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            epoch_time // 60, epoch_time % 60))
        print()

        # If fine validation loss did not improve for more than "EARLY_STOPPING_PATIENCE" consecutive epochs, stop training.
        if epoch >= 1 and val_fine_losses[MODEL_SELECTION_DATASET][-1] < val_fine_losses[MODEL_SELECTION_DATASET][-2]:
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1
            if num_epochs_no_improvement > args.patience:
                break


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("==== All losses =====")
    print('train fine:', train_fine_losses["CFIS_2016"])
    print("val fine:", val_fine_losses["CFIS_2016"])
    print('=================== At best val fine (epoch ' + str(best_val_fine_epoch) + ') ========================')
    print('train coarse losses:', train_coarse_loss_at_best_val_fine)
    print('train fine loss:', train_fine_loss_at_best_val_fine)
    print('val coarse loss:', val_coarse_loss_at_best_val_fine)
    print('val fine loss:', val_fine_loss_at_best_val_fine)
    print('=================== At best val coarse (epoch ' + str(best_val_coarse_epoch) + ') ========================')
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
    return model, train_coarse_losses, val_coarse_losses, train_fine_losses, val_fine_losses, train_other_losses, val_other_losses


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
# print("Means", train_means)
# print("Stds", train_stds)
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
if args.normalize:
    normalize_transform = tile_transforms.NormalizeReflectance(reflectance_bands=list(range(0, 9)))
    standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds, bands_to_transform=list(range(9, 12)))
    clip_transform = tile_transforms.ClipTile(min_input=args.min_input, max_input=args.max_input, bands_to_clip=list(range(9, 12)))
else:
    standardize_transform = tile_transforms.StandardizeTile(band_means, band_stds, bands_to_transform=list(range(0, 12))) #, min_input=MIN_INPUT, max_input=MAX_INPUT)
    clip_transform = tile_transforms.ClipTile(min_input=args.min_input, max_input=args.max_input, bands_to_transform=list(range(0, 12)))
# color_distortion_transform = tile_transforms.ColorDistortion(continuous_bands=list(range(0, 12)), standard_deviation=args.color_distortion_std)
noise_transform = tile_transforms.GaussianNoise(continuous_bands=list(range(0, 9)), standard_deviation=args.gaussian_noise_std)
multiplicative_noise_transform = tile_transforms.MultiplicativeGaussianNoise(continuous_bands=list(range(0, 9)), standard_deviation=args.mult_noise_std)
flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()
compute_vi_transform = tile_transforms.ComputeVegetationIndices()
jigsaw_transform = tile_transforms.RandomJigsaw()
resize_transform = tile_transforms.ResizeTile(target_dim=[args.resize_dim, args.resize_dim])
random_crop_transform = tile_transforms.RandomCrop(crop_dim=args.crop_dim)
cutout_transform = tile_transforms.Cutout(cutout_dim=args.cutout_dim, prob=args.cutout_prob)
transform_list_train = []  #[standardize_transform, clip_transform] # [standardize_transform, noise_transform]
transform_list_val = [] # [standardize_transform, clip_transform] #[standardize_transform]

# Add vegetation index calculations at beginning, BEFORE standardizing. Note that vegetation
# indices are already normalized to a reasonalbe range, so no need to standardize them.
if args.compute_vi:
    transform_list_train.append(compute_vi_transform)
    transform_list_val.append(compute_vi_transform)

# Apply cutout at beginning, since it is supposed to zero pixels out BEFORE standardizing
if args.cutout:
    transform_list_train.append(cutout_transform)

# Add multiplicative noise at beginning
if args.multiplicative_noise:
    transform_list_train.append(multiplicative_noise_transform)

# Standardize/clip the continuous variables
if args.normalize:
    transform_list_train.extend([normalize_transform, standardize_transform, clip_transform])
    transform_list_val.extend([normalize_transform, standardize_transform, clip_transform])
else:
    transform_list_train.extend([standardize_transform, clip_transform]) # [standardize_transform, noise_transform]
    transform_list_val.extend([standardize_transform, clip_transform])

# More augmentations
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

    if '2018' in dataset_name:
        metadata['SIF'] /= 1.52
        # print(metadata['SIF'].head())

    # Read dataset splits
    if dataset_name == 'OCO2_2016' or dataset_name == 'CFIS_2016':
        train_set = metadata[(metadata['fold'].isin(TRAIN_FOLDS)) &
                            (metadata['date'].isin(TRAIN_DATES))].copy()
        val_set = metadata[(metadata['fold'].isin(VAL_FOLDS)) &
                        (metadata['date'].isin(TRAIN_DATES))].copy()
        test_set = metadata[(metadata['fold'].isin(TEST_FOLDS)) &
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
PARAM_STRING += ('Min SIF clip: ' + str(args.min_sif_clip) + '\n')
PARAM_STRING += ('Max cloud cover (coarse CFIS): ' + str(MAX_CFIS_CLOUD_COVER) + '\n')
PARAM_STRING += ('Min fraction valid pixels in CFIS tile: ' + str(MIN_COARSE_FRACTION_VALID_PIXELS) + '\n')
PARAM_STRING += ('Train features: ' + str(BANDS) + '\n')
PARAM_STRING += ("Clip input features: " + str(args.min_input) + " to " + str(args.max_input) + " standard deviations from mean\n")
# if REMOVE_PURE_TRAIN:
#     PARAM_STRING += ('Removing pure train tiles above ' + str(PURE_THRESHOLD_TRAIN) + '\n')
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
if args.label_noise != 0:
    PARAM_STRING += ("\tLabel noise: std " + str(args.label_noise))
PARAM_STRING += ("Fraction outputs to average: " + str(args.fraction_outputs_to_average) + '\n')
PARAM_STRING += ("SIF range: " + str(args.min_sif) + " to " + str(args.max_sif) + '\n')
PARAM_STRING += ("SIF statistics: mean " + str(sif_mean) + ", std " + str(sif_std) + '\n')
if args.pretrain_contrastive:
    PARAM_STRING += ('============ CONTRASTIVE PARAMS ===========\n')
    PARAM_STRING += ("Learning rate: " + str(args.contrastive_learning_rate) + '\n')
    PARAM_STRING += ("Weight decay: " + str(args.contrastive_weight_decay) + '\n')
    PARAM_STRING += ("Batch size: " + str(args.contrastive_batch_size) + '\n')
    PARAM_STRING += ("Num epochs: " + str(args.contrastive_epochs) + '\n')
    PARAM_STRING += ("Temperature: " + str(args.contrastive_temp) + '\n')
PARAM_STRING += ("==============================================================\n")
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
if args.model == 'unet2':
    unet_model = UNet2(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, recon_channels=RECON_CHANNELS,
                       reduced_channels=args.reduced_channels, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'unet2_with_reconstruction':
    unet_model = UNet2WithReconstruction(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, recon_channels=RECON_CHANNELS,
                                        reduced_channels=args.reduced_channels, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'pixel_nn':
    unet_model = PixelNN(input_channels=INPUT_CHANNELS, output_dim=OUTPUT_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'unet':
    unet_model = UNet(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, recon_channels=RECON_CHANNELS,
                      reduced_channels=args.reduced_channels, min_output=min_output, max_output=max_output).to(device)   
else:
    print('Model type not supported')
    exit(1)


# If we're loading a pre-trained model, read model params from file
if args.from_pretrained:
    unet_model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE, map_location=device))

# Initialize loss and optimizer
criterion = nn.MSELoss(reduction='mean')
# criterion = nn.SmoothL1Loss(reduction='mean')
if args.optimizer == "Adam":
    optimizer = optim.Adam(unet_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer == "AdamW":
    optimizer = optim.AdamW(unet_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
else:
    print("Optimizer not supported")
    exit(1)

if args.pretrain_contrastive:
    contrastive_loss = nt_xent.NTXentLoss(device, CONTRASTIVE_BATCH_SIZE, args.contrastive_temp, True)
    contrastive_optimizer = optim.Adam(unet_model.parameters(), lr=args.contrastive_learning_rate, weight_decay=args.contrastive_weight_decay)
    contrastive_dataloader = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size, 
                                                         shuffle=True, num_workers=args.num_workers, drop_last=True)
    print('Contrastive training!')
    # Train pixel embedding
    unet_model = train_contrastive(args, unet_model, contrastive_dataloader, contrastive_loss, contrastive_optimizer, device,
                                   pixel_pairs_per_image=args.pixel_pairs_per_image)

# Freeze pixel embedding layers
if args.freeze_pixel_encoder:
    unet_model.dimensionality_reduction_1.requires_grad = False
    unet_model.inc.requires_grad = False

# Train model to predict SIF
unet_model, train_coarse_losses, val_coarse_losses, train_fine_losses, val_fine_losses, train_other_losses, val_other_losses = train_model(args, unet_model, dataloaders, criterion, optimizer, device, sif_mean, sif_std)
torch.save(unet_model.state_dict(), MODEL_FILE)

# Plot loss curves: NRMSE
epoch_list = range(len(train_coarse_losses['CFIS_2016']))
plots = []
# print("Coarse Train NRMSE:", train_coarse_losses['CFIS_2016'])
train_coarse_plot, = plt.plot(epoch_list, train_coarse_losses['CFIS_2016'], color='blue', label='Coarse Train NRMSE (CFIS)')
plots.append(train_coarse_plot)
# print("Coarse train recon loss:", train_other_losses['CFIS_2016'])
if args.recon_loss:
    train_other_plot, = plt.plot(epoch_list, train_other_losses['CFIS_2016'], color='black', label='Other train loss')
    plots.append(train_other_plot)
elif args.gradient_penalty:
    train_other_plot, = plt.plot(epoch_list, train_other_losses['CFIS_2016'], color='black', label='Gradient penalty')
    plots.append(train_other_plot)
if 'OCO2_2016' in COARSE_SIF_DATASETS:
    # print("OCO2 Train NRMSE:", train_coarse_losses['OCO2_2016'])
    train_oco2_plot, = plt.plot(epoch_list, train_coarse_losses['OCO2_2016'], color='purple', label='Coarse Train NRMSE (OCO-2)')
    plots.append(train_oco2_plot)
# print("Coarse Val NRMSE:", val_coarse_losses['CFIS_2016'])
val_coarse_plot, = plt.plot(epoch_list, val_coarse_losses['CFIS_2016'], color='green', label='Coarse Val NRMSE')
plots.append(val_coarse_plot)
# print("Coarse val recon loss:", val_other_losses['CFIS_2016'])
if args.recon_loss:
    val_recon_plot, = plt.plot(epoch_list, val_other_losses['CFIS_2016'], color='gray', label='Val reconstruction loss')
    plots.append(val_recon_plot)
# print("Fine Train NRMSE:", train_fine_losses['CFIS_2016'])
train_fine_plot, = plt.plot(epoch_list, train_fine_losses['CFIS_2016'], color='red', label='Fine Train (Interpolated) NRMSE')
plots.append(train_fine_plot)
# print("Fine Val NRMSE:", val_fine_losses['CFIS_2016'])
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
