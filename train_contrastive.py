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

from datasets import CombinedDataset, CoarseSIFDataset, FineSIFDataset
from unet.unet_model import UNetContrastive, UNet2Contrastive, UNet, UNetSmall, UNet2, UNet2PixelEmbedding, UNet2Larger, PixelNN, UNet2Spectral
import visualization_utils
import sif_utils
import tile_transforms
import tqdm
import mtadam

# Folds
TRAIN_FOLDS = [0, 1, 2]
VAL_FOLDS = [3]
TEST_FOLDS = [4]

# Data files
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF"
METADATA_DIR = os.path.join(DATA_DIR, "metadata/CFIS_OCO2_dataset")
DATASET_2018_DIR = os.path.join(DATA_DIR, "metadata/dataset_2018")
DATASET_FILES = {'CFIS_2016': os.path.join(METADATA_DIR, 'cfis_coarse_metadata.csv'),
                 'OCO2_2016': os.path.join(METADATA_DIR, 'oco2_metadata.csv'),
                 'OCO2_2018': os.path.join(DATASET_2018_DIR, 'oco2_metadata.csv'),
                 'TROPOMI_2018': os.path.join(DATASET_2018_DIR, 'tropomi_metadata.csv')}
COARSE_SIF_DATASETS = {'train': ['CFIS_2016', 'OCO2_2016'],
                       'val': ['CFIS_2016']} 
FINE_SIF_DATASETS = {'train': ['CFIS_2016'],
                     'val': ['CFIS_2016']}
MODEL_SELECTION_DATASET = 'CFIS_2016'

# Ratios - how often a batch from each dataset should be sampled during training
UPDATE_FRACTIONS = {'CFIS_2016': 1,
                    'OCO2_2016': 1,
                    'OCO2_2018': 0,
                    'TROPOMI_2018': 0}

# Dataset resolution/scale
RES = (0.00026949458523585647, 0.00026949458523585647)  # in degrees
TILE_PIXELS = 100
TILE_SIZE_DEGREES = RES[0] * TILE_PIXELS

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-prefix', "--prefix", default='10d_', type=str, help='prefix')
parser.add_argument('-model', "--model", default='unet2', type=str, help='model type')

# Random seed
parser.add_argument('-seed', "--seed", default=0, type=int)

# Optimizer params
parser.add_argument('-optimizer', "--optimizer", default='AdamW', choices=["Adam", "AdamW", "MTAdam"], type=str, help='optimizer type')
parser.add_argument('-lr', "--learning_rate", default=1e-4, type=float, help='initial learning rate')
parser.add_argument('-wd', "--weight_decay", default=1e-4, type=float, help='weight decay rate')

# Training params
parser.add_argument('-epoch', "--max_epoch", default=100, type=int, help='max epoch to train')
parser.add_argument('-patience', "--patience", default=5, type=int, help="Early stopping patience (stop if val loss doesn't improve for this many epochs)")
parser.add_argument('-bs', "--batch_size", default=100, type=int, help="Batch size")
parser.add_argument('-num_workers', "--num_workers", default=4, type=int, help="Number of dataloader workers")
parser.add_argument('-from_pretrained', "--from_pretrained", default=False, action='store_true', help='Whether to initialize from pre-trained model')
parser.add_argument('-visualize', "--visualize", default=False, action='store_true', help='Plot visualizations')

# ============ Smoothness/contrastive losses : general params ============
parser.add_argument('-num_pixels', "--num_pixels", default=1000, type=int, help='Number of pixels to sample per batch for smoothness/contrastive losses')
parser.add_argument('-lambduh', "--lambduh", default=1, type=float, help="How much to weight the smoothness/contrastive loss compared to coarse")
parser.add_argument('-adaptive_lambda', "--adaptive_lambda", default=False, action='store_true', help='Plot visualizations')
parser.add_argument('-pretrain_epochs', "--pretrain_epochs", default=0, type=int, help='If this is greater than 0, pre-train this number of epochs with ONLY unsupervised contrastive loss, then introduce supervised coarse loss')

# Pixel contrastive loss
parser.add_argument('-pixel_contrastive_loss', "--pixel_contrastive_loss", default=False, action='store_true', help='Whether to add a "smoothness loss" term (encouraging pixels with similar features to have similar SIF)')
parser.add_argument('-temperature', "--temperature", default=0.1, type=float, help='')
parser.add_argument('-similarity_threshold', "--similarity_threshold", default=0.05, type=float, help='Threshold for considering pixels to be "similar". Defined as Euclidean distance between normalized reflectances (over bands defined in SIMILARITY_INDICES)')

# Smoothness loss
parser.add_argument('-smoothness_loss', "--smoothness_loss", default=False, action='store_true', help='Whether to add a "smoothness loss" term (encouraging pixels with similar features to have similar SIF)')
parser.add_argument('-spread', "--spread", default=50, type=float, help='A parameter when computing similarity between pixel reflectances. The higher this parameter is, the faster the similarities decay towards 0 as distance increases.')

# Other smoothness loss variants
parser.add_argument('-smoothness_loss_contrastive', "--smoothness_loss_contrastive", default=False, action='store_true', help='Whether to add a "smoothness loss" term (encouraging pixels to have more similar SIF to its most similar pixel, and different SIF from other pixels)')
parser.add_argument('-similarity_loss', "--similarity_loss", default=False, action='store_true', help='Whether to add a "similarity loss" term (encouraging pixels with similar features to have similar SIF)')

# Gradient penalty
parser.add_argument('-gradient_penalty', "--gradient_penalty", default=False, action='store_true', help="DOES NOT SEEM TO WORK. Whether to penalize gradient of SIF wrt input, to make function Lipschitz.")
parser.add_argument('-norm_penalty_threshold', "--norm_penalty_threshold", default=50, type=float, help='Only penalize gradient norms above this threshold.')

# Whether to train using fine-resolution supervision
parser.add_argument('-fine_supervision', "--fine_supervision", default=False, action='store_true', help='Use this flag to train with full *fine-resolution* labels')

# Restricting output and input values
parser.add_argument('-min_sif', "--min_sif", default=None, type=float, help="If (min_sif, max_sif) are set, the model uses a tanh function to ensure the output is within that range.")
parser.add_argument('-max_sif', "--max_sif", default=None, type=float, help="If (min_sif, max_sif) are set, the model uses a tanh function to ensure the output is within that range.")
parser.add_argument('-min_sif_clip', "--min_sif_clip", default=0.1, type=float, help="Before computing loss, clip outputs below this to this value.")
parser.add_argument('-min_input', "--min_input", default=-3, type=float, help="Clip extreme input values to this many standard deviations below mean")
parser.add_argument('-max_input', "--max_input", default=3, type=float, help="Clip extreme input values to this many standard deviations above mean")
parser.add_argument('-reduced_channels', "--reduced_channels", default=None, type=int, help="If this is set, add a 'dimensionality reduction' layer to the front of the model, to reduce the number of channels to this.")

# Augmentations. None are enabled by default.
parser.add_argument('-normalize', "--normalize", action='store_true', help='Whether to normalize the reflectance bands to have norm 1. If this is enabled, the reflectance bands are NOT standardized.')
parser.add_argument('-fraction_outputs_to_average', "--fraction_outputs_to_average", default=1, type=float, help="Fraction of outputs to average when computing loss.")
parser.add_argument('-flip_and_rotate', "--flip_and_rotate", action='store_true')
parser.add_argument('-jigsaw', "--jigsaw", action='store_true')
parser.add_argument('-compute_vi', "--compute_vi", action='store_true', help="Whether to compute vegetation indices per pixel")
parser.add_argument('-multiplicative_noise', "--multiplicative_noise", action='store_true')
parser.add_argument('-multiplicative_noise_end', "--multiplicative_noise_end", action='store_true')
parser.add_argument('-mult_noise_std', "--mult_noise_std", default=0, type=float, help="If the 'multiplicative_noise' augmentation is used, multiply each channel by (1+eps), where eps ~ N(0, mult_noise_std)")
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

# Contrastive training settings - TODO

args = parser.parse_args()

# Set random seeds
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Band statistics (mean/std) for standardizing each channel. Depends on whether we're first normalizing reflectance vectors to norm 1 or not.
if args.normalize:
    BAND_STATISTICS_FILE = os.path.join(METADATA_DIR, 'normalized_cfis_band_statistics_train.csv')
else:
    BAND_STATISTICS_FILE = os.path.join(METADATA_DIR, 'cfis_band_statistics_train.csv')

# Create param string
PARAM_SETTING = "{}_{}_optimizer-{}_bs-{}_lr-{}_wd-{}_fractionoutputs-{}_seed-{}".format(  #_T0-{}_etamin-{}_step-{}_gamma-{}_
    args.prefix, args.model, args.optimizer, args.batch_size, args.learning_rate, args.weight_decay, args.fraction_outputs_to_average, args.seed  #args.T0, args.eta_min, args.lrsteps, args.gamma,
)
args.special_options = ""
if args.normalize:
    args.special_options += "_normalize"
if args.flip_and_rotate:
    args.special_options += "_fliprotate"
if args.jigsaw:
    args.special_options += "_jigsaw"
if args.multiplicative_noise:
    args.special_options += "_multiplicativenoise-{}".format(args.mult_noise_std)
if args.multiplicative_noise_end:
    args.special_options += "_multiplicativenoiseend-{}".format(args.mult_noise_std)
if args.gaussian_noise:
    args.special_options += "_gaussiannoise-{}".format(args.mult_noise_std)
if args.cutout:
    args.special_options += "_cutout-{}_prob-{}".format(args.cutout_dim, args.cutout_prob)
if args.resize:
    args.special_options += "_resize-{}".format(args.resize_dim)
if args.random_crop:
    args.special_options += "_randomcrop-{}".format(args.crop_dim)
if args.fine_supervision:
    args.special_options += "_finesupervision"
if args.similarity_loss:
    args.special_options += "_similarityloss-{}".format(args.lambduh)
if args.smoothness_loss:
    args.special_options += "_smoothnessloss-{}-s{}".format(args.lambduh, args.spread)
if args.smoothness_loss_contrastive:
    args.special_options += "_smoothnesslosscontrastive-{}".format(args.lambduh)
if args.pixel_contrastive_loss:
    args.special_options += "_pixelcontrastive-{}".format(args.lambduh)
PARAM_SETTING += args.special_options

# Model files
results_dir = os.path.join(DATA_DIR, "unet_results", PARAM_SETTING)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

PRETRAINED_MODEL_FILE = os.path.join(results_dir, "model.ckpt")
MODEL_FILE = os.path.join(results_dir, "model.ckpt")

# Results files/plots
CFIS_RESULTS_CSV_FILE = os.path.join(results_dir, 'cfis_results_' + PARAM_SETTING + '.csv')
LOSS_PLOT = os.path.join(results_dir, 'losses')

# Summary csv file of all results. Create this if it doesn't exist
RESULTS_SUMMARY_FILE = os.path.join(DATA_DIR, "unet_results/results_summary.csv")
if not os.path.isfile(RESULTS_SUMMARY_FILE):
    with open(RESULTS_SUMMARY_FILE, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['model', 'git_commit', 'command', 'optimizer', 'batch_size', 'lr', 'wd', 'mult_noise', 'special_options', 'fine_val_nrmse', 'path_to_model', 'best_val_fine_epoch'])
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

# Cover columns to use
ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']

# Bands to use
# BANDS = list(range(0, 7)) + [12, 13, 14, 16, 17, 19, 23, 24, 25, 28, 34] + [42]
# REFLECTANCE_INDICES = list(range(0, 7))  # Indices of reflectance bands, used for multiplicative noise/normalization
# CONTINUOUS_INDICES = list(range(0, 7))  # Indices of continuous bands to standardize/clip
# SIMILARITY_INDICES = list(range(0, 7))  # Indices to use when computing pixel similarity (within BANDS)
# CROP_TYPE_INDICES = list(range(7, 18))  # Indices of crop type bands (within BANDS)

BANDS = list(range(0, 7)) + list(range(9, 12)) + [12, 13, 14, 16, 17, 19, 23, 24, 25, 28, 34] + [42]
REFLECTANCE_INDICES = list(range(0, 7))  # Indices of reflectance bands, used for multiplicative noise/normalization
CONTINUOUS_INDICES = list(range(0, 12))  # Indices of continuous bands to standardize/clip
SIMILARITY_INDICES = list(range(0, 7))  # Indices to use when computing pixel similarity (within BANDS)
CROP_TYPE_INDICES = list(range(10, 21))  # Indices of crop type bands (within BANDS)

# BANDS = list(range(0, 12)) + [12, 13, 14, 16, 17, 19, 23, 24, 25, 28, 34] + [42]
# SIMILARITY_INDICES = list(range(0, 7))
# CROP_TYPE_INDICES = list(range(12, 23))
MISSING_REFLECTANCE_IDX = len(BANDS) - 1
INPUT_CHANNELS = len(BANDS)
OUTPUT_CHANNELS = 1 





def get_neighbor_pixel_indices(images, radius=10, pixel_pairs_per_image=10, num_tries=3):
    """Selects indices of pixels that are within "radius" of each other, ideally with the same cover type.

    Args:
        images: batch of images with shape [batch, channels, height, width]
        pixel_pairs_per_image: how many pairs of nearby pixels to extract per image
        num_tries: max number of attempts when searching for nearby pixels with same cover type. If
                   this number of attempts is exceeded, give up on trying to find nearby pixels with
                   the same cover type, and just pick a random pair of nearby pixels.

    Returns: two Tensors, each of shape [batch, pixel_pairs_per_image, 2], where "2" is (height_idx, width_idx)

    TODO "radius" is not quite the right terminology
    """
    indices1 = torch.zeros((images.shape[0], pixel_pairs_per_image, 2))
    indices2 = torch.zeros((images.shape[0], pixel_pairs_per_image, 2))
    for image_idx in range(images.shape[0]):
        for pair_idx in range(pixel_pairs_per_image):
            # Randomly choose anchor pixel
            for i in range(num_tries):
                anchor_height_idx = np.random.randint(0, images.shape[2])
                anchor_width_idx = np.random.randint(0, images.shape[3])

                # If the pixel is missing reflectance data, choose anther pixel
                if images[image_idx, MISSING_REFLECTANCE_IDX, anchor_height_idx, anchor_width_idx] == 1:
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

    return indices1, indices2


# def train_contrastive(args, model, dataloader, criterion, optimizer, device, pixel_pairs_per_image):
#     """Train contrastive pixel embedding, using nearby pixels of same cover type as positive pairs, and everything else as negative pairs.
#     """ 
#     model.train()
#     for epoch in range(args.contrastive_epochs):
#         train_loss, total = 0, 0
#         pbar = tqdm.tqdm(dataloader, total=len(dataloader), desc='train epoch {}'.format(epoch))
#         for combined_sample in pbar:
#             for dataset_name, sample in combined_sample.items():

#                 input_tiles = sample['input_tile'].to(device)
#                 _, pixel_embeddings = model(input_tiles)

#                 indices1, indices2 = get_neighbor_pixel_indices(input_tiles, pixel_pairs_per_image=pixel_pairs_per_image)
#                 # print(indices1.shape)
#                 batch_size = indices1.shape[0]
#                 assert batch_size == input_tiles.shape[0]
#                 assert batch_size == indices2.shape[0]
#                 num_points = indices1.shape[1]
#                 assert num_points == indices2.shape[1]
#                 indices1 = indices1.reshape((batch_size * num_points, 2)).long()
#                 indices2 = indices2.reshape((batch_size * num_points, 2)).long()
#                 # print('indices1 shape', indices1.shape)
#                 bselect = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, num_points).flatten()
#                 # print('Bselect shape', bselect.shape)

#                 embeddings1 = pixel_embeddings[bselect, :, indices1[:, 0], indices1[:, 1]]
#                 embeddings2 = pixel_embeddings[bselect, :, indices2[:, 0], indices2[:, 1]]
#                 # print('Embeddings shape', embeddings1.shape)  # [batch_size * num_points, reduced_dim]

#                 #normalize
#                 embeddings1 = F.normalize(embeddings1, dim=1)
#                 embeddings2 = F.normalize(embeddings2, dim=1)

#                 loss = criterion(embeddings1, embeddings2)
#                 # output_perm = torch.cat((output1[2:], output1[:2]))
#                 #loss, l_n, l_d = triplet_loss(output1, output2, output_perm, margin=args.triplet_margin, l2=args.triplet_l2)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()
#                 total += batch_size
#                 pbar.set_postfix(avg_loss=train_loss/total)
#     return model


def calc_gradient_penalty(args, input_tiles_std, predicted_coarse_sifs, return_gradients=False):
    """Place a penalty if gradient norm (of output w.r.t. input pixels) exceeds "norm_penalty_threshold".
    Before this method, you first need to call "input_tiles_std.requires_grad_(True)", then pass tiles
    through the model to obtain "predicted_coarse_sifs".

    Args:
        input_tiles_std: standardized input images, shape [batch, channels, height, width].
        predicted_coarse_sifs: predicted coarse tile SIFs, shape [batch]
        norm_penalty_threshold: penalize gradient norms that are above this threshold.
                                If the norm of the gradient is below this threshold, no penalty.
        return_gradients: whether to return raw gradients (of output w.r.t. input pixels) for visualization

    In addition, the following command-line arguments in "args" are used:
        args.norm_penalty_threshold: Only penalize gradient norms above this threshold

    Returns:
        If "return_gradients" is True, returns a tuple of gradient_penalty (scalar loss) and raw_gradients [batch, channels, height, width].
        Otherwise, just return the gradient_penalty (scalar loss).
    """
    assert input_tiles_std.shape[0] == predicted_coarse_sifs.shape[0]
    assert len(input_tiles_std.shape) == 4

    # print("Predicted coarse sifs", predicted_coarse_sifs.shape, "Input tiles std", input_tiles_std.shape)
    raw_gradients = autograd.grad(outputs=predicted_coarse_sifs, inputs=input_tiles_std,
                                  grad_outputs=torch.ones(predicted_coarse_sifs.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
    # print("Gradients shape", gradients.shape)
    gradients = raw_gradients[:, SIMILARITY_INDICES, :, :] * 10000 #* args.batch_size
    # print("Gradients", gradients[0, 0, 50:55, 50:55])

    # Reshape gradient such that each pixel is its own row: [# pixels, input features]
    gradients = gradients.view(gradients.size(0), -1)
    # print("After view", gradients.shape)
    gradient_norms = gradients.norm(2, dim=1)
    gradient_penalty = (torch.clamp(gradient_norms - args.norm_penalty_threshold, min=0) ** 2).mean() # (torch.max(torch.zeros_like(gradient_norms), gradient_norms - 1) ** 2).mean() * lambduh
    if return_gradients:
        return gradient_penalty, raw_gradients
    return gradient_penalty



# # Computes similarity between 0 and 1 between the two given pixel feature vectors.
# # If the pixels are not of the same crop type, or if one of them is missing, assign
# # a similarity of 0 (so that they are ignored in the smoothness loss). Otherwise,
# # the similarity is a function of the distance between the feature vectors.
# def pixel_similarity(pixel1, pixel2):
#     if not torch.equal(pixel1[CROP_TYPE_INDICES], pixel2[CROP_TYPE_INDICES]):
#         return 0
#     if pixel1[MISSING_REFLECTANCE_IDX] == 1 or pixel2[MISSING_REFLECTANCE_IDX] == 1:
#         return 0
#     similarity = torch.exp((-1 / len(SIMILARITY_INDICES)) * (torch.linalg.norm(pixel1[SIMILARITY_INDICES] - pixel2[SIMILARITY_INDICES]) ** 2))
#     return similarity
#     # similarity = torch.exp((-1 / len(SIMILARITY_INDICES)) * (torch.linalg.norm(pixel1[SIMILARITY_INDICES] - pixel2[SIMILARITY_INDICES]) ** 2))
#     # return similarity  
#     # if similarity > -1: # > 0.5:
#     #     return similarity
#     # else:
#     #     return 0


def smoothness_loss(args, input_tiles_std, predicted_fine_sifs, device):
    """Implements the smoothness loss term described by (Kotzias et al., 2015). Sample "args.num_pixels" random pixels.
    For each pixel pair, we penalize the similarity between pixel reflectance (raw vectors, normalized to norm 1),
    times the difference in SIF. So this places a high penalty on similar pixels that have very different SIF predictions.

    Args:
        input_tiles_std: standardized input images, shape [batch, channels, height, width]
        predicted_fine_sifs: predicted SIF map, shape [batch, height, width]
    
    In addition, the following command-line arguments in "args" are used:
        args.spread: a parameter when computing similarity between pixel reflectances. 
                     The higher this parameter is, the faster the similarities decay towards 0 as distance increases.
        args.num_pixels: number of pixels to randomly sample
        args.means, args.stds: means and stds for each band (used to undo standardization when computing pixel similarity)
    """
    assert input_tiles_std.shape[0] == predicted_fine_sifs.shape[0]
    assert input_tiles_std.shape[2] == predicted_fine_sifs.shape[1]
    assert input_tiles_std.shape[3] == predicted_fine_sifs.shape[2]
    num_features = input_tiles_std.shape[1]

    # Reshape input so that each pixel is its own row 
    input_pixels = input_tiles_std.permute(0, 2, 3, 1)
    input_pixels = input_pixels.reshape(-1, num_features) # [total pixels, num_features]
    sif_pixels = predicted_fine_sifs.flatten()  # [total pixels]
    sif_pixels.requires_grad_(True)

    # Filter for pixels that actually have reflectance data, and sample some random pixels
    indices_with_reflectance_data = (input_pixels[:, MISSING_REFLECTANCE_IDX] == 0)
    input_pixels = input_pixels[indices_with_reflectance_data]
    sif_pixels = sif_pixels[indices_with_reflectance_data]
    random_indices = torch.randint(low=0, high=input_pixels.shape[0], size=(args.num_pixels,))
    input_pixels = input_pixels[random_indices]  # [num_pixels, num_features]
    sif_pixels = sif_pixels[random_indices]  # [num_pixels]

    # Extract reflectance and cover type vectors
    input_reflectance = input_pixels[:, SIMILARITY_INDICES]
    input_cover_types = input_pixels[:, CROP_TYPE_INDICES]

    # When computing reflectance similarity between pixels, undo standardization, normalize the raw 
    # vectors to norm 1
    input_reflectance = input_reflectance * args.stds[SIMILARITY_INDICES] + args.means[SIMILARITY_INDICES]
    input_reflectance = input_reflectance / torch.linalg.norm(input_reflectance, dim=1, keepdim=True)

    # Computes the Euclidean norm between every pair of rows - see https://pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html
    reflectance_distances = torch.norm(input_reflectance[:, None] - input_reflectance, dim=2, p=2)  # [num_pixels, num_pixels]
    cover_distances = torch.norm(input_cover_types[:, None] - input_cover_types, dim=2, p=1)  # [num_pixels, num_pixels]
    sif_distances = torch.square(sif_pixels[:, None] - sif_pixels)  # torch.abs(sif_pixels[:, None] - sif_pixels)  # [num_pixels, num_pixels]
    sif_distances.requires_grad_(True)

    # Convert reflectance distance into reflectance similarity
    reflectance_similarities = torch.exp(-args.spread * (reflectance_distances ** 2))  # [num_pixels, num_pixels]

    # reflectance_similarities = torch.exp((-1 / len(SIMILARITY_INDICES)) * (reflectance_distances ** 2))  # [num_pixels, num_pixels]
    # reflectance_similarities = torch.matmul(input_reflectance, torch.transpose(input_reflectance, 0, 1))
    # Debugging
    # print("=================== Samples ===================")
    # print("Pixels", input_pixels[0:5])
    # print("SIFs", sif_pixels[0:5])
    # print("Reflectance distances", reflectance_distances[0:5, 0:5])
    # print("Converted to smilarities", reflectance_similarities[0:5, 0:5])
    # print("Cover distances", cover_distances[0:5, 0:5])
    # print("SIF distances", sif_distances[0:5, 0:5])

    # If two pixels are of different cover types, set their similarity to 0. so that these pairs are
    # ignored in the loss function. Same for the similarity of a pixel with itself
    reflectance_similarities[cover_distances > 0.5] = 0  
    reflectance_similarities.fill_diagonal_(0)

    # Compute loss
    loss = torch.sum(reflectance_similarities * sif_distances) / torch.count_nonzero(reflectance_similarities)  # (cover_distances <= 0.5)
    return loss

    # # Below - naive implementation
    # # Generate random pairs of pixels
    # indices = torch.randint(low=0, high=input_pixels.shape[0], size=(1000, 2))
    # total_loss = 0
    # num_points = 0
    # num_zero_similarity = 0

    # # TODO: Potentially vectorize this using "cdist" to improve efficiency, instead of for loop
    # for i in range(indices.shape[0]):
    #     idx1 = indices[i, 0]
    #     idx2 = indices[i, 1]
    #     input_similarity = pixel_similarity(input_pixels[idx1], input_pixels[idx2])
    #     if input_similarity != 0:
    #         sif_difference = torch.abs(sif_pixels[idx1] - sif_pixels[idx2])
    #         total_loss += input_similarity * sif_difference
    #         num_points += 1
    #         # if i % 100 == 0:
    #         #     print("Input 1", input_pixels[idx1][SIMILARITY_INDICES])
    #         #     print("Input 2", input_pixels[idx2][SIMILARITY_INDICES])
    #         #     print("Input similarity", input_similarity, "SIF difference", sif_difference.item())
    #     else:
    #         num_zero_similarity += 1
    # # print("Num points with nonzero similarity", num_points, "with zero similarity", num_zero_similarity)
    # return total_loss / num_points


def pixel_contrastive_loss(args, input_tiles_std, projections, device):
    """Computes contrastive loss; pixels with similar reflectances are pushed closer together 
    while pixels with different reflectances are pushed further apart in the projection space

    Args:
        input_tiles_std: the original input tiles [batch, # channels, width, height]
        projections: embeddings for each pixel [batch, projection dim, width, height]

    In addition, the following command-line arguments in "args" are used:
        args.temperature: the temperature hyperparameter in the contrastive loss function. The lower this is,
                          more penalty is placed on the hardest negative samples (Wang and Liu 2021).
        args.similarity_threshold: if reflectance distance (between normalized reflectance vectors)
                                   is lower than this threshold, pixels are considered similar.
        args.num_pixels: number of pixels to randomly sample
        args.means, args.stds: means and stds for each band (used to undo standardization when computing pixel similarity)
    """
    num_channels = input_tiles_std.shape[1]
    input_pixels = input_tiles_std.permute(0, 2, 3, 1)
    input_pixels = input_pixels.reshape(-1, num_channels)
    projection_dim = projections.shape[1]
    pixel_projections = projections.permute(0, 2, 3, 1)
    pixel_projections = pixel_projections.reshape(-1, projection_dim)

    # To reduce computational cost, sample some pixels
    indices_with_reflectance_data = (input_pixels[:, MISSING_REFLECTANCE_IDX] == 0)
    input_pixels = input_pixels[indices_with_reflectance_data]
    pixel_projections = pixel_projections[indices_with_reflectance_data]
    indices = torch.randint(low=0, high=input_pixels.shape[0], size=(args.num_pixels,))
    input_pixels = input_pixels[indices, :]  # [pixels, input_channels]
    pixel_projections = pixel_projections[indices, :]  # [pixels, proj_dim]

    # Extract reflectance and cover type vectors
    input_reflectance = input_pixels[:, SIMILARITY_INDICES]
    input_cover_types = input_pixels[:, CROP_TYPE_INDICES]

    # When computing reflectance similarity between pixels, undo standardization, normalize the raw 
    # vectors to norm 1 (thus undo-ing multiplicative noise), and then standardize again.
    # print("Input reflectance", input_reflectance)
    input_reflectance = input_reflectance * args.stds[SIMILARITY_INDICES] + args.means[SIMILARITY_INDICES]
    # print("After unstd", input_reflectance)
    input_reflectance = input_reflectance / torch.linalg.norm(input_reflectance, dim=1, keepdim=True)
    # print("Sample reflectance after normalize", input_reflectance[0:10])
    # # print("After unstd, normalize", input_reflectance)
    # input_reflectance = (input_reflectance - args.means[SIMILARITY_INDICES]) / args.stds[SIMILARITY_INDICES]
    # # print("After unstd, normalize, std", input_reflectance)
    # # exit(1)

    # Computes the Euclidean norm between every pair of rows - see https://pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html
    reflectance_distances = torch.norm(input_reflectance[:, None] - input_reflectance, dim=2, p=2)
    cover_distances = torch.norm(input_cover_types[:, None] - input_cover_types, dim=2, p=1)

    # Identify positive and negative pairs. Positive pairs have similar reflectance and the same cover type.
    # Negative pairs have different reflectance.
    pos_mask = ((reflectance_distances <= args.similarity_threshold) & (cover_distances < 0.5)).fill_diagonal_(0) # [pixels, pixels] # .fill_diagonal_(0)
    neg_mask = ((reflectance_distances > args.similarity_threshold) | (cover_distances > 0.5)).fill_diagonal_(0)  # [pixels, pixels]
    # print("Pos mask shape", pos_mask.shape)
    # print("Pos mask per row", torch.count_nonzero(pos_mask[0:10, :], dim=1))

    # # Debugging
    # print("Input pixels shape", input_pixels.shape)
    # print("Pixel projections shape", pixel_projections.shape)
    # print("Input pixels", input_pixels[0:5])
    # print("Ref distances", reflectance_distances[0:5, 0:5])
    # print("Cover distances", cover_distances[0:5, 0:5])
    # print("pos mask", pos_mask[0:5, 0:5])
    # print("neg mask", neg_mask[0:5, 0:5])
    # print("Projections", pixel_projections[0:5, :])
    # print("Projection norms", torch.norm(pixel_projections[0:5, :], dim=1, p=2))

    # Code largely borrowed from https://github.com/tfzhou/ContrastiveSeg/blob/a2f7392d0c50ecdda36fd41205b77a7bc74f7cb8/lib/loss/loss_contrast.py
    contrast_feature = pixel_projections
    anchor_feature = pixel_projections

    # Compute all pairwise similarities (dot product) between pixel projections
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),  # [pixels, pixels]
                                    args.temperature)
    # print("Anchor dot contrast", anchor_dot_contrast[0:5, 0:5])
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # [pixels]
    logits = anchor_dot_contrast - logits_max.detach()  # [pixels, pixels]

    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)

    exp_logits = torch.exp(logits)

    log_prob = logits - torch.log(exp_logits + neg_logits)
    # print("Log prob", log_prob[0:5, 0:5])

    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / torch.clamp(pos_mask.sum(1), min=1)

    loss = - mean_log_prob_pos  # TODO assume base_temperature is the same as temperature
    loss = loss.nanmean()

    return loss



def smoothness_loss_contrastive(args, input_tiles_std, predicted_fine_sifs, device):
    """More contrastive version of the smoothness loss proposed by (Kotzias et al., 2015).

    Sample "num_pixels" random pixels. For each pixel, encourage its predicted SIF to be closer
    to the predicted SIF of the pixel with most similar reflectance, while being further from
    the predicted SIF of other pixels.

    Args:
        input_tiles_std: standardized input images, shape [batch, channels, height, width]
        predicted_fine_sifs: predicted SIF map, shape [batch, height, width]

    In addition, the following command-line arguments in "args" are used:
        args.temperature: the temperature hyperparameter in the contrastive loss function. The lower this is,
                          more penalty is placed on the hardest negative samples (Wang and Liu 2021).
        args.similarity_threshold: if reflectance distance (between normalized reflectance vectors)
                                   is lower than this threshold, pixels are considered similar.
        args.num_pixels: number of pixels to randomly sample
        args.means, args.stds: means and stds for each band (used to undo standardization when computing pixel similarity)

    """
    assert input_tiles_std.shape[0] == predicted_fine_sifs.shape[0]
    assert input_tiles_std.shape[2] == predicted_fine_sifs.shape[1]
    assert input_tiles_std.shape[3] == predicted_fine_sifs.shape[2]
    num_features = input_tiles_std.shape[1]

    # Reshape input so that each pixel is its own row 
    input_pixels = input_tiles_std.permute(0, 2, 3, 1)
    input_pixels = input_pixels.reshape(-1, num_features) # [total pixels, num_features]
    sif_pixels = predicted_fine_sifs.flatten()  # [total pixels]

    # Filter for missing reflectance, and sample some random pixels
    indices_with_reflectance_data = (input_pixels[:, MISSING_REFLECTANCE_IDX] == 0)
    input_pixels = input_pixels[indices_with_reflectance_data]
    sif_pixels = sif_pixels[indices_with_reflectance_data]
    random_indices = torch.randint(low=0, high=input_pixels.shape[0], size=(args.num_pixels,))
    input_pixels = input_pixels[random_indices]  # [num_pixels, num_features]
    sif_pixels = sif_pixels[random_indices]  # [num_pixels]

    # Extract reflectance and cover type vectors
    input_reflectance = input_pixels[:, SIMILARITY_INDICES]
    input_cover_types = input_pixels[:, CROP_TYPE_INDICES]

    # When computing reflectance similarity between pixels, undo standardization, normalize the raw 
    # vectors to norm 1 (thus undo-ing multiplicative noise), and then standardize again.
    # print("Input reflectance", input_reflectance)
    input_reflectance = input_reflectance * args.stds[SIMILARITY_INDICES] + args.means[SIMILARITY_INDICES]
    input_reflectance = input_reflectance / torch.linalg.norm(input_reflectance, dim=1, keepdim=True)

    # Computes the Euclidean norm between every pair of rows - see https://pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html
    reflectance_distances = torch.norm(input_reflectance[:, None] - input_reflectance, dim=2, p=2)  # [num_pixels, num_pixels]
    cover_distances = torch.norm(input_cover_types[:, None] - input_cover_types, dim=2, p=1)  # [num_pixels, num_pixels]

    # Identify positive and negative pairs. Positive pairs have similar reflectance and the same cover type.
    # Negative pairs have different reflectance.
    pos_mask = ((reflectance_distances <= args.similarity_threshold) & (cover_distances < 0.5)).fill_diagonal_(0) # [pixels, pixels] # .fill_diagonal_(0)
    neg_mask = ((reflectance_distances > args.similarity_threshold) | (cover_distances > 0.5)).fill_diagonal_(0)  # [pixels, pixels]

    # Code largely borrowed from https://github.com/tfzhou/ContrastiveSeg/blob/a2f7392d0c50ecdda36fd41205b77a7bc74f7cb8/lib/loss/loss_contrast.py
    sif_distances = torch.abs(sif_pixels[:, None] - sif_pixels)  # [num_pixels, num_pixels]

    # Compute all pairwise similarities (1 - distance) between SIFs
    sif_similarities = (1 - sif_distances) / args.temperature # [num_pixels, num_pixels]

    logits_max, _ = torch.max(sif_similarities, dim=1, keepdim=True)  # [pixels]
    logits = sif_similarities - logits_max.detach()  # [pixels, pixels]

    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)

    exp_logits = torch.exp(logits)

    log_prob = logits - torch.log(exp_logits + neg_logits)
    # print("Log prob", log_prob[0:5, 0:5])

    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / torch.clamp(pos_mask.sum(1), min=1)

    loss = - mean_log_prob_pos  # TODO assume base_temperature is the same as temperature
    loss = loss.nanmean()

    return loss




    # sif_distances = torch.abs(sif_pixels[:, None] - sif_pixels)  # [num_pixels, num_pixels]
    # sif_similarities =  torch.exp(-10 * sif_distances)  # [num_pixels, num_pixels] torch.exp(-10 * torch.norm(sif_pixels[:, None] - sif_pixels, dim=2, p=1))
    # sif_similarities_sum = torch.sum(sif_similarities, dim=1)  # [num_pixels]

    # # When computing the nearest neighbor for each pixel, we want to ignore itself (diagonal),
    # # as well as pixel pairs with different crop types.
    # reflectance_distances.fill_diagonal_(9999)  # We want to find the most similar pixel apart from itself.
    # reflectance_distances[cover_distances > 0.1] = 9999
    # # print("Reflectance distances", reflectance_distances)
    # # print("SIF distances", sif_distances)

    # # For each pixel, compute the index of its nearest neighbor.
    # index_nearest_neighbors = torch.argmin(reflectance_distances, axis=1)
    # loss = 0

    # # For each pixel i, compute loss as the difference between its predicted SIF and the predicted SIF of the nearest neighbor,
    # # divided by the sum of the differences between its predicted SIF and all other pixels' SIF.
    # for i in range(index_nearest_neighbors.shape[0]):
    #     loss += (sif_similarities[i, index_nearest_neighbors[i]] / sif_similarities_sum[i])

    # return loss / index_nearest_neighbors.shape[0]



def similarity_loss(args, input_tiles_std, predicted_fine_sifs, device, similarity_threshold=2):
    """Tries to encourage pixels with similar reflectance to have similar SIF,
    by penalizing the SIF differences among pixel pairs whose reflectance distance is below "similarity_threshold"

    Args:
        input_tiles_std: standardized input images, shape [batch, channels, height, width]
        predicted_fine_sifs: predicted SIF map, shape [batch, height, width]
        similarity_threshold: threshld at which : number of pixels to randomly sample
    """
    # # print("SIMILARITY LOSS")
    # # print("Input tiles std shape", input_tiles_std.shape) # [batch x 24 x 100 x 100]
    # # print("predicted sif shape", predicted_fine_sifs.shape)  # [batch x 100 x 100]
    # height_idx = np.random.randint(low=0, high=input_tiles_std.shape[2])
    # width_idx = np.random.randint(low=0, high=input_tiles_std.shape[3])

    # # "Features" are actually predicted SIF. We want similar pixels to have similar predicted SIF.
    # features = predicted_fine_sifs[:, height_idx, width_idx]

    assert input_tiles_std.shape[0] == predicted_fine_sifs.shape[0]
    assert input_tiles_std.shape[2] == predicted_fine_sifs.shape[1]
    assert input_tiles_std.shape[3] == predicted_fine_sifs.shape[2]
    num_features = input_tiles_std.shape[1]

    # Reshape input so that each pixel is its own row 
    input_pixels = input_tiles_std.permute(0, 2, 3, 1)
    input_pixels = input_pixels.reshape(-1, num_features) # [total pixels, num_features]
    sif_pixels = predicted_fine_sifs.flatten()  # [total pixels]

    # Filter for missing reflectance, and sample some random pixels
    indices_with_reflectance_data = (input_pixels[:, MISSING_REFLECTANCE_IDX] == 0)
    input_pixels = input_pixels[indices_with_reflectance_data]
    sif_pixels = sif_pixels[indices_with_reflectance_data]
    random_indices = torch.randint(low=0, high=input_pixels.shape[0], size=(args.num_pixels,))
    input_pixels = input_pixels[random_indices]  # [num_pixels, num_features]
    sif_pixels = sif_pixels[random_indices]  # [num_pixels]

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

    mask = (pixel_differences < similarity_threshold).to(device)
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

    # Loop through each epoch
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
                    # Read input tile
                    input_tiles_std = sample['input_tile'][:, BANDS, :, :].to(device)  # [batch, # channels, height, width]
                    input_tiles_without_mult_noise = sample['input_tile_without_mult_noise'][:, BANDS, :, :].to(device)
                    # print("input tiles shape", input_tiles_std.shape)

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
                        predicted_fine_sifs_std, pixel_projections = model(input_tiles_std)  # predicted_fine_sifs_std: (batch size, output dim, H, W)
                        predicted_fine_sifs_std.requires_grad_(True)

                        if torch.isnan(predicted_fine_sifs_std).any():
                            print("Predicted fine sifs contained nan")
                            print(predicted_fine_sifs_std)
                            exit(0)

                        predicted_fine_sifs_std = torch.squeeze(predicted_fine_sifs_std, dim=1)
                        predicted_fine_sifs = predicted_fine_sifs_std * sif_std + sif_mean
                        predicted_fine_sifs.requires_grad_(True)
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

                        # Compute contrastive/smoothness loss
                        if args.smoothness_loss:
                            other_loss = smoothness_loss(args, input_tiles_without_mult_noise, predicted_fine_sifs, device)
                            loss = coarse_loss + other_loss * args.lambduh
                        elif args.smoothness_loss_contrastive:
                            other_loss = smoothness_loss_contrastive(args, input_tiles_without_mult_noise, predicted_fine_sifs, device)
                            loss = coarse_loss + other_loss * args.lambduh
                        elif args.similarity_loss:
                            other_loss = similarity_loss(args, input_tiles_without_mult_noise, predicted_fine_sifs, device)
                            loss = coarse_loss + other_loss * args.lambduh
                        elif args.pixel_contrastive_loss:
                            other_loss = pixel_contrastive_loss(args, input_tiles_without_mult_noise, pixel_projections, device)
                            if epoch < args.pretrain_epochs:
                                print("Attention - pretraining!")
                                loss = other_loss
                            else:
                                loss = coarse_loss + other_loss * args.lambduh
                            # loss = coarse_loss + other_loss * args.lambduh
                        elif args.gradient_penalty and phase == 'train':
                            other_loss, gradients = calc_gradient_penalty(args, input_tiles_std, predicted_coarse_sifs, return_gradients=True)
                            # print("Coarse loss", coarse_loss.item(), "--- Gradient penalty", other_loss.item())
                            loss = coarse_loss + other_loss * args.lambduh
                        else:
                            loss = coarse_loss

                        # Backpropagate coarse loss
                        if phase == 'train' and random.random() < UPDATE_FRACTIONS[dataset_name] and dataset_name in COARSE_SIF_DATASETS[phase] and not args.fine_supervision: # and not np.isnan(fine_loss.item()):
                            optimizer.zero_grad()
                            if args.optimizer == "MTAdam":
                                print("MTadam")
                                ranks = [1] * 2
                                optimizer.step([coarse_loss, other_loss], ranks, None)
                            else:
                                loss.backward()
                                # print('Grad', model.down1.maxpool_conv[1].double_conv[0].weight.grad)
                                optimizer.step()


                        # Compute/record losses
                        with torch.set_grad_enabled(args.fine_supervision):
                            if dataset_name in COARSE_SIF_DATASETS[phase]:
                                running_coarse_loss[dataset_name] += coarse_loss.item() * len(true_coarse_sifs)
                                if args.smoothness_loss or args.gradient_penalty or args.smoothness_loss_contrastive or args.pixel_contrastive_loss:
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

                                # Backpropagate fine loss (SHOULD NOT BE USED EXCEPT FOR FULLY-SUPERVISED TRAINING)
                                if phase == 'train' and args.fine_supervision: # and not np.isnan(fine_loss.item()):
                                    if len(all_true_fine_sifs[dataset_name]) == 1:
                                        print("Training with fine SIF!!! are you sure you want to do this?")
                                    optimizer.zero_grad()
                                    fine_loss.backward()
                                    optimizer.step()

                                running_fine_loss[dataset_name] += fine_loss.item() * len(true_fine_sifs_filtered)
                                num_fine_datapoints[dataset_name] += len(true_fine_sifs_filtered)
                                all_true_fine_sifs[dataset_name].append(true_fine_sifs_filtered.cpu().detach().numpy())
                                all_predicted_fine_sifs[dataset_name].append(predicted_fine_sifs_filtered.cpu().detach().numpy())

                                # VISUALIZATIONS
                                if epoch % 10 == 5 and len(all_true_fine_sifs[dataset_name]) == 1 and 'CFIS' in dataset_name and args.visualize:  # Only do first val batch
                                    base_plot_dir = os.path.join(results_dir, "example_predictions_epoch_" + str(epoch))
                                    for i in range(0, 5, 1):
                                        if phase == 'val' or phase == 'train':
                                            center_lat = sample['lat'][i].item()
                                            center_lon = sample['lon'][i].item()
                                            date = sample['date'][i]
                                            tile_description = phase + '_lat_' + str(round(center_lat, 4)) + '_lon_' + str(round(center_lon, 4)) + '_' + date
                                            title = 'Lon ' + str(round(center_lon, 4)) + ', Lat ' + str(round(center_lat, 4)) + ', ' + date
                                            plot_dir = os.path.join(base_plot_dir, tile_description)
                                            if not os.path.exists(plot_dir):
                                                os.makedirs(plot_dir)

                                            # Plot input tile
                                            visualization_utils.plot_tile(input_tiles_std[i].cpu().detach().numpy(),
                                                                center_lon, center_lat, date, TILE_SIZE_DEGREES, 
                                                                tile_description=tile_description, title=title, plot_dir=plot_dir,
                                                                cdl_bands=CROP_TYPE_INDICES)

                                            # Plot input tile without mult noise
                                            visualization_utils.plot_tile(input_tiles_without_mult_noise[i].cpu().detach().numpy(),
                                                                center_lon, center_lat, date, TILE_SIZE_DEGREES, 
                                                                tile_description=tile_description + "_no_mult_noise", title=title + " without mult noise", plot_dir=plot_dir,
                                                                cdl_bands=CROP_TYPE_INDICES)

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
                                                                        plot_dir=plot_dir,
                                                                        cdl_bands=CROP_TYPE_INDICES)

                                        # # Plot gradients wrt input
                                        if phase == 'train' and args.gradient_penalty:
                                            visualization_utils.plot_individual_bands(gradients[i].cpu().detach().numpy(),
                                                                                      title, center_lon, center_lat, TILE_SIZE_DEGREES, 
                                                                                      os.path.join(plot_dir, "input_gradients.png"),
                                                                                      crop_type_start_idx=CROP_TYPE_INDICES[0], num_grid_squares=4,
                                                                                      decimal_places=3, min_feature=None, max_feature=None)

                                            # Print norm of gradients in model
                                            total_norm = 0
                                            for p in model.parameters():
                                                param_norm = p.grad.detach().data.norm(2)
                                                total_norm += param_norm.item() ** 2
                                            total_norm = total_norm ** 0.5
                                            print("Overall gradient norm", total_norm)

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
                # Record other loss
                epoch_other_loss = running_other_loss[coarse_dataset] / num_coarse_datapoints[coarse_dataset]

                # Compute and record coarse SIF loss
                epoch_coarse_nrmse = math.sqrt(running_coarse_loss[coarse_dataset] / num_coarse_datapoints[coarse_dataset]) / sif_mean
                if phase == 'train':
                    train_other_losses[coarse_dataset].append(epoch_other_loss)
                    train_coarse_losses[coarse_dataset].append(epoch_coarse_nrmse)
                    if args.adaptive_lambda:
                        if len(train_other_losses[coarse_dataset]) > 1:
                            if train_other_losses[coarse_dataset][-1] < train_other_losses[coarse_dataset][-2]:
                                if train_coarse_losses[coarse_dataset][-1] < train_coarse_losses[coarse_dataset][-2]:
                                    print("Both losses went down")
                                else:
                                    print("Only coarse loss went up")
                                    args.lambduh *= 0.9
                            else:
                                if train_coarse_losses[coarse_dataset][-1] < train_coarse_losses[coarse_dataset][-2]:
                                    print("Only smoothness loss went up")
                                    args.lambduh *= 1.1
                                else:
                                    print("Both losses went up :(")
                            print("New lambduh", args.lambduh)

                else:
                    val_other_losses[coarse_dataset].append(epoch_other_loss)
                    val_coarse_losses[coarse_dataset].append(epoch_coarse_nrmse)
                true_coarse = np.concatenate(all_true_coarse_sifs[coarse_dataset])
                predicted_coarse = np.concatenate(all_predicted_coarse_sifs[coarse_dataset])
                print('===== ', phase, coarse_dataset, 'Coarse stats ====')
                if args.pixel_contrastive_loss:
                    print("Pixel contrastive loss", epoch_other_loss)
                if args.smoothness_loss:
                    print("Smoothness loss", epoch_other_loss)
                if args.smoothness_loss_contrastive:
                    print("Smoothness loss contrastive", epoch_other_loss)
                if args.gradient_penalty:
                    print("Gradient penalty", epoch_other_loss)
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



            # If the model performed better on the COARSE "MODEL_SELECTION_DATASET" validation set than the
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

            # If the model performed better on the FINE "MODEL_SELECTION_DATASET" validation set than the
            # best previous model, record losses for this epoch, and save this model
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
                    csv_writer.writerow([args.model, GIT_COMMIT, COMMAND_STRING, args.optimizer, args.batch_size, args.learning_rate, args.weight_decay, args.mult_noise_std, args.special_options, val_fine_loss_at_best_val_fine['CFIS_2016'], MODEL_FILE, best_val_fine_epoch])



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

    # Print full loss progressions
    print("==== All losses =====")
    if "CFIS_2016" in train_fine_losses:
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
        csv_writer.writerow([args.model, GIT_COMMIT, COMMAND_STRING, args.optimizer, args.batch_size, args.learning_rate, args.weight_decay, args.mult_noise_std, args.special_options, val_fine_loss_at_best_val_fine['CFIS_2016'], MODEL_FILE, best_val_fine_epoch])

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
args.means = train_means[:-1]
sif_mean = train_means[-1]
args.stds = train_stds[:-1]
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
    normalize_transform = tile_transforms.NormalizeReflectance(reflectance_bands=REFLECTANCE_INDICES)

standardize_transform = tile_transforms.StandardizeTile(args.means, args.stds, bands_to_transform=CONTINUOUS_INDICES)
clip_transform = tile_transforms.ClipTile(min_input=args.min_input, max_input=args.max_input, bands_to_transform=CONTINUOUS_INDICES)
# color_distortion_transform = tile_transforms.ColorDistortion(continuous_bands=REFLECTANCE_INDICES), standard_deviation=args.color_distortion_std)
noise_transform = tile_transforms.GaussianNoise(bands_to_transform=REFLECTANCE_INDICES, standard_deviation=args.gaussian_noise_std)

# If we do multiplicative noise at the end
if args.multiplicative_noise_end:
    multiplicative_noise_end_transform = tile_transforms.MultiplicativeGaussianNoise(args.means, args.stds, bands_to_transform=REFLECTANCE_INDICES, standard_deviation=args.mult_noise_std)
    multiplicative_noise_end_transform = transforms.Compose([multiplicative_noise_end_transform, clip_transform])
else:
    multiplicative_noise_end_transform = None
raw_multiplicative_noise_transform = tile_transforms.MultiplicativeGaussianNoiseRaw(bands_to_transform=REFLECTANCE_INDICES, standard_deviation=args.mult_noise_std)
flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()
compute_vi_transform = tile_transforms.ComputeVegetationIndices()
jigsaw_transform = tile_transforms.RandomJigsaw()
resize_transform = tile_transforms.ResizeTile(target_dim=[args.resize_dim, args.resize_dim])
random_crop_transform = tile_transforms.RandomCrop(crop_dim=args.crop_dim)
cutout_transform = tile_transforms.Cutout(cutout_dim=args.cutout_dim, prob=args.cutout_prob, reflectance_indices=REFLECTANCE_INDICES)
transform_list_train = []
transform_list_val = []

# Convert means/stds to tensor
args.means = torch.tensor(args.means, device=device)
args.stds = torch.tensor(args.stds, device=device)

# Add vegetation index calculations at beginning, BEFORE standardizing. Note that vegetation
# indices are already normalized to a reasonable range, so no need to standardize them.
if args.compute_vi:
    transform_list_train.append(compute_vi_transform)
    transform_list_val.append(compute_vi_transform)

# Apply cutout at beginning, since it is supposed to zero pixels out BEFORE standardizing
if args.cutout:
    transform_list_train.append(cutout_transform)

# Add multiplicative noise at beginning
if args.multiplicative_noise:
    transform_list_train.append(raw_multiplicative_noise_transform)

# If normalize, normalize each pixel at the very beginning
if args.normalize:
    print("Normalizing")
    transform_list_train.append(normalize_transform)
    transform_list_val.append(normalize_transform)
else:
    print("Not normalizing")

# Standardize/clip the continuous variables
transform_list_train.extend([standardize_transform, clip_transform])
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
            train_datasets[dataset_name] = FineSIFDataset(train_set, train_transform, multiplicative_noise_end_transform)
        else:
            train_datasets[dataset_name] = CoarseSIFDataset(train_set, train_transform, multiplicative_noise_end_transform)
    if dataset_name in COARSE_SIF_DATASETS['val'] or dataset_name in FINE_SIF_DATASETS['val']:
        if 'CFIS' in dataset_name:
            val_datasets[dataset_name] = FineSIFDataset(val_set, val_transform, None)
        else:
            val_datasets[dataset_name] = CoarseSIFDataset(val_set, val_transform, None)


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
if args.fine_supervision:
    PARAM_STRING += ("Training with full supervision (fine-resolution labels)\n")
if args.similarity_loss:
    PARAM_STRING += ("Similarity Loss (simple penalty on SIF difference between similar pixels). Lambda: " + str(args.lambduh) + "\n")
if args.smoothness_loss:
    PARAM_STRING += ("Smoothness Loss (penalty on SIF difference * pixel similarity). Lambda: " + str(args.lambduh) + "\n")
if args.smoothness_loss_contrastive:
    PARAM_STRING += ("Smoothness Loss Contrastive. Lambda: " + str(args.lambduh) + "\n")
if args.pixel_contrastive_loss:
    PARAM_STRING += ("Pixel contrastive loss. Lambda: " + str(args.lambduh) + ", Temperature: " + str(args.temperature) + "\n")

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
    PARAM_STRING += ("\tLabel noise: std " + str(args.label_noise) + '\n')
PARAM_STRING += ("Fraction outputs to average: " + str(args.fraction_outputs_to_average) + '\n')
PARAM_STRING += ("SIF range: " + str(args.min_sif) + " to " + str(args.max_sif) + '\n')
PARAM_STRING += ("SIF statistics: mean " + str(sif_mean) + ", std " + str(sif_std) + '\n')
PARAM_STRING += ("==============================================================\n")
print(PARAM_STRING)


# Create dataset/dataloaders
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)
# print("Train datasets", [label + " " + str(len(dataset)) for label, dataset in train_datasets.items()])
# print("Val datasets", [label + " " + str(len(dataset)) for label, dataset in val_datasets.items()])
datasets = {'train': CombinedDataset(train_datasets),
            'val': CombinedDataset(val_datasets)}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
               for x in ['train', 'val']}

# Initialize model
if args.model == 'unet2':
    unet_model = UNet2(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS,
                       reduced_channels=args.reduced_channels, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'unet2_contrastive':
    unet_model = UNet2Contrastive(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'unet2_spectral':
    unet_model = UNet2Spectral(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'pixel_nn':
    unet_model = PixelNN(input_channels=INPUT_CHANNELS, output_dim=OUTPUT_CHANNELS, min_output=min_output, max_output=max_output).to(device)
elif args.model == 'unet':
    unet_model = UNet(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS,
                      reduced_channels=args.reduced_channels, min_output=min_output, max_output=max_output).to(device)   
elif args.model == 'unet_contrastive':
    unet_model = UNetContrastive(n_channels=INPUT_CHANNELS, n_classes=OUTPUT_CHANNELS, min_output=min_output, max_output=max_output).to(device)
else:
    print('Model type not supported', args.model)
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
elif args.optimizer == "MTAdam":
    optimizer = mtadam.MTAdam(unet_model.parameters(), lr=args.learning_rate)
else:
    print("Optimizer not supported")
    exit(1)

# Train model to predict SIF
unet_model, train_coarse_losses, val_coarse_losses, train_fine_losses, val_fine_losses, train_other_losses, val_other_losses = train_model(args, unet_model, dataloaders, criterion, optimizer, device, sif_mean, sif_std)
torch.save(unet_model.state_dict(), MODEL_FILE)

# Rescale other losses by a constant so that they fit on the same graph
if 'CFIS_2016' in train_other_losses and 'CFIS_2016' in train_fine_losses:
    scale_factor = sum(train_fine_losses['CFIS_2016']) / sum(train_other_losses['CFIS_2016'])
    train_other_losses['CFIS_2016'] = [element * scale_factor for element in train_other_losses['CFIS_2016']]

# Plot loss curves: NRMSE
epoch_list = range(len(train_coarse_losses['CFIS_2016']))
plots = []
# print("Coarse Train NRMSE:", train_coarse_losses['CFIS_2016'])
train_coarse_plot, = plt.plot(epoch_list, train_coarse_losses['CFIS_2016'], color='blue', label='Coarse Train NRMSE (CFIS)')
plots.append(train_coarse_plot)
if args.gradient_penalty:
    train_other_plot, = plt.plot(epoch_list, train_other_losses['CFIS_2016'], color='black', label='Gradient penalty')
    plots.append(train_other_plot)
if args.smoothness_loss or args.smoothness_loss_contrastive or args.similarity_loss or args.pixel_contrastive_loss:
    train_other_plot, = plt.plot(epoch_list, train_other_losses['CFIS_2016'], color='black', label='Contrastive/smoothness loss')
    plots.append(train_other_plot)
if 'OCO2_2016' in COARSE_SIF_DATASETS:
    # print("OCO2 Train NRMSE:", train_coarse_losses['OCO2_2016'])
    train_oco2_plot, = plt.plot(epoch_list, train_coarse_losses['OCO2_2016'], color='purple', label='Coarse Train NRMSE (OCO-2)')
    plots.append(train_oco2_plot)
# print("Coarse Val NRMSE:", val_coarse_losses['CFIS_2016'])
val_coarse_plot, = plt.plot(epoch_list, val_coarse_losses['CFIS_2016'], color='green', label='Coarse Val NRMSE')
plots.append(val_coarse_plot)
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
plt.savefig(LOSS_PLOT + '_scatter_fine_vs_coarse_train.png')
plt.close()

# Plot val coarse vs val fine losses
print('============== Val: Fine vs Coarse Losses ===============')
sif_utils.print_stats(val_fine_losses['CFIS_2016'], val_coarse_losses['CFIS_2016'], sif_mean, ax=plt.gca())
plt.xlabel('Val Coarse Losses')
plt.ylabel('Val Fine Losses')
plt.title('Fine vs Coarse val losses: ' + args.model)
plt.savefig(LOSS_PLOT + '_scatter_fine_vs_coarse_val.png')
plt.close()
