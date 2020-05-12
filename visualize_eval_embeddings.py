import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import sys
sys.path.append('../')
from tile2vec.src.tilenet import make_tilenet
import tile_transforms
import torchvision
import torchvision.transforms as transforms
from eval_subtile_dataset import EvalSubtileDataset
from sif_utils import plot_histogram

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
EVAL_SUBTILE_DATASET_FILE = os.path.join(DATA_DIR, "dataset_2016-07-16/eval_subtiles.csv")
TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_recon_5/TileNet.ckpt")  #"models/tile2vec_dim512_neighborhood100/TileNet.ckpt")
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-07-16")
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")

eval_metadata = pd.read_csv(EVAL_SUBTILE_DATASET_FILE)
Z_DIM = 256  # 256  #512
INPUT_CHANNELS = 43
RGB_BANDS = [3, 2, 1]
COVER_BANDS = list(range(12, 42))
BATCH_SIZE = 4
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
print("Eval samples", len(eval_metadata))
print("Means", train_means)
print("Stds", train_stds)
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
transform = transforms.Compose(transform_list)

# Set up Dataset and Dataloader
dataset_size = len(eval_metadata)
dataset = EvalSubtileDataset(eval_metadata, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

# Load Tile2Vec model
tile2vec_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=Z_DIM).to(device)
tile2vec_model.load_state_dict(torch.load(TILE2VEC_MODEL_FILE, map_location=device))
tile2vec_model.eval()

# Calculate all subtile embeddings
subtile_embeddings = np.zeros((len(eval_metadata), Z_DIM)) # INPUT_CHANNELS))
subtile_files = []
lats = []
lons = []
sifs = []
i = 0
for sample in dataloader:
    input_tile_standardized = sample['subtile'].to(device)
    #print('Input tile standardized dims', input_tile_standardized.shape)

    batch = input_tile_standardized.shape[0]
    with torch.set_grad_enabled(False):
        embeddings = tile2vec_model(input_tile_standardized).cpu().numpy()
        #print('Embedding shape', embeddings.shape)
        #print('Embedding', embeddings[0])
        subtile_embeddings[i:i+batch] = embeddings #tile2vec_model(input_tile_standardized).cpu().numpy() # torch.mean(input_tile_standardized, dim=(2, 3)).cpu().numpy() #  # tile2vec_model(input_tile_standardized)
    subtile_files.extend(sample['subtile_file'])
    lats.extend(sample['lat'].tolist())
    lons.extend(sample['lon'].tolist())
    sifs.extend(sample['SIF'].tolist())
    i += batch
    #if i >= 5000:
    #    break

# Loads tile from file and transforms it into the format that imshow wants
def tile_to_image(tile):
    tile = tile.transpose((1, 2, 0))
    return tile[:, :, RGB_BANDS] / 2000

def get_title_string(lat, lon, sif):
    return 'Lat' + str(round(lat, 5)) + ', Lon' + str(round(lon, 5)) + ' (SIF = ' + str(round(sif, 3)) + ')'

indices = [3, 13, 23, 33, 43, 53, 63]
for idx in indices:
    plot_histogram(subtile_embeddings[:, idx], "tile2vec_embedding_idx_" + str(idx) + ".png")

NUM_NEIGHBORS = 5

for anchor_idx in range(10):
    distances_to_anchor = np.zeros((subtile_embeddings.shape[0]))
    for i in range(subtile_embeddings.shape[0]):
        distances_to_anchor[i] = np.linalg.norm(subtile_embeddings[anchor_idx] - subtile_embeddings[i])
    sorted_indices = distances_to_anchor.argsort()  # Ascending order of distance
    closest_indices = sorted_indices[:NUM_NEIGHBORS]
    furthest_indices = sorted_indices[-NUM_NEIGHBORS:][::-1]
    print('Closest indices', closest_indices)
    print('Furthest indices', furthest_indices)
    fig, axeslist = plt.subplots(ncols=NUM_NEIGHBORS, nrows=3, figsize=(30, 30))

    # Display anchor subtile
    anchor_tile = np.load(subtile_files[anchor_idx])
    axeslist[0][0].imshow(tile_to_image(anchor_tile))
    axeslist[0][0].set_title('Anchor tile: ' + get_title_string(lats[anchor_idx], lons[anchor_idx], sifs[anchor_idx]))
    axeslist[0][0].set_axis_off()
    print('Anchor tile crops', anchor_tile[COVER_BANDS, :, :].mean(axis=(1,2)))

    # display closest subtiles
    i = 0
    for close_idx in closest_indices:
        close_tile = np.load(subtile_files[close_idx])
        axeslist[1][i].imshow(tile_to_image(close_tile))
        axeslist[1][i].set_title('Close: ' + get_title_string(lats[close_idx], lons[close_idx], sifs[close_idx]))
        axeslist[1][i].set_axis_off()
        print('close tile crops', close_tile[COVER_BANDS, :, :].mean(axis=(1,2)))
        i += 1

    # display closest subtiles
    i = 0
    for far_idx in furthest_indices:
        far_tile = np.load(subtile_files[far_idx])
        axeslist[2][i].imshow(tile_to_image(far_tile))
        axeslist[2][i].set_title('Far: ' + get_title_string(lats[far_idx], lons[far_idx], sifs[far_idx]))
        axeslist[2][i].set_axis_off()
        print('far tile crops', far_tile[COVER_BANDS, :, :].mean(axis=(1,2)))
        i += 1

    plt.savefig('exploratory_plots/embedding_neighbors_' + os.path.basename(subtile_files[anchor_idx]) + '.png')
    plt.close()
   


