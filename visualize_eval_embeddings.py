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

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
EVAL_SUBTILE_DATASET_FILE = os.path.join(DATA_DIR, "dataset_2016-08-01/eval_subtiles.csv")
TILE2VEC_MODEL_FILE = os.path.join(DATA_DIR, "models/tile2vec_dim256_rgb/TileNet.ckpt")  #"models/tile2vec_dim512_neighborhood100/TileNet.ckpt")
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01")
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")

eval_metadata = pd.read_csv(EVAL_SUBTILE_DATASET_FILE)
Z_DIM = 256  #512
INPUT_CHANNELS = 3 # 43
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
tile2vec_model = make_tilenet(in_channels=INPUT_CHANNELS, z_dim=Z_DIM)
tile2vec_model.load_state_dict(torch.load(TILE2VEC_MODEL_FILE, map_location=device))

# Calculate all subtile embeddings
subtile_embeddings = np.zeros((2000, Z_DIM))  # (len(eval_dataset)z_dim))
i = 0
for sample in dataloader:
    input_tile_standardized = sample['subtile'][:, RGB_BANDS, :, :].to(device)
    print('Input tile standardized dims', input_tile_standardized.shape)

    batch = input_tile_standardized.shape[0]
    with torch.set_grad_enabled(False):
        subtile_embeddings[i:i+batch] = tile2vec_model(input_tile_standardized)
    i += batch
    if i >= 2000:
        break

# Loads tile from file and transforms it into the format that imshow wants
def tile_to_image(tile):
    tile = tile.transpose((1, 2, 0))
    return tile[:, :, RGB_BANDS] / 1000

def get_title_string(image_row):
    return 'Lat' + str(round(image_row['lat'], 6)) + ', Lon' + str(round(image_row['lon'], 6)) + ' (SIF = ' + str(round(image_row['SIF'], 3)) + ')'


NUM_NEIGHBORS = 5

for anchor_idx in range(5):
    distances_to_anchor = np.zeros((subtile_embeddings.shape[0]))
    for i in range(subtile_embeddings.shape[0]):
        distances_to_anchor[i] = np.linalg.norm(subtile_embeddings[anchor_idx] - subtile_embeddings[i])
    sorted_indices = distances_to_anchor.argsort()  # Ascending order of distance
    closest_indices = sorted_indices[:NUM_NEIGHBORS]
    furthest_indices = sorted_indices[-NUM_NEIGHBORS:][::-1]
    print('Closest indices', closest_indices)
    closest_subtile_rows = eval_metadata.iloc[closest_indices]
    furthest_subtile_rows = eval_metadata.iloc[furthest_indices]
    anchor_subtile_row = eval_metadata.iloc[anchor_idx]

    fig, axeslist = plt.subplots(ncols=NUM_NEIGHBORS, nrows=3, figsize=(30, 30))

    # Display anchor subtile
    anchor_tile = np.load(anchor_subtile_row['subtile_file'])
    axeslist[0][0].imshow(tile_to_image(anchor_tile))
    axeslist[0][0].set_title('Anchor tile: ' + get_title_string(anchor_subtile_row))
    axeslist[0][0].set_axis_off()
    print('Anchor tile crops', anchor_tile[COVER_BANDS, :, :].mean(axis=(1,2)))

    # display closest subtiles
    i = 0
    for idx, close_subtile_row in closest_subtile_rows.iterrows():
        close_tile = np.load(close_subtile_row['subtile_file'])
        axeslist[1][i].imshow(tile_to_image(close_tile))
        axeslist[1][i].set_title('close: ' + get_title_string(close_subtile_row))
        axeslist[1][i].set_axis_off()
        print('close tile crops', close_tile[COVER_BANDS, :, :].mean(axis=(1,2)))
        i += 1

    # display closest subtiles
    i = 0
    for idx, far_subtile_row in furthest_subtile_rows.iterrows():
        far_tile = np.load(far_subtile_row['subtile_file'])
        axeslist[2][i].imshow(tile_to_image(far_tile))
        axeslist[2][i].set_title('far: ' + get_title_string(far_subtile_row))
        axeslist[2][i].set_axis_off()
        print('far tile crops', far_tile[COVER_BANDS, :, :].mean(axis=(1,2)))
        i += 1

    plt.savefig('exploratory_plots/embedding_neighbors_' + os.path.basename(anchor_subtile_row['subtile_file']) + '.png')
    plt.close()
   


