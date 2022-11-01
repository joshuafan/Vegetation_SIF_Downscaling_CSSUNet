from sif_utils import plot_histogram
import numpy as np
import os
import pandas as pd
from visualize_dataset import plot_images


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
SUBTILE_EMBEDDING_FILE = os.path.join(DATA_DIR, 'dataset_2018-08-01/avg_embeddings_train.csv')
PLOT_FILE_PREFIX = "avg_subtile_embeddings_idx_"
large_tiles = 1000
subtiles_per_large_tile = 1369
z_dim = 25

subtile_embedding_dataset = pd.read_csv(SUBTILE_EMBEDDING_FILE)
random_subset = subtile_embedding_dataset.sample(n=large_tiles)
subtile_embeddings = np.zeros((large_tiles * subtiles_per_large_tile, z_dim))

for idx in range(len(random_subset)):
    embeddings = np.load(random_subset.iloc[idx].loc['embedding_file'])
    print('Embeddings shape', embeddings.shape)
    subtile_embeddings[idx*subtiles_per_large_tile:(idx+1)*subtiles_per_large_tile] = embeddings

for j in range(z_dim):
    plot_histogram(subtile_embeddings[:, j], PLOT_FILE_PREFIX + str(j))


