from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import sif_utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class SubtileEmbeddingDataset(Dataset):

    def __init__(self, tile_rows):
        """
        Args:
        """
        self.tile_rows = tile_rows

    def __len__(self):
        return len(self.tile_rows)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        subtile_embeddings = self.tile_rows[idx][0]
        sif = self.tile_rows[idx][1]
 
        sample = {'subtile_embeddings': subtile_embeddings,
                  'SIF': sif}
        return sample
