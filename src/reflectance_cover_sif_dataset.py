from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import numpy.ma as ma
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import sif_utils
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class CombinedDataset(Dataset):
    """
    Used to produce a Dataloader that can load samples from multiple Datasets in one "batch".
    """
    def __init__(self, datasets):
        self.datasets = datasets
    
    """
    Length is equal to the length of the longest dataset
    """
    def __len__(self):
        max_len = 0
        for dataset in self.datasets.values():
            if len(dataset) > max_len:
                max_len = len(dataset)
        return max_len

    def __getitem__(self, idx):
        batch = dict()
        for dataset_name, dataset in self.datasets.items():
            dataset_idx = idx % len(dataset)
            batch[dataset_name] = dataset[dataset_idx]
        return batch


class CoarseSIFDataset(Dataset):
    """
    Dataset mapping a tile (with reflectance/cover bands) to a single SIF value
    """
    def __init__(self, tile_info, transform, tile_file_column='tile_file', coarse_sif_column='SIF'):
        """
        Args:
            tile_info: Pandas dataframe containing metadata for each tile.
            The tile is assumed to have shape (band x lat x long)
        """
        self.tile_info = tile_info
        self.transform = transform
        self.tile_file_column = tile_file_column
        self.coarse_sif_column = coarse_sif_column

    def __len__(self):
        return len(self.tile_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_tile_info = self.tile_info.iloc[idx]
        tile = np.load(current_tile_info.loc[self.tile_file_column])

        if self.transform:
            tile = self.transform(tile)
 
        sample = {'input_tile': torch.tensor(tile, dtype=torch.float),
                  'coarse_sif': torch.tensor(current_tile_info[self.coarse_sif_column], dtype=torch.float),
                  'lon': current_tile_info.loc['lon'],
                  'lat': current_tile_info.loc['lat'],
                  'tile_file': current_tile_info.loc[self.tile_file_column],
                  'date': current_tile_info.loc['date']}
        return sample


class FineSIFDataset(Dataset):
    """
    Dataset mapping a tile (with reflectance/cover bands) to fine pixel-level SIF values and a
    coarse SIF value.
    """
    def __init__(self, tile_info, transform,
                 tile_file_column='tile_file',
                 fine_sif_file_column='fine_sif_file',
                 fine_soundings_file_column='fine_soundings_file',
                 coarse_sif_column='SIF',
                 coarse_soundings_column='num_soundings'):
        """
        Args:
            tile_info: Pandas dataframe containing metadata for each tile.
                        The tile is assumed to have shape (band x lat x long)
            "fine_sif_file_column": column which contains fine SIF files, which are MASKED NUMPY 
                                    ARRAYS. The mask is set to True if the pixel is invalid.
        """
        self.tile_info = tile_info
        self.transform = transform
        self.tile_file_column = tile_file_column
        self.fine_sif_file_column = fine_sif_file_column
        self.fine_soundings_file_column = fine_soundings_file_column
        self.coarse_sif_column = coarse_sif_column
        self.coarse_soundings_column = coarse_soundings_column


    def __len__(self):
        return len(self.tile_info)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}

        # Read CFIS tile
        current_tile_info = self.tile_info.iloc[idx]
        input_tile = np.load(current_tile_info.loc[self.tile_file_column], allow_pickle=True)
        fine_sif_tile = np.load(current_tile_info.loc[self.fine_sif_file_column], allow_pickle=True)
        fine_soundings_tile = np.load(current_tile_info.loc[self.fine_soundings_file_column], allow_pickle=True)
        if self.transform:
            # If we're rotating, 
            consolidated_tile = np.concatenate([input_tile,
                                                np.expand_dims(fine_sif_tile.data, axis=0),
                                                np.expand_dims(fine_sif_tile.mask, axis=0),
                                                np.expand_dims(fine_soundings_tile, axis=0)], axis=0)
            consolidated_tile = self.transform(consolidated_tile)

            input_tile = consolidated_tile[:-3]
            fine_sif_tile = consolidated_tile[-3]
            fine_sif_mask = consolidated_tile[-2]
            fine_soundings_tile = consolidated_tile[-1]

            # Mark cloudy pixels as invalid ("fine_sif_mask" is 1 for invalid pixels,
            # and input_tile[-1] is the missing reflectance mask, which is 1 when a pixel
            # is covered by clouds)
            fine_sif_mask = np.logical_or(fine_sif_mask, input_tile[-1])

        sample = {'input_tile': torch.tensor(input_tile, dtype=torch.float),
                'fine_sif': torch.tensor(fine_sif_tile, dtype=torch.float),
                'fine_sif_mask': torch.tensor(fine_sif_mask, dtype=torch.bool),
                'fine_soundings': torch.tensor(fine_soundings_tile, dtype=torch.float),
                'coarse_sif': torch.tensor(current_tile_info[self.coarse_sif_column], dtype=torch.float),  #torch.tensor(cfis_coarse_sif, dtype=torch.float),
                'coarse_soundings': current_tile_info[self.coarse_soundings_column],
                'tile_file': current_tile_info.loc[self.tile_file_column],
                'lon': current_tile_info.loc['lon'],
                'lat': current_tile_info.loc['lat'],
                'date': current_tile_info.loc['date'],
                'fraction_valid': current_tile_info.loc['fraction_valid']}

        return sample



