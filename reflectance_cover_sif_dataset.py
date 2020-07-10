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

class ReflectanceCoverSIFDataset(Dataset):
    """Dataset mapping a tile (with reflectance/cover bands) to total SIF"""

    def __init__(self, tile_info, transform, tile_file_column='tile_file'):
        """
        Args:
            tile_info: Pandas dataframe containing metadata for each tile.
            The tile is assumed to have shape (band x lat x long)
        """
        self.tile_info = tile_info
        self.transform = transform
        self.tile_file_column = tile_file_column

    def __len__(self):
        return len(self.tile_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_tile_info = self.tile_info.iloc[idx]
        tile = np.load(current_tile_info.loc[self.tile_file_column]) 

        # print('Idx', idx, 'Band means before transform', np.mean(tile, axis=(1, 2)))
        # tile_description = str(round(current_tile_info.loc['lat'], 5)) + '_lon_' + str(round(current_tile_info.loc['lon'], 5)) + '_' + current_tile_info.loc['date']
        # sif_utils.plot_tile(tile,  'lat_before_augment_lat_' + tile_description)

        if self.transform:
            tile = self.transform(tile)

        # sif_utils.plot_tile(tile,  'lat_after_augment_lat_' + tile_description)
        # print('Idx', idx, 'Band means after transform', np.mean(tile, axis=(1,2)))

        tile = torch.tensor(tile, dtype=torch.float)
 
        sample = {'tile': tile,
                  'SIF': current_tile_info.loc["SIF"]}
                # 'lon': current_tile_info.loc['lon'],
                #   'lat': current_tile_info.loc['lat'],
                #   'tile_file': current_tile_info.loc[self.tile_file_column],
                #   'source': current_tile_info.loc['source'],
                #   'date': current_tile_info.loc['date'],
                  #'year': year,
                  #'day_of_year': day_of_year,
        return sample

class CombinedDataset(Dataset):
    """
    Dataset mapping a tile (with reflectance/cover bands) to total SIF, combining both
    OCO-2 and TROPOMI datapoints.
    """

    def __init__(self, tropomi_tile_info, oco2_tile_info, transform, subtile_dim, tile_file_column='tile_file'):
        """
        Args:
            tropomi_tile_info: Pandas dataframe containing metadata for each TROPOMI tile.
                               The tile is assumed to have shape (band x lat x long)
            oco2_file_info: Pandas dataframe containing metadata for each OCO-2 tile.
                            The tile is assumed to have shape (band x lat x long)
        """
        assert((self.tropomi_tile_info is not None) or (self.oco2_tile_info is not None))
        self.tropomi_tile_info = tropomi_tile_info
        self.oco2_tile_info = oco2_tile_info
        self.transform = transform
        self.subtile_dim = subtile_dim
        self.tile_file_column = tile_file_column

    def __len__(self):
        if self.tropomi_tile_info is None:
            return len(self.oco2_tile_info)
        elif self.oco2_tile_info is None:
            return len(self.tropomi_tile_info)
        return max(len(self.tropomi_tile_info), len(self.oco2_tile_info))


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}

        # Read the TROPOMI tile if it exists
        if self.tropomi_tile_info is not None:
            # If the index is larger than the length of one of the datasets, loop around
            tropomi_idx = idx % len(self.tropomi_tile_info)

            # Load the TROPOMI tile
            tropomi_tile_info = self.tropomi_tile_info.iloc[tropomi_idx]
            tropomi_tile = np.load(tropomi_tile_info.loc[self.tile_file_column])

            # Transform the TROPOMI tile
            if self.transform:
                tropomi_tile = self.transform(tropomi_tile)

            tropomi_subtiles = sif_utils.get_subtiles_list(tropomi_tile, self.subtile_dim, MAX_SUBTILE_CLOUD_COVER)

            # Add the TROPOMI tile to the sample
            sample['tropomi_subtiles'] = torch.tensor(tropomi_subtiles, dtype=torch.float)
            sample['tropomi_sif'] = tropomi_tile_info.loc['SIF']

        # Read the OCO-2 tile if it exists
        if self.oco2_tile_info is not None:
            oco2_idx = idx % len(self.oco2_tile_info)
            oco2_tile_info = self.oco2_tile_info.iloc[oco2_idx]
            oco2_tile = np.load(oco2_tile_info.loc[self.tile_file_column]) 
            if self.transform:
                oco2_tile = self.transform(oco2_tile)
            sample['oco2_tile'] = torch.tensor(oco2_tile, dtype=torch.float)
            sample['oco2_sif'] = oco2_tile_info.loc['SIF']}

        return sample



class SubtileListDataset(Dataset):
    """Dataset mapping a sub-tile list to total SIF"""
    def __init__(self, tile_info, transform, tile_file_column='tile_file', num_subtiles=50):
        """
        Args:
            tile_info_file (string): Pandas dataframe containing metadata for each tile.
            The tile is assumed to have shape (band x lat x long)

            num_subtiles: number of subtiles to sample
        """
        self.tile_info = tile_info
        self.transform = transform
        self.tile_file_column = tile_file_column
        self.num_subtiles = num_subtiles

    def __len__(self):
        return len(self.tile_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_tile_info = self.tile_info.iloc[idx]
        tile = np.load(current_tile_info.loc[self.tile_file_column])

        # Chose some random sub-tiles
        if self.num_subtiles is not None:
            # Without replacement, unless we want more sub-tiles than exist
            if self.num_subtiles < tile.shape[0]:
                subtile_indices = np.random.choice(tile.shape[0], self.num_subtiles, replace=False)
            else:
                subtile_indices = np.random.choice(tile.shape[0], self.num_subtiles, replace=True)

        tile = tile[subtile_indices]
        if self.transform:
            tile = self.transform(tile)

        tile = torch.tensor(tile, dtype=torch.float)
 
        sample = {'lon': current_tile_info.loc['lon'],
                  'lat': current_tile_info.loc['lat'],
                  'tile_file': current_tile_info.loc[self.tile_file_column],
                  'source': current_tile_info.loc['source'],
                  'date': current_tile_info.loc['date'],
                  #'year': year,
                  #'day_of_year': day_of_year,
                  'tile': tile,
                  'SIF': current_tile_info.loc["SIF"]}
        return sample
