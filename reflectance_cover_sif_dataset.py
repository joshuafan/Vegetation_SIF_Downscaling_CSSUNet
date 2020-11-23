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
    Used to produce a Dataloader that can load samples from multiple Datasets simultaneously.
    """
    def __init__(self, datasets):
        self.datasets = datasets


class CombinedCfisOco2Dataset(Dataset):
    """Dataset mapping a tile (with reflectance/cover bands) to a coarse or fine resolution SIF map"""

    def __init__(self, cfis_tile_info, oco2_tile_info, transform,
                 min_cfis_soundings,
                 tile_file_column='tile_file',
                 cfis_fine_sif_file_column='fine_sif_file',
                 cfis_fine_soundings_file_column='fine_soundings_file',
                 cfis_coarse_sif_column='SIF',
                 cfis_coarse_soundings_column='num_soundings',
                 oco2_sif_column='SIF',
                 oco2_soundings_column='num_soundings'):
        """
        Args:
            tile_info: Pandas dataframe containing metadata for each tile.
            The tile is assumed to have shape (band x lat x long)
        """
        self.cfis_tile_info = cfis_tile_info
        self.oco2_tile_info = oco2_tile_info
        self.transform = transform
        self.min_cfis_soundings = min_cfis_soundings  # TODO remove this
        self.tile_file_column = tile_file_column
        self.cfis_fine_sif_file_column = cfis_fine_sif_file_column
        self.cfis_fine_soundings_file_column = cfis_fine_soundings_file_column
        self.cfis_coarse_sif_column = cfis_coarse_sif_column
        self.cfis_coarse_soundings_column = cfis_coarse_soundings_column
        self.oco2_sif_column = oco2_sif_column
        self.oco2_soundings_column = oco2_soundings_column

        # Store lengths of datasets
        if cfis_tile_info is None:
            self.cfis_len = 0
        else:
            self.cfis_len = len(cfis_tile_info)
        if oco2_tile_info is None:
            self.oco2_len = 0
        else:
            self.oco2_len = len(oco2_tile_info)


    def __len__(self):
        return max(self.cfis_len, self.oco2_len)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}

        # Read CFIS tile
        if self.cfis_tile_info is not None:
            cfis_idx = idx % self.cfis_len
            current_cfis_tile_info = self.cfis_tile_info.iloc[cfis_idx]
            cfis_input_tile = np.load(current_cfis_tile_info.loc[self.tile_file_column], allow_pickle=True)
            cfis_fine_sif_tile = np.load(current_cfis_tile_info.loc[self.cfis_fine_sif_file_column], allow_pickle=True)
            cfis_fine_soundings_tile = np.load(current_cfis_tile_info.loc[self.cfis_fine_soundings_file_column], allow_pickle=True)
            # print('Cfis input tile', cfis_input_tile[])
            # Mark fine SIF entries with too few soundings as invalid (so that they don't get counted in the loss)
            # cfis_fine_sif_tile.mask[cfis_fine_soundings_tile < self.min_cfis_soundings] = True
            # cfis_coarse_sif = current_tile_info[self.cfis_coarse_sif_column]  # np.load(current_cfis_tile_info.loc[self.cfis_coarse_sif_column], allow_pickle=True)
            if self.transform:
                consolidated_tile = np.concatenate([cfis_input_tile, np.expand_dims(cfis_fine_sif_tile.data, axis=0),
                                                    np.expand_dims(cfis_fine_sif_tile.mask, axis=0),
                                                    np.expand_dims(cfis_fine_soundings_tile, axis=0)], axis=0)
                # print('random', consolidated_tile[:, 0, 0])
                # print('random', consolidated_tile[:, 1, 1])
                # print('random', consolidated_tile[:, 2, 2])
                # print('random', consolidated_tile[:, 3, 3])
                # print('consolidated tile', consolidated_tile.shape)
                consolidated_tile = self.transform(consolidated_tile)
                # print('after transform', consolidated_tile.shape)
                # print('random', consolidated_tile[:, 0, 0])
                # print('random', consolidated_tile[:, 1, 1])
                # print('random', consolidated_tile[:, 2, 2])
                # print('random', consolidated_tile[:, 3, 3])

                cfis_input_tile = consolidated_tile[:-3]
                cfis_fine_sif_tile = consolidated_tile[-3]
                cfis_fine_sif_mask = consolidated_tile[-2]
                cfis_fine_soundings_tile = consolidated_tile[-1]

            sample = {'cfis_input_tile': torch.tensor(cfis_input_tile, dtype=torch.float),
                    'cfis_fine_sif': torch.tensor(cfis_fine_sif_tile, dtype=torch.float),
                    'cfis_fine_sif_mask': torch.tensor(cfis_fine_sif_mask, dtype=torch.bool),
                    'cfis_fine_soundings': torch.tensor(cfis_fine_soundings_tile, dtype=torch.float),
                    'cfis_coarse_sif': torch.tensor(current_cfis_tile_info[self.cfis_coarse_sif_column], dtype=torch.float),  #torch.tensor(cfis_coarse_sif, dtype=torch.float),
                    'cfis_coarse_soundings': current_cfis_tile_info[self.cfis_coarse_soundings_column],
                    'cfis_tile_file': current_cfis_tile_info.loc[self.tile_file_column],
                    'cfis_lon': current_cfis_tile_info.loc['lon'],
                    'cfis_lat': current_cfis_tile_info.loc['lat'],
                    'cfis_date': current_cfis_tile_info.loc['date']}

        # print('Idx', idx, 'Band means before transform', np.mean(tile, axis=(1, 2)))
        # tile_description = str(round(current_tile_info.loc['lat'], 5)) + '_lon_' + str(round(current_tile_info.loc['lon'], 5)) + '_' + current_tile_info.loc['date']
        # sif_utils.plot_tile(tile,  'lat_before_augment_lat_' + tile_description)

        # Read OCO-2 tile
        if self.oco2_tile_info is not None:
            oco2_idx = idx % self.oco2_len
            current_oco2_tile_info = self.oco2_tile_info.iloc[oco2_idx]
            oco2_input_tile = np.load(current_oco2_tile_info.loc[self.tile_file_column], allow_pickle=True)
            oco2_sif = current_oco2_tile_info.loc[self.oco2_sif_column]  # TODO temporary
            oco2_soundings = current_oco2_tile_info.loc[self.oco2_soundings_column]

            if self.transform:
                oco2_input_tile = self.transform(oco2_input_tile)

            # sif_utils.plot_tile(input_tile,  'lat_after_augment_lat_' + tile_description)
            # print('Idx', idx, 'Band means after transform', np.mean(tile, axis=(1,2)))
 
            sample['oco2_input_tile'] = torch.tensor(oco2_input_tile, dtype=torch.float)
            sample['oco2_sif'] = torch.tensor(oco2_sif, dtype=torch.float)
            sample['oco2_soundings'] = oco2_soundings
            sample['oco2_lon'] = current_oco2_tile_info.loc['lon']
            sample['oco2_lat'] = current_oco2_tile_info.loc['lat']
            sample['oco2_date'] = current_oco2_tile_info.loc['date']

        return sample



class CFISDataset(Dataset):
    """Dataset mapping a tile (with reflectance/cover bands) to a coarse or fine resolution SIF map"""

    def __init__(self, tile_info, transform, tile_file_column='tile_file', 
                 sif_fine_file_column='sif_fine_file', sif_coarse_file_column='sif_coarse_file'):
        """
        Args:
            tile_info: Pandas dataframe containing metadata for each tile.
            The tile is assumed to have shape (band x lat x long)
        """
        self.tile_info = tile_info
        self.transform = transform
        self.tile_file_column = tile_file_column
        self.sif_fine_file_column = sif_fine_file_column
        self.sif_coarse_file_column = sif_coarse_file_column


    def __len__(self):
        return len(self.tile_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_tile_info = self.tile_info.iloc[idx]
        input_tile = np.load(current_tile_info.loc[self.tile_file_column], allow_pickle=True)
        fine_sif_tile = np.load(current_tile_info.loc[self.sif_fine_file_column], allow_pickle=True)
        coarse_sif_tile = np.load(current_tile_info.loc[self.sif_coarse_file_column], allow_pickle=True)

        # print('Idx', idx, 'Band means before transform', np.mean(tile, axis=(1, 2)))
        # tile_description = str(round(current_tile_info.loc['lat'], 5)) + '_lon_' + str(round(current_tile_info.loc['lon'], 5)) + '_' + current_tile_info.loc['date']
        # sif_utils.plot_tile(tile,  'lat_before_augment_lat_' + tile_description)

        if self.transform:
            input_tile = self.transform(input_tile)

        # sif_utils.plot_tile(input_tile,  'lat_after_augment_lat_' + tile_description)
        # print('Idx', idx, 'Band means after transform', np.mean(tile, axis=(1,2)))
 
        sample = {'input_tile': torch.tensor(input_tile, dtype=torch.float),
                  'fine_sif': torch.tensor(fine_sif_tile.data, dtype=torch.float),
                  'fine_sif_mask': torch.tensor(fine_sif_tile.mask, dtype=torch.bool),
                  'coarse_sif': torch.tensor(coarse_sif_tile.data, dtype=torch.float),
                  'coarse_sif_mask': torch.tensor(coarse_sif_tile.mask, dtype=torch.bool),
                  'tile_file': current_tile_info.loc[self.tile_file_column],
                  'lon': current_tile_info.loc['lon'],
                  'lat': current_tile_info.loc['lat'],
                  'date': current_tile_info.loc['date']}
        return sample


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
                  'SIF': current_tile_info.loc["SIF"],
                  'lon': current_tile_info.loc['lon'],
                  'lat': current_tile_info.loc['lat'],
                  'tile_file': current_tile_info.loc[self.tile_file_column],
                  'source': current_tile_info.loc['source'],
                  'date': current_tile_info.loc['date']}
        return sample


class CombinedDataset(Dataset):
    """
    Dataset mapping a tile (with reflectance/cover bands) to total SIF, combining both
    OCO-2 and TROPOMI datapoints.
    """

    def __init__(self, tropomi_tile_info, oco2_tile_info, transform, return_subtiles=True, subtile_dim=100, tile_file_column='tile_file'):
        """
        Args:
            tropomi_tile_info: Pandas dataframe containing metadata for each TROPOMI tile.
                               The tile is assumed to have shape (band x lat x long)
            oco2_file_info: Pandas dataframe containing metadata for each OCO-2 tile.
                            The tile is assumed to have shape (band x lat x long)
        """
        assert((tropomi_tile_info is not None) or (oco2_tile_info is not None))
        self.tropomi_tile_info = tropomi_tile_info
        self.oco2_tile_info = oco2_tile_info
        self.transform = transform
        self.tile_file_column = tile_file_column
        self.return_subtiles = return_subtiles
        self.subtile_dim = subtile_dim
        
        # Store lengths of datasets
        if tropomi_tile_info is None:
            self.tropomi_len = 0
        else:
            self.tropomi_len = len(tropomi_tile_info)
        if oco2_tile_info is None:
            self.oco2_len = 0
        else:
            self.oco2_len = len(oco2_tile_info)


    def __len__(self):
        return max(self.tropomi_len, self.oco2_len)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}

        # Read the TROPOMI tile if it exists
        if (self.tropomi_tile_info is not None):
            # If the index is larger than the length of one of the datasets, loop around
            tropomi_idx = idx % self.tropomi_len

            # Load the TROPOMI tile
            tropomi_tile_info = self.tropomi_tile_info.iloc[tropomi_idx]
            tropomi_tile = np.load(tropomi_tile_info.loc[self.tile_file_column])

            tile_description = 'dataloader_lat_' + str(round(tropomi_tile_info.loc['lat'], 5)) + '_lon_' + str(round(tropomi_tile_info.loc['lon'], 5)) + '_' + tropomi_tile_info.loc['date']
            # sif_utils.plot_tile(tropomi_subtiles[0], tile_description + "_before_subtile_0.png")

            # Transform the TROPOMI tile
            if self.transform:
                tropomi_tile = self.transform(tropomi_tile)

            # print('Idx', idx, 'Upper left pixel', tropomi_tile[:, 0, 0])
            # print('Idx', idx, 'Upper right pixel', tropomi_tile[:, 0, 370])
            # print('Idx', idx, 'Bottom right pixel', tropomi_tile[:, 370, 370])
            # sif_utils.plot_tile(tropomi_tile, tile_description + "_after_subtile_0.png")

            # print('plotted')
            if self.return_subtiles:
                tropomi_subtiles = sif_utils.get_subtiles_list(tropomi_tile, self.subtile_dim)
                sample['tropomi_subtiles'] = torch.tensor(tropomi_subtiles, dtype=torch.float)
            else:
                sample['tropomi_tile'] = torch.tensor(tropomi_tile, dtype=torch.float)
            sample['tropomi_sif'] = tropomi_tile_info.loc['SIF']
            sample['tropomi_description'] = tile_description
            sample['tropomi_lon'] = tropomi_tile_info.loc['lon']
            sample['tropomi_lat'] = tropomi_tile_info.loc['lat']
            sample['tropomi_date'] = tropomi_tile_info.loc['date']

            # print('Upper left pixel', tropomi_subtiles[0, :, 0, 0])
            # print('Upper right pixel', tropomi_subtiles[3, :, 0, 99])
            # print('Bottom right pixel', tropomi_subtiles[15, :, 99, 99])
            # sif_utils.plot_tile(tropomi_subtiles[0], 'lat_after_transform_' + tile_description)
            # sif_utils.plot_tile(tropomi_subtiles[15], 'lat_after_transform_' + tile_description)

            # before_tensor = time.time()
            # Add the TROPOMI tile to the sample
            # loaded = time.time()

            # print('======= Dataloader timing =======')
            # print('Load tile from disk', before_transform-before_load)
            # print('Transform tile', before_subtiles-before_transform)
            # print('Get subtiles', before_tensor-before_subtiles)
            # print('Convert to Torch tensor', loaded-before_tensor)

        # Read the OCO-2 tile if it exists
        if (self.oco2_tile_info is not None):
            oco2_idx = idx % self.oco2_len
            oco2_tile_info = self.oco2_tile_info.iloc[oco2_idx]
            oco2_tile = np.load(oco2_tile_info.loc[self.tile_file_column])
            if self.transform:
                oco2_tile = self.transform(oco2_tile)
            if self.return_subtiles:
                oco2_subtiles = sif_utils.get_subtiles_list(oco2_tile, self.subtile_dim) #, self.max_subtile_cloud_cover)
                sample['oco2_subtiles'] = torch.tensor(oco2_subtiles, dtype=torch.float)
            else:
                sample['oco2_tile'] = torch.tensor(oco2_tile, dtype=torch.float)
            sample['oco2_sif'] = oco2_tile_info.loc['SIF']

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
