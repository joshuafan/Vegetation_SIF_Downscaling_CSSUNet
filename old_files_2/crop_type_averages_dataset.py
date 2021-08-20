from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import sif_utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class CropTypeAveragesDataset(Dataset):
    """
    Dataset mapping crop-specific features (reflectance/cover for each crop) to total SIF.
    
    Each element contains a "features" and "crop_fractions" entry. 
    
    "features" is a dictionary mapping from crop type (name) to a torch tensor of features (given by feature_names)

    "cover_fractions" is a dictionary mapping from crop type to the percentage of that crop type
     """

    def __init__(self, tile_info, crop_types, feature_names):
        """
        Args:
            tile_info_file (string): Pandas dataframe containing metadata (features)
        """
        self.tile_info = tile_info
        self.crop_types = crop_types
        self.feature_names = feature_names

    def __len__(self):
        return len(self.tile_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_tile_features = self.tile_info.iloc[idx]

        # Maps from crop type to numpy array of features (in the order given by "self.feature_names")
        features = dict()

        # Maps from crop type to fraction of area covered by that crop type
        cover_fractions = dict()

        # Loop through all crop types
        for crop_type in list(self.crop_types.keys()) + ['other']:
            # Extract fraction of area covered by crop type
            area_fraction = current_tile_features.loc[crop_type + '_cover']
            cover_fractions[crop_type] = area_fraction

            # Extract all features for this crop
            crop_features = []
            for feature_name in self.feature_names:
                feature_value = current_tile_features.loc[crop_type + '_' + feature_name]
                if np.isnan(feature_value):
                    feature_value = 0.
                crop_features.append(feature_value)
            features[crop_type] = torch.tensor(crop_features, dtype=torch.float)

        sample = {'lon': current_tile_features.loc['lon'],
                  'lat': current_tile_features.loc['lat'],
                  'features': features,
                  'cover_fractions': cover_fractions,
                  'SIF': current_tile_features.loc["SIF"]}
        return sample


class CropTypeAveragesFromTileDataset(Dataset):
    """
    Dataset mapping crop-specific features (reflectance/cover for each crop) to total SIF.
    The crop-specific features are extracted from an image tile.
    
    Each element contains a "features" and "crop_fractions" entry. 
    
    "features" is a dictionary mapping from crop type (name) to a torch tensor of features (given by feature_names)

    "cover_fractions" is a dictionary mapping from crop type to the percentage of that crop type
     """

    def __init__(self, tile_info, crop_types, feature_names, transform, missing_idx=-1):
        """
        Args:
            tile_info_file (string): Pandas dataframe containing metadata (features)

            Here crop_types should exclude other
        """
        self.tile_info = tile_info
        self.crop_types = crop_types
        self.feature_names = feature_names
        self.transform = transform
        self.missing_idx = missing_idx

    def __len__(self):
        return len(self.tile_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_tile_features = self.tile_info.iloc[idx]
        tile = np.load(current_tile_features.loc['tile_file'])
        tile = self.transform(tile)

        # Reshape tile into a list of pixels (pixels x channels)
        pixels = np.moveaxis(tile, 0, -1)
        pixels = pixels.reshape((-1, pixels.shape[2]))

        # Maps from crop type to numpy array of features (in the order given by "self.feature_names")
        features = dict()

        # Maps from crop type to fraction of area covered by that crop type
        cover_fractions = dict()

        for crop_type, crop_idx in self.crop_types.items():
            # Compute fraction land cover for this crop
            crop_cover = np.mean(pixels[:, crop_idx])
            cover_fractions[crop_type] = crop_cover
            assert(crop_cover >= 0 and crop_cover <= 1)

            # Extract pixels belonging to this crop & are not obscured by clouds
            crop_pixels = pixels[(pixels[:, crop_idx] == 1) & (pixels[:, self.missing_idx] == 1)]
            # print('crop cover', crop_cover)
            # print('Crop type', crop_type, 'shape', crop_pixels.shape)
            # print(crop_pixels.shape)
            crop_features = []
            for feature_name, feature_idx in self.feature_names.items():
                if crop_pixels.shape[0] > 0:
                    feature_mean = np.mean(crop_pixels[:, feature_idx])
                else:
                    # If coverage is 0%, it should get ignored
                    feature_mean = -99999.
                crop_features.append(feature_mean)
            features[crop_type] = torch.tensor(crop_features)

        # "any_cover": 1 if pixel is covered by any of the crop types; 0 if it is not
        crop_indices = list(self.crop_types.values())
        any_cover = np.sum(pixels[:, crop_indices], axis=1)
        other_pixels = pixels[any_cover == 0]  # Pixels not covered by any of the given crop types

        # Compute featiures for the region that's not covered by any crop
        other_cover = 1 - np.mean(any_cover)
        assert(other_cover >= 0 and other_cover <= 1)
        cover_fractions['other'] = other_cover
        other_features = []
        for feature_name, feature_idx in self.feature_names.items():
            if other_pixels.shape[0] > 0:
                feature_mean = np.mean(other_pixels[:, feature_idx])
            else:
                feature_mean = -99999.
            other_features.append(feature_mean)
        features['other'] = torch.tensor(other_features)

        sample = {'lon': current_tile_features.loc['lon'],
                  'lat': current_tile_features.loc['lat'],
                  'date': current_tile_features.loc['date'],
                  'tile': tile,
                  'tile_file': current_tile_features.loc['tile_file'],
                  'features': features,
                  'cover_fractions': cover_fractions,
                  'SIF': current_tile_features.loc["SIF"]}
                #   'cloud_fraction': current_tile_features.loc['cloud_fraction'],
                #   'num_soundings': current_tile_features.loc['num_soundings']}
        return sample
