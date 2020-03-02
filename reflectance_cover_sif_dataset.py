from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")



class ReflectanceCoverSIFDataset(Dataset):
    """Dataset mapping a tile (with reflectance/cover bands) to total SIF"""

    def __init__(self, tile_info_file):  # root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tile_info = pd.read_csv(tile_info_file)
        #self.root_dir = root_dir
        #self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_tile_info = self.tile_info.iloc[idx]
        year, month, day_of_year = utils.parse_date_string(current_tile_info.loc['date'])

        SAMPLE_ID_IDX = 0
        ORIENTATION_IDX = 4
        SIZE_IDX = 6
        ORIENTATION_TO_IMAGE_IDX = {"front": 0,
                                    "left": 1,
                                    "right": 2,
                                    "back": 3,
                                    "top": 4,
                                    "bottom": 5}

        folder_path = os.path.join(self.root_dir, str(self.features.iloc[idx].loc['SampleID']))
        sorted_images = sorted(os.listdir(folder_path))

        which_image_in_folder = ORIENTATION_TO_IMAGE_IDX[self.features.iloc[idx].loc['view']]

        if len(sorted_images) < 6:
            print("Folder " + folder_path + " had " + str(len(sorted_images)) + " images")
        if which_image_in_folder >= len(sorted_images):
            which_image_in_folder = len(sorted_images) - 1
        img_name = os.path.join(folder_path, sorted_images[which_image_in_folder])
        image = io.imread(img_name)
        image = Image.fromarray(image, mode="RGB")

        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)

        if self.transform:
            image = self.transform(image)
        sample = {'lon': current_tile_info.loc['lon'],
                  'lat': current_tile_info.loc['lat'],
                  'current_tile_info.loc["SIF"], 'tile': image, 'SIF': current_tile_info.loc["SIF"]}


        return sample