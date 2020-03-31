import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class EvalSubtileDataset(Dataset):

    def __init__(self, eval_tile_info, transform):
        self.eval_tile_info = eval_tile_info
        self.transform = transform

    def __len__(self):
        return len(self.eval_tile_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_tile_info = self.eval_tile_info.iloc[idx]
        subtile = np.load(current_tile_info.loc['subtile_file'])
        if self.transform:
            subtile = self.transform(subtile)
        subtile = torch.tensor(subtile, dtype=torch.float)

        sample = {'lon': current_tile_info.loc['lon'],
                  'lat': current_tile_info.loc['lat'],
                  'subtile': subtile,
                  'SIF': current_tile_info.loc['SIF']}
        return sample

