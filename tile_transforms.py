import numpy as np
import torch

class StandardizeTile(object):
    """
    Standardizes images so that each band has mean 0, standard deviation 1 
    """
    def __init__(self, band_means, band_stds):
        self.band_means = band_means
        self.band_stds = band_stds

    def __call__(self, tile):
        return (tile - self.band_means) / self.band_stds



