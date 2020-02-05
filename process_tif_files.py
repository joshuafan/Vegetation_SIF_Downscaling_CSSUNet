# Import all packages in their own cell at the top of your notebook
import numpy as np
import os
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.plot import show

# import earthpy as et

FILE_PATH = "datasets/GEE_data/41-42N_92-93W_reflectance.tif"
# FILE_PATH = "datasets/GEE_data/44-45N_87-88W_reflectance.tif" # "datasets/GEE_data/41-42N_92-93W_reflectance.tif"  # "datasets/GEE_data/44-45N_87-88W_cdl.tif"

with rio.open(FILE_PATH) as tiff_dataset:
    print('Bounds:', tiff_dataset.bounds)
    print('Metadata:', tiff_dataset.meta)
    print('Resolution:', tiff_dataset.res)
    print('Number of layers:', tiff_dataset.count)
    print('Coordinate reference system:', tiff_dataset.crs)
    print('Shape:', tiff_dataset.shape)

    red = tiff_dataset.read(4) / 1000.
    green = tiff_dataset.read(3) / 1000.
    blue = tiff_dataset.read(2) / 1000.
    rgb = np.stack([red, green, blue], axis=0)
    print('rgb shape', rgb.shape)
    show(rgb)

    print(green)
    plt.imshow(green, cmap='Greens', vmin=0, vmax=1)
    plt.show()

    left, top = (-88.0, 45.0)
    right, bottom = (-87.8, 44.8)
    left_idx, top_idx = tiff_dataset.index(left, top)
    right_idx, bottom_idx = tiff_dataset.index(right, bottom)
    print(left_idx, right_idx, top_idx, bottom_idx)


# image_stack = tifffile.imread(FILE_PATH)
# print(image_stack.shape)
# print(image_stack.dtype)