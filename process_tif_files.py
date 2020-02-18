# Import all packages in their own cell at the top of your notebook
import numpy as np
import os
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.plot import show

# import earthpy as et

#FILE_PATH = "datasets/GEE_data/41-42N_92-93W_reflectance.tif"
FILE_PATH = "datasets/GEE_data/44-45N_88-89W_reflectance_august.tif" # "datasets/GEE_data/41-42N_92-93W_reflectance.tif"
COVER_PATH = "datasets/GEE_data/44-45N_88-89W_cdl.tif"

with rio.open(FILE_PATH) as tiff_dataset:
    print('Bounds:', tiff_dataset.bounds)
    print('Transform:', tiff_dataset.transform)
    print('Metadata:', tiff_dataset.meta)
    print('Resolution:', tiff_dataset.res)
    print('Number of layers:', tiff_dataset.count)
    print('Coordinate reference system:', tiff_dataset.crs)
    print('Shape:', tiff_dataset.shape)

    all_bands_numpy = tiff_dataset.read()
    print('numpy array shape', all_bands_numpy.shape)

    # Read off RGB bands
    red = tiff_dataset.read(4) / 1000.
    green = tiff_dataset.read(3) / 1000.
    blue = tiff_dataset.read(2) / 1000.
    rgb = np.stack([red, green, blue], axis=0)
    print('rgb shape', rgb.shape)
    show(rgb)

    print(green)
    plt.imshow(green, cmap='Greens', vmin=0, vmax=1)
    plt.show()

    print('Left boudn', tiff_dataset.bounds.left)
    left, top = (-88.99991, 45.0)
    right, bottom = (-88.8, 44.8)
    left_idx, top_idx = tiff_dataset.index(left, top)  # tiff_dataset.bounds.left, tiff_dataset.bounds.top)
    right_idx, bottom_idx = tiff_dataset.index(right, bottom)
    print(left_idx, right_idx, top_idx, bottom_idx)
    print('Lat/Long of Upper Left Corner', tiff_dataset.xy(0, 0))
    print('Lat/Long of index (1000, 1000)', tiff_dataset.xy(1000, 1000))

with rio.open(COVER_PATH) as cover_dataset:
    print('Bounds:', cover_dataset.bounds)
    print('Transform', cover_dataset.transform)
    print('Metadata:', cover_dataset.meta)
    print('Resolution:', cover_dataset.res)
    print('Number of layers:', cover_dataset.count)
    print('Coordinate reference system:', cover_dataset.crs)
    covers = cover_dataset.read(1)

    # Plot distribution of specific crop
    mask = np.zeros_like(covers)
    mask[covers == 1] = 1.
    plt.imshow(mask, cmap='Greens', vmin=0, vmax=1)
    plt.savefig('datasets/GEE_data/visualizations/corn_pixels.png')
    plt.close()

    # Count how many pixels contain each crop
    total_pixels = covers.shape[0] * covers.shape[1]
    crop_to_count = dict()
    for i in range(covers.shape[0]):
        for j in range(covers.shape[1]):
            crop_type = covers[i][j]
            if crop_type in crop_to_count:
                crop_to_count[crop_type] += (1 / total_pixels)
            else:
                crop_to_count[crop_type] = 1 / total_pixels
    sorted_crops = sorted(crop_to_count.items(), key=lambda x: x[1], reverse=True)
    for crop, fraction in sorted_crops:
        print(str(crop) + ': ' + str(round(fraction * 100, 2)) + '%')



# image_stack = tifffile.imread(FILE_PATH)
# print(image_stack.shape)
# print(image_stack.dtype)