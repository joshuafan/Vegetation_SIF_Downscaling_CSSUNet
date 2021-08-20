# Import all packages in their own cell at the top of your notebook
import numpy as np
import os
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.plot import show

# import earthpy as et

#FILE_PATH = "datasets/GEE_data/41-42N_92-93W_reflectance.tif"
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
FILE_PATH = os.path.join(DATA_DIR, "LandsatReflectance/2018-08-01/corn_belt_reflectance_2018-08-01_box_0_0.tif")  # "datasets/GEE_data/41-42N_92-93W_reflectance.tif"
COVER_PATH = os.path.join(DATA_DIR, "CDL_2018/corn_belt_cdl_2018-08-01_epsg.tif")  # clip_20200316232647_1548459805.tif"

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
    red = tiff_dataset.read(4)
    green = tiff_dataset.read(3)
    blue = tiff_dataset.read(2)
    print('max red', np.max(red))
    print('mean read', np.mean(red))
    red = red / (2*np.mean(red))
    green = green / (2*np.mean(green))
    blue = blue / (2*np.mean(blue))
    rgb = np.stack([red, green, blue], axis=0)
    print('rgb shape', rgb.shape)
    show(rgb)
    plt.savefig('exploratory_plots/dataset_full_rgb_visualization.png')
    plt.close()

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
    plt.savefig('exploratory_plots/dataset_corn_pixels.png')
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
