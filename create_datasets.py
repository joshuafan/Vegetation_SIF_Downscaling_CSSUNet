# Import all packages in their own cell at the top of your notebook
import numpy as np
import os
import rasterio as rio
import rasterio.crs
from rasterio.warp import calculate_default_transform, reproject, Resampling

import matplotlib.pyplot as plt
from rasterio.plot import show

# import earthpy as et

# REFLECTANCE_FILES = {"2017-07-07": "datasets/GEE_data/44-45N_88-89W_reflectance_max.tif"}

# TODO Change to directory
REFLECTANCE_FILES = {pd.daterange(start="2018-08-01", end="2018-08-15"): ["datasets/LandsatReflectance/44-45N_88-89W_reflectance_max.tif"]}  # datasets/GEE_data/"}
COVER_FILE = "datasets/CDL_2019/CDL_2019_clip_20200218171505_325973588.tif"  #CDL_2019_clip_20200217173819_149334469.tif"  #"datasets/GEE_data/44-45N_88-89W_cdl.tif"
#REPROJECTED_COVER_FILE = "datasets/GEE_data/REPROJECTED_44-45N_88-89W_cdl.tif"
SIF_FILES = {7: "datasets/TROPOMI_SIF_2018/TROPO_SIF_07-2018.nc",
             8: "datasets/TROPOMI_SIF_2018/TROPO_SIF_08-2018.nc"}


def plot_and_print_covers(covers, filename):
    print('Covers!', covers)
    mask = np.zeros_like(covers)
    mask[covers == 1] = 1.
    plt.imshow(mask, cmap='Greens', vmin=0, vmax=1)
    plt.savefig('datasets/CDL/visualizations/' + filename)
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


def lat_long_to_index(lat, long, left_bound, top_bound, resolution):
    height_idx = (top_bound - lat) / resolution[0]
    width_idx = (long - left_bound) / resolution[1]
    return int(height_idx), int(width_idx)


# Dataset format: image file name, SIF, date
dataset_rows = []
with rio.open(COVER_FILE) as cover_dataset:
    print('COVER DATASET')
    print('Bounds:', cover_dataset.bounds)
    print('Transform', cover_dataset.transform)
    print('Metadata:', cover_dataset.meta)
    print('Resolution:', cover_dataset.res)
    print('Number of layers:', cover_dataset.count)
    print('Coordinate reference system:', cover_dataset.crs)
    print('Shape:', cover_dataset.shape)
    print('Width:', cover_dataset.width)
    print('Height:', cover_dataset.height)
    print('ORIGINAL COVER STATS')
    # plot_and_print_covers(cover_dataset.read(1), 'original_covers_corn.png')

    for date, reflectance_folder in REFLECTANCE_FILES.items():
        for reflectance_file in reflectance_folder:  # os.path.listdir(reflectance_folder)
        with rio.open(reflectance_file) as reflectance_dataset:
            print('REFLECTANCE DATASET')
            print('Bounds:', reflectance_dataset.bounds)
            print('Transform:', reflectance_dataset.transform)
            print('Metadata:', reflectance_dataset.meta)
            print('Resolution:', reflectance_dataset.res)
            print('Number of layers:', reflectance_dataset.count)
            print('Coordinate reference system:', reflectance_dataset.crs)
            print('Shape:', reflectance_dataset.shape)

            # resample cover data to target shape
            height_upscale_factor = cover_dataset.res[0] / reflectance_dataset.res[0]
            width_upscale_factor = cover_dataset.res[1] / reflectance_dataset.res[1]

            print('Upscale factor: height', height_upscale_factor, 'width', width_upscale_factor)
            reprojected_covers = cover_dataset.read(
                out_shape=(
                    int(cover_dataset.height * height_upscale_factor),
                    int(cover_dataset.width * width_upscale_factor)
                ),
                resampling=Resampling.nearest
            )

            # scale image transform
            new_cover_transform = cover_dataset.transform * cover_dataset.transform.scale(
                (cover_dataset.width / reprojected_covers.shape[-2]),
                (cover_dataset.height / reprojected_covers.shape[-1])
            )
            print('Transform', new_cover_transform)

            print('REPROJECTED COVER DATASET')
            reprojected_covers = np.squeeze(reprojected_covers)
            print('Shape:', reprojected_covers.shape)

            # Plot distribution of specific crop
            # plot_and_print_covers(reprojected_covers, filename="reprojected_cover_corn.png")

            all_bands_numpy = reflectance_dataset.read()
            print('numpy array shape', all_bands_numpy.shape)
            # print('Lat/Long of Upper Left Corner', reflectance_dataset.xy(0, 0))
            # print('Lat/Long of index (1000, 1000)', reflectance_dataset.xy(1000, 1000))

            point = (44.9, -88.9)
            left_idx, top_idx = reflectance_dataset.index(reflectance_dataset.bounds.left, reflectance_dataset.bounds.top)
            print('Using index method', left_idx, top_idx)

            reflectance_height_idx, reflectance_width_idx = lat_long_to_index(point[0], point[1], reflectance_dataset.bounds.left, reflectance_dataset.bounds.top, reflectance_dataset.res)
            print("indices in reflectance:", reflectance_height_idx, reflectance_width_idx)
            cover_height_idx, cover_width_idx = lat_long_to_index(point[0], point[1], cover_dataset.bounds.left, cover_dataset.bounds.top, reflectance_dataset.res)
            print("indices in cover:", cover_height_idx, cover_width_idx)
            lengths = 0.2 / reflectance_dataset.res[0], 0.2 / reflectance_dataset.res[1]
            print('Lengths', lengths)

            # Todo: 1) Download reflectance/crop data for larger regions
            #  2) iterate across images at 0.2 degree intervals. Extract those pixels.
            #  3) Check to make sure no more than 10% pixels covered with cloud. If so, create dataset with many bands
            # plus total SIF ground truth.
