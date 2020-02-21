# Import all packages in their own cell at the top of your notebook
import numpy as np
import os
import rasterio as rio
import rasterio.crs
from rasterio.warp import calculate_default_transform, reproject, Resampling

import matplotlib.pyplot as plt
from rasterio.plot import show

# import earthpy as et

REFLECTANCE_FILES = {"2017-07-07": "datasets/GEE_data/44-45N_88-89W_reflectance_max.tif"}
COVER_FILE = "datasets/CDL/CDL_2019_big.tif"  #"datasets/GEE_data/44-45N_88-89W_cdl.tif"
SIF_FILES = {7: "datasets/TROPOMI_SIF_2018/TROPO_SIF_07-2018.nc",
             8: "datasets/TROPOMI_SIF_2018/TROPO_SIF_08-2018.nc"}


# Plot corn pixels and print the most frequent crop types (sorted by percentage)
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
                crop_to_count[crop_type] += 1.
            else:
                crop_to_count[crop_type] = 1.
    sorted_crops = sorted(crop_to_count.items(), key=lambda x: x[1], reverse=True)
    for crop, count in sorted_crops:
        print(str(crop) + ': ' + str(round((count / total_pixels) * 100, 2)) + '%')


def lat_long_to_index(lat, long, dataset_top_bound, dataset_left_bound, resolution):
    height_idx = (dataset_top_bound - lat) / resolution[0]
    width_idx = (long - dataset_left_bound) / resolution[1]
    return int(height_idx), int(width_idx)


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
    plot_and_print_covers(cover_dataset.read(1), 'original_covers_corn_big.png')

    for date, reflectance_file in REFLECTANCE_FILES.items():
        with rio.open(reflectance_file) as reflectance_dataset:
            print('REFLECTANCE DATASET')
            print('Bounds:', reflectance_dataset.bounds)
            print('Transform:', reflectance_dataset.transform)
            print('Metadata:', reflectance_dataset.meta)
            print('Resolution:', reflectance_dataset.res)
            print('Number of layers:', reflectance_dataset.count)
            print('Coordinate reference system:', reflectance_dataset.crs)
            print('Shape:', reflectance_dataset.shape)

            # Resample cover data to match the resolution of the reflectance dataset
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
            # new_cover_transform = cover_dataset.transform * cover_dataset.transform.scale(
            #     (cover_dataset.width / reprojected_covers.shape[-2]),
            #     (cover_dataset.height / reprojected_covers.shape[-1])
            # )
            # print('Transform', new_cover_transform)

            print('REPROJECTED COVER DATASET')
            reprojected_covers = np.squeeze(reprojected_covers)
            print('Shape:', reprojected_covers.shape)

            # Plot distribution of specific crop
            plot_and_print_covers(reprojected_covers, filename="reprojected_cover_corn_big.png")

            all_bands_numpy = reflectance_dataset.read()
            print('numpy array shape', all_bands_numpy.shape)
            # print('Lat/Long of Upper Left Corner', reflectance_dataset.xy(0, 0))
            # print('Lat/Long of index (1000, 1000)', reflectance_dataset.xy(1000, 1000))

            point = (44.9, -88.9)
            left_idx, top_idx = reflectance_dataset.index(reflectance_dataset.bounds.left, reflectance_dataset.bounds.top)
            print('Using index method', left_idx, top_idx)

            reflectance_height_idx, reflectance_width_idx = lat_long_to_index(point[0], point[1],
                                                                              reflectance_dataset.bounds.top,
                                                                              reflectance_dataset.bounds.left,
                                                                              reflectance_dataset.res)
            print("indices in reflectance:", reflectance_height_idx, reflectance_width_idx)
            cover_height_idx, cover_width_idx = lat_long_to_index(point[0], point[1], cover_dataset.bounds.top,
                                                                  cover_dataset.bounds.left, reflectance_dataset.res)
            print("indices in cover:", cover_height_idx, cover_width_idx)
            lengths = 0.2 / reflectance_dataset.res[0], 0.2 / reflectance_dataset.res[1]
            print('Lengths', lengths)

            LEFT_BOUND = -100.2
            RIGHT_BOUND = -81.6
            BOTTOM_BOUND = 38.2
            TOP_BOUND = 46.6
            SIF_TILE_SIZE = 0.2
            MAX_MISSING_FRACTION = 0.1  # If more than 10% of pixels in the tile are missing, throw the tile out

            for left_edge in range(LEFT_BOUND, RIGHT_BOUND, SIF_TILE_SIZE):
                for bottom_edge in range(BOTTOM_BOUND, TOP_BOUND, SIF_TILE_SIZE):
                    right_edge = left_edge + SIF_TILE_SIZE
                    top_edge = bottom_edge + SIF_TILE_SIZE

                    # Find indices in datasets
                    cover_top_idx, cover_left_idx = lat_long_to_index(top_edge, left_edge, cover_dataset.bounds.top,
                                                                      cover_dataset.bounds.left, cover_dataset.res)
                    cover_bottom_idx, cover_right_idx = lat_long_to_index(bottom_edge, right_edge,
                                                                          cover_dataset.bounds.top,
                                                                          cover_dataset.bounds.left,
                                                                          cover_dataset.res)
                    reflectance_top_idx, reflectance_left_idx = lat_long_to_index(top_edge, left_edge,
                                                                                  reflectance_dataset.bounds.top,
                                                                                  reflectance_dataset.bounds.left,
                                                                                  reflectance_dataset.res)
                    reflectance_bottom_idx, reflectance_right_idx = lat_long_to_index(bottom_edge, right_edge,
                                                                                      reflectance_dataset.bounds.top,
                                                                                      reflectance_dataset.bounds.left,
                                                                                      reflectance_dataset.res)
                    print("Cover bounds: top", cover_top_idx, "bottom", cover_bottom_idx, "left", cover_left_idx,
                          "right", cover_right_idx)
                    cover_tile = reprojected_covers[cover_top_idx:cover_bottom_idx, cover_left_idx:cover_right_idx]
                    reflectance_tile = reprojected_covers[reflectance_top_idx:reflectance_bottom_idx,
                                                          reflectance_left_idx:reflectance_right_idx]
                    print("Cover tile shape", cover_tile.shape)
                    print("Nonzeros in cover tile:", np.count_nonzero(cover_tile), "of", len(cover_tile))
                    print("Reflectance tile shape", reflectance_tile.shape)
                    print("Nonzeros in reflectance tile:", np.count_nonzero(reflectance_tile), "of", len(reflectance_tile))
                    assert(cover_tile.shape[0:2] == reflectance_tile.shape[0:2])
                    if np.count_nonzero(cover_tile) / len(cover_tile) < 1 - MAX_MISSING_FRACTION:
                        continue
                    if np.count_nonzero(reflectance_tile) / len(reflectance_tile) < 1 - MAX_MISSING_FRACTION:
                        continue
                    COVERS_TO_MASK = [1, 5, 176, 141]
                    cover_masks = np.zeros((cover_tile.shape, len(COVERS_TO_MASK)))
                    print("Cover mask shape", cover_masks.shape)
                    reflectance_and_cover_tile = np.concatenate(reflectance_tile, cover_masks, dim=2)
                    print("Combined tile shape", reflectance_and_cover_tile.shape)

                    sif_lat_idx, sif_long_idx = lat_long_to_index(left_edge + SIF_TILE_SIZE / 2,
                                                                  bottom_edge + SIF_TILE_SIZE / 2,
                                                                  90., -180., SIF_TILE_SIZE)
                    print("SIF indices", sif_lat_idx, sif_long_idx)





            # Todo: 1) Download reflectance/crop data for larger regions
            #  2) iterate across images at 0.2 degree intervals. Extract those pixels.
            #  3) Check to make sure no more than 10% pixels covered with cloud. If so, create dataset with many bands
            # plus total SIF ground truth.
