# Import all packages in their own cell at the top of your notebook
import csv
import math

import numpy as np
import os
import pandas as pd
import rasterio as rio
import rasterio.crs
from rasterio.warp import calculate_default_transform, reproject, Resampling

import matplotlib.pyplot as plt
import xarray as xr

from rasterio.plot import show



REFLECTANCE_FILES = [(pd.date_range(start="2018-08-16", end="2018-08-31"), "datasets/LandsatReflectance/aug_16")]  # ["datasets/LandsatReflectance/44-45N_88-89W_reflectance_max.tif"])]  # datasets/GEE_data/"}
COVER_FILE = "datasets/CDL_2019/CDL_2019_big.tif"  # CDL_2019_clip_20200218171505_325973588.tif"  #CDL_2019_clip_20200217173819_149334469.tif"  #"datasets/GEE_data/44-45N_88-89W_cdl.tif"
OUTPUT_CSV_FILE = "datasets/generated/reflectance_cover_to_sif.csv"
#REPROJECTED_COVER_FILE = "datasets/GEE_data/REPROJECTED_44-45N_88-89W_cdl.tif"
SIF_FILES = {7: "datasets/TROPOMI_SIF_2018/TROPO_SIF_07-2018.nc",
             8: "datasets/TROPOMI_SIF_2018/TROPO_SIF_08-2018.nc"}


# Plot corn pixels and print the most frequent crop types (sorted by percentage)
# "covers" should be a 2-D array of pixels, with a single integer at each pixel representin the crop type (according
# to the CDL classification)
def plot_and_print_covers(covers, filename):
    print('Covers!', covers)
    mask = np.zeros_like(covers)
    mask[covers == 1] = 1.
    plt.imshow(mask, cmap='Greens', vmin=0, vmax=1)
    plt.savefig('datasets/CDL_2019/visualizations/' + filename)
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


# Dataset format: image file name, SIF, date
dataset_rows = ["lon, lat, date, tile_file, SIF"]

# Open crop cover dataset
with rio.open(COVER_FILE) as cover_dataset:
    print('===================================================')
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
    print('===================================================')

    # print('ORIGINAL COVER STATS')
    # plot_and_print_covers(cover_dataset.read(1), 'original_covers_corn_big.png')

    # Loop through all time-periods for reflectance data
    for (date_range, reflectance_folder) in REFLECTANCE_FILES:
        # Open up the SIF file corresponding to that time period (start date)
        month = date_range[0].month
        sif_filename = SIF_FILES[month]
        sif_dataset = xr.open_dataset(sif_filename)
        sif_array = sif_dataset.dcSIF.sel(time=pd.date_range(start=date_range.date[0], end=date_range.date[-1]))
        print("SIF array shape", sif_array.shape)
        sif_array = sif_array.mean(dim='time')

        # If you select a large region, Google Drive breaks the reflectance data into multiple files; loop through
        # all of them.
        for reflectance_file in os.listdir(reflectance_folder):
            try:
                with rio.open(os.path.join(reflectance_folder, reflectance_file)) as reflectance_dataset:
                    print('===================================================')
                    print('REFLECTANCE DATASET')
                    print('Bounds:', reflectance_dataset.bounds)
                    print('Transform:', reflectance_dataset.transform)
                    print('Metadata:', reflectance_dataset.meta)
                    print('Resolution:', reflectance_dataset.res)
                    print('Number of layers:', reflectance_dataset.count)
                    print('Coordinate reference system:', reflectance_dataset.crs)
                    print('Shape:', reflectance_dataset.shape)
                    print('===================================================')

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
                    # plot_and_print_covers(reprojected_covers, filename="reprojected_cover_corn_big.png")

                    reflectance_numpy = reflectance_dataset.read()
                    print('Reflectance numpy array shape', reflectance_numpy.shape)
                    # print('Lat/Long of Upper Left Corner', reflectance_dataset.xy(0, 0))
                    # print('Lat/Long of index (1000, 1000)', reflectance_dataset.xy(1000, 1000))

                    # Just for testing
                    point = (44.9, -88.9)
                    left_idx, top_idx = reflectance_dataset.index(point[1], point[0])  # reflectance_dataset.bounds.left, reflectance_dataset.bounds.top)
                    print('===================================================')
                    print('TEST CASE: Point lat=', point[0], 'long=', point[1])
                    print('Using index method', left_idx, top_idx)

                    reflectance_height_idx, reflectance_width_idx = lat_long_to_index(point[0], point[1],
                                                                                      reflectance_dataset.bounds.top,
                                                                                      reflectance_dataset.bounds.left,
                                                                                      reflectance_dataset.res)
                    print("indices in reflectance:", reflectance_height_idx, reflectance_width_idx)
                    cover_height_idx, cover_width_idx = lat_long_to_index(point[0], point[1], cover_dataset.bounds.top,
                                                                          cover_dataset.bounds.left, reflectance_dataset.res)
                    print("indices in cover:", cover_height_idx, cover_width_idx)
                    print('===================================================')

                    # TODO
                    # Round boundaries to the nearest 0.2 degree
                    LEFT_BOUND = math.ceil(reflectance_dataset.bounds.left * 5) / 5  # -100.2
                    RIGHT_BOUND = math.floor(reflectance_dataset.bounds.right * 5) / 5  #-81.6
                    BOTTOM_BOUND = math.ceil(reflectance_dataset.bounds.bottom * 5) / 5  # 38.2
                    TOP_BOUND = math.floor(reflectance_dataset.bounds.top * 5) / 5  # 46.6
                    SIF_TILE_DEGREE_SIZE = 0.2
                    MAX_MISSING_FRACTION = 0.1  # If more than 10% of pixels in the tile are missing, throw the tile out
                    tile_width_pixels = round(SIF_TILE_DEGREE_SIZE / reflectance_dataset.res[0])
                    tile_height_pixels = round(SIF_TILE_DEGREE_SIZE / reflectance_dataset.res[1])
                    print('Tile pixels', tile_width_pixels, tile_height_pixels)
                    # For each "SIF tile", extract the tile of the reflectance data that maps to it
                    for left_degrees in np.arange(LEFT_BOUND, RIGHT_BOUND, SIF_TILE_DEGREE_SIZE):
                        for bottom_degrees in np.arange(BOTTOM_BOUND, TOP_BOUND, SIF_TILE_DEGREE_SIZE):
                            right_edge = left_degrees + SIF_TILE_DEGREE_SIZE
                            top_edge = bottom_degrees + SIF_TILE_DEGREE_SIZE

                            # Find indices in datasets. Note that for the cover dataset, we're actually extracting from the
                            # version that has already been re-sampled to the same resolution as reflectance, so we use
                            # reflectance_dataset's resolution.
                            cover_bottom_idx, cover_left_idx = lat_long_to_index(bottom_degrees, left_degrees,
                                                                                 cover_dataset.bounds.top,
                                                                                 cover_dataset.bounds.left,
                                                                                 reflectance_dataset.res)
                            # cover_bottom_idx, cover_right_idx = lat_long_to_index(bottom_edge, right_edge,
                            #                                                       cover_dataset.bounds.top,
                            #                                                       cover_dataset.bounds.left,
                            #                                                       reflectance_dataset.res)
                            reflectance_bottom_idx, reflectance_left_idx = lat_long_to_index(bottom_degrees, left_degrees,
                                                                                          reflectance_dataset.bounds.top,
                                                                                          reflectance_dataset.bounds.left,
                                                                                          reflectance_dataset.res)
                            # reflectance_bottom_idx, reflectance_right_idx = lat_long_to_index(bottom_edge, right_edge,
                            #                                                                   reflectance_dataset.bounds.top,
                            #                                                                   reflectance_dataset.bounds.left,
                            #                                                                   reflectance_dataset.res)
                            cover_top_idx = cover_bottom_idx - tile_height_pixels
                            reflectance_top_idx = reflectance_bottom_idx - tile_height_pixels
                            cover_right_idx = cover_left_idx + tile_width_pixels
                            reflectance_right_idx = reflectance_left_idx + tile_width_pixels
                            #print("Cover shape", reprojected_covers.shape)
                            #print("Cover idx: top", cover_top_idx, "bottom", cover_bottom_idx, "left", cover_left_idx,
                            #      "right", cover_right_idx)
                            #print("Reflectance shape", reflectance_dataset.shape)
                            #print("Reflectance idx: top", reflectance_top_idx, "bottom", reflectance_bottom_idx,
                            #      "left", reflectance_left_idx, "right", reflectance_right_idx)

                            if cover_top_idx < 0 or cover_left_idx < 0 or reflectance_top_idx < 0 or reflectance_left_idx < 0:
                                print("Index was negative!")
                                continue
                            if (cover_bottom_idx >= reprojected_covers.shape[0] or
                                    cover_right_idx >= reprojected_covers.shape[1] or
                                    reflectance_bottom_idx >= reflectance_dataset.shape[0] or
                                    reflectance_right_idx >= reflectance_dataset.shape[1]):
                                print("Index went beyond edge of array!")
                                continue

                            # Extract the cover and reflectance tiles (covering the same region as the SIF tile)
                            cover_tile = reprojected_covers[cover_top_idx:cover_bottom_idx, cover_left_idx:cover_right_idx]
                            reflectance_tile = reflectance_numpy[:, reflectance_top_idx:reflectance_bottom_idx,
                                                                 reflectance_left_idx:reflectance_right_idx]
                            print("Cover tile shape", cover_tile.shape)
                            print("Fraction of nonzeros in cover tile:", np.count_nonzero(cover_tile) / (cover_tile.shape[0] * cover_tile.shape[1]))
                            print("Reflectance tile shape", reflectance_tile.shape)
                            print("Fraction of nonzeros in reflectance tile:", np.count_nonzero(reflectance_tile) / (reflectance_tile.shape[0] * reflectance_tile.shape[1] * reflectance_tile.shape[2]))
                            assert(cover_tile.shape[0:2] == reflectance_tile.shape[1:3])
                            if np.count_nonzero(cover_tile) / len(cover_tile) < 1 - MAX_MISSING_FRACTION:
                                continue
                            if np.count_nonzero(reflectance_tile) / len(reflectance_tile) < 1 - MAX_MISSING_FRACTION:
                                continue
                            COVERS_TO_MASK = [1, 5, 176, 141]
                            cover_masks = []
                            for i, cover_type in enumerate(COVERS_TO_MASK):
                                crop_mask = np.zeros_like(cover_tile)
                                crop_mask[cover_tile == cover_type] = 1.
                                cover_masks.append(crop_mask)
                            cover_masks = np.stack(cover_masks, axis=0)
                            #print("Cover masks shape", cover_masks.shape)
                            reflectance_and_cover_tile = np.concatenate((reflectance_tile, cover_masks), axis=0)
                            np.transpose(reflectance_and_cover_tile, (1, 2, 0))
                            print("Combined tile shape", reflectance_and_cover_tile.shape)

                            # sif_lat_idx, sif_long_idx = lat_long_to_index(left_degrees + SIF_TILE_DEGREE_SIZE / 2,
                            #                                               bottom_degrees + SIF_TILE_DEGREE_SIZE / 2,
                            #                                               90., -180., [SIF_TILE_DEGREE_SIZE,
                            #                                                            SIF_TILE_DEGREE_SIZE])
                            # print("SIF indices", sif_lat_idx, sif_long_idx)
                            center_lat = bottom_degrees + SIF_TILE_DEGREE_SIZE / 2
                            center_lon = left_degrees + SIF_TILE_DEGREE_SIZE / 2
                            sif_tile = sif_array.sel(lat=center_lat, lon=center_lon, method='nearest').values
                            print("SIF TILE", sif_tile)

                            # Write reflectance/cover pixels tile (as Numpy array) to .npy file
                            npy_filename = "datasets/generated/reflectance_lat_" + str(bottom_degrees + SIF_TILE_DEGREE_SIZE / 2) + "_lon_" + str(left_degrees + SIF_TILE_DEGREE_SIZE / 2) + ".npy"
                            np.save(npy_filename, reflectance_and_cover_tile)
                            print("date", date_range.date[0].isoformat())
                            dataset_rows.append([center_lon, center_lat, date_range.date[0].isoformat(), npy_filename, sif_tile])
            except Exception as error:
                print("Reading reflectance file", reflectance_file, "failed")
                print(error)

with open(OUTPUT_CSV_FILE, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in dataset_rows:
        csv_writer.writerow(row)

            # Todo: 1) Download reflectance/crop data for larger regions
            #  2) iterate across images at 0.2 degree intervals. Extract those pixels.
            #  3) Check to make sure no more than 10% pixels covered with cloud. If so, create dataset with many bands
            # plus total SIF ground truth.
