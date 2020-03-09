# Import all packages in their own cell at the top of your notebook
import csv
import math

import numpy as np
import os
import pandas as pd
import rasterio as rio
import rasterio.crs
from rasterio.warp import calculate_default_transform, reproject, Resampling
import traceback
import matplotlib.pyplot as plt
import xarray as xr

from rasterio.plot import show
from sif_utils import lat_long_to_index


REFLECTANCE_FILES = [(pd.date_range(start="2018-08-16", end="2018-08-31"), "datasets/LandsatReflectance/aug_16")]  # ["datasets/LandsatReflectance/44-45N_88-89W_reflectance_max.tif"])]  # datasets/GEE_data/"}
COVER_FILE = "datasets/CDL_2019/CDL_2019_big.tif"  # CDL_2019_clip_20200218171505_325973588.tif"  #CDL_2019_clip_20200217173819_149334469.tif"  #"datasets/GEE_data/44-45N_88-89W_cdl.tif"
OUTPUT_CSV_FILE = "datasets/generated/reflectance_cover_to_sif.csv"
SIF_FILE = "datasets/SIF/TROPO-SIF_01deg_biweekly_Apr18-Jan20.nc"


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



# Dataset format: image file name, SIF, date
dataset_rows = [["lon", "lat", "date", "tile_file", "SIF"]]

# Open crop cover dataset
with rio.open(COVER_FILE) as cover_dataset:
    # Print stats about cover dataset
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
        sif_dataset = xr.open_dataset(SIF_FILE)
        sif_array = sif_dataset.sif_dc.sel(time=slice(date_range.date[0], date_range.date[-1]))
        print("SIF array shape", sif_array.shape)
        print("SIF array:", sif_array)
        sif_array = sif_array.mean(dim='time')

        # If you select a large region, Google Drive breaks the reflectance data into multiple files; loop through
        # all of them.
        for reflectance_file in os.listdir(reflectance_folder):
            try:
                with rio.open(os.path.join(reflectance_folder, reflectance_file)) as reflectance_dataset:
                    # Print stats about reflectance file
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

                    # Resample cover data into target resolution
                    SIF_TILE_DEGREE_SIZE = 0.1
                    target_res = (reflectance_dataset.res[0], reflectance_dataset.res[1])  #SIF_TILE_DEGREE_SIZE / TARGET_TILE_SIZE
                    TARGET_TILE_SIZE = int(SIF_TILE_DEGREE_SIZE / target_res[0])
                    print("target tile size", TARGET_TILE_SIZE)
                    cover_height_upscale_factor = cover_dataset.res[0] / target_res[0]  # reflectance_dataset.res[0]
                    cover_width_upscale_factor = cover_dataset.res[1] / target_res[1]  # reflectance_dataset.res[1]
                    print('Upscale factor: height', cover_height_upscale_factor, 'width', cover_width_upscale_factor)
                    reprojected_covers = cover_dataset.read(
                        out_shape=(
                            int(cover_dataset.height * cover_height_upscale_factor),
                            int(cover_dataset.width * cover_width_upscale_factor)
                        ),
                        resampling=Resampling.mode
                    )
                    print('REPROJECTED COVER DATASET')
                    reprojected_covers = np.squeeze(reprojected_covers)
                    print('Shape:', reprojected_covers.shape)

                    # Resample reflectance data into target resolution
                    #reflectance_height_upscale_factor = reflectance_dataset.res[0] / target_res[0]
                    #reflectance_width_upscale_factor = reflectance_dataset.res[1] / target_res[1]
                    #reprojected_reflectances = reflectance_dataset.read(
                    #    out_shape=(
                    #        int(reflectance_dataset.height * reflectance_height_upscale_factor),
                    #        int(reflectance_dataset.width * reflectance_width_upscale_factor)
                    #    ),
                    #    resampling=Resampling.bilinear
                    # 
                    #)
                    reprojected_reflectances = reflectance_dataset.read()
                    print('REPROJECTED REFLECTANCE DATASET')
                    print('Shape:', reprojected_reflectances.shape)

                    # Plot distribution of specific crop
                    # plot_and_print_covers(reprojected_covers, filename="reprojected_cover_corn_big.png")

                    # Read reflectance data into numpy array
                    #reflectance_numpy = reflectance_dataset.read()
                    #print('Reflectance numpy array shape', reflectance_numpy.shape)
                    # print('Lat/Long of Upper Left Corner', reflectance_dataset.xy(0, 0))
                    # print('Lat/Long of index (1000, 1000)', reflectance_dataset.xy(1000, 1000))

                    # Just for testing
                    #point = (44.9, -88.9)
                    #left_idx, top_idx = reprojected_reflectances.index(point[1], point[0])  # reflectance_dataset.bounds.left, reflectance_dataset.bounds.top)
                    #print('===================================================')
                    #print('TEST CASE: Point lat=', point[0], 'long=', point[1])
                    #print('Using index method', left_idx, top_idx)

                    #reflectance_height_idx, reflectance_width_idx = lat_long_to_index(point[0], point[1],
                    #                                                                  reflectance_dataset.bounds.top,
                    #                                                                  reflectance_dataset.bounds.left,
                    #                                                                  target_res)
                    #print("indices in reflectance:", reflectance_height_idx, reflectance_width_idx)
                    #cover_height_idx, cover_width_idx = lat_long_to_index(point[0], point[1], cover_dataset.bounds.top,
                    #                                                      cover_dataset.bounds.left, target_res)
                    #print("indices in cover:", cover_height_idx, cover_width_idx)
                    #print('===================================================')

                    # Round boundaries to the nearest 0.1 degree
                    LEFT_BOUND = math.ceil(reflectance_dataset.bounds.left * 10) / 10  # -100.2
                    RIGHT_BOUND = math.floor(reflectance_dataset.bounds.right * 10) / 10  #-81.6
                    BOTTOM_BOUND = math.ceil(reflectance_dataset.bounds.bottom * 10) / 10  # 38.2
                    TOP_BOUND = math.floor(reflectance_dataset.bounds.top * 10) / 10  # 46.6
                    MAX_MISSING_FRACTION = 0.3  # If more than 30% of pixels in the tile are missing, throw the tile out

                    # For each "SIF tile", extract the tile of the reflectance data that maps to it
                    for left_degrees in np.arange(LEFT_BOUND, RIGHT_BOUND, SIF_TILE_DEGREE_SIZE):
                        for bottom_degrees in np.arange(BOTTOM_BOUND, TOP_BOUND, SIF_TILE_DEGREE_SIZE):
                            right_edge = left_degrees + SIF_TILE_DEGREE_SIZE
                            top_edge = bottom_degrees + SIF_TILE_DEGREE_SIZE

                            # Find indices in datasets.
                            cover_bottom_idx, cover_left_idx = lat_long_to_index(bottom_degrees, left_degrees,
                                                                                 cover_dataset.bounds.top,
                                                                                 cover_dataset.bounds.left,
                                                                                 target_res)
                            reflectance_bottom_idx, reflectance_left_idx = lat_long_to_index(bottom_degrees, left_degrees,
                                                                                          reflectance_dataset.bounds.top,
                                                                                          reflectance_dataset.bounds.left,
                                                                                          target_res)
                            cover_top_idx = cover_bottom_idx - TARGET_TILE_SIZE  #tile_height_pixels
                            reflectance_top_idx = reflectance_bottom_idx - TARGET_TILE_SIZE  #tile_height_pixels
                            cover_right_idx = cover_left_idx + TARGET_TILE_SIZE
                            reflectance_right_idx = reflectance_left_idx + TARGET_TILE_SIZE
                            print("Cover shape", reprojected_covers.shape)
                            print("Cover idx: top", cover_top_idx, "bottom", cover_bottom_idx, "left", cover_left_idx,
                                  "right", cover_right_idx)
                            print("Reflectance shape", reprojected_reflectances.shape)
                            print("Reflectance idx: top", reflectance_top_idx, "bottom", reflectance_bottom_idx,
                                  "left", reflectance_left_idx, "right", reflectance_right_idx)

                            # If the selected region (box) goes outside the range of the cover or reflectance dataset, ignore
                            if cover_top_idx < 0 or cover_left_idx < 0 or reflectance_top_idx < 0 or reflectance_left_idx < 0:
                                print("Index was negative!")
                                continue
                            if (cover_bottom_idx >= reprojected_covers.shape[0] or
                                    cover_right_idx >= reprojected_covers.shape[1] or
                                    reflectance_bottom_idx >= reprojected_reflectances.shape[1] or
                                    reflectance_right_idx >= reprojected_reflectances.shape[2]):
                                print("Index went beyond edge of array!")
                                continue

                            # Extract the cover and reflectance tiles (covering the same region as the SIF tile)
                            cover_tile = reprojected_covers[cover_top_idx:cover_bottom_idx, cover_left_idx:cover_right_idx]
                            reflectance_tile = reprojected_reflectances[:, reflectance_top_idx:reflectance_bottom_idx,
                                                                 reflectance_left_idx:reflectance_right_idx]
                            print("Cover tile shape", cover_tile.shape)
                            cover_fraction_nonzero = np.count_nonzero(cover_tile) / (cover_tile.shape[0] * cover_tile.shape[1])
                            print("Fraction of nonzeros in cover tile:", cover_fraction_nonzero)
                            print("Reflectance tile shape", reflectance_tile.shape)
                            reflectance_fraction_nonzero = np.count_nonzero(reflectance_tile) / (reflectance_tile.shape[0] * reflectance_tile.shape[1] * reflectance_tile.shape[2])
                            print("Fraction of nonzeros in reflectance tile:", reflectance_fraction_nonzero)
                            assert(cover_tile.shape[0:2] == reflectance_tile.shape[1:3])

                            # If too much data is missing, throw this tile out
                            if cover_fraction_nonzero < 1 - MAX_MISSING_FRACTION:
                                continue
                            if reflectance_fraction_nonzero < 1 - MAX_MISSING_FRACTION:
                                continue

                            # Create cover bands (binary masks)
                            COVERS_TO_MASK = [1, 5, 176, 141]
                            masks = []
                            for i, cover_type in enumerate(COVERS_TO_MASK):
                                crop_mask = np.zeros_like(cover_tile)
                                crop_mask[cover_tile == cover_type] = 1.
                                masks.append(crop_mask)
                            
                            # Also create a binary mask, which is 1 for pixels where reflectance
                            # data (for all bands) is missing (due to cloud cover)
                            reflectance_tile_sum_bands = reflectance_tile.sum(axis=0)
                            missing_reflectance_mask = np.zeros_like(reflectance_tile_sum_bands)
                            missing_reflectance_mask[reflectance_tile_sum_bands == 0] = 1.
                            #print("Missing reflectance mask", missing_reflectance_mask.shape)
                            masks.append(missing_reflectance_mask)

                            # Stack masks on top of each other
                            masks = np.stack(masks, axis=0)

                            # Stack reflectance bands and masks on top of each other
                            reflectance_and_cover_tile = np.concatenate((reflectance_tile, masks), axis=0)
                            #print("Combined tile shape", reflectance_and_cover_tile.shape)

                            # Extract corresponding SIF value
                            center_lat = round(bottom_degrees + SIF_TILE_DEGREE_SIZE / 2, 2)
                            center_lon = round(left_degrees + SIF_TILE_DEGREE_SIZE / 2, 2)
                            total_sif = sif_array.sel(lat=center_lat, lon=center_lon, method='nearest').values
                            if np.isnan(total_sif):
                                continue
                            print("total_sif", total_sif)

                            # Write reflectance/cover pixels tile (as Numpy array) to .npy file
                            npy_filename = "datasets/generated/reflectance_lat_" + str(center_lat) + "_lon_" + str(center_lon) + ".npy"
                            np.save(npy_filename, reflectance_and_cover_tile)
                            #print("date", date_range.date[0].isoformat())
                            dataset_rows.append([center_lon, center_lat, date_range.date[0].isoformat(), npy_filename, total_sif])

            except Exception as error:
                print("Reading reflectance file", reflectance_file, "failed")
                print(traceback.format_exc())


with open(OUTPUT_CSV_FILE, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in dataset_rows:
        csv_writer.writerow(row) 
            # Todo: 1) Download reflectance/crop data for larger regions
