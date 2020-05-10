"""
Outputs a dataset mapping reflectance/cover tiles to SIF.
Each row in the dataset contains a filename of an .npy file. The ``tile_file'' field contains the
name of a file, which stores a tensor of shape (band x lat x long); each band is either a 
reflectance band or a mask of a specific crop cover type.
Each row also contains the latitude and longitude of that tile, as well as the total SIF of the tile.

"""
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
from sif_utils import lat_long_to_index, plot_histogram

DATE_RANGE = pd.date_range(start="2018-07-16", end="2018-07-31")
SIF_DATE_RANGE = pd.date_range(start="2018-08-01", end="2018-08-16")
START_DATE = str(DATE_RANGE.date[0])
YEAR = "2018"
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
REFLECTANCE_DIR = os.path.join(DATA_DIR, "LandsatReflectance", START_DATE + "_r2")
COVER_FILE = os.path.join(DATA_DIR, "CDL_" + YEAR + "/CDL_2018_r2.tif") #corn_belt_cdl_2018-08-01_epsg.tif")  # "CDL_2016_big.tif"
OUTPUT_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + START_DATE)  # Directory containing list of tiles
OUTPUT_IMAGES_DIR = os.path.join(DATA_DIR, "images_" + START_DATE)  # Directory containing large images
OUTPUT_TILES_DIR = os.path.join(DATA_DIR, "tiles_" + START_DATE)  # Directory containing 0.1x0.1 degree tiles
SIF_FILE = os.path.join(DATA_DIR, "TROPOMI_SIF/TROPO-SIF_01deg_biweekly_Apr18-Jan20.nc")
FLDAS_FILE = os.path.join(DATA_DIR, "FLDAS/FLDAS_NOAH01_C_GL_M.A" + YEAR + "08.001.nc.SUB.nc4")

# List of cover types to include (see https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL
# for what these numbers correspond to). I included all cover types that are >1% of the region.
COVERS_TO_MASK = [176, 1, 5, 152, 141, 142, 23, 121, 37, 24, 195, 190, 111, 36, 61, 4, 122, 131, 22, 31, 6, 42, 123, 29, 41, 28, 143, 53, 21, 52]  # [176, 152, 1, 5, 141, 142, 23, 121, 37, 190, 195, 111, 36, 24, 61, 0]
# MAX_MISSING_FRACTION = 0.5  # If more than 50% of pixels in the tile are missing, throw the tile out
SIF_TILE_DEGREE_SIZE = 0.1
FLOAT_EQUALITY_TOLERANCE = 1e-10

# True if you want to append to the output csv file, False to overwrite
APPEND = False #True
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DATASET_DIR, "reflectance_cover_to_sif_r2.csv")

if not os.path.exists(OUTPUT_DATASET_DIR):
    os.makedirs(OUTPUT_DATASET_DIR)
if not os.path.exists(OUTPUT_IMAGES_DIR):
    os.makedirs(OUTPUT_IMAGES_DIR)
if not os.path.exists(OUTPUT_TILES_DIR):
    os.makedirs(OUTPUT_TILES_DIR)



# Plot corn pixels and print the most frequent crop types (sorted by percentage)
# "covers" should be a 2-D array of pixels, with a single integer at each pixel representin the crop type (according
# to the CDL classification)
def plot_and_print_covers(covers, filename):
    print('Covers!', covers)
    mask = np.zeros_like(covers)
    mask[covers == 1] = 1.
    #plt.imshow(mask, cmap='Greens', vmin=0, vmax=1)
    #plt.savefig('datasets/CDL_2019/visualizations/' + filename)
    #plt.close()

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




# Dataset format: lon/lat, date, image file name, SIF
dataset_rows = []
if not APPEND:
    dataset_rows.append(["lon", "lat", "date", "missing_reflectance", "tile_file", "SIF"])

# For each tile, keep track of how much reflectance and cover data is present
reflectance_coverage = []
cover_coverage = []

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

    # Open up FLDAS dataset
    fldas_dataset = xr.open_dataset(FLDAS_FILE).mean(dim='time')

    # Print stats about cover dataset
    # plot_and_print_covers(cover_dataset.read(1), 'original_covers_corn_big.png')

    # Open up the SIF file
    sif_dataset = xr.open_dataset(SIF_FILE)

    # Read SIF values that fall in the appropriate date range
    sif_array = sif_dataset.sif_dc.sel(time=slice(SIF_DATE_RANGE.date[0], SIF_DATE_RANGE.date[-1]))
    print("SIF array shape", sif_array.shape)
    print("SIF array:", sif_array)

    # Check if SIF is available for any date in time range. If there is, take the mean
    # over all dates in the time period. Otherwise, ask if we should still create the
    # dataset, but without the SIF label.
    if len(sif_array['time'].values) >= 1:
        sif_array = sif_array.mean(dim='time')
    else:
        response = input("No SIF data available for any date between " + str(SIF_DATE_RANGE.date[0]) +
                         " and " + str(SIF_DATE_RANGE.date[-1]) +
                         ". Create dataset anyways without total SIF label? (y/n) ")
        if response != 'y' and response != 'Y':
            exit(1)

    # Stores a version of cover dataset, reprojected to the resolution of
    # the reflectance dataset
    reprojected_covers = None
    reprojected_fldas = None

    # If you select a large region, Google Earth Engine breaks the reflectance data
    # into multiple files; loop through all of them.
    for reflectance_file in os.listdir(REFLECTANCE_DIR):
        try:
            with rio.open(os.path.join(REFLECTANCE_DIR, reflectance_file)) as reflectance_dataset:
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

                target_res = (reflectance_dataset.res[0], reflectance_dataset.res[1])
                TARGET_TILE_SIZE = int(SIF_TILE_DEGREE_SIZE / target_res[0])
                print("target tile size", TARGET_TILE_SIZE)

                # Resample cover data into target resolution, if we haven't done so already and the
                # cover dataset is in a different resolution
                if reprojected_covers is None:
                    if abs(cover_dataset.res[0] - target_res[0]) < FLOAT_EQUALITY_TOLERANCE and abs(cover_dataset.res[1] - target_res[1]) < FLOAT_EQUALITY_TOLERANCE:
                        reprojected_covers = cover_dataset.read(1)
                        print('No need to reproject cover dataset, as it already has the same resolution as the reflectance dataset')
                    else:
                        cover_height_upscale_factor = cover_dataset.res[0] / target_res[0]
                        cover_width_upscale_factor = cover_dataset.res[1] / target_res[1]
                        print('Reprojecting cover dataset to match reflectance!')
                        print('Upscale factor: height', cover_height_upscale_factor, 'width', cover_width_upscale_factor)
                        reprojected_covers = cover_dataset.read(
                            out_shape=(
                                int(cover_dataset.height * cover_height_upscale_factor),
                                int(cover_dataset.width * cover_width_upscale_factor)
                            ),
                            resampling=Resampling.mode
                        )
                    reprojected_covers = np.squeeze(reprojected_covers)

                print('REPROJECTED COVER DATASET: shape', reprojected_covers.shape, 'dtype:', reprojected_covers.dtype)

                # Resample reflectance data into target resolution (don't need this now since we're
                # projecting everything to the reflectance dataset's resolution)
                # reflectance_height_upscale_factor = reflectance_dataset.res[0] / target_res[0]
                # reflectance_width_upscale_factor = reflectance_dataset.res[1] / target_res[1]
                # reprojected_reflectances = reflectance_dataset.read(
                #    out_shape=(
                #        int(reflectance_dataset.height * reflectance_height_upscale_factor),
                #        int(reflectance_dataset.width * reflectance_width_upscale_factor)
                #    ),
                #    resampling=Resampling.bilinear
                # 
                # )

                # Read reflectance dataset into numpy array: CxHxW
                reprojected_reflectances = reflectance_dataset.read()
                print('REPROJECTED REFLECTANCE DATASET: shape', reprojected_reflectances.shape, 'Dtype:', reprojected_reflectances.dtype)
                #plot_histogram(reprojected_reflectances[1].flatten(), 'blue_reflectance_values.png')
                #plot_histogram(reprojected_reflectances[2].flatten(), 'green_reflectance_values.png')
                #plot_histogram(reprojected_reflectances[3].flatten(), 'red_reflectance_values.png')

                # Plot distribution of specific crop
                # plot_and_print_covers(reprojected_covers, filename="reprojected_cover_corn_big.png")

                # Read reflectance data into numpy array
                # reflectance_numpy = reflectance_dataset.read()
                # print('Reflectance numpy array shape', reflectance_numpy.shape)
                # print('Lat/Long of Upper Left Corner', reflectance_dataset.xy(0, 0))
                # print('Lat/Long of index (1000, 1000)', reflectance_dataset.xy(1000, 1000))

                # Just for testing
                # point = (44.9, -88.9)
                # left_idx, top_idx = reprojected_reflectances.index(point[1], point[0])  # reflectance_dataset.bounds.left, reflectance_dataset.bounds.top)
                # print('===================================================')
                # print('TEST CASE: Point lat=', point[0], 'long=', point[1])
                # print('Using index method', left_idx, top_idx)

                # reflectance_height_idx, reflectance_width_idx = lat_long_to_index(point[0], point[1],
                #                                                                  reflectance_dataset.bounds.top,
                #                                                                  reflectance_dataset.bounds.left,
                #                                                                  target_res)
                # print("indices in reflectance:", reflectance_height_idx, reflectance_width_idx)
                # cover_height_idx, cover_width_idx = lat_long_to_index(point[0], point[1], cover_dataset.bounds.top,
                #                                                      cover_dataset.bounds.left, target_res)
                # print("indices in cover:", cover_height_idx, cover_width_idx)
                # print('===================================================')

                # Extract bounds of the intersection of reflectance/cover coverage
                combined_left_bound = max(reflectance_dataset.bounds.left, cover_dataset.bounds.left)
                combined_right_bound = min(reflectance_dataset.bounds.right, cover_dataset.bounds.right)
                combined_bottom_bound = max(reflectance_dataset.bounds.bottom, cover_dataset.bounds.bottom)
                combined_top_bound = min(reflectance_dataset.bounds.top, cover_dataset.bounds.top)

                # Convert bounds to indices in the cover and reflectance datasets. Note that (0,0)
                # is the upper-left corner!
                #cover_top_idx, cover_left_idx = lat_long_to_index(combined_top_bound,
                #                                                  combined_left_bound,
                #                                                  cover_dataset.bounds.top,
                #                                                  cover_dataset.bounds.left,
                #                                                  target_res)
                #reflectance_top_idx, reflectance_left_idx = lat_long_to_index(combined_top_bound,
                #                                                              combined_left_bound,
                #                                                              reflectance_dataset.bounds.top,
                #                                                              reflectance_dataset.bounds.left,
                #                                                              target_res)
                #height_pixels = int((combined_top_bound - combined_bottom_bound) / target_res[0])
                #width_pixels = int((combined_right_bound - combined_left_bound) / target_res[1])
                #cover_right_idx = cover_left_idx + width_pixels
                #reflectance_right_idx = reflectance_left_idx + width_pixels
                #cover_bottom_idx = cover_top_idx + height_pixels
                #reflectance_bottom_idx = reflectance_top_idx + height_pixels
                #print('Cover: top', cover_top_idx, 'bottom', cover_bottom_idx, 'left', cover_left_idx, 'right', cover_right_idx)
                #print('Reflectance: top', reflectance_top_idx, 'bottom', reflectance_bottom_idx, 'left', reflectance_left_idx, 'right', reflectance_right_idx)
                #assert(reflectance_top_idx >= 0)
                #assert(cover_top_idx >= 0)
                #assert(cover_right_idx <= reprojected_covers.shape[1])  # Recall right_idx is exclusive
                #assert(reflectance_right_idx <= reprojected_reflectances.shape[2])

                # Round boundaries to the nearest 0.1 degree
                LEFT_BOUND = math.ceil(combined_left_bound * 10) / 10  # -100.2
                RIGHT_BOUND = math.floor(combined_right_bound * 10) / 10  # -81.6
                BOTTOM_BOUND = math.ceil(combined_bottom_bound * 10) / 10  # 38.2
                TOP_BOUND = math.floor(combined_top_bound * 10) / 10  # 46.6

                # For each "SIF tile", extract the tile of the reflectance data that maps to it
                for left_degrees in np.arange(LEFT_BOUND, RIGHT_BOUND, SIF_TILE_DEGREE_SIZE):
                    for top_degrees in np.arange(TOP_BOUND, BOTTOM_BOUND, -1*SIF_TILE_DEGREE_SIZE):
                        
                        bottom_degrees = top_degrees - (TARGET_TILE_SIZE * target_res[0])
                        right_degrees = left_degrees + (TARGET_TILE_SIZE * target_res[1])
                        print('-----------------------------------------------------------')
                        print('Extracting tile: longitude', left_degrees, 'to', right_degrees, 'latitude', bottom_degrees, 'to', top_degrees)

                        # Find indices of tile in reflectance and cover datasets
                        reflectance_top_idx, reflectance_left_idx = lat_long_to_index(top_degrees, left_degrees, reflectance_dataset.bounds.top, reflectance_dataset.bounds.left, target_res)
                        reflectance_bottom_idx = reflectance_top_idx + TARGET_TILE_SIZE
                        reflectance_right_idx = reflectance_left_idx + TARGET_TILE_SIZE
                        cover_top_idx, cover_left_idx = lat_long_to_index(top_degrees, left_degrees, cover_dataset.bounds.top, cover_dataset.bounds.left, target_res)
                        cover_bottom_idx = cover_top_idx + TARGET_TILE_SIZE
                        cover_right_idx = cover_left_idx + TARGET_TILE_SIZE

                        print("Reflectance dataset idx: top", reflectance_top_idx, "bottom", reflectance_bottom_idx,
                              "left", reflectance_left_idx, "right", reflectance_right_idx)
                        print("Cover dataset idx: top", cover_top_idx, "bottom", cover_bottom_idx,
                              "left", cover_left_idx, "right", cover_right_idx)

                        # If the selected region (box) goes outside the range of the cover or reflectance dataset, ignore
                        if reflectance_top_idx < 0 or reflectance_left_idx < 0:
                            print("Reflectance index was negative!")
                            continue
                        if (reflectance_bottom_idx >= reprojected_reflectances.shape[1] or reflectance_right_idx >= reprojected_reflectances.shape[2]):
                            print("Reflectance index went beyond edge of array!")
                            continue
                        if cover_top_idx < 0 or cover_left_idx < 0:
                            print("Cover index was negative!")
                            continue
                        if (cover_bottom_idx >= reprojected_covers.shape[0] or cover_right_idx >= reprojected_covers.shape[1]):
                            print("Cover index went beyond edge of array!")
                            continue

                        # Resample FLDAS data for this tile into target resolution (if we haven't already)
                        new_lat = np.linspace(top_degrees, bottom_degrees, TARGET_TILE_SIZE)  # bottom_bound, combined_top_bound, height_pixels)
                        new_lon = np.linspace(left_degrees, right_degrees, TARGET_TILE_SIZE)  # combined_left_bound, combined_right_bound, width_pixels)
                        reprojected_fldas_dataset = fldas_dataset.interp(X=new_lon, Y=new_lat)
                        fldas_layers = []
                        print('FLDAS data vars', reprojected_fldas_dataset.data_vars)
                        for data_var in reprojected_fldas_dataset.data_vars:
                            #print('var', data_var)
                            fldas_layers.append(reprojected_fldas_dataset[data_var].data)
                        fldas_tile = np.stack(fldas_layers)
                        if np.isnan(fldas_tile).any():
                            print('ATTENTION: FLDAS tile had NaNs!!!')
                            continue

                        # Extract relevant areas from cover and reflectance datasets
                        cover_tile = reprojected_covers[cover_top_idx:cover_bottom_idx,
                                                        cover_left_idx:cover_right_idx]
                        reflectance_tile = reprojected_reflectances[:, reflectance_top_idx:reflectance_bottom_idx,
                                                                    reflectance_left_idx:reflectance_right_idx]
                        print('Cover tile shape', cover_tile.shape, 'dtype', cover_tile.dtype)
                        print('Reflectance tile shape (should be the same!)', reflectance_tile.shape, 'dtype', reflectance_tile.dtype)
                        print('FLDAS tile shape (should be the same!)', fldas_tile.shape, 'dtype', fldas_tile.dtype)

                        # Create cover bands (binary masks)
                        masks = []
                        for i, cover_type in enumerate(COVERS_TO_MASK):
                            crop_mask = np.zeros_like(cover_tile, dtype=bool)
                            crop_mask[cover_tile == cover_type] = 1.
                            masks.append(crop_mask)

                        # Also create a binary mask, which is 1 for pixels where reflectance
                        # data (for all bands) is missing (due to cloud cover)
                        reflectance_sum_bands = reflectance_tile.sum(axis=0)
                        #print("Reflectance sum bands dtype", reflectance_sum_bands.dtype)
                        missing_reflectance_mask = np.zeros_like(reflectance_sum_bands, dtype=bool)
                        missing_reflectance_mask[reflectance_sum_bands == 0] = 1.
                        #print("Missing reflectance mask dtype", missing_reflectance_mask.dtype)
                        masks.append(missing_reflectance_mask)

                        # Stack masks on top of each other
                        masks = np.stack(masks, axis=0)

                        # Stack reflectance bands and masks on top of each other
                        combined_tile = np.concatenate((reflectance_tile, fldas_tile, masks), axis=0)
                        print("Combined tile shape", combined_tile.shape, 'dtype', combined_tile.dtype)


                        # Extract the cover and reflectance tiles (covering the same region as the SIF tile)
                        # reflectance_and_cover_tile = combined_area[:, top_idx:bottom_idx, left_idx:right_idx]
                        reflectance_fraction_missing = np.sum(combined_tile[-1, :, :].flatten()) / \
                                                       (combined_tile.shape[1] *
                                                        combined_tile.shape[2])
                        print("Fraction of reflectance pixels missing:", reflectance_fraction_missing)
                        reflectance_coverage.append(1 - reflectance_fraction_missing)
                        #cover_fraction_missing = np.sum(reflectance_and_cover_tile[-2, :, :].flatten()) / \
                        #                                (reflectance_and_cover_tile.shape[1] *
                        #                                 reflectance_and_cover_tile.shape[2])
                        #print("Fraction of cover pixels missing:", cover_fraction_missing)
                        #cover_coverage.append(1 - cover_fraction_missing)
                        
                        # If too much data is missing, throw this tile out
                        #if reflectance_fraction_missing > MAX_MISSING_FRACTION:
                        #    continue
                        #if cover_fraction_missing > MAX_MISSING_FRACTION:
                        #    continue

                        # Extract corresponding SIF value
                        center_lat = round(top_degrees - SIF_TILE_DEGREE_SIZE / 2, 2)
                        center_lon = round(left_degrees + SIF_TILE_DEGREE_SIZE / 2, 2)
                        if sif_array is not None:
                            total_sif = sif_array.sel(lat=center_lat, lon=center_lon, method='nearest').values
                            if np.isnan(total_sif):
                                continue
                        else:
                            total_sif = float.nan
                        # print("total_sif", total_sif)

                        # Write reflectance/cover pixels tile (as Numpy array) to .npy file
                        npy_filename = os.path.join(OUTPUT_TILES_DIR, "reflectance_lat_" + str(
                            center_lat) + "_lon_" + str(center_lon) + ".npy")
                        np.save(npy_filename, combined_tile)
                        dataset_rows.append([center_lon, center_lat, START_DATE, reflectance_fraction_missing, npy_filename, total_sif])
 
        except Exception as error:
            print("Reading reflectance file", reflectance_file, "failed")
            print(traceback.format_exc())

if APPEND:
    mode = "a+"
else:
    mode = "w"
with open(OUTPUT_CSV_FILE, mode) as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in dataset_rows:
        csv_writer.writerow(row)

plot_histogram(np.array(reflectance_coverage), "reflectance_coverage_" + START_DATE + ".png")
# plot_histogram(np.array(cover_coverage), "cover_coverage_" + START_DATE + ".png")
