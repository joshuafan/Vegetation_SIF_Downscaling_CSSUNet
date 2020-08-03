"""
Outputs a dataset mapping large tiles to TROPOMI SIF.
Each row in the dataset contains a filename of an .npy file. The ``tile_file'' field contains the
name of a file, which stores a tensor of shape (band x lat x long); each band is either a 
reflectance band, a FLDAS band, a mask of a specific crop cover type.
Each row also contains the latitude and longitude of that tile, as well as the total SIF of the tile.
"""
import csv
import math
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.crs
import xarray as xr
from rasterio.plot import show
from rasterio.warp import Resampling, calculate_default_transform, reproject

from sif_utils import lat_long_to_index, plot_histogram


# Plot corn pixels and print the most frequent crop types (sorted by percentage)
# "covers" should be a 2-D array of pixels, with a single integer at each pixel representin the crop type (according
# to the CDL classification)
def plot_and_print_covers(covers, filename):
    print('Covers!', covers)
    mask = np.zeros_like(covers)
    mask[covers == 1] = 1.
    plt.imshow(mask, cmap='Greens', vmin=0, vmax=1)
    plt.savefig('exploratory_plots/' + filename)
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


# Root directory of datasets
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"

COLUMNS = ['lon', 'lat', 'date', 'tile_file', 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                        'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                        'grassland_pasture', 'corn', 'soybean', 'shrubland',
                        'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                        'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                        'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                        'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                        'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                        'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                        'lentils', 'missing_reflectance', 'SIF', 'tropomi_cloud_fraction', 'num_soundings']

# List of cover types to include (see https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL
# for what these numbers correspond to). I included all cover types that are >1% of the region.
COVERS_TO_MASK = [176, 1, 5, 152, 141, 142, 23, 121, 37, 24, 195, 190, 111, 36, 61, 4, 122, 131, 22, 31, 6, 42, 123, 29, 41, 28, 143, 53, 21, 52]  # [176, 152, 1, 5, 141, 142, 23, 121, 37, 190, 195, 111, 36, 24, 61, 0]

SIF_TILE_DEGREE_SIZE = 0.1  # Size of output tiles, in degrees
FLOAT_EQUALITY_TOLERANCE = 1e-10
REFLECTANCE_BANDS = list(range(0, 9))
MISSING_REFLECTANCE_IDX = -1

# True if you want to append to the output csv file, False to overwrite
APPEND = False

# Date ranges of Landsat data
# DATE_RANGES = [pd.date_range(start="2018-04-29", end="2018-05-12"),
#                pd.date_range(start="2018-05-13", end="2018-05-26"),
#                pd.date_range(start="2018-05-27", end="2018-06-09"),
#                pd.date_range(start="2018-06-10", end="2018-06-23"),
#                pd.date_range(start="2018-06-24", end="2018-07-07"),
#                pd.date_range(start="2018-07-08", end="2018-07-21"),
#                pd.date_range(start="2018-07-22", end="2018-08-04"),
#                pd.date_range(start="2018-08-05", end="2018-08-18"),
#                pd.date_range(start="2018-08-19", end="2018-09-01"),
#                pd.date_range(start="2018-09-02", end="2018-09-15"),
#                pd.date_range(start="2018-09-16", end="2018-09-29")]

DATE_RANGES = [pd.date_range(start="2016-08-01", end="2016-08-16")]


for DATE_RANGE in DATE_RANGES:
    start_date = DATE_RANGE.date[0]
    start_date_string = str(start_date)
    middle_date = DATE_RANGE.date[int(len(DATE_RANGE.date) / 2)]
    year = str(middle_date.year)
    month = str(middle_date.month)
    month = month.rjust(2, '0')  # Pad month to 2-digits (e.g. 5 becomes 05)

    # Directory containing Landsat data (may contain multiple .tif files)
    REFLECTANCE_DIR = os.path.join(DATA_DIR, "LandsatReflectance", start_date_string)

    # File containing FLDAS data
    FLDAS_FILE = os.path.join(DATA_DIR, "FLDAS/FLDAS_NOAH01_C_GL_M.A" + year + month + ".001.nc.SUB.nc4")
    print("FLDAS file", FLDAS_FILE)
    FLDAS_VARS = ["Rainf_f_tavg", "SWdown_f_tavg", "Tair_f_tavg"]

    # Directory containing CDL (crop type) data
    COVER_DIR = os.path.join(DATA_DIR, "CDL_" + year)

    # File containing SIF data
    SIF_FILE = os.path.join(DATA_DIR, "TROPOMI_SIF/TROPO-SIF_01deg_biweekly_Apr18-Jan20.nc")

    # Output directories
    OUTPUT_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + start_date_string)  # Directory containing list of tiles
    OUTPUT_TILES_DIR = os.path.join(DATA_DIR, "tiles_" + start_date_string)  # Directory containing 0.1x0.1 degree tiles
    OUTPUT_CSV_FILE = os.path.join(OUTPUT_DATASET_DIR, "reflectance_cover_to_sif.csv")  # Output csv file referencing all tiles
    if not os.path.exists(OUTPUT_DATASET_DIR):
        os.makedirs(OUTPUT_DATASET_DIR)
    if not os.path.exists(OUTPUT_TILES_DIR):
        os.makedirs(OUTPUT_TILES_DIR)


    # Dataset format: lon/lat, date, image file name, SIF
    dataset_rows = []
    if not APPEND:
        dataset_rows.append(COLUMNS)  #(["lon", "lat", "date", "missing_reflectance", "tile_file", "SIF", "cloud_fraction", "num_soundings"])

    # For each tile, keep track of how much reflectance and cover data is present
    reflectance_coverage = []
    cover_coverage = []

    # Open up the SIF file
    sif_dataset = xr.open_dataset(SIF_FILE)

    # Read SIF values that fall in the appropriate date range
    tropomi_sifs = sif_dataset.sif_dc.sel(time=slice(DATE_RANGE.date[0], DATE_RANGE.date[-1]))
    tropomi_cloud_fraction = sif_dataset.cloud_fraction.sel(time=slice(DATE_RANGE.date[0], DATE_RANGE.date[-1]))
    tropomi_n = sif_dataset.n.sel(time=slice(DATE_RANGE.date[0], DATE_RANGE.date[-1]))

    print("SIF array:", tropomi_sifs)
    exit(0)
    # print("FLDAS file", FLDAS_FILE)

    # Check if SIF is available for any date in time range. If there is, take the mean
    # over all dates in the time period. Otherwise, ask if we should still create the
    # dataset, but without the SIF label.
    if len(tropomi_sifs['time'].values) >= 1:
        tropomi_sifs = tropomi_sifs.mean(dim='time')
        tropomi_cloud_fraction = tropomi_cloud_fraction.mean(dim='time')
        tropomi_n = tropomi_n.mean(dim='time')
    else:
        response = input("No SIF data available for any date between " + str(DATE_RANGE.date[0]) +
                            " and " + str(DATE_RANGE.date[-1]) +
                            ". Create dataset anyways without total SIF label? (y/n) ")
        if response != 'y' and response != 'Y':
            exit(1)
        tropomi_sifs = None
        tropomi_cloud_fraction = None
        tropomi_n = None

    # Open up FLDAS dataset
    fldas_dataset = xr.open_dataset(FLDAS_FILE).mean(dim='time')
    # print("FLDAS dataset", fldas_dataset)

    # Open crop cover files
    for cover_file in os.listdir(COVER_DIR):
        with rio.open(os.path.join(COVER_DIR, cover_file)) as cover_dataset:
            # Print stats about cover dataset
            print('===================================================')
            print('COVER DATASET', cover_file)
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

            # Print stats about cover dataset
            #plot_and_print_covers(cover_dataset.read(1), 'original_covers_corn_big_r2.png')
            #exit(1)

            # Stores a version of cover dataset, reprojected to the resolution of
            # the reflectance dataset
            reprojected_covers = None
            reprojected_fldas = None

            # If you select a large region, Google Earth Engine breaks the reflectance data
            # into multiple files; loop through all of them.
            for reflectance_file in sorted(os.listdir(REFLECTANCE_DIR)):
                try:
                    with rio.open(os.path.join(REFLECTANCE_DIR, reflectance_file)) as reflectance_dataset:
                        # Print stats about reflectance file
                        print('===================================================')
                        print('REFLECTANCE DATASET', reflectance_file)
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
                        #print("target tile size", TARGET_TILE_SIZE)

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
                        # Note that index (0, 0) is at the upper left corner!
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
                        #                                                       cover_dataset.bounds.left, target_res)
                        # print("indices in cover:", cover_height_idx, cover_width_idx)
                        # print('===================================================')

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

                        # Extract bounds of the intersection of reflectance/cover coverage
                        combined_left_bound = max(reflectance_dataset.bounds.left, cover_dataset.bounds.left)
                        combined_right_bound = min(reflectance_dataset.bounds.right, cover_dataset.bounds.right)
                        combined_bottom_bound = max(reflectance_dataset.bounds.bottom, cover_dataset.bounds.bottom)
                        combined_top_bound = min(reflectance_dataset.bounds.top, cover_dataset.bounds.top)
                        print("Bounds: lon:", combined_left_bound, "to", combined_right_bound, "lat:", combined_bottom_bound, "to", combined_top_bound)
        
                        # Round boundaries to the nearest 0.1 degree
                        LEFT_BOUND = math.ceil(combined_left_bound * 10) / 10  # -100.2
                        RIGHT_BOUND = math.floor(combined_right_bound * 10) / 10  # -81.6
                        BOTTOM_BOUND = math.ceil(combined_bottom_bound * 10) / 10  # 38.2
                        TOP_BOUND = math.floor(combined_top_bound * 10) / 10  # 46.6
                        num_tiles_lon = round((RIGHT_BOUND - LEFT_BOUND) / SIF_TILE_DEGREE_SIZE)
                        num_tiles_lat = round((TOP_BOUND - BOTTOM_BOUND) / SIF_TILE_DEGREE_SIZE)
                        if num_tiles_lon <= 0 or num_tiles_lat <= 0:
                            print("No overlap between reflectance and cover dataset!")
                            continue

                        # Iterate through all 0.1 degree intervals
                        tile_lefts = np.linspace(LEFT_BOUND, RIGHT_BOUND, num_tiles_lon, endpoint=False)
                        tile_tops = np.linspace(TOP_BOUND, BOTTOM_BOUND, num_tiles_lat, endpoint=False)
                        # print('Tile lefts', tile_lefts)
                        # print('Tile tops', tile_tops)

                        # For each "SIF tile", extract the tile of the reflectance data that maps to it
                        for left_degrees in tile_lefts:
                            for top_degrees in tile_tops:
                                bottom_degrees = top_degrees - (TARGET_TILE_SIZE * target_res[0])
                                right_degrees = left_degrees + (TARGET_TILE_SIZE * target_res[1])
                                # print('-----------------------------------------------------------')
                                print('Extracting tile: longitude', left_degrees, 'to', right_degrees, 'latitude', bottom_degrees, 'to', top_degrees)

                                # Find indices of tile in reflectance and cover datasets
                                reflectance_top_idx, reflectance_left_idx = lat_long_to_index(top_degrees, left_degrees, reflectance_dataset.bounds.top, reflectance_dataset.bounds.left, target_res)
                                reflectance_bottom_idx = reflectance_top_idx + TARGET_TILE_SIZE
                                reflectance_right_idx = reflectance_left_idx + TARGET_TILE_SIZE
                                cover_top_idx, cover_left_idx = lat_long_to_index(top_degrees, left_degrees, cover_dataset.bounds.top, cover_dataset.bounds.left, target_res)
                                cover_bottom_idx = cover_top_idx + TARGET_TILE_SIZE
                                cover_right_idx = cover_left_idx + TARGET_TILE_SIZE

                                #print("Reflectance dataset idx: top", reflectance_top_idx, "bottom", reflectance_bottom_idx,
                                #      "left", reflectance_left_idx, "right", reflectance_right_idx)
                                # print("Cover dataset idx: top", cover_top_idx, "bottom", cover_bottom_idx,
                                #     "left", cover_left_idx, "right", cover_right_idx)

                                # If the selected region (box) goes outside the range of the cover or reflectance dataset, that's a bug!
                                if reflectance_top_idx < 0 or reflectance_left_idx < 0:
                                    print("Reflectance index was negative!")
                                    exit(1)
                                if (reflectance_bottom_idx >= reprojected_reflectances.shape[1] or reflectance_right_idx >= reprojected_reflectances.shape[2]):
                                    print("Reflectance index went beyond edge of array!")
                                    exit(1)
                                if cover_top_idx < 0 or cover_left_idx < 0:
                                    print("Cover index was negative!")
                                    exit(1)
                                if (cover_bottom_idx >= reprojected_covers.shape[0] or cover_right_idx >= reprojected_covers.shape[1]):
                                    print("Cover index went beyond edge of array!")
                                    exit(1)

                                # Resample FLDAS data for this tile into target resolution (if we haven't already)
                                new_lat = np.linspace(top_degrees, bottom_degrees, TARGET_TILE_SIZE, endpoint=False)  # bottom_bound, combined_top_bound, height_pixels)
                                new_lon = np.linspace(left_degrees, right_degrees, TARGET_TILE_SIZE, endpoint=False)  # combined_left_bound, combined_right_bound, width_pixels)
                                reprojected_fldas_dataset = fldas_dataset.interp(X=new_lon, Y=new_lat)
                                fldas_layers = []
                                #print('FLDAS data vars', reprojected_fldas_dataset.data_vars)
                                for data_var in reprojected_fldas_dataset.data_vars:
                                    assert(data_var in FLDAS_VARS)
                                    fldas_layers.append(reprojected_fldas_dataset[data_var].data)
                                fldas_tile = np.stack(fldas_layers)
                                if np.isnan(fldas_tile).any():
                                    print('ATTENTION: FLDAS tile had NaNs!!!')
                                    continue

                                # Extract the areas from cover and reflectance datasets that map to this SIF value.
                                # Again, index (0, 0) is in the upper-left corner. For "reprojected_covers",
                                # the first axis is latitude (higher indices = lower latitudes / south), and the
                                # second axis is longitude (higher indices = higher longitudes / east). For 
                                # "reprojected_reflectances", it's similar, but there is a "channel" axis first.
                                cover_tile = reprojected_covers[cover_top_idx:cover_bottom_idx,
                                                                cover_left_idx:cover_right_idx]
                                reflectance_tile = reprojected_reflectances[:, reflectance_top_idx:reflectance_bottom_idx,
                                                                            reflectance_left_idx:reflectance_right_idx]
                                #print('Cover tile shape', cover_tile.shape, 'dtype', cover_tile.dtype)
                                #print('Reflectance tile shape (should be the same!)', reflectance_tile.shape, 'dtype', reflectance_tile.dtype)
                                #print('FLDAS tile shape (should be the same!)', fldas_tile.shape, 'dtype', fldas_tile.dtype)

                                # Create cover bands (binary masks)
                                masks = []
                                for i, cover_type in enumerate(COVERS_TO_MASK):
                                    crop_mask = np.zeros_like(cover_tile, dtype=bool)
                                    crop_mask[cover_tile == cover_type] = 1.
                                    masks.append(crop_mask)

                                # Also create a binary mask, which is 1 for pixels where reflectance
                                # data (for all bands) is missing (due to cloud cover)
                                reflectance_sum_bands = reflectance_tile.sum(axis=0)
                                missing_reflectance_mask = np.zeros_like(reflectance_sum_bands, dtype=bool)
                                missing_reflectance_mask[reflectance_sum_bands == 0] = 1.
                                masks.append(missing_reflectance_mask)

                                # Stack masks on top of each other
                                masks = np.stack(masks, axis=0)

                                # Stack reflectance bands and masks on top of each other
                                combined_tile = np.concatenate((reflectance_tile, fldas_tile, masks), axis=0)
                                #print("Combined tile shape", combined_tile.shape, 'dtype', combined_tile.dtype)

                                # reflectance_and_cover_tile = combined_area[:, top_idx:bottom_idx, left_idx:right_idx]
                                reflectance_fraction_missing = np.sum(combined_tile[MISSING_REFLECTANCE_IDX, :, :].flatten()) / \
                                                            (combined_tile.shape[1] * combined_tile.shape[2])
                                # print("Fraction of reflectance pixels missing:", reflectance_fraction_missing)
                                reflectance_coverage.append(1 - reflectance_fraction_missing)

                                # Extract corresponding SIF value
                                center_lat = round(top_degrees - SIF_TILE_DEGREE_SIZE / 2, 2)
                                center_lon = round(left_degrees + SIF_TILE_DEGREE_SIZE / 2, 2)
                                if tropomi_sifs is not None:
                                    total_sif = tropomi_sifs.sel(lat=center_lat, lon=center_lon, method='nearest').values.item()
                                    cloud_fraction = tropomi_cloud_fraction.sel(lat=center_lat, lon=center_lon, method='nearest').values.item()
                                    num_soundings = tropomi_n.sel(lat=center_lat, lon=center_lon, method='nearest').values.item()
                                    if np.isnan(total_sif):  # If there's no SIF value, ignore this tile
                                        continue
                                else:
                                    total_sif = float("nan")
                                    cloud_fraction = float("nan")
                                    num_soundings = float("nan")

                                # Compute averages of each band (over non-cloudy pixels)
                                # Reshape tile into a list of pixels (pixels x channels)
                                pixels = np.moveaxis(combined_tile, 0, -1)
                                pixels = pixels.reshape((-1, pixels.shape[2]))

                                # Compute averages of each feature (band) over all pixels
                                tile_averages = np.mean(pixels, axis=0)
                                # print('=============================')
                                # print('averages all', tile_averages)

                                # NOTE: Exclude missing (cloudy) pixels for reflectance band averages
                                pixels_with_data = pixels[pixels[:, MISSING_REFLECTANCE_IDX] == 0]

                                # Remove tiles where no pixels have data (it's completely covered by clouds)
                                if pixels_with_data.shape[0] == 0:
                                    continue

                                # Compute average of the reflectance, over the non-cloudy pixels
                                reflectance_averages = np.mean(pixels_with_data[:, REFLECTANCE_BANDS], axis=0)
                                tile_averages[REFLECTANCE_BANDS] = reflectance_averages
                                # print('averages (reflectance computed over non-cloudy):', tile_averages)
                                    
                                # Remove tiles with any NaNs
                                if np.isnan(tile_averages).any():
                                    print('tile contained nan:', tile_filename)
                                    continue

                                # Write reflectance/cover pixels tile (as Numpy array) to .npy file
                                tile_filename = os.path.join(OUTPUT_TILES_DIR, "reflectance_lat_" + str(
                                    center_lat) + "_lon_" + str(center_lon) + ".npy")
                                np.save(tile_filename, combined_tile)

                                # Add metadata about the tile to csv
                                csv_row = [center_lon, center_lat, start_date_string, tile_filename] + tile_averages.tolist() + [total_sif, cloud_fraction, num_soundings]
                                dataset_rows.append(csv_row)

        
                except Exception as error:
                    print("Reading reflectance file", reflectance_file, "failed")
                    print(traceback.format_exc())
                    exit(1)

    # If APPEND is true, we're appending rows to an existing .csv.
    # Otherwise, we're overwriting.
    if APPEND:
        mode = "a+"
    else:
        mode = "w"

    # Write information about each tile to the output csv file
    with open(OUTPUT_CSV_FILE, mode) as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in dataset_rows:
            csv_writer.writerow(row)

    # Plot histogram of reflectance coverage per tile
    plot_histogram(np.array(reflectance_coverage), "reflectance_coverage_" + start_date_string + ".png")
    # plot_histogram(np.array(cover_coverage), "cover_coverage_" + START_DATE + ".png")
