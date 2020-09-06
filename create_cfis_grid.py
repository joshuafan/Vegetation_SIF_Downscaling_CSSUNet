"""
Creates tiles of CFIS points. TODO - add more documentation
"""
import csv
import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import visualization_utils
import sif_utils


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
OCO2_DIR = os.path.join(DATA_DIR, "OCO2")
MONTHS = ["Jun", "Aug"]
DATES = ["2016-06-15", "2016-08-01"]
OCO2_FILES = [os.path.join(OCO2_DIR, "oco2_20160615_20160629_3km.nc"),
              os.path.join(OCO2_DIR, "oco2_20160801_20160816_3km.nc")]

MIN_SOUNDINGS_FINE = 1
MIN_COARSE_DATAPOINTS_PER_TILE = 1
MAX_INVALID_SIF_PIXELS = 0.99  # Max fraction of invalid pixels when computing coarse-resolution SIF
MIN_CDL_COVERAGE = 0.5  # Exclude areas where there is no CDL coverage (e.g. Canada)
CDL_INDICES = list(range(12, 42))
# MIN_SIF = 0.2
FRACTION_VAL = 0.2
FRACTION_TEST = 0.2

# Scaling factors to correct for differing wavelengths
OCO2_SCALING_FACTOR = 1.69 / 1.52

# # Columns for U-Net tile dataset (mapping tile to per-pixel SIF and coarse SIF)
# CFIS_TILE_METADATA_COLUMNS = ['lon', 'lat', 'date', 'tile_file', 'sif_fine_file', 'sif_coarse_file', 
#                               'fine_soundings', 'coarse_soundings']

# # CFIS U-Net tile datasets
# CFIS_TILE_METADATA_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_tile_metadata_train.csv')
# CFIS_TILE_METADATA_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_tile_metadata_val.csv')
# CFIS_TILE_METADATA_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_tile_metadata_test.csv')
# cfis_tile_metadata_train = []
# cfis_tile_metadata_val = []
# cfis_tile_metadata_test = []

# OCO-2 tile datasets
OCO2_METADATA_TRAIN_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_train.csv')
OCO2_METADATA_VAL_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_val.csv')
OCO2_METADATA_TEST_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_test.csv')
oco2_metadata_train = []
oco2_metadata_val = []
oco2_metadata_test = []

# Columns for "band average" datasets, at both fine/coarse resolution
BAND_AVERAGE_COLUMNS = ['lon', 'lat', 'date', 'tile_file', 'ref_1', 'ref_2', 'ref_3', 'ref_4',
                        'ref_5', 'ref_6', 'ref_7', 'ref_10', 'ref_11', 'Rainf_f_tavg',
                        'SWdown_f_tavg', 'Tair_f_tavg', 'grassland_pasture', 'corn',
                        'soybean', 'shrubland', 'deciduous_forest', 'evergreen_forest',
                        'spring_wheat', 'developed_open_space',
                        'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                        'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                        'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                        'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                        'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                        'lentils', 'missing_reflectance', 'SIF', 'num_soundings']
FINE_CFIS_AVERAGE_COLUMNS = BAND_AVERAGE_COLUMNS + ['coarse_sif']
COARSE_CFIS_AVERAGE_COLUMNS = BAND_AVERAGE_COLUMNS + ['num_valid_pixels', 'fine_sif_file', 'fine_soundings_file']

# CFIS coarse/fine averages
FINE_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_train.csv')
FINE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_val.csv')
FINE_AVERAGES_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_test.csv')
fine_averages_train = []
fine_averages_val = []
fine_averages_test = []

COARSE_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_train.csv')
COARSE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_val.csv')
COARSE_AVERAGES_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_test.csv')
coarse_averages_train = []
coarse_averages_val = []
coarse_averages_test = []

# Columns to compute statistics for
STATISTICS_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                      'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
                      'grassland_pasture', 'corn', 'soybean', 'shrubland',
                      'deciduous_forest', 'evergreen_forest', 'spring_wheat',
                      'developed_open_space', 'other_hay_non_alfalfa', 'winter_wheat',
                      'herbaceous_wetlands', 'woody_wetlands', 'open_water', 'alfalfa',
                      'fallow_idle_cropland', 'sorghum', 'developed_low_intensity',
                      'barren', 'durum_wheat',
                      'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                      'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                      'lentils', 'missing_reflectance', 'SIF']
BAND_STATISTICS_FILE = os.path.join(CFIS_DIR, 'cfis_band_statistics_train.csv')

# Lat/lon bounds. Note: *region indices* refer to the pixel index relative to the
# top left corner (TOP_BOUND, LEFT_BOUND). (0, 0) is the top-left pixel, and
# each pixel has size "RES" (0.00026949458523585647) degrees. 
LEFT_BOUND = -108
RIGHT_BOUND = -82
BOTTOM_BOUND = 38
TOP_BOUND = 48.7
RES = (0.00026949458523585647, 0.00026949458523585647)  # Degrees per Landsat pixel
TILE_SIZE_PIXELS = 100  # Size of output tile, in Landsat pixels
TILE_SIZE_DEGREES = TILE_SIZE_PIXELS * RES[0]
COARSE_SIF_PIXELS = 25  # Size of each (generated) coarse SIF datapoint, in Landsat pixels

REFLECTANCE_PIXELS = 371
INPUT_CHANNELS = 43
MISSING_REFLECTANCE_IDX = -1


# Loop through all dates
for date_idx, DATE in enumerate(DATES):
    MONTH = MONTHS[date_idx]
    INPUT_TILES_DIR = os.path.join(DATA_DIR, "tiles_" + DATE)
    OUTPUT_TILES_DIR = os.path.join(DATA_DIR, "tiles_cfis_" + DATE)
    if not os.path.exists(OUTPUT_TILES_DIR):
        os.makedirs(OUTPUT_TILES_DIR)

    # Output tile prefixes
    INPUT_TILE_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'input_tile_')
    # COARSE_SIF_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'coarse_sif_')
    FINE_SIF_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'fine_sif_')
    # COARSE_SOUNDINGS_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'coarse_soundings_')
    FINE_SOUNDINGS_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'fine_soundings_')

    # Maps from *region indices* of the upper/left corner of each output tile to:
    # sum SIF, avg SIF, number of soundings
    tile_to_sum_cfis_sif_array = dict()
    tile_to_avg_cfis_sif_array = dict()
    tile_to_cfis_soundings_array = dict()
    tile_to_oco2_sif = dict()
    tile_to_oco2_soundings = dict()


    # Read OCO-2 dataset
    dataset = xr.open_dataset(OCO2_FILES[date_idx])
    oco2_times = dataset.time.values
    oco2_lons = dataset.lon.values
    oco2_lats = dataset.lat.values
    oco2_sifs = dataset.dcSIF.values
    oco2_num_soundings = dataset.n.values
    print(dataset)
    print('Times', oco2_times.shape)
    print('Lons', oco2_lons.shape)
    print('Lats', oco2_lats.shape)
    print('Sifs', oco2_sifs.shape)

    # In order to align CFIS and OCO-2 grids, we need to know where to "start" the OCO-2 grid.
    # (OCO2_LAT_OFFSET, OCO2_LON_OFFSET) are the *region indices* of the upper left corner of
    # the OCO-2 grid (where the grid starts). 
    # (Recall: *region indices* are relative to TOP_BOUND/LEFT_BOUND of the entire dataset.)
    oco2_grid_top_degrees = oco2_lats[0] + TILE_SIZE_DEGREES / 2
    oco2_grid_left_degrees = oco2_lons[0] - TILE_SIZE_DEGREES / 2
    oco2_grid_top_idx, oco2_grid_left_idx = sif_utils.lat_long_to_index(oco2_grid_top_degrees, oco2_grid_left_degrees, TOP_BOUND, LEFT_BOUND, RES)
    OCO2_LON_OFFSET = oco2_grid_left_idx % TILE_SIZE_PIXELS
    OCO2_LAT_OFFSET = oco2_grid_top_idx % TILE_SIZE_PIXELS
    # print('OCO2 upper-left degrees', oco2_grid_top_degrees, oco2_grid_left_degrees)
    # print('OCO2 upper-left idx', oco2_grid_top_idx, oco2_grid_left_idx)
    # print('OCO2 upper-left idx offset', OCO2_LAT_OFFSET, OCO2_LON_OFFSET)
    # exit(0)

    # Loop through all OCO-2 grid squares
    for oco2_lon_idx in range(len(oco2_lons)):
        for oco2_lat_idx in range(len(oco2_lats)):
            oco2_sif = oco2_sifs[oco2_lat_idx, oco2_lon_idx, 0]
            oco2_num_soundings_point = oco2_num_soundings[oco2_lat_idx, oco2_lon_idx, 0]

            # Ignore grid squares where there is no SIF data
            if math.isnan(oco2_sif):
                continue

            # Find lon/lat of the OCO-2 tile's top-left corner 
            oco2_left_lon = oco2_lons[oco2_lon_idx] - TILE_SIZE_DEGREES / 2
            oco2_top_lat = oco2_lats[oco2_lat_idx] + TILE_SIZE_DEGREES / 2    

            # Compute *region indices* of the top-left corner
            lat_idx, lon_idx = sif_utils.lat_long_to_index(oco2_top_lat, oco2_left_lon, TOP_BOUND, LEFT_BOUND, RES)

            # Record OCO-2 SIF / num soundings in dictionary
            tile_indices = (lat_idx, lon_idx)
            tile_to_oco2_sif[tile_indices] = oco2_sif * OCO2_SCALING_FACTOR
            tile_to_oco2_soundings[tile_indices] = oco2_num_soundings_point


    # Read CFIS data
    lons = np.load(os.path.join(CFIS_DIR, "lons_" + MONTH + ".npy"), allow_pickle=True)
    lats = np.load(os.path.join(CFIS_DIR, "lats_" + MONTH + ".npy"), allow_pickle=True)
    sifs = np.load(os.path.join(CFIS_DIR, "dcsif_" + MONTH + ".npy"), allow_pickle=True)
    print('Lons shape', lons.shape)
    print('Lats shape', lats.shape)
    print('Sifs shape', sifs.shape)

    # Loop through all soundings. The arrays in "tile_to_sum_sif_array" will store the SUM
    # of all SIF soundings per pixel. Then we will divide by the number of soundings (computed in
    # "cfis_") to get the AVERAGE.
    for i in range(lons.shape[0]):
        if lats[i] < BOTTOM_BOUND or lats[i] > TOP_BOUND or lons[i] < LEFT_BOUND or \
                                     lons[i] > RIGHT_BOUND:
            # print('Sounding was outside region of interest! lat', lats[i], 'lon', lons[i])
            continue

        # Find the *region indices* (relative to the (top, left) corner of the entire region)
        # that the sounding falls in.
        lat_idx, lon_idx = sif_utils.lat_long_to_index(lats[i], lons[i], TOP_BOUND, LEFT_BOUND, RES)
        if (lat_idx < OCO2_LAT_OFFSET or lon_idx < OCO2_LON_OFFSET):
            print('Attention - point outside OCO2 grid: lat', lats[i], 'lon', lons[i])
            continue

        # We're breaking the region into large tiles of size "TILE_SIZE_PIXELS", but with the upper-left
        # corner of the grid being at (OCO2_LAT_OFFSET, OCO2_LON_OFFSET).
        # For example, if OCO2_LAT_OFFSET is 3 and TILE_SIZE_PIXELS is 100, the latitude grid lines are at
        # indices (3, 103, 203, 303, etc.) This code computes the upper-left corner of the grid square
        # containing the sounding at (lat_idx, lon_idx)
        tile_indices = (sif_utils.round_down(lat_idx - OCO2_LAT_OFFSET, TILE_SIZE_PIXELS) + OCO2_LAT_OFFSET,
                        sif_utils.round_down(lon_idx - OCO2_LON_OFFSET, TILE_SIZE_PIXELS) + OCO2_LON_OFFSET)

        # If this SIF tile is not yet in "tile_to_sum_cfis_sif_array", add it. Otherwise, extract
        # the SIF tile.
        if tile_indices not in tile_to_sum_cfis_sif_array:
            tile_sum_sifs = np.zeros([TILE_SIZE_PIXELS, TILE_SIZE_PIXELS])
            tile_soundings = np.zeros([TILE_SIZE_PIXELS, TILE_SIZE_PIXELS])
        else:
            tile_sum_sifs = tile_to_sum_cfis_sif_array[tile_indices]
            tile_soundings = tile_to_cfis_soundings_array[tile_indices]

        # Get the index of the current pixel within this SIF tile
        within_tile_lat_idx = lat_idx - tile_indices[0]
        within_tile_lon_idx = lon_idx - tile_indices[1]

        # Add this sounding's SIF to the sum of SIFs for this pixel
        tile_sum_sifs[within_tile_lat_idx, within_tile_lon_idx] += sifs[i]

        # Increment sounding count for this pixel
        tile_soundings[within_tile_lat_idx, within_tile_lon_idx] += 1

        # Store the SIF tile and sounding counts
        tile_to_sum_cfis_sif_array[tile_indices] = tile_sum_sifs
        tile_to_cfis_soundings_array[tile_indices] = tile_soundings


    # Now, compute the average SIF for each fine pixel
    for tile_indices in tile_to_sum_cfis_sif_array:
        sum_sif_array = tile_to_sum_cfis_sif_array[tile_indices]
        soundings_array = tile_to_cfis_soundings_array[tile_indices]
        num_pixels_with_data = np.count_nonzero(soundings_array)

        # Create a mask of which pixels have enough SIF soundings to be reliable
        invalid_mask = soundings_array < MIN_SOUNDINGS_FINE
        valid_mask = soundings_array >= MIN_SOUNDINGS_FINE

        # Create average SIF array
        avg_sif_array = np.zeros_like(sum_sif_array)
        avg_sif_array[valid_mask] = (sum_sif_array[valid_mask] / soundings_array[valid_mask])

        # # Also filter out low-SIF pixels
        # low_sif_mask = avg_sif_array < MIN_SIF
        # invalid_mask = np.logical_or(invalid_mask, low_sif_mask)
        tile_to_avg_cfis_sif_array[tile_indices] = np.ma.array(avg_sif_array, mask=invalid_mask)


    # For each CFIS and/or OCO2 SIF tile, extract reflectance/crop cover data.
    all_tile_indices = tile_to_avg_cfis_sif_array.keys() | tile_to_oco2_sif.keys()  # Union ("OR") of CFIS/OCO-2 tiles
    print('CFIS tile indices', len(tile_to_avg_cfis_sif_array))
    print('OCO2 tile indices', len(tile_to_oco2_sif))
    print('All tile indices', len(all_tile_indices))
    for tile_indices in all_tile_indices:
        # Randomly assign this tile into train/val/test
        random_number = random.random()
        if random_number < 1 - FRACTION_VAL - FRACTION_TEST:
            split = 'train'
        elif random_number < 1 - FRACTION_TEST:
            split = 'val'
        else:
            split = 'test'

        # From region indices, compute lat/lon bounds on this tile
        tile_max_lat = TOP_BOUND - (tile_indices[0] * RES[0])
        tile_min_lat = tile_max_lat - (TILE_SIZE_PIXELS * RES[0])
        tile_min_lon = LEFT_BOUND + (tile_indices[1] * RES[1])
        tile_max_lon = tile_min_lon + (TILE_SIZE_PIXELS * RES[1])

        # Extract input data for this region from files
        input_tile = sif_utils.extract_input_subtile(tile_min_lon, tile_max_lon, tile_min_lat, tile_max_lat,
                                                     INPUT_TILES_DIR, TILE_SIZE_PIXELS, RES)
        if input_tile is None:
            print('No input tiles found')
            continue

        # Tile description
        tile_center_lat = round((tile_min_lat + tile_max_lat) / 2, 5)
        tile_center_lon = round((tile_min_lon + tile_max_lon) / 2, 5)
        tile_description = 'lat_' + str(tile_center_lat) + '_lon_' + str(tile_center_lon) + '_' + DATE

        # Check CDL coverage
        cdl_coverage = np.mean(np.sum(input_tile[CDL_INDICES, :, :], axis=0))
        assert cdl_coverage >= 0 and cdl_coverage <= 1
        if cdl_coverage < MIN_CDL_COVERAGE:
            print('CDL coverage too low', tile_description)
            continue

        landsat_cloud_mask = input_tile[MISSING_REFLECTANCE_IDX, :, :]

        # Save input data to file
        input_tile_filename = INPUT_TILE_PREFIX + tile_description + '.npy'
        np.save(input_tile_filename, input_tile)

        # If there is OCO-2 data for this tile, get OCO-2 SIF and number of soundings
        if tile_indices in tile_to_oco2_sif:
            oco2_sif = tile_to_oco2_sif[tile_indices]
            oco2_soundings = tile_to_oco2_soundings[tile_indices]
            average_input_features = sif_utils.compute_band_averages(input_tile, input_tile[MISSING_REFLECTANCE_IDX])
            oco2_tile_metadata = [tile_center_lon, tile_center_lat, DATE, input_tile_filename] + average_input_features.tolist() + [oco2_sif, oco2_soundings]

            if split == 'train':
                oco2_metadata_train.append(oco2_tile_metadata)
            elif split == 'val':
                oco2_metadata_val.append(oco2_tile_metadata)
            elif split == 'test':
                oco2_metadata_test.append(oco2_tile_metadata)
            else:
                print('Invalid split!!!', split)
                exit(1)   

        # If there is CFIS data for this tile, get CFIS fine-resolution and coarse-resolution SIF
        if tile_indices in tile_to_avg_cfis_sif_array:
            fine_sif_array = tile_to_avg_cfis_sif_array[tile_indices]
            fine_soundings_array = tile_to_cfis_soundings_array[tile_indices]

            # If the corresponding Landsat pixel has cloud cover, mark this SIF pixel as invalid
            new_invalid_mask = np.logical_or(fine_sif_array.mask, landsat_cloud_mask)
            fine_sif_array.mask = new_invalid_mask

            # Check how many SIF pixels in this tile actually have data. If not enough do, skip this tile.
            num_invalid_pixels = np.count_nonzero(fine_sif_array.mask)
            num_valid_pixels = fine_sif_array.size - num_invalid_pixels
            if num_invalid_pixels / fine_sif_array.size > MAX_INVALID_SIF_PIXELS:
                continue

            # Compute the average SIF and total number of soundings for this subregion
            tile_sif = fine_sif_array.mean()
            tile_soundings = fine_soundings_array.sum()
            average_input_features = sif_utils.compute_band_averages(input_tile, fine_sif_array.mask)

            # # Compute coarse-resolution version
            # NUM_COARSE_PIXELS_PER_TILE = int(TILE_SIZE_PIXELS / COARSE_SIF_PIXELS)
            # coarse_sif_array = np.zeros([NUM_COARSE_PIXELS_PER_TILE, NUM_COARSE_PIXELS_PER_TILE])
            # coarse_soundings_array = np.zeros([NUM_COARSE_PIXELS_PER_TILE, NUM_COARSE_PIXELS_PER_TILE])
            # coarse_invalid_mask = np.zeros([NUM_COARSE_PIXELS_PER_TILE, NUM_COARSE_PIXELS_PER_TILE])
            # NUM_COARSE_SIF_DATAPOINTS = 0
            # coarse_sif_points = []
    
            # for i in range(0, TILE_SIZE_PIXELS, COARSE_SIF_PIXELS):
            #     for j in range(0, TILE_SIZE_PIXELS, COARSE_SIF_PIXELS):
            #         # For each coarse area, grab the corresponding subregion from input tile and SIF
            #         input_subregion = input_tile[:, i:i+COARSE_SIF_PIXELS, j:j+COARSE_SIF_PIXELS]
            #         sif_subregion = fine_sif_array[i:i+COARSE_SIF_PIXELS, j:j+COARSE_SIF_PIXELS]
            #         soundings_subregion = fine_soundings_array[i:i+COARSE_SIF_PIXELS, j:j+COARSE_SIF_PIXELS]

            #         # Check how many SIF pixels in this "subregion" actually have data.
            #         # If many are missing data, mark this subregion as invalid
            #         num_invalid_pixels = np.count_nonzero(sif_subregion.mask)
            #         num_valid_pixels = sif_subregion.size - num_invalid_pixels
            #         if num_invalid_pixels / sif_subregion.size > MAX_INVALID_SIF_PIXELS:
            #             coarse_invalid_mask[i // COARSE_SIF_PIXELS, j // COARSE_SIF_PIXELS] = True
            #             continue

            #         # Compute the average SIF and total number of soundings for this subregion
            #         subregion_sif = sif_subregion.mean()
            #         subregion_soundings = soundings_subregion.sum()

            #         # # Compute avg sif using for loop
            #         # num_points = 0
            #         # sum_sif_subregion = 0.
            #         # for idx1 in range(sif_subregion.shape[0]):
            #         #     for idx2 in range(sif_subregion.shape[1]):
            #         #         if not sif_subregion.mask[idx1, idx2]:
            #         #             sum_sif_subregion += sif_subregion.data[idx1, idx2]
            #         #             num_points += 1
            #         # subregion_sif_for_loop = sum_sif_subregion / num_points
            #         # print('Subregion SIF: by mean method', subregion_sif, 'by for loop', subregion_sif_for_loop)
            #         # print('Subregion SIF:', subregion_sif, '- based on', subregion_soundings, 'soundings (from', (sif_subregion.size - num_invalid_pixels), 'pixels)')
            #         # print('Subregion:', sif_subregion)
            #         # print('Num soundings', subregion_soundings)
            #         # assert subregion_sif_for_loop == subregion_sif

            #         coarse_sif_array[i // COARSE_SIF_PIXELS, j // COARSE_SIF_PIXELS] = subregion_sif
            #         coarse_soundings_array[i // COARSE_SIF_PIXELS, j // COARSE_SIF_PIXELS] = subregion_soundings
            #         NUM_COARSE_SIF_DATAPOINTS += 1

            #         # Compute average input features for this coarse SIF area
            #         average_input_features = sif_utils.compute_band_averages(input_subregion, sif_subregion.mask)

            #         # Add a row to "coarse SIF metadata"
            #         subregion_lat = max_lat - RES[0] * (i + COARSE_SIF_PIXELS / 2)
            #         subregion_lon = min_lon + RES[1] * (j + COARSE_SIF_PIXELS / 2)
            #         coarse_sif_points.append([subregion_lon, subregion_lat, DATE, input_tile_filename] + average_input_features.tolist() + [subregion_sif, subregion_soundings, num_valid_pixels])

            # If this tile has too few coarse SIF datapoints, remove it
            # if NUM_COARSE_SIF_DATAPOINTS < MIN_COARSE_DATAPOINTS_PER_TILE:
            #     continue
            # coarse_sif_array_masked = np.ma.array(coarse_sif_array, mask=coarse_invalid_mask)

            # Fine SIF/soundings file names to output to
            fine_sif_filename = FINE_SIF_PREFIX + tile_description + '.npy'
            fine_soundings_filename = FINE_SOUNDINGS_PREFIX + tile_description + '.npy'
            # coarse_sif_filename = COARSE_SIF_PREFIX + tile_description + '.npy'
            # coarse_soundings_filename = COARSE_SOUNDINGS_PREFIX + tile_description + '.npy'

            # Write fine SIF tile and fine soundings tile to files
            fine_sif_array.dump(fine_sif_filename)
            np.save(fine_soundings_filename, fine_soundings_array)
            # coarse_sif_array_masked.dump(coarse_sif_filename)
            # np.save(coarse_soundings_filename, coarse_soundings_array)

            # Plot tile
            # cdl_utils.plot_tile(input_tile, coarse_sif_array_masked, fine_sif_array, center_lon, center_lat, TILE_SIZE_DEGREES, tile_description)

            # Create row: reflectance/crop cover tile, fine SIF, coarse SIF.
            tile_metadata = [tile_center_lon, tile_center_lat, DATE, input_tile_filename] + average_input_features.tolist() + [tile_sif, tile_soundings, num_valid_pixels, fine_sif_filename, fine_soundings_filename]

            # tile_metadata = [center_lon, center_lat, DATE, input_tile_filename, fine_sif_filename, coarse_sif_filename, fine_soundings_filename, coarse_soundings_filename]

            # For each *valid* fine-resolution SIF pixel, extract features, and add to dataset. To-do: vectorize
            fine_sif_points = []
            for i in range(fine_sif_array.shape[0]):
                for j in range(fine_sif_array.shape[1]):
                    # Exclude pixels where SIF is invalid
                    if fine_sif_array.mask[i, j]:
                        continue
                    input_features = input_tile[:, i, j]
                    if input_features[MISSING_REFLECTANCE_IDX] == 1:
                        print('BUG!! SIF was not marked invalid, even though the Landsat pixel is cloudy')
                        exit(1)
                    pixel_lat = tile_max_lat - RES[0] * i
                    pixel_lon = tile_min_lon + RES[1] * j
                    pixel_sif = fine_sif_array[i, j]
                    # subregion_sif = coarse_sif_array[i // COARSE_SIF_PIXELS, j // COARSE_SIF_PIXELS]
                    pixel_soundings = fine_soundings_array[i, j]
                    assert pixel_soundings >= MIN_SOUNDINGS_FINE
                    fine_sif_points.append([pixel_lon, pixel_lat, DATE, input_tile_filename] + input_features.tolist() + [pixel_sif, pixel_soundings, tile_sif])

            if split == 'train':
                coarse_averages_train.append(tile_metadata)
                fine_averages_train.extend(fine_sif_points)
            elif split == 'val':
                coarse_averages_val.append(tile_metadata)
                fine_averages_val.extend(fine_sif_points)
            elif split == 'test':
                coarse_averages_test.append(tile_metadata)
                fine_averages_test.extend(fine_sif_points)
            else:
                print('Invalid split!!!', split)
                exit(1)
    


print('Number of fine SIF points (train/val/test)', len(fine_averages_train), len(fine_averages_val), len(fine_averages_test))
print('Number of coarse SIF points (train/val/test)', len(coarse_averages_train), len(coarse_averages_val), len(coarse_averages_test))
# print('Number of U-Net tiles (train/val/test)', len(cfis_tile_metadata_train), len(cfis_tile_metadata_val), len(cfis_tile_metadata_test))

# Construct DataFrames
# cfis_tile_metadata_train_df = pd.DataFrame(cfis_tile_metadata_train, columns=CFIS_TILE_METADATA_COLUMNS)
# cfis_tile_metadata_val_df = pd.DataFrame(cfis_tile_metadata_val, columns=CFIS_TILE_METADATA_COLUMNS)
# cfis_tile_metadata_test_df = pd.DataFrame(cfis_tile_metadata_test, columns=CFIS_TILE_METADATA_COLUMNS)
oco2_metadata_train_df = pd.DataFrame(oco2_metadata_train, columns=BAND_AVERAGE_COLUMNS)
oco2_metadata_val_df = pd.DataFrame(oco2_metadata_val, columns=BAND_AVERAGE_COLUMNS)
oco2_metadata_test_df = pd.DataFrame(oco2_metadata_test, columns=BAND_AVERAGE_COLUMNS)
fine_averages_train_df = pd.DataFrame(fine_averages_train, columns=FINE_CFIS_AVERAGE_COLUMNS)
fine_averages_val_df = pd.DataFrame(fine_averages_val, columns=FINE_CFIS_AVERAGE_COLUMNS)
fine_averages_test_df = pd.DataFrame(fine_averages_test, columns=FINE_CFIS_AVERAGE_COLUMNS)
coarse_averages_train_df = pd.DataFrame(coarse_averages_train, columns=COARSE_CFIS_AVERAGE_COLUMNS)
coarse_averages_val_df = pd.DataFrame(coarse_averages_val, columns=COARSE_CFIS_AVERAGE_COLUMNS)
coarse_averages_test_df = pd.DataFrame(coarse_averages_test, columns=COARSE_CFIS_AVERAGE_COLUMNS)

# Write DataFrames to files
# cfis_tile_metadata_train_df.to_csv(CFIS_TILE_METADATA_TRAIN_FILE)
# cfis_tile_metadata_val_df.to_csv(CFIS_TILE_METADATA_VAL_FILE)
# cfis_tile_metadata_test_df.to_csv(CFIS_TILE_METADATA_TEST_FILE)
oco2_metadata_train_df.to_csv(OCO2_METADATA_TRAIN_FILE)
oco2_metadata_val_df.to_csv(OCO2_METADATA_VAL_FILE)
oco2_metadata_test_df.to_csv(OCO2_METADATA_TEST_FILE)
fine_averages_train_df.to_csv(FINE_AVERAGES_TRAIN_FILE)
fine_averages_val_df.to_csv(FINE_AVERAGES_VAL_FILE)
fine_averages_test_df.to_csv(FINE_AVERAGES_TEST_FILE)
coarse_averages_train_df.to_csv(COARSE_AVERAGES_TRAIN_FILE)
coarse_averages_val_df.to_csv(COARSE_AVERAGES_VAL_FILE)
coarse_averages_test_df.to_csv(COARSE_AVERAGES_TEST_FILE)

# Compute averages for each band
selected_columns = fine_averages_train_df[STATISTICS_COLUMNS]
print("Band values ARRAY shape", selected_columns.shape)
band_means = selected_columns.mean(axis=0)
band_stds = selected_columns.std(axis=0)
print("Band means", band_means)
print("Band stds", band_stds)

# Write band averages to file
statistics_rows = [['mean', 'std']]
for i, mean in enumerate(band_means):
    statistics_rows.append([band_means[i], band_stds[i]])
with open(BAND_STATISTICS_FILE, 'w') as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in statistics_rows:
        csv_writer.writerow(row)