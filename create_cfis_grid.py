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

# Set random seed
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
OCO2_DIR = os.path.join(DATA_DIR, "OCO2")
MONTHS = ["Jun", "Aug"]
DATES = ["2016-06-15", "2016-08-01"]
OCO2_FILES = [os.path.join(OCO2_DIR, "oco2_20160615_20160629_3km.nc"),
              os.path.join(OCO2_DIR, "oco2_20160801_20160816_3km.nc")]

MIN_SOUNDINGS_FINE = 1
MIN_FRACTION_VALID_PIXELS = 0.1  # Max fraction of invalid pixels when computing coarse-resolution SIF
MIN_CDL_COVERAGE = 0.5  # Exclude areas where there is no CDL coverage (e.g. Canada)
CDL_INDICES = list(range(12, 42))
FRACTION_VAL = 0.2
FRACTION_TEST = 0.2

# Scaling factors to correct for differing wavelengths
OCO2_SCALING_FACTOR = 1.69 / 1.52

# Number of folds
NUM_FOLDS = 5
TRAIN_FOLDS = [1, 2, 3]  # 1-indexed

# OCO-2 tile datasets to output to
OCO2_METADATA_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_overlap.csv')
oco2_metadata = []
# OCO2_METADATA_TRAIN_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_train.csv')
# OCO2_METADATA_VAL_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_val.csv')
# OCO2_METADATA_TEST_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_test.csv')

# Columns for "band average" datasets, at both fine/coarse resolution
BAND_AVERAGE_COLUMNS = ['fold', 'grid_fold', 'lon', 'lat', 'date', 'tile_file', 'ref_1', 'ref_2', 'ref_3', 'ref_4',
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
COARSE_CFIS_AVERAGE_COLUMNS = BAND_AVERAGE_COLUMNS + ['fraction_valid', 'fine_sif_file', 'fine_soundings_file']

# CFIS coarse/fine averages
CFIS_FINE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_fine_metadata.csv')
cfis_fine_metadata = []
# FINE_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_train.csv')
# FINE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_val.csv')
# FINE_AVERAGES_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_test.csv')
# fine_averages_folds = []
# fine_averages_val = []
# fine_averages_test = []

CFIS_COARSE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_metadata.csv')
cfis_coarse_metadata = []
# COARSE_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_train.csv')
# COARSE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_val.csv')
# COARSE_AVERAGES_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_test.csv')
# coarse_averages_train = []
# coarse_averages_val = []
# coarse_averages_test = []


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

REFLECTANCE_PIXELS = 371
INPUT_CHANNELS = 43
MISSING_REFLECTANCE_IDX = -1

# Divide the region into 0.5x0.5 degree large grid areas. Split them between folds.
GRID_AREA_DEGREES = 0.2
num_lat_squares = int((50-38) / GRID_AREA_DEGREES)
num_lon_squares = int((108-82) / GRID_AREA_DEGREES)
LATS = np.linspace(50, 38, num_lat_squares, endpoint=False)
LONS = np.linspace(-108, -82, num_lon_squares, endpoint=False)  # These lat/lons are the UPPER LEFT corner of the large grid areas
print('lats', LATS)
print('lons', LONS)
large_grid_areas = dict()
for lat in LATS:
    for lon in LONS:
        fold_number = random.randint(0, NUM_FOLDS - 1)
        large_grid_areas[(lat, lon)] = fold_number



# Loop through all dates
for date_idx, DATE in enumerate(DATES):
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

    # We divide the region into 1x1 grid squares. We keep track of which grid squares contained
    # CFIS points for this date, so that we can remove OCO-2 points that are in those grid 
    # squares (for these experiments, we want OCO-2 and CFIS to be from distinct geographic regions).
    # The elements of this set will contain upper-left corners of 1x1-degree grid squares
    # that contained the *upper-left corners* of CFIS tiles.
    grid_squares_with_cfis = set()

    # Data directories for this date
    MONTH = MONTHS[date_idx]
    INPUT_TILES_DIR = os.path.join(DATA_DIR, "tiles_" + DATE)
    OUTPUT_TILES_DIR = os.path.join(DATA_DIR, "tiles_cfis_" + DATE)
    if not os.path.exists(OUTPUT_TILES_DIR):
        os.makedirs(OUTPUT_TILES_DIR)

    # Output tile prefixes
    INPUT_TILE_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'input_tile_')
    FINE_SIF_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'fine_sif_')
    FINE_SOUNDINGS_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'fine_soundings_')

    # Maps from *region indices* of the upper/left corner of each output tile to:
    # sum SIF, avg SIF, number of soundings
    tile_to_sum_cfis_sif_array = dict()
    tile_to_avg_cfis_sif_array = dict()
    tile_to_cfis_soundings_array = dict()
    tile_to_oco2_sif = dict()
    tile_to_oco2_soundings = dict()

    # Read CFIS data
    lons = np.load(os.path.join(CFIS_DIR, "lons_" + MONTH + ".npy"), allow_pickle=True)
    lats = np.load(os.path.join(CFIS_DIR, "lats_" + MONTH + ".npy"), allow_pickle=True)
    sifs = np.load(os.path.join(CFIS_DIR, "dcsif_" + MONTH + ".npy"), allow_pickle=True)
    print('Lons shape', lons.shape)
    print('Lats shape', lats.shape)
    print('Sifs shape', sifs.shape)

    # Loop through all CFIS soundings. The arrays in "tile_to_sum_sif_array" will store the SUM
    # of all SIF soundings per pixel. Then we will divide by the number of soundings (computed in
    # "tile_to_cfis_soundings_array") to get the AVERAGE.
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


    # Now, compute the average CFIS SIF for each fine pixel
    for tile_indices in tile_to_sum_cfis_sif_array:
        sum_sif_array = tile_to_sum_cfis_sif_array[tile_indices]
        soundings_array = tile_to_cfis_soundings_array[tile_indices]

        # Create a mask of which pixels have enough SIF soundings to be reliable
        invalid_mask = soundings_array < MIN_SOUNDINGS_FINE
        valid_mask = soundings_array >= MIN_SOUNDINGS_FINE

        # Create average SIF array
        avg_sif_array = np.zeros_like(sum_sif_array)
        avg_sif_array[valid_mask] = (sum_sif_array[valid_mask] / soundings_array[valid_mask])

        # Store masked array in dictionary
        tile_to_avg_cfis_sif_array[tile_indices] = np.ma.array(avg_sif_array, mask=invalid_mask)

        # Compute the lat/lon grid square these tile indices are in. (This computation
        # is based on the upper-left corner of the tile.) Mark that this grid square 
        # contains a CFIS tile.
        lat = TOP_BOUND - tile_indices[0] * RES[0]
        lon = LEFT_BOUND + tile_indices[1] * RES[1]
        grid_square_upper_left = sif_utils.get_large_grid_area_coordinates_lat_first(lat, lon, GRID_AREA_DEGREES)
        print('Grid square upper left', grid_square_upper_left)
        if grid_square_upper_left not in grid_squares_with_cfis:
            grid_squares_with_cfis.add(grid_square_upper_left)


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

            # Compute the lat/lon grid square this OCO-2 tile is in. Again this computation is based
            # on the upper-left corner of the tile. If it overlaps with a CFIS region, skip this tile.
            grid_square = sif_utils.get_large_grid_area_coordinates_lat_first(oco2_top_lat, oco2_left_lon, GRID_AREA_DEGREES)
            if grid_square in grid_squares_with_cfis:
                print('Grid square', grid_square, 'had CFIS data!!!!!')
                # continue
            # else:
                # print('Grid square', grid_square, 'has no CFIS data :)')
            # print('=============================')

            # Compute *region indices* of the top-left corner
            lat_idx, lon_idx = sif_utils.lat_long_to_index(oco2_top_lat, oco2_left_lon, TOP_BOUND, LEFT_BOUND, RES)

            # Record OCO-2 SIF / num soundings in dictionary
            tile_indices = (lat_idx, lon_idx)
            tile_to_oco2_sif[tile_indices] = oco2_sif * OCO2_SCALING_FACTOR
            tile_to_oco2_soundings[tile_indices] = oco2_num_soundings_point


    # For each CFIS and/or OCO2 SIF tile, extract reflectance/crop cover data.
    all_tile_indices = tile_to_avg_cfis_sif_array.keys() | tile_to_oco2_sif.keys()  # Union ("OR") of CFIS/OCO-2 tiles
    print('CFIS tile indices', len(tile_to_avg_cfis_sif_array))
    print('OCO2 tile indices', len(tile_to_oco2_sif))
    print('All tile indices', len(all_tile_indices))
    for tile_indices in all_tile_indices:

        # From region indices, compute lat/lon bounds on this tile
        tile_max_lat = TOP_BOUND - (tile_indices[0] * RES[0])
        tile_min_lat = tile_max_lat - (TILE_SIZE_PIXELS * RES[0])
        tile_min_lon = LEFT_BOUND + (tile_indices[1] * RES[1])
        tile_max_lon = tile_min_lon + (TILE_SIZE_PIXELS * RES[1])

        # Compute which 1x1-degree grid square the tile is in, and what fold it is in
        random_fold_number = random.randint(0, NUM_FOLDS - 1)
        grid_square = sif_utils.get_large_grid_area_coordinates_lat_first(tile_max_lat, tile_min_lon, GRID_AREA_DEGREES)
        if grid_square not in large_grid_areas:
            print('Large grid areas:', large_grid_areas)
            print('Grid square', grid_square, 'not present in large_grid_areas :((((')
            exit(1)
        grid_fold_number = large_grid_areas[grid_square]

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

        # Save input data to file
        input_tile_filename = INPUT_TILE_PREFIX + tile_description + '.npy'
        np.save(input_tile_filename, input_tile)

        # Tile should definitely not be in both OCO-2 and CFIS
        # assert not (tile_indices in tile_to_oco2_sif and tile_indices in tile_to_avg_cfis_sif_array)

        # If there is OCO-2 data for this tile, get OCO-2 SIF and number of soundings
        if tile_indices in tile_to_oco2_sif:
            oco2_sif = tile_to_oco2_sif[tile_indices]
            oco2_soundings = tile_to_oco2_soundings[tile_indices]
            average_input_features = sif_utils.compute_band_averages(input_tile, input_tile[MISSING_REFLECTANCE_IDX])
            oco2_tile_metadata = [random_fold_number, grid_fold_number, tile_center_lon, tile_center_lat, DATE, input_tile_filename] + \
                                  average_input_features.tolist() + [oco2_sif, oco2_soundings]
            oco2_metadata.append(oco2_tile_metadata)

        # If there is CFIS data for this tile, get CFIS fine-resolution and coarse-resolution SIF
        if tile_indices in tile_to_avg_cfis_sif_array:
            fine_sif_array = tile_to_avg_cfis_sif_array[tile_indices]
            fine_soundings_array = tile_to_cfis_soundings_array[tile_indices]

            # If the corresponding Landsat pixel has cloud cover, mark this SIF pixel as invalid
            landsat_cloud_mask = input_tile[MISSING_REFLECTANCE_IDX, :, :]
            new_invalid_mask = np.logical_or(fine_sif_array.mask, landsat_cloud_mask)
            fine_sif_array.mask = new_invalid_mask

            # Check how many SIF pixels in this tile actually have data. If not enough do, skip this tile.
            fraction_valid = 1 - (np.count_nonzero(fine_sif_array.mask) / fine_sif_array.size)
            if fraction_valid < MIN_FRACTION_VALID_PIXELS:
                continue

            # Compute the average SIF (over valid pixels with enough soundings and no cloud cover)
            # and total number of soundings for this subregion
            tile_sif = fine_sif_array.mean()
            tile_soundings = fine_soundings_array.sum()
            average_input_features = sif_utils.compute_band_averages(input_tile, fine_sif_array.mask)

            # Fine SIF/soundings file names to output to
            fine_sif_filename = FINE_SIF_PREFIX + tile_description + '.npy'
            fine_soundings_filename = FINE_SOUNDINGS_PREFIX + tile_description + '.npy'

            # Write fine SIF tile and fine soundings tile to files
            fine_sif_array.dump(fine_sif_filename)
            np.save(fine_soundings_filename, fine_soundings_array)

            # Plot tile
            # cdl_utils.plot_tile(input_tile, coarse_sif_array_masked, fine_sif_array, center_lon, center_lat, TILE_SIZE_DEGREES, tile_description)

            # Create row: reflectance/crop cover tile, fine SIF, coarse SIF.
            tile_metadata = [random_fold_number, grid_fold_number, tile_center_lon, tile_center_lat, DATE, input_tile_filename] + \
                            average_input_features.tolist() + \
                            [tile_sif, tile_soundings, fraction_valid, fine_sif_filename, fine_soundings_filename]

            # For each *valid* fine-resolution SIF pixel, extract features, and add to dataset.
            fine_sif_points = []
            pixel_sif_sum = 0
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
                    pixel_sif_sum += pixel_sif
                    pixel_soundings = fine_soundings_array[i, j]
                    assert pixel_soundings >= MIN_SOUNDINGS_FINE
                    fine_sif_points.append([random_fold_number, grid_fold_number, pixel_lon, pixel_lat, DATE, input_tile_filename] + input_features.tolist() + [pixel_sif, pixel_soundings, tile_sif])

            cfis_coarse_metadata.append(tile_metadata)
            cfis_fine_metadata.extend(fine_sif_points)


# Construct DataFrames
oco2_metadata_df = pd.DataFrame(oco2_metadata, columns=BAND_AVERAGE_COLUMNS)
oco2_metadata_df.to_csv(OCO2_METADATA_FILE)
cfis_fine_metadata_df = pd.DataFrame(cfis_fine_metadata, columns=FINE_CFIS_AVERAGE_COLUMNS)
cfis_fine_metadata_df.to_csv(CFIS_FINE_METADATA_FILE)
cfis_coarse_metadata_df = pd.DataFrame(cfis_coarse_metadata, columns=COARSE_CFIS_AVERAGE_COLUMNS)
cfis_coarse_metadata_df.to_csv(CFIS_COARSE_METADATA_FILE)

print('Number of OCO-2 SIF points:', len(oco2_metadata_df))
print('OCO2 by random fold:', oco2_metadata_df['fold'].value_counts())
print('OCO2 by grid fold:', oco2_metadata_df['grid_fold'].value_counts())
print('Number of coarse CFIS SIF points:', len(cfis_coarse_metadata_df))
print('Coarse CFIS by fold:', cfis_coarse_metadata_df['fold'].value_counts())
print('Coarse CFIS by grid fold:', cfis_coarse_metadata_df['grid_fold'].value_counts())
print('Number of fine CFIS SIF points:', len(cfis_fine_metadata_df))

# oco2_metadata_train_df = pd.DataFrame(oco2_metadata_train, columns=BAND_AVERAGE_COLUMNS)
# oco2_metadata_val_df = pd.DataFrame(oco2_metadata_val, columns=BAND_AVERAGE_COLUMNS)
# oco2_metadata_test_df = pd.DataFrame(oco2_metadata_test, columns=BAND_AVERAGE_COLUMNS)
# fine_averages_train_df = pd.DataFrame(fine_averages_train, columns=FINE_CFIS_AVERAGE_COLUMNS)
# fine_averages_val_df = pd.DataFrame(fine_averages_val, columns=FINE_CFIS_AVERAGE_COLUMNS)
# fine_averages_test_df = pd.DataFrame(fine_averages_test, columns=FINE_CFIS_AVERAGE_COLUMNS)
# coarse_averages_train_df = pd.DataFrame(coarse_averages_train, columns=COARSE_CFIS_AVERAGE_COLUMNS)
# coarse_averages_val_df = pd.DataFrame(coarse_averages_val, columns=COARSE_CFIS_AVERAGE_COLUMNS)
# coarse_averages_test_df = pd.DataFrame(coarse_averages_test, columns=COARSE_CFIS_AVERAGE_COLUMNS)

# # Write DataFrames to files
# oco2_metadata_train_df.to_csv(OCO2_METADATA_TRAIN_FILE)
# oco2_metadata_val_df.to_csv(OCO2_METADATA_VAL_FILE)
# oco2_metadata_test_df.to_csv(OCO2_METADATA_TEST_FILE)
# fine_averages_train_df.to_csv(FINE_AVERAGES_TRAIN_FILE)
# fine_averages_val_df.to_csv(FINE_AVERAGES_VAL_FILE)
# fine_averages_test_df.to_csv(FINE_AVERAGES_TEST_FILE)
# coarse_averages_train_df.to_csv(COARSE_AVERAGES_TRAIN_FILE)
# coarse_averages_val_df.to_csv(COARSE_AVERAGES_VAL_FILE)
# coarse_averages_test_df.to_csv(COARSE_AVERAGES_TEST_FILE)

# Compute averages for each band
fine_averages_train_df = cfis_fine_metadata_df[cfis_fine_metadata_df['fold'].isin(TRAIN_FOLDS)]
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
