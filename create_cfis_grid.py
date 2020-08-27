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
import cdl_utils
import sif_utils


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
MONTHS = ["Jun", "Aug"]
DATES = ["2016-06-15", "2016-08-01"]

MIN_SOUNDINGS_FINE = 1
MIN_COARSE_DATAPOINTS_PER_TILE = 1
MAX_INVALID_SIF_PIXELS = 0.8  # Max fraction of invalid pixels when computing coarse-resolution SIF
MIN_SIF = 0.2
FRACTION_VAL = 0.2
FRACTION_TEST = 0.2

# Columns for U-Net tile dataset (mapping tile to per-pixel SIF and coarse SIF)
TILE_COLUMNS = ['lon', 'lat', 'date', 'tile_file', 'sif_fine_file', 'sif_coarse_file', 'fine_soundings', 'coarse_soundings']

# U-Net tile datasets
TILE_METADATA_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_tile_metadata_train_1soundings.csv')
TILE_METADATA_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_tile_metadata_val_1soundings.csv')
TILE_METADATA_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_tile_metadata_test_1soundings.csv')
tile_metadata_train = []
tile_metadata_val = []
tile_metadata_test = []

# Columns for "band average" datasets, at both fine/coarse resolution
CFIS_AVERAGE_COLUMNS = ['lon', 'lat', 'date', 'tile_file', 'ref_1', 'ref_2', 'ref_3', 'ref_4',
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
FINE_CFIS_AVERAGE_COLUMNS = CFIS_AVERAGE_COLUMNS + ['coarse_sif']
COARSE_CFIS_AVERAGE_COLUMNS = CFIS_AVERAGE_COLUMNS + ['num_valid_pixels']

FINE_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_train_1soundings.csv')
FINE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_val_1soundings.csv')
FINE_AVERAGES_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_fine_averages_test_1soundings.csv')
fine_averages_train = []
fine_averages_val = []
fine_averages_test = []
COARSE_AVERAGES_TRAIN_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_train_1soundings.csv')
COARSE_AVERAGES_VAL_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_val_1soundings.csv')
COARSE_AVERAGES_TEST_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_averages_test_1soundings.csv')
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
BAND_STATISTICS_CSV_FILE = os.path.join(CFIS_DIR, 'cfis_band_statistics_train_1soundings.csv')

# Lat/lon bounds
LEFT_BOUND = -108
RIGHT_BOUND = -82
BOTTOM_BOUND = 38
TOP_BOUND = 48.7
RES = (0.00026949458523585647, 0.00026949458523585647)  # Degrees per Landsat pixel
TILE_SIZE_PIXELS = 200  # Size of output tile, in Landsat pixels
TILE_SIZE_DEGREES = TILE_SIZE_PIXELS * RES[0]
COARSE_SIF_PIXELS = 25  # Size of each (generated) coarse SIF datapoint, in Landsat pixels
REFLECTANCE_PIXELS = 371

INPUT_CHANNELS = 43
MISSING_REFLECTANCE_IDX = -1


for date_idx, DATE in enumerate(DATES):
    MONTH = MONTHS[date_idx]
    INPUT_TILES_DIR = os.path.join(DATA_DIR, "tiles_" + DATE)
    OUTPUT_TILES_DIR = os.path.join(DATA_DIR, "tiles_cfis_" + DATE)
    if not os.path.exists(OUTPUT_TILES_DIR):
        os.makedirs(OUTPUT_TILES_DIR)

    # Output tile prefixes
    INPUT_TILE_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'input_tile_')
    COARSE_SIF_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'coarse_sif_')
    FINE_SIF_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'fine_sif_')
    COARSE_SOUNDINGS_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'coarse_soundings_')
    FINE_SOUNDINGS_PREFIX = os.path.join(OUTPUT_TILES_DIR, 'fine_soundings_')

    # Maps from INDICES (upper/left corner of each output tile) to:
    # sum SIF, avg SIF, number of soundings
    tile_to_sum_sif_array = dict()
    tile_to_avg_sif_array = dict()
    tile_to_soundings_array = dict()

    # Read CFIS data
    lons = np.load(os.path.join(CFIS_DIR, "lons_" + MONTH + ".npy"), allow_pickle=True)
    lats = np.load(os.path.join(CFIS_DIR, "lats_" + MONTH + ".npy"), allow_pickle=True)
    sifs = np.load(os.path.join(CFIS_DIR, "dcsif_" + MONTH + ".npy"), allow_pickle=True)
    print('Lons shape', lons.shape)
    print('Lats shape', lats.shape)
    print('Sifs shape', sifs.shape)

    # # Plot CFIS points
    # plt.figure(figsize=(30, 10))
    # scatterplot = plt.scatter(lons, lats, c=sifs, cmap=plt.get_cmap('YlGn'), vmin=0.2, vmax=1.5)
    # plt.colorbar(scatterplot)
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('CFIS points, date: ' + DATE)
    # plt.savefig('exploratory_plots/cfis_points_' + DATE + '.png')
    # plt.close()

    # TODO: Build up an OCO-2 grid, each footprint is 100x100 pixels


    # Loop through all soundings. The arrays in "tile_to_sum_sif_array" will store the SUM
    # of all SIF soundings per pixel. Then we will divide by the number of soundings (computed in
    # "tile_to_soundings_array") to get the AVERAGE.
    for i in range(lons.shape[0]):
        if lats[i] < BOTTOM_BOUND or lats[i] > TOP_BOUND or lons[i] < LEFT_BOUND or \
                                     lons[i] > RIGHT_BOUND:
            # print('Sounding was outside region of interest! lat', lats[i], 'lon', lons[i])
            continue

        # Relative to the (top, left) corner of the entire region, find the indices of the Landsat
        # pixel the sounding falls in.
        lat_idx, lon_idx = sif_utils.lat_long_to_index(lats[i], lons[i], TOP_BOUND, LEFT_BOUND, RES)

        # We're breaking the region into large tiles of size "TILE_SIZE_PIXELS".
        # Compute the upper-left corner of the large tile that contains the sounding.
        tile_indices = (sif_utils.round_down(lat_idx, TILE_SIZE_PIXELS),
                        sif_utils.round_down(lon_idx, TILE_SIZE_PIXELS))

        # If this SIF tile is not yet in "tile_to_sif_array", add it. Otherwise, extract
        # the SIF tile.
        if tile_indices not in tile_to_sum_sif_array:
            tile_sum_sifs = np.zeros([TILE_SIZE_PIXELS, TILE_SIZE_PIXELS])
            tile_soundings = np.zeros([TILE_SIZE_PIXELS, TILE_SIZE_PIXELS])
        else:
            tile_sum_sifs = tile_to_sum_sif_array[tile_indices]
            tile_soundings = tile_to_soundings_array[tile_indices]

        # Get the index of the current pixel within the SIF tile
        within_tile_lat_idx = lat_idx % TILE_SIZE_PIXELS
        within_tile_lon_idx = lon_idx % TILE_SIZE_PIXELS

        # Add this sounding's SIF to the sum of SIFs for this pixel
        tile_sum_sifs[within_tile_lat_idx, within_tile_lon_idx] += sifs[i]

        # Increment sounding count for this pixel
        tile_soundings[within_tile_lat_idx, within_tile_lon_idx] += 1

        # Store the SIF tile and sounding counts
        tile_to_sum_sif_array[tile_indices] = tile_sum_sifs
        tile_to_soundings_array[tile_indices] = tile_soundings


    # Now, compute the average SIF for each pixel
    for tile_indices in tile_to_sum_sif_array:
        sum_sif_array = tile_to_sum_sif_array[tile_indices]
        soundings_array = tile_to_soundings_array[tile_indices]
        num_pixels_with_data = np.count_nonzero(soundings_array)

        # Create a mask of which pixels have enough SIF soundings to be reliable
        invalid_mask = soundings_array < MIN_SOUNDINGS_FINE
        valid_mask = soundings_array >= MIN_SOUNDINGS_FINE

        # Create average SIF array
        avg_sif_array = np.zeros_like(sum_sif_array)
        avg_sif_array[valid_mask] = sum_sif_array[valid_mask] / soundings_array[valid_mask]

        # # Also filter out low-SIF pixels
        # low_sif_mask = avg_sif_array < MIN_SIF
        # invalid_mask = np.logical_or(invalid_mask, low_sif_mask)
        tile_to_avg_sif_array[tile_indices] = np.ma.array(avg_sif_array, mask=invalid_mask)


    # For each CFIS SIF tile, extract reflectance/crop cover data.
    for tile_indices, fine_sif_array in tile_to_avg_sif_array.items():
        soundings_array = tile_to_soundings_array[tile_indices]

        max_lat = TOP_BOUND - (tile_indices[0] * RES[0])
        min_lat = max_lat - (TILE_SIZE_PIXELS * RES[0])
        min_lon = LEFT_BOUND + (tile_indices[1] * RES[1])
        max_lon = min_lon + (TILE_SIZE_PIXELS * RES[1])

        # Tile description
        center_lat = round((min_lat + max_lat) / 2, 5)
        center_lon = round((min_lon + max_lon) / 2, 5)
        tile_description = 'lat_' + str(center_lat) + '_lon_' + str(center_lon) + '_' + DATE

        # File names to output to
        input_tile_filename = INPUT_TILE_PREFIX + tile_description + '_1soundings.npy'
        fine_sif_filename = FINE_SIF_PREFIX + tile_description + '_1soundings.npy'
        coarse_sif_filename = COARSE_SIF_PREFIX + tile_description + '_1soundings.npy'
        fine_soundings_filename = FINE_SOUNDINGS_PREFIX + tile_description + '_1soundings.npy'
        coarse_soundings_filename = COARSE_SOUNDINGS_PREFIX + tile_description + '_1soundings.npy'


        # Figure out which reflectance files to open. For each edge of the bounding box,
        # find the left/top bound of the surrounding reflectance large tile.
        min_lon_tile_left = (math.floor(min_lon * 10) / 10)
        max_lon_tile_left = (math.floor(max_lon * 10) / 10)
        min_lat_tile_top = (math.ceil(min_lat * 10) / 10)
        max_lat_tile_top = (math.ceil(max_lat * 10) / 10)
        num_tiles_lon = round((max_lon_tile_left - min_lon_tile_left) * 10) + 1
        num_tiles_lat = round((max_lat_tile_top - min_lat_tile_top) * 10) + 1
        file_left_lons = np.linspace(min_lon_tile_left, max_lon_tile_left, num_tiles_lon,
                                     endpoint=True)

         # Go through lats from top to bottom, because indices are numbered from top to bottom
        file_top_lats = np.linspace(min_lat_tile_top, max_lat_tile_top, num_tiles_lat,
                                    endpoint=True)[::-1]
        # print("File left lons", file_left_lons)
        # print("File top lats", file_top_lats)


        # Because a sub-tile could span multiple files, patch together all of the files that
        # contain any portion of the sub-tile
        columns = []
        FILE_EXISTS = False  # Set to True if at least one file exists
        for file_left_lon in file_left_lons:
            rows = []
            for file_top_lat in file_top_lats:
                # Find what reflectance file to read from
                file_center_lon = round(file_left_lon + 0.05, 2)
                file_center_lat = round(file_top_lat - 0.05, 2)
                large_tile_filename = INPUT_TILES_DIR + "/reflectance_lat_" + str(file_center_lat) +  \
                                      "_lon_" + str(file_center_lon) + ".npy"
                if not os.path.exists(large_tile_filename):
                    print('Needed data file', large_tile_filename, 'does not exist!')
                    # For now, consider the data for this section as missing
                    missing_tile = np.zeros((INPUT_CHANNELS, REFLECTANCE_PIXELS,
                                             REFLECTANCE_PIXELS))
                    missing_tile[-1, :, :] = 1
                    rows.append(missing_tile)
                else:
                    # print('Large tile filename', large_tile_filename)
                    large_tile = np.load(large_tile_filename)
                    rows.append(large_tile)
                    FILE_EXISTS = True

            column = np.concatenate(rows, axis=1)
            columns.append(column)

        # If no input files exist, ignore this tile
        if not FILE_EXISTS:
            continue

        combined_large_tiles = np.concatenate(columns, axis=2)
        # print('All large tiles shape', combined_large_tiles.shape)

        # Find indices of bounding box within this combined large tile
        top_idx, left_idx = sif_utils.lat_long_to_index(max_lat, min_lon, max_lat_tile_top,
                                                        min_lon_tile_left, RES)
        bottom_idx = top_idx + TILE_SIZE_PIXELS
        right_idx = left_idx + TILE_SIZE_PIXELS
        # print('From combined large tile: Top', top_idx, 'Bottom', bottom_idx, 'Left', left_idx, 'Right', right_idx)

        # If the selected region (box) goes outside the range of the cover or reflectance dataset, that's a bug!
        if top_idx < 0 or left_idx < 0:
            print("Index was negative!")
            exit(1)
        if (bottom_idx >= combined_large_tiles.shape[1] or right_idx >= combined_large_tiles.shape[2]):
            print("Reflectance index went beyond edge of array!")
            exit(1)

        input_tile = combined_large_tiles[:, top_idx:bottom_idx, left_idx:right_idx]
        landsat_cloud_mask = input_tile[MISSING_REFLECTANCE_IDX, :, :]

        # If the corresponding Landsat pixel has cloud cover, mark this SIF pixel as invalid
        new_invalid_mask = np.logical_or(fine_sif_array.mask, landsat_cloud_mask)
        fine_sif_array.mask = new_invalid_mask

        # Compute coarse-resolution version
        NUM_COARSE_PIXELS_PER_TILE = int(TILE_SIZE_PIXELS / COARSE_SIF_PIXELS)
        coarse_sif_array = np.zeros([NUM_COARSE_PIXELS_PER_TILE, NUM_COARSE_PIXELS_PER_TILE])
        coarse_soundings_array = np.zeros([NUM_COARSE_PIXELS_PER_TILE, NUM_COARSE_PIXELS_PER_TILE])
        coarse_invalid_mask = np.zeros([NUM_COARSE_PIXELS_PER_TILE, NUM_COARSE_PIXELS_PER_TILE])
        NUM_COARSE_SIF_DATAPOINTS = 0
        coarse_sif_points = []
    
        for i in range(0, TILE_SIZE_PIXELS, COARSE_SIF_PIXELS):
            for j in range(0, TILE_SIZE_PIXELS, COARSE_SIF_PIXELS):
                # For each coarse area, grab the corresponding subregion from input tile and SIF
                input_subregion = input_tile[:, i:i+COARSE_SIF_PIXELS, j:j+COARSE_SIF_PIXELS]
                sif_subregion = fine_sif_array[i:i+COARSE_SIF_PIXELS, j:j+COARSE_SIF_PIXELS]
                soundings_subregion = soundings_array[i:i+COARSE_SIF_PIXELS, j:j+COARSE_SIF_PIXELS]

                # Check how many SIF pixels in this "subregion" actually have data.
                # If many are missing data, mark this subregion as invalid
                num_invalid_pixels = np.count_nonzero(sif_subregion.mask)
                num_valid_pixels = sif_subregion.size - num_invalid_pixels
                if num_invalid_pixels / sif_subregion.size > MAX_INVALID_SIF_PIXELS:
                    coarse_invalid_mask[i // COARSE_SIF_PIXELS, j // COARSE_SIF_PIXELS] = True
                    continue

                # Compute the average SIF and total number of soundings for this subregion
                subregion_sif = sif_subregion.mean()
                subregion_soundings = soundings_subregion.sum()

                # # Compute avg sif using for loop
                # num_points = 0
                # sum_sif_subregion = 0.
                # for idx1 in range(sif_subregion.shape[0]):
                #     for idx2 in range(sif_subregion.shape[1]):
                #         if not sif_subregion.mask[idx1, idx2]:
                #             sum_sif_subregion += sif_subregion.data[idx1, idx2]
                #             num_points += 1
                # subregion_sif_for_loop = sum_sif_subregion / num_points
                # print('Subregion SIF: by mean method', subregion_sif, 'by for loop', subregion_sif_for_loop)
                # print('Subregion SIF:', subregion_sif, '- based on', subregion_soundings, 'soundings (from', (sif_subregion.size - num_invalid_pixels), 'pixels)')
                # print('Subregion:', sif_subregion)
                # print('Num soundings', subregion_soundings)
                # assert subregion_sif_for_loop == subregion_sif


                coarse_sif_array[i // COARSE_SIF_PIXELS, j // COARSE_SIF_PIXELS] = subregion_sif
                coarse_soundings_array[i // COARSE_SIF_PIXELS, j // COARSE_SIF_PIXELS] = subregion_soundings
                NUM_COARSE_SIF_DATAPOINTS += 1

                # Compute average input features for this subregion, over the pixels that have SIF data
                input_pixels = np.moveaxis(input_subregion, 0, -1)
                input_pixels = input_pixels.reshape((-1, input_pixels.shape[2]))
                invalid_pixels = sif_subregion.mask.flatten()
                pixels_with_data = input_pixels[invalid_pixels == 0, :]
                average_input_features = np.mean(pixels_with_data, axis=0)

                # Only change the "missing reflectance" feature to be the average across all pixels
                # (not just non-missing ones)
                average_input_features[MISSING_REFLECTANCE_IDX] = np.mean(input_subregion[MISSING_REFLECTANCE_IDX, :, :])
                # print('Avg input features', average_input_features)

                # Add a row to "coarse SIF metadata"
                subregion_lat = max_lat - RES[0] * (i + COARSE_SIF_PIXELS / 2)
                subregion_lon = min_lon + RES[1] * (j + COARSE_SIF_PIXELS / 2)
                coarse_sif_points.append([subregion_lon, subregion_lat, DATE, input_tile_filename] + average_input_features.tolist() + [subregion_sif, subregion_soundings, num_valid_pixels])

        # If this tile has too few coarse SIF datapoints, remove it
        if NUM_COARSE_SIF_DATAPOINTS < MIN_COARSE_DATAPOINTS_PER_TILE:
            continue
        coarse_sif_array_masked = np.ma.array(coarse_sif_array, mask=coarse_invalid_mask)

        # Write reflectance/cover tile, fine SIF tile, and coarse SIF tile to files
        np.save(input_tile_filename, input_tile)
        fine_sif_array.dump(fine_sif_filename)
        coarse_sif_array_masked.dump(coarse_sif_filename)
        np.save(fine_soundings_filename, soundings_array)
        np.save(coarse_soundings_filename, coarse_soundings_array)


        # Plot tile
        # cdl_utils.plot_tile(input_tile, coarse_sif_array_masked, fine_sif_array, center_lon, center_lat, TILE_SIZE_DEGREES, tile_description)

        # Create row: reflectance/crop cover tile, fine SIF, coarse SIF.
        tile_metadata = [center_lon, center_lat, DATE, input_tile_filename, fine_sif_filename, coarse_sif_filename, fine_soundings_filename, coarse_soundings_filename]

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
                pixel_lat = max_lat - RES[0] * i
                pixel_lon = min_lon + RES[1] * j
                pixel_sif = fine_sif_array[i, j]
                subregion_sif = coarse_sif_array[i // COARSE_SIF_PIXELS, j // COARSE_SIF_PIXELS]
                pixel_soundings = soundings_array[i, j]
                assert pixel_soundings >= MIN_SOUNDINGS_FINE
                fine_sif_points.append([pixel_lon, pixel_lat, DATE, input_tile_filename] + input_features.tolist() + [pixel_sif, pixel_soundings, subregion_sif])

        # Randomly assign to train/val/test
        random_number = random.random()
        if random_number < 1 - FRACTION_VAL - FRACTION_TEST:
            tile_metadata_train.append(tile_metadata)
            fine_averages_train.extend(fine_sif_points)
            coarse_averages_train.extend(coarse_sif_points)
        elif random_number < 1 - FRACTION_TEST:
            tile_metadata_val.append(tile_metadata)
            fine_averages_val.extend(fine_sif_points)
            coarse_averages_val.extend(coarse_sif_points)
        else:
            tile_metadata_test.append(tile_metadata)
            fine_averages_test.extend(fine_sif_points)
            coarse_averages_test.extend(coarse_sif_points)

print('Number of fine SIF points (train/val/test)', len(fine_averages_train), len(fine_averages_val), len(fine_averages_test))
print('Number of coarse SIF points (train/val/test)', len(coarse_averages_train), len(coarse_averages_val), len(coarse_averages_test))
print('Number of U-Net tiles (train/val/test)', len(tile_metadata_train), len(tile_metadata_val), len(tile_metadata_test))

# Construct DataFrames
tile_metadata_train_df = pd.DataFrame(tile_metadata_train, columns=TILE_COLUMNS)
tile_metadata_val_df = pd.DataFrame(tile_metadata_val, columns=TILE_COLUMNS)
tile_metadata_test_df = pd.DataFrame(tile_metadata_test, columns=TILE_COLUMNS)
fine_averages_train_df = pd.DataFrame(fine_averages_train, columns=FINE_CFIS_AVERAGE_COLUMNS)
fine_averages_val_df = pd.DataFrame(fine_averages_val, columns=FINE_CFIS_AVERAGE_COLUMNS)
fine_averages_test_df = pd.DataFrame(fine_averages_test, columns=FINE_CFIS_AVERAGE_COLUMNS)
coarse_averages_train_df = pd.DataFrame(coarse_averages_train, columns=COARSE_CFIS_AVERAGE_COLUMNS)
coarse_averages_val_df = pd.DataFrame(coarse_averages_val, columns=COARSE_CFIS_AVERAGE_COLUMNS)
coarse_averages_test_df = pd.DataFrame(coarse_averages_test, columns=COARSE_CFIS_AVERAGE_COLUMNS)

# Write DataFrames to files
tile_metadata_train_df.to_csv(TILE_METADATA_TRAIN_FILE)
tile_metadata_val_df.to_csv(TILE_METADATA_VAL_FILE)
tile_metadata_test_df.to_csv(TILE_METADATA_TEST_FILE)
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
with open(BAND_STATISTICS_CSV_FILE, 'w') as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in statistics_rows:
        csv_writer.writerow(row)

# Plot distribution of each band
for idx, column in enumerate(STATISTICS_COLUMNS):
    if band_stds[idx] == 0:
        continue
    sif_utils.plot_histogram(fine_averages_train_df[column].to_numpy(),
                             "histogram_pixels_" + column + "_cfis.png",
                             title=column + ' (CFIS pixels)')
    sif_utils.plot_histogram(coarse_averages_train_df[column].to_numpy(),
                             "histogram_coarse_" + column + "_cfis.png",
                             title=column + ' (CFIS coarse subregions)')
    standardized_pixel_values = (fine_averages_train_df[column] - band_means[idx]) / band_stds[idx]
    sif_utils.plot_histogram(standardized_pixel_values.to_numpy(),
                             "histogram_pixels_" + column + "_cfis_std.png",
                             title=column + ' (CFIS pixels, std. by pixel std dev)')
    standardized_coarse_values = (coarse_averages_train_df[column] - band_means[idx]) / band_stds[idx]
    sif_utils.plot_histogram(standardized_coarse_values.to_numpy(),
                             "histogram_coarse_" + column + "_cfis_std.png",
                             title=column + ' (CFIS coarse subregions, std. by pixel std dev)')

sif_utils.plot_histogram(fine_averages_train_df['num_soundings'].to_numpy(),
                         "histogram_pixels_num_soundings.png",
                         title='Num soundings (CFIS pixels)')
sif_utils.plot_histogram(coarse_averages_train_df['num_soundings'].to_numpy(),
                         "histogram_coarse_num_soundings.png",
                         title='Num soundings (CFIS coarse subregions)')
sif_utils.plot_histogram(coarse_averages_train_df['num_valid_pixels'].to_numpy(),
                         "histogram_coarse_num_valid_pixels.png",
                         title='Num valid pixels (CFIS coarse subregions)')