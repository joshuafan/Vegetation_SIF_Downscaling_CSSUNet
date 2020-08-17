


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
MONTH = "Jun"
DATE = "2016-06-15"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + DATE)
OUTPUT_DATASET_DIR = os.path.join(DATA_DIR, "dataset_cfis_" + DATE)  

MIN_SOUNDINGS = 5
MAX_FRACTION_INVALID_PIXELS = 0.5

# Columns for "band average" dataset
CFIS_AVERAGE_COLUMNS = ['lon', 'lat', 'date', 'tile_file', 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                        'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                        'grassland_pasture', 'corn', 'soybean', 'shrubland',
                        'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                        'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                        'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                        'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                        'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                        'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                        'lentils', 'missing_reflectance', 'sif', 'num_soundings']

# Columns for U-Net dataset (mapping tile to per-pixel SIF)
TILE_COLUMNS = ['lon', 'lat', 'date', 'tile_file', 'sif_fine_file', 'sif_coarse_file']

CFIS_FINE_TRAIN = os.path.join(CFIS_DIR, 'cfis_fine_train.csv')
CFIS_FINE_VAL = os.path.join(CFIS_DIR, 'cfis_fine_val.csv')
CFIS_FINE_TEST = os.path.join(CFIS_DIR, 'cfis_fine_test.csv')
CFIS_COARSE_TRAIN = os.path.join(CFIS_DIR, 'cfis_coarse_train.csv')
CFIS_COARSE_VAL = os.path.join(CFIS_DIR, 'cfis_coarse_val.csv')
CFIS_COARSE_TEST = os.path.join(CFIS_DIR, 'cfis_coarse_test.csv')
TILES_TRAIN = os.path.join(CFIS_DIR, 'cfis_tiles_train.csv')
TILES_VAL = os.path.join(CFIS_DIR, 'cfis_tiles_val.csv')
TILES_TEST = os.path.join(CFIS_DIR, 'cfis_tiles_test.csv')

# Lat/lon bounds
LEFT_BOUND = -108
RIGJT_BOUND = -82
BOTTOM_BOUND = 38
TOP_BOUND = 48.7
RES = (0.00026949458523585647, 0.00026949458523585647)
TILE_SIZE_PIXELS = 300
COARSE_SIF_PIXELS = 20

# Maps from INDICES (upper/left) to SIF
tile_to_sif_array = dict()
tile_to_soundings_array = dict()

# Read CFIS data
lons = np.load(os.path.join(CFIS_DIR, "lons_" + MONTH + ".npy"))
lats = np.load(os.path.join(CFIS_DIR, "lats_" + MONTH + ".npy"))
sifs = np.load(os.path.join(CFIS_DIR, "dcsif_" + MONTH + ".npy"))
print('Lons shape', lons.shape)
print('Lats shape', lats.shape)
print('Sifs shape', sifs.shape)


# Loop through all soundings. For now, the arrays in "tile_to_sif_array" will store the SUM
# of all SIF soundings per pixel. Then we will divide by the number of soundings (computed in
# "tile_to_soundings_array") to get the AVERAGE.
for i in range(lons.shape[0]):
    lat_idx, lon_idx = lat_long_to_index(lats[i], lons[i], TOP_BOUND, LEFT_BOUND, RES)
    print('Lat', lats[i], 'Lon', lons[i], 'Indices', lat_idx, lon_idx)
    exit(0)
    tile_indices = (lat_idx // TILE_SIZE_PIXELS, lon_idx // TILE_SIZE_PIXELS)  # Upper left corner of tile
    if tile_indices not in tile_to_sif_array:
        tile_sifs = np.zeros([TILE_SIZE_PIXELS, TILE_SIZE_PIXELS])
        tile_soundings = np.zeros([TILE_SIZE_PIXELS, TILE_SIZE_PIXELS])
    else:
        tile_sifs = tile_to_sif_array[tile_indices]
        tile_soundings = tile_to_soundings_array[tile_indices]

    within_tile_lat_idx = lat_idx % TILE_SIZE_PIXELS
    within_tile_lon_idx = lon_idx % TILE_SIZE_PIXELS
    tile_sifs[within_tile_lat_idx, within_tile_lon_idx] += sifs[i]
    tile_soundings[within_tile_lat_idx, within_tile_lon_idx] += 1
    tile_to_sif_array[tile_indices] = tile_sifs
    tile_to_soundings_array[tile_indices] = tile_soundings


# Now, compute the average SIF for each pixel
for tile_indices in tile_to_sif_array:
    sif_array = tile_to_sif_array[tile_indices]
    soundings_array = tile_to_soundings_array[tile_indices]
    num_pixels_with_data = np.count_nonzero(soundings_array)
    print('=========================================')
    print('Tile indices', tile_indices, 'Pixels with >=1 soundings', num_pixels_with_data, 'Nonzero SIF', np.count_nonzero(sif_array))

    nonzero_indices = soundings_array[soundings_array > 0]
    invalid_mask = soundings_array[soundings_array < MIN_SOUNDINGS]
    valid_mask = soundings_array[soundings_array >= MIN_SOUNDINGS]
    print('Invalid', np.count_nonzero(invalid_mask))
    print('Valid mask', sif_array[valid_mask])
    print('Pixels with >=1 soundings', np.count_nonzero(soundings_array[soundings_array >= 1]))
    print('Pixels with >=2 soundings', np.count_nonzero(soundings_array[soundings_array >= 2]))
    print('Pixels with >=3 soundings', np.count_nonzero(soundings_array[soundings_array >= 3]))
    print('Pixels with >=5 soundings', np.count_nonzero(soundings_array[soundings_array >= 5]))
    print('Pixels with >=10 soundings', np.count_nonzero(soundings_array[soundings_array >= 10]))
    exit(0)
    sif_array[valid_mask] = sif_array[valid_mask] / soundings_array[valid_mask]
    tile_to_sif_array[tile_indices] = np.ma.array(sif_array, mask=invalid_mask)


# Compute a coarse-resolution version
tile_to_corase_sif_array = dict()
for tile_indices, sif_array in tile_to_sif_array.items():
    coarse_sif_array = np.zeros([TILE_SIZE_PIXELS / COARSE_SIF_PIXELS, TILE_SIZE_PIXELS / COARSE_SIF_PIXELS])
    coarse_invalid_mask = np.zeros([TILE_SIZE_PIXELS / COARSE_SIF_PIXELS, TILE_SIZE_PIXELS / COARSE_SIF_PIXELS])
    for i in range(0, TILE_SIZE_PIXELS, COARSE_SIF_PIXELS):
        for j in range(0, TILE_SIZE_PIXELS, COARSE_SIF_PIXELS):
            sif_subregion = sif_array[i:i+COARSE_SIF_PIXELS, j:j+COARSE_SIF_PIXELS]
            # soundings_subregion = soundings_array[i:i+COARSE_SIF_PIXELS, j:j+COARSE_SIF_PIXELS]
            print('*****************')
            print('SIF subregion', sif_subregion)

            # Check how many pixels in this "subregion" actually have data.
            # If many are missing data, mark this subregion as invalid
            num_invalid_pixels = np.count_nonzero(sif_subregion.mask)
            print('Num invalid', num_invalid_pixels)
            if num_invalid_pixels / (COARSE_SIF_PIXELS ** 2) > MAX_FRACTION_INVALID_PIXELS:
                coarse_invalid_mask[i / COARSE_SIF_PIXELS, j / COARSE_SIF_PIXELS] = True
                continue
            # mask = soundings_subregion[soundings_subregion >= MIN_SOUNDINGS]
            subregion_sif = sif_subregion.mean()
            print('Average SIF', )
            coarse_sif_array[i / COARSE_SIF_PIXELS, j / COARSE_SIF_PIXELS] = subregion_sif
    tile_to_coarse_sif_array[tile_indices] = np.ma.array(coarse_sif_array, mask=coarse_invalid_mask)  



# For each SIF tile, extract reflectance/crop cover data. Compute distribution of pixels.

    # If there are too few valid SIF labels, skip

    # Randomly assign to train/val/test

    # Create row: reflectance/crop cover tile, fine SIF, coarse SIF. Write all 3 to files.

    

    # For each *valid* fine-resolution SIF pixel, extract features, and add to dataset

    # For each *valid* coarse-resolution SIF pixel, compute feature averages, and add to dataset

# Write band statistics
