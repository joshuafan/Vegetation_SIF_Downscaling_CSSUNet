import numpy as np
import os
import pandas as pd
import shutil

# Directories
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
NEW_DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF_AAAI/test_data"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
OCO2_DIR = os.path.join(DATA_DIR, "OCO2")

# Train files
CFIS_COARSE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_metadata.csv')
CFIS_FINE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_fine_metadata.csv')
OCO2_METADATA_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_overlap.csv')
BAND_STATISTICS_FILE = os.path.join(CFIS_DIR, 'cfis_band_statistics_train.csv')

# Only include CFIS tiles where at least this fraction of pixels have CFIS
# fine-resolution data
MIN_COARSE_FRACTION_VALID_PIXELS = [0.1]

# Only EVALUATE on CFIS fine-resolution pixels with at least this number of soundings (measurements)
MIN_FINE_CFIS_SOUNDINGS = [30] #[30] #[1, 5, 10, 20, 30] # # 100, 250] #[100, 300, 1000, 3000]
eps = 1e-5
RANDOM_STATE = 0

# For resolutions greater than 30m, only evaluate on grid cells where at least this fraction
# of 30m pixels have any CFIS data
MIN_FINE_FRACTION_VALID_PIXELS = [0.9-eps] #[0.1, 0.3, 0.5, 0.7] # [0.5] #[0.5]

# Resolutions to consider
RESOLUTION_METERS = [30, 90, 150, 300, 600]

# Dates/sources
DATES = ["2016-06-15", "2016-08-01"]
TRAIN_DATES = ["2016-06-15", "2016-08-01"]
TEST_DATES = ["2016-06-15", "2016-08-01"]

# METHOD = "9a_Ridge_Regression_cfis" #_5soundings"
# METHOD = "9b_Gradient_Boosting_Regressor_cfis" #_5soundings"
# METHOD = "9c_MLP_cfis" #_10soundings"
# METHOD = "10a_Ridge_Regression_both"
# METHOD_READABLE = "Ridge Regression"
# METHOD = "10b_Gradient_Boosting_Regressor_both"
# METHOD = "10c_MLP_both"
# METHOD = "11a_Ridge_Regression_oco2"
# METHOD = "11b_Gradient_Boosting_Regressor_oco2"
# METHOD = "11c_MLP_oco2"

# List of sources to use (either CFIS or OCO-2)
TRAIN_SOURCES = ['CFIS', 'OCO2']

# For evaluation purposes, we consider a grid cell to be "pure" if at least this fraction
# of the cell is of a given land cover type
PURE_THRESHOLD = 0.7

# Only train on OCO-2 datapoints with at least this number of soundings
MIN_OCO2_SOUNDINGS = 3

# Remove OCO-2 and CFIS tiles with cloud cover that exceeds this threshold
MAX_OCO2_CLOUD_COVER = 0.5
MAX_CFIS_CLOUD_COVER = 0.5

# Clip inputs to this many standard deviations from mean
MIN_INPUT = -3
MAX_INPUT = 3

# Clip SIF predictions to be within this range, and exclude
# datapoints whose true SIF is outside this range
MIN_SIF_CLIP = 0.1
MAX_SIF_CLIP = None

# Range of SIF values to plot
MIN_SIF_PLOT = 0
MAX_SIF_PLOT = 1.5


# Input feature names
BAND_AVERAGE_COLUMNS = ['fold', 'grid_fold', 'lon', 'lat', 'date', 'tile_file', 'ref_1', 'ref_2', 'ref_3', 'ref_4',
                        'ref_5', 'ref_6', 'ref_7', 'ref_10', 'ref_11', 'Rainf_f_tavg',
                        'SWdown_f_tavg', 'Tair_f_tavg', 'grassland_pasture', 'corn', 'soybean',
                        'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                        'woody_wetlands', 'open_water', 'alfalfa',
                        'developed_low_intensity', 'developed_med_intensity', 'missing_reflectance', 'SIF', 'num_soundings']
FINE_CFIS_AVERAGE_COLUMNS = BAND_AVERAGE_COLUMNS + ['coarse_sif', 'fraction_valid']
COARSE_CFIS_AVERAGE_COLUMNS = BAND_AVERAGE_COLUMNS + ['fraction_valid', 'fine_sif_file', 'fine_soundings_file']
BANDS = list(range(0, 12)) + [12, 13, 14, 16, 17, 19, 23, 24, 25, 28, 34] + [42]

INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
                    'grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity', 'missing_reflectance']
# INPUT_COLUMNS = ["NDVI"]
ALL_COVER_COLUMNS = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']
REFLECTANCE_BANDS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11']
# INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
#                     'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
#                     'grassland_pasture', 'corn', 'soybean', 'shrubland',
#                     'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
#                     'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
#                     'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
#                     'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
#                     'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
#                     'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
#                     'lentils', 'missing_reflectance']
# INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
#                     'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
# COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
#                     'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
#                     'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
#                     'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
#                     'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
#                     'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
#                     'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
#                     'lentils']

COLUMNS_TO_STANDARDIZE = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg']
OUTPUT_COLUMN = ['SIF']


# Crop types to look at when analyzing results
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean'] # Temporary function: copy file to target_dir, and return the new file path





def copy_tile(file_path, target_dir):
    new_file_path = os.path.join(target_dir, os.path.basename(file_path))
    if not os.path.exists(new_file_path):
        tile = np.load(file_path, allow_pickle=True).astype(np.float32)
        assert(tile.shape[0] == 43)
        tile = tile[BANDS, :, :]
        assert(tile.shape[0] == 24)
        np.save(new_file_path, tile)
    # np.save(new_file_path + "_continuous", tile[0:12])
    # np.save(new_file_path + "_mask", tile[12:].astype(bool))
    # exit(1)

    return new_file_path

def copy_other_file(file_path, target_dir):
    new_file_path = os.path.join(target_dir, os.path.basename(file_path))
    if not os.path.exists(new_file_path):
        shutil.copy(file_path, new_file_path)
    # if not os.path.exists(new_file_path):
    #     tile = np.load(file_path, allow_pickle=True).astype(dtype)
    #     if np.max(tile) > 255:
    #         print("More than 256 soundings", np.max(tile))
    #     if np.max(tile) > 65535:
    #         print("more than 2^16 soundings", np.max(tile))
    #     np.save(new_file_path, tile)
    return new_file_path




for min_coarse_fraction_valid in MIN_COARSE_FRACTION_VALID_PIXELS:
    # Filter OCO2 tiles
    oco2_metadata = pd.read_csv(OCO2_METADATA_FILE)[BAND_AVERAGE_COLUMNS]
    oco2_metadata = oco2_metadata[(oco2_metadata['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                                    (oco2_metadata['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                                    (oco2_metadata['SIF'] >= MIN_SIF_CLIP)]
    oco2_metadata = oco2_metadata[oco2_metadata[ALL_COVER_COLUMNS].sum(axis=1) >= 0.5]
    oco2_metadata = oco2_metadata.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)[0:250]

    # Read CFIS coarse datapoints - only include CFIS tiles with enough valid pixels
    cfis_coarse_metadata = pd.read_csv(CFIS_COARSE_METADATA_FILE)[COARSE_CFIS_AVERAGE_COLUMNS]
    cfis_coarse_metadata = cfis_coarse_metadata[(cfis_coarse_metadata['fraction_valid'] >= min_coarse_fraction_valid) &
                                                (cfis_coarse_metadata['SIF'] >= MIN_SIF_CLIP) &
                                                (cfis_coarse_metadata['missing_reflectance'] <= MAX_CFIS_CLOUD_COVER)]
    cfis_coarse_metadata = cfis_coarse_metadata[cfis_coarse_metadata[ALL_COVER_COLUMNS].sum(axis=1) >= 0.5]
    cfis_coarse_metadata = cfis_coarse_metadata.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)[0:250]
    selected_cfis_tiles = set(cfis_coarse_metadata['tile_file'])

    TILE_DIR = os.path.join(NEW_DATA_DIR, "tiles")
    if not os.path.exists(TILE_DIR):
        os.makedirs(TILE_DIR)
    oco2_metadata["tile_file"] = oco2_metadata["tile_file"].apply(lambda f: copy_tile(f, TILE_DIR))
    cfis_coarse_metadata["tile_file"] = cfis_coarse_metadata["tile_file"].apply(lambda f: copy_tile(f, TILE_DIR))
    cfis_coarse_metadata["fine_sif_file"] = cfis_coarse_metadata["fine_sif_file"].apply(lambda f: copy_other_file(f, TILE_DIR))
    cfis_coarse_metadata["fine_soundings_file"] = cfis_coarse_metadata["fine_soundings_file"].apply(lambda f: copy_other_file(f, TILE_DIR))
    cfis_coarse_metadata.to_csv(os.path.join(NEW_DATA_DIR, "cfis_coarse_metadata.csv"))
    oco2_metadata.to_csv(os.path.join(NEW_DATA_DIR, "oco2_metadata.csv"))

    for resolution in RESOLUTION_METERS:
        # Read fine metadata at particular resolution, and do initial filtering
        CFIS_FINE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_metadata_' + str(resolution) + 'm.csv')
        cfis_fine_metadata = pd.read_csv(CFIS_FINE_METADATA_FILE)[FINE_CFIS_AVERAGE_COLUMNS]
        cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['num_soundings'] >= min(MIN_FINE_CFIS_SOUNDINGS)) &
                                                (cfis_fine_metadata['fraction_valid'] >= min(MIN_FINE_FRACTION_VALID_PIXELS))]  # Avoid roundoff errors
        cfis_fine_metadata = cfis_fine_metadata[(cfis_fine_metadata['SIF'] >= MIN_SIF_CLIP) &
                                                (cfis_fine_metadata['tile_file'].isin(set(selected_cfis_tiles)))]
        cfis_fine_metadata["tile_file"] = cfis_fine_metadata["tile_file"].apply(lambda f: copy_tile(f, TILE_DIR))

        # print("Resolution", resolution)
        cfis_fine_metadata.to_csv(os.path.join(NEW_DATA_DIR, "cfis_fine_metadata_" + str(resolution) + "m.csv"))
        
