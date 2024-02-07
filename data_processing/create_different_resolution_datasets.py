"""
Downsample fine-resolution CFIS SIF (30m) to various resolutions 
(e.g. 30, 90, 150, 300, 600m) for experimentation
"""
import numpy as np
import os
import pandas as pd
import sif_utils
import time
import torch

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")

COARSE_AVERAGE_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_metadata.csv')
FINE_AVERAGE_FILE = os.path.join(CFIS_DIR, 'cfis_fine_metadata.csv')

RES_AVERAGE_COLUMNS = ['fold', 'grid_fold', 'lon', 'lat', 'date', 'tile_file', 'ref_1', 'ref_2', 'ref_3', 'ref_4',
                        'ref_5', 'ref_6', 'ref_7', 'ref_10', 'ref_11', 'Rainf_f_tavg',
                        'SWdown_f_tavg', 'Tair_f_tavg', 'grassland_pasture', 'corn',
                        'soybean', 'shrubland', 'deciduous_forest', 'evergreen_forest',
                        'spring_wheat', 'developed_open_space',
                        'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                        'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                        'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                        'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                        'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                        'lentils', 'missing_reflectance', 'SIF', 'num_soundings', 'coarse_sif', 'fraction_valid']
FINE_PIXELS_PER_COARSE = [1, 3, 5, 10, 20]  # Resolutions to create, in units of 30m pixels
RES_DEGREES = (0.00026949458523585647, 0.00026949458523585647)  # Degrees per Landsat pixel
TILE_SIZE_PIXELS = 100  # Size of output tile, in Landsat pixels
TILE_SIZE_DEGREES = TILE_SIZE_PIXELS * RES_DEGREES[0]
MISSING_REFLECTANCE_IDX = -1

# Read CFIS coarse metadata
cfis_metadata = pd.read_csv(COARSE_AVERAGE_FILE) 

for resolution_pixels in FINE_PIXELS_PER_COARSE:
    resolution_meters = str(30 * resolution_pixels)
    res_metadata = []
    RES_AVERAGES_FILE = os.path.join(CFIS_DIR, 'cfis_metadata_' + resolution_meters + 'm.csv')
    for idx, row in cfis_metadata.iterrows():
        fine_sifs_numpy = np.load(row['fine_sif_file'], allow_pickle=True)
        fine_soundings_numpy = np.load(row['fine_soundings_file'], allow_pickle=True)
        input_tile = np.load(row['tile_file'], allow_pickle=True)
        tile_max_lat = row['lat'] + (TILE_SIZE_DEGREES / 2)
        tile_min_lon = row['lon'] - (TILE_SIZE_DEGREES / 2)

        sif_array = torch.tensor(fine_sifs_numpy.data)
        invalid_sif_mask = torch.tensor(fine_sifs_numpy.mask)
        valid_sif_mask = torch.logical_not(invalid_sif_mask)
        soundings_array = torch.tensor(fine_soundings_numpy)
        before = time.time()
        res_sifs, fraction_valid, res_soundings = sif_utils.downsample_sif(torch.unsqueeze(sif_array, 0),
                                                                            torch.unsqueeze(valid_sif_mask, 0),
                                                                            torch.unsqueeze(soundings_array, 0),
                                                                            resolution_pixels)
        res_sifs = torch.squeeze(res_sifs, 0)
        fraction_valid = torch.squeeze(fraction_valid, 0)
        res_soundings = torch.squeeze(res_soundings, 0)
        before = time.time()
        res_sifs_2, fraction_valid_2, res_soundings_2 = sif_utils.downsample_sif_for_loop(sif_array, valid_sif_mask, soundings_array, resolution_pixels)

        for i in range(res_sifs.shape[0]):
            for j in range(res_sifs.shape[1]):
                if fraction_valid[i, j] <= 0:
                    continue
                assert abs(res_sifs[i, j].item() - res_sifs_2[i, j]) < 1e-5, 'methods should compute same sif'
                assert abs(res_soundings[i, j].item() - res_soundings_2[i, j]) < 1e-5, 'methods should compute same num soundings'
                assert abs(fraction_valid[i, j].item() - fraction_valid_2[i, j]) < 1e-5, 'methods should compute same fraction_valid'
                input_subregion = input_tile[:, i*resolution_pixels:(i+1)*resolution_pixels,
                                                j*resolution_pixels:(j+1)*resolution_pixels]
                invalid_mask_subregion = invalid_sif_mask[i*resolution_pixels:(i+1)*resolution_pixels,
                                                          j*resolution_pixels:(j+1)*resolution_pixels]
                average_input_features = sif_utils.compute_band_averages(input_subregion, invalid_mask_subregion)
                subregion_lat = tile_max_lat - RES_DEGREES[0] * resolution_pixels * (i + 0.5)
                subregion_lon = tile_min_lon + RES_DEGREES[1] * resolution_pixels * (j + 0.5)
                subregion_sif = res_sifs[i, j]
                subregion_row = [row['fold'], row['grid_fold'], subregion_lon, subregion_lat, row['date'], row['tile_file']] + average_input_features.tolist() + \
                                [res_sifs[i, j].item(), round(res_soundings[i, j].item(), 1), row['SIF'], fraction_valid[i, j].item()]
                res_metadata.append(subregion_row)
    res_df = pd.DataFrame(res_metadata, columns=RES_AVERAGE_COLUMNS)
    res_df.to_csv(RES_AVERAGES_FILE)


