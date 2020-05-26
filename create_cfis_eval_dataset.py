"""
Constructs validation datasets from CFIS data. 

The main dataset (written to OUTPUT_CSV_FILE) contains
(longitude, latitude, subtile file name, surrounding large tile file name, SIF).

The "tile average" dataset (written to TILE_AVERAGE_CSV_FILE) contains
the band averages for the surrounding large tile.

The "subtile average" dataset (written to SUBTILE_AVERAGE_CSV_FILE) contains
the band averages for the subtile.
"""

import csv
import math
import numpy as np
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sif_utils import lat_long_to_index, plot_histogram, get_top_bound, get_left_bound
DATE = "2016-08-01" #"2016-07-16"
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
TILES_DIR = os.path.join(DATA_DIR, "tiles_" + DATE)
SUBTILES_DIR = os.path.join(DATA_DIR, "subtiles_" + DATE)  # Directory to output subtiles to
DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + DATE)

assert os.path.exists(TILES_DIR)
if not os.path.exists(SUBTILES_DIR):
    os.makedirs(SUBTILES_DIR)
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

OUTPUT_CSV_FILE = os.path.join(DATASET_DIR, "eval_subtiles.csv")
TILE_AVERAGE_CSV_FILE = os.path.join(DATASET_DIR, "eval_large_tile_averages.csv")
CFIS_FILE = os.path.join(DATA_DIR, "CFIS/CFIS_201608a_300m_soundings.npy")
TILE_SIZE_DEGREES = 0.1
SUBTILE_SIZE_PIXELS = 10
MAX_FRACTION_MISSING = 0.1  # If more than this fraction of reflectance pixels is missing, ignore the data point
MIN_SIF = 0.2
MIN_SOUNDINGS = 100
column_names = ['lat', 'lon', 'date', 'tile_file', 'subtile_file', 'num_soundings',
                    'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils', 'missing_reflectance', 'SIF']
csv_rows = [column_names]
tile_averages = [column_names]

# Each row is a datapoint. First column is the dc_sif. Second/third columns lon/lat of the grid center.
validation_points = np.load(CFIS_FILE)
print("Validation points shape", validation_points.shape)

plot_histogram(validation_points[:, 0], "sif_distribution_cfis_all.png", title="CFIS SIF distribution (longitude: -108 to -82, latitude: 38 to 48.7)")
plot_histogram(validation_points[:, 3], "cfis_soundings.png")
print('Total points', validation_points.shape[0])
print('More than 100 soundings', validation_points[validation_points[:, 3] >= 100].shape[0])
print('More than 200 soundings', validation_points[validation_points[:, 3] >= 200].shape[0])
print('More than 500 soundings', validation_points[validation_points[:, 3] >= 500].shape[0])

# Scatterplot of CFIS points
green_cmap = plt.get_cmap('Greens')
plt.scatter(validation_points[:, 1], validation_points[:, 2], c=validation_points[:, 0], cmap=green_cmap)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('All CFIS points')
plt.savefig('exploratory_plots/cfis_points_all.png')
plt.close()
#print('Longitude extremes', np.max(validation_points[:,1]), np.min(validation_points[validation_points[:, 1] > -110, 1]))
#print('Latitude extremes', np.max(validation_points[:,2]), np.min(validation_points[validation_points[:, 2] > 38.2, 2]))

subtile_reflectance_coverage = []
sifs = []

points_no_reflectance = 0  # Number of points outside bounds of reflectance dataset
points_missing_reflectance = 0  # Number of points with missing reflectance data (due to cloud cover)
points_with_reflectance = 0  # Number of points with reflectance data

# Loop through all CFIS data points
for i in range(validation_points.shape[0]):
    sif = validation_points[i, 0]
    if math.isnan(sif):
        continue
    if sif < MIN_SIF:
        continue
    num_soundings = validation_points[i, 3]
    if num_soundings < MIN_SOUNDINGS:
        continue

    #sif *= 1.52  # TROPOMI SIF is roughly 1.52 times CFIS SIF
    point_lon = validation_points[i, 1]
    point_lat = validation_points[i, 2]

    # Compute the box of 0.1 degrees surrounding this point
    top_bound = get_top_bound(point_lat)
    left_bound = get_left_bound(point_lon)
    # Find the 0.1-degree large tile this point is in
    large_tile_center_lat = round(top_bound - (TILE_SIZE_DEGREES / 2), 2)
    large_tile_center_lon = round(left_bound + (TILE_SIZE_DEGREES / 2), 2)
   
    large_tile_filename = TILES_DIR + "/reflectance_lat_" + str(large_tile_center_lat) + "_lon_" + str(large_tile_center_lon) + ".npy"
    if not os.path.exists(large_tile_filename):
        print('Needed data file', large_tile_filename, 'does not exist.')
        points_no_reflectance += 1
        continue

    print('(GOOD) Needed data file', large_tile_filename, 'DOES exist')
    large_tile = np.load(large_tile_filename)
    print("large tile shape", large_tile.shape)
    res = (TILE_SIZE_DEGREES / large_tile.shape[1], TILE_SIZE_DEGREES / large_tile.shape[2])
    print("Resolution:", res)

    # Find the point's index in large tile
    point_lat_idx, point_lon_idx = lat_long_to_index(point_lat, point_lon, top_bound, left_bound, res)
    eps = int(SUBTILE_SIZE_PIXELS / 2)

    # Check if ENTIRE subtile is inside bounds. If not, ignore.
    # TODO This could be unneccesarily throwing away points, if there is another file that contains
    # the missing pixels.
    if point_lat_idx+eps > large_tile.shape[1] or point_lat_idx-eps <= 0 or point_lon_idx+eps > large_tile.shape[2] or point_lon_idx-eps <= 0:
        print("Subtile went beyond edge of large tile")
        continue

    # Tile dimensions assumed to be (band x lat x lon)
    subtile = large_tile[:, point_lat_idx-eps:point_lat_idx+eps,
                         point_lon_idx-eps:point_lon_idx+eps]
    subtile_filename = SUBTILES_DIR + "/lat_" + str(point_lat) + "_lon_" + str(point_lon) + ".npy"

    # Check how much of the subtile's reflectance data is missing (due to cloud cover)
    print('Subtile shape', subtile.shape)
    fraction_missing_pixels = subtile[-1, :, :].sum() / (subtile.shape[1] * subtile.shape[2])
    print('Missing pixels', fraction_missing_pixels)
    subtile_reflectance_coverage.append(1 - fraction_missing_pixels)

    # If too much reflectance data is missing, ignore this point
    if fraction_missing_pixels > MAX_FRACTION_MISSING:
        points_missing_reflectance += 1
        continue
    points_with_reflectance += 1

    # Save subtile to file
    np.save(subtile_filename, subtile)

    # We're constructing 3 datasets. "csv_rows" contains the filename of the surrounding large
    # tile and the subtile. "tile_averages" contains the band averages of the surrounding large
    # tile. "subtile_averages" contains the band averages of the subtile.
    csv_rows.append([point_lat, point_lon, large_tile_filename, subtile_filename, num_soundings] +
                    np.nanmean(subtile, axis=(1,2)).tolist() + [sif])
    tile_averages.append([point_lat, point_lon, large_tile_filename, subtile_filename, num_soundings] +
                         np.nanmean(large_tile, axis=(1,2)).tolist() + [sif])
    sifs.append(sif)

print('=====================================================')
print('Number of points with NO reflectance data', points_no_reflectance)
print('Number of points with MISSING reflectance data', points_missing_reflectance)
print('Number of points WITH reflectance data', points_with_reflectance)


# Plot histograms of reflectance coverage percentage and SIF
plot_histogram(np.array(subtile_reflectance_coverage), "CFIS_subtile_reflectance_coverage.png")

# Write datasets to CSV files
with open(OUTPUT_CSV_FILE, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in csv_rows:
        csv_writer.writerow(row) 
with open(TILE_AVERAGE_CSV_FILE, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in tile_averages:
        csv_writer.writerow(row)
 
