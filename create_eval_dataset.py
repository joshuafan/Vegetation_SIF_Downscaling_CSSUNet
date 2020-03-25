import csv
import math
import numpy as np
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sif_utils import lat_long_to_index, plot_histogram

DATE = "2016-08-01"
TILES_DIR = "datasets/tiles_" + DATE
SUBTILES_DIR = "datasets/subtiles_" + DATE  # Directory to output subtiles to
DATASET_DIR = "datasets/dataset_" + DATE

assert os.path.exists(TILES_DIR)
if not os.path.exists(SUBTILES_DIR):
    os.makedirs(SUBTILES_DIR)
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

OUTPUT_CSV_FILE = os.path.join(DATASET_DIR, "eval_subtiles.csv")
TILE_AVERAGE_CSV_FILE = os.path.join(DATASET_DIR, "eval_large_tile_averages.csv")
SUBTILE_AVERAGE_CSV_FILE = os.path.join(DATASET_DIR, "eval_subtile_averages.csv")
CFIS_FILE = "datasets/CFIS/CFIS_201608a_300m.npy"
headers = ["lat", "lon", "SIF", "tile_file", "subtile_file"]
csv_rows = [headers]
TILE_SIZE_DEGREES = 0.1
SUBTILE_SIZE_PIXELS = 10
MAX_FRACTION_MISSING = 0.5

column_names = ['lat', 'lon', 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                'ref_10', 'ref_11', 'corn', 'soybean', 'grassland', 'deciduous_forest',
                'percent_missing', 'SIF']
tile_averages = [column_names]
subtile_averages = [column_names]

#for cfis_file in os.listdir(CFIS_DIR):
#    print('===============================================')
#    print('CFIS file', cfis_file)
#    data_array = xr.open_dataset(os.path.join(CFIS_DIR, cfis_file)).SIF.mean(dim='time')
#    print('data array', data_array)
    #print('Values', data_array.values)
    #print('Coords', data_array.coords)

# Each row is a datapoint. First column is the dc_sif. Second/third columns lon/lat of the grid center.
validation_points = np.load(CFIS_FILE)
print("Validation points shape", validation_points.shape)

# Scatterplot of CFIS points
green_cmap = plt.get_cmap('Greens')
plt.scatter(validation_points[:, 1], validation_points[:, 2], c=validation_points[:, 0], cmap=green_cmap)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Validation points')
plt.savefig('exploratory_plots/validation_points_sif.png')
#print('Longitude extremes', np.max(validation_points[:,1]), np.min(validation_points[validation_points[:, 1] > -110, 1]))
#print('Latitude extremes', np.max(validation_points[:,2]), np.min(validation_points[validation_points[:, 2] > 38.2, 2]))

subtile_reflectance_coverage = []
points_no_reflectance = 0
points_missing_reflectance = 0
points_with_reflectance = 0

for i in range(validation_points.shape[0]):
    sif = validation_points[i, 0]
    if math.isnan(sif):
        continue
    sif *= 1.52  # TROPOMI SIF is roughly 1.52 times CFIS SIF
    point_lon = validation_points[i, 1]
    point_lat = validation_points[i, 2]

    top_bound = math.ceil(point_lat * 10) / 10
    left_bound = math.floor(point_lon * 10) / 10
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

    # Tile dimensions assumed to be (band x lat x lon)
    subtile = large_tile[:, point_lat_idx-eps:point_lat_idx+eps,
                         point_lon_idx-eps:point_lon_idx+eps]
    subtile_filename = SUBTILES_DIR + "/lat_" + str(point_lat) + "_lon_" + str(point_lon) + ".npy"

    # Check if the subtile reflectance data is missing
    print('Subtile shape', subtile.shape)
    fraction_missing_pixels = subtile[-1, :, :].sum() / (subtile.shape[1] * subtile.shape[2])
    print('Missing pixels', fraction_missing_pixels)
    subtile_reflectance_coverage.append(1 - fraction_missing_pixels)

    if fraction_missing_pixels > MAX_FRACTION_MISSING:
        points_missing_reflectance += 1
        continue

    points_with_reflectance += 1

    # Save subtile to file also
    np.save(subtile_filename, subtile)
    csv_rows.append([point_lat, point_lon, sif, large_tile_filename, subtile_filename])

    tile_averages.append([point_lat, point_lon] + np.nanmean(large_tile, axis=(1,2)).tolist() + [sif])
    subtile_averages.append([point_lat, point_lon] + np.nanmean(subtile, axis=(1,2)).tolist() + [sif])

print('=====================================================')
print('Number of points with NO reflectance data', points_no_reflectance)
print('Number of points with MISSING reflectance data', points_missing_reflectance)
print('Number of points WITH reflectance data', points_with_reflectance)

plot_histogram(np.array(subtile_reflectance_coverage), "CFIS_subtile_reflectance_coverage.png")


with open(OUTPUT_CSV_FILE, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in csv_rows:
        csv_writer.writerow(row) 
with open(TILE_AVERAGE_CSV_FILE, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in tile_averages:
        csv_writer.writerow(row) 
with open(SUBTILE_AVERAGE_CSV_FILE, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in subtile_averages:
        csv_writer.writerow(row) 
 
