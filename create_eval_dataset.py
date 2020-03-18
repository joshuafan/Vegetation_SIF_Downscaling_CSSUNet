import csv
import math
import numpy as np
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sif_utils import lat_long_to_index

DATE = "2016-08-01"
TILES_DIR = "datasets/tiles_" + DATE
SUBTILES_DIR = "datasets/subtiles_" + DATE
DATASET_DIR = "datasets/dataset_" + DATE

assert os.path.exists(TILES_DIR)
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

#plt.scatter(validation_points[:, 1], validation_points[:, 2])
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.title('Validation points')
#plt.savefig('exploratory_plots/validation_points.png')
#print('Longitude extremes', np.max(validation_points[:,1]), np.min(validation_points[validation_points[:, 1] > -110, 1]))
#print('Latitude extremes', np.max(validation_points[:,2]), np.min(validation_points[validation_points[:, 2] > 38.2, 2]))


for i in range(validation_points.shape[0]):
    sif = validation_points[i, 0]
    if math.isnan(sif):
        continue
    point_lon = validation_points[i, 1]
    point_lat = validation_points[i, 2]

        #for j in range(len(data_array.values[i])):
            #print('point')
            #print(data_array[i, j])
            #print('lat', data_array[i,j].lat.values)
            #print('lon', data_array[i,j].lon.values)
            #print('SIF', data_array[i,j].values)

            #point_lat = data_array[i,j].lat.values
            #point_lon = data_array[i,j].lon.values
            #sif = data_array[i,j].values
            #if math.isnan(sif):
            #    continue
    top_bound = math.ceil(point_lat * 10) / 10
    left_bound = math.floor(point_lon * 10) / 10
    large_tile_center_lat = round(top_bound - (TILE_SIZE_DEGREES / 2), 2)
    large_tile_center_lon = round(left_bound + (TILE_SIZE_DEGREES / 2), 2)
    large_tile_filename = TILES_DIR + "/reflectance_lat_" + str(large_tile_center_lat) + "_lon_" + str(large_tile_center_lon) + ".npy"
    if not os.path.exists(large_tile_filename):
        print('Needed data file', large_tile_filename, 'does not exist.')
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

    # Save subtile to file also
    np.save(subtile_filename, subtile)
    csv_rows.append([point_lat, point_lon, sif, large_tile_filename, subtile_filename])

    tile_averages.append([point_lat, point_lon] + np.nanmean(large_tile, axis=(1,2)).tolist() + [sif])
    subtile_averages.append([point_lat, point_lon] + np.nanmean(subtile, axis=(1,2)).tolist() + [sif])

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
 
