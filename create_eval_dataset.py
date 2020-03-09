import numpy as np
import os
import pandas as pd
import xarray as xr

CFIS_FOLDER = "datasets/CFIS/CFIS_Campaign_008_USA/L3_netCDF/lowRes/jul_16_2017"
OUTPUT_CSV_FILE = "datasets/generated_subtiles/eval_subtiles.csv"

headers = ["lat", "lon", "SIF", "tile_file", "subtile_file"]
csv_rows = [headers]
TILE_SIZE_DEGREES = 0.1
SUBTILE_SIZE_PIXELS = 8

for cfis_file in os.listdir(CFIS_FOLDER):
    data_array = xr.open_dataset(os.path.join(CFIS_FOLDER, cfis_file)).SIF.mean(dim='time')
    print('Values', data_array.values)
    print('Coords', data_array.coords)
    for i in range(len(data_array.values)):
        for j in range(len(data_array.values[i])):
            print('point')
            print(data_array[i, j])
            print('lat', data_array[i,j].lat.values)
            print('lon', data_array[i,j].lon.values)
            print('SIF', data_array[i,j].values)

            point_lat = data_array[i,j].lat.values
            point_lon = data_array[i,j].lon.values
            sif = data_array[i,j].values
            top_bound = math.ceil(point_lat * 10) / 10
            left_bound = math.floor(point_lon * 10) / 10
            large_tile_center_lat = round(top_bound - (TILE_SIZE_DEGREES / 2), 2)
            large_tile_center_lon = round(left_bound + (TILE_SIZE_DEGREES / 2), 2)
            large_tile_filename = "datasets/generated/reflectance_lat" + str(large_tile_center_lat) + "_lon_" + str(large_tile_center_lon) + ".npy"
            large_tile = np.load(large_tile_filename)
            print("large tile shape", large_tile.shape)
            res = (TILE_SIZE_DEGREES / large_tile.shape[1], TILE_SIZE_DEGREES / large_tile.shape[2])
            print("Resolution:", res)
            
            # Find the point's index in large tile
            point_lat_idx, point_lon_idx = lat_long_to_index(point_lat, point_lon, top_bound, left_bound, res)
            eps = SUBTILE_SIZE_PIXELS / 2

            # Tile dimensions assumed to be (band x lat x lon)
            subtile = large_tile[:, point_lat_idx-eps:point_lat_idx+eps,
                                 point_lon_idx-eps:point_lon_idx+eps]
            subtile_filename = "datasets/generated_subtiles/lat_" + str(point_lat) + "_lon_" + str(point_lon) + ".npy"
            csv_rows.append([point_lat, point_lon, point.SIF, large_tile_filename, subtile_filename])

with open(OUTPUT_CSV_FILE, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in dataset_rows:
        csv_writer.writerow(row) 
 
