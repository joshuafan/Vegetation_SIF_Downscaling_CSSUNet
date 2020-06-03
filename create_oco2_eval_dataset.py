import csv
import math
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import os
import xarray as xr
from sif_utils import plot_histogram, lat_long_to_index


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
OCO2_FILE = os.path.join(DATA_DIR, "OCO2/oco2_20180801_20180816_1800m.nc")
DATE = "2018-08-01" #"2016-07-16"
TILES_DIR = os.path.join(DATA_DIR, "tiles_" + DATE)
DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + DATE)
OCO2_SUBTILES_DIR = os.path.join(DATA_DIR, "oco2_subtiles_" + DATE)  # Directory to output subtiles to
OUTPUT_CSV_FILE = os.path.join(DATASET_DIR, "oco2_eval_subtiles.csv")

if not os.path.exists(OCO2_SUBTILES_DIR):
    os.makedirs(OCO2_SUBTILES_DIR)

dataset_rows = []
dataset_rows.append(["lon", "lat", "date",
                     "subtile_file", 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils', 'missing_reflectance', "SIF"])



# For plotting
point_lons = []
point_lats = []
point_sifs = []
RES = (0.00026949458523585647, 0.00026949458523585647)
LARGE_TILE_PIXELS = 371
SUBTILE_PIXELS = 60
SUBTILE_DEGREES = RES[0] * SUBTILE_PIXELS
MAX_FRACTION_MISSING_SUBTILE = 0.1
MAX_FRACTION_MISSING_OVERALL = 0.1
PURE_THRESHOLD = 0.7
MIN_SIF = 0.2
INPUT_CHANNELS = 43
CDL_BANDS = list(range(12, 42))
MIN_CDL_COVERAGE = 0.5

pure_corn_points = 0
pure_soy_points = 0
missing_points = 0

dataset = xr.open_dataset(OCO2_FILE)
print(dataset)
dataset = dataset.mean('time')
lons = dataset.lon.values
lats = dataset.lat.values
sifs = dataset.dcSIF.values

plot_histogram(sifs, "sif_distribution_oco2_all.png", title="OCO2 SIF distribution (longitude: -108 to -82, latitude: 38 to 48.7)")
print('Lons', lons.shape)
print('Lats', lats.shape)
print('Sifs', sifs.shape)


points_no_reflectance = 0  # Number of points outside bounds of reflectance dataset
points_missing_reflectance = 0  # Number of points with missing reflectance data (due to cloud cover)
points_with_reflectance = 0  # Number of points with reflectance data

# Loop through all OCO-2 data points
for lon_idx in range(len(lons)):
    for lat_idx in range(len(lats)):
        lon = lons[lon_idx]
        lat = lats[lat_idx]
        sif = sifs[lat_idx, lon_idx]
        if math.isnan(sif):
            continue

        # Compute rectangle that bounds this OCO-2 observation
        min_lon = lon - SUBTILE_DEGREES / 2
        max_lon = lon + SUBTILE_DEGREES / 2
        min_lat = lat - SUBTILE_DEGREES / 2
        max_lat = lat + SUBTILE_DEGREES / 2
        print('====================================')
        print("Lon: min", min_lon, "max:", max_lon)
        print("Lat: min", min_lat, "max:", max_lat)

        # Figure out which reflectance files to open. For each edge of the bounding box,
        # find the center of the surrounding reflectance large tile.
        min_lon_tile_left = (math.floor(min_lon * 10) / 10)
        max_lon_tile_left = (math.floor(max_lon * 10) / 10)
        min_lat_tile_top = (math.ceil(min_lat * 10) / 10)
        max_lat_tile_top = (math.ceil(max_lat * 10) / 10)
        num_tiles_lon = round((max_lon_tile_left - min_lon_tile_left) * 10) + 1
        num_tiles_lat = round((max_lat_tile_top - min_lat_tile_top) * 10) + 1
        file_left_lons = np.linspace(min_lon_tile_left, max_lon_tile_left, num_tiles_lon, endpoint=True)
        file_top_lats = np.linspace(min_lat_tile_top, max_lat_tile_top, num_tiles_lat, endpoint=True)[::-1]  # Go through lats from top to bottom, because indices are numbered from top to bottom
        print("File left lons", file_left_lons)
        print("File top lats", file_top_lats)

        # Because a sub-tile could span multiple files, patch all of the files together
        columns = []
        for file_left_lon in file_left_lons:
            rows = []
            for file_top_lat in file_top_lats:
                # Find what reflectance file to read from
                file_center_lon = round(file_left_lon + 0.05, 2)
                file_center_lat = round(file_top_lat - 0.05, 2)
                large_tile_filename = TILES_DIR + "/reflectance_lat_" + str(file_center_lat) + "_lon_" + str(file_center_lon) + ".npy"
                if not os.path.exists(large_tile_filename):
                    print('Needed data file', large_tile_filename, 'does not exist!')
                    # For now, consider the data for this section as missing
                    missing_tile = np.zeros((INPUT_CHANNELS, LARGE_TILE_PIXELS, LARGE_TILE_PIXELS))
                    missing_tile[-1, :, :] = 1
                    rows.append(missing_tile)
                else:
                    print('Large tile filename', large_tile_filename)
                    large_tile = np.load(large_tile_filename)
                    rows.append(large_tile)

            column = np.concatenate(rows, axis=1)
            columns.append(column)
        
        combined_large_tiles = np.concatenate(columns, axis=2)
        print('All large tiles shape', combined_large_tiles.shape)

        # Find indices of bounding box within this combined large tile 
        top_idx, left_idx = lat_long_to_index(max_lat, min_lon, max_lat_tile_top, min_lon_tile_left, RES)
        bottom_idx = top_idx + SUBTILE_PIXELS
        right_idx = left_idx + SUBTILE_PIXELS
        print('From combined large tile: Top', top_idx, 'Botom', bottom_idx, 'Left', left_idx, 'Right', right_idx)

        # If the selected region (box) goes outside the range of the cover or reflectance dataset, that's a bug!
        if top_idx < 0 or left_idx < 0:
            print("Index was negative!")
            exit(1)
        if (bottom_idx >= combined_large_tiles.shape[1] or right_idx >= combined_large_tiles.shape[2]):
            print("Reflectance index went beyond edge of array!")
            exit(1)

        subtile = combined_large_tiles[:, top_idx:bottom_idx, left_idx:right_idx]

        # Calculate band averages across this sub-tile
        band_averages = np.mean(subtile, axis=(1, 2))
        print("Band averages", band_averages)
        fraction_missing_reflectance = band_averages[-1]

        # If there's too much missing reflectance data, skip this point
        if fraction_missing_reflectance > MAX_FRACTION_MISSING_OVERALL:
            missing_points += 1
            print('Too much missing')
            continue

        # If SIF is low, we don't care about this point / it's likely to be noisy, so ignore it
        if sif < MIN_SIF:
            continue

        # If too much CDL data is missing, skip this point
        cdl_coverage = np.sum(band_averages[CDL_BANDS])
        if cdl_coverage < MIN_CDL_COVERAGE:
            print('CDL coverage too low:', cdl_coverage)
            continue

        if band_averages[13] > PURE_THRESHOLD:
            pure_corn_points += 1
            print('Pure corn')
        if band_averages[14] > PURE_THRESHOLD:
            pure_soy_points += 1
            print('Pure soy')


        # Save the sub-tiles array to file
        subtile_filename = OCO2_SUBTILES_DIR + "/lat_" + str(lat) + "_lon_" + str(lon) + ".npy"
        np.save(subtile_filename, subtile)

        point_lons.append(lon)
        point_lats.append(lat)
        point_sifs.append(sif)

        dataset_rows.append([lon, lat, DATE, subtile_filename] + band_averages.tolist() + 
                             [sif])
        

# Write dataset to file
with open(OUTPUT_CSV_FILE, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in dataset_rows:
        csv_writer.writerow(row) 

print('Missing points', missing_points)
print('Pure corn points', pure_corn_points)
print('Pure soy points', pure_soy_points)

point_sifs = np.array(point_sifs)
print('SIFs', np.min(point_sifs), np.max(point_sifs))

# Plot histogram of SIFs
plot_histogram(point_sifs, "sif_distribution_oco2.png")

# Scatterplot of OCO-2 points
green_cmap = plt.get_cmap('Greens')
plt.scatter(point_lons, point_lats, c=point_sifs, cmap=green_cmap)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('All OCO-2 points')
plt.savefig('exploratory_plots/oco2_points.png')
plt.close()
