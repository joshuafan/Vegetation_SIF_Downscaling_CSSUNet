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
OCO2_DIR = os.path.join(DATA_DIR, "fluo.gps.caltech.edu/data/OCO2/sif_lite_B8100/2018/08")
DATE = "2018-08-01" #"2016-07-16"
TILES_DIR = os.path.join(DATA_DIR, "tiles_" + DATE)
DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + DATE)
OCO2_SUBTILES_DIR = os.path.join(DATA_DIR, "oco2_subtiles_" + DATE)  # Directory to output subtiles to
OUTPUT_CSV_FILE = os.path.join(DATASET_DIR, "oco2_eval_subtiles.csv")

if not os.path.exists(OCO2_SUBTILES_DIR):
    os.makedirs(OCO2_SUBTILES_DIR)

dataset_rows = []
dataset_rows.append(["center_lon", "center_lat", "lon_0", "lat_0", "lon_1", "lat_1", "lon_2", "lat_2", "lon_3", "lat_3", "date",
                     "subtiles_file", 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
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
patches = []
point_lons = []
point_lats = []
sifs = []
RES = (0.00026949458523585647, 0.00026949458523585647)
LARGE_TILE_PIXELS = 371
SUBTILE_PIXELS = 10
MAX_FRACTION_MISSING_SUBTILE = 0.1
MAX_FRACTION_MISSING_OVERALL = 0.1
PURE_THRESHOLD = 0.7
MIN_SIF = 0.2

pure_corn_points = 0
pure_soy_points = 0
missing_points = 0

# Loop through OCO-2 files
for oco2_file in os.listdir(OCO2_DIR):
    # Extract the date from the filename, restrict it to only days between August 1 and 16
    oco2_date_string = oco2_file.split('_LtSIF_')[1][0:6]
    day_string = oco2_date_string[4:6]
    print("OCO2 Date string", oco2_date_string)
    if day_string > '16':
        print("Ignore")
        continue
    print("Keep")
    date_string = "20" + oco2_date_string[0:2] + "-" + oco2_date_string[2:4] + "-" + oco2_date_string[4:6]
    print("Standardized date string", date_string)

    # Open netCDF file, extract SIF and vertex lat/lon
    dataset = xr.open_dataset(os.path.join(OCO2_DIR, oco2_file))
    print(dataset)
    vertex_lons = dataset.footprint_vertex_longitude.values
    vertex_lats = dataset.footprint_vertex_latitude.values
    sif_757 = dataset.SIF_757nm.values
    sif_771 = dataset.SIF_771nm.values
    daily_correction_factor = dataset.daily_correction_factor.values
    measurement_mode = dataset.measurement_mode.values
    IGBP_index = dataset.IGBP_index.values
    sounding_id = dataset.sounding_id.values
    orbit_number = dataset.orbit_number.values
    center_lon = dataset.longitude.values
    center_lat = dataset.latitude.values
    
    assert(vertex_lats.shape[0] == vertex_lons.shape[0])
    assert(vertex_lats.shape[1] == vertex_lons.shape[1])
    assert(vertex_lats.shape[0] == sif_757.shape[0])
    assert(vertex_lats.shape[0] == sif_771.shape[0])
    assert(vertex_lats.shape[0] == center_lat.shape[0])
    assert(vertex_lats.shape[0] == center_lon.shape[0])

    # Loop through all points
    for i in range(vertex_lats.shape[0]):
        # Restrict region
        if not ((-108 < vertex_lons[i, 0] < -82) and (38 < vertex_lats[i, 0] < 48)):
            continue
        
        # Only include points where measurement mode is 0
        if measurement_mode[i] != 0:
            continue

        # Read vertices of observation, construct polygon
        vertices = np.zeros((vertex_lats.shape[1], 2))
        vertices[:, 0] = vertex_lons[i, :]
        vertices[:, 1] = vertex_lats[i, :]

        if (vertices[:, 0] < -180).any() or (vertices[:, 0] > 180).any() or (vertices[:, 1] < -90).any() or (vertices[:, 1] > 90).any():
            print("illegal vertices!", vertices)
            continue
        if math.isnan(sif_757[i]) or math.isnan(sif_771[i]):
            print("sif was nan!", sif_757[i], sif_771[i])
            continue
        vertex_list = vertices.tolist()
        vertex_list.append(vertex_list[0])  # Make the path return to starting point
        print("Vertex list", vertex_list)
        parallelogram = path.Path(vertex_list)

        # Compute rectangle that bounds this OCO-2 observation
        min_lon = np.min(vertices[:, 0])
        max_lon = np.max(vertices[:, 0])
        min_lat = np.min(vertices[:, 1])
        max_lat = np.max(vertices[:, 1])
        print('====================================')
        print("Vertices:", vertices)
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
        file_top_lats = np.linspace(min_lat_tile_top, max_lat_tile_top, num_tiles_lat, endpoint=True)
        print("File left lons", file_left_lons)
        print("File top lats", file_top_lats)
        all_subtiles = []
        for file_left_lon in file_left_lons:
            for file_top_lat in file_top_lats:
                # Find what reflectance file to read from
                file_center_lon = round(file_left_lon + 0.05, 2)
                file_center_lat = round(file_top_lat - 0.05, 2)
                large_tile_filename = TILES_DIR + "/reflectance_lat_" + str(file_center_lat) + "_lon_" + str(file_center_lon) + ".npy"
                if not os.path.exists(large_tile_filename):
                    print('Needed data file', large_tile_filename, 'does not exist.')
                    continue
                print('Large tile filename', large_tile_filename)
                large_tile = np.load(large_tile_filename)

                # Find indices of bounding box within this file 
                bottom_idx, left_idx = lat_long_to_index(min_lat, min_lon, file_top_lat, file_left_lon, RES)
                top_idx, right_idx = lat_long_to_index(max_lat, max_lon, file_top_lat, file_left_lon, RES)

                # Note: if the bounding box extends off the edge of the file, clip the indices
                top_idx = max(top_idx, 0)
                bottom_idx = min(bottom_idx, LARGE_TILE_PIXELS)
                left_idx = max(left_idx, 0)
                right_idx = min(right_idx, LARGE_TILE_PIXELS)
                print('Indices: top', top_idx, 'bottom', bottom_idx, 'left', left_idx, 'right', right_idx)
                subtile_lon_indices = np.arange(left_idx, right_idx - SUBTILE_PIXELS + 1, SUBTILE_PIXELS) # left_idx + num_subtiles_lon * SUBTILE_PIXELS, num_subtiles_lon, endpoint=False)
                subtile_lat_indices = np.arange(top_idx, bottom_idx - SUBTILE_PIXELS + 1, SUBTILE_PIXELS) # + num_subtiles_lat * SUBTILE_PIXELS, num_subtiles_lat, endpoint=False)
                print("subtile lon indices", subtile_lon_indices)
                print("subtile lat indices", subtile_lat_indices)
                for subtile_lon_idx in subtile_lon_indices:
                    for subtile_lat_idx in subtile_lat_indices:
                        subtile_center_lon = file_left_lon + RES[1] * (subtile_lon_idx + SUBTILE_PIXELS / 2)
                        subtile_center_lat = file_top_lat - RES[0] * (subtile_lat_idx + SUBTILE_PIXELS / 2)

                        # Check if this point actually falls in the parallelogram
                        in_region = parallelogram.contains_point((subtile_center_lon, subtile_center_lat))
                        print('Subtile lon', subtile_center_lon, 'lat', subtile_center_lat, 'In region:', in_region)
                        if not in_region:
                            continue

                        # Read the sub-tile from the large tile
                        point_lons.append(subtile_center_lon)
                        point_lats.append(subtile_center_lat)
                        subtile = large_tile[:, subtile_lat_idx:subtile_lat_idx+SUBTILE_PIXELS,
                                                subtile_lon_idx:subtile_lon_idx+SUBTILE_PIXELS]
                        if np.mean(subtile[-1, :, :]) > MAX_FRACTION_MISSING_SUBTILE:
                            print('Subtile had too much missing data!')
                            continue
                        else:
                            print('Subtile shape', subtile.shape)
                            all_subtiles.append(subtile)

        if len(all_subtiles) == 0:
            print('No subtiles found??????')
            continue

        # Stack all sub-tiles on top of each other (creating a new axis)
        all_subtiles_numpy = np.stack(all_subtiles)
        print('All subtiles numpy dim', all_subtiles_numpy.shape)

        # Save the sub-tiles array to file
        subtiles_filename = OCO2_SUBTILES_DIR + "/lat_" + str(center_lat[i]) + "_lon_" + str(center_lon[i]) + ".npy"
        # if os.path.exists(subtiles_filename):
        #     print('Uh-oh! duplicate subtile location!', subtiles_filename)
        #     exit(1)
        np.save(subtiles_filename, all_subtiles_numpy)

        # Calculate band averages across this OCO-2 region
        band_averages = np.mean(all_subtiles_numpy, axis=(0, 2, 3))
        print("Band averages", band_averages)
        fraction_missing_reflectance = band_averages[-1]
        sif = daily_correction_factor[i] * (sif_757[i] + 1.5 * sif_771[i]) / 2

        # If there's too much missing data, skip this point
        if fraction_missing_reflectance > MAX_FRACTION_MISSING_OVERALL:
            missing_points += 1
            print('Too much missing')
            continue

        # If SIF is low, we don't care about this point / it's likely to be noisy, so ignore it
        if sif < MIN_SIF:
            continue

        cdl_coverage = np.sum(band_averages[CDL_BANDS])
        if cdl_coverage < MIN_CDL_COVERAGE:
            print('CDL coverage too low:', cdl_coverage)
            print(row.loc['tile_file'])
            continue

        if band_averages[13] > PURE_THRESHOLD:
            pure_corn_points += 1
            print('Pure corn')
        if band_averages[14] > PURE_THRESHOLD:
            pure_soy_points += 1
            print('Pure soy')
        sifs.append(sif)
        polygon = Polygon(vertices, True)
        patches.append(polygon)

        dataset_rows.append([center_lon[i], center_lat[i],
                             vertices[0,0], vertices[0,1], vertices[1,0], vertices[1,1],
                             vertices[2,0], vertices[2,1], vertices[3,0], vertices[3,1],
                             date_string, subtiles_filename] + band_averages.tolist() + 
                             [sif])
        

# Write dataset to file
with open(OUTPUT_CSV_FILE, "w") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    for row in dataset_rows:
        csv_writer.writerow(row) 

print('Missing points', missing_points)
print('Pure corn points', pure_corn_points)
print('Pure soy points', pure_soy_points)

sifs = np.array(sifs)
print('SIFs', np.min(sifs), np.max(sifs))

# Plot histogram of SIFs
plot_histogram(sifs, "sif_distribution_oco2.png")

# Plot OCO-2 regions
fig, ax = plt.subplots(figsize=(10, 10))
p = PatchCollection(patches, alpha=1, cmap="YlGn")
p.set_array(sifs)
p.set_clim(0, 2)
ax.add_collection(p)
ax.autoscale()
fig.colorbar(p, ax=ax)
plt.savefig("exploratory_plots/oco2_coverage.png")
plt.close()

