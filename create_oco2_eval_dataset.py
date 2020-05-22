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
TILES_DIR = os.path.join(DATA_DIR, "tiles_2_" + DATE)
# For plotting
patches = []
sifs = []
RES = (0.00026949, 0.00026949)
LARGE_TILE_PIXELS = 371
SUBTILE_PIXELS = 10
MAX_FRACTION_MISSING_SUBTILE = 0.1
MAX_FRACTION_MISSING_OVERALL = 0.1
PURE_THRESHOLD = 0.7

pure_corn_points = 0
pure_soy_points = 0

# Loop through OCO-2 files
for oco2_file in os.listdir(OCO2_DIR):
    # Extract the date from the filename, restrict it to only days between August 1 and 16
    date_string = oco2_file.split('_LtSIF_')[1][4:6]
    if date_string > '16':
       continue

    # Open netCDF file, extract SIF and vertex lat/lon
    dataset = xr.open_dataset(os.path.join(OCO2_DIR, oco2_file))
    print(dataset)
    vertex_lats = dataset.footprint_vertex_latitude.values
    vertex_lons = dataset.footprint_vertex_longitude.values
    sif_757 = dataset.SIF_757nm.values
    sif_771 = dataset.SIF_771nm.values
    assert(vertex_lats.shape[0] == vertex_lons.shape[0])
    assert(vertex_lats.shape[1] == vertex_lons.shape[1])
    assert(vertex_lats.shape[0] == sif_757.shape[0])
    assert(vertex_lats.shape[0] == sif_771.shape[0])

    # Loop through all points
    for i in range(vertex_lats.shape[0]):
        # Restrict region
        if not ((-108 < vertex_lons[i, 0] < -82) and (38 < vertex_lats[i, 0] < 48)):
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
        vertex_list.append(vertex_list[0])
        print("Vertex list", vertex_list)
        p = path.Path(vertex_list)

        min_lon = np.min(vertices[:, 0])
        max_lon = np.max(vertices[:, 0])
        min_lat = np.min(vertices[:, 1])
        max_lat = np.max(vertices[:, 1])
        print('====================================')
        print("Vertices:", vertices)
        #left_bound = math.floor(min_lon, 1)
        #right_bound = round(max_lon, 1)
        #bottom_bound = round(min_lat, 1)
        #top_bound = round(max_lat, 1)
        print("Lon: min", min_lon, "max:", max_lon)
        #print("Lon (after rounding): min", left_bound, "max:", right_bound)
        print("Lat: min", min_lat, "max:", max_lat)
        #print("Lat (after rounding): min", bottom_bound, "max:", top_bound)

        # Figure out which reflectance files to open. For each edge of the bounding box,
        # find the center of the surrounding reflectance large tile.
        left_tile = (math.floor(min_lon * 10) / 10)
        right_tile = (math.floor(max_lon * 10) / 10)
        bottom_tile = (math.ceil(min_lat * 10) / 10)
        top_tile = (math.ceil(max_lat * 10) / 10)
        num_tiles_lon = int((right_tile - left_tile) / 10) + 1
        num_tiles_lat = int((top_tile - bottom_tile) / 10) + 1
        file_left_lons = np.linspace(left_tile, right_tile, num_tiles_lon, endpoint=True)
        file_top_lats = np.linspace(bottom_tile, top_tile, num_tiles_lat, endpoint=True)
        all_subtiles = []
        for file_left_lon in file_left_lons:
            for file_top_lat in file_top_lats:
                file_center_lon = round(file_left_lon + 0.05, 2)
                file_center_lat = round(file_top_lat - 0.05, 2)
                large_tile_filename = TILES_DIR + "/reflectance_lat_" + str(file_center_lat) + "_lon_" + str(file_center_lon) + ".npy"
                if not os.path.exists(large_tile_filename):
                    print('Needed data file', large_tile_filename, 'does not exist.')
                    continue
                print('Large tile filename', large_tile_filename)
                large_tile = np.load(large_tile_filename)

                bottom_idx, left_idx = lat_long_to_index(min_lat, min_lon, file_top_lat, file_left_lon, RES)
                top_idx, right_idx = lat_long_to_index(max_lat, max_lon, file_top_lat, file_left_lon, RES)
                top_idx = max(top_idx, 0)
                bottom_idx = min(bottom_idx, LARGE_TILE_PIXELS)
                left_idx = max(left_idx, 0)
                right_idx = min(right_idx, LARGE_TILE_PIXELS)
                print('Indices: top', top_idx, 'bottom', bottom_idx, 'left', left_idx, 'right', right_idx)
                #num_subtiles_lon = math.floor((right_idx - left_idx) / SUBTILE_PIXELS)
                #num_subtiles_lat = math.floor((bottom_idx - top_idx) / SUBTILE_PIXELS)
                subtile_lon_indices = np.arange(left_idx, right_idx - SUBTILE_PIXELS, SUBTILE_PIXELS) # left_idx + num_subtiles_lon * SUBTILE_PIXELS, num_subtiles_lon, endpoint=False)
                subtile_lat_indices = np.arange(top_idx, bottom_idx - SUBTILE_PIXELS, SUBTILE_PIXELS) # + num_subtiles_lat * SUBTILE_PIXELS, num_subtiles_lat, endpoint=False)
                print("subtile lon indices", subtile_lon_indices)
                print("subtile lat indices", subtile_lat_indices)
                for subtile_lon_idx in subtile_lon_indices:
                    for subtile_lat_idx in subtile_lat_indices:
                        subtile_lon = file_left_lon + RES[1] * subtile_lon_idx
                        subtile_lat = file_top_lat - RES[0] * subtile_lat_idx
                        in_region = p.contains_point((subtile_lon, subtile_lat))
                        print('Subtile lon', subtile_lon, 'lat', subtile_lat, 'In region:', in_region)
                        if not in_region:
                            continue
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

        all_subtiles_numpy = np.stack(all_subtiles)
        print('All subtiles numpy dim', all_subtiles_numpy.shape)
        band_averages = np.mean(all_subtiles_numpy, axis=(0, 2, 3))
        print("Band averages", band_averages)
        if band_averages[-1] > MAX_FRACTION_MISSING_OVERALL:
            continue
        if band_averages[13] > PURE_THRESHOLD:
            pure_corn_points += 1
            print('Pure corn')
        if band_averages[14] > PURE_THRESHOLD:
            pure_soy_points += 1
            print('Pure soy')
        sif = (sif_757[i] + 1.5 * sif_771[i]) / 2
        sifs.append(sif)

print('Pure corn points', pure_corn_points)
print('Pure soy points', pure_soy_points)

sifs = np.array(sifs)
print('SIFs', np.min(sifs), np.max(sifs))

# Plot histogram of SIFs
plot_histogram(sifs, "sif_distribution_oco2.png")

# Plot OCO-2 regions
fig, ax = plt.subplots(figsize=(40, 40))
p = PatchCollection(patches, alpha=1, cmap="YlGn")
p.set_array(sifs)
p.set_clim(0, 2)
ax.add_collection(p)
ax.autoscale()
fig.colorbar(p, ax=ax)
plt.savefig("exploratory_plots/oco2_coverage.png")
plt.close()

