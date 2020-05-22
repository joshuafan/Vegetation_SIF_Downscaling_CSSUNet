import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import os
import xarray as xr
from sif_utils import plot_histogram

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
OCO2_DIR = os.path.join(DATA_DIR, "fluo.gps.caltech.edu/data/OCO2/sif_lite_B8100/2018/08")

# For plotting
patches = []
sifs = []

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
        if not((-108 < vertex_lons[i, 0] < -82) and (38 < vertex_lats[i, 0] < 48)):
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
        polygon = Polygon(vertices, True)
        patches.append(polygon)
        sif = (sif_757[i] + 1.5 * sif_771[i]) / 2
        sifs.append(sif)

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

