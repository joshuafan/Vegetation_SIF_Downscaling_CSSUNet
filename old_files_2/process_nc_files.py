"""
Visualizes TROPOMI SIF dataset
"""
from scipy.io import netcdf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import xarray as xr

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
FILE_PATH = os.path.join(DATA_DIR, "TROPOMI_SIF/TROPO-SIF_01deg_biweekly_Apr18-Jan20.nc")
START_DATE = '2018-07-08'
END_DATE = '2018-09-15'
PLOT_FOLDER = './exploratory_plots/TROPOMI'

if not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)

MIN_SIF = 0.0
MAX_SIF = 1.0

dataset = xr.open_dataset(FILE_PATH)
print("====================================================")
print("Dataset info:", dataset)
print("====================================================")
print("Times:", dataset.time)
print("====================================================")

# Take average SIF between August 1-16, 2018
data_array = dataset.sif_dc.sel(time=slice(START_DATE, END_DATE)) #.mean(dim='time', skipna=True)
print("SIF array shape", data_array.shape)
print("Array", data_array)

# Plot the distribution of dcSIF (across all time/space)
all_dcSIF = data_array.data.flatten()
all_dcSIF = all_dcSIF[~np.isnan(all_dcSIF)]
# all_dcSIF = all_dcSIF[all_dcSIF > 0]
n, bins, patches = plt.hist(all_dcSIF, 100, facecolor='blue', alpha=0.5)
plt.title('sif_dc values (worldwide, average from ' + START_DATE + ' to ' + END_DATE + ')')
plt.xlabel('n')
plt.ylabel('Number of pixels (across all days)')
plt.savefig(os.path.join(PLOT_FOLDER, 'TROPOMI_sif_dc.png'))
plt.close()

# data_array = dataset.dcSIF.sel(time=START_DATE)
# # bfill('time', 10)
# # SIF should never be negative, so replace any negatives or NaN with 0
# data_array = data_array.fillna(0)
# data_array = data_array.where(data_array >= 0, 0)
# print(data_array)
data_array = data_array.mean(dim='time', skipna=True)
plt.figure(figsize=(21,9))
color_map = plt.get_cmap('YlGn')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
data_array.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', cmap=color_map, vmin=MIN_SIF, vmax=MAX_SIF)
ax.coastlines()
plt.title('sif_dc, average from ' + START_DATE + ' to ' + END_DATE)
plt.savefig(os.path.join(PLOT_FOLDER, 'TROPOMI_sif_dc_' + START_DATE + '_to_' + END_DATE + '_global.png'))
plt.close()

# Example: get SIF at specific lat/long
print("Value at 100W, 45N:", data_array.sel(lat=40.95, lon=-100.15, method='nearest'))
