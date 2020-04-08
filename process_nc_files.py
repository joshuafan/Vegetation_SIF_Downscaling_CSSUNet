from scipy.io import netcdf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import xarray as xr

FILE_PATH = './datasets/TROPOMI_SIF/TROPO-SIF_01deg_biweekly_Apr18-Jan20.nc'
START_DATE = '2018-08-01'
END_DATE = '2018-08-15'
PLOT_FOLDER = './datasets/SIF/visualizations/'

MIN_SIF = 0.0
MAX_SIF = 1.0

dataset = xr.open_dataset(FILE_PATH)
print("dcSIF:", dataset)

# Plot the distribution of dcSIF (across all time/space)
all_dcSIF = dataset.sif_dc.values.flatten()
all_dcSIF = all_dcSIF[~np.isnan(all_dcSIF)]
n, bins, patches = plt.hist(all_dcSIF, 100, facecolor='blue', alpha=0.5)
plt.title('dcSIF values, Apr 2018 - Jan 2020 (over all days / locations, excluding NaN values)')
plt.xlabel('dcSIF')
plt.ylabel('Number of pixels (across all days)')

plt.savefig(os.path.join(PLOT_FOLDER, 'all_SIF.png'))
plt.close()

# Select one date (but backfill missing data that's present in the next few days, up to 10 days later)
data_array = dataset.sif_dc.sel(time=slice(START_DATE, END_DATE)).mean(dim='time')
print("SIF array shape", data_array.shape)
print("Array", data_array)

# data_array = dataset.dcSIF.sel(time=START_DATE)
# # bfill('time', 10)
# # SIF should never be negative, so replace any negatives or NaN with 0
# data_array = data_array.fillna(0)
# data_array = data_array.where(data_array >= 0, 0)
# print(data_array)

plt.figure(figsize=(21,9))
color_map = plt.get_cmap('Greens')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
data_array.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', vmin=MIN_SIF,
                                vmax=MAX_SIF, cmap=color_map)
ax.coastlines()
plt.title('dcSIF, average from ' + START_DATE + ' to ' + END_DATE)
plt.savefig(os.path.join(PLOT_FOLDER, 'TROPOMI_SIF_' + START_DATE + '_to_' + END_DATE + '_global.png'))
plt.close()

# Example: get SIF at specific lat/long
print("Value at 100W, 45N:", data_array.sel(lat=40.95, lon=-100.15, method='nearest'))
