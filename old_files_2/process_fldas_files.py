from scipy.io import netcdf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import xarray as xr
from sif_utils import plot_histogram

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
FILE_PATH = os.path.join(DATA_DIR, 'FLDAS/FLDAS_NOAH01_C_GL_M.A201808.001.nc.SUB.nc4')
START_DATE = '2018-08-01'
END_DATE = '2018-08-16'
PLOT_FOLDER = './exploratory_plots/FLDAS'

if not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)


dataset = xr.open_dataset(FILE_PATH)

# Variables: Rainf_f_tavg, Rainf_f_tavg, Rainf_f_tavg
print("======================================================")
print("Dataset variables:", dataset)
print("======================================================")

# Plot the distribution of temperature (across all time/space)
# all_rainfall = dataset.Rainf_f_tavg.values.flatten()
# all_rainfall = all_rainfall[~np.isnan(all_rainfall)]
# n, bins, patches = plt.hist(all_rainfall, 100, facecolor='blue', alpha=0.5)
# plt.title('Temp values: August 1-16, 2018')
# plt.xlabel('Temp')
# plt.ylabel('Number of pixels')

# plt.savefig(os.path.join(PLOT_FOLDER, 'all_temp.png'))
# plt.close()

# data_array = dataset.dcSIF.sel(time=START_DATE)
# # bfill('time', 10)
# # SIF should never be negative, so replace any negatives or NaN with 0
# data_array = data_array.fillna(0)
# data_array = data_array.where(data_array >= 0, 0)
# print(data_array)


# Select date range
data_array = dataset.Rainf_f_tavg.sel(time=slice(START_DATE, END_DATE)).mean(dim='time')
print("Temp array shape", data_array.shape)
print("Array", data_array)

# Interpolate to higher resolution
new_lat = np.linspace(38, 48.7, 1000)
new_lon = np.linspace(-108, -82, 1000)
reprojected_fldas_dataset = dataset.interp(X=new_lon, Y=new_lat).mean(dim='time')
interpolated_temps = reprojected_fldas_dataset.Rainf_f_tavg

# Compute the min/max values for plotting
interpolated_values = interpolated_temps.values.flatten()
interpolated_values = interpolated_values[~np.isnan(interpolated_values)]
plot_histogram(interpolated_values, "FLDAS/Rainf_f_tavg_region_histogram.png", title="Rainf_f_tavg")
min_value = np.min(interpolated_values)
max_value = np.max(interpolated_values)

# Plot uninterpolated data
plt.figure(figsize=(21,9))
color_map = plt.get_cmap('Blues')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
data_array.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='X', y='Y', vmin=min_value,
                                vmax=max_value, cmap=color_map)
ax.coastlines()
plt.title('Rainf_f_tavg, average from ' + START_DATE + ' to ' + END_DATE)
plt.savefig(os.path.join(PLOT_FOLDER, 'Rainf_f_tavg_' + START_DATE + '_to_' + END_DATE + '_global.png'))
plt.close()

# Plot interpolated data
plt.figure(figsize=(21,9))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
interpolated_temps.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='X', y='Y', vmin=min_value,
                                vmax=max_value, cmap=color_map)
ax.coastlines()
plt.title('(INTERPOLATED) Rainf_f_tavg, average from ' + START_DATE + ' to ' + END_DATE)
plt.savefig(os.path.join(PLOT_FOLDER, 'Rainf_f_tavg_INTERPOLATED_' + START_DATE + '_to_' + END_DATE + '_global.png'))
plt.close()




# Example: get value at specific lat/long
print("Value at 100W, 45N:", data_array.sel(Y=40.95, X=-100.15, method='nearest'))
