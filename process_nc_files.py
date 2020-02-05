from scipy.io import netcdf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr

FILE_PATH = './datasets/TROPOMI_SIF_2018/TROPO_SIF_08-2018.nc'
DATE = '2018-08-01'
PLOT_FOLDER = './datasets/TROPOMI_SIF_2018/visualizations/'

MIN_SIF = 0.0
MAX_SIF = 1.0

dataset = xr.open_dataset(FILE_PATH)

# Plot the distribution of dcSIF (across all time/space)
all_dcSIF = dataset.dcSIF.values.flatten()
all_dcSIF = all_dcSIF[~np.isnan(all_dcSIF)]
n, bins, patches = plt.hist(all_dcSIF, 100, facecolor='blue', alpha=0.5)
plt.title('dcSIF values, August 2018 (over all days / locations, excluding NaN values)')
plt.xlabel('dcSIF')
plt.ylabel('Number of pixels (across all days)')

plt.savefig(os.path.join(PLOT_FOLDER, 'all_dcSIF_08-2018.png'))
plt.close()

# Select one date (but backfill missing data that's present in the next few days, up to 10 days later)
data_array = dataset.dcSIF.sel(time=DATE)
# bfill('time', 10)
# SIF should never be negative, so replace any negatives or NaN with 0
data_array = data_array.fillna(0)
data_array = data_array.where(data_array >= 0, 0)
print(data_array)




plt.figure(figsize=(21,9))
color_map = plt.get_cmap('Greens')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
data_array.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', vmin=MIN_SIF, vmax=MAX_SIF,
                           cmap=color_map)  # , add_colorbar=False)
ax.coastlines()
plt.title('dcSIF for ' + DATE)
plt.savefig(os.path.join(PLOT_FOLDER, 'TROPOMI_SIF_' + DATE + '_global.png'))
plt.close()

#ax.set_ylim([0,90]);
#plt.imshow(dcSIF, interpolation='none')
#plt.show()
