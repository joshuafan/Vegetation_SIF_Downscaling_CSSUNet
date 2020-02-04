from scipy.io import netcdf
import xarray as xr

FILE_PATH = './datasets/TROPOMI_SIF_2018/TROPO_SIF_08-2018.nc'

# single file
dataset = xr.open_dataset(FILE_PATH)
dcSIF = dataset.relSIF.sel(time='2018-08-02').values
print(dcSIF)