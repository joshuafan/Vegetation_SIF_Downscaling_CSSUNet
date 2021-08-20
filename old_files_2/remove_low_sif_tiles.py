"""
Creates an eval dataset with low SIF tiles removed (where
SIF is less than MIN_SIF)
"""

import csv
import numpy as np
import os
import pandas as pd
import xarray as xr

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-07-16")
TILE_INFO_FILE = os.path.join(DATASET_DIR, "tile_info_val.csv") #reflectance_cover_to_sif_r2.csv")  #"eval_subtiles.csv")
TILE_AVERAGES_FILE = os.path.join(DATASET_DIR, "tile_averages_val.csv")
#BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
SIF_FILE = os.path.join(DATA_DIR, "TROPOMI_SIF/TROPO-SIF_01deg_biweekly_Apr18-Jan20.nc")
SIF_DATE_RANGE_2018 = pd.date_range(start="2018-08-01", end="2018-08-16")
SIF_DATE_RANGE_2019 = pd.date_range(start="2019-08-01", end="2019-08-16")


tile_metadata = pd.read_csv(TILE_INFO_FILE)
tile_metadata = tile_metadata.dropna()
tile_averages = pd.read_csv(TILE_AVERAGES_FILE)
print('Before removing', len(tile_averages))
tile_averages = tile_averages.dropna()
print('After removing', len(tile_averages))
tile_averages.to_csv(TILE_AVERAGES_FILE, index=False)
exit(0)


sif_mean = np.mean(tile_metadata['SIF'])
sif_std = np.std(tile_metadata['SIF'])
print('SIf mean', sif_mean, 'std', sif_std)
band_statistics = pd.read_csv(BAND_STATISTICS_FILE)
band_statistics.iloc[-1] = {'mean': sif_mean, 'std': sif_std}
band_statistics.to_csv(BAND_STATISTICS_FILE, index=False)

print('After removing nan', len(tile_metadata))
tile_metadata.to_csv(TILE_INFO_FILE, index=False)

exit(0)
# Remove unnamed columns
tile_metadata = tile_metadata.loc[:, ~tile_metadata.columns.str.contains('^Unnamed')]

# Open SIF array
sif_dataset = xr.open_dataset(SIF_FILE)
sif_array_2018 = sif_dataset.sif_dc.sel(time=slice(SIF_DATE_RANGE_2018.date[0], SIF_DATE_RANGE_2018.date[-1])).mean(dim='time')
sif_array_2019 = sif_dataset.sif_dc.sel(time=slice(SIF_DATE_RANGE_2019.date[0], SIF_DATE_RANGE_2019.date[-1])).mean(dim='time')

# Loop through all values and replace SIF with correct value
for index, row in tile_metadata.iterrows():
    if row['date'] == '2018-07-16':
        correct_sif = sif_array_2018.sel(lat=row['lat'], lon=row['lon'], method='nearest').values
    else:
        correct_sif = sif_array_2019.sel(lat=row['lat'], lon=row['lon'], method='nearest').values

    if ((row['lon'] == -93.35 and row['lat'] == 46.95) or (row['lon'] == -91.75 and row['lat'] == 43.65)):
        print('Index', index, 'Date', row['date'], 'Lat', row['lat'], 'Lon', row['lon'], 'SIF', correct_sif)
    tile_metadata.at[index, 'SIF'] = correct_sif
tile_metadata.to_csv(TILE_INFO_FILE, index=False)
exit(0)



EVAL_SUBTILE_AVERAGES_FILE = os.path.join(EVAL_DATASET_DIR, "tile_averages_train.csv") # "eval_subtile_averages.csv")
FILTERED_SUBTILES_FILE = os.path.join(EVAL_DATASET_DIR, "filtered_tile_info_train.csv") # eval_subtiles.csv")
FILTERED_SUBTILE_AVERAGES_FILE = os.path.join(EVAL_DATASET_DIR, "filtered_tile_averages_train.csv")  #"filtered_eval_subtile_averages.csv")
MIN_SIF = 0.2

subtile_metadata = pd.read_csv(EVAL_SUBTILES_FILE)
subtile_averages = pd.read_csv(EVAL_SUBTILE_AVERAGES_FILE)
assert(len(subtile_metadata) == len(subtile_averages))
print('Originally:', str(len(subtile_metadata)), 'subtiles')

# Remove tiles that have a low SIF value
filtered_subtiles = subtile_metadata.loc[subtile_metadata['SIF'] >= MIN_SIF]
filtered_averages = subtile_averages.loc[subtile_averages['SIF'] >= MIN_SIF]
assert(len(filtered_subtiles) == len(filtered_averages))

print('After filtering:', str(len(filtered_subtiles)), 'subtiles')
filtered_subtiles.to_csv(FILTERED_SUBTILES_FILE)
filtered_averages.to_csv(FILTERED_SUBTILE_AVERAGES_FILE)

