import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sif_utils
import torch
import visualization_utils
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.patches as patches


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
OCO2_DIR = os.path.join(DATA_DIR, "OCO2")
PLOT_DIR = os.path.join(DATA_DIR, "exploratory_plots")
CFIS_COARSE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_metadata.csv')
# CFIS_FINE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_fine_metadata.csv')
CFIS_FINE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_metadata_30m.csv')
OCO2_METADATA_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_overlap.csv')
DATES = ["2016-06-15", "2016-08-01"]
NUM_FOLDS = 5
RES = (0.00026949458523585647, 0.00026949458523585647)  # Degrees per Landsat pixel
TILE_SIZE_PIXELS = 100  # Size of output tile, in Landsat pixels
TILE_SIZE_DEGREES = TILE_SIZE_PIXELS * RES[0]
TRAIN_FOLDS = [0, 1, 2]

# OCO-2 filtering
MIN_OCO2_SOUNDINGS = 3
MAX_OCO2_CLOUD_COVER = 0.5
MIN_SIF_CLIP = 0.1

# CFIS filtering
MAX_CFIS_CLOUD_COVER = 0.5
MIN_FINE_CFIS_SOUNDINGS = 1 #30
MIN_COARSE_FRACTION_VALID_PIXELS = 0.1

STATISTICS_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                      'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg',
                      'grassland_pasture', 'corn', 'soybean', 'shrubland',
                      'deciduous_forest', 'evergreen_forest', 'spring_wheat',
                      'developed_open_space', 'other_hay_non_alfalfa', 'winter_wheat',
                      'herbaceous_wetlands', 'woody_wetlands', 'open_water', 'alfalfa',
                      'fallow_idle_cropland', 'sorghum', 'developed_low_intensity',
                      'barren', 'durum_wheat',
                      'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                      'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                      'lentils', 'missing_reflectance', 'SIF']
BAND_STATISTICS_FILE = os.path.join(CFIS_DIR, 'cfis_band_statistics_train.csv')
# COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean', 'deciduous_forest']
COVER_COLUMN_NAMES = ['grassland_pasture', 'corn', 'soybean',
                    'deciduous_forest', 'evergreen_forest', 'developed_open_space',
                    'woody_wetlands', 'open_water', 'alfalfa',
                    'developed_low_intensity', 'developed_med_intensity']

# Read datasets from files
cfis_fine_metadata_df = pd.read_csv(CFIS_FINE_METADATA_FILE)
cfis_coarse_metadata_df = pd.read_csv(CFIS_COARSE_METADATA_FILE)
oco2_metadata_df = pd.read_csv(OCO2_METADATA_FILE)

# Filter OCO-2 dataset
oco2_metadata_df = oco2_metadata_df[(oco2_metadata_df['num_soundings'] >= MIN_OCO2_SOUNDINGS) &
                                    (oco2_metadata_df['missing_reflectance'] <= MAX_OCO2_CLOUD_COVER) &
                                    (oco2_metadata_df['SIF'] >= MIN_SIF_CLIP)]

# Filter noisy coarse pixels
cfis_coarse_metadata_df = cfis_coarse_metadata_df[(cfis_coarse_metadata_df['fraction_valid'] >= MIN_COARSE_FRACTION_VALID_PIXELS) &
                                                  (cfis_coarse_metadata_df['SIF'] >= MIN_SIF_CLIP) &
                                                  (cfis_coarse_metadata_df['missing_reflectance'] <= MAX_CFIS_CLOUD_COVER)]

# Filter noisy fine pixels
cfis_fine_metadata_df = cfis_fine_metadata_df[(cfis_fine_metadata_df['num_soundings'] >= MIN_FINE_CFIS_SOUNDINGS) &
                                              (cfis_fine_metadata_df['SIF'] >= MIN_SIF_CLIP) &
                                              (cfis_fine_metadata_df['tile_file'].isin(set(cfis_coarse_metadata_df['tile_file'])))]

print('Total OCO2', len(oco2_metadata_df))
print('Total coarse CFIS', len(cfis_coarse_metadata_df))
print('Total fine CFIS', len(cfis_fine_metadata_df))



# Read county map
MAP_FILE = "/mnt/beegfs/bulk/mirror/jyf6/datasets/exploratory_plots/States_21basic/geo_export_ac3fb136-4f60-4584-8fd2-5d077cc8bc7e.shp"  # /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/gz_2010_us_050_00_20m/"
state_map = gpd.read_file(MAP_FILE)
# print("State map", state_map)

cfis_tile_points = [Point(xy) for xy in zip(cfis_coarse_metadata_df["lon"], cfis_coarse_metadata_df["lat"])]
cfis_df = gpd.GeoDataFrame(cfis_coarse_metadata_df, geometry=cfis_tile_points)
oco2_tile_points = [Point(xy) for xy in zip(oco2_metadata_df["lon"], oco2_metadata_df["lat"])]
oco2_df = gpd.GeoDataFrame(oco2_metadata_df, geometry=oco2_tile_points)
fig, ax = plt.subplots(figsize=(10.62, 6))
min_lon, max_lon, min_lat, max_lat = -108, -82, 38, 48.7
width = max_lon - min_lon
height = max_lat - min_lat
ax.set_xlim(-125, -66)  #min_lon - 5, max_lon + 5)
ax.set_ylim(24, 50) #min_lat - 5, max_lat + 2)

# Draw rectangle around study region
ax.add_patch(
    patches.Rectangle(
        xy=(min_lon, min_lat),  # point of origin.
        width=width, height=height, linewidth=2,
        color='red', fill=False))

state_map.plot(ax=ax, color='white', edgecolor='lightgray')
oco2_df.plot(ax=ax, marker='o', color='darkgray', markersize=10, zorder=2, label="OCO-2")
cfis_df.plot(ax=ax, marker='o', color='black', markersize=10, zorder=2, label="CFIS")
ax.set_title("CFIS and OCO-2 locations", fontsize=14)
ax.legend(loc="lower right", fontsize=12)
plt.savefig(os.path.join(PLOT_DIR, "locations_cfis_and_oco2.png"))
plt.close()
exit(0)

# Choose arbitrary tile, load data
single_metadata = cfis_coarse_metadata_df.iloc[0]
# print('Single metadata', single_metadata)
true_fine_sifs = np.load(single_metadata.loc['fine_sif_file'], allow_pickle=True)
valid_fine_sif_mask = np.logical_not(true_fine_sifs.mask)
fine_soundings = np.load(single_metadata.loc['fine_soundings_file'], allow_pickle=True)
# print('Fine soundings', fine_soundings)
# print('Valid fine sif mask', valid_fine_sif_mask)
true_eval_sifs, eval_fraction_valid, eval_soundings = sif_utils.downsample_sif(torch.tensor(true_fine_sifs.data).unsqueeze(0),
                                                                               torch.tensor(valid_fine_sif_mask).unsqueeze(0),
                                                                               torch.tensor(fine_soundings).unsqueeze(0), 3)

# Plot soundings/SIF - squeeze
fig, axeslist = plt.subplots(ncols=2, nrows=1, figsize=(24, 12))
visualization_utils.plot_2d_array(fig, axeslist[0], true_eval_sifs.squeeze().numpy(), 'Fine SIF',
                                  single_metadata['lon'], single_metadata['lat'],
                                  TILE_SIZE_DEGREES, colorbar=True, cmap='RdYlGn')
visualization_utils.plot_2d_array(fig, axeslist[1], eval_soundings.squeeze().numpy(), 'Fine soundings',
                                  single_metadata['lon'], single_metadata['lat'],
                                  TILE_SIZE_DEGREES, colorbar=True, cmap='RdYlGn')
tile_description = 'lat_' + str(round(single_metadata['lat'], 4)) + '_lon_' + str(round(single_metadata['lon'], 4)) + '_' \
                    + single_metadata['date']
title = 'Lat' + str(round(single_metadata['lat'], 6)) + ', Lon' + str(round(single_metadata['lon'], 6)) + ' (SIF = ' + str(round(single_metadata['SIF'], 3)) + ')'
plt.suptitle(title)
fig.subplots_adjust(top=0.96)
plt.savefig(os.path.join(PLOT_DIR, tile_description + '_sif_soundings_90m.png'))
plt.close()


# Print scatterplot of soundings in this area, from original CFIS data
MONTH = "Jun"
lons = np.load(os.path.join(CFIS_DIR, "lons_" + MONTH + ".npy"), allow_pickle=True).flatten()
lats = np.load(os.path.join(CFIS_DIR, "lats_" + MONTH + ".npy"), allow_pickle=True).flatten()
sifs = np.load(os.path.join(CFIS_DIR, "dcsif_" + MONTH + ".npy"), allow_pickle=True).flatten()
# print('Lons', lons.shape, 'Lats', lats.shape, 'SIFs', sifs.shape)
eps = TILE_SIZE_DEGREES / 2
indices = (lons > single_metadata['lon']-eps) & (lons < single_metadata['lon']+eps) & \
          (lats > single_metadata['lat']-eps) & (lats < single_metadata['lat']+eps)
# print('Indices', indices)

plt.figure(figsize=(70, 70))
scatterplot = plt.scatter(lons[indices], lats[indices], c=sifs[indices], cmap=plt.get_cmap('RdYlGn'), vmin=0, vmax=1.5)
plt.colorbar(scatterplot)
plt.xlabel('Longitude')
plt.xlim(single_metadata['lon']-eps, single_metadata['lon']+eps)
plt.ylabel('Latitude')
plt.ylim(single_metadata['lat']-eps, single_metadata['lat']+eps)
plt.title('CFIS points, date: ' + single_metadata['date'])
plt.savefig(os.path.join(PLOT_DIR, tile_description + '_soundings_plot.png'))
plt.close()


# Print crop type distribution of each fold
for fold_num in range(NUM_FOLDS):
    oco2_fold = oco2_metadata_df[oco2_metadata_df['fold'] == fold_num]
    cfis_fold = cfis_coarse_metadata_df[cfis_coarse_metadata_df['fold'] == fold_num]
    fine_cfis_fold = cfis_fine_metadata_df[cfis_fine_metadata_df['fold'] == fold_num]
    print('=============== OCO-2 Fold', fold_num, '===============')
    print('Num OCO2 datapoints', len(oco2_fold))
    print(oco2_fold[COVER_COLUMN_NAMES + ['SIF']].mean(axis=0))
    print('=============== CFIS Fold', fold_num, '===============')
    print('Num CFIS datapoints', len(cfis_fold))
    print(cfis_fold[COVER_COLUMN_NAMES + ['SIF']].mean(axis=0))
    print('Num fine CFIS pixels', len(fine_cfis_fold))

# Print histogram for each crop type and date
cfis_coarse_metadata_train = cfis_coarse_metadata_df[cfis_coarse_metadata_df['fold'].isin(TRAIN_FOLDS)].copy()
cfis_fine_metadata_train = cfis_fine_metadata_df[cfis_fine_metadata_df['fold'].isin(TRAIN_FOLDS)].copy()

for date in DATES:
    date_tiles = cfis_coarse_metadata_train[cfis_coarse_metadata_train['date'] == date].copy()
    date_pixels = cfis_fine_metadata_train[cfis_fine_metadata_train['date'] == date].copy()
    for cover_col in COVER_COLUMN_NAMES:
        cover_date_pixels = cfis_fine_metadata_train[(cfis_fine_metadata_train[cover_col] > 0.5) &
                                                     (cfis_fine_metadata_train['date'] == date)].copy()
        print('Cover', cover_col, 'date', date, 'num pixels', len(cover_date_pixels))
        sif_utils.plot_histogram(cover_date_pixels['SIF'].to_numpy(),
                                 "histogram_train_pixels_SIF_" + cover_col + "_" + date + ".png",
                                 title='SIF: ' + cover_col + ', ' + date)        
        sif_utils.plot_histogram(date_pixels['SIF'].to_numpy(),
                                 "histogram_train_pixels_SIF_" + cover_col + "_" + date + "_weighted.png",
                                 title='SIF: ' + cover_col + ', ' + date,
                                 weights=date_pixels[cover_col])
        sif_utils.plot_histogram(date_tiles['SIF'].to_numpy(),
                                 "histogram_train_tiles_SIF_" + cover_col + "_" + date + "_weighted.png",
                                 title='SIF: ' + cover_col + ', ' + date,
                                 weights=date_tiles[cover_col])
exit(0)

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
band_means = train_statistics['mean'].values
band_stds = train_statistics['std'].values


# Plot distribution of each band
for fold in range(NUM_FOLDS):
    cfis_fine_metadata_fold = cfis_fine_metadata_df[cfis_fine_metadata_df['fold'] == fold]
    cfis_coarse_metadata_fold = cfis_coarse_metadata_df[cfis_coarse_metadata_df['fold'] == fold]
    oco2_metadata_fold = oco2_metadata_df[oco2_metadata_df['fold'] == fold]

    for idx, column in enumerate(STATISTICS_COLUMNS):
        if band_stds[idx] == 0:
            continue
        sif_utils.plot_histogram(cfis_fine_metadata_fold[column].to_numpy(),
                                "histogram_pixels_" + column + "_cfis_fold" + str(fold) + ".png",
                                title=column + ' (CFIS pixels, fold ' + str(fold) + ')')
        sif_utils.plot_histogram(cfis_coarse_metadata_fold[column].to_numpy(),
                                "histogram_coarse_" + column + "_cfis_fold" + str(fold) + ".png",
                                title=column + ' (CFIS coarse subregions, fold ' + str(fold) + ')')
        sif_utils.plot_histogram(oco2_metadata_fold[column].to_numpy(),
                                "histogram_coarse_" + column + "_oco2_fold" + str(fold) + ".png",
                                title=column + ' (OCO-2, fold ' + str(fold) + ')')
        standardized_pixel_values = (cfis_fine_metadata_fold[column] - band_means[idx]) / band_stds[idx]
        sif_utils.plot_histogram(standardized_pixel_values.to_numpy(),
                                "histogram_pixels_" + column + "_cfis_fold" + str(fold) + "_std.png",
                                title=column + ' (CFIS pixels, std. by pixel std dev, fold ' + str(fold) + ')')
        standardized_coarse_values = (cfis_coarse_metadata_fold[column] - band_means[idx]) / band_stds[idx]
        sif_utils.plot_histogram(standardized_coarse_values.to_numpy(),
                                "histogram_coarse_" + column + "_cfis_fold" + str(fold) + "_std.png",
                                title=column + ' (CFIS coarse subregions, std. by pixel std dev, fold ' + str(fold) + ')')
        standardized_oco2_values = (oco2_metadata_fold[column] - band_means[idx]) / band_stds[idx]
        sif_utils.plot_histogram(standardized_oco2_values.to_numpy(),
                                "histogram_coarse_" + column + "_oco2_fold" + str(fold) + "_std.png",
                                title=column + ' (OCO-2, std. by pixel std dev, fold ' + str(fold) + 's)')

sif_utils.plot_histogram(cfis_fine_metadata_df['num_soundings'].to_numpy(),
                         "histogram_pixels_num_soundings.png",
                         title='Num soundings (CFIS pixels)')
sif_utils.plot_histogram(cfis_coarse_metadata_df['num_soundings'].to_numpy(),
                         "histogram_coarse_num_soundings.png",
                         title='Num soundings (CFIS coarse subregions)')
sif_utils.plot_histogram(oco2_metadata_df['num_soundings'].to_numpy(),
                         "histogram_oco2_num_soundings.png",
                         title='Num soundings (OCO-2)')
sif_utils.plot_histogram(cfis_coarse_metadata_df['fraction_valid'].to_numpy(),
                         "histogram_coarse_fraction_valid.png",
                         title='Num valid pixels (CFIS coarse subregions)')


# Display tiles with largest/smallest OCO-2 SIFs
highest_oco2_sifs = oco2_metadata_df.nlargest(25, 'SIF')
visualization_utils.plot_rgb_images(highest_oco2_sifs, 'tile_file', os.path.join(PLOT_DIR, 'oco2_sif_high_subtiles.png'))
visualization_utils.plot_band_images(highest_oco2_sifs, 'tile_file', os.path.join(PLOT_DIR, 'oco2_sif_high_subtiles'))
visualization_utils.plot_cdl_layers_multiple(highest_oco2_sifs, 'tile_file', os.path.join(PLOT_DIR, 'oco2_sif_high_subtiles_cdl.png'))
lowest_oco2_sifs = oco2_metadata_df.nsmallest(25, 'SIF')
visualization_utils.plot_rgb_images(lowest_oco2_sifs, 'tile_file', os.path.join(PLOT_DIR, 'oco2_sif_low_subtiles.png'))
visualization_utils.plot_band_images(lowest_oco2_sifs, 'tile_file', os.path.join(PLOT_DIR, 'oco2_sif_low_subtiles'))
visualization_utils.plot_cdl_layers_multiple(lowest_oco2_sifs, 'tile_file', os.path.join(PLOT_DIR, 'oco2_sif_low_subtiles_cdl.png'))

# Display tiles with largest/smallest CFIS SIFs
highest_cfis_sifs = cfis_coarse_metadata_df.nlargest(25, 'SIF')
visualization_utils.plot_rgb_images(highest_cfis_sifs, 'tile_file', os.path.join(PLOT_DIR, 'cfis_sif_high_subtiles.png'))
visualization_utils.plot_band_images(highest_cfis_sifs, 'tile_file', os.path.join(PLOT_DIR, 'cfis_sif_high_subtiles'))
visualization_utils.plot_cdl_layers_multiple(highest_cfis_sifs, 'tile_file', os.path.join(PLOT_DIR, 'cfis_sif_high_subtiles_cdl.png'))
lowest_cfis_sifs = cfis_coarse_metadata_df.nsmallest(25, 'SIF')
visualization_utils.plot_rgb_images(lowest_cfis_sifs, 'tile_file', os.path.join(PLOT_DIR, 'cfis_sif_low_subtiles.png'))
visualization_utils.plot_band_images(lowest_cfis_sifs, 'tile_file', os.path.join(PLOT_DIR, 'cfis_sif_low_subtiles'))
visualization_utils.plot_cdl_layers_multiple(lowest_cfis_sifs, 'tile_file', os.path.join(PLOT_DIR, 'cfis_sif_low_subtiles_cdl.png'))



# Plot geographic distributions of each grid fold
for date in DATES:
    for fold in range(NUM_FOLDS):
        # Plot coarse CFIS tiles
        cfis_coarse_points_date = cfis_coarse_metadata_df[(cfis_coarse_metadata_df['date'] == date) &
                                                          (cfis_coarse_metadata_df['grid_fold'] == fold)]
        cfis_coarse_points_date.head()


        plt.figure(figsize=(30, 10))
        scatterplot = plt.scatter(cfis_coarse_points_date['lon'], cfis_coarse_points_date['lat'],
                                  c=cfis_coarse_points_date['SIF'], cmap=plt.get_cmap('RdYlGn'), vmin=0, vmax=1.5)
        plt.colorbar(scatterplot)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('CFIS points, date: ' + date)
        plt.savefig(os.path.join(PLOT_DIR, 'cfis_coarse_points_' + date + '_fold_' + str(fold) + '.png'))
        plt.close()

        # Plot fine CFIS pixels
        cfis_fine_points_date = cfis_fine_metadata_df[(cfis_fine_metadata_df['date'] == date) &
                                                      (cfis_fine_metadata_df['grid_fold'] == fold)]
        plt.figure(figsize=(30, 10))
        scatterplot = plt.scatter(cfis_fine_points_date['lon'], cfis_fine_points_date['lat'],
                                c=cfis_fine_points_date['SIF'], cmap=plt.get_cmap('RdYlGn'), vmin=0, vmax=1.5)
        plt.colorbar(scatterplot)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('CFIS points, date: ' + date)
        plt.savefig(os.path.join(PLOT_DIR, 'cfis_fine_points_' + date + '_fold_' + str(fold) + '.png'))
        plt.close()

        # Plot OCO-2 tiles
        oco2_points_date = oco2_metadata_df[(oco2_metadata_df['date'] == date) &
                                            (oco2_metadata_df['grid_fold'] == fold)]
        plt.figure(figsize=(30, 10))
        scatterplot = plt.scatter(oco2_points_date['lon'], oco2_points_date['lat'],
                                c=oco2_points_date['SIF'], cmap=plt.get_cmap('RdYlGn'), vmin=0, vmax=1.5)
        plt.colorbar(scatterplot)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('OCO-2 points, date: ' + date)
        plt.savefig(os.path.join(PLOT_DIR, 'oco2_points_' + date + '_fold_' + str(fold) + '.png'))
        plt.close()
