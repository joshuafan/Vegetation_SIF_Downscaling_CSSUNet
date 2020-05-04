import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.linear_model import Lasso, Ridge, LinearRegression
import torch
import torchvision.transforms as transforms
import xarray as xr
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from numpy import poly1d

from sif_utils import lat_long_to_index, plot_histogram, get_subtiles_list
import simple_cnn
import tile_transforms

# Taken from https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plot_figures(output_file, figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    #fig = plt.figure(figsize=(8, 20))
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 20))
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title])  #, cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.savefig(output_file)
    plt.close()

def plot_images(image_rows, image_filename_column, output_file):
    images = {}
    for idx, image_row in image_rows.iterrows():
        subtile = np.load(image_row[image_filename_column]).transpose((1, 2, 0))
        title = 'Lat' + str(round(image_row['lat'], 6)) + ', Lon' + str(round(image_row['lon'], 6)) + ' (SIF = ' + str(round(image_row['SIF'], 3)) + ')'
        #print('BLUE: max', np.max(subtile[:, :, 1]), 'min', np.min(subtile[:, :, 1]))
        #print('GREEN: max', np.max(subtile[:, :, 2]), 'min', np.min(subtile[:, :, 2]))
        #print('RED: max', np.max(subtile[:, :, 3]), 'min', np.min(subtile[:, :, 3]))
        images[title] = subtile[:, :, RGB_BANDS] / 2000

    plot_figures(output_file, images, nrows=math.ceil(len(images) / 5), ncols=5)
 

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
TRAIN_DATE = "2018-07-17"
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + TRAIN_DATE)
TILE_AVERAGE_TRAIN_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_averages_train.csv")
TILE_AVERAGE_VAL_FILE = os.path.join(TRAIN_DATASET_DIR, "tile_averages_val.csv")
BAND_STATISTICS_FILE = os.path.join(TRAIN_DATASET_DIR, "band_statistics_train.csv")
TRAIN_TILE_DATASET = os.path.join(TRAIN_DATASET_DIR, "tile_info_train.csv")
ALL_TILE_DATASET = os.path.join(TRAIN_DATASET_DIR, "reflectance_cover_to_sif.csv")


TILES_DIR = os.path.join(DATA_DIR, "tiles_2016-07-17")
LAT = 42.65
LON = -93.35
LAT_LON = 'lat_' + str(LAT) + '_lon_' + str(LON)
TILE_DEGREES = 0.1
eps = TILE_DEGREES / 2
IMAGE_FILE = os.path.join(TILES_DIR, "reflectance_" + LAT_LON + ".npy")
CFIS_SIF_FILE = os.path.join(DATA_DIR, "CFIS/CFIS_201608a_300m.npy")
TROPOMI_SIF_FILE = os.path.join(DATA_DIR, "TROPOMI_SIF/TROPO-SIF_01deg_biweekly_Apr18-Jan20.nc")
TROPOMI_DATE_RANGE = slice("2018-08-01", "2018-08-16")
EVAL_SUBTILE_DATASET = os.path.join(DATA_DIR, "dataset_2016-07-17/eval_subtiles.csv")
RGB_BANDS = [3, 2, 1]
SUBTILE_SIF_MODEL_FILE = os.path.join(DATA_DIR, "models/subtile_sif_simple_cnn_7")
INPUT_CHANNELS = 43
SUBTILE_DIM = 10

# Check if any CUDA devices are visible. If so, pick a default visible device.
# If not, use CPU.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)

# Load subtile SIF model
subtile_sif_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=15, output_dim=1).to(device)
subtile_sif_model.load_state_dict(torch.load(SUBTILE_SIF_MODEL_FILE, map_location=device))

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
transform = transforms.Compose(transform_list)

# Read an input tile
tile = np.load(IMAGE_FILE)
input_tile_non_standardized = torch.tensor(tile, dtype=torch.float).unsqueeze(0).to(device)
subtiles_non_standardized = get_subtiles_list(input_tile_non_standardized, SUBTILE_DIM, device)[0]
subtile_averages = torch.mean(subtiles_non_standardized, dim=(2,3))

# Train linear regression model
train_set = pd.read_csv(TILE_AVERAGE_TRAIN_FILE).dropna()
EXCLUDE_FROM_INPUT = ['lat', 'lon', 'SIF']
INPUT_COLUMNS = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Tair_f_tavg', 
                    'grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils', 'missing_reflectance']

print('input columns', INPUT_COLUMNS)
OUTPUT_COLUMN = ['SIF']


# Obtain linear regression SIF predictions
X_train = train_set[INPUT_COLUMNS]
Y_train = train_set[OUTPUT_COLUMN].values.ravel()
linear_regression = LinearRegression().fit(X_train, Y_train)
#print('First train tile', X_train.iloc[0])
#print('Second train tile', X_train.iloc[1])
#print('subtile averages', subtile_averages[0])
predicted_sifs_linear = linear_regression.predict(subtile_averages).reshape((37, 37))
print('Predicted sifs linear', predicted_sifs_linear)

# Plot map of linear_regression's subtile SIF predictions
plt.imshow(predicted_sifs_linear, cmap='Greens', vmin=0, vmax=1.5)
plt.savefig("exploratory_plots/" + LAT_LON + "_subtile_sif_map_linear.png")
plt.close()

# Standardize input tile
input_tile_standardized = torch.tensor(transform(tile), dtype=torch.float).unsqueeze(0).to(device)
print('Input tile dim', input_tile_standardized.shape)
print('Random pixel', input_tile_standardized[:, :, 8, 8])

# Obtain model's subtile SIF predictions
subtiles_standardized = get_subtiles_list(input_tile_standardized, SUBTILE_DIM, device)[0]  # (batch x num subtiles x bands x subtile_dim x subtile_dim)
print('Subtile shape', subtiles_standardized.shape)
with torch.set_grad_enabled(False):
    predicted_sifs_standardized = subtile_sif_model(subtiles_standardized).detach().numpy()
print('Predicted SIFs standardized', predicted_sifs_standardized.shape)
predicted_sifs_non_standardized = (predicted_sifs_standardized * sif_std + sif_mean).reshape((37, 37))

# Plot map of CNN's subtile SIF predictions
plt.imshow(predicted_sifs_non_standardized, cmap='Greens', vmin=0, vmax=1.5)
plt.savefig("exploratory_plots/" + LAT_LON + "_subtile_sif_map_7.png")
plt.close()




# Display tiles with largest/smallest TROPOMI SIFs
train_metadata = pd.read_csv(TRAIN_TILE_DATASET)
highest_tropomi_sifs = train_metadata.nlargest(25, 'SIF')
plot_images(highest_tropomi_sifs, 'tile_file', 'exploratory_plots/tropomi_sif_high_subtiles.png')
lowest_tropomi_sifs = train_metadata.nsmallest(25, 'SIF')
plot_images(lowest_tropomi_sifs, 'tile_file', 'exploratory_plots/tropomi_sif_low_subtiles.png')
all_metadata = pd.read_csv(ALL_TILE_DATASET)

# Display tiles with largest/smallest CFIS SIFs
eval_metadata = pd.read_csv(EVAL_SUBTILE_DATASET)
highest_cfis_sifs = eval_metadata.nlargest(25, 'SIF')
plot_images(highest_cfis_sifs, 'subtile_file', 'exploratory_plots/cfis_sif_high_subtiles.png')
lowest_cfis_sifs = eval_metadata.nsmallest(25, 'SIF')
plot_images(lowest_cfis_sifs, 'subtile_file', 'exploratory_plots/cfis_sif_low_subtiles.png')

# Open CFIS SIF evaluation dataset
all_cfis_points = np.load(CFIS_SIF_FILE)
print("CFIS points total", all_cfis_points.shape[0])
print('CFIS points with reflectance data', len(eval_metadata))

# Open TROPOMI SIF dataset
tropomi_dataset = xr.open_dataset(TROPOMI_SIF_FILE)
tropomi_array = tropomi_dataset.sif_dc.sel(time=TROPOMI_DATE_RANGE).mean(dim='time')

# For each CFIS SIF point, find TROPOMI SIF of surrounding tile
tropomi_sifs = []  # TROPOMI SIF corresponding to each CFIS point
for i in range(len(eval_metadata)):  # range(cfis_points.shape[0]):
    point_lon = eval_metadata['lon'][i]  # cfis_points[i, 1]
    point_lat = eval_metadata['lat'][i]  # cfis_points[i, 2]
    tropomi_sif = tropomi_array.sel(lat=point_lat, lon=point_lon, method='nearest')
    tropomi_sifs.append(tropomi_sif)

# Plot histogram of CFIS and TROPOMI SIFs
plot_histogram(np.array(all_cfis_points[:, 0]), "sif_distribution_cfis_all.png")
plot_histogram(np.array(eval_metadata['SIF']), "sif_distribution_cfis_filtered.png") #  cfis_points[:, 0])
plot_histogram(np.array(tropomi_sifs), "sif_distribution_tropomi_eval_area.png")
plot_histogram(np.array(train_metadata['SIF']), "sif_distribution_tropomi_train.png")
plot_histogram(np.array(all_metadata['SIF']), "sif_distribution_tropomi_all.png")

# sif_mean = np.mean(train_metadata['SIF'])
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
sif_mean = train_statistics['mean'].values[-1]
print('SIF mean (TROPOMI, train set)', sif_mean)

# Scatterplot of CFIS points (all)
green_cmap = plt.get_cmap('Greens')
plt.figure(figsize=(10, 10))
plt.scatter(all_cfis_points[:, 1], all_cfis_points[:, 2], c=all_cfis_points[:, 0], cmap=green_cmap, vmin=0, vmax=1.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('CFIS points (all)')
plt.savefig('exploratory_plots/cfis_points_all.png')
plt.close()

# Scatterplot of CFIS points (eval)
plt.figure(figsize=(10, 10))
plt.scatter(eval_metadata['lon'], eval_metadata['lat'], c=eval_metadata['SIF'], cmap=green_cmap, vmin=0, vmax=1.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('CFIS points (reflectance data available, eval set)')
plt.savefig('exploratory_plots/cfis_points_filtered.png')
plt.close()

# Scatterplot of CFIS points in the dense area
plt.figure(figsize=(10, 10))
cfis_dense_area = all_cfis_points[all_cfis_points[:, 1] > -94]
plt.scatter(cfis_dense_area[:, 1], cfis_dense_area[:, 2], c=cfis_dense_area[:, 0], cmap=green_cmap, vmin=0, vmax=1.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('CFIS points (area)')
plt.savefig('exploratory_plots/cfis_points_dense.png')
plt.close()

# Scatterplot of CFIS points in the particular area
plt.figure(figsize=(10, 10))
cfis_area = all_cfis_points[(all_cfis_points[:, 1] > LON-eps) & (all_cfis_points[:, 1] < LON+eps) & (all_cfis_points[:, 2] > LAT-eps) & (all_cfis_points[:, 2] < LAT+eps)]
plt.scatter(cfis_area[:, 1], cfis_area[:, 2], c=cfis_area[:, 0], cmap=green_cmap, vmin=0, vmax=1.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('CFIS points (area)')
plt.savefig('exploratory_plots/' + LAT_LON + '_cfis_points.png')
plt.close()

# Convert CFIS into matrix
cfis_tile = np.zeros_like(predicted_sifs_non_standardized)
print('CFIS tile shape', cfis_tile.shape)
top_bound = LAT + eps
left_bound = LON - eps
for p in range(cfis_area.shape[0]):
    res = (TILE_DEGREES / cfis_tile.shape[0], TILE_DEGREES / cfis_tile.shape[1])
    height_idx, width_idx = lat_long_to_index(cfis_area[p, 2], cfis_area[p, 1], top_bound, left_bound, res)
    cfis_tile[height_idx, width_idx] = cfis_area[p, 0] * 1.52

plt.imshow(cfis_tile, cmap='Greens', vmin=0, vmax=1.5)
plt.savefig("exploratory_plots/" + LAT_LON + "_cfis_sifs.png")
plt.close()

# Compare stats!
print('===================== Comparing stats ======================')
print('Ground-truth CFIS SIF for this tile: mean', np.mean(cfis_area[:, 0]), 'std', np.std(cfis_area[:,0]), 'min', np.min(cfis_area[:, 0]), 'max', np.max(cfis_area[:, 0]))
print('Linear predictions for this tile: mean', np.mean(predicted_sifs_linear), 'std', np.std(predicted_sifs_linear), 'min', np.min(predicted_sifs_linear), 'max', np.max(predicted_sifs_linear))
print('CNN predictions for this tile: mean', np.mean(predicted_sifs_non_standardized), 'std', np.std(predicted_sifs_non_standardized), 'min', np.min(predicted_sifs_non_standardized), 'max', np.max(predicted_sifs_non_standardized))
print('TROPOMI SIF for this tile', tropomi_array.sel(lat=LAT, lon=LON, method='nearest'))
print('============================================================')

# Plot TROPOMI vs SIF (and linear regression)
x = eval_metadata['SIF']  # cfis_points[:, 0]
y = tropomi_sifs
coef = np.polyfit(x, y, 1)
print('Linear regression: x=CFIS, y=TROPOMI', coef)
poly1d_fn = np.poly1d(coef) 
plt.plot(x, y, 'bo', x, poly1d_fn(x), '--k')
plt.xlabel('CFIS SIF (small tile, 2016)')
plt.ylabel('TROPOMI SIF (surrounding large tile, 2018)')
plt.title('TROPOMI vs CFIS SIF')
plt.savefig('exploratory_plots/TROPOMI_vs_CFIS_SIF')
plt.close()

# Calculate NRMSE and correlation
nrmse = math.sqrt(mean_squared_error(y, x)) / sif_mean
corr, _ = pearsonr(y, x)
print('NRMSE', round(nrmse, 3))
print('Correlation', round(corr, 3))

# Show example tiles (RGB)
tile = np.load(IMAGE_FILE)
array = tile.transpose((1, 2, 0))
print('Array shape', array.shape)
plt.imshow(array[:, :, RGB_BANDS] / 2000)
plt.savefig("exploratory_plots/" + LAT_LON + "_rgb.png")
plt.close()

fig, axeslist = plt.subplots(ncols=6, nrows=8, figsize=(24, 24))
for band in range(0, 43):
    layer = array[:, :, band]
    axeslist.ravel()[band].imshow(layer, cmap='Greens', vmin=np.min(layer), vmax=np.max(layer))
    axeslist.ravel()[band].set_title('Band ' + str(band))
    axeslist.ravel()[band].set_axis_off()
plt.tight_layout() # optional
plt.savefig('exploratory_plots/' + LAT_LON +'_all_bands.png')
plt.close()




