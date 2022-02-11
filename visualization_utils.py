import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import os
import sif_utils
import tile_transforms
import torchvision.transforms as transforms

COVERS_TO_MASK = [176, 1, 5, 152, 141, 142, 23, 121, 37, 24, 195, 190, 111, 36, 61, 4, 122, 131, 22, 31, 6, 42, 123, 29, 41, 28, 143, 53, 21, 52]  # [176, 152, 1, 5, 141, 142, 23, 121, 37, 190, 195, 111, 36, 24, 61, 0]

# List of crop types (in order of array)
COVERS_TO_MASK = [176, 1, 5, 152, 141,
                  142, 23, 121, 37, 24,
                  195, 190, 111, 36, 61,
                  4, 122, 131, 22, 31,
                  6, 42, 123, 29, 41,
                  28, 143, 53, 21, 52]  # [176, 152, 1, 5, 141, 142, 23, 121, 37, 190, 195, 111, 36, 24, 61, 0]

# # Names of cover types (same order,plus none first)
# COVER_NAMES = ['none',
#                'grassland_pasture', 'corn', 'soybean', 'shrubland', 'deciduous_forest',
#                'evergreen_forest', 'spring_wheat', 'developed_open_space', 'other_hay_non_alfalfa', 'winter_wheat',
#                'herbaceous_wetlands', 'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
#                'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat', 'canola',
#                'sunflower', 'dry_beans', 'developed_med_intensity', 'millet', 'sugarbeets',
#                'oats', 'mixed_forest', 'peas', 'barley', 'lentils']

# # Colors (same order, plus white for zeros)
# CDL_COLORS = ["white", 
#               "#e8ffbf", "#ffd300", "#267000", "#c6d69e", "#93cc93",
#               "#93cc93", "#d8b56b", "#999999", "#a5f28c", "#a57000",
#               "#7cafaf", "#7cafaf", "#4970a3", "#ffa5e2", "#bfbf77",
#               "#ff9e0a", "#999999", "#ccbfa3", "#896054", "#d1ff00",
#               "#ffff00", "#a50000", "#999999", "#700049", "#a800e2",
#               "#a05989", "#93cc93", "#54ff00", "#e2007c", "#00ddaf"]


# Names of cover types (adjusted for reduced # of crops)
COVER_NAMES = ['none',
               'grassland_pasture', 'corn', 'soybean', 'deciduous_forest',
               'evergreen_forest', 'developed_open_space',
               'woody_wetlands', 'open_water', 'alfalfa',
               'developed_low_intensity', 
               'developed_med_intensity']

# Colors (same order, plus white for zeros)
CDL_COLORS = ["white", 
              "#e8ffbf", "#ffd300", "#267000", "#93cc93",
              "#93cc93", "#999999", 
              "#7cafaf", "#4970a3", "#ffa5e2",
              "#999999",
              "#999999"]

# Taken from https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plot_figures(output_file, figures, nrows = 1, ncols=1, cmap=None):
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
        axeslist.ravel()[ind].imshow(figures[title], vmin=0, vmax=1, cmap=cmap)  #, cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        #axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.savefig(output_file)
    plt.close()

def plot_rgb_images(image_rows, image_filename_column, output_file, RGB_BANDS=[3, 2, 1]):
    images = {}
    for idx, image_row in image_rows.iterrows():
        subtile = np.load(image_row[image_filename_column]).transpose((1, 2, 0))
        title = 'Lat' + str(round(image_row['lat'], 6)) + ', Lon' + str(round(image_row['lon'], 6)) + ' (SIF = ' + str(round(image_row['SIF'], 3)) + ')'
        #print('BLUE: max', np.max(subtile[:, :, 1]), 'min', np.min(subtile[:, :, 1]))
        #print('GREEN: max', np.max(subtile[:, :, 2]), 'min', np.min(subtile[:, :, 2]))
        #print('RED: max', np.max(subtile[:, :, 3]), 'min', np.min(subtile[:, :, 3]))
        images[title] = subtile[:, :, RGB_BANDS] / 1000

    plot_figures(output_file, images, nrows=math.ceil(len(images) / 5), ncols=5)



def plot_band_images(image_rows, image_filename_column, output_file_prefix):
    band_to_max = {4: 5000, 5: 3000, 6: 2000}
    band_images = {}
    for band in band_to_max:
        band_images[band] = {}

    for idx, image_row in image_rows.iterrows():
        subtile = np.load(image_row[image_filename_column]).transpose((1, 2, 0))
        title = 'Lat' + str(round(image_row['lat'], 6)) + ', Lon' + str(round(image_row['lon'], 6)) + ' (SIF = ' + str(round(image_row['SIF'], 3)) + ')'
        for band, max_value in band_to_max.items():
            band_images[band][title] = subtile[:, :, band] / max_value

    for band, images in band_images.items():
        plot_figures(output_file_prefix + '_band_' + str(band) + '.png', images, nrows=math.ceil(len(images) / 5), ncols=5, cmap=plt.get_cmap('YlGn'))


# Note there's a lot of redundant code here
def plot_cdl_layers_multiple(image_rows, image_filename_column, output_file, cdl_bands):
    # Load all tiles and store the CDL bands
    images = {}
    for idx, image_row in image_rows.iterrows():
        cdl_layers = np.load(image_row[image_filename_column])[cdl_bands, :, :]
        title = 'Lat' + str(round(image_row['lat'], 6)) + ', Lon' + str(round(image_row['lon'], 6)) + ' (SIF = ' + str(round(image_row['SIF'], 3)) + ')'
        images[title] = cdl_layers

    # Set up plots
    fig, axeslist = plt.subplots(ncols=5, nrows=math.ceil(len(images) / 5), figsize=(20, 20))
    # Custom CDL colormap
    cmap = matplotlib.colors.ListedColormap(CDL_COLORS)

    for ind, title in enumerate(images):
        # Convert CDL bands into a single layer (each pixel has one number representing the crop type)
        cover_bands = images[title]
        cover_tile = np.zeros((cover_bands.shape[1], cover_bands.shape[2]))
        for i in range(cover_bands.shape[0]):
            # Reserving 0 for no cover, so add 1
            cover_tile[cover_bands[i, :, :] == 1] = i + 1
        img = axeslist.ravel()[ind].imshow(cover_tile, interpolation='nearest',
                     cmap=cmap, vmin=-0.5, vmax=len(CDL_COLORS)-0.5)
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()

    ticks_loc = np.arange(0, len(CDL_COLORS), 1) #len(COVERS_TO_MASK) / len(CDL_COLORS))
    cb = plt.colorbar(img, ax=axeslist[:, 4], cmap=cmap)
    cb.set_ticks(ticks_loc)
    cb.set_ticklabels(COVER_NAMES)
    cb.ax.tick_params(labelsize='small')
    plt.tight_layout() # optional
    plt.savefig(output_file)
    plt.close()


def plot_cdl_layers(tile, title, center_lon, center_lat, tile_size_degrees, plot_file,
                    cdl_bands, num_grid_squares=4, decimal_places=3):
    eps = tile_size_degrees / 2
    num_ticks = num_grid_squares + 1
    cover_bands = tile[cdl_bands, :, :]
    assert(len(cover_bands.shape) == 3)
    cover_tile = np.zeros((cover_bands.shape[1], cover_bands.shape[2]))
    for i in range(cover_bands.shape[0]):
        # Reserving 0 for no cover, so add 1
        cover_tile[cover_bands[i, :, :] == 1] = i + 1

    crop_cmap = matplotlib.colors.ListedColormap(CDL_COLORS)

    # Weird bounds because if len(CDL_COLORS) is 31, the range of
    # possible values is [0, 1, ..., 30]. Therefore, the bounds should
    # be [-0.5, 30.5], because if this interval is split into 31 sections,
    # 0 will map to the first section [-0.5, 0.5], 1 will map to the second
    # section [0.5, 1.5], etc 
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    img = ax.imshow(cover_tile, interpolation='nearest',
                     cmap=crop_cmap, vmin=-0.5, vmax=len(CDL_COLORS)-0.5)
    add_grid_lines(ax, center_lon, center_lat, tile.shape[2], tile.shape[1], tile_size_degrees, num_grid_squares, decimal_places)

    # ax.set_xticks(np.linspace(-0.5, tile.shape[2]-0.5, num_ticks))
    # ax.set_yticks(np.linspace(-0.5, tile.shape[1]-0.5, num_ticks))
    # ax.set_xticklabels(np.round(np.linspace(center_lon-eps, center_lon+eps, num_ticks), decimal_places))
    # ax.set_yticklabels(np.round(np.linspace(center_lat+eps, center_lat-eps, num_ticks), decimal_places))
    # ax.grid(color='gray', linestyle='-', linewidth=2)
    ticks_loc = np.arange(0, len(CDL_COLORS), 1) #len(COVERS_TO_MASK) / len(CDL_COLORS))
    cb = fig.colorbar(img, ax=ax, cmap=crop_cmap)
    cb.set_ticks(ticks_loc)
    cb.set_ticklabels(COVER_NAMES)
    cb.ax.tick_params(labelsize='small')
    ax.set_title('Crop types: ' + title)
    plt.savefig(plot_file)
    plt.close()


# Plots a single 2-D array ("array") to the given axes ("ax"), with grid lines
def plot_2d_array(fig, ax, array, title, center_lon, center_lat, tile_size_degrees,
                  num_grid_squares=4, decimal_places=3, min_feature=None, max_feature=None,
                  colorbar=True, cmap='Greens'):

    pcm = ax.imshow(array, cmap=cmap, vmin=min_feature, vmax=max_feature)
    add_grid_lines(ax, center_lon, center_lat, array.shape[1], array.shape[0], tile_size_degrees, num_grid_squares, decimal_places)
    fig.colorbar(pcm, ax=ax)
    ax.set_title(title)



# For a single tile, plot each band in its own plot
def plot_individual_bands(tile, title, center_lon, center_lat, tile_size_degrees, plot_file, crop_type_start_idx,
                          num_grid_squares=4, decimal_places=3, min_feature=-3, max_feature=3, diverging_cmap=False):

    # Set up a plot for each band
    num_bands = tile.shape[0]
    num_cols, num_rows = 6, math.ceil(num_bands/6)
    fig, axeslist = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=(6*num_cols, 6*num_rows))
    fig.suptitle('All bands: ' + title)

    if diverging_cmap:
        cmap = ""

    for band in range(0, tile.shape[0]): #range(0, 43):
        layer = tile[band, :, :]
        ax = axeslist.ravel()[band]
        title = 'Band' + str(band)

        # Get colorbar bounds for this band
        if band >= crop_type_start_idx:
            # Binary masks (crop type or missing reflectance) range from 0 to 1
            vmin = 0
            vmax = 1
        else:
            if min_feature is not None:
                vmin = min_feature
            else:
                vmin = np.nanmin(layer)
            if max_feature is not None:
                vmax = max_feature
            else:
                vmax = np.nanmax(layer)

        
        # Plot band (channel)
        plot_2d_array(fig, ax, layer, title, center_lon, center_lat, tile_size_degrees,
                      num_grid_squares, decimal_places, vmin, vmax)

    plt.tight_layout() # optional
    fig.subplots_adjust(top=0.94)
    plt.savefig(plot_file)
    plt.close()


# Plot the RGB bands to "ax"
def plot_rgb_bands(tile, title, center_lon, center_lat, tile_size_degrees, ax,
                   rgb_bands=[3, 2, 1], num_grid_squares=4, decimal_places=3):
    # Plot the RGB bands
    array = tile.transpose((1, 2, 0))
    rgb_tile = (array[:, :, rgb_bands] + 2) / 4
    ax.imshow(rgb_tile)
    add_grid_lines(ax, center_lon, center_lat, tile.shape[2], tile.shape[1], tile_size_degrees, num_grid_squares, decimal_places)
    ax.set_title('RGB bands' + title, fontsize=20)


def add_grid_lines(ax, center_lon, center_lat, tile_width, tile_height, tile_size_degrees, num_grid_squares, decimal_places):
    eps = tile_size_degrees / 2
    num_ticks = num_grid_squares + 1
    ax.set_xticks(np.linspace(-0.5, tile_width-0.5, num_ticks))
    ax.set_yticks(np.linspace(-0.5, tile_height-0.5, num_ticks))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # ax.set_xticklabels(np.round(np.linspace(center_lon-eps, center_lon+eps, num_ticks), decimal_places), fontsize=12)
    # ax.set_yticklabels(np.round(np.linspace(center_lat+eps, center_lat-eps, num_ticks), decimal_places), fontsize=12)
    ax.grid(color='blue', linestyle='-', linewidth=2)


# Just plots visualizations for a single tile
def plot_tile(tile, center_lon, center_lat, date, tile_size_degrees,
              tile_description=None, title=None,
              num_grid_squares=4, decimal_places=3, rgb_bands=[3, 2, 1], cdl_bands=range(12, 42),
              plot_dir="/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/exploratory_plots"):

    if tile_description is None:
        tile_description = 'lat_' + str(round(center_lat, 4)) + '_lon_' + str(round(center_lon, 4)) + '_' + date
    if title is None:
        title = 'Lon ' + str(round(center_lon, 4)) + ', Lat ' + str(round(center_lat, 4)) + ', ' + date

    # Plot individual bands
    plot_individual_bands(tile, title, center_lon, center_lat, tile_size_degrees,
                          crop_type_start_idx=cdl_bands[0],
                          plot_file=os.path.join(plot_dir, tile_description + '_all_bands.png'),
                          num_grid_squares=4, decimal_places=3)

    # Plot crop cover
    plot_cdl_layers(tile, title, center_lon, center_lat, tile_size_degrees,
                    plot_file=os.path.join(plot_dir, tile_description + '_cdl.png'),
                    cdl_bands=cdl_bands, num_grid_squares=num_grid_squares, decimal_places=decimal_places)

    # Plot the RGB bands
    ax = plt.gca()
    plot_rgb_bands(tile, title, center_lon, center_lat, tile_size_degrees, ax,
                   rgb_bands=rgb_bands, num_grid_squares=num_grid_squares,
                   decimal_places=decimal_places)

    plt.savefig(os.path.join(plot_dir, tile_description + "_rgb.png"))
    plt.close()


def plot_tile_prediction_only(tile, predicted_sif_tile, valid_mask, center_lon, center_lat, date, tile_size_degrees,
                              tile_description=None, title=None,
                              num_grid_squares=4, decimal_places=3, rgb_bands=[3, 2, 1], cdl_bands=range(12, 42),
                              plot_dir="/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/exploratory_plots"):    
    if tile_description is None:
        tile_description = 'lat_' + str(round(center_lat, 4)) + '_lon_' + str(round(center_lon, 4)) + '_' + date
    if title is None:
        title = 'Lon ' + str(round(center_lon, 4)) + ', Lat ' + str(round(center_lat, 4)) + ', ' + date

    # Plot individual bands
    plot_individual_bands(tile, title, center_lon, center_lat, tile_size_degrees,
                          plot_file=os.path.join(plot_dir, tile_description + '_all_bands.png'),
                          num_grid_squares=4, decimal_places=3)

    # Plot crop cover
    plot_cdl_layers(tile, title, center_lon, center_lat, tile_size_degrees,
                    plot_file=os.path.join(plot_dir, tile_description + '_cdl.png'),
                    cdl_bands=cdl_bands, num_grid_squares=num_grid_squares, decimal_places=decimal_places)

    # Plot the RGB bands
    fig, axeslist = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))
    plot_rgb_bands(tile, title, center_lon, center_lat, tile_size_degrees, axeslist[0],
                   rgb_bands=rgb_bands, num_grid_squares=num_grid_squares,
                   decimal_places=decimal_places)
    
    # Plot SIF predictions
    sif_cmap = plt.get_cmap('RdYlGn')
    sif_cmap.set_bad(color='black')
    predicted_sif_tile[valid_mask == 0] = np.nan
    pcm = axeslist[1].imshow(predicted_sif_tile, cmap=sif_cmap, vmin=0, vmax=1.5)
    add_grid_lines(axeslist[1], center_lon, center_lat, predicted_sif_tile.shape[1], predicted_sif_tile.shape[0], tile_size_degrees, num_grid_squares, decimal_places)
    fig.colorbar(pcm, ax=axeslist, cmap=sif_cmap)
    plt.title('Pixel SIF predictions: ' + title)
    plt.savefig(os.path.join(plot_dir, tile_description + "_predictions.png"))
    plt.close()




# Plots all bands of the tile, and RGB/CDL bands. Tile is assumed to have shape (CxHxW)
# def plot_tile(tile, coarse_sif_tile, fine_sif_tile, coarse_predicted_sifs, fine_predicted_sifs,
#               prediction_methods, center_lon, center_lat, date, tile_size_degrees,
#               num_grid_squares=4, decimal_places=3, rgb_bands=[3, 2, 1],
#               cdl_bands=range(12, 42)):


def outline_masked_areas(mapimg, ax):
    """Outline areas that are set to "True" (1) in the given "mapimg"

    Code taken from https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
    """

    # a vertical line segment is needed, when the pixels next to each other horizontally
    #   belong to diffferent groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates 
    ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])

    # the same is repeated for horizontal segments
    hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add Nones in between segments
    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan,np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))

    # now we transform the list into a numpy array of Nx2 shape
    segments = np.array(l)

    # # now we need to know something about the image which is shown
    # #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    # #   drawn with origin='lower'
    # # with this information we can rescale our points
    # segments[:,0] = x0 + (x1-x0) * segments[:,0] / mapimg.shape[1]
    # segments[:,1] = y0 + (y1-y0) * segments[:,1] / mapimg.shape[0]

    # and now there isn't anything else to do than plot it
    ax.plot(segments[:,0]-0.5, segments[:,1]-0.5, color='black', linewidth=0.5)


# Plot tile information along with true/predicted pixel SIFs.
# "predicted_sif_tiles" and "prediction_methods" should be lists (they can be empty if we have no predictions)
def plot_tile_predictions(tile, tile_description, true_sif_tile, predicted_sif_tiles, valid_mask, non_noisy_mask, prediction_methods,
                          center_lon, center_lat, date, tile_size_degrees,
                          res, soundings_tile=None, num_grid_squares=4, decimal_places=3, rgb_bands=[3, 2, 1],
                          cdl_bands=range(12, 42),
                          plot_dir="/mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/exploratory_plots"): 

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot an augmented version of the tile
    flip_and_rotate_transform = tile_transforms.RandomFlipAndRotate()
    jigsaw_transform = tile_transforms.RandomJigsaw()
    multiplicative_noise_transform = tile_transforms.MultiplicativeGaussianNoiseRaw(bands_to_transform=list(range(0, 9)), standard_deviation=0.2)
    # transform_list_train = [flip_and_rotate_transform] #, jigsaw_transform, multiplicative_noise_transform]
    # train_transform = transforms.Compose(transform_list_train)
    augmented_tile = None # multiplicative_noise_transform(flip_and_rotate_transform(tile))

    # Get indices of non-noisy pixels
    non_noisy_indices = np.argwhere(non_noisy_mask)
    # print('Non-noisy mask', non_noisy_mask[0, :])
    # print('Non-noisy indices', non_noisy_indices)
    # non_noisy_indices = np.array([[0, 99]])


    eps = tile_size_degrees / 2
    num_ticks = num_grid_squares + 1
    title = 'Lon ' + str(round(center_lon, 4)) + ', Lat ' + str(round(center_lat, 4)) + ', ' + date

    # Plot individual bands
    plot_individual_bands(tile, title, center_lon, center_lat, tile_size_degrees,
                          crop_type_start_idx=cdl_bands[0],
                          plot_file=os.path.join(plot_dir, tile_description + '_all_bands.png'),
                          num_grid_squares=4, decimal_places=3)

    # Plot crop cover
    plot_cdl_layers(tile, title, center_lon, center_lat, tile_size_degrees,
                    plot_file=os.path.join(plot_dir, tile_description + '_cdl.png'),
                    cdl_bands=cdl_bands, num_grid_squares=num_grid_squares, decimal_places=decimal_places)

    # Set up subplots
    num_cols = 1 + len(predicted_sif_tiles)
    right_idx = 1 + len(predicted_sif_tiles)  # Index of soundings tile if it exists, otherwise right edge (exclusive) of SIF plots
    if soundings_tile is not None or augmented_tile is not None:
        num_cols += 1

    fig, axeslist = plt.subplots(ncols=num_cols, nrows=2, figsize=(15*num_cols, 22))
    fig.suptitle(title, fontsize=40)

    # Plot the RGB bands
    ax = axeslist[0, 0]
    plot_rgb_bands(tile, "", center_lon, center_lat, tile_size_degrees, ax,
                   rgb_bands=rgb_bands, num_grid_squares=num_grid_squares,
                   decimal_places=decimal_places)

    # Plot (predicted - true) SIF differences for each prediction method
    sif_cmap = plt.get_cmap('RdYlBu')
    sif_cmap.set_bad(color='gray')
    for idx, sif_tile in enumerate(predicted_sif_tiles):
        sif_difference = sif_tile - true_sif_tile
        ax = axeslist[0, idx+1]
        sif_difference[valid_mask == 0] = np.nan
        pcm = ax.imshow(sif_difference, cmap=sif_cmap, vmin=-0.5, vmax=0.5)

        # outline_masked_areas(non_noisy_mask, ax)
        # ax.scatter(x=non_noisy_indices[:, 1], y=non_noisy_indices[:, 0], c='black', s=0.3)
        add_grid_lines(ax, center_lon, center_lat, sif_difference.shape[1], sif_difference.shape[0], tile_size_degrees, num_grid_squares, decimal_places)
        ax.set_title(prediction_methods[idx] + ': difference from ground-truth', fontsize=32)

    # Plot SIF difference colorbar
    cbar = fig.colorbar(pcm, ax=axeslist[0, :right_idx], cmap=sif_cmap)
    cbar.ax.tick_params(labelsize=32)

    # Plot ground-truth SIF
    sif_cmap = plt.get_cmap('BuGn')
    sif_mean = sif_utils.masked_average_numpy(true_sif_tile, valid_mask, dims_to_average=(0, 1))
    ax = axeslist[1, 0]
    true_sif_tile[valid_mask == 0] = np.nan

    pcm = ax.imshow(true_sif_tile, cmap=sif_cmap, vmin=0, vmax=1)
    # ax.scatter(x=non_noisy_indices[:, 1], y=non_noisy_indices[:, 0], c='black', s=0.5)
    add_grid_lines(ax, center_lon, center_lat, true_sif_tile.shape[1], true_sif_tile.shape[0], tile_size_degrees, num_grid_squares, decimal_places)
    ax.set_title('Ground truth (average SIF: ' + str(round(sif_mean, 4)) + ')', fontsize=32)

    # If outline: https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph

    # Plot predicted SIFs
    sif_cmap.set_bad(color='gray')
    for idx, sif_tile in enumerate(predicted_sif_tiles):
        sif_mean = sif_utils.masked_average_numpy(sif_tile, valid_mask, dims_to_average=(0, 1)) # np.sum(sif_tile) / np.count_nonzero(sif_tile)
        ax = axeslist[1, idx+1]
        sif_tile[valid_mask == 0] = np.nan

        pcm = ax.imshow(sif_tile, cmap=sif_cmap, vmin=0, vmax=1)
        # ax.scatter(x=non_noisy_indices[:, 1], y=non_noisy_indices[:, 0], c='black', s=0.5)
        add_grid_lines(ax, center_lon, center_lat, sif_tile.shape[1], sif_tile.shape[0], tile_size_degrees, num_grid_squares, decimal_places)
        ax.set_title(prediction_methods[idx] + ' (average SIF: ' + str(round(sif_mean, 4)) + ')', fontsize=32)

    # Plot SIF colorbar
    cbar = fig.colorbar(pcm, ax=axeslist[1, :right_idx], cmap=sif_cmap)
    cbar.ax.tick_params(labelsize=32) 

    # Plot soundings, if soundings tile was given
    if soundings_tile is not None:
        plot_2d_array(fig, axeslist[0, right_idx], soundings_tile, 'Num soundings', center_lon, center_lat, tile_size_degrees,
                      min_feature=None, max_feature=None, colorbar=True, cmap='Greys')

    if augmented_tile is not None:
        plot_rgb_bands(augmented_tile, "", center_lon, center_lat, tile_size_degrees, axeslist[0, right_idx],
                       rgb_bands=rgb_bands, num_grid_squares=num_grid_squares,
                       decimal_places=decimal_places)

    # fig.subplots_adjust(top=0.94)

    plt.savefig(os.path.join(plot_dir, tile_description + "_predictions.png"))
    plt.close()
    print('plotted', os.path.join(plot_dir, tile_description + "_predictions.png"))

    # # Plot coarse SIF
    # ax = axeslist[1, 0]
    # ax.imshow(coarse_sif_tile, cmap='YlGn', vmin=0.2, vmax=1.5)
    # ax.set_xticks(np.linspace(-0.5, coarse_sif_tile.shape[1]-0.5, num_ticks))
    # ax.set_yticks(np.linspace(-0.5, coarse_sif_tile.shape[0]-0.5, num_ticks))
    # ax.set_xticklabels(np.round(np.linspace(center_lon-eps, center_lon+eps, num_ticks), decimal_places))
    # ax.set_yticklabels(np.round(np.linspace(center_lat+eps, center_lat-eps, num_ticks), decimal_places))
    # ax.grid(color='black', linestyle='-', linewidth=2)
    # ax.set_title('True Coarse SIF')

    # # Plot fine SIF
    # ax = axeslist[1, 1]
    # pcm = ax.imshow(fine_sif_tile, cmap='YlGn', vmin=0.2, vmax=1.5)
    # ax.set_xticks(np.linspace(-0.5, fine_sif_tile.shape[1]-0.5, num_ticks))
    # ax.set_yticks(np.linspace(-0.5, fine_sif_tile.shape[0]-0.5, num_ticks))
    # ax.set_xticklabels(np.round(np.linspace(center_lon-eps, center_lon+eps, num_ticks), decimal_places))
    # ax.set_yticklabels(np.round(np.linspace(center_lat+eps, center_lat-eps, num_ticks), decimal_places))
    # ax.grid(color='black', linestyle='-', linewidth=2)
    # ax.set_title('True Fine SIF')

    # # Plot SIF colorbar
    # fig.colorbar(pcm, ax=axeslist[1:, :], cmap=cmap)

    # for idx, coarse_predictions in enumerate(coarse_predicted_sifs):
    #     # Plot coarse predicted SIF
    #     ax = axeslist[idx+2, 0]
    #     ax.imshow(coarse_predictions, cmap='YlGn', vmin=0.2, vmax=1.5)
    #     ax.set_xticks(np.linspace(-0.5, coarse_predictions.shape[1]-0.5, num_ticks))
    #     ax.set_yticks(np.linspace(-0.5, coarse_predictions.shape[0]-0.5, num_ticks))
    #     ax.set_xticklabels(np.round(np.linspace(center_lon-eps, center_lon+eps, num_ticks), decimal_places))
    #     ax.set_yticklabels(np.round(np.linspace(center_lat+eps, center_lat-eps, num_ticks), decimal_places))
    #     ax.grid(color='black', linestyle='-', linewidth=2)
    #     ax.set_title('Predicted Coarse SIF (' + prediction_methods[idx] + '): ' + title)

    #     # Plot fine predicted SIF
    #     ax = axeslist[idx+2, 1]
    #     fine_predictions = fine_predicted_sifs[idx]
    #     ax.imshow(fine_predictions, cmap='YlGn', vmin=0.2, vmax=1.5)
    #     ax.set_xticks(np.linspace(-0.5, fine_predictions.shape[1]-0.5, num_ticks))
    #     ax.set_yticks(np.linspace(-0.5, fine_predictions.shape[0]-0.5, num_ticks))
    #     ax.set_xticklabels(np.round(np.linspace(center_lon-eps, center_lon+eps, num_ticks), decimal_places))
    #     ax.set_yticklabels(np.round(np.linspace(center_lat+eps, center_lat-eps, num_ticks), decimal_places))
    #     ax.grid(color='black', linestyle='-', linewidth=2)
    #     ax.set_title('Predicted Fine SIF (' + prediction_methods[idx] + '): ' + title)
    # plt.tight_layout()


# # Test code
# if __name__ == "__main__":
#     example_cdl = [[[0, 0, 0], [0, 1, 1], [0, 1, 1]],
#                    [[1, 1, 1], [0, 0, 0], [0, 0, 0]]]
#     example_cdl = np.array(example_cdl)
#     print('Example CDL shape', example_cdl.shape)
#     plot_cdl_layers(example_cdl, "exploratory_plots/cdl_test.png")



