import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

COVERS_TO_MASK = [176, 1, 5, 152, 141, 142, 23, 121, 37, 24, 195, 190, 111, 36, 61, 4, 122, 131, 22, 31, 6, 42, 123, 29, 41, 28, 143, 53, 21, 52]  # [176, 152, 1, 5, 141, 142, 23, 121, 37, 190, 195, 111, 36, 24, 61, 0]

# List of crop types (in order of array)
COVERS_TO_MASK = [176, 1, 5, 152, 141,
                  142, 23, 121, 37, 24,
                  195, 190, 111, 36, 61,
                  4, 122, 131, 22, 31,
                  6, 42, 123, 29, 41,
                  28, 143, 53, 21, 52]  # [176, 152, 1, 5, 141, 142, 23, 121, 37, 190, 195, 111, 36, 24, 61, 0]

# Names of cover types (same order,plus none first)
COVER_NAMES = ['none',
               'grassland_pasture', 'corn', 'soybean', 'shrubland', 'deciduous_forest',
               'evergreen_forest', 'spring_wheat', 'developed_open_space', 'other_hay_non_alfalfa', 'winter_wheat',
               'herbaceous_wetlands', 'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
               'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat', 'canola',
               'sunflower', 'dry_beans', 'developed_med_intensity', 'millet', 'sugarbeets',
               'oats', 'mixed_forest', 'peas', 'barley', 'lentils']

# Colors (same order, plus white for zeros)
CDL_COLORS = ["white", 
              "#e8ffbf", "#ffd300", "#267000", "#c6d69e", "#93cc93",
              "#93cc93", "#d8b56b", "#999999", "#a5f28c", "#a57000",
              "#7cafaf", "#7cafaf", "#4970a3", "#ffa5e2", "#bfbf77",
              "#ff9e0a", "#999999", "#ccbfa3", "#896054", "#d1ff00",
              "#ffff00", "#a50000", "#999999", "#700049", "#a800e2",
              "#a05989", "#93cc93", "#54ff00", "#e2007c", "#00ddaf"]




# Plots all bands of the tile, and RGB/CDL bands. Tile is assumed to have shape (CxHxW)
def plot_tile(tile, center_lon, center_lat, tile_size_degrees, tile_description, num_grid_squares=5, decimal_places=2, title='', rgb_bands=[3, 2, 1], cdl_bands=range(12, 42), sif=None):
    eps = tile_size_degrees / 2
    num_ticks = num_grid_squares + 1

    # Plot each band in its own plot
    fig, axeslist = plt.subplots(ncols=6, nrows=8, figsize=(36, 48))
    fig.suptitle('All bands: ' + title)

    for band in range(0, 43):
        layer = tile[band, :, :]
        ax = axeslist.ravel()[band]
        if band >= 12:
            # Binary masks (crop type or missing reflectance) range from 0 to 1
            ax.imshow(layer, cmap='Greens', vmin=0, vmax=1)
        else:
            # Other channels range from -3 to 3
            ax.imshow(layer, cmap='Greens', vmin=-5, vmax=5)

        ax.set_xticks(np.linspace(-0.5, tile.shape[2]-0.5, num_ticks))
        ax.set_yticks(np.linspace(-0.5, tile.shape[1]-0.5, num_ticks))
        ax.set_xticklabels(np.round(np.linspace(center_lon-eps, center_lon+eps, num_ticks), decimal_places))
        ax.set_yticklabels(np.round(np.linspace(center_lat+eps, center_lat-eps, num_ticks), decimal_places))
        ax.grid(color='black', linestyle='-', linewidth=2)
        ax.set_title('Band ' + str(band))


    plt.tight_layout() # optional
    fig.subplots_adjust(top=0.94)
    plt.savefig('exploratory_plots/' + tile_description + '_all_bands.png')
    plt.close()

    # Plot the RGB bands
    array = tile.transpose((1, 2, 0))
    rgb_tile = (array[:, :, rgb_bands] + 4) / 8
    fig, axeslist = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    ax = axeslist.ravel()[0]
    ax.imshow(rgb_tile)
    ax.set_xticks(np.linspace(-0.5, tile.shape[2]-0.5, num_ticks))
    ax.set_yticks(np.linspace(-0.5, tile.shape[1]-0.5, num_ticks))
    ax.set_xticklabels(np.round(np.linspace(center_lon-eps, center_lon+eps, num_ticks), decimal_places))
    ax.set_yticklabels(np.round(np.linspace(center_lat+eps, center_lat-eps, num_ticks), decimal_places))
    ax.grid(color='black', linestyle='-', linewidth=2)
    ax.set_title('RGB bands: ' + title)

    # Plot crop cover
    cover_bands = tile[cdl_bands, :, :]
    assert(len(cover_bands.shape) == 3)
    cover_tile = np.zeros((cover_bands.shape[1], cover_bands.shape[2]))
    for i in range(cover_bands.shape[0]):
        # Reserving 0 for no cover, so add 1
        cover_tile[cover_bands[i, :, :] == 1] = i + 1

    cmap = matplotlib.colors.ListedColormap(CDL_COLORS)
    
    # Weird bounds because if there len(CDL_COLORS) is 31, the range of
    # possible values is [0, 1, ..., 30]. Therefore, the bounds should
    # be [-0.5, 30.5], because if this interval is split into 31 sections,
    # 0 will map to the first section [-0.5, 0.5], 1 will map to the second
    # section [0.5, 1.5], etc 
    ax = axeslist.ravel()[1]
    img = ax.imshow(cover_tile, interpolation='nearest',
                     cmap=cmap, vmin=-0.5, vmax=len(CDL_COLORS)-0.5)
    ax.set_xticks(np.linspace(-0.5, tile.shape[2]-0.5, num_ticks))
    ax.set_yticks(np.linspace(-0.5, tile.shape[1]-0.5, num_ticks))
    ax.set_xticklabels(np.round(np.linspace(center_lon-eps, center_lon+eps, num_ticks), decimal_places))
    ax.set_yticklabels(np.round(np.linspace(center_lat+eps, center_lat-eps, num_ticks), decimal_places))
    ax.grid(color='black', linestyle='-', linewidth=2)
    ticks_loc = np.arange(0, len(CDL_COLORS), 1) #len(COVERS_TO_MASK) / len(CDL_COLORS))
    cb = fig.colorbar(img, ax=axeslist.ravel().tolist(), cmap=cmap)
    cb.set_ticks(ticks_loc)
    cb.set_ticklabels(COVER_NAMES)
    cb.ax.tick_params(labelsize='small')
    ax.set_title('Crop types: ' + title)

    plt.savefig("exploratory_plots/" + tile_description + "_rgb_cdl.png")
    plt.close()
    return rgb_tile


# # Test code
# if __name__ == "__main__":
#     example_cdl = [[[0, 0, 0], [0, 1, 1], [0, 1, 1]],
#                    [[1, 1, 1], [0, 0, 0], [0, 0, 0]]]
#     example_cdl = np.array(example_cdl)
#     print('Example CDL shape', example_cdl.shape)
#     plot_cdl_layers(example_cdl, "exploratory_plots/cdl_test.png")



