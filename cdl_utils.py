import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

def plot_cdl_layers(cover_bands, filename):
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
    img = plt.imshow(cover_tile, interpolation='nearest',
                     cmap=cmap, vmin=-0.5, vmax=len(CDL_COLORS)-0.5)
    ticks_loc = np.arange(0, len(CDL_COLORS), 1) #len(COVERS_TO_MASK) / len(CDL_COLORS))
    print('ticks_loc', ticks_loc)
    cb = plt.colorbar(img, cmap=cmap)
    cb.set_ticks(ticks_loc)
    cb.set_ticklabels(COVER_NAMES)
    cb.ax.tick_params(labelsize='small')

    #cb = plt.colorbar()
    #cb.set_ticks(loc)
    #cb.set_ticklabels(CDL_COLORS)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    example_cdl = [[[0, 0, 0], [0, 1, 1], [0, 1, 1]],
                   [[1, 1, 1], [0, 0, 0], [0, 0, 0]]]
    example_cdl = np.array(example_cdl)
    print('Example CDL shape', example_cdl.shape)
    plot_cdl_layers(example_cdl, "exploratory_plots/cdl_test.png")



