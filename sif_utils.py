import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime


# Given a date string (e.g. "20150608"), returns the year, month, and day-of-year (e.g. June 8 is day #159 of the year)
def parse_date_string(date_string):
    dt = datetime.strptime(date_string, '%Y-%m-%d')
    day_of_year = dt.timetuple().tm_yday
    month = dt.timetuple().tm_mon
    year = dt.timetuple().tm_year
    return year, month, day_of_year


def lat_long_to_index(lat, lon, dataset_top_bound, dataset_left_bound, resolution):
    height_idx = (dataset_top_bound - lat) / resolution[0]
    width_idx = (lon - dataset_left_bound) / resolution[1]
    eps = 1e-6
    return int(height_idx+eps), int(width_idx+eps)


def plot_histogram(column, plot_filename):
    column = column.flatten()
    print('Column', column)
    column = column[~np.isnan(column)]
    print(plot_filename)
    print('Number of datapoints:', len(column))
    print('Mean:', round(np.mean(column), 4))
    print('Std:', round(np.std(column), 4))
    n, bins, patches = plt.hist(column, 20, facecolor='blue', alpha=0.5)
    plt.savefig('exploratory_plots/' + plot_filename)
    plt.close()


# For each tile in the batch, returns a list of subtiles.
# Given a Tensor of tiles, with shape (batch x C x W x H), returns a Tensor of
# shape (batch x SUBTILE x C x subtile_dim x subtile_dim)
def get_subtiles_list(tile, subtile_dim, device):
    batch_size, bands, width, height = tile.shape
    num_subtiles_along_width = int(width / subtile_dim)
    num_subtiles_along_height = int(height / subtile_dim)
    num_subtiles = num_subtiles_along_width * num_subtiles_along_height
    assert(num_subtiles_along_width == 37)
    assert(num_subtiles_along_height == 37)
    subtiles = torch.empty((batch_size, num_subtiles, bands, subtile_dim, subtile_dim), device=device)
    for b in range(batch_size):
        subtile_idx = 0
        for i in range(num_subtiles_along_width):
            for j in range(num_subtiles_along_height):
                subtile = tile[b, :, subtile_dim*i:subtile_dim*(i+1), subtile_dim*j:subtile_dim*(j+1)].to(device)
                subtiles[b, subtile_idx, :, :, :] = subtile
                subtile_idx += 1
    return subtiles


