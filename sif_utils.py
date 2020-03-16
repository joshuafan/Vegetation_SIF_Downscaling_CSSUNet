import numpy as np
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
    return int(height_idx), int(width_idx)


def plot_histogram(column, plot_filename):
    column = column.flatten()
    column = column[~np.isnan(column)]
    print(plot_filename)
    print('Number of datapoints:', len(column))
    print('Mean:', round(np.mean(column), 4))
    print('Std:', round(np.std(column), 4))
    n, bins, patches = plt.hist(column, 20, facecolor='blue', alpha=0.5)
    plt.savefig('exploratory_plots/' + plot_filename)
    plt.close()


