"""
Randomly splits tiles into 80% train, 20% validation.
Creates a dataset of tile averages.
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection

INFO_CSV_FILE = "datasets/generated/reflectance_cover_to_sif.csv"
SPLIT_INFO_CSV_FILES = {"train": "datasets/generated/tile_info_train.csv",
                        "val": "datasets/generated/tile_info_val.csv"}
TILE_AVERAGE_CSV_FILES = {"train": "datasets/generated/tile_averages_train.csv",
                          "val": "datasets/generated/tile_averages_val.csv"}


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


tile_metadata = pd.read_csv(INFO_CSV_FILE)

# Split into train/val, and write the split to a file (to ensure that all methos use the same
# train/val split)
train_set, val_set = sklearn.model_selection.train_test_split(tile_metadata, test_size=0.2)
datasets = {"train": train_set, "val": val_set}

for split in ["train", "val"]:
    # Write original tile-info dataset to file, to record split
    datasets[split].to_csv(SPLIT_INFO_CSV_FILES[split])

    # Create a dataset of tile-average values
    csv_rows = []
    column_names = ['lat', 'lon', 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                    'ref_10', 'ref_11', 'corn', 'soybean', 'grassland', 'deciduous_forest',
                    'percent_missing', 'SIF']
    csv_rows.append(column_names)
    dataset = datasets[split]
    for index, row in dataset.iterrows():
        # Tile assumed to be (band x lat x long)
        tile = np.load(row.loc['tile_file'])
        tile_averages = np.mean(tile, axis=(1,2))
        csv_row = [row.loc['lat'], row.loc['lon']] + tile_averages.tolist() + [row.loc['SIF']]
        csv_rows.append(csv_row)

    # Write rows to .csv file
    with open(TILE_AVERAGE_CSV_FILES[split], "w") as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            csv_writer.writerow(row)

    # Plot histogram of each column
    data_array = np.array(csv_rows[1:])  # Ignore header row
    for i in range(len(column_names)):
        print("Column:", column_names[i])
        plot_histogram(data_array[:, i], column_names[i] + "_" + split + ".png")

