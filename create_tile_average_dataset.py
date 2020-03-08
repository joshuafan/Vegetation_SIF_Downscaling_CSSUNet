import numpy as np
import pandas as pd

INFO_CSV_FILE = "datasets/generated/reflectance_cover_to_sif.csv"
SPLIT_INFO_CSV_FILES = ["datasets/generated/tile_info_train.csv", "datasets/generated/tile_info_val.csv"]
TILE_AVERAGE_CSV_FILES = ["datasets/generated/tile_averages_train.csv", "datasets/generated/tile_averages_val.csv"]


tile_metadata = pd.read_csv(INFO_CSV_FILE)

# Split into train/val, and write the split to a file (to ensure that all methos use the same
# train/val split)
train_set, val_set = pd.train_test_split(tile_metadata, test_size=0.2)

datasets = [train_set, val_set]
for i in range(len(datasets)):
    # Write original tile-info dataset to file, to record split
    datasets[i].to_csv(SPLIT_INFO_CSV_FILES[i])

    # Create a dataset of tile-average values
    csv_rows = []
    csv_rows.append(['lat', 'lon', 'ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7',
                     'ref_10', 'ref_11', 'corn', 'soybean', 'grassland', 'deciduous_forest',
                     'percent_missing', 'SIF'])
    dataset = datasets[i]
    for index, row in dataset.iterrows():
        tile = np.load(row.loc['tile_file'])
        tile_averages = np.mean(tile, axis=(1,2))
        print('tile averages shape', tile_averages.shape)
        exit(1)
        csv_row = [row.loc['lat'], row.loc['lon']] + np.tolist(tile_averages) + [row.loc['SIF']]
        csv_rows.append(csv_row)


    with open(TILE_AVERAGE_CSV_FILES[i], "w") as output_csv_file:
        csv_writer = csv_writer(output_csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            csv_writer.writerow(row)

