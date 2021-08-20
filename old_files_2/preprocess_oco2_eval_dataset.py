import numpy as np
import pandas as pd

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATE = "2018-08-01"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + DATE)
OCO2_SUBTILES_FILE = os.path.join(DATASET_DIR, "oco2_eval_subtiles.csv")
PROCESSED_OCO2_SUBTILES_FILE = os.path.join(DATASET_DIR, "processed_oco2_eval_subtiles.csv")

PURE_THRESHOLD = 0.7
NUM_NEIGHBORS = 5
crop_types = ['grassland_pasture', 'corn', 'soybean', 'shrubland',
                    'deciduous_forest', 'evergreen_forest', 'spring_wheat', 'developed_open_space',
                    'other_hay_non_alfalfa', 'winter_wheat', 'herbaceous_wetlands',
                    'woody_wetlands', 'open_water', 'alfalfa', 'fallow_idle_cropland',
                    'sorghum', 'developed_low_intensity', 'barren', 'durum_wheat',
                    'canola', 'sunflower', 'dry_beans', 'developed_med_intensity',
                    'millet', 'sugarbeets', 'oats', 'mixed_forest', 'peas', 'barley',
                    'lentils']

oco2_footprints = pd.read_csv(OCO2_SUBTILES_FILE)

crop_frames = []

for crop_type in crop_types:
    crop_rows = oco2_footprints.loc[oco2_footprints[crop_type] > PURE_THRESHOLD]
    crop_lons = crop_rows['center_lon'].to_numpy()
    crop_lats = crop_rows['center_lat'].to_numpy()
    num_rows_crop = len(crop_rows)
    for i in range(len(crop_rows)):
        row = crop_rows.iloc[i]
        diff_lons = crop_lons - row['lon'] * np.ones(num_rows_crop)
        diff_lats = crop_lats - row['lat'] * np.ones(num_rows_crop)
        dist_center = diff_lons ** 2 + diff_lats ** 2
        sorted_indices = np.argsort(dist_center)
        neighbor_indices = sorted_indices[:NUM_NEIGHBORS]
        neighbor_rows = crop_rows.iloc[neighbor_indices]
        crop_rows.at[i, 'SIF'] = np.mean(neighbor_rows['SIF'].to_numpy())
    crop_frames.append(crop_rows)

combined_frame = pd.concat(crop_frames)
pd.to_csv(PROCESSED_OCO2_SUBTILES_FILE)

