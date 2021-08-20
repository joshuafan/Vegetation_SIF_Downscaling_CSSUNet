import os
import pandas as pd
import random
import sif_utils


DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
CFIS_DIR = os.path.join(DATA_DIR, "CFIS")
OCO2_DIR = os.path.join(DATA_DIR, "OCO2")
CFIS_COARSE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_coarse_metadata.csv')
CFIS_FINE_METADATA_FILE = os.path.join(CFIS_DIR, 'cfis_fine_metadata.csv')
OCO2_METADATA_FILE = os.path.join(OCO2_DIR, 'oco2_metadata_overlap.csv')

cfis_coarse_metadata = pd.read_csv(CFIS_COARSE_METADATA_FILE)
cfis_fine_metadata = pd.read_csv(CFIS_FINE_METADATA_FILE)
oco2_metadata = pd.read_csv(OCO2_METADATA_FILE)

# Divide the region into 0.5x0.5 degree large grid areas. Split them between folds.
GRID_AREA_DEGREES = 0.5
NUM_FOLDS = 5
LATS = list(range(50, 38, -GRID_AREA_DEGREES))
LONS = list(range(-108, -82, GRID_AREA_DEGREES))  # These lat/lons are the UPPER LEFT corner of the large grid areas
large_grid_areas = dict()
for lat in LATS:
    for lon in LONS:
        fold_number = random.randint(0, NUM_FOLDS-1)
        large_grid_areas[(lat, lon)] = fold_number

cfis_coarse_metadata['fold'] = cfis_coarse_metadata.apply(lambda row: sif_utils.determine_split(large_grid_areas, row, GRID_AREA_DEGREES), axis=1)
cfis_fine_metadata['fold'] = cfis_fine_metadata.apply(lambda row: sif_utils.determine_split(large_grid_areas, row, GRID_AREA_DEGREES), axis=1)
oco2_metadata['fold'] = oco2_metadata.apply(lambda row: sif_utils.determine_split(large_grid_areas, row, GRID_AREA_DEGREES), axis=1)
cfis_coarse_metadata.to_csv(CFIS_COARSE_METADATA_FILE)
cfis_fine_metadata.to_csv(CFIS_FINE_METADATA_FILE)
oco2_metadata.to_csv(OCO2_METADATA_FILE)
