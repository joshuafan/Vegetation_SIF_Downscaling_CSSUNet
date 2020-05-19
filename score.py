import os
#import pickle
import json
import numpy as np
import pandas as pd
import time
import torch
import torchvision.transforms as transforms

import tile_transforms
import simple_cnn

MODEL_FILENAME = "subtile_sif_simple_cnn_9"
BAND_STATISTICS_FILENAME = "band_statistics_train.csv"
INPUT_CHANNELS = 43
REDUCED_CHANNELS = 30
SUBTILE_DIM = 10

# For each tile in the batch, returns a list of subtiles.
# Given a Tensor of tiles, with shape (batch x C x H x W), returns a Tensor of
# shape (batch x SUBTILE x C x subtile_dim x subtile_dim)
def get_subtiles_list(tile, subtile_dim, device):
    batch_size, bands, height, width = tile.shape
    num_subtiles_along_height = int(height / subtile_dim)
    num_subtiles_along_width = int(width / subtile_dim)
    num_subtiles = num_subtiles_along_height * num_subtiles_along_width
    subtiles = torch.empty((batch_size, num_subtiles, bands, subtile_dim, subtile_dim), device=device)
    for b in range(batch_size):
        subtile_idx = 0
        for i in range(num_subtiles_along_height):
            for j in range(num_subtiles_along_width):
                subtile = tile[b, :, subtile_dim*i:subtile_dim*(i+1), subtile_dim*j:subtile_dim*(j+1)].to(device)
                subtiles[b, subtile_idx, :, :, :] = subtile
                subtile_idx += 1
    return subtiles


# Called when the deployed service starts
def init():
    # global band_means
    # global band_stds
    # global sif_mean
    # global sif_std
    global device
    global subtile_sif_model
    global transform
    global sif_mean
    global sif_std

    # Get the path where the deployed model can be found.
    model_dir = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './models')

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    subtile_sif_model = simple_cnn.SimpleCNN(input_channels=INPUT_CHANNELS, reduced_channels=REDUCED_CHANNELS, output_dim=1).to(device)
    print('Model path', model_path)
    subtile_sif_model.load_state_dict(torch.load(model_path, map_location=device))
    subtile_sif_model.eval()

    # Read statistics from file
    train_statistics = pd.read_csv(os.path.join(model_dir, BAND_STATISTICS_FILENAME))
    train_means = train_statistics['mean'].values
    train_stds = train_statistics['std'].values
    band_means = train_means[:-1]
    sif_mean = train_means[-1]
    band_stds = train_stds[:-1]
    sif_std = train_stds[-1]

    # Set up data transformation (e.g. standardizing)
    # Set up image transforms
    transform_list = []
    transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
    transform = transforms.Compose(transform_list)
    #transform = tile_transforms.StandardizeTile(band_means, band_stds) 


# Handle requests to the service
def run(raw_data):
    try:
        # Pick out the tile from the JSON request.
        # This expects a request in the form of {"data": [serialized 3d numpy array in CxHxW format]}
        tile = np.array(json.loads(raw_data)['data'])
        prediction = predict(tile)
        return prediction
    except Exception as e:
        result = str(e)
        # return error message back to the client
        return json.dumps({"error": result})

# Predict SIF using the model
def predict(tile):
    start_at = time.time()

    # Standardize input tile, and convert it to a Tensor of shape (1 x C x H x W)
    input_tile_standardized = torch.tensor(transform(tile), dtype=torch.float).unsqueeze(0).to(device)
    subtiles_standardized = get_subtiles_list(input_tile_standardized, SUBTILE_DIM, device)[0]  # (num subtiles x bands x subtile_dim x subtile_dim)
    num_subtiles_along_height = int(input_tile_standardized.shape[2] / SUBTILE_DIM)
    num_subtiles_along_width = int(input_tile_standardized.shape[3] / SUBTILE_DIM)

    # Obttain predicted SIFs for each subtile
    with torch.set_grad_enabled(False):
        predicted_sifs_simple_cnn_standardized = subtile_sif_model(subtiles_standardized).cpu().numpy()
    predicted_sifs_simple_cnn_non_standardized = (predicted_sifs_simple_cnn_standardized * sif_std + sif_mean) \
            .reshape((num_subtiles_along_height, num_subtiles_along_width))
    return {"sifs": predicted_sifs_simple_cnn_non_standardized.tolist(),
            "elapsed_time": time.time()-start_at}  