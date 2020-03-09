import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler

import time
import torch
import torchvision
import torchvision.transforms as transforms
import resnet
import torch.nn as nn
import torch.optim as optim



EVAL_FILE = "datasets/generated_subtiles/eval_subtiles.csv" 
TRAINED_MODEL_FILE = "models/large_tile_sif_prediction"


eval_points = pd.read_csv(EVAL_FILE)

def eval_model(model, dataloader, dataset_size, device, average_sif):
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0

    # Iterate over data.
    for sample in dataloader:
        input_tile = sample['tile'].to(device)
        true_sif = sample['SIF'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            predicted_sif = model(input_tile).flatten()
            loss = criterion(predicted_sif, true_sif)

        # statistics
        running_loss += loss.item() * len(sample['SIF'])

    loss = math.sqrt(running_loss / dataset_size) / average_sif
    return loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", device)

# Read train/val tile metadata
eval_metadata = pd.read_csv(EVAL_FILE)
average_sif = eval_metadata['SIF'].mean()
print("Average sif", average_sif)
print("Eval samples", len(eval_metadata))
dataset_size = len(eval_metadata)
dataset = ReflectanceCoverSIFDataset(eval_metadata)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                         shuffle=True, num_workers=4)

resnet_model = resnet.resnet18(input_channels=14).to(device)
if FROM_PRETRAINED:
    resnet_model.load_state_dict(torch.load(TRAINED_MODEL_FILE))

loss = eval_model(resnet_model, dataloader, dataset, device, average_sif)
print("Eval Loss", loss)

