"""
Trains a ResNet (with full supervision) to predict the total SIF of a large tile (0.1 x 0.1 degree)
"""

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
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

from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
import tile_transforms

DATASET_DIR = "datasets/dataset_2018-08-01"
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
TRAINED_MODEL_FILE = "models/large_tile_sif_prediction"
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
FROM_PRETRAINED = False
RGB_BANDS = [1, 2, 3]
NUM_EPOCHS = 10 

# Visualize images (RGB bands only)
# Tile is assumed to be in Pytorch format: CxWxH
def imshow(tile):
    print("================= Per-band averages: =====================")
    for i in range(tile.shape[0]):
        print("Band", i, ":", np.mean(tile[i].flatten()))
    print("==========================================================")
    img = tile[RGB_BANDS, :, :]
    print("Image shape", img.shape)

    #img = img / 2 + 0.5     # unnormalize
    #npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


# "means" is a list of band averages (+ average SIF at end)
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for sample in dataloaders[phase]:
                #if band_means_batch is not None:
                #    print("Band means shape", band_means_batch.shape)
                #print("Tile shape", sample['tile'].shape)

                # Recall: sample['tile'] has shape (batch x band x lat x long)
                #batch_size = sample['tile'].shape[0]

                # If batch size changed (or if this is the first time), compute repeated mean/std vectors 
                #if band_means_batch is None or band_means_batch.shape[0] != batch_size:
                #    band_means_batch = torch.tensor(np.repeat(band_means[np.newaxis, :, np.newaxis, np.newaxis], batch_size, axis=0), dtype=torch.float)
                #    band_stds_batch = torch.tensor(np.repeat(band_stds[np.newaxis, :, np.newaxis, np.newaxis], batch_size, axis=0), dtype=torch.float)
                #    print('Now band means shape', band_means_batch.shape)
                #input_tile_standardized = ((sample['tile'] - band_means_batch) / band_stds_batch).to(device)
                input_tile_standardized = sample['tile']
                true_sif_non_standardized = sample['SIF'].to(device)
                true_sif_standardized = ((true_sif_non_standardized - sif_mean) / sif_std).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    predicted_sif_standardized = model(input_tile_standardized).flatten()
                    #_, preds = torch.max(outputs, 1)
                    #print('Predicted (standardized):', predicted_sif_standardized)
                    #print('True (standardized):', true_sif_standardized)
                    loss = criterion(predicted_sif_standardized, true_sif_standardized)
                    #print("loss", loss)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                #print('========================')
                #print('Predicted', predicted_sif_non_standardized)
                #print('True', true_sif_non_standardized)
                non_standardized_loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)
                running_loss += non_standardized_loss.item() * len(sample['SIF'])
                # running_corrects += torch.sum(preds == labels.data)

            #if phase == 'train':
            #    scheduler.step()

            epoch_loss = math.sqrt(running_loss / dataset_sizes[phase]) / sif_mean
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            # Record loss
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, best_loss


# get some random training images
#dataiter = iter(dataloaders['train'])
#samples = dataiter.next()
#for sample in samples:
#    imshow(sample['tile'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", device)

# Read train/val tile metadata
train_metadata = pd.read_csv(INFO_FILE_TRAIN)
val_metadata = pd.read_csv(INFO_FILE_VAL)

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
print("Train samples", len(train_metadata))
print("Validation samples", len(val_metadata))
print("Means", train_means)
print("Stds", train_stds)
band_means = train_means[:-1]
sif_mean = train_means[-1]
band_stds = train_stds[:-1]
sif_std = train_stds[-1]

# Set up image transforms
transform_list = []
transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
transform = transforms.Compose(transform_list)

# Set up Datasets and Dataloaders
# resize_transform = torchvision.transforms.Resize((224, 224))
datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, transform),
            'val': ReflectanceCoverSIFDataset(val_metadata, transform)}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4,
                                              shuffle=True, num_workers=1)
              for x in ['train', 'val']}

print("Dataloaders")
resnet_model = resnet.resnet18(input_channels=14)  #.to(device)
print("Got model")
resnet_model = resnet_model.to(device)

if FROM_PRETRAINED:
    resnet_model.load_state_dict(torch.load(TRAINED_MODEL_FILE))
criterion = nn.MSELoss(reduction='mean')
#optimizer = optim.SGD(resnet_model.parameters(), lr=1e-4, momentum=0.9)
optimizer = optim.Adam(resnet_model.parameters(), lr=1e-4)
dataset_sizes = {'train': len(train_metadata),
                 'val': len(val_metadata)}

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Train model
print("Starting to train")
resnet_model, train_losses, val_losses, best_loss = train_model(resnet_model, dataloaders,
    dataset_sizes, criterion, optimizer, device, train_means, train_stds, num_epochs=NUM_EPOCHS)

# Save model to file
torch.save(resnet_model.state_dict(), TRAINED_MODEL_FILE)

# Plot loss curves
print("Train losses:", train_losses)
print("Validation losses:", val_losses)
epoch_list = range(NUM_EPOCHS)
train_plot, = plt.plot(epoch_list, train_losses, color='blue', label='Train NRMSE')
val_plot, = plt.plot(epoch_list, val_losses, color='red', label='Validation NRMSE')
plt.legend(handles=[train_plot, val_plot])
plt.xlabel('Epoch #')
plt.ylabel('Normalized Root Mean Squared Error')
plt.savefig('plots/resnet_sif_prediction.png')
plt.close()
