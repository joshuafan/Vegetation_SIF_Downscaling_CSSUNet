import copy
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

from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset

CSV_INFO_FILE = "datasets/generated/reflectance_cover_to_sif.csv"
RGB_BANDS = [1, 2, 3]

# Read tile metadata
tile_metadata = pd.read_csv(CSV_INFO_FILE)

# Randomly split into train/validation/test
train_and_validation, test_metadata = train_test_split(tile_metadata, test_size=0.2)
train_metadata, val_metadata = train_test_split(train_and_validation, test_size=0.25)
print("Train samples", len(train_metadata))
print("Validation samples", len(val_metadata))
print("Test samples", len(test_metadata))

# Set up Datasets and Dataloaders
resize_transform = torchvision.transforms.Resize((224, 224))
datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, resize_transform),
            'val': ReflectanceCoverSIFDataset(val_metadata, resize_transform),
            'test': ReflectanceCoverSIFDataset(test_metadata, resize_transform)}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}


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


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

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
                input_tile = sample['tile'].to(device)
                true_sif = sample['SIF'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    predicted_sif = model(input_tile)
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(predicted_sif, true_sif)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * sample.size(0)
                # running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# get some random training images
dataiter = iter(dataloaders['train'])
samples = dataiter.next()
for sample in samples:
    imshow(sample['tile'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", device)
resnet_model = resnet.resnet18(input_channels=14).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
resnet_model = train_model(resnet_model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, device,
                           num_epochs=25)