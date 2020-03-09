"""
Trains a ResNet (with full supervision) to predict the total SIF of a large tile (0.1 x 0.1 degree)
"""

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

from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset

INFO_FILE_TRAIN = "datasets/generated/tile_info_train.csv"
INFO_FILE_VAL = "datasets/generated/tile_info_val.csv"
TRAINED_MODEL_FILE = "models/large_tile_sif_prediction"
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


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, average_sif, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_losses = []
    val_losses = []

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
                    predicted_sif = model(input_tile).flatten()
                    #_, preds = torch.max(outputs, 1)
                    #print('Predicted:', predicted_sif)
                    #print('True:', true_sif)
                    loss = criterion(predicted_sif, true_sif)
                    #print("loss", loss)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * len(sample['SIF'])
                # running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = math.sqrt(running_loss / dataset_sizes[phase]) / average_sif
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
average_sif = train_metadata['SIF'].mean()
print("Average sif", average_sif)
print("Train samples", len(train_metadata))
print("Validation samples", len(val_metadata))

# Set up Datasets and Dataloaders
# resize_transform = torchvision.transforms.Resize((224, 224))
datasets = {'train': ReflectanceCoverSIFDataset(train_metadata), # , resize_transform),
            'val': ReflectanceCoverSIFDataset(val_metadata)} # resize_transform),
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

print("Dataloaders")
resnet_model = resnet.resnet18(input_channels=14).to(device)
if FROM_PRETRAINED:
    resnet_model.load_state_dict(torch.load(TRAINED_MODEL_FILE))
criterion = nn.MSELoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=1e-4, momentum=0.9)
dataset_sizes = {'train': len(train_metadata),
                 'val': len(val_metadata)}

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Train model
print("Starting to train")
resnet_model, train_losses, val_losses, best_loss = train_model(resnet_model, dataloaders,
    dataset_sizes, criterion, optimizer, exp_lr_scheduler, device, average_sif, num_epochs=NUM_EPOCHS)

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
