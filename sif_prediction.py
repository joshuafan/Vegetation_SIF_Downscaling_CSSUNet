import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
train_dataset = ReflectanceCoverSIFDataset(train_metadata, resize_transform)
val_dataset = ReflectanceCoverSIFDataset(val_metadata, resize_transform)
test_dataset = ReflectanceCoverSIFDataset(test_metadata, resize_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                           shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4,
                                           shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                           shuffle=True, num_workers=2)

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

# get some random training images
dataiter = iter(train_loader)
samples = dataiter.next()
for sample in samples:
    imshow(sample['tile'])

resnet_model = resnet.resnet18(input_channels=14)
criterion = nn.MSELoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, sample in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        tile = sample['tile']
        true_sif = sample['SIF']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        predicted_sif = resnet_model(tile)
        loss = criterion(predicted_sif, true_sif)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
