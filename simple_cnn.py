import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_dim, reduced_channels=43):
        super(SimpleCNN, self).__init__()
        self.dimensionality_reduction = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(reduced_channels, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        #self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        #self.conv3 = nn.Conv2d(128, 256, kernel_size=2, stride=1)
        #self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.dimensionality_reduction(x)
        x = self.pool(F.relu(self.conv1(x)))
        #print('After layer 1', x.shape)
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        #x = self.pool(F.relu(self.conv2(x)))
        #print('After layer 2', x.shape)
        x = x.view(-1, 128 * 2 * 2)
        #x = x.view(-1, 256 * 2 * 2)
        #print('Before fully-connected', x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
