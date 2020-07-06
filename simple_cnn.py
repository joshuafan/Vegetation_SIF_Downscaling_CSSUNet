import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_dim, reduced_channels=43, min_output=None, max_output=None):
        super(SimpleCNN, self).__init__()
        self.dimensionality_reduction = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(reduced_channels, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        #self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        #self.conv3 = nn.Conv2d(128, 256, kernel_size=2, stride=1)
        #self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, output_dim)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
        else:
            self.restrict_output = False

    def forward(self, x):
        x = F.relu(self.dimensionality_reduction(x))
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
        if self.restrict_output:
            x = (F.tanh(x) * self.scale_factor) + self.mean_output
            #x = torch.clamp(x, min=self.min_output, max=self.max_output)
        return x
    
class SimpleCNNSmall(nn.Module):
    def __init__(self, input_channels, output_dim, crop_type_start_idx, crop_type_embedding_dim=15, reduced_channels=20, min_output=None, max_output=None):
        super(SimpleCNNSmall, self).__init__()

        # Create list of crop type indices and non-crop indices
        self.non_crop_type_bands = list(range(0, crop_type_start_idx)) + [input_channels - 1]  # Last channel is the "missing reflectance" mask and is not a crop type
        self.crop_type_bands = list(range(crop_type_start_idx, input_channels - 1))
        # print('non crop type indices', self.non_crop_type_bands)
        # print('crop type indices', self.crop_type_bands)

        # # Embedding vector for each crop type's pixels
        self.crop_type_embedding = nn.Conv2d(len(self.crop_type_bands), crop_type_embedding_dim, kernel_size=1, stride=1)
        
        # # Embed each pixel. Each pixel's vector should contain semantic information about
        # # the crop type + reflectance + other features
        channels_after_embedding = input_channels - len(self.crop_type_bands) + crop_type_embedding_dim  # Number of features after embedding crop type
        # print('Channels after embedding', channels_after_embedding)

        self.dimensionality_reduction_1 = nn.Conv2d(channels_after_embedding, reduced_channels, kernel_size=1, stride=1)
        self.dimensionality_reduction_2 = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(reduced_channels, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
        else:
            self.restrict_output = False

    def forward(self, x):
        crop_masks = x[:, self.crop_type_bands, :, :]
        # print('===================================================')
        # print('Random pixel (7, 7)', x[0, :, 7, 7])
        # print('Random pixel (7, 8)', x[0, :, 7, 8])
        # print('Random pixel (7, 9)', x[0, :, 7, 9])
        # print('Crop masks shape', crop_masks.shape)
        crop_embeddings = self.crop_type_embedding(crop_masks)
        # print('Crop embeddings shape', crop_embeddings.shape)
        x = torch.cat([x[:, self.non_crop_type_bands, :, :], crop_embeddings], dim=1)

        # print('After stacking', x.shape)
        # print('(After embedding) Random pixel (7, 7)', x[0, :, 7, 7])
        # print('Random pixel (7, 8)', x[0, :, 7, 8])
        # print('Random pixel (7, 9)', x[0, :, 7, 9])

        # Dimensionality reduction (1x1 convolution)
        x = F.relu(self.dimensionality_reduction_1(x))
        # print('Reduced dimensionality', x[0, :, 7, 8])
        # print('Reduced dimensionality', x[0, :, 7, 8])
        x = F.relu(self.dimensionality_reduction_2(x))

        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 1 * 1)

        # Fully-connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.restrict_output:
            x = (F.tanh(x) * self.scale_factor) + self.mean_output
            #x = torch.clamp(x, min=self.min_output, max=self.max_output)
        return x

class SimpleCNNSmall2(nn.Module):
    def __init__(self, input_channels, output_dim, crop_type_start_idx, crop_type_embedding_dim=15, reduced_channels=20, min_output=None, max_output=None):
        super(SimpleCNNSmall2, self).__init__()

        # Create list of crop type indices and non-crop indices
        self.non_crop_type_bands = list(range(0, crop_type_start_idx)) + [input_channels - 1]  # Last channel is the "missing reflectance" mask and is not a crop type
        self.crop_type_bands = list(range(crop_type_start_idx, input_channels - 1))
        # print('non crop type indices', self.non_crop_type_bands)
        # print('crop type indices', self.crop_type_bands)

        # # Embedding vector for each crop type's pixels
        self.crop_type_embedding = nn.Conv2d(len(self.crop_type_bands), crop_type_embedding_dim, kernel_size=1, stride=1)
        
        # # Embed each pixel. Each pixel's vector should contain semantic information about
        # # the crop type + reflectance + other features
        channels_after_embedding = input_channels - len(self.crop_type_bands) + crop_type_embedding_dim  # Number of features after embedding crop type
        # print('Channels after embedding', channels_after_embedding)

        self.dimensionality_reduction_1 = nn.Conv2d(channels_after_embedding, reduced_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(reduced_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, output_dim)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
        else:
            self.restrict_output = False

    def forward(self, x):
        crop_masks = x[:, self.crop_type_bands, :, :]
        # print('===================================================')
        # print('Random pixel (7, 7)', x[0, :, 7, 7])
        # print('Random pixel (7, 8)', x[0, :, 7, 8])
        # print('Random pixel (7, 9)', x[0, :, 7, 9])
        # print('Crop masks shape', crop_masks.shape)
        crop_embeddings = self.crop_type_embedding(crop_masks)
        # print('Crop embeddings shape', crop_embeddings.shape)
        x = torch.cat([x[:, self.non_crop_type_bands, :, :], crop_embeddings], dim=1)

        # print('After stacking', x.shape)
        # print('(After embedding) Random pixel (7, 7)', x[0, :, 7, 7])
        # print('Random pixel (7, 8)', x[0, :, 7, 8])
        # print('Random pixel (7, 9)', x[0, :, 7, 9])

        # Dimensionality reduction (1x1 convolution)
        x = F.relu(self.dimensionality_reduction_1(x))
        # print('Reduced dimensionality', x[0, :, 7, 8])
        # print('Reduced dimensionality', x[0, :, 7, 8])

        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 1 * 1)

        # Fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.restrict_output:
            x = (F.tanh(x) * self.scale_factor) + self.mean_output
            #x = torch.clamp(x, min=self.min_output, max=self.max_output)
        return x


class SimpleCNNSmall3(nn.Module):
    def __init__(self, input_channels, output_dim,min_output=None, max_output=None):
        super(SimpleCNNSmall3, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, output_dim)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
        else:
            self.restrict_output = False

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        # print('After conv 1', x.shape)
        x = F.relu(self.conv2(x))
        # print('After conv 2', x.shape)
        x = self.pool(x)
        # print('After pool 2', x.shape)
        x = F.relu(self.conv3(x))
        # print('After conv 3', x.shape)
        x = self.pool(x)
        # print('After pool 3', x.shape)
        x = x.view(-1, 128 * 2 * 2)
        # print('Reshaped', x.shape)

        # Fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.restrict_output:
            x = (F.tanh(x) * self.scale_factor) + self.mean_output
            #x = torch.clamp(x, min=self.min_output, max=self.max_output)
        return x

class SimpleCNNSmall4(nn.Module):
    def __init__(self, input_channels, output_dim, min_output=None, max_output=None):
        super(SimpleCNNSmall4, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, output_dim)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
        else:
            self.restrict_output = False

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        # print('After conv 1', x.shape)
        x = F.relu(self.conv2(x))
        # print('After conv 2', x.shape)
        x = self.pool(x)
        # print('After pool 2', x.shape)
        x = F.relu(self.conv3(x))
        # print('After conv 3', x.shape)
        x = self.pool(x)
        # print('After pool 3', x.shape)
        x = x.view(-1, 64 * 2 * 2)
        # print('Reshaped', x.shape)

        # Fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.restrict_output:
            x = (F.tanh(x) * self.scale_factor) + self.mean_output
            #x = torch.clamp(x, min=self.min_output, max=self.max_output)
        return x

class SimpleCNNSmall5(nn.Module):
    def __init__(self, input_channels, output_dim, min_output=None, max_output=None):
        super(SimpleCNNSmall5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 32)
        self.fc2 = nn.Linear(32, output_dim)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
        else:
            self.restrict_output = False

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        # print('After conv 1', x.shape)
        x = F.relu(self.conv2(x))
        # print('After conv 2', x.shape)
        x = self.pool(x)
        # print('After pool 2', x.shape)
        x = F.relu(self.conv3(x))
        # print('After conv 3', x.shape)
        x = self.pool(x)
        # print('After pool 3', x.shape)
        x = x.view(-1, 256 * 2 * 2)
        # print('Reshaped', x.shape)

        # Fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.restrict_output:
            x = (F.tanh(x) * self.scale_factor) + self.mean_output
            #x = torch.clamp(x, min=self.min_output, max=self.max_output)
        return x


class PixelNN(nn.Module):
    def __init__(self, input_channels, output_dim, min_output=None, max_output=None):
        super(SimpleCNNSmall5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
        else:
            self.restrict_output = False

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        if self.restrict_output:
            x = (F.tanh(x) * self.scale_factor) + self.mean_output
        return x



