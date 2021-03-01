# TODO - this can be consolidated with train_unet.py

import os
import pandas as pd
import resnet
import tile_transforms
from sif_utils import train_single_model
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
from SAN import SAN

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-08-01") #07-16")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
METHOD = "5_SAN"
LOSS_PLOT_FILE = "exploratory_plots/losses_" + METHOD + ".png"
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
RESNET_FROM_PRETRAINED = True  #True
RESNET_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_large_tile_resnet")
SAN_FROM_PRETRAINED = False # False #True #False #True  # Falsei
SAN_MODEL_FILE = os.path.join(DATA_DIR, "models/AUG_SAN")
NUM_EPOCHS = 50
INPUT_CHANNELS = 43
OPTIMIZER_TYPE = "Adam"
LEARNING_RATE = 1e-4 #5
WEIGHT_DECAY = 1e-5 
BATCH_SIZE = 16
NUM_WORKERS = 4
INPUT_SIZE = 371
OUTPUT_SIZE = 37
NUM_FEATURES = OUTPUT_SIZE*3
MIN_SIF = 0.2
MAX_SIF = 1.7
AUGMENT = True

# Print params for reference
print("=========================== PARAMS ===========================")
print("Method:", METHOD)
print("Dataset:", os.path.basename(DATASET_DIR))
if RESNET_FROM_PRETRAINED:
    print("From pretrained ResNet:", os.path.basename(RESNET_MODEL_FILE))
else:
    print("Training ResNet (front-end layers) from scratch")
if SAN_FROM_PRETRAINED:
    print("From pretrained SAN")
else:
    print("Training SAN from scratch")
print("Output model:", os.path.basename(SAN_MODEL_FILE))
print("---------------------------------")
print("Num features:", NUM_FEATURES)
print("Optimizer:", OPTIMIZER_TYPE)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Batch size:", BATCH_SIZE)
print("Num epochs:", NUM_EPOCHS)
print("Augment:", AUGMENT)
print("SIF range:", MIN_SIF, "to", MAX_SIF)
print("==============================================================")


if __name__ == "__main__":
    # Check if any CUDA devices are visible. If so, pick a default visible device.
    # If not, use CPU.
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    print("Device", device)

    # Read train/val tile metadata
    train_metadata = pd.read_csv(INFO_FILE_TRAIN) #.iloc[0:100]
    val_metadata = pd.read_csv(INFO_FILE_VAL) #.iloc[0:100]

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

    # Constrain predicted SIF to be between 0.2 and 1.7 (unstandardized)
    # Don't forget to standardize
    min_output = (MIN_SIF - sif_mean) / sif_std
    max_output = (MAX_SIF - sif_mean) / sif_std

    # Set up image transforms
    transform_list = []
    transform_list.append(tile_transforms.StandardizeTile(band_means, band_stds))
    if AUGMENT:
        transform_list.append(tile_transforms.RandomFlipAndRotate())
    transform_list.append(tile_transforms.ToFloatTensor())
    transform = transforms.Compose(transform_list)

    #; Set up Datasets and Dataloaders
    datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, transform),
                'val': ReflectanceCoverSIFDataset(val_metadata, transform)}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS)
              for x in ['train', 'val']}

    # Set up models (and load pre-trained if specified)
    resnet_model = resnet.resnet18(input_channels=INPUT_CHANNELS)
    if RESNET_FROM_PRETRAINED:
        resnet_model.load_state_dict(torch.load(RESNET_MODEL_FILE, map_location=device))

    model = SAN(resnet_model, input_height=INPUT_SIZE, input_width=INPUT_SIZE,
                output_height=OUTPUT_SIZE, output_width=OUTPUT_SIZE,
                feat_width=NUM_FEATURES, feat_height=NUM_FEATURES,
                in_channels=INPUT_CHANNELS, min_output=min_output, max_output=max_output).to(device)
    if SAN_FROM_PRETRAINED:
        model.load_state_dict(torch.load(SAN_MODEL_FILE, map_location=device))
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)

    # Set up loss/optimizer
    criterion = nn.MSELoss(reduction='mean')

    if OPTIMIZER_TYPE == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        print("Optimizer type not supported.")
        exit(1)

    dataset_sizes = {'train': len(train_metadata),
                 'val': len(val_metadata)}

    # Train model
    print("Starting to train")
    model, train_losses, val_losses, best_loss = train_single_model(model, dataloaders,
            dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, SAN_MODEL_FILE,
            num_epochs=NUM_EPOCHS)

    # Plot loss curves
    print("Train losses:", train_losses)
    print("Validation losses:", val_losses)
    epoch_list = range(NUM_EPOCHS)
    train_plot, = plt.plot(epoch_list, train_losses, color='blue', label='Train NRMSE')
    val_plot, = plt.plot(epoch_list, val_losses, color='red', label='Validation NRMSE')
    plt.legend(handles=[train_plot, val_plot])
    plt.xlabel('Epoch #')
    plt.ylabel('Normalized Root Mean Squared Error')
    plt.savefig(LOSS_PLOT_FILE)
    plt.close()



