import os
import pandas as pd
import resnet
import tile_transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from reflectance_cover_sif_dataset import ReflectanceCoverSIFDataset
from SAN import SAN

DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_2018-07-16")
INFO_FILE_TRAIN = os.path.join(DATASET_DIR, "tile_info_train.csv")
INFO_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")
LOSS_PLOT_FILE = "exploratory_plots/losses_SAN.png"
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
RESNET_FROM_PRETRAINED = True
RESNET_MODEL_FILE = os.path.join(DATA_DIR, "models/large_tile_resnet50")
SAN_FROM_PRETRAINED = False
SAN_MODEL_FILE = os.path.join(DATA_DIR, "models/SAN")
NUM_EPOCHS = 50
INPUT_CHANNELS = 43
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
BATCH_SIZE = 32
NUM_WORKERS = 4
INPUT_SIZE = 371
OUTPUT_SIZE = 37



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
    train_metadata = pd.read_csv(INFO_FILE_TRAIN).iloc[0:100]
    val_metadata = pd.read_csv(INFO_FILE_VAL).iloc[0:100]

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
    datasets = {'train': ReflectanceCoverSIFDataset(train_metadata, transform),
                'val': ReflectanceCoverSIFDataset(val_metadata, transform)}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS)
              for x in ['train', 'val']}

    # Set up models (and load pre-trained if specified)
    resnet_model = resnet.resnet50(input_channels=INPUT_CHANNELS)
    if RESNET_FROM_PRETRAINED:
        resnet_model.load_state_dict(torch.load(RESNET_MODEL_FILE, map_location=device))
    
    model = SAN(resnet_model, input_height=INPUT_SIZE, input_width=INPUT_SIZE,
                output_height=OUTPUT_SIZE, output_width=OUTPUT_SIZE,
                feat_width=OUTPUT_SIZE, feat_height=OUTPUT_SIZE,
                in_channels=INPUT_CHANNELS).to(device)
    if SAN_FROM_PRETRAINED:
        model.load_state_dict(torch.load(SAN_MODEL_FILE, map_location=device))

    # Set up loss/optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(resnet_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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



