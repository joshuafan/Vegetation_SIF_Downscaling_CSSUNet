import torch.nn as nn
import torch.optim as optim

# Don't know how to properly import from Tile2Vec
from tile2vec.src.tilenet import make_tileNet
from embedding_to_sif_model import EmbeddingToSIFModel

TILE2VEC_MODEL_FILE = "models/tile2vec/TileNet_epoch50.ckpt"
EMBEDDING_TO_SIF_MODEL_FILE = "models/embedding_to_sif"
FREEZE_TILE2VEC = True
Z_DIM = 20
INPUT_CHANNELS= 14
LEARNING_RATE = 1e-4

# Given the tile (C x W x H), returns a list of subtiles, each of dimension (C x subtile_dim x subtile_dim)
def get_subtiles_list(tile, subtile_dim):
    bands, width, height = tile.shape
    num_subtiles_along_width = int(width / subtile_dim)
    num_subtiles_along_height = int(height / subtile_height)
    assert(num_subtiles_along_width == 37)
    assert(num_subtiles_along_height == 37)
    subtiles = []
    for i in range(num_subtiles_along_width):
        for j in range(num_subtiles_along_height):
            subtile = tile[:, subtile_dim*i:subtile_dim*(i+1), subtile_dim*j:subtile_dim*(j+1)]
            subtiles.append(subtile)
    return subtiles


# "means" is a list of band averages (+ average SIF at end)
def train_model(tile2vec_model, embedding_to_sif_model, freeze_tile2vec, dataloaders, dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    print('SIF mean', sif_mean)
    print('SIF std', sif_std)
    sif_mean = torch.tensor(sif_mean).to(device)
    sif_std = torch.tensor(sif_std).to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if freeze_tile2vec:
                    tile2vec_model.eval()
                else:
                    tile2vec_model.train()
                embedding_to_sif_model.train()
            else:
                tile2vec_model.eval()
                embedding_to_sif_model.eval()

            running_loss = 0.0

            # Iterate over data.
            for sample in dataloaders[phase]:
                input_tile_standardized = sample['tile'].to(device)
                true_sif_non_standardized = sample['SIF'].to(device)
                true_sif_standardized = ((true_sif_non_standardized - sif_mean) / sif_std).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Obtain subtiles (NOTE Pay attention to standardization :( )
                subtiles = get_subtile_list(input_tile_standardized)
                predicted_subtile_sifs = torch.new_empty((len(subtiles)), device=device)
                
                # Forward pass: feed subtiles through embedding model and then the
                # embedding -> SIF model
                with torch.set_grad_enabled(phase == 'train'):
                    for i in range(len(subtiles)):
                        embedding = tile2vec_model(subtiles[i].to(device))
                        predicted_sif = embedding_to_sif_model(embedding)
                        predicted_subtile_sifs[i] = predicted_sif
                    
                    # Predicted SIF for full tile
                    predicted_sif_standardized = torch.sum(predicted_subtile_sifs)
                    loss = criterion(predicted_sif_standardized, true_sif_standardized)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                predicted_sif_non_standardized = torch.tensor(predicted_sif_standardized * sif_std + sif_mean, dtype=torch.float).to(device)
                print('========================')
                print('Predicted', predicted_sif_non_standardized)
                print('True', true_sif_non_standardized)
                non_standardized_loss = criterion(predicted_sif_non_standardized, true_sif_non_standardized)
                running_loss += non_standardized_loss.item() * len(sample['SIF'])

            epoch_loss = math.sqrt(running_loss / dataset_sizes[phase]) / sif_mean

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = [copy.deepcopy(tile2vec_model.state_dict()),
                                  copy.deepcopy(embedding_to_sif_model.state_dict())]

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
    tile2vec_model.load_state_dict(best_model_wts[0])
    embedding_to_sif_model.load_state_dict(best_model_wts[1])
    return tile2vec_model, embedding_to_sif_model, train_losses, val_losses, best_loss


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

tile2vec_model = make_tilenet(input_channels=INPUT_CHANNELS, z_dim=Z_DIM).to(device)
tile2vec_model.load_state_dict(torch.load(TILE2VEC_MODEL_FILE))
embedding_to_sif_model = EmbeddingToSIFModel(embedding_size=Z_DIM)
criterion = nn.MSELoss(reduction='mean')
#optimizer = optim.SGD(resnet_model.parameters(), lr=1e-4, momentum=0.9)

# Don't optimize Tile2vec model; just use pre-trained version
if FREEZE_TILE2VEC:
    params_to_optimize = embedding_to_sif_model.parameters()
else:
    params_to_optimize = list(tile2vec_model.parameters()) + list(embedding_to_sif_model.parameters())
optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE)
dataset_sizes = {'train': len(train_metadata),
                 'val': len(val_metadata)}


tile2vec_model, embedding_to_sif_model, train_losses, val_losses, best_loss = train_model(tile2vec_model, embedding_to_sif_model, FREEZE_TILE2VEC, dataloaders, dataset_sizes, criterion, optimizer, device, sif_mean, sif_std, num_epochs=NUM_EPOCHS)

torch.save(embedding_to_sif_model.state_dict(), EMBEDDING_TO_SIF_MODEL_FILE)

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
