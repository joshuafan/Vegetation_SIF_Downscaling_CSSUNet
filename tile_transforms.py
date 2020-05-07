import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize

class StandardizeTile(object):
    """
    Standardizes the given bands (listed in "bands_to_transform") so that each band has
    mean 0, standard deviation 1.
    Note: do not standardize bands that are already binary masks
    """
    def __init__(self, band_means, band_stds, bands_to_transform=list(range(0,12))):
        self.bands_to_transform = bands_to_transform 
        self.band_means = band_means[bands_to_transform, np.newaxis, np.newaxis]
        self.band_stds = band_stds[bands_to_transform, np.newaxis, np.newaxis]

    def __call__(self, tile):
        tile[self.bands_to_transform, :, :] = (tile[self.bands_to_transform, :, :] - self.band_means) / self.band_stds
        return tile


class RandomFlipAndRotate(object):
    """
    Code taken from Tile2Vec.
    Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """
    def __call__(self, tile):
        # Randomly horizontal flip
        if np.random.rand() < 0.5:
            tile = np.flip(tile, axis=2).copy()
        # Randomly vertical flip
        if np.random.rand() < 0.5:
            tile = np.flip(tile, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0:
            tile = np.rot90(tile, k=rotations, axes=(1,2)).copy()
        return tile


class ResizeTile(object):
    def __init__(self, target_dim=[371,371], discrete_bands=list(range(12, 43))):
        self.target_dim = target_dim
        self.discrete_bands = discrete_bands

    def __call__(self, tile):
        # Convert tile into Scikit learn order (channels as last dimension, not first)
        tile_numpy = np.moveaxis(tile, 0, -1)
        #print('Numpy tile shape', tile_numpy.shape)
        resized_tile = resize(tile_numpy, self.target_dim, mode='edge')
        #print('After reize', resized_tile.shape)
        resized_tile = np.moveaxis(resized_tile, -1, 0)
        #print('Resized tile shape', resized_tile.shape)
        resized_tile[self.discrete_bands, :, :] = np.round(resized_tile[self.discrete_bands, :, :])
        
        #new_shape = [num_bands] + self.target_dim
        #print('New shape', new_shape)
        #tile = torch.from_numpy(tile).float().unsqueeze(0)
        #print('SHape', tile.shape)

        #resized_tile = F.interpolate(tile, size=self.target_dim, mode='bilinear')
        #resized_tile[self.discrete_bands, :, :] = torch.round(resized_tile[self.discrete_bands, :, :])
        return resized_tile


class ToFloatTensor(object):
    def __call__(self, tile):
        return torch.from_numpy(tile).float()


class ShrinkTile(object):
    """
    Shrinks tile down to designated size (e.g. from 371 x 371 to 10 x 10).
    Contains custom logic to ensure that binary bands (1/0) stay that way.
    """
    def __init__(self, target_dim=10, continuous_bands=list(range(0, 12)), cover_bands=list(range(12, 42)), missing_band=42):
        print('Continuous bands', continuous_bands)
        self.target_dim = target_dim
        self.continuous_bands = continuous_bands
        self.cover_bands = cover_bands
        self.missing_band = missing_band

    # tile is assumed to be CxWxH
    def __call__(self, tile):
        # print('Original tile shape', tile.shape)
        bands, original_height, original_width = tile.shape
        resized_tile = np.zeros((bands, self.target_dim, self.target_dim))
        
        # Loop through each pixel in the target smaller tile
        for i in range(self.target_dim):
            for j in range(self.target_dim):
                # Find which pixels in original tile correspond to this pixel
                top = int((i / self.target_dim) * original_height)
                bottom = int(((i+1) / self.target_dim) * original_height)
                left = int((j / self.target_dim) * original_width)
                right = int(((j+1) / self.target_dim) * original_width)
                #print('Top', top, 'Bottom', bottom)
                #print('Left', left, 'Right', right)

                # Area of original tile mapping to this pixel
                original_pixels = tile[:, top:bottom, left:right]
                
                # Sample center pixel
                #resized_tile[:, i, j] = tile[:, int((top+bottom)/2), int((left+right)/2)]
                #continue

                # Count number of pixels with and without reflectance data (by looking at the missing_band mask)
                pixels_without_reflectance = np.sum(original_pixels[self.missing_band, :, :])
                pixels_with_reflectance = (original_pixels.shape[1] * original_pixels.shape[2]) - pixels_without_reflectance
                # print('Pixels without', pixels_without_reflectance, 'Pixels with', pixels_with_reflectance)

                # Take the average reflectance value, but only consider nonzero pixels (where data is NOT missing) in denominator.
                # Note: if no pixels contain reflectance, make the new pixel values 0 (already 
                # set by default)
                # if pixels_with_reflectance >= 1:
                #   resized_tile[self.continuous_bands, i, j] = np.sum(original_pixels[self.continuous_bands, :, :], axis=(1,2)) / pixels_with_reflectance
                # print('Shape of mean',  np.mean(original_pixels[self.continuous_bands, :, :], axis=(1,2)).shape)
                resized_tile[self.continuous_bands, i, j] = np.mean(original_pixels[self.continuous_bands, :, :], axis=(1,2)) 

                # Compute percent covered by each crop type.
                # Pick the crop type with the plurality of pixels
                # ("no crop type" is a type)
                cover_fractions = np.mean(original_pixels[self.cover_bands, :, :], axis=(1,2))
                fraction_no_cover = 1 - np.sum(cover_fractions)
                #print('Cover fractions', cover_fractions)
                #print('Fraction no cover', fraction_no_cover)
                index = np.argmax(cover_fractions)
                #print('argmax index', index)
                value = cover_fractions[index]
                if value > 0.2:
                    most_common_cover = self.cover_bands[index]
                    #print('Most common cover:', most_common_cover)
                    resized_tile[most_common_cover, i, j] = 1
                #else:
                    #print('No cover is most common.')
                # Set missing mask to 1 if majority of pixels were missing
                if pixels_without_reflectance > pixels_with_reflectance:
                    resized_tile[self.missing_band, i, j] = 1
        #print('Random pixel', resized_tile[:, 2, 6])
        #print('Random pixel', resized_tile[:, 3, 8])
        return resized_tile

if __name__ == '__main__':
    example_tensor = [[[0.4, 0.5, 1.8], [0.9, 1.0, 3.5]], [[100., 200., 1000.], [150., 250., 5000.]], [[0, 0, 0], [1, 1, 0]]]
    example_tensor = np.array(example_tensor)
    transform = ResizeTile(target_dim=[4, 6], discrete_bands=[2])
    example_tensor = transform(example_tensor)
    print(example_tensor)
    print(example_tensor.shape)
