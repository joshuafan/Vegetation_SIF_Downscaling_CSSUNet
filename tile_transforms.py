import numpy as np
import random
import torch
import torch.nn.functional as F
from skimage.transform import resize


class NormalizeReflectance(object):
    """
    Normalizes reflectance bands (listed in "reflectance_bands") so that over those bands, each *pixel* has norm 1.
    """
    def __init__(self, reflectance_bands=list(range(0, 9))):
        self.reflectance_bands = reflectance_bands
    def __call__(self, tile):
        normalized_bands = tile[self.reflectance_bands, :, :] / np.linalg.norm(tile[self.reflectance_bands, :, :], axis=0, keepdims=True)
        tile[self.reflectance_bands, :, :] = np.nan_to_num(normalized_bands, nan=0, posinf=0, neginf=0)
        return tile


class StandardizeTile(object):
    """
    Standardizes the given bands (listed in "bands_to_transform") so that each band has
    mean 0, standard deviation 1.
    Note: do not standardize bands that are already binary masks.
    """
    def __init__(self, band_means, band_stds, bands_to_transform=list(range(0, 12))):
        self.bands_to_transform = bands_to_transform 
        self.band_means = band_means[bands_to_transform, np.newaxis, np.newaxis]
        self.band_stds = band_stds[bands_to_transform, np.newaxis, np.newaxis]

    def __call__(self, tile):
        tile[self.bands_to_transform, :, :] = (tile[self.bands_to_transform, :, :] - self.band_means) / self.band_stds
        return tile


class TanhTile(object):
    """
    Divides "bands_to_transform" by "tanh_stretch", then passes through a tanh function
    """
    def __init__(self, tanh_stretch=3, bands_to_transform=list(range(0, 12))):
        self.bands_to_transform = bands_to_transform
        self.tanh_stretch = tanh_stretch
    
    def __call__(self, tile):
        tile[self.bands_to_transform, :, :] = np.tanh(tile[self.bands_to_transform, :, :] / self.tanh_stretch)
        return tile

class ClipTile(object):
    """
    Clips the specified bands ("bands_to_clip") to within the range [min_input, max_input].
    """
    def __init__(self, min_input, max_input, bands_to_transform=list(range(0, 12))):
        self._min_input = min_input
        self._max_input = max_input
        self.bands_to_transform = bands_to_transform

    def __call__(self, tile):
        if self._min_input is not None and self._max_input is not None:
            tile[self.bands_to_transform, :, :] = np.clip(tile[self.bands_to_transform, :, :], a_min=self._min_input, a_max=self._max_input)
        return tile


class GaussianNoise(object):
    def __init__(self, bands_to_transform, standard_deviation=0.1):
        self.bands_to_transform = bands_to_transform
        self.standard_deviation = standard_deviation

    def __call__(self, tile):
        #print('Before noise', tile[self.continuous_bands, 0:3, 0:3])
        continuous_bands_shape = tile[self.bands_to_transform, :, :].shape
        noise = np.random.normal(loc=0, scale=self.standard_deviation, size=continuous_bands_shape)
        tile[self.bands_to_transform, :, :] = tile[self.bands_to_transform, :, :] + noise
        #print('After noise', tile[self.bands_to_transform, 0:3, 0:3])
        return tile

class MultiplicativeGaussianNoise(object):
    """Assumes input has already been standardized. Need to un-standardize, add multiplicative noise, then re-standardize
    """
    def __init__(self, band_means, band_stds, bands_to_transform=list(range(0, 9)), standard_deviation=0.1):
        self.bands_to_transform = bands_to_transform 
        self.band_means = band_means[bands_to_transform, np.newaxis, np.newaxis]
        self.band_stds = band_stds[bands_to_transform, np.newaxis, np.newaxis]
        self.standard_deviation = standard_deviation

    def __call__(self, tile):
        # Undo standardization
        # print("======================")
        # print("Start - pixel", tile[self.bands_to_transform, 50, 50])
        continuous_bands = (tile[self.bands_to_transform, :, :] * self.band_stds) + self.band_means
        # print("After unstd", continuous_bands[:, 50, 50])

        # Apply noise
        noise = np.random.normal(loc=0, scale=self.standard_deviation)
        # print("Noise", noise)
        continuous_bands = continuous_bands * (1 + noise)
        # print("After noise", continuous_bands[:, 50, 50])

        # Re-standardize
        tile[self.bands_to_transform, :, :] = (continuous_bands - self.band_means) / self.band_stds
        # print("After std", tile[self.bands_to_transform, 50, 50])

        return tile


class MultiplicativeGaussianNoiseRaw(object):
    """Assumes input is NOT standardized
    """
    def __init__(self, bands_to_transform=list(range(0, 9)), standard_deviation=0.1):
        self.bands_to_transform = bands_to_transform
        self.standard_deviation = standard_deviation

    def __call__(self, tile):
        noise = np.random.normal(loc=0, scale=self.standard_deviation)
        tile[self.bands_to_transform, :, :] = tile[self.bands_to_transform, :, :] * (1 + noise)
        return tile



class ColorDistortion(object):
    def __init__(self, bands_to_transform, standard_deviation=0.1):
        self.bands_to_transform = bands_to_transform
        self.standard_deviation = standard_deviation

    def __call__(self, tile):
        noise = np.random.normal(loc=0, scale=self.standard_deviation, size=(len(self.bands_to_transform)))
        noise = noise[:, np.newaxis, np.newaxis].astype(np.float32)
        tile[self.bands_to_transform, :, :] = tile[self.bands_to_transform, :, :] + noise
        return tile


class GaussianNoiseSubtiles(object):
    def __init__(self, bands_to_transform, standard_deviation=0.1):
        self.bands_to_transform = bands_to_transform
        self.standard_deviation = standard_deviation

    def __call__(self, subtiles):
        #print('Before noise', tile[self.bands_to_transform, 0:3, 0:3])
        continuous_bands_shape = subtiles[:, self.bands_to_transform, :, :].shape
        noise = np.random.normal(loc=0, scale=self.standard_deviation, size=continuous_bands_shape)
        subtiles[:, self.bands_to_transform, :, :] = subtiles[:, self.bands_to_transform, :, :] + noise
        #print('After noise', tile[self.bands_to_transform, 0:3, 0:3])
        return subtiles


class RandomJigsaw(object):
    def __call__(self, tile):
        new_tile = np.copy(tile)

        # Randomly re-arrange horizontally
        if np.random.rand() < 0.5:
            width = tile.shape[2]
            horizontal_idx = np.random.randint(1, width)
            new_tile[:, :, width-horizontal_idx:width] = tile[:, :, 0:horizontal_idx]
            new_tile[:, :, 0:width-horizontal_idx] = tile[:, :, horizontal_idx:width]

        # Randomly re-arrange vertically
        if np.random.rand() < 0.5:
            height = tile.shape[1]
            vertical_idx = np.random.randint(1, height)
            new_tile[:, height-vertical_idx:height, :] = tile[:, 0:vertical_idx, :]
            new_tile[:, 0:height-vertical_idx, :] = tile[:, vertical_idx:height, :]

        return new_tile


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



class RandomFlipAndRotateSubtiles(object):
    def __call__(self, subtiles):
        # Generate random 0/1 integer for each sub-tile. 1 means that it should be
        # horizontally flipped.
        horizontal_flips = np.random.choice(2, size=subtiles.shape[0])
        vertical_flips = np.random.choice(2, size=subtiles.shape[0])

        # For each sub-tile, generate an integer from [0, 1, 2, 3]. This
        # represents how many times that sub-tile should be rotated.
        rotations = np.random.choice(4, size=subtiles.shape[0])

        # print('=================================')
        # print('subtile shape', subtiles.shape)
        # print('10th sub-tile: horizontal', horizontal_flips[10], 'vertical', vertical_flips[10],
        #       'rotations', rotations[10])
        # print('Before transformation:', subtiles[10, 0, :, :])
        # print('Upper left pixel:', subtiles[10, :, 0, 0])

        # Randomly horizontal flip
        subtiles[horizontal_flips == 1] = np.flip(subtiles[horizontal_flips == 1], axis=3).copy()
 
        # Randomly vertical flip
        subtiles[vertical_flips == 1] = np.flip(subtiles[vertical_flips == 1], axis=2).copy()

        for i in [1, 2, 3]:
            subtiles[rotations == i] = np.rot90(subtiles[rotations == i], k=i, axes=(2, 3)).copy()
 
        # print('After transformation:', subtiles[10, 0, :, :])
        # print('Upper left pixel:', subtiles[10, :, 0, 0])
        return subtiles  


class ResizeTile(object):
    def __init__(self, target_dim=[371,371], discrete_bands=list(range(12, 43))):
        self.target_dim = target_dim
        self.discrete_bands = discrete_bands

    def __call__(self, tile):
        # If tile is already of the desired size, do nothing
        if tile.shape[1] == self.target_dim[0] and tile.shape[2] == self.target_dim[1]:
            return tile

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

class ResizeTileRandom(object):
    def __init__(self, small_dim=50, large_dim=300, discrete_bands=list(range(12, 43))):
        self.small_dim = small_dim
        self.large_dim = large_dim
        self.discrete_bands = discrete_bands

    def __call__(self, tile):
        target_dim = random.randint(self.small_dim, self.large_dim)

        # Convert tile into Scikit learn order (channels as last dimension, not first)
        tile_numpy = np.moveaxis(tile, 0, -1)
        print('Numpy tile shape', tile_numpy.shape)
        resized_tile = resize(tile_numpy, [target_dim, target_dim], mode='edge')
        print('After reize', resized_tile.shape)
        resized_tile = np.moveaxis(resized_tile, -1, 0)
        print('Resized tile shape', resized_tile.shape)
        resized_tile[self.discrete_bands, :, :] = np.round(resized_tile[self.discrete_bands, :, :])
        return resized_tile


class RandomCrop(object):
    def __init__(self, crop_dim):
        self.crop_dim = crop_dim
    
    def __call__(self, tile):
        top_index = np.random.randint(0, tile.shape[1] - self.crop_dim)
        left_index = np.random.randint(0, tile.shape[2] - self.crop_dim)
        return tile[:, top_index:top_index+self.crop_dim, left_index:left_index+self.crop_dim]

class Cutout(object):
    def __init__(self, cutout_dim, prob, reflectance_indices=list(range(0, 9)), missing_reflectance_idx=-1):  # TODO Change!
        self.cutout_dim = cutout_dim
        self.prob = prob
        self.reflectance_indices = reflectance_indices
        self.missing_reflectance_idx = missing_reflectance_idx
    
    def __call__(self, tile):
        if random.random() < self.prob:
            # print('Cutout')
            top_index = np.random.randint(0, tile.shape[1] - self.cutout_dim)
            left_index = np.random.randint(0, tile.shape[2] - self.cutout_dim)
            tile[self.reflectance_indices, top_index:top_index+self.cutout_dim, left_index:left_index+self.cutout_dim] = 0
            tile[self.missing_reflectance_idx, top_index:top_index+self.cutout_dim, left_index:left_index+self.cutout_dim] = 1
        return tile


#https://www.usgs.gov/core-science-systems/nli/landsat/landsat-surface-reflectance-derived-spectral-indices?qt-science_support_page_related_con=0#qt-science_support_page_related_con
class ComputeVegetationIndices(object):
    def __call__(self, tile):
        ndvi = (tile[4] - tile[3]) / (tile[4] + tile[3])
        evi = 2.5 * ((tile[4] - tile[3]) / (tile[4] + 6 * tile[3] - 7.5 * tile[1] + 1))
        savi = ((tile[4] - tile[3]) / (tile[4] + tile[3] + 0.5)) * 1.5
        msavi = (2 * tile[4] + 1 - np.sqrt((2 * tile[4] + 1) ** 2 - 8 * (tile[4] - tile[3]))) / 2
        ndmi = (tile[4] - tile[5]) / (tile[4] + tile[5])
        # if np.isfinite(ndvi).any():
        #     print("NDVI nan", ndvi)
        # if np.isfinite(evi).any():
        #     print("EVI nan", evi)
        # if np.isfinite(savi).any():
        #     print("SAVI nan", savi)
        # if np.isfinite(msavi).any():
        #     print("MSAVI nan", msavi)
        # if np.isfinite(ndmi).any():
        #     print("NDMI nan", ndmi)
        
        new_tile = np.concatenate((tile[0:12],
                                   np.expand_dims(ndvi, axis=0),
                                   np.expand_dims(evi, axis=0),
                                   np.expand_dims(savi, axis=0),
                                   np.expand_dims(msavi, axis=0),
                                   np.expand_dims(ndmi, axis=0),
                                   tile[12:]), axis=0)
        new_tile = np.nan_to_num(new_tile, nan=0., posinf=0., neginf=0.)

        return new_tile



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
        bands, original_height, original_width = tile.shape
        resized_tile = np.zeros((bands, self.target_dim, self.target_dim))
        
        # Loop through each pixel in the target smaller tile
        for i in range(self.target_dim):
            for j in range(self.target_dim):
                # Find which pixels in original tile correspond to this pixel
                top = round((i / self.target_dim) * original_height)
                bottom = round(((i+1) / self.target_dim) * original_height)
                left = round((j / self.target_dim) * original_width)
                right = round(((j+1) / self.target_dim) * original_width)

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
