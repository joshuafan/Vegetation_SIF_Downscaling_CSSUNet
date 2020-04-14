import numpy as np
import torch

class StandardizeTile(object):
    """
    Standardizes images so that each band has mean 0, standard deviation 1 
    """
    def __init__(self, band_means, band_stds):
        self.band_means = band_means[:, np.newaxis, np.newaxis]
        self.band_stds = band_stds[:, np.newaxis, np.newaxis]

    def __call__(self, tile):
        return (tile - self.band_means) / self.band_stds


class ShrinkTile(object):
    """
    Shrinks tile down to designated size (e.g. from 371 x 371 to 10 x 10).
    Contains custom logic to ensure that binary bands (1/0) stay that way.
    """
    def __init__(self, target_dim=10, continuous_bands=list(range(0, 12)), cover_bands=list(range(12, 27)), missing_band=[27, 28]):
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
                resized_tile[:, i, j] = tile[:, int((top+bottom)/2), int((left+right)/2)]
                continue

                # Count number of pixels with and without reflectance data (by looking at the missing_band mask)
                pixels_without_reflectance = np.sum(original_pixels[self.missing_band, :, :])
                pixels_with_reflectance = (original_pixels.shape[1] * original_pixels.shape[2]) - pixels_without_reflectance
                # print('Pixels without', pixels_without_reflectance, 'Pixels with', pixels_with_reflectance)

                # Take the average reflectance value, but only consider nonzero pixels (where data is NOT missing) in denominator.
                # Note: if no pixels contain reflectance, make the new pixel values 0 (already 
                # set by default)
                # if pixels_with_reflectance >= 1:
                #   resized_tile[self.continuous_bands, i, j] = np.sum(original_pixels[self.continuous_bands, :, :], axis=(1,2)) / pixels_with_reflectance
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
                if value > 0.3:
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
