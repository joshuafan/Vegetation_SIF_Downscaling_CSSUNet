import tifffile

FILE_PATH = "datasets/GEE_data/imageToDriveExample.tif"

image_stack = tifffile.imread(FILE_PATH)
print(image_stack.shape)
print(image_stack.dtype)