import ee

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()


bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
print('Hi 1')

# Cloud masking function. Specifically, we mask pixels for which its 'pixel_qa'
# field has bit 3 (cloud shadow) or bit 5 (cloud) set to 1.
def maskL8sr(image):
  cloudShadowBitMask = ee.Number(2).pow(3).int()
  cloudsBitMask = ee.Number(2).pow(5).int()
  qa = image.select('pixel_qa')
  mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
    qa.bitwiseAnd(cloudsBitMask).eq(0))
  return image.updateMask(mask).select(bands).divide(10000)

# The image input data is a 2018 cloud-masked median composite.
cloud_masked_image = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
                  .filterDate('2018-08-01', '2018-08-15')
                  .map(maskL8sr)
                  .select(bands)
                  .max()
print('Masked clouds')

# Clip to the output image to the Iowa state boundary.
#fc = (ee.FeatureCollection('TIGER/2018/States')
#      .filter(ee.Filter().eq('NAME', 'Iowa')))
#print('Got Iowa boundary')
#iowa_image = cloud_masked_image.clipToCollection(fc)
#print('Clipped image')

## Define the visualization parameters.
#vizParams = {
#  'bands': ['B4', 'B3', 'B2'],
#  'min': 0,
#  'max': 0.5,
#  'gamma': [0.95, 1.1, 1]
#};

# Center the map and display the image.
#Map.setCenter(-122.1899, 37.5010, 10);  #  San Francisco Bay
#Map.addLayer(image, vizParams, 'false color composite');

# Change the following two lines to use your own training data.
#labels = ee.FeatureCollection('GOOGLE/EE/DEMOS/demo_landcover_labels')
#label = 'landcover'

# Sample the image at the points and add a random column.
#sample = iowa_image.sampleRegions(
#  collection=labels, properties=[label], scale=30).randomColumn()
#print('Sampled image')

# Partition the sample approximately 70-30.
#training = sample.filter(ee.Filter.lt('random', 0.7))
#testing = sample.filter(ee.Filter.gte('random', 0.7))

#from pprint import pprint

# Print the first couple points to verify.
#print('Training full!', training)
#pprint({'training': training.first().getInfo()})
#pprint({'testing': testing.first().getInfo()})


# # Use folium to visualize the imagery.
# mapIdDict = iowa_image.getMapId({'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3})
# map = folium.Map(location=[38., -122.5])
# folium.TileLayer(
#     tiles=mapIdDict['tile_fetcher'].url_format,
#     attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
#     overlay=True,
#     name='median composite',
#   ).add_to(map)
# map.add_child(folium.LayerControl())
# map.show_in_browser()

# Select the red, green and blue bands.
# image = iowa_image.select('B3', 'B2', 'B1')
# ee.mapclient.addToMap(image, {'gain': [1.4, 1.4, 1.1]})


# # Use folium to visualize the imagery.
# mapIdDict = image.getMapId({'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3})
# map = folium.Map(location=[38., -122.5])
# folium.TileLayer(
#     tiles=mapIdDict['tile_fetcher'].url_format,
#     attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
#     overlay=True,
#     name='median composite',
#   ).add_to(map)
# map.add_child(folium.LayerControl())
# map
