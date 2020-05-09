import requests
import json
import numpy as np
import matplotlib.pyplot as plt

LAT = 47.55
LON = -101.35
LAT_LON = 'lat_' + str(LAT) + '_lon_' + str(LON)

scoring_uri = 'scoring uri for your service'
headers = {'Content-Type':'application/json'}


sample_tile = np.load("./datasets/tiles_2016-07-16/reflectance_lat_" + LAT_LON + ".npy")
sample_input = json.dumps({
    'data': sample_tile.tolist()
})

response = requests.post(scoring_uri, data=sample_input, headers=headers)
print(response.status_code)
print(response.elapsed)
print(response.json())

plt.imshow(response.sifs, cmap='Greens', vmin=0, vmax=1.5)
plt.savefig("exploratory_plots/" + LAT_LON + "_subtile_sif_map_9.png")
plt.close()
