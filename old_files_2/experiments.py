import json
import numpy as np
import requests
import time

URI = "http://20.190.236.238:80/api/v1/service/myservice-aks/score"
key = 'taCahn3FNMAXFM5yT0BsupNwbaC4S4Nc'

headers = {'Content-Type':'application/json'}
headers['Authorization'] = f'Bearer {key}'

sample_tile = np.load("./datasets/tiles_2016-07-16/reflectance_lat_47.55_lon_-101.35.npy")
sample_tile = sample_tile[:, :100, :100]
print('Tile shape', sample_tile.shape)
sample_input = json.dumps({
    'data': sample_tile.tolist()
})

#service.reload()
sample_input = bytes(sample_input, encoding='utf8')

print('About to call')
before_request = time.time()
response = requests.post(URI, data=sample_input, headers=headers)
request_time = time.time()-before_request
print('Response status', response.status_code)

predictions = np.array(response.json()['sifs'])
print('Prediction shape', predictions.shape)
print('Prediction time:', response.json()['elapsed_time'])
print('Total request time:', request_time)