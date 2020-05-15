# Send many asynchronous requests to the server at the same time
import asyncio
import json
import numpy as np
import requests
import time
import concurrent.futures

URI = "http://52.167.126.124:80/api/v1/service/myservice-aks-6/score"
KEY = "qipTFgoatuKDfOSaarwehJDWqfNtFNd2"

HEADERS = {'Content-Type':'application/json'}
HEADERS['Authorization'] = f'Bearer {KEY}'

async def main():

    sample_tile = np.load("./datasets/tiles_2016-07-16/reflectance_lat_47.55_lon_-101.35.npy")
    sample_tile = sample_tile[:, :100, :100]
    sample_input = json.dumps({
        'data': sample_tile.tolist()
    })

    sample_input = bytes(sample_input, encoding='utf8')
    
    def send_request(uri, data, headers, i):
        time.sleep(0.05*i)
        success = False
        while not success:
            response = requests.post(uri, data=data, headers=headers)
            print('Response status', response.status_code)
            if response.status_code == 200:
                success = True
            else:
                time.sleep(1)
        return response
        

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        start_time = time.time()
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor, 
                send_request, 
                URI,
                sample_input,
                HEADERS,
                i
            )
            for i in range(20)
        ]
        for response in await asyncio.gather(*futures):
            predictions = np.array(response.json()['sifs'])
            print('Prediction shape', predictions.shape)
        elapsed_time = time.time() - start_time
        print('Total elapsed time', elapsed_time)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())