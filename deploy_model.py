# %%
import azureml.core
azureml.core.VERSION
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
import json
import numpy as np
import requests
from azureml.core.webservice import LocalWebservice

# %%
# Create workspace
ws = Workspace.from_config(path="./datasets/config.json")

# %%
# Create AKS cluster
from azureml.core.compute import AksCompute, ComputeTarget

# Use the default configuration (you can also provide parameters to customize this).
# For example, to create a dev/test cluster, use:
# prov_config = AksCompute.provisioning_configuration(cluster_purpose = AksCompute.ClusterPurpose.DEV_TEST)
prov_config = AksCompute.provisioning_configuration()

aks_name = 'myaks'
# Create the cluster
aks_target = ComputeTarget.create(workspace = ws,
                                    name = aks_name,
                                    provisioning_configuration = prov_config)

# Wait for the create process to complete
aks_target.wait_for_completion(show_output = True)

# %%
# Create AKS cluster (GPU)
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.exceptions import ComputeTargetException

# Choose a name for your cluster
aks_name = "aks-gpu"

# Check to see if the cluster already exists
try:
    gpu_aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    # Provision AKS cluster with GPU machine
    prov_config = AksCompute.provisioning_configuration(vm_size="Standard_NC6")

    # Create the cluster
    gpu_aks_target = ComputeTarget.create(
        workspace=ws, name=aks_name, provisioning_configuration=prov_config
    )

    gpu_aks_target.wait_for_completion(show_output=True)

# %%
# Register the model with Azure's machine learning workspace
#model = Model.register(model_path = "./models",
#                       model_name = "subtile_sif",
#                       description = "Predicts SIF for subtile",
#                       workspace = ws)

# %%
# Create the environment
myenv = Environment(name="myenv")
conda_dep = CondaDependencies()

# Define the packages needed by the model and scripts
#conda_dep.add_conda_package("torchvision")
#conda_dep.add_conda_package("json")
conda_dep.add_conda_package("numpy")
#conda_dep.add_conda_package("os")
conda_dep.add_conda_package("pandas")
conda_dep.add_conda_package("pytorch")
conda_dep.add_conda_package("scikit-image")
conda_dep.add_conda_package("torchvision")
#conda_dep.add_conda_package("time")

#conda_dep.add_conda_package("scikit-learn")
# You must list azureml-defaults as a pip dependency
conda_dep.add_pip_package("azureml-defaults")
#conda_dep.add_pip_package("keras")
#conda_dep.add_pip_package("gensim")

# Adds dependencies to PythonSection of myenv
myenv.python.conda_dependencies=conda_dep

inference_config = InferenceConfig(entry_script="score.py",
                                   source_directory=".",
                                   environment=myenv)


# %%
# AKS deploy
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import Model

aks_target = AksCompute(ws,"myaks")
# If deploying to a cluster configured for dev/test, ensure that it was created with enough
# cores and memory to handle this deployment configuration. Note that memory is also used by
# things such as dependencies and AML components.
deployment_config = AksWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)



model = Model(ws, name='subtile_sif')
service = Model.deploy(ws, "myservice-aks", [model], inference_config, deployment_config, aks_target)
service.wait_for_deployment(show_output = True)
print(service.state)
print(service.get_logs())

# %%
# AKS deploy (GPU)
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import Model
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment, DEFAULT_GPU_IMAGE

myenv.docker.base_image = DEFAULT_GPU_IMAGE
gpu_inference_config = InferenceConfig(entry_script="score.py", environment=myenv)

gpu_aks_target = AksCompute(ws,"aks-gpu")
# If deploying to a cluster configured for dev/test, ensure that it was created with enough
# cores and memory to handle this deployment configuration. Note that memory is also used by
# things such as dependencies and AML components.
gpu_aks_config = AksWebservice.deploy_configuration(autoscale_enabled=False,
                                                    num_replicas=3,
                                                    cpu_cores=2,
                                                    memory_gb=4)


model = Model(ws, name='subtile_sif')
service = Model.deploy(ws, "myservice-aks", [model], gpu_inference_config, gpu_deployment_config, gpu_aks_target)
service.wait_for_deployment(show_output = True)
print(service.state)
print(service.get_logs())

# %%
# Local deploy
deployment_config = LocalWebservice.deploy_configuration(port=8890)

# Deploy model
model = Model(ws, name='subtile_sif')
service = Model.deploy(ws, 'myservice', [model], inference_config, deployment_config)
service.wait_for_deployment(show_output=True)
print(service.state)
print("scoring URI: " + service.scoring_uri)
print('Local service port: {}'.format(service.port))

# %%

headers = {'Content-Type':'application/json'}
sample_tile = np.load("./datasets/tiles_2016-07-16/reflectance_lat_47.55_lon_-101.35.npy")
print('Tile shape', sample_tile.shape)
sample_input = json.dumps({
    'data': sample_tile.tolist()
})

#service.reload()
sample_input = bytes(sample_input, encoding='utf8')
try:
    prediction = service.run(input_data=sample_input)
    print('PREDICTION!', prediction)
except Exception as e:
    print('LOGS!', service.get_logs())

# response = requests.post(service.scoring_uri, data=sample_input, headers=headers)
# print(response.status_code)
# print(response.elapsed)
# print(response.json())

# %%
import matplotlib.pyplot as plt

RGB_BANDS = [3, 2, 1]

img = np.moveaxis(sample_tile[RGB_BANDS, :, :], 0, -1)
print(img.shape)
plt.imshow(img / 1000)



# %%
prediction_array = np.array(prediction['sifs']) #, dtype=np.float)
sif_cmap = plt.get_cmap('YlGn')
sif_cmap.set_bad(color='red')
plt.imshow(prediction_array, cmap=sif_cmap, vmin=0.0, vmax=1.0)


# %%
uri = service.scoring_uri
print(uri)
print(service.get_keys())
