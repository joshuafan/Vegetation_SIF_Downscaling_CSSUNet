#%%
import azureml.core
azureml.core.VERSION
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
import json
import numpy as np
from azureml.core.webservice import LocalWebservice

# %%
# Create workspace
ws = Workspace.from_config(path="./datasets/config.json")

# Register the model with Azure's machine learning workspace
model = Model.register(model_path = "./models",
                       model_name = "subtile_sif",
                       description = "Predicts SIF for subtile",
                       workspace = ws)

# %%
# Create the environment
myenv = Environment(name="myenv")
conda_dep = CondaDependencies()

# Define the packages needed by the model and scripts
conda_dep.add_conda_package("pytorch")
conda_dep.add_conda_package("torchvision")
conda_dep.add_conda_package("numpy")
conda_dep.add_conda_package("pandas")
#conda_dep.add_conda_package("scikit-learn")
# You must list azureml-defaults as a pip dependency
conda_dep.add_pip_package("azureml-defaults")
conda_dep.add_pip_package("keras")
conda_dep.add_pip_package("gensim")

# Adds dependencies to PythonSection of myenv
myenv.python.conda_dependencies=conda_dep

inference_config = InferenceConfig(entry_script="score.py",
                                   environment=myenv)


# %%
# Deployment config
deployment_config = LocalWebservice.deploy_configuration()

# Deploy model
#model = Model(ws, name='subtile_sif')
service = Model.deploy(ws, 'myservice', [model], inference_config, deployment_config)
service.wait_for_deployment(True)
print(service.state)
print("scoring URI: " + service.scoring_uri)
print('Local service port: {}'.format(service.port))
print(service.get_logs())

# %%
sample_tile = np.load("./datasets/tiles_2016-07-16/reflectance_lat_47.55_lon_-101.35.npy")
sample_input = json.dumps({
    'data': sample_tile.tolist()
})

sample_input = bytes(sample_input, encoding='utf-8')
service.run(input_data=sample_input)