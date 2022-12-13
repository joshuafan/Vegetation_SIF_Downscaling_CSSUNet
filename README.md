# Code for paper "Monitoring Vegetation at Extremely Fine Resolutions via Coarsely-Supervised Smooth U-Net Regression"

To train CS-SUNet in a coarsely-supervised way, `run_train_contrastive.sh` provides example usage. The 
main file for that is `train_contrastive.py`.

Other experimental settings:

- `run_train_pixel_nn.sh` trains a per-pixel MLP

- `run_train_vanilla_unet.sh` trains more vanilla U-Net approaches without smoothness loss or early stopping.

To evaluate a deep model at a fine resolution, see `run_eval.sh` and `eval_contrastive.py`.

To train averaging-based baselines and test them at a fine resolution, see `train_downscaling_averages.py`

# Installation instructions (not complete, TODO)

conda create --name sif
conda activate sif
conda install numpy pandas matplotlib
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge tqdm
