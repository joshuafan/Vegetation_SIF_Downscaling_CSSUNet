# Super Fine-Resolution SIF via Coarsely-Supervised U-Net Regression

Code for the paper "Super Fine-Resolution SIF via Coarsely-Supervised U-Net Regression", as well as a subset of the dataset. (Because CMT limits file size to 100MB, we can only upload 500 tiles, even though our paper uses over 3000 tiles.)

To run the code, first change to the `src` directory.

## Baselines

To run baselines, run

`./run_baselines.sh`

A summary of the results will be written to `baseline_results/results_summary_BASELINE.csv` (inside `src`). 

## CSR-U-Net
To train CSR-U-Net, run

`./run_train.sh`

To evaluate CSR-U-Net, run

`./run_eval.sh`

A summary of the results will be written to `unet_results/results_summary_EVAL_train.csv` (inside `src`).

## Dependencies

This code was tested on Python 3.7 on Linux (CentOS 7), using the following libraries:

- Matplotlib 3.3.4
- Numpy 1.18.1
- Pandas 1.1.3
- PyTorch 1.7.0
- Scikit-Learn 0.24.1
- Scipy 1.4.1


