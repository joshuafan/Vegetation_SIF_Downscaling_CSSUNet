# Train on TROPOMI only
python3 train_downscaling_unet.py --prefix 1d_TROPOMI --model unet2 --optimizer AdamW -lr 1e-6 -wd 0 -sche const -epoch 100 -bs 16 \
    --fraction_outputs_to_average 0.1 --flip_and_rotate --label_noise 0.05  --cutout --cutout_dim 50 --multiplicative_noise --mult_noise_std 0.2 # --smoothness_loss_contrastive --lambduh 1