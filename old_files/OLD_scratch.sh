
python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 1e-3 -wd 0 -sche const -epoch 100 -bs 64 \
    --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 -seed 0 \
    --cutout --cutout_dim 20 --cutout_prob 0.5


# for s in 0 1 2
# do 
#     python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 2e-4 -wd 1e-4 -sche const -epoch 100 -bs 64 \
#         --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.5 -seed $s \
#         --cutout --cutout_dim 20 --cutout_prob 0.5
# done

# --cutout --cutout_dim 20 --cutout_prob 0.5

# for m in 0.3 0.5 0.7 0.1 0
# do
#     python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 2e-4 -wd 1e-4 -sche const -epoch 100 -bs 64 \
#         --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std $m -seed 0 \
#         --cutout --cutout_dim 20 --cutout_prob 0.5

#     # python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 3e-4 -wd 3e-4 -sche const -epoch 100 -bs 64 \
#     #     --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 -seed $s \
#     #     --cutout --cutout_dim 20 --cutout_prob 0.5

#     # python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 3e-4 -wd 1e-4 -sche const -epoch 100 -bs 64 \
#     #     --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 \
#     #     -seed $s --cutout --cutout_dim 20 --cutout_prob 0.5

#     # python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 1e-4 -wd 1e-3 \
#     #     -sche const -epoch 100 -bs 50 --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 -seed $s \
#     #     --cutout --cutout_dim 20 --cutout_prob 0.5
# done