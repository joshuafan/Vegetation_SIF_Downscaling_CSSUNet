
# for l in 2e-4 5e-4 1e-3
# do
#     for w in 1e-4 3e-4 1e-3
#     do
#         for opt in AdamW
#         do
#             python3 train_downscaling_unet.py --prefix 10d_reduced10 --model unet2 --optimizer $opt --reduced_channels 10 \
#                 -lr $l -wd $w -sche const -epoch 100 -bs 64 --fraction_outputs_to_average 0.2 \
#                 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 -seed 0 --cutout --cutout_dim 20 --cutout_prob 0.5
#         done 
#     done 
# done

for m in 0 0.1 0.2 0.3 0.4
do
    python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 2e-4 -wd 1e-4 -sche const -epoch 100 -bs 64 \
        --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std $m -seed 0 \
        --cutout --cutout_dim 20 --cutout_prob 0.5
done

    # python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 3e-4 -wd 3e-4 -sche const -epoch 100 -bs 64 \
    #     --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 -seed $s \
    #     --cutout --cutout_dim 20 --cutout_prob 0.5

    # python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 3e-4 -wd 1e-4 -sche const -epoch 100 -bs 64 \
    #     --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 \
    #     -seed $s --cutout --cutout_dim 20 --cutout_prob 0.5

    # python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 1e-4 -wd 1e-3 \
    #     -sche const -epoch 100 -bs 50 --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 -seed $s \
    #     --cutout --cutout_dim 20 --cutout_prob 0.5
# done

# for s in 0 1 2
# do
#     train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 3e-4 -wd 1e-4 -sche const -epoch 5 -bs 64 \
#         --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 \
#         -seed $s --cutout --cutout_dim 20 --cutout_prob 0.5

#     # python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 1e-4 -wd 1e-3 \
#     #     -sche const -epoch 100 -bs 50 --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 -seed $s \
#     #     --cutout --cutout_dim 20 --cutout_prob 0.5
# done