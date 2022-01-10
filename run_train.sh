# for m in 0 0.1 0.2 0.3 0.4
# do
#     python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 2e-4 -wd 1e-4 -sche const -epoch 100 -bs 64 \
#         --fraction_outputs_to_average 0.2 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std $m -seed 0 \
#         --cutout --cutout_dim 20 --cutout_prob 0.5
# done

# 10d_no_FLDAS
# 10d_no_crop
# 10d_no_G_R_NIR
# 10d_no_G
# python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer Adam -lr 2e-4 -wd 1e-4 -sche const -epoch 50 -bs 64 \
#     --fraction_outputs_to_average 1 --flip_and_rotate --jigsaw --multiplicative_noise --mult_noise_std 0.2 \
#     --seed 0 --label_noise 0.05  #--cutout --cutout_dim 25 --cutout_prob 1 #--gradient_penalty --lambduh 1e-6

    #--label_noise 0.05 # --gradient_penalty --lambduh 1e-4
    # --similarity_loss --similarity_temp 0.2
    # --cutout --cutout_dim 20 --cutout_prob 0.5 --recon_loss

for l in 0.01 0.1 1 10 0
do
    python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer AdamW -lr 1e-4 -wd 0 -sche const -epoch 100 -bs 64 \
        --fraction_outputs_to_average 1 --smoothness_loss --lambduh $l --flip_and_rotate --multiplicative_noise --mult_noise_std 0.2
done