# Smoothness loss
for l in 0.1
do
    python3 train_contrastive.py --prefix 10e_smoothness --model unet2_contrastive --optimizer AdamW -lr 5e-4 -wd 1e-4 -sche const -epoch 100 -bs 64 \
        --fraction_outputs_to_average 1 --flip_and_rotate --pixel_contrastive_loss --lambduh $l --multiplicative_noise --mult_noise_std 0.2
done