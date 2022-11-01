for l in 0.5
do
    for s in 50
    do
        for lr in 1e-4 3e-4 1e-3
        do
            for m in 0.1 0.2
            do
                python3 train_contrastive.py --prefix 10f_contrastive --model unet2_contrastive --optimizer AdamW -lr $lr -wd 1e-4 -epoch 100 -bs 128 \
                    --flip_and_rotate --smoothness_loss --lambduh $l --spread $s --num_pixels 1000 --visualize  --multiplicative_noise --mult_noise_std $m
            done
        done
    done
done


# # Smoothness loss
# for l in 0.01 0.1
# do
#     for s in 20 100
#     do
#         for lr in 1e-4 3e-4 1e-3
#         do
#             python3 train_contrastive.py --prefix 10f_contrastive --model unet2_contrastive --optimizer AdamW -lr $lr -wd 1e-4 -sche const -epoch 60 -bs 128 \
#                 --flip_and_rotate --smoothness_loss --lambduh $l --spread $s --num_pixels 1000
#         done
#     done
# done
