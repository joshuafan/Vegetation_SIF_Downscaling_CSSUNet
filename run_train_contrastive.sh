# Smoothness loss
for l in 0.1
do
    for s in 10 50
    do
        for lr in  3e-4  #1e-4 3e-4 1e-3
        do
            python3 train_contrastive.py --prefix 10f_contrastive --model unet2_contrastive --optimizer AdamW -lr $lr -wd 1e-4 -sche const -epoch 100 -bs 128 \
                --flip_and_rotate --smoothness_loss --lambduh $l --spread $s --num_pixels 1000 --visualize
        done
    done
done


# # Smoothness loss contrastive - DOES NOT WORK
# for l in 0.01 0.1 1
# do
#     for s in 0.05
#     do
#         for lr in 1e-4 3e-4 1e-3
#         do
#             python3 train_contrastive.py --prefix 10f_contrastive --model unet2_contrastive --optimizer AdamW -lr $lr -wd 1e-4 -sche const -epoch 100 -bs 128 \
#                 --flip_and_rotate --smoothness_loss_contrastive --lambduh $l --similarity_threshold $s --temperature 0.2 --num_pixels 1000 --visualize
#         done
#     done
# done



# # Try no FLDAS
# python3 train_contrastive.py --prefix 10f_contrastive --model unet2_contrastive --optimizer AdamW -lr 3e-4 -wd 1e-4 -sche const -epoch 50 -bs 128 \
#     --flip_and_rotate --visualize

# # What type of multiplicative noise
# for m in 0.2 0.3 0.4
# do
#     python3 train_contrastive.py --prefix 10f_contrastive_without_thermal --model unet2_contrastive --optimizer AdamW -lr 3e-4 -wd 1e-4 -sche const -epoch 50 -bs 128 \
#         --flip_and_rotate --multiplicative_noise_end --mult_noise_std $m --visualize
# done



# # Pretrain
# for lr in 1e-4 3e-4 1e-3
# do
#     python3 train_contrastive.py --prefix 10f_PRETRAIN20_contrastive --model unet2_contrastive --optimizer AdamW -lr $lr -wd 1e-4 -sche const -epoch 100 -bs 128 \
#         --fraction_outputs_to_average 1 --flip_and_rotate --pixel_contrastive_loss --lambduh 0.001 --temperature 0.2 --multiplicative_noise --mult_noise_std 0.2 #--num_workers 1
# done



# # Smoothness loss
# for l in 0 0.001 0.01 0.1 1
# do
#     python3 train_contrastive.py --prefix 10f_contrastive --model unet2_contrastive --optimizer AdamW -lr 2e-4 -wd 1e-4 -sche const -epoch 100 -bs 128 \
#         --fraction_outputs_to_average 1 --flip_and_rotate --pixel_contrastive_loss --lambduh $l --temperature 0.2  --multiplicative_noise --mult_noise_std 0.2
# done