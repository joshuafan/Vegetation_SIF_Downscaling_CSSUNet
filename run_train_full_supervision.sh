# Final run with additional seeds
for s in 1 2
do
    python3 train_contrastive.py --prefix 10g_FULL_SUPERVISION --model unet2_contrastive --optimizer AdamW -lr 1e-3 -wd 1e-3 \
        -epoch 100 -bs 128 --flip_and_rotate --visualize --fine_supervision --seed $s
done


# Hyperparam tuning
# for lr in 1e-4 3e-4 1e-3
# do
#     for w in 0 1e-4 1e-3
#     do
#         python3 train_contrastive.py --prefix 10g_FULL_SUPERVISION --model unet2_contrastive --optimizer AdamW -lr $lr -wd $w -epoch 100 -bs 128 \
#             --flip_and_rotate --visualize --fine_supervision --seed 0
#     done
# done
