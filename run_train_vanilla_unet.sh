# Final params
for s in 1 2
do
    # Best params - early stopping
    python3 train.py --prefix 10d_unet --model unet2_contrastive --optimizer AdamW -lr 3e-4 -wd 1e-3 -epoch 100 -bs 128 --flip_and_rotate --seed $s

    # Best params - no early stopping
    python3 train.py --prefix 10d_unet --model unet2_contrastive --optimizer AdamW -lr 1e-4 -wd 0 -epoch 100 -bs 128 --flip_and_rotate --seed $s
done


# Hyperparam tuning
# for w in 1e-3 0 1e-4
# do
#     for lr in 1e-4 3e-4 1e-3
#     do
#         python3 train.py --prefix 10d_unet --model unet2_contrastive --optimizer AdamW -lr $lr -wd $w -epoch 100 -bs 128 \
#             --flip_and_rotate --visualize --seed 0
#     done
# done
