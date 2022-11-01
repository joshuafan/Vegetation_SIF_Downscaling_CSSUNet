for s in 1 2
do
    python3 train_contrastive.py --prefix 10f_contrastive_SQUARE --model unet2_contrastive --optimizer AdamW -lr 3e-4 -wd 1e-4 -epoch 100 -bs 128 \
        --flip_and_rotate --smoothness_loss --lambduh 0.5 --spread 0.5 --num_pixels 1000 --visualize --seed $s
done