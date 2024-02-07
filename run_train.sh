# Smoothness loss
for l in 0.5
do
    for s in 0.5
    do
        for w in 1e-3 0
        do
            for lr in 3e-4
            do
                for seed in 0 1 2
                do
                    python3 train.py --prefix 10f_contrastive_SQUARE --model unet2_contrastive --optimizer AdamW -lr $lr -wd $w -epoch 100 -bs 128 \
                        --flip_and_rotate --smoothness_loss --lambduh $l --spread $s --num_pixels 1000 --visualize --seed $seed
                done
            done
        done
    done
done
