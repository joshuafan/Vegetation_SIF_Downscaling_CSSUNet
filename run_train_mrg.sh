# for l in 0.5
# do
#     for s in 0.5
#     do
#         for w in 1e-4
#         do
#             for lr in 1e-4 1e-3 1e-2 1e-1
#             do
#                 for seed in 0
#                 do
#                     for model in unet2_contrastive
#                     do
#                         python3 train_contrastive.py --prefix 10f --model $model --optimizer AdamW -lr $lr -wd $w -epoch 100 -bs 256 \
#                             --flip_and_rotate --seed $seed  --smoothness_loss --lambduh $l --spread $s --num_pixels 1000

#                         python3 train_contrastive.py --prefix 10f --model $model --optimizer AdamW -lr $lr -wd $w -epoch 100 -bs 256 \
#                             --flip_and_rotate --seed $seed  --smoothness_loss --lambduh $l --spread $s --num_pixels 1000 --batch_norm
#                     done
#                 done
#             done
#         done
#     done
# done

for l in 0.5
do
    for s in 0.5
    do
        for w in 1e-4
        do
            for lr in 1e-4 3e-4 1e-3 3e-3
            do
                for seed in 0
                do
                    for model in mrg_unet_plus_plus
                    do
                        python3 train_contrastive.py --prefix 10j_mrg --model $model --optimizer AdamW -lr $lr -wd $w -epoch 100 -bs 64 \
                            --flip_and_rotate --seed $seed  --smoothness_loss --lambduh $l --spread $s --num_pixels 1000 --dropout_prob 0.1
                    done
                done
            done
        done
    done
done

# for l in 0.5
# do
#     for s in 0.5
#     do
#         for w in 1e-4
#         do
#             for lr in 1e-4 1e-3 1e-2 1e-1
#             do
#                 for seed in 0
#                 do
#                     for model in mrg_unet
#                     do
#                         python3 train_contrastive.py --prefix 10j_mrg --model $model --optimizer AdamW -lr $lr -wd $w -epoch 100 -bs 128 \
#                             --flip_and_rotate --seed $seed  --smoothness_loss --lambduh $l --spread $s --num_pixels 1000 --dropout_prob 0.1
#                     done
#                 done
#             done
#         done
#     done
# done


# Next: full batchnorm experiment (bs 256, with/without batchnorm, 1e-4 1e-3 1e-2 1e-1)
# Next: MRG UNet and UNet++, batchnorm, bs256, 1e-4 1e-3 1e-2 1e-1 
