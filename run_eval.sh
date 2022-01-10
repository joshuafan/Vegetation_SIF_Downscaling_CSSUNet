
for s in 0
do
    for t in train
    do
        python3 eval_downscaling_unet.py --model_path unet_results/10d_unet2_optimizer-AdamW_bs-64_lr-0.001_wd-0.0_maxepoch-100_sche-const_fractionoutputs-0.2_seed-0_fliprotate_jigsaw_multiplicativenoise-0.2_cutout-20_prob-0.5/model.ckpt_best_val_coarse \
            --model unet2 --seed $s --test_set $t --plot_examples
    done
done
