
for s in 0
do
    for t in train
    do
        python3 eval_downscaling_unet.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10f_contrastive_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.0001_fractionoutputs-1.0_seed-0_fliprotate_multiplicativenoise-0.2_pixelcontrastive-0.01/model.ckpt_best_val_coarse  #unet_results/10d_unet2_optimizer-AdamW_bs-64_lr-0.001_wd-0.0_maxepoch-100_sche-const_fractionoutputs-0.2_seed-0_fliprotate_jigsaw_multiplicativenoise-0.2_cutout-20_prob-0.5/model.ckpt_best_val_coarse \
            --model unet2_contrastive --seed $s --test_set $t --plot_examples
    done
done
