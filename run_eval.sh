
# Evaluates trained models on the given fine-resolution test sets
for t in train
do
    for s in 0
    do
        # CS-SUNet 
        python3 eval.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10f_contrastive_SQUARE_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.0001_fractionoutputs-1_seed-${s}_fliprotate_smoothnessloss-0.5-s0.5/model.ckpt \
            --model unet2_contrastive --seed $s --test_set $t --match_coarse --plot_examples

        # # CS-SUNet without smoothness loss
        # python3 eval.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10d_unet_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.001_fractionoutputs-1_seed-${s}_fliprotate/model.ckpt \
        #     --model unet2_contrastive --seed $s --test_set $t --match_coarse

        # # Vanilla U-Net (without smoothness loss and early stopping)
        # python3 eval.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10d_unet_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0001_wd-0.0_fractionoutputs-1_seed-${s}_fliprotate/model.ckpt_last \
        #     --model unet2_contrastive --seed $s --test_set $t --match_coarse

        # # Fine supervision
        # python3 eval.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10g_FULL_SUPERVISION_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.001_wd-0.001_fractionoutputs-1_seed-${s}_fliprotate_finesupervision/model.ckpt \
        #     --model unet2_contrastive --seed $s --test_set $t --match_coarse

        # # Pixel NN
        # python3 eval.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10h_pixel_nn_pixel_nn_optimizer-AdamW_bs-128_lr-0.001_wd-0.001_fractionoutputs-1_seed-${s}_fliprotate/model.ckpt \
        #     --model pixel_nn --seed $s --test_set $t --match_coarse

    done
done
