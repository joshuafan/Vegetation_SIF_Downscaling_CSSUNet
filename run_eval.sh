
for s in 0
do
    for t in train
    do
        # CS-SUNet not normalized
        python3 eval_contrastive.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10f_contrastive_ABS_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.0001_fractionoutputs-1_seed-0_fliprotate_smoothnessloss-0.1-s0.5/model.ckpt \
            --model unet2_contrastive --seed $s --test_set $t        

        # CS-SUNet OLD (normalized)
        # python3 eval_contrastive.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10f_contrastive_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.0001_fractionoutputs-1_seed-0_fliprotate_smoothnessloss-0.5-s0.5/model.ckpt \
        #     --model unet2_contrastive --seed $s --test_set $t
        # python3 eval_contrastive.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10f_contrastive_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.0001_fractionoutputs-1_seed-${s}_fliprotate_smoothnessloss-0.5-s50.0/model.ckpt \
        #     --model unet2_contrastive --seed $s --test_set $t --plot_examples


        # python3 eval_contrastive.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10f_contrastive_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.0001_fractionoutputs-1_seed-${s}_fliprotate_smoothnessloss-0.2-s20.0/model.ckpt \
        # --model unet2_contrastive --seed $s --test_set $t 


        # # CS-SUNet without smoothness loss
        # python3 eval_contrastive.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10f_contrastive_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.0001_fractionoutputs-1_seed-${s}_fliprotate_smoothnessloss-0.0-s20.0/model.ckpt \
        #     --model unet2_contrastive --seed $s --test_set $t 

        # # Vanilla U-Net
        # python3 eval_contrastive.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10f_contrastive_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0001_wd-0.0001_fractionoutputs-1_seed-${s}_fliprotate_smoothnessloss-0.0-s50.0/model.ckpt_last \
        #     --model unet2_contrastive --seed $s --test_set $t

        # # Pixel NN
        # python3 eval_contrastive.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10h_pixel_nn_pixel_nn_optimizer-AdamW_bs-128_lr-0.001_wd-0.0001_fractionoutputs-1_seed-${s}_fliprotate/model.ckpt \
        #     --model pixel_nn --seed $s --test_set $t

        # # Fine supervision
        # python3 eval_contrastive.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10g_FULL_SUPERVISION_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.0001_fractionoutputs-1_seed-${s}_fliprotate_finesupervision/model.ckpt \
        #     --model unet2_contrastive --seed $s --test_set $t
    done
done
