for t in train test 
do
    python3 eval_contrastive.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10f_contrastive_SQUARE_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.0_fractionoutputs-1_seed-${1}_fliprotate_smoothnessloss-0.5-s0.5/model.ckpt \
        --model unet2_contrastive --seed ${1} --test_set $t --match_coarse
done

# python3 eval_contrastive.py --model_path /mnt/beegfs/bulk/mirror/jyf6/datasets/SIF/unet_results/10f_contrastive_SQUARE_unet2_contrastive_optimizer-AdamW_bs-128_lr-0.0003_wd-0.0001_fractionoutputs-1_seed-0_fliprotate_smoothnessloss-0.5-s0.5/model.ckpt \
#     --model unet2_contrastive --seed 0 --test_set train --match_coarse