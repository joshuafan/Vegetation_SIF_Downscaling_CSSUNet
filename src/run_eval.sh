# Evaluates pre-trained CSR-U-Net on a given fine-resolution set (set "--test_set" to train, val, or test).

for s in 0 1 2
do
    python3 eval_downscaling_unet.py --model_path unet_results/10d_unet2_optimizer-AdamW_bs-64_lr-0.0002_wd-0.0001_maxepoch-100_sche-const_fractionoutputs-0.2_seed-${s}_fliprotate_jigsaw_multiplicativenoise-0.2_cutout-20_prob-0.5/model.ckpt \
        --model unet2 --seed $s --test_set train
done