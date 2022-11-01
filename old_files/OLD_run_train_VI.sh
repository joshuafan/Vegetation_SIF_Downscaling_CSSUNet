# OLD - not working
python3 train_downscaling_unet.py --prefix 10d --model unet2 --optimizer Adam -lr 1e-4 -wd 1e-4 -sche const -epoch 50 -bs 64 \
    --fraction_outputs_to_average 1 --flip_and_rotate --jigsaw  --seed 0 #--compute_vi