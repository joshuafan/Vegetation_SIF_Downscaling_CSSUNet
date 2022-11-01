

# python3 train_downscaling_averages.py --method Ridge_Regression # --multiplicative_noise --mult_noise_std 0.2 --mult_noise_repeats 10

for m in Random_Forest  # Ridge_Regression Gradient_Boosting_Regressor MLP Nearest_Neighbors
do
    python3 train_downscaling_averages.py --method $m --standardize --match_coarse   # --multiplicative_noise --mult_noise_std 0.2 --mult_noise_repeats 10
done