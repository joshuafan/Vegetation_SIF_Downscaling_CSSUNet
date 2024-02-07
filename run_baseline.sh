for m in Random_Forest  # Ridge_Regression Gradient_Boosting_Regressor MLP Nearest_Neighbors
do
    python3 train_downscaling_averages.py --method $m --standardize --match_coarse
done