# Trains baselines (that use tile-average features) and reports results at different resolutions.

# To add multiplicative noise, you can add "--multiplicative_noise --mult_noise_std 0.2 --mult_noise_repeats 10",
# where mult_noise_std is the standard deviation of the multiplicative factor, and mult_noise_repeats is how many
# times to repeat each data point (with a different random noise factor)

for m in Ridge_Regression Gradient_Boosting_Regressor MLP
do
    python3 train_downscaling_averages.py --method $m
done