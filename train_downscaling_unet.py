

# Read band statistics

# MSELoss: reduction='sum'
# Read input tiles (standardized)

# Pass inputs through model

# AvgPool to get coarse-resolution predictions

# Flatten predictions, labels, and label mask. Extract indices where mask is False (0). Run MSELoss on those.
# Don't forget to record the number of valid indices / datapoints.

# Compute epoch loss. Divide loss by number of valid datapoints.