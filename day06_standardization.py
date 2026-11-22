"""Standardizing Feature Vectors"""

def standardize_features(x):
    '''
    Returns the Z-scaled feature vectors and the means and standard deviations of them
    '''
    # Validate Input Data
    if len(x) == 0:
        return None
    num_vars = len(x[0])
    if num_vars == 0 or any(i != num_vars for i in x):
        return None
    # Get Means
    means = [0.0] * num_vars
    stds = [0.0] * num_vars
    x_scaled = []
    for x_val in x:
        for k in range(num_vars):
            means[k] += x_val[k]
    means = [i/len(x) for i in means]
    # Get Standard Deviations
    for x_val in x:
        for k in range(num_vars):
            stds[k] += (x_val[k] - means[k]) ** 2
    stds = [(i/(len(x))) ** 0.5 for i in stds]
    # Validate Standard Deviation
    if any(i==0 for i in stds):
        return None
    # Z Score Normalization
    for x_val in x:
        x_scaled_val = [0.0] * num_vars
        for k in range(num_vars):
            x_scaled_val[k] = (x_val[k] - means[k]) / stds[k]
        x_scaled.append(x_scaled_val)
    return x_scaled, means, stds

X = [
    [1000, 2],
    [1500, 3],
    [2000, 3],
    [2500, 4],
    [3000, 5]
]
X_SCALED, MEANS, STDS = standardize_features(X)
print('Scaled Values Are - ', X_SCALED)
print('Means are - ', MEANS)
print('Stds are - ', STDS)  