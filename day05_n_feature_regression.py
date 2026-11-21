"""Multivariate regression with n variables"""
def train_linear_regression(x, y, learning_rate, epochs):
    '''
    Runs multivariate regression to identify weights, bias
    '''
    # Run Validation Checks
    if len(x) == 0 or len(y) == 0 or len(x) != len(y) \
    or not isinstance(learning_rate, (int, float)) \
    or not isinstance(epochs, int) or learning_rate <= 0 \
    or epochs <= 0:
        return None
    num_vars = len(x[0])
    if num_vars == 0 or len([i for i in x if len(i) != num_vars]) > 0:
        return None
    # Weights and Bias Calculation
    weights = [0.0] * num_vars
    bias = 0.0
    while epochs > 0:
        grad = [0.0] * num_vars
        grad_b = 0.0
        for x_val, y_val in zip(x, y):
            e = 0.0
            for k in range(num_vars):
                e += weights[k] * x_val[k]
            e = e + bias - y_val
            for k in range(num_vars):
                grad[k] += e * x_val[k]
            grad_b += e
        for k in range(num_vars):
            grad[k] = 2 / len(x) * grad[k]
        grad_b = 2 / len(x) * grad_b
        for k in range(num_vars):
            weights[k] = weights[k] - (learning_rate * grad[k])
        bias = bias - (learning_rate * grad_b)
        epochs -= 1
    return weights, bias


X = [
    [1, 1, 2],
    [2, 1, 3],
    [3, 2, 1],
    [4, 3, 2],
    [5, 2, 4],
    [6, 4, 3]
]
Y = [9, 12, 12, 17, 25, 26]
LR = 0.01
EPOCHS = 10_000
WTS, BIAS = train_linear_regression(X,Y,LR,EPOCHS)
print(f"Final Values -\nweights: {WTS};\nbias: {BIAS}")