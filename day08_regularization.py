"""Ridge Regularization with regression with n variables"""
def train_ridge_regression(x, y, learning_rate, epochs, lambda_reg):
    '''
    Runs L2 regression with multivariate regression to identify weights
    '''
    # Run Validation Checks
    if len(x) == 0 or len(y) == 0 or len(x) != len(y) \
    or not isinstance(learning_rate, (int, float)) \
    or not isinstance(epochs, int) or learning_rate <= 0 \
    or epochs <= 0 or not isinstance(lambda_reg, (int, float)) \
    or lambda_reg < 0:
        return None
    num_vars = len(x[0])
    if num_vars == 0 or any(len(i) != num_vars for i in x):
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
            grad[k] = grad[k] + 2 * lambda_reg * weights[k]
        grad_b = 2 / len(x) * grad_b
        for k in range(num_vars):
            weights[k] = weights[k] - (learning_rate * grad[k])
        bias = bias - (learning_rate * grad_b)
        epochs -= 1
    return weights, bias


X = [
    [1, 0],
    [2, 1],
    [3, 0],
    [4, 1],
    [5, 0],
    [6, 1]
]
Y = [4, 8, 8, 12, 12, 16]
LR = 0.01
EPOCHS = 10000
LAMBDA = 0.1
WTS, BIAS = train_ridge_regression(X,Y,LR,EPOCHS,LAMBDA)
print(f"Final Values -\nweights: {WTS};\nbias: {BIAS}")