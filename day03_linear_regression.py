"""Implementation of Linear Regression - Basic Maths"""

def train_linear_regression(x, y, learning_rate, epochs):
    '''
    This function calculates the bias and weight at each epoch and returns the final values
    '''
    w = 0.0
    b = 0.0

    if len(x) == 0 or len(y) == 0 or len(x) != len(y) or not isinstance(epochs, (int, float)) \
        or not isinstance(learning_rate, (int, float)):
        return None, None

    while epochs > 0:
        grad_w = 0.0
        grad_b = 0.0
        for x_val, y_val in zip(x, y):
            grad_w += x_val * ((w * x_val) + b - y_val)
            grad_b += (w * x_val) + b - y_val
        grad_w = 2 / len(x) * grad_w
        grad_b = 2 / len(x) * grad_b
        w = w - (learning_rate * grad_w)
        b = b - (learning_rate * grad_b)
        epochs -= 1
    return w, b

X = [1, 2, 3, 4]
Y = [5, 8, 11, 14]
LR = 0.05
EPOCHS = 2000
W, B = train_linear_regression(X, Y, LR, EPOCHS)
print(f'Final Weight and Bias are - \nWeight: {W};\nBias: {B}')