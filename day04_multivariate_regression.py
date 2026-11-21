"""This program runs the multi variate regression"""
def train_multivariate_regression(x, y, learning_rate, epochs):
    '''
    Multivariate analysis to find weights and basis
    '''
    w1 = 0.0
    w2 = 0.0
    b = 0.0
    if len(x) == 0 or len(y) == 0 or len(x) != len(y) \
    or not isinstance(learning_rate, (int, float)) \
    or not isinstance(epochs, int) or learning_rate <=0 or epochs <= 0:
        return None
    num_vars = len(x[0])
    if num_vars != 2 or len([i for i in x if len(i) != num_vars]) > 0:
        return None
    while epochs > 0:
        grad_w1 = 0.0
        grad_w2 = 0.0
        grad_b = 0.0
        for x_val, y_val in zip(x, y):
            e = (w1 * x_val[0]) + (w2 * x_val[1]) + b - y_val
            grad_w1 += x_val[0] * e
            grad_w2 += x_val[1] * e
            grad_b += e
        grad_w1 = 2 / len(x) * grad_w1
        grad_w2 = 2 / len(x) * grad_w2
        grad_b = 2 / len(x) * grad_b
        w1 = w1 - (learning_rate * grad_w1)
        w2 = w2 - (learning_rate * grad_w2)
        b = b - (learning_rate * grad_b)
        epochs -= 1
    return w1, w2, b

X = [
    [1, 1],
    [2, 1],
    [2, 2],
    [3, 2],
    [4, 3]
]
Y = [6, 8, 11, 13, 18]
LR = 0.05
EPOCHS = 10000
W1, W2, BIAS = train_multivariate_regression(X, Y, LR, EPOCHS)
print (f"Final Values - \nw1: {W1};\nw2: {W2};\nb: {BIAS}") 