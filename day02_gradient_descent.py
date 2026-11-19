"""Implementation of Gradient Descent using MSE"""

def gradient_descent(x, y, learning_rate, epochs):
    '''
    This function calculates the weight at the end of all epochs and returns it
    '''
    w = 0.0
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return None
    while epochs > 0:
        grad = 0.0
        for x_val, y_val in zip(x, y):
            grad += x_val * (w * x_val - y_val)
        grad = 2 / len(x) * grad
        w -= learning_rate * grad
        epochs -= 1
    return w

X = [1, 2, 3, 4]
Y = [2, 4, 6, 8]

LR = 0.05
EPOCHS = 100
print ("Final Weight Value is : ", gradient_descent(X, Y, LR, EPOCHS))
