"""Train Test Split from scratch"""
import random
def train_test_split(x, y, test_size=0.2, seed=42):
    '''
    This function returns the split train and test sets
    '''
    if len(x) == 0 or len(y) == 0 or len(x) != len(y) \
        or test_size <= 0 or test_size >= 1 or seed <= 0:
        return None
    num_rows = len(x[0])
    if num_rows == 0 or any(len(i) != num_rows for i in x):
        return None
    # Generate random set of indices
    n_test = round(test_size * len(x))
    random.seed(seed)
    test_indices = random.sample(range(len(x)), n_test)
    x_train = [x[i] for i in range(len(x)) if i not in test_indices]
    y_train = [y[i] for i in range(len(y)) if i not in test_indices]
    x_test = [x[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    return x_train, x_test, y_train, y_test

X = [
    [1000, 2],
    [1500, 3],
    [2000, 3],
    [2500, 4],
    [3000, 5],
    [3500, 4],
    [4000, 5],
    [4500, 6],
    [5000, 6],
    [5500, 7]
]
Y = [200, 250, 310, 370, 430, 490, 560, 620, 690, 750]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST =  train_test_split(X, Y, 0.2, 42)
print (X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)