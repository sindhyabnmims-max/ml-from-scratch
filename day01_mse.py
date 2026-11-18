"""Module providing the mean squared error when provided with the actual and predicted"""
def mean_squared_error(actual, predicted):
    """
    Calculate mean squared error using actual and predicted series
    """
    if len(actual) < 0 or len(predicted) < 0 or len(actual) != len(predicted):
        return None

    summation = 0
    for i, act in enumerate(actual):
        for j, pred in enumerate(predicted):
            if i == j:
                summation += (act - pred)**2
    mse_final = summation/len(actual)
    return mse_final

actual_lst = [300, 450, 500, 600]
predicted_lst = [310, 430, 520, 590]
mse = mean_squared_error(actual_lst, predicted_lst)
print (mse)
