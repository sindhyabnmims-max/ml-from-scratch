"""Module providing the mean squared error when provided with the actual and predicted"""
def mean_squared_error(actual, predicted):
    """
    Calculate mean squared error using actual and predicted series
    """
    if len(actual) == 0 or len(predicted) == 0 or len(actual) != len(predicted):
        return None

    summation = 0
    for tup in zip(actual, predicted):
        summation += (tup[0] - tup[1])**2
    mse_final = summation/len(actual)
    return mse_final


# Test 1 - Existing Loss
actual_lst = [300, 450, 500, 600]
predicted_lst = [310, 430, 520, 590]
mse = mean_squared_error(actual_lst, predicted_lst)
print ("Prediction not the same Actual: MSE = ", mse)

# Test 2 - Existing Loss
actual_lst = [300, 450, 500, 600]
predicted_lst = [300, 450, 500, 600]
mse = mean_squared_error(actual_lst, predicted_lst)
print ("Prediction is the same as Actual: MSE = ", mse)
