import numpy as np
from gradient import sigmoid


def accuracy(X, y, theta):
    y_pred = sigmoid(np.dot(X, theta)) # Hypothesis function (sigmoid)
    y_pred_class = (y_pred >= 0.5).astype(int) # Predicted class
    accuracy = np.mean(y_pred_class == y) # Accuracy calculation
    return accuracy
