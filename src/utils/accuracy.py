import numpy as np
from gradient import sigmoid


def accuracy(X, y, theta):
    y_pred = sigmoid(np.dot(X, theta))
    y_pred_class = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y)
    return accuracy