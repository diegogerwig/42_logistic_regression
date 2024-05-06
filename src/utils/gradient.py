import numpy as np
from tqdm import tqdm


def sigmoid(z):
    sig = 1 / (1 + np.exp(-z)) # Sigmoid function (logistic function)
    return sig


def gradient_descent(X, y, theta, lr, num_iters, threshold=1e-5):
    m = len(y)
    J_history = []
    for _ in tqdm(range(num_iters)):
        h = sigmoid(np.dot(X, theta)) # Hypothesis function (sigmoid)
        J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) # Cost function
        J_history.append(J)
        gradient = (1 / m) * np.dot(X.T, (h - y)) # Gradient of the cost
        theta -= lr * gradient  # Update theta with gradient descent
        if len(J_history) > 1 and abs(J_history[-1] - J_history[-2]) < threshold: # Convergence check with threshold
            break
    return theta, J_history
