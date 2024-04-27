import numpy as np
from tqdm import tqdm


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for _ in tqdm(range(num_iters)):
        h = sigmoid(np.dot(X, theta))
        J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        J_history.append(J)
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
    return theta, J_history