import numpy as np


def normalize_xset(x):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    X_norm = (x - means) / stds
    return X_norm
