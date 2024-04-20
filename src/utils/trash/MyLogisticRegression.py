"""My Logistic Regression module"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
import time
# nd arrays
import numpy as np
# for decorators
import inspect
from functools import wraps
# for progress bar
from tqdm import tqdm
# for plot
import matplotlib.pyplot as plt
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from regularization import l2
from validators import shape_validator, type_validator


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
# sigmoid
@type_validator
@shape_validator({'x': ('m', 1)})
def sigmoid_(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid of a vector."""
    try:
        return 1 / (1 + np.exp(-x))
    except:
        return None


# l2 regularizations with decorators
# regulatization of loss
def regularize_loss(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        y, y_hat = args
        m, _ = y.shape
        loss = func(self, y, y_hat)
        regularization_term = (self.lambda_ / (2 * m)) * l2(self.thetas)
        return loss + regularization_term
    return wrapper


# regulatization of gradient
def regularize_grad(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        x, y = args
        m, _ = x.shape
        gradient = func(self, x, y)
        # add regularization
        theta_prime = self.thetas.copy()
        theta_prime[0][0] = 0
        return gradient + (self.lambda_ * theta_prime) / m
    return wrapper


# yield batches from a np array
@type_validator
@shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
def get_batches(x: np.ndarray, y: np.ndarray, size: int = 32) -> np.ndarray:
    """Yields batch of data from original array"""
    m, n = x.shape
    if m < size or size <= 0:
        yield x, y
    else:
        start = size * -1
        stop = size
        while start != m - 1 and stop != m - 1:
            start = start + size if start + size < m else m - 1
            stop = start + size if start + size < m else m - 1
            yield x[start:stop, :], y[start:stop, :]


# shuffle an x and y set
@type_validator
@shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
def shuffle_set(x: np.ndarray, y: np.ndarray) -> tuple:
    """Returns a tuple x and y after shuffling"""
    # stick x and y then shuffle
    full_set = np.c_[x, y]
    np.random.shuffle(full_set)
    # re detach x and y
    x_shuf = full_set[:, :-1]
    y_shuf = full_set[:, -1:]
    return x_shuf, y_shuf


# -----------------------------------------------------------------------------
# MyLogisticRegression class with l2 regularization
# -----------------------------------------------------------------------------
class MyLogisticRegression():
    """My personnal logistic regression to classify things."""

    # We consider l2 penalty only. One may wants to implement other penalties
    supported_penalties = ['l2']

    @type_validator
    @shape_validator({'thetas': ('n', 1)})
    def __init__(self, thetas: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000, penalty: str = 'l2',
                 lambda_: float = 1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas).reshape((-1, 1))
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalties else 0
        self.gd_functions = {'GD' : self.gradient_descent,
                             'SGD' : self.stochastic_gradient_descent,
                             'MBGD' : self.mini_batch_gradient_descent}

    @type_validator
    @shape_validator({'x': ('m', 'n')})
    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the vector of prediction y_hat from two non-empty
        numpy.ndarray.
        """
        try:
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            return sigmoid_(x_prime.dot(self.thetas))
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Calculates the loss by element."""
        try:
            # values check
            min_val = 0
            max_val = 1
            valid_values = np.logical_and(min_val <= y_hat, y_hat <= max_val)
            if not np.all(valid_values):
                print('y / y_hat val must be between 0 and 1', file=sys.stderr)
                return None
            # add a little value to y_hat to avoid log(0) problem
            eps: float=1e-15
            # y_hat = np.clip(y_hat, eps, 1 - eps) < good options for eps
            return -(y * np.log(y_hat + eps) + (1 - y) \
                    * np.log(1 - y_hat + eps))
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    @regularize_loss
    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Compute the logistic loss value."""
        try:
            m, _= y.shape
            return np.sum(self.loss_elem_(y, y_hat)) / m
        except:
            return None

    @type_validator
    @shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
    @regularize_grad
    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the regularized linear gradient of three non-empty
        numpy.ndarray.
        """
        try:
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            y_hat = x_prime.dot(self.thetas)
            return x_prime.T.dot(y_hat - y) / m
        except:
            return None

    @type_validator
    @shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
    def fit_(self, x: np.ndarray, y: np.ndarray,
             plot: bool = False, ax = None, gd: str = 'GD') -> np.ndarray:
        """
        Fits the model to the training dataset contained in x and y.
        """
        try:
            learn_c = None
            if plot is True:
                # initialize the line plots
                learn_c, = ax.plot([], [], 'r-')
                # label axis
                ax.set_xlabel = 'number of iteration'
                ax.set_ylabel = 'cost'
            progress = tqdm(range(self.max_iter))
            progress.set_description(" Training model...      ")
            trained = False
            for it, _ in enumerate(progress):
                # calculate the gradient for current thetas
                if trained:
                    pass
                elif bool(self.gd_functions[gd](x, y, ax=ax, learn_c=learn_c,
                                              it=it)) is True:
                    progress.set_description(f"\033[32m{ f' Trained in {it} iters' :<24}"
                                             "\033[0m")
                    trained = True
                elif it == self.max_iter - 1 and gd == 'GD':
                    progress.set_description("\033[33m Model not optimal...   "
                                             "\033[0m")
                elif it == self.max_iter - 1 and gd != 'GD':
                    pass
                    progress.set_description("\033[32m Training complete !    "
                                             "\033[0m")

            return self.thetas
        except:
            return None

    # -------------------------------------------------------------------------
    # Gradient descent algorithms
    # -------------------------------------------------------------------------
    # batch gradient descent
    @type_validator
    def gradient_descent(self, x: np.ndarray, y: np.ndarray, ax = None,
                         learn_c = None, it: int = 0) -> bool:
        # update learning curve
        if ax is not None and learn_c is not None:
            self.update_learning_curve(x, y, ax, learn_c, it)
        # epsilon to check if the derivatives are evolving
        eps = 1e-6
        # gradient descent
        gradient = self.gradient_(x, y)
        previous_thetas = self.thetas.copy()
        self.thetas -= self.alpha * gradient
        # early stopping if the parameters are not evolving anymore
        return np.sum(np.absolute(self.thetas - previous_thetas)) < eps

    # stochastic gradient descent
    @type_validator
    def stochastic_gradient_descent(self, x: np.ndarray, y: np.ndarray,
                                    ax = None, learn_c= None,
                                    it: int = 0) -> bool:
        # shuffle set
        x_shuf, y_shuf = shuffle_set(x, y)
        # loop on every x, and do and save the gradient for each
        for i in range(x_shuf.shape[0]):
            # current x position for plot
            x_plot = it * x_shuf.shape[0] + i
            # update learning curve
            if ax is not None and learn_c is not None:
                self.update_learning_curve(x, y, ax, learn_c, x_plot)
            # check the gradient for every loop
            gradient = self.gradient_(x_shuf[i, :].reshape(1, -1),
                                      y_shuf[i].reshape(1, 1))
            previous_thetas = self.thetas.copy()
            self.thetas -= self.alpha * gradient
        # no early stopping
        return False

    # minibatch gradient descent
    @type_validator
    def mini_batch_gradient_descent(self, x: np.ndarray, y: np.ndarray,
                                    ax = None, learn_c= None,
                                    it: int = 0) -> bool:
        # shuffle set
        x_shuf, y_shuf = shuffle_set(x, y)
        # loop on every x, and do and save the gradient for each
        batch_size = 50
        for index, batch in enumerate(get_batches(x_shuf, y_shuf,
                                      size=batch_size)):
            batch_x, batch_y = batch
            # current x position for plot
            x_plot = (it * x_shuf.shape[0]) // 50 + index
            # update learning curve
            if ax is not None and learn_c is not None:
                self.update_learning_curve(x, y, ax, learn_c, x_plot)
            # check the gradient for every loop
            gradient = self.gradient_(batch_x, batch_y)
            previous_thetas = self.thetas.copy()
            self.thetas -= self.alpha * gradient
        # no early stopping
        return False

    # -------------------------------------------------------------------------
    # Plot learning curves
    # -------------------------------------------------------------------------
    @type_validator
    def update_learning_curve(self, x: np.ndarray, y: np.ndarray,
                              ax, learn_c, it: int):
        # calculate current loss
        loss = self.loss_(y, self.predict_(x))
        # Update the lines data
        learn_c.set_data(np.append(learn_c.get_xdata(), it),
                         np.append(learn_c.get_ydata(), loss))
        # Update the axis limits
        ax.relim()
        ax.autoscale_view()

        # Wait to make plot visible
        # time.sleep(0.03)
