# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import sys
import os
# nd arrays
import numpy as np
# user modules
from validators import type_validator, shape_validator


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
# helper function to calculate tp, fp, tn, fn
@type_validator
@shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
def tf_metrics(y: np.ndarray, y_hat: np.ndarray, pos_label=None) -> tuple:
    """
    Returns as a tuple in that order :
        true positive number
        false positive number
        true negative number
        false negative number
    """
    # initialize variables
    tp, fp, tn, fn = [0]*4
    # if global for all classes
    if pos_label==None:
        # loop on every class in the original y vector
        for class_ in np.unique(y):
            tp += np.sum(np.logical_and(y_hat == class_, y == class_))
            fp += np.sum(np.logical_and(y_hat == class_, y != class_))
            tn += np.sum(np.logical_and(y_hat != class_, y != class_))
            fn += np.sum(np.logical_and(y_hat != class_, y == class_))
    else: # focus on one class
        tp += np.sum(np.logical_and(y_hat == pos_label, y == pos_label))
        fp += np.sum(np.logical_and(y_hat == pos_label, y != pos_label))
        tn += np.sum(np.logical_and(y_hat != pos_label, y != pos_label))
        fn += np.sum(np.logical_and(y_hat != pos_label, y == pos_label))

    return (tp, fp, tn, fn)


def accuracy_score_(y: np.ndarray, y_hat: np.ndarray,
                    pos_label: int = 1) -> float:
    """
    Compute the accuracy score : how many predictions are correct.
    """
    try:
        tp, fp, tn, fn = tf_metrics(y, y_hat, pos_label)
        return (tp + tn) / (tp + fp + tn + fn)
    except:
        return None

@type_validator
@shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
def precision_score_(y: np.ndarray, y_hat: np.ndarray,
                     pos_label: int = 1) -> float:
    """
    Compute the precision score : model's ability to not classify positive
                                  examples as negative.
    """
    try:
        tp, fp, _, _ = tf_metrics(y, y_hat, pos_label)
        return tp / (tp + fp)
    except:
        return None


@type_validator
@shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
def recall_score_(y: np.ndarray, y_hat: np.ndarray,
                  pos_label: int = 1) -> float:
    """
    Compute the recall score : model's ability to detect positive examples.
    """
    try:
        tp, _, _, fn = tf_metrics(y, y_hat, pos_label)
        return tp / (tp + fn)
    except:
        return None


@type_validator
@shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
def f1_score_(y: np.ndarray, y_hat: np.ndarray,
              pos_label: int = 1) -> float:
    """
    Compute the f1 score : harmonic mean of precision and recall. often used
                           for imbalanced datasets where it is important to
                           minimize false negatives while minimizing false
                           positives.
    """
    try:
        precision = precision_score_(y, y_hat, pos_label=pos_label)
        recall = recall_score_(y, y_hat, pos_label=pos_label)
        return (2 * precision * recall) / (precision + recall)
    except:
        return None
