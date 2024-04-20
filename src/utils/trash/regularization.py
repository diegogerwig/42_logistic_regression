"""Regularization decorators"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
# nd arrays
import numpy as np
# user modules
from validators import shape_validator, type_validator


@type_validator
@shape_validator({'theta': ('n', 1)})
def l2(theta: np.ndarray) -> float:
    """Computes the L2 regularization of a non-empty numpy.ndarray."""
    try:
        # l2
        theta_prime = theta
        theta_prime[0][0] = 0
        return theta_prime.T.dot(theta_prime)[0][0]
    except:
        return None
