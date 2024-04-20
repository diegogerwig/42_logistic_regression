"""
Decorators to validate types and nd array dimensions
"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# nd arrays
import numpy as np
# to get information from the decorated function
import inspect
# to keep decorated function doc string
from functools import wraps

# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
# shape decorator helper
def shape_ok(expected_shape: tuple, actual_shape: tuple, m_n: list) -> bool:
    """
    Helper to calculate if the expected shape matches the actual shape,
    taking m and n possibilities in account
    """
    m, n = expected_shape
    am, an = actual_shape
    # Generic check
    if expected_shape == actual_shape:
        return True
    # numeric dimensions
    if isinstance(m, int) and m != am:
        return False
    if isinstance(n, int) and n != an:
        return False
    # Specific dimensions 'm', 'n', 'n + 1'
    if m == 'm' and m_n[0] is not None and am != m_n[0]:
        return False
    if m == 'n' and m_n[1] is not None and am != m_n[1]:
        return False
    if n == 'n' and m_n[1] is not None and an != m_n[1]:
        return False
    if m == 'n + 1' and m_n[1] is not None and am != m_n[1] + 1:
        return False
    # if the param is the first with specific dimensions to be tested
    if m == 'm' and m_n[0] is None:
        m_n[0] = am
    if m == 'n' and m_n[1] is None:
        m_n[1] = am
    if n == 'n' and m_n[1] is None:
        m_n[1] = an
    return True


# -----------------------------------------------------------------------------
# type validator
# -----------------------------------------------------------------------------
# generic nd array shape validation based on function signature
def shape_validator(shape_mapping: dict):
    """
    Decorator that will loop on the attributes in function signature and for
    each one checks if a specific 2D shape is expected in the dictionnary
    provided to the decorator.
    If the expected shape is not the current shape, the decorator prints an
    error and return None.
    This decorator does not do a lot of type checks, it must be verified before
    """
    def decorator(func):
        # extract information about the function's parameters.
        sig = inspect.signature(func)
        # preserve name and docstring of decorated function
        @wraps(func)
        def wrapper(*args, **kwargs):
            # init m and n so they can be used for comparison
            m_n: list = [None, None]
            # check positional arguments
            for i, (param_name, param) in enumerate(sig.parameters.items()):
                if param.annotation == np.ndarray and i < len(args):
                    arg = args[i]
                    expected_shape = shape_mapping.get(param_name)
                    # check the shape if there is something to check
                    if expected_shape is not None:
                        # dim check
                        if arg.ndim != 2:
                            raise TypeError(f"function '{func.__name__}' : "
                                            f"wrong dimension on "
                                            f"'{param_name}'")
                        # shape check
                        if not shape_ok(expected_shape, arg.shape, m_n):
                            print(f"function '{func.__name__}' : " \
                                  f"{param_name} has an invalid shape. "\
                                  f"Expected {expected_shape}, " \
                                  f"got {arg.shape}.")
                            return None

            # check keyword arguments
            for arg_name, expected_shape in shape_mapping.items():
                param = sig.parameters.get(arg_name)
                if param and param.annotation == np.ndarray:
                    arg = kwargs.get(arg_name)
                    if arg is not None:
                        # dim check
                        if arg.ndim != 2:
                            raise TypeError(f"function '{func.__name__}' : "
                                            f"wrong dimension on '{arg_name}'")
                            return None
                        # shape check
                        if not shape_ok(expected_shape, arg.shape, m_n):
                            raise TypeError(f"function '{func.__name__}' : "
                                            f"{arg_name} has an invalid shape."
                                            f" Expected {expected_shape}, "
                                            f"got {arg.shape}.")
                            return None

            return func(*args, **kwargs)
        return wrapper
    return decorator


# -----------------------------------------------------------------------------
# type validator
# -----------------------------------------------------------------------------
# generic type validation based on type annotation in function signature
def type_validator(func):
    """
    Decorator that will rely on the types and attributes declaration in the
    function signature to check the actual types of the parameter against the
    expected types
    """
    # extract information about the function's parameters and return type.
    sig = inspect.signature(func)
    # preserve name and docstring of decorated function
    @wraps(func)
    def wrapper(*args, **kwargs):
        # map the parameter from signature to their corresponding values
        bound_args = sig.bind(*args, **kwargs)
        # check for each name of param if value has the declared type
        for name, value in bound_args.arguments.items():
            if name in sig.parameters:
                param = sig.parameters[name]
                if (param.annotation != param.empty
                        and not isinstance(value, param.annotation)):
                    raise TypeError(f"function '{func.__name__}' : expected "
                                    f"type '{param.annotation}' for argument "
                                    f"'{name}' but got {type(value)}.")
        return func(*args, **kwargs)
    return wrapper
