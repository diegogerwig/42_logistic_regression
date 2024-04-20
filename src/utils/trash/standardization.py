import numpy as np
from functools import wraps
import inspect

# Funciones de validación de tipo y forma
def shape_ok(expected_shape, actual_shape, m_n):
    """
    Helper para calcular si la forma esperada coincide con la forma actual,
    teniendo en cuenta las posibilidades de m y n.
    """
    m, n = expected_shape
    am, an = actual_shape
    if expected_shape == actual_shape:
        return True
    if isinstance(m, int) and m != am:
        return False
    if isinstance(n, int) and n != an:
        return False
    if m == 'm' and m_n[0] is not None and am != m_n[0]:
        return False
    if m == 'n' and m_n[1] is not None and am != m_n[1]:
        return False
    if n == 'n' and m_n[1] is not None and an != m_n[1]:
        return False
    if m == 'n + 1' and m_n[1] is not None and am != m_n[1] + 1:
        return False
    if m == 'm' and m_n[0] is None:
        m_n[0] = am
    if m == 'n' and m_n[1] is None:
        m_n[1] = am
    if n == 'n' and m_n[1] is None:
        m_n[1] = an
    return True

def shape_validator(shape_mapping):
    """
    Decorador que verificará si la forma esperada coincide con la forma actual
    """
    def decorator(func):
        sig = inspect.signature(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            m_n = [None, None]
            bound_args = sig.bind(*args, **kwargs)
            for name, value in bound_args.arguments.items():
                if name in sig.parameters:
                    param = sig.parameters[name]
                    if param.annotation == np.ndarray:
                        arg = value
                        expected_shape = shape_mapping.get(name)
                        if expected_shape:
                            if arg.ndim != 2:
                                raise TypeError(f"Función '{func.__name__}': "
                                                f"Dimensión incorrecta en "
                                                f"'{name}'")
                            if not shape_ok(expected_shape, arg.shape, m_n):
                                print(f"Función '{func.__name__}': " \
                                      f"{name} tiene una forma inválida. " \
                                      f"Se esperaba {expected_shape}, " \
                                      f"se obtuvo {arg.shape}.")
                                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator

def type_validator(func):
    """
    Decorador que verificará si los tipos de los argumentos coinciden con los
    tipos esperados
    """
    sig = inspect.signature(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        for name, value in bound_args.arguments.items():
            if name in sig.parameters:
                param = sig.parameters[name]
                if (param.annotation != param.empty
                        and not isinstance(value, param.annotation)):
                    raise TypeError(f"Función '{func.__name__}': se esperaba "
                                    f"tipo '{param.annotation}' para el "
                                    f"argumento '{name}' pero se obtuvo "
                                    f"{type(value)}.")
        return func(*args, **kwargs)
    return wrapper

# Funciones de normalización
@type_validator
@shape_validator({'x': ('m', 'n')})
def normalize_xset(x: np.ndarray, p_means=None, p_stds=None):
    """Normalizar cada característica en un conjunto de datos completo"""
    try:
        m, n = x.shape
        x_norm = np.empty((m, 0))
        means = []
        stds = []
        for feature in range(n):
            serie = x[:, feature].reshape(-1, 1)
            if p_means is not None and p_stds is not None:
                mean = p_means[0][feature][0]
                std = p_stds[0][feature][0]
            else:
                mean = np.mean(serie)
                std = np.std(serie)
            x_norm = np.c_[x_norm, z_score(serie, mean, std)]
            means.append(mean)
            stds.append(std)
        return x_norm, means, stds
    except ValueError as exp:
        print(exp)
        return None

@type_validator
@shape_validator({'x': ('m', 1)})
def z_score(x: np.ndarray, mean: float, std: float):
    """
    Calcula la versión normalizada de un np.ndarray no vacío utilizando la
    estandarización de la puntuación z.
    """
    try:
        z_score_formula = lambda x, mean, std: (x - mean) / std
        zscore_normalize = np.vectorize(z_score_formula)
        x_prime = zscore_normalize(x, mean, std)
        return x_prime
    except:
        return None

@staticmethod
def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray) or x.size <= 1:
            return None

        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return None
        return (x - mean) / std

    except:
        return None



# """Standardization functions"""
# # -----------------------------------------------------------------------------
# # Module imports
# # -----------------------------------------------------------------------------
# # system
# import os
# import sys
# # nd arrays
# import numpy as np
# # user modules
# # from validators import shape_validator, type_validator

# import numpy as np
# # to get information from the decorated function
# import inspect
# # to keep decorated function doc string
# from functools import wraps



# def shape_ok(expected_shape: tuple, actual_shape: tuple, m_n: list) -> bool:
#     """
#     Helper to calculate if the expected shape matches the actual shape,
#     taking m and n possibilities in account
#     """
#     m, n = expected_shape
#     am, an = actual_shape
#     # Generic check
#     if expected_shape == actual_shape:
#         return True
#     # numeric dimensions
#     if isinstance(m, int) and m != am:
#         return False
#     if isinstance(n, int) and n != an:
#         return False
#     # Specific dimensions 'm', 'n', 'n + 1'
#     if m == 'm' and m_n[0] is not None and am != m_n[0]:
#         return False
#     if m == 'n' and m_n[1] is not None and am != m_n[1]:
#         return False
#     if n == 'n' and m_n[1] is not None and an != m_n[1]:
#         return False
#     if m == 'n + 1' and m_n[1] is not None and am != m_n[1] + 1:
#         return False
#     # if the param is the first with specific dimensions to be tested
#     if m == 'm' and m_n[0] is None:
#         m_n[0] = am
#     if m == 'n' and m_n[1] is None:
#         m_n[1] = am
#     if n == 'n' and m_n[1] is None:
#         m_n[1] = an
#     return True


# # -----------------------------------------------------------------------------
# # type validator
# # -----------------------------------------------------------------------------
# # generic nd array shape validation based on function signature
# def shape_validator(shape_mapping: dict):
#     """
#     Decorator that will loop on the attributes in function signature and for
#     each one checks if a specific 2D shape is expected in the dictionnary
#     provided to the decorator.
#     If the expected shape is not the current shape, the decorator prints an
#     error and return None.
#     This decorator does not do a lot of type checks, it must be verified before
#     """
#     def decorator(func):
#         # extract information about the function's parameters.
#         sig = inspect.signature(func)
#         # preserve name and docstring of decorated function
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             # init m and n so they can be used for comparison
#             m_n: list = [None, None]
#             # check positional arguments
#             for i, (param_name, param) in enumerate(sig.parameters.items()):
#                 if param.annotation == np.ndarray and i < len(args):
#                     arg = args[i]
#                     expected_shape = shape_mapping.get(param_name)
#                     # check the shape if there is something to check
#                     if expected_shape is not None:
#                         # dim check
#                         if arg.ndim != 2:
#                             raise TypeError(f"function '{func.__name__}' : "
#                                             f"wrong dimension on "
#                                             f"'{param_name}'")
#                         # shape check
#                         if not shape_ok(expected_shape, arg.shape, m_n):
#                             print(f"function '{func.__name__}' : " \
#                                   f"{param_name} has an invalid shape. "\
#                                   f"Expected {expected_shape}, " \
#                                   f"got {arg.shape}.")
#                             return None

#             # check keyword arguments
#             for arg_name, expected_shape in shape_mapping.items():
#                 param = sig.parameters.get(arg_name)
#                 if param and param.annotation == np.ndarray:
#                     arg = kwargs.get(arg_name)
#                     if arg is not None:
#                         # dim check
#                         if arg.ndim != 2:
#                             raise TypeError(f"function '{func.__name__}' : "
#                                             f"wrong dimension on '{arg_name}'")
#                             return None
#                         # shape check
#                         if not shape_ok(expected_shape, arg.shape, m_n):
#                             raise TypeError(f"function '{func.__name__}' : "
#                                             f"{arg_name} has an invalid shape."
#                                             f" Expected {expected_shape}, "
#                                             f"got {arg.shape}.")
#                             return None

#             return func(*args, **kwargs)
#         return wrapper
#     return decorator


# # -----------------------------------------------------------------------------
# # type validator
# # -----------------------------------------------------------------------------
# # generic type validation based on type annotation in function signature
# def type_validator(func):
#     """
#     Decorator that will rely on the types and attributes declaration in the
#     function signature to check the actual types of the parameter against the
#     expected types
#     """
#     # extract information about the function's parameters and return type.
#     sig = inspect.signature(func)
#     # preserve name and docstring of decorated function
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         # map the parameter from signature to their corresponding values
#         bound_args = sig.bind(*args, **kwargs)
#         # check for each name of param if value has the declared type
#         for name, value in bound_args.arguments.items():
#             if name in sig.parameters:
#                 param = sig.parameters[name]
#                 if (param.annotation != param.empty
#                         and not isinstance(value, param.annotation)):
#                     raise TypeError(f"function '{func.__name__}' : expected "
#                                     f"type '{param.annotation}' for argument "
#                                     f"'{name}' but got {type(value)}.")
#         return func(*args, **kwargs)
#     return wrapper




# # -----------------------------------------------------------------------------
# # Functions
# # -----------------------------------------------------------------------------
# # normalization functions
# @type_validator
# @shape_validator({'x': ('m', 'n')})
# def normalize_xset(x: np.ndarray, p_means: list = None,
#                    p_stds: list = None) -> np.ndarray:
#     """Normalize each feature an entire set of data"""
#     try:
#         m, n = x.shape
#         x_norm = np.empty((m, 0))
#         means = []
#         stds = []
#         for feature in range(n):
#             serie = x[:, feature].reshape(-1, 1)
#             if p_means is not None and p_stds is not None:
#                 mean = p_means[0][feature][0]
#                 std = p_stds[0][feature][0]
#             else:
#                 mean = np.mean(serie)
#                 std = np.std(serie)
#             x_norm = np.c_[x_norm, z_score(serie, mean, std)]
#             means.append(mean)
#             stds.append(std)
#         return x_norm, means, stds
#     except ValueError as exp:
#         print(exp)
#         return None


# @type_validator
# @shape_validator({'x': ('m', 1)})
# def z_score(x: np.ndarray, mean: float, std: float) -> np.ndarray:
#     """
#     Computes the normalized version of a non-empty numpy.ndarray using the
#     z-score standardization.
#     """
#     try:
#         z_score_formula = lambda x, mean, std: (x - mean) / std
#         zscore_normalize = np.vectorize(z_score_formula)
#         x_prime = zscore_normalize(x, mean, std)
#         return x_prime
#     except:
#         return None
