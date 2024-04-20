"""
Custom class to perform metrics on a dataset
"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
import math
# nd arrays
import numpy as np
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from validators import shape_validator, type_validator


class Metrics():
    """Metrics class"""

    @type_validator
    @shape_validator({'x': ('m', 1)})
    def __init__(self, x: np.ndarray):
        """Constructor"""
        self.x = x[~np.isnan(x)] # filtering nan values
        self.m, self.n = x.shape

    def mean(self) -> float:
        """Computes the mean of a given non-empty list or array x"""
        result: float = 0
        try:
            return float(np.sum(self.x) / self.m)
        except:
            return None

    def median(self) -> float:
        """Computes the median of a given non-empty list or array x"""
        return float(self.percentile(50))

    @type_validator
    def percentile(self, p: int) -> float:
        """
        computes the expected percentile of a given non-empty list or array x.
        """
        x_sorted = np.sort(self.x)
        try:
            if p in (0, 100):
                return x_sorted[self.m - 1] if p == 100 else x_sorted[0]
            fractional_rank: float = (p / 100) * (self.m - 1)
            int_part = int(fractional_rank)
            frac_part = fractional_rank % 1
            return (x_sorted[int_part] + frac_part * (x_sorted[int_part + 1]
                    - x_sorted[int_part]))
        except:
            return None

    def quartiles(self) -> np.ndarray:
        """Computes the 1st and 3rd quartiles of a given non-empty array x"""
        return ([float(self.percentile(25)), float(self.percentile(75))])

    def var(self) -> float:
        """computes the variance of a given non-empty list or array x"""
        result = 0
        try:
            for num in range(self.m):
                result += (num - self.mean()) ** 2
            return float(result / (self.m - 1))
        except:
            return None

    def std(self) -> float:
        """
        computes the standard deviation of a given non-empty list or array x
        """
        return math.sqrt(self.var())

    def __str__(self, first: bool = False, sp: float = 13,
                name: str = None) -> str:
        """
        display metrics
        """
        res = (f'{self.mean():{sp}.6f}\n'
               f'{self.std():{sp}.6f}\n'
               f'{self.percentile(0):{sp}.6f}\n'
               f'{self.percentile(25):{sp}.6f}\n'
               f'{self.percentile(50):{sp}.6f}\n'
               f'{self.percentile(75):{sp}.6f}\n'
               f'{self.percentile(100):{sp}.6f}\n')

        if first is True:
            rows = (f'{"mean":{sp}}\n'
                    f'{"std":{sp}}\n'
                    f'{"min":{sp}}\n'
                    f'{"25%":{sp}}\n'
                    f'{"50%":{sp}}\n'
                    f'{"75%":{sp}}\n'
                    f'{"max":{sp}}\n')
            res = '\n'.join([col1 + col2 for col1, col2 in
                                zip(rows.split('\n'), res.split('\n'))])
        if name is not None:
            name = name[:sp - 3] + '..' if len(name) > sp else name
            col_name = (f'{"":{sp}}{name:>{sp}.{sp - 1}}\n' if first is True
                        else f'{name:>{sp}.{sp - 1}}\n')
            res = col_name + res

        return res
