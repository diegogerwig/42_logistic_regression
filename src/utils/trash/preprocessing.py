"""
Data preprocessing utils
"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
# read csv files
import csv
# nd arrays
import numpy as np
# dataframes
import pandas as pd
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'classes'))
from validators import type_validator
from Metrics import Metrics

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
@type_validator
def get_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters data to get only numeric features.
    This function also remove duplicated rows.
    This creates a new dataframe.
    """
    # Copy df
    df_res: pd.DataFrame = df.copy()
    # Clean duplicated data
    df_res: pd.DataFrame = df_res[~df.duplicated()]
    # filter non numerical features and remove "Index" column
    numeric_features: list = [feature for feature in df.columns
                              if pd.api.types.is_numeric_dtype(df[feature])
                              and feature != "Index"]
    return df_res[numeric_features]


@type_validator
def replace_empty_nan_mean(df: pd.DataFrame):
    """
    Replace all empty or nan values in a dataframe with the mean of the serie
    the value is in.
    The replacement is done in place on a dataframe with only numerical cols.
    """
    for i, col in enumerate(df.columns):
        # replace nan values with the mean of the feature
        tmp_mean = df[col].sum(skipna=True) / df[col].count()
        df.loc[:, col] = df[col].fillna(value=tmp_mean)


@type_validator
def replace_empty_nan_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all empty or nan values in a dataframe with the median of the serie
    the value is in.
    The replacement is done in place on a dataframe with only numerical cols.
    """
    for i, col in enumerate(df.columns):
        # replace nan values with the median of the feature
        tmp_med = Metrics(np.array(df[col]).reshape(-1, 1)).median()
        df.loc[:, col] = df[col].fillna(value=tmp_med)