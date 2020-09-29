"""
Cleans data (outliers, NaN,...)


TODO: I'm not happy with replacing infty by np.nan, should be winsorization-like
"""

import sklearn
import scipy
import numpy as np
import pandas as pd

# Paths to input data 
path_data_input = "data/01_anomalies_filtered/data.pkl"

# Path to output data
path_data_output = "data/02_anomalies_cleaned/data.pkl"

def replace_inf(df):
    return df.replace([np.inf, -np.inf], np.nan)

def winsorize(df):
    t = scipy.stats.mstats.winsorize(df, axis=0, limits=(0.01,0.01))
    return pd.DataFrame(t, columns=df.columns, index=df.index)

def center(df):
    """
    removes the mean of the data, columnwise
    """
    scaler = sklearn.preprocessing.StandardScaler(with_std=False)
    t = scaler.fit_transform(df)
    return pd.DataFrame(t, columns=df.columns, index=df.index)
    
def normalize(df):
    max_abs_scaler = sklearn.preprocessing.MaxAbsScaler()
    t = max_abs_scaler.fit_transform(df)
    return pd.DataFrame(t, columns=df.columns, index=df.index)

def impute_nan(df):
    imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    t = imputer.fit_transform(df)
    return pd.DataFrame(t, columns=df.columns, index=df.index)

def clean(df):
    # Clean data using yearly cross-sections
    df = replace_inf(df)
    df = df.groupby(pd.Grouper(level=1, freq="Y")).apply(winsorize)
    df = df.groupby(pd.Grouper(level=1, freq="Y")).apply(center)
    df = df.groupby(pd.Grouper(level=1, freq="Y")).apply(normalize)
    df = impute_nan(df)
    return df.sort_index()

# Load data

# Clean
# Save

