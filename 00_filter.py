"""
Filters out firms that do not pass certain criteria.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Paths to necessary data 
path_universe_filter = 'data/00_anomalies_unfiltered/DST_universe_filter.gzip'  # use pd.read_parquet()
path_signals = 'data/00_anomalies_unfiltered/DST_signals.gzip'  # use pd.read_parquet()
path_returns = 'data/00_anomalies_unfiltered/DST_returns.gzip'  # use pd.read_parquet()

# Paths to output data 
path_data_filtered = "data/01_anomalies_filtered/data.pkl"

# Filter signals
universe_filter = pd.read_parquet(path_universe_filter)
signals = pd.read_parquet(path_signals)
signals_filtered = signals[signals['DTID'].isin(universe_filter.DTID.unique().tolist())]

# Clear memory 
del signals

# Filter returns
returns = pd.read_parquet(path_returns)
returns_filtered = returns[returns['DTID'].isin(universe_filter.DTID.unique().tolist())]

# Clear memory
del returns 

# Concatenate returns and signals
data = pd.concat(signals.set_index(['DTID', 'date']), returns.set_index(['DTID', 'date'])], axis=1, join='inner')
data = pd.reset_index()
data.to_pickle(path_data_filtered)
