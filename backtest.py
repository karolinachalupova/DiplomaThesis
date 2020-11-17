"""
Backtesting out of sample
"""
import pandas as pd
import numpy as np
import os

from empyrical import max_drawdown, sharpe_ratio
from empyrical.stats import cum_returns, aggregate_returns


YTRAIN_NAMES = ["12","13","14", "15", "16"]
NN_DICT = {
        "-1":"LR",
        "32":"NN1",
        "32,16":"NN2",
        "32,16,8":"NN3",
        "32,16,8,4":"NN4",
        "32,16,8,4,2":"NN5"
        }
HIDDEN_LAYERS = list(NN_DICT.keys())
NN_NAMES = list(NN_DICT.values())


def get_metrics(ret): 
    raise NotImplementedError


def get_returns(
        path_to_models="models/selected/ensembles", 
        hidden_layers="32,16,8,4,2", 
        percent_long=10, percent_short=10):
    
    # Get df with columns 'actual' and 'prediction' for all testing years 
    # of the network with specified hidden_layers
    dfs = []
    for ytrain in YTRAIN_NAMES: 
        try: 
            df = pd.read_csv(os.path.join(
                path_to_models, 
                'y={},y=12,y=1,hl={},nm=9,o=adam'.format(ytrain, hidden_layers), 
                'backtest.csv'),index_col=[0,1])
        except: 
            df = pd.read_csv(os.path.join(
                path_to_models, 
                'y={},y=12,y=1,hl={},nm=8,o=adam'.format(ytrain, hidden_layers), 
                'backtest.csv'),index_col=[0,1])
        df.index =df.index.set_levels([df.index.levels[0], pd.to_datetime(df.index.levels[1])])
        dfs.append(df)
    df = pd.concat(dfs)
    
    def get_p_largest(df, percent):
        n = int(0.01* percent * len(df))
        return df.nlargest(n, columns="prediction")

    def get_p_smallest(df, percent):
        n = int(0.01* percent * len(df))
        return df.nsmallest(n, columns="prediction")
     
    # Get df of shorts 
    srt = df.groupby(pd.Grouper(level=1, freq="M")).apply(get_p_smallest, percent=percent_short)
    srt.index = srt.index.droplevel(2)

    # Get df of longs 
    lng = df.groupby(pd.Grouper(level=1, freq="M")).apply(get_p_largest, percent=percent_long)
    lng.index = lng.index.droplevel(2)

    # Get mean returns of short, long, long-short and all 
    l = lng.groupby(pd.Grouper(level=0)).apply(np.mean).actual 
    s = srt.groupby(pd.Grouper(level=0)).apply(np.mean).actual
    ls = l - s
    al = df.groupby(pd.Grouper(level=1, freq="M")).apply(np.mean).actual
    
    # Return dataframe with mean returns 
    return pd.DataFrame(np.array([l.values, s.values, ls.values, al.values]).T,
             columns = ["Long", "Short", "Long-short", "All"],
             index = l.index)