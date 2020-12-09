"""
Backtesting out of sample
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from empyrical import max_drawdown, sharpe_ratio
from empyrical.stats import cum_returns, aggregate_returns
from scipy.stats import skew, kurtosis


from figures import YTRAIN_NAMES, NN_DICT, HIDDEN_LAYERS, NN_NAMES




def get_metrics_single_model(ret): 
    return {
        "Mean": ret.mean(),
        "Mean (Yearly)": aggregate_returns(ret, convert_to="yearly").mean(),
        "Standard Deviation": ret.std(),
        "Sharpe Ratio": sharpe_ratio(ret,period='monthly'),
        "Skewness": skew(ret),
        "Kurtosis": kurtosis(ret),
        "Max Drawdown": max_drawdown(ret),
    }

def get_metrics_multiple_longshort(path_to_backtests, hidden_layers="32"):
    metrics = dict()
    longs = [0.5, 1, 5, 10, 20]
    shorts = [0.5, 1, 5, 10, 20]
    for l, s in zip(longs, shorts):
        _, _, ret, market = get_returns(path_to_backtests, hidden_layers, l, s)
        m = get_metrics_single_model(ret)
        m_market = get_metrics_single_model(market)
        metrics["Market"] = m_market
        metrics["{}-{}".format(l, s)] = m
    return pd.DataFrame(metrics)

def get_metrics_all_models(path_to_backtests, percent_long=10, percent_short=10, HIDDEN_LAYERS=HIDDEN_LAYERS, NN_DICT=NN_DICT):
    metrics = dict()
    for hidden_layers in HIDDEN_LAYERS: 
        _, _, ret, _  = get_returns(path_to_backtests, hidden_layers, percent_long, percent_short)
        m = get_metrics_single_model(ret)
        metrics[NN_DICT.get(hidden_layers)] = m
    return pd.DataFrame(metrics)

def get_cumulative_returns_all_models(path_to_backtests, percent_long=10, percent_short=10, HIDDEN_LAYERS=HIDDEN_LAYERS, NN_DICT=NN_DICT): 
    returns = dict() 
    for hidden_layers in HIDDEN_LAYERS:
        _, _, ret, _  = get_cumulative_returns(path_to_backtests, hidden_layers, percent_long, percent_short)
        returns[NN_DICT.get(hidden_layers)] = ret
    return pd.DataFrame(returns)

def get_nfirms_in_portfolios(
        path_to_backtests="models/selected/ensembles", 
        hidden_layers="32,16,8,4,2", percentages = [0.5, 1, 5, 10, 20]):
    nfirms = dict()
    for p in percentages:
        _, lng, _ = get_portfolios(path_to_backtests, hidden_layers, p, p)
        nfirms["{}".format(p)] = int(lng["prediction"].groupby(pd.Grouper(level=0)).count().mean())
    return nfirms

def get_portfolios(
        path_to_backtests="models/selected/ensembles", 
        hidden_layers="32,16,8,4,2", 
        percent_long=10, percent_short=10):
    # Get df with columns 'actual' and 'prediction' for all testing years 
    # of the network with specified hidden_layers
    dfs = []
    for ytrain in YTRAIN_NAMES: 
        df = pd.read_csv(os.path.join(
            path_to_backtests, 
            'y={},y=4,y=1,hl={},nm=10,o=adam'.format(ytrain, hidden_layers), 
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
    return df, lng, srt

def get_returns(
        path_to_backtests="models/selected/ensembles", 
        hidden_layers="32,16,8,4,2", 
        percent_long=10, percent_short=10):

    
    df, lng, srt = get_portfolios(path_to_backtests=path_to_backtests, hidden_layers=hidden_layers, 
        percent_long=percent_long, percent_short=percent_short)

    # Get mean returns of short, long, long-short and all 
    l = lng.groupby(pd.Grouper(level=0)).apply(np.mean).actual 
    s = srt.groupby(pd.Grouper(level=0)).apply(np.mean).actual
    ls = l - s
    al = df.groupby(pd.Grouper(level=1, freq="M")).apply(np.mean).actual
    
    return l, s, ls, al

def get_cumulative_returns(path_to_backtests="models/selected/ensembles", 
        hidden_layers="32,16,8,4,2", 
        percent_long=10, percent_short=10):
    l, s, ls, al = get_returns(path_to_backtests, hidden_layers, percent_long, percent_short)
    cum_l = cum_returns(l)
    cum_s = cum_returns(-s)
    cum_ls = cum_l + cum_s
    cum_al = cum_returns(al)

    return cum_l, cum_s, cum_ls, cum_al