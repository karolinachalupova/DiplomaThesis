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

from matplotlib.ticker import MaxNLocator

from figures import LatexFigure


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


def returns_histogram(returns):
    """
    Arguments: 
        returns: pd.Series
    """
    returns = returns*100
    returns.plot.hist(grid=True, bins=100, rwidth=0.9)

    sdate = pd.to_datetime(str(returns.index.values.min())).strftime('%b %Y')
    edate = pd.to_datetime(str(returns.index.values.max())).strftime('%b %Y')

    plt.ylabel("Number of Months ({} to {})".format(sdate, edate))
    plt.xlabel("Monthly Return (percentage points)")
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig = LatexFigure(plt.gcf())
    fig.fit()


def plot_cumulative_returns(returns):
    axis = cum_returns(returns, starting_value=1).plot()
    axis.set_ylabel("Gross Cumulative Return")
    axis.set_xlabel("")
    fig = LatexFigure(plt.gcf())
    fig.fit()


def get_metrics_single_model(ret): 
    return {
        "Mean": ret.mean(),
        "Mean (Yearly)": aggregate_returns(ret, convert_to="yearly").mean(),
        "Standard Deviation": ret.std(),
        "Sharpe Ratio": sharpe_ratio(ret,period='monthly'),
        "Skewness": skew(ret),
        "Kurtosis": kurtosis(ret),
        "Max Drawdown": max_drawdown(ret)
    }

def get_metrics_multiple_longshort(path_to_models, hidden_layers="32"):
    metrics = dict()
    longs = [0.5, 1, 5, 10, 20]
    shorts = [0.5, 1, 5, 10, 20]
    for l, s in zip(longs, shorts):
        ret = get_returns(path_to_models, hidden_layers, l, s)["Long-short"]
        m = get_metrics_single_model(ret)
        metrics["{}-{}".format(l, s)] = m
    return pd.DataFrame(metrics)

def get_metrics_all_models(path_to_models, percent_long=10, percent_short=10):
    metrics = dict()
    for hidden_layers in HIDDEN_LAYERS: 
        ret = get_returns(path_to_models, hidden_layers, percent_long, percent_short)["Long-short"]
        m = get_metrics_single_model(ret)
        metrics[NN_DICT.get(hidden_layers)] = m
    return pd.DataFrame(metrics)


def get_returns_all_models(path_to_models, percent_long=10, percent_short=10): 
    returns = dict() 
    for hidden_layers in HIDDEN_LAYERS:
        ret = get_returns(path_to_models, hidden_layers, percent_long, percent_short)["Long-short"]
        returns[NN_DICT.get(hidden_layers)] = ret
    return pd.DataFrame(returns)

def get_nfirms_in_portfolios(
        path_to_models="models/selected/ensembles", 
        hidden_layers="32,16,8,4,2", percentages = [0.5, 1, 5, 10, 20]):
    nfirms = dict()
    for p in percentages:
        lng, _ = get_portfolios(path_to_models, hidden_layers, p, p)
        nfirms["{}".format(p)] = int(lng["prediction"].groupby(pd.Grouper(level=0)).count().mean())
    return nfirms

def get_portfolios(
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

    return lng, srt

def get_returns(
        path_to_models="models/selected/ensembles", 
        hidden_layers="32,16,8,4,2", 
        percent_long=10, percent_short=10):
    
    lng, srt = get_portfolios(path_to_models=path_to_models, hidden_layers=hidden_layers, 
        percent_long=percent_long, percent_short=percent_short)

    # Get mean returns of short, long, long-short and all 
    l = lng.groupby(pd.Grouper(level=0)).apply(np.mean).actual 
    s = srt.groupby(pd.Grouper(level=0)).apply(np.mean).actual
    ls = l - s
    al = df.groupby(pd.Grouper(level=1, freq="M")).apply(np.mean).actual
    
    # Return dataframe with mean returns 
    return pd.DataFrame(np.array([l.values, s.values, ls.values, al.values]).T,
             columns = ["Long", "Short", "Long-short", "All Stocks"],
             index = l.index)