"""
Generates all figures and tables in the thesis.
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import pickle 
import os 
import itertools
import argparse

from matplotlib import pyplot as plt
import matplotlib.ticker as tkr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from empyrical.stats import cum_returns
from matplotlib.ticker import MaxNLocator

from utils import chunks, get_parts, get_orders, divide_by_max
from data import MinMaxed, Normalized, Subset, Meta


meta = Meta()
meta.load()
mpl.use('pgf')
sns.set()

SORTING = ['52WH',
 'WWI',
 'EFoP',
 'MomLag',
 'LB5',
 'STR',
 'Seas6t10A',
 'DurE',
 'VolMV',
 'Seas',
 'Seas2t5A',
 'Max',
 'IdioRisk',
 'LB3',
 'MomRev',
 'NOA',
 'Accr',
 'Seas2t5N',
 'Coskew',
 'Seas11t15N',
 'PM',
 'OPoA',
 'Seas6t10N',
 'RDoMV',
 'CVoST',
 'LiqShck',
 'Amihud',
 'dCE',
 'LCoBP',
 'EPred']

N_SEEDS = 10
YTRAIN_NAMES = [str(i) for i in range(6,25)]
YTRAIN_NAMES_TO_TEST_YEAR = {n:str(int(n)+1994) for n in YTRAIN_NAMES}
SEED_NAMES = [str(i) for i in list(range(1,N_SEEDS+1))]
SEED_INT_TO_STR = {int(i):i for i in SEED_NAMES} 
N_YTRAIN = len(YTRAIN_NAMES)


NN_DICT = {
        "-1":"LR",
        "32": "NN1",
        "32,16":"NN2",
        "32,16,8":"NN3",
        "32,16,8,4":"NN4",
        }
HIDDEN_LAYERS = list(NN_DICT.keys())
NN_NAMES = list(NN_DICT.values())
SORTING_LATEX = [meta.sc_to_latex.get(s) for s in SORTING]

import backtest # must come after defining the ABOVE variables



def plot_dummy(p=None):
    plt.plot([1, 2, 3, 4])
    plt.ylabel('Some Numbers')
    fig = LatexFigure(plt.gcf())
    fig.fit(square=True)
    if p is not None: 
        fig.save(p)


class LatexTable():
    def __init__(self, tab:str):
        if (isinstance(tab, pd.DataFrame) or isinstance(tab, pd.Series)):
            tab = tab.to_latex()
        self.tab = tab
    
    def save(self, p):
        if p is not None: 
            with open(p,'w') as tf:
                tf.write(self.tab)
            print("Table  saved to {}".format(p))


class LatexFigure():
    def __init__(self, fig):
        """
        Arguments: 
            fig: matplotlib fig
        """
        self.fig = fig
        self._setup()

    def fit(self, scale=1, square=False): 
        """
        Scales matplotplib figure to fit page size using provided scale
        """
        self.fig.set_size_inches(self._figsize(scale,square=square))
    
    def save(self, p:str):
        if p is not None: 
            self.fig.savefig(p, bbox_inches='tight')
            print("Figure saved to {}".format(p))
    
    

    @staticmethod
    def _figsize(scale:int, square:bool):
        fig_width_pt = 401.18405                        # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        if square: 
            fig_height=fig_width
        else: 
            golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
            fig_height = fig_width*golden_mean              # height in inches
        fig_size = [fig_width,fig_height]
        return fig_size

    def _setup(self):
        """
        Sets up matplolib so that all figures look latexy
        """
        pgf_with_latex = {                      # setup matplotlib to use latex for output
            "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
            "text.usetex": True,                # use LaTeX to write all text
            "font.family": "serif",
            "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
            "font.sans-serif": [],
            "font.monospace": [],
            "axes.labelsize": 10,               # LaTeX default is 10pt font.
            "font.size": 10,
            "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.figsize": self._figsize(0.9, square=False),     # default fig size of 0.9 textwidth
            "pgf.preamble": "\n".join([
                r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
                r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
                ])
            }
        mpl.rcParams.update(pgf_with_latex)


def tabulate_meta(SORTING_LATEX=SORTING_LATEX, p=None):
    df = meta.signals
    df = df[["name_tex", "class", "class2", "tex_cite", "freq"]]
    df.set_index('name_tex', inplace=True)
    df.index.name = "Feature"
    df = df.loc[SORTING_LATEX]

    class_dict = {"frictions":"Market", "fund":"Accounting","IBES":"Analyst Forecasts"}
    freq_dict = {"monthly":"M", "annual_july":"Y"}
    tex_cite_dict = {old: "\cite{" + old + "}" for old in list(df.tex_cite.values)}
    df.replace({"class": class_dict, "freq":freq_dict, "tex_cite":tex_cite_dict}, inplace=True)

    cdict = {
        'name_tex':"Feature", 
        "tex_cite":"Author", 
        "freq":"Frequency", 
        "class":"Data Source", 
        "class2":"Category"}
    df = df.rename(columns = cdict)

    tab = LatexTable(df.to_latex(escape=False))
    tab.save(p)
    return df

def tabulate_characteristics_motivation(SORTING_LATEX = SORTING_LATEX, p=None):
    df = meta.signals[["name_tex", "tex_cite", "class2", "journal", "sign"]]
    tex_cite_dict = {old: "\cite{" + old + "}" for old in list(df.tex_cite.values)}
    df.replace({"tex_cite":tex_cite_dict}, inplace=True)
    df = df[df["name_tex"].isin(SORTING_LATEX)]
    df["class2"] = df["class2"].str.title() 
    cdict = {
        'name_tex':"Feature", 
        "tex_cite":"Author", 
        "class2":"Category",
        "journal":"Journal", 
        "sign":"Sign"}
    df = df.rename(columns = cdict)
    df.set_index(["Category", "Feature"],inplace=True)
    df.sort_index(inplace=True)

    tab = LatexTable(df.to_latex(escape=False))
    tab.save(p)
    return df

def plot_missing_observations(df, p=None):
    nas = (df.isna().sum()/df.isna().count())*100
    fig, axis = plt.subplots(figsize=(8,8))
    plt.plot(nas, np.arange(len(nas)), 'o')
    
    # Y Ticks
    labels = [meta.sc_to_latex.get(s) for s in nas.index.values.tolist()]
    plt.yticks(np.arange(0, len(nas.index), 1), labels=labels)
    axis.invert_yaxis()
    
    # X label 
    axis.set_xlabel("Percentage of Missing Values")
    
    fig = LatexFigure(plt.gcf())
    fig.fit(square=True)
    fig.save(p)

def _numfmt(x, pos): # your custom formatter function: divide by 100.0
    s = '{}'.format(x / 1000.0)
    return s

def plot_histograms(df, p=None):
    df.columns = [meta.sc_to_latex.get(s) for s in df.columns.tolist()]
    figure = df.hist(sharex=True, sharey=False, xlabelsize=16, ylabelsize=16, xrot=90)
    [x.title.set_size(26) for x in figure.ravel()]
    yfmt = tkr.FuncFormatter(_numfmt)
    [x.yaxis.set_major_formatter(yfmt) for x in figure.ravel()]
    fig = LatexFigure(plt.gcf())
    fig.fit(scale=5)
    fig.save(p)
    return fig

def plot_correlation_matrix(df, p=None):
    corr = df.corr()
    corr.rename(index=meta.sc_to_latex, inplace=True)
    corr.columns = corr.index.values
    _corrplot(corr, size_scale=30, legend=True)
    fig = LatexFigure(plt.gcf())
    fig.fit(square=True)
    fig.save(p)

def plot_standard_deviation(df,p=None):
    std = df.std()
    std.rename(index=meta.sc_to_latex, inplace=True)
    std.columns = std.index.values
    fig, axis = plt.subplots(figsize=(8,8))
    plt.plot(std, np.arange(len(std)), 'o')
    plt.yticks(np.arange(0, len(std.index), 1), std.index.values.tolist())
    axis.invert_yaxis()
    fig = LatexFigure(plt.gcf())
    fig.fit(square=True)
    fig.save(p)

def plot_correlation_matrix_highest(df, p=None):
    corr = df.corr()
    corr.rename(index=meta.sc_to_latex, inplace=True)
    corr.columns = corr.index.values
    so = corr.unstack().abs().sort_values()
    hi = so[so!=1].iloc[::2].tail(10).sort_values(ascending=False)
    highest = list(set(hi.index.get_level_values(0)).union(set(hi.index.get_level_values(1))))
    df = corr.loc[highest,highest]
    ordered_index = [s for s in SORTING_LATEX if s in list(df.columns)]
    df.columns = ordered_index
    df.index = ordered_index
    _corrplot(df, size_scale=250, legend=True)
    fig = LatexFigure(plt.gcf())
    fig.fit(square=True)
    fig.save(p)
    
def tabulate_correlation_matrix(df, p=None):
    corr = df.corr()
    corr.rename(index=meta.sc_to_latex, inplace=True)
    corr.columns = corr.index.values
    corr = corr.round(3)
    tab = corr.to_latex()

    # Rotate header 
    break_one, break_two = "\\toprule\n", "\\\\\n\\midrule"
    first, second = tab.split(break_one)
    second, third = second.split(break_two)
    second = "&".join(["\\rot{" +s + "}" for s in second.split("&")])
    tab = LatexTable(first +  break_one + second + break_two + third)
    tab.save(p)
    return corr

def tabulate_most_correlated_pairs(df, p=None):
    corr = df.corr()
    corr.rename(index=meta.sc_to_latex, inplace=True)
    corr.columns = corr.index.values
    so = corr.unstack().abs().sort_values()
    hi = so[so!=1].iloc[::2].tail(10).sort_values(ascending=False)
    highest = list(set(hi.index.get_level_values(0)).union(set(hi.index.get_level_values(1))))
    so = corr.loc[highest,highest].unstack().sort_values()
    most = so[so!=1].iloc[::2].abs().sort_values(ascending=False).head(10).round(3).index
    df = pd.DataFrame(so.loc[most].round(3))
    df.columns = ["Correlation Coefficient"]
    tab = LatexTable(df)
    tab.save(p)
    return df 

def tabulate_descriptives(df, p=None):
    df = df.describe().transpose().round(4)
    df.rename(index=meta.sc_to_latex, inplace=True)
    df = df[df.columns.tolist()[1:]] # Omit count
    df.columns = ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    tab = LatexTable(df)
    tab.save(p)
    return df

def plot_returns_descriptives(r, p=None):
    # Create subplot of 3 figures
    fig, axes = plt.subplots(nrows=3,ncols=1, sharex=True, figsize=(20,15))
    fontsize = 25

    # Histogram 
    r.plot.hist(grid=True, bins=100, rwidth=0.9, ax=axes[0])
    axes[0].set_ylabel('Number of Observations', fontsize = fontsize)

    # Boxplot
    sns.boxplot(x=r, ax=axes[1], showfliers=False)

    # Deciles 
    out, percentiles = pd.qcut(r,100, retbins=True)
    percentiles = pd.DataFrame(percentiles, index = list(range(0,101,1)),columns=["r"])
    percentiles.reset_index(inplace=True)
    percentiles.plot(x="r",y="index", ax = axes[2], legend=False)
    axes[2].set_ylabel('Return Percentile', fontsize = fontsize)

    # Common x label 
    plt.xlabel("Monthly Return", fontsize = fontsize)

    # Increase font size of ticks
    for ax in axes:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

    # Convert to LatexFigure
    fig = LatexFigure(fig)
    fig.save(p)
    return fig

def tabulate_return_descriptives(r, p=None):
    df = pd.DataFrame(r.describe()[1:]).round(3).transpose()
    df.columns = ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    df.index = ["Return"]
    tab = LatexTable(df)
    tab.save(p)
    return df

def tabulate_return_deciles(r, p=None):
    out, bins = pd.qcut(r,10,retbins=True)
    bins = bins[1:-1]
    deciles = pd.DataFrame(bins,index=["{}%".format(i) for i in list(range(10,100,10))])
    df = deciles.round(3).transpose()
    df.index = ["Return"]
    tab = LatexTable(df)
    tab.save(p)
    return df 

def tabulate_backtest_descriptives_ls(path_to_backtests, hidden_layers, p=None):
    df = backtest.get_metrics_multiple_longshort(path_to_backtests, hidden_layers=hidden_layers)
    df = df.round(3)
    tab = LatexTable(df.to_latex())
    tab.save(p)
    return df

def plot_backtest_cumreturns_ls(path_to_backtests, hidden_layers, p=None):
    l, s, ls, al = backtest.get_cumulative_returns(path_to_backtests=path_to_backtests, hidden_layers=hidden_layers)
    df = pd.DataFrame(l, columns=["Long"])
    df["Short"] = - s 
    df["Long-Short"] = ls 
    df["Market"] = al
    axis = df.plot()
    axis.set_ylabel("Cumulative Return")
    axis.set_xlabel("")
    fig = LatexFigure(plt.gcf())
    fig.fit()
    fig.save(p)
    return fig

def plot_backtest_histogram(path_to_backtests, hidden_layers, p=None):
    """
    Arguments: 
        returns: pd.Series
    """
    _, _, ls, market =  backtest.get_returns(path_to_backtests=path_to_backtests, hidden_layers=hidden_layers)
    returns = ls
    returns = returns*100
    market = market*100

    fig, axis = plt.subplots(figsize=(8,8))
    returns.plot.hist(grid=True, bins=15, rwidth=0.9,ax=axis, alpha=0.5, label="NN1")
    market.plot.hist(grid=True, bins=15, rwidth=0.9,ax=axis, alpha=0.5, label="Market")

    sdate = pd.to_datetime(str(returns.index.values.min())).strftime('%b %Y')
    edate = pd.to_datetime(str(returns.index.values.max())).strftime('%b %Y')

    axis.set_ylabel("Number of Months ({} to {})".format(sdate, edate))
    axis.set_xlabel("Monthly Return (Percentage Points)")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.legend(loc='upper right')
    fig = LatexFigure(plt.gcf())
    fig.fit()
    fig.save(p)

def tabulate_backtest_descriptive_models(paths_to_backtests, hl_dict, p=None):
    metrics = dict()
    for p in paths_to_backtests:
        _, _, ret, market  = backtest.get_returns(p, hl_dict.get(p), percent_long=10, percent_short=10)
        m = backtest.get_metrics_single_model(ret)
        m_market = backtest.get_metrics_single_model(market)
        metrics["Market"] = m_market
        metrics[NN_DICT.get(hl_dict.get(p))] = m
    df = pd.DataFrame(metrics)
    df = df.round(3)
    tab = LatexTable(df)
    tab.save(p)
    return df

def plot_backtest_cumreturns_models(path_to_backtests, hl_dict, p=None):
    returns = dict() 
    for  p in paths_to_backtests:
        _, _, ret, market  = backtest.get_cumulative_returns(p, hl_dict.get(p), 10, 10)
        returns["Market"] = market
        returns[NN_DICT.get(hl_dict.get(p))] = ret
    df = pd.DataFrame(returns)

    axis = df.plot()
    axis.set_ylabel("Cumulative Return")
    axis.set_xlabel("")
    fig = LatexFigure(plt.gcf())
    fig.fit()
    fig.save(p)
    return fig

def tabulate_performance(dp, p=None):
    dp["decile"] = [s.split("_")[0] for s in dp.index.values.tolist()]
    dp["measure"] = ["_".join(s.split("_")[1:]) for s in dp.index.values.tolist()]
    dp = dp.groupby('measure').mean()
    dp = dp.loc[["mae", "mse", "rmse","r2"]]
    dp.index=["Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "R Square"]
    dp = dp[["NN1","NN2", "NN3", "NN4"]]
    dp.columns = ["NN1","NN2", "NN3", "NN4"]
    dp.loc["R Square"] = dp.loc["R Square"] * 100
    dp = dp.round(3)
    tab = LatexTable(dp)
    tab.save(p)
    return dp

def plot_r2(dp, p=None):
    dp["decile"] = [s.split("_")[0] for s in dp.index.values.tolist()]
    dp["measure"] = ["_".join(s.split("_")[1:]) for s in dp.index.values.tolist()]
    dp = dp[dp['measure']=="r2"]
    dp.set_index(["decile"],inplace=True)
    dp = dp[["NN1","NN2", "NN3", "NN4"]]
    dp.columns = ["NN1","NN2", "NN3", "NN4"]
    dp = dp*100
    dp.plot(xlabel="Return Decile", ylabel="R Square (Percentage Points)")
    fig = LatexFigure(plt.gcf())
    fig.save(p)
    return dp


class Results():
    def __init__(self, path, SORTING=SORTING, NN_DICT=NN_DICT, NN_NAMES=NN_NAMES):
        """
        """
        self.path = path
        self.SORTING = SORTING
        self.NN_DICT=NN_DICT
        self.NN_NAMES=NN_NAMES

        self.ar = None 
        self.pe = None 
        self.ig = None 
        self.mr = None 
        self.pr = None
    
    def load(self, sort_features = True, suffix="_test", adjust_index=True):
        """
        Models are in columns, features are in rows
        """
        self.ar = pd.read_csv(os.path.join(self.path, "args.csv"), index_col=0)
        self.pe = pd.read_csv(os.path.join(self.path, "performance.csv"), index_col=0)
        self.ig = pd.read_csv(os.path.join(self.path, "integrated_gradients_global{}.csv".format(suffix)), index_col=0)
        self.mr = pd.read_csv(os.path.join(self.path, "model_reliance{}.csv".format(suffix)), index_col=0)
        self.pr = pd.read_csv(os.path.join(self.path, "portfolio_reliance.csv"), index_col=0)
        if adjust_index: 
            self.ar.index = [s.split(': ')[1] for s in self.ar.index.values]  # Get rid of the class name in index
            self.pe.index = [s.split(': ')[1] for s in self.pe.index.values]
            self.ig.index = [s.split(': ')[1] for s in self.ig.index.values]
            self.mr.index = [s.split(': ')[1] for s in self.mr.index.values]
            self.pr.index = [s.split(': ')[1] for s in self.pr.index.values]
        self.ar.sort_index(inplace=True)
        self.pe.sort_index(inplace=True)
        self.ig.sort_index(inplace=True)
        self.mr.sort_index(inplace=True)
        self.pr.sort_index(inplace=True)
        try: 
            self.dp = pd.read_csv(os.path.join(self.path, "decile_performance.csv"), index_col=0)
            if adjust_index: 
                self.dp.index = [s.split(': ')[1] for s in self.dp.index.values]
            self.dp.sort_index(inplace=True)
        except: 
            print("File {}/decile_performance.csv not found, skipping".format(self.path))

        self.ar["hidden_layers"] = self.ar["hidden_layers"].astype(str)
        self.ar["nn_name"] = self.ar[["hidden_layers"]].replace(self.NN_DICT)
        self.ar["nn_name_short"] = [s[-1:] for s in self.ar.nn_name]

        if sort_features: 
            self.ig = self.ig[self.SORTING]
            self.mr = self.mr[self.SORTING]
            self.pr = self.pr[self.SORTING]
    
    def subset(self, key_name, value):
        if not isinstance(value, list):
            value =[value]
        sub = (self.ar[key_name].isin(value)) 
        self.ar = self.ar[sub]
        self.pe = self.pe[sub]
        self.ig = self.ig[sub]
        self.mr = self.mr[sub]
        if self.pr is not None: 
            self.pr = self.pr[sub]
    
    def rename(self, keys):
        """
        keys_select = {
            "ensemble": ["nn_name"],
            "seeds": ['nn_name', "seed"],
            "ensemble_time": ["nn_name", "ytrain"]
        }
        """
        old_names = list(self.ar.index)
        lists = [self.ar.loc[old_names][var].values.astype(str).tolist() for var in keys]
        new_names = ["-".join(l) for l in list(zip(*lists))]
        self.ar.index = new_names
        self.pe.index = new_names
        self.mr.index = new_names
        self.ig.index = new_names
        if self.pr is not None: 
            self.pr.index = new_names  

class LocalIG():
    def __init__(self, path_to_models, hidden_layers):
        self.path_to_models = path_to_models
        self.hidden_layers = hidden_layers
    
    def load(self, sorting=None):
        dfs = []
        for ytrain in YTRAIN_NAMES: 
            df = pd.read_csv(os.path.join(
                self.path_to_models, 
                'y={},y=4,y=1,hl={},nm=10,o=adam'.format(ytrain, self.hidden_layers), 
                'integrated_gradients_test.csv'), dtype={'DTID':str}, index_col=[0,1])
            df.index =df.index.set_levels([df.index.levels[0], pd.to_datetime(df.index.levels[1])])
            dfs.append(df)
        df = pd.concat(dfs)
        if sorting is not None: 
            df = df[sorting]
        self.df = df

    @staticmethod
    def plot_all_observations(df, xlabel, p=None):
        df_melted = pd.DataFrame([(colname, df[colname].iloc[i]) for i in range(len(df)) for colname in df.columns], 
                    columns=['col', 'values'])
        
        # Plot
        fig, axis = plt.subplots(1,1)
        axis = sns.stripplot(x = 'values', y='col', data=df_melted)

        # Axis Labels
        axis.set_ylabel("")
        axis.set_xlabel(xlabel)

        # Y ticks 
        labels = [meta.sc_to_latex.get(label) for label in list(df.columns)]
        axis.set_yticklabels(labels)

        # Convert to LatexFigure to change font and figsize
        fig = LatexFigure(plt.gcf())
        fig.fit(scale=2)
        fig.save(p)



#==========================================================================================
#                                    Auxiliary 
#==========================================================================================


class Styling(): 
    @staticmethod
    def heatmap(df):
        return df.apply(lambda x: get_parts(x,6), axis=0)
    
    @staticmethod
    def relative(df):
        return df.apply(lambda x: divide_by_max(x), axis=0)
    
    @staticmethod
    def order(df):
        return len(df) - df.apply(lambda x: get_orders(x), axis=0)
    
    @staticmethod
    def identity(df):
        return df 


def _corrplot(data, size_scale=500, marker='s', legend=True):
    """
    Args:
        data (pd.DataFrame): df as output of pd.DataFrame.corr
    """
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    _heatmap(
        corr['x'], corr['y'], legend=legend,
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale,
    )

def _heatmap(x, y, legend, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid

    if legend: 
        ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot
    else: 
        ax = plt.subplot(plot_grid[:,:])

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    if legend:
        if color_min < color_max:
            ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

            col_x = [0]*len(palette) # Fixed x coordinate for the bars
            bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

            bar_height = bar_y[1] - bar_y[0]
            ax.barh(
                y=bar_y,
                width=[5]*len(palette), # Make bars 5 units wide
                left=col_x, # Make bars start at 0
                height=bar_height,
                color=palette,
                linewidth=0
            )
            ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
            ax.grid(False) # Hide grid
            ax.set_facecolor('white') # Make background white
            ax.set_xticks([]) # Remove horizontal ticks
            ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
            ax.yaxis.tick_right() # Show vertical ticks on the right 

def _plot_df(
            df, column_groups:list=None, cmap=plt.cm.Blues, 
            flip_cbar:bool=False, show_cbar:bool=True, 
            scale:int=2, vmin=None, group_xticks=False, convert_labels=True):
        
        # Create as many column subplots as there are column groups
        if column_groups == None: 
            column_groups = [df.columns.tolist()]
            
        fig, axes = plt.subplots(nrows=1, ncols=len(column_groups), figsize=(6.5,7),
                            gridspec_kw={"width_ratios":[len(group) for group in column_groups]})
        # fix axes if there is only one columns group from integer to list
        if len(column_groups) == 1: 
            axes = np.array(axes)
            axes = axes.reshape(-1)
        
        # Populate all axes
        vmin = df.min().min() if vmin is None else vmin
        kw = {'cmap':cmap, 'vmin':vmin, 'vmax':df.max().max()}
        for i, group in enumerate(column_groups): 
            im = axes[i].pcolor(df[group], **kw)
        
        # Y labels for first group are the index of the df
        if convert_labels:
            labels = [meta.sc_to_latex.get(label) for label in df.index.values.tolist()]
        else: 
            labels = df.index.values.tolist()
        axes[0].set_yticks(np.arange(0.5, len(df.index)))
        axes[0].set_yticklabels(labels)
        axes[0].invert_yaxis()
        
        # Y labels for rest of groups are empty
        for i in range(1,len(column_groups)):
            axes[i].invert_yaxis()
            axes[i].set_yticks([])
        
        # Set X labels for each group (column names of the df) 
        for i, group in enumerate(column_groups): 
            if group_xticks:
                axes[i].set_xticks([np.median(np.arange(0.5, len(group), 1))])
                axes[i].set_xticklabels([group[0].split("-")[0]])
            else: 
                axes[i].set_xticks(np.arange(0.5, len(group), 1))
                axes[i].set_xticklabels(group)
            axes[i].xaxis.set_ticks_position('top')    
        
        
        # Make the width of space between subplots smaller
        plt.subplots_adjust(wspace=0.025)
        
        if show_cbar: 
            #fig.colorbar(im, ax=axes.ravel().tolist())
            cbar = fig.colorbar(im, ax=axes.ravel().tolist())
            if flip_cbar:
                cbar.ax.invert_yaxis() 
        
        # Convert to LatexFigure to change font and figsize
        fig = LatexFigure(plt.gcf())
        fig.fit(scale=scale)
        return fig

#==========================================================================================
#                                     MAIN
#==========================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_figures", default="latex/Figures", type=str, help="Folder where to save the figures.")
    parser.add_argument("--path_tables", default="latex/Tables", type=str, help="Folder where to save the figures.")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    mpl.use('pgf')
    sns.set()

    plot_dummy(p=os.path.join(args.path_figures,"dummy.pdf"))

    #==========================================================================================
    #                                  Meta Information about Predictors 
    #==========================================================================================
    # Table 3.1 Predictors 
    tabulate_meta(p=os.path.join(args.path_tables,"meta.tex"),SORTING_LATEX=SORTING_LATEX)

    # Table 2.1 Economic Motivation of Predictors Used in This Thesis
    tabulate_characteristics_motivation(p=os.path.join(args.path_tables,"characteristics_motivation.tex"),SORTING_LATEX=SORTING_LATEX)
    
    #==========================================================================================
    #                                Descriptive Statistics On Not Cleaned Data
    #==========================================================================================
    # Load Data
    dt = Subset()
    dt.load()
    df = dt.features[SORTING]

    # Figure 3.1 Amount of Missing Observations in Individual Features
    plot_missing_observations(df, p=os.path.join(args.path_figures,"missing_observations.pdf"))
    

    #==========================================================================================
    #                                 Descriptive Statistics On Cleaned Data
    #==========================================================================================
    # Load data 
    dt = MinMaxed()
    dt.load()
    df = dt.features[SORTING]
    r = dt.targets["r"]*100
    r_without_outliers = r[np.abs((r - r.mean())/r.std(ddof=0)<8)] 
    
    # Figure 3.2 Standard Deviation of the Features 
    plot_standard_deviation(df,p=os.path.join(args.path_figures,"standard_deviation.pdf"))

    # Figure 3.3 Features Correlation Matrix 
    plot_correlation_matrix(df, p=os.path.join(args.path_figures,"correlation_matrix.pdf"))

    # Figure 3.4 10 Most Correlated Pairs of Features
    # Panel (a)
    plot_correlation_matrix_highest(df, p=os.path.join(args.path_figures,"correlation_matrix_highest.pdf"))
    # Panel (b)
    tabulate_most_correlated_pairs(df, p=os.path.join(args.path_tables,"most_correlated_pairs.tex"))

    # Figure A.1 Histograms of All Features 
    plot_histograms(df, p=os.path.join(args.path_figures,"histograms.pdf"))

    # Figure 3.5 Descriptive Statistics of Monthly Returns
    # Panel (a) 
    plot_returns_descriptives(r_without_outliers, p=os.path.join(args.path_figures, "returns_descriptives.pdf"))
    # Panel (b) Return Descriptives
    tabulate_return_deciles(r_without_outliers,  p=os.path.join(args.path_tables,"return_descriptives.tex")))
    # Panel (c) Return Deciles
    tabulate_return_deciles(r_without_outliers, p=os.path.join(args.path_tables,"return_deciles.tex")))

    # Table A.1 Descpriptive Statistics of All Features
    tabulate_descriptives(df, p=os.path.join(args.path_tables,"descriptives.tex"))
    
    # Table A.2 Features Correlation Matrix
    tabulate_correlation_matrix(df, p=os.path.join(args.path_tables,"correlation_matrix.tex"))
    
    #==========================================================================================
    #                                      Predictive Performance
    #==========================================================================================
    r_all = Results('results/all/ensembles')
    r_all.load(adjust_index=False)
    dp = r_all.dp.groupby(r_all.ar.nn_name).median().transpose()

    # Figure 4.1 Out-of-Sample Predictive Ability of the Networks
    # Panel (a) 
    tabulate_performance(dp, p=os.path.join(args.path_tables,"performance.tex"))
    # Panel (b)
    plot_r2(dp, p=os.path.join(args.path_figures,"r2.pdf"))

    #==========================================================================================
    #                                      Backtest
    #==========================================================================================
    paths_to_backtests = ['models/nn1/ensembles','models/nn2/ensembles','models/nn3/ensembles','models/nn4/ensembles']
    hl_dict = {
        'models/nn1/ensembles':'32',
        'models/nn2/ensembles':'32,16',
        'models/nn3/ensembles':'32,16,8',
        'models/nn4/ensembles':'32,16,8,4', 
        'models/lr/ensembles':'-1'}
    
    tabulate_backtest_descriptive_models(paths_to_backtests, hl_dict, p=os.path.join(args.path_tables,"backtest_descriptive_models.tex"))
    
    # Figure 4.2 Cumulative Returns on the Long-Short Portfolio
    # Panel (a)
    plot_backtest_cumreturns_models(paths_to_backtests, hl_dict, p=os.path.join(args.path_figures,"backtest_cumreturns_models.pdf"))
    # Panel (b)
    plot_backtest_cumreturns_ls(path_to_backtests='models/nn1/ensembles', hidden_layers='32', 
                                          p=os.path.join(args.path_figures,"backtest_cumreturns_ls.pdf"))
    
    # Figure 4.3 Descriptive Statistics of the Returns on Long-Short Portfolios
    # Panel (a)
    tabulate_backtest_descriptives_ls('models/nn1/ensembles', "32",p=os.path.join(args.path_tables,"backtest_descriptives_ls.tex"))
    # Panel (b)
    plot_backtest_histogram('models/nn1/ensembles', "32", p=os.path.join(args.path_figures,"backtest_histogram.pdf"))
    

    
    
    #==========================================================================================
    #                                    Global Interpretation Plots
    #==========================================================================================

    # -----------------------------------------------------------------------------------------
    # AGGREGATE ENSEMBLE RESULTS 
    # -----------------------------------------------------------------------------------------
    r_nn1 = Results('results/nn1/ensembles')  
    r_nn2 = Results('results/nn2/ensembles')  
    r_nn3 = Results('results/nn3/ensembles')  
    r_nn4 = Results('results/nn4/ensembles')
    r_lr = Results('results/lr/ensembles')  

    for r in [r_lr, r_nn1, r_nn2, r_nn3, r_nn4]: 
        r.load()

    ar = pd.concat([r_lr.ar, r_nn1.ar, r_nn2.ar, r_nn3.ar, r_nn4.ar])
    pe = pd.concat([r_lr.pe, r_nn1.pe, r_nn2.pe, r_nn3.pe, r_nn4.pe])
    ig = pd.concat([r_lr.ig, r_nn1.ig, r_nn2.ig, r_nn3.ig, r_nn4.ig])
    mr = pd.concat([r_lr.mr, r_nn1.mr, r_nn2.mr, r_nn3.mr, r_nn4.mr])
    pr = pd.concat([r_lr.pr, r_nn1.pr, r_nn2.pr, r_nn3.pr, r_nn4.pr])
    dp =  pd.concat([r_lr.dp, r_nn1.dp, r_nn2.dp, r_nn3.dp, r_nn4.dp])

    ar.to_csv('results/all/ensembles/args.csv')
    pe.to_csv('results/all/ensembles/performance.csv')
    ig.to_csv('results/all/ensembles/integrated_gradients_global_test.csv')
    mr.to_csv('results/all/ensembles/model_reliance_test.csv')
    pr.to_csv('results/all/ensembles/portfolio_reliance.csv')
    dp.to_csv('results/all/ensembles/decile_performance.csv')

    # -----------------------------------------------------------------------------------------
    # AGGREGATE SEED RESULTS
    # -----------------------------------------------------------------------------------------
    r_nn1 = Results('results/nn1/seeds')  
    r_nn2 = Results('results/nn2/seeds')  
    r_nn3 = Results('results/nn3/seeds')  
    r_nn4 = Results('results/nn4/seeds')  

    for r in [r_nn1, r_nn2, r_nn3, r_nn4]: 
        r.load()

    ar = pd.concat([r_nn1.ar, r_nn2.ar, r_nn3.ar, r_nn4.ar])
    pe = pd.concat([r_nn1.pe, r_nn2.pe, r_nn3.pe, r_nn4.pe])
    ig = pd.concat([r_nn1.ig, r_nn2.ig, r_nn3.ig, r_nn4.ig])
    mr = pd.concat([r_nn1.mr, r_nn2.mr, r_nn3.mr, r_nn4.mr])
    pr = pd.concat([r_nn1.pr, r_nn2.pr, r_nn3.pr, r_nn4.pr])

    ar.to_csv('results/all/seeds/args.csv')
    pe.to_csv('results/all/seeds/performance.csv')
    ig.to_csv('results/all/seeds/integrated_gradients_global_test.csv')
    mr.to_csv('results/all/seeds/model_reliance_test.csv')
    pr.to_csv('results/all/seeds/portfolio_reliance.csv')

    # -----------------------------------------------------------------------------------------
    # PLOT MAIN INTERPRETABILITY RESULTS
    # ----------------------------------------------------------------------------------------- 
    r = Results('results/all/ensembles')
    r.load(adjust_index=False)
    r.rename(['nn_name'])

    dfs_to_plot = {
        'ig':r.ig.groupby(r.ar.nn_name).mean().transpose()*100,
        'mr':r.mr.groupby(r.ar.nn_name).mean().transpose(),
        'pr':r.pr.groupby(r.ar.nn_name).mean().transpose()*100}

    vmins = {
        'ig':0,
        'mr':1,
        'pr':0
    }

    SORTINGS = {
        'ig': dfs_to_plot.get('ig')[["NN1","NN2","NN3","NN4"]].mean(axis=1).sort_values(ascending=False).index,
        'mr': dfs_to_plot.get('mr')[["NN1","NN2","NN3","NN4"]].mean(axis=1).sort_values(ascending=False).index,
        'pr': dfs_to_plot.get('pr')[["NN1","NN2","NN3","NN4"]].mean(axis=1).sort_values(ascending=False).index,
    }

    for name, df in dfs_to_plot.items(): 
        df["Mean"] = df[["NN1","NN2","NN3","NN4"]].mean(axis=1)


    # blues (with vmin)
    for name, df in dfs_to_plot.items(): 
        fig = _plot_df(
                    df.loc[SORTINGS.get(name)], column_groups=[["LR"],["NN1", "NN2", "NN3", "NN4"], ["Mean"]], group_xticks=False, 
                    cmap=plt.cm.Blues, flip_cbar=False, show_cbar=True, 
                    vmin=vmins.get(name), convert_labels=True)
        fig.save('latex/Figures/{}_blues.pdf'.format(name))

    for name, df in dfs_to_plot.items(): 
        df = df.loc[SORTINGS.get(name)]
        df.index = [meta.sc_to_latex.get(s) for s in df.index.values.tolist()]
        df.columns = ["LR", "NN1", "NN2", "NN3", "NN4", "Mean"]
        df = df.round(3)
        tab = LatexTable(df)
        tab.save('latex/Tables/{}_blues.tex'.format(name))


    # order
    for name, df in dfs_to_plot.items():
        df = Styling.order(df)
        fig = _plot_df(
                    df.loc[SORTINGS.get(name)], column_groups=[["LR"],["NN1", "NN2", "NN3", "NN4"], ["Mean"]], group_xticks=False, 
                    cmap=plt.cm.inferno_r, flip_cbar=True, show_cbar=True, 
                    vmin=None, convert_labels=True)
        fig.save('latex/Figures/{}_order.pdf'.format(name))


    # relative
    for name, df in dfs_to_plot.items(): 
        df = Styling.relative(df)
        fig = _plot_df(
                    df.loc[SORTINGS.get(name)], column_groups=[["LR"],["NN1", "NN2", "NN3", "NN4"], ["Mean"]], group_xticks=False, 
                    cmap=plt.cm.Greens, flip_cbar=False, show_cbar=True, 
                    vmin=vmins.get(name), convert_labels=True)
        fig.save('latex/Figures/{}_relative.pdf'.format(name))


    # relative without LR
    for name, df in dfs_to_plot.items(): 
        df = Styling.relative(df)
        fig = _plot_df(
                    df.loc[SORTINGS.get(name)], column_groups=[["NN1", "NN2", "NN3", "NN4"]], group_xticks=False, 
                    cmap=plt.cm.Greens, flip_cbar=False, show_cbar=True, 
                    vmin=vmins.get(name), convert_labels=True)
        fig.save('latex/Figures/{}_relative_without_lr.pdf'.format(name))

        lig = LocalIG('models/nn1/ensembles', hidden_layers="32")

    # ----------------------------------------------------------------------------------------- 
    # PLOT COMPARISON OF INTEGRATED GRADIENTS AND PORTFOLIO RELIANCE
    # -----------------------------------------------------------------------------------------

    r = Results('results/all/ensembles')
    r.load(adjust_index=False)
    r.rename(['nn_name'])

    ig = r.ig.groupby(r.ar.nn_name).mean().transpose()*100
    pr = r.pr.groupby(r.ar.nn_name).mean().transpose()*100
    ig = ig[["NN1","NN2","NN3","NN4"]]
    ig["Mean"] = ig.mean(axis=1)
    ig.columns = ["NN1-IG","NN2-IG","NN3-IG","NN4-IG", "Mean-IG"]
    pr = pr[["NN1","NN2","NN3","NN4"]]
    pr["Mean"] = pr.mean(axis=1)
    pr.columns = ["NN1-PR","NN2-PR","NN3-PR","NN4-PR", "Mean-PR"]

    df = pd.concat([ig,pr],axis=1)
    df = df.loc[SORTINGS.get('ig')]

    df = Styling.relative(df)
    fig = _plot_df(
                df, column_groups=[["NN1-IG", "NN1-PR"], ["NN2-IG","NN2-PR"], ["NN3-IG","NN3-PR"], ["NN4-IG","NN4-PR"],["Mean-IG","Mean-PR"]], group_xticks=False, 
                cmap=plt.cm.Greens, flip_cbar=False, show_cbar=True, 
                vmin=0, convert_labels=True)
    fig.save('latex/Figures/ig_pr_comparison.pdf')


    # ----------------------------------------------------------------------------------------- 
    # PLOT TIME DECOMPOSITIONS
    # ----------------------------------------------------------------------------------------- 
    r = Results('results/all/ensembles')
    r.load(adjust_index=False)
    r.rename(['nn_name', 'ytrain'])
    r.subset("hidden_layers",["32","32,16","32,16,8","32,16,8,4"])

    dfs_to_plot = {
        'ig': r.ig.transpose(),
        'mr': r.mr.transpose(),
        'pr': r.pr.transpose()
    }

    for name, df in dfs_to_plot.items(): 
        df = Styling.relative(df)
        column_groups = list(
            utils.chunks(["{}-{}".format(n,s) for n,s in itertools.product(
                NN_NAMES[1:],YTRAIN_NAMES)],N_YTRAIN))
        fig = _plot_df(
                    df.loc[SORTINGS.get(name)], column_groups=column_groups, group_xticks=True, 
                    cmap=plt.cm.Greens, flip_cbar=False, show_cbar=True, 
                    vmin=vmins.get(name), convert_labels=True)
        fig.save('latex/Figures/{}_time_relative.pdf'.format(name))

    dfs_to_plot = {
        'ig': r.ig.groupby(r.ar["ytrain"]).mean().transpose(),
        'mr': r.mr.groupby(r.ar["ytrain"]).mean().transpose(),
        'pr': r.pr.groupby(r.ar["ytrain"]).mean().transpose()
    }

    for name, df in dfs_to_plot.items(): 
        df = Styling.relative(df)
        df.columns = [str(i) for i in dfs_to_plot.get("ig").columns.tolist()]
        df.rename(columns=YTRAIN_NAMES_TO_TEST_YEAR, inplace=True)
        fig = _plot_df(
                    df.loc[SORTINGS.get(name)], column_groups=None, group_xticks=False, 
                    cmap=plt.cm.Greens, flip_cbar=False, show_cbar=True, 
                    vmin=vmins.get(name), convert_labels=True)
        fig.save('latex/Figures/{}_time_relative_mean.pdf'.format(name))

    # ----------------------------------------------------------------------------------------- 
    # PLOT SEED DECOMPOSITIONS
    # ----------------------------------------------------------------------------------------- 
    r = Results('results/all/seeds')
    r.load(adjust_index=False)
    r.rename(['nn_name', 'ytrain', 'seed'])

    ig = r.ig.groupby([r.ar.nn_name, r.ar.seed]).mean()
    ig.index = ["{}-{}".format(n,s) for n,s in itertools.product(NN_NAMES[1:],SEED_NAMES)]
    ig = ig.transpose()

    mr = r.mr.groupby([r.ar.nn_name, r.ar.seed]).mean()
    mr.index = ["{}-{}".format(n,s) for n,s in itertools.product(NN_NAMES[1:],SEED_NAMES)]
    mr = mr.transpose()

    pr = r.pr.groupby([r.ar.nn_name, r.ar.seed]).mean()
    pr.index = ["{}-{}".format(n,s) for n,s in itertools.product(NN_NAMES[1:],SEED_NAMES)]
    pr = pr.transpose()

    dfs_to_plot = {
        'ig': ig,
        'mr': mr,
        'pr': pr
    }

    for name, df in dfs_to_plot.items(): 
        df = Styling.relative(df)
        column_groups = list(
            utils.chunks(["{}-{}".format(n,s) for n,s in itertools.product(NN_NAMES[1:],SEED_NAMES)],N_SEEDS))
        fig = _plot_df(
                    df.loc[SORTINGS.get(name)], column_groups=column_groups, group_xticks=True, 
                    cmap=plt.cm.Greens, flip_cbar=False, show_cbar=True, 
                    vmin=vmins.get(name), convert_labels=True)
        fig.save('latex/Figures/{}_seeds_relative.pdf'.format(name))


    #==========================================================================================
    #                                    Local Interpretation Plots
    #==========================================================================================

    # ----------------------------------------------------------------------------------------- 
    # CALCULATE INTEGRATED GRADIENT COEFFICIENTS
    # -----------------------------------------------------------------------------------------
    lig = figures.LocalIG('models/nn1/ensembles', hidden_layers="32")

    lig.load(sorting=SORTINGS.get('ig').values.tolist())
    lig.df = lig.df*100

    from data import MinMaxed
    dt = MinMaxed()
    dt.load()

    features = dt.features.loc[lig.df.index]
    features = features[lig.df.columns.tolist()]

    coefs = lig.df / features

    # ----------------------------------------------------------------------------------------- 
    # PLOT INTEGRATED GRADIENT COEFFICIENTS
    # -----------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1,1)
    axis = sns.boxplot(data=coefs, orient="h", showfliers=False)

    # Axis Labels
    axis.set_xlabel("Local Integrated Gradient / Feature Value")
    axis.set_ylabel("")

    # Y ticks 
    labels = [meta.sc_to_latex.get(label) for label in list(coefs.columns)]
    axis.set_yticklabels(labels)

    # Convert to LatexFigure to change font and figsize
    fig = figures.LatexFigure(plt.gcf())
    fig.fit(scale=2)
    fig.save('latex/Figures/ig_coefs.pdf')

    # ----------------------------------------------------------------------------------------- 
    # TABULATE INTEGRATED GRADIENT COEFFICIENTS
    # -----------------------------------------------------------------------------------------
    coefs = lig.df / features
    coefs = coefs.describe().transpose()
    coefs = coefs[["mean", "std"]]
    coefs["Mean / Std"] = coefs["mean"] / coefs["std"]

    df = meta.signals[meta.signals["important_otmh_global_liquid"]<=30][["sc", "sign"]]
    df.set_index("sc", inplace=True)
    coefs = coefs.join(df)

    coefs["Sign (Here)"] = np.sign(coefs["mean"]).astype(int)
    coefs.rename(columns={"mean":"Mean", "std":"Std", "sign":"Sign (Original)"}, inplace=True)

    coefs.index = [meta.sc_to_latex.get(s) for s in coefs.index]
    coefs = coefs.round(3)

    tab = LatexTable(coefs)
    tab.save("latex/Tables/coefs.tex")

    # ----------------------------------------------------------------------------------------- 
    # PLOT INTEGRATED GRADIENT BOXPLOT
    # -----------------------------------------------------------------------------------------
    df = lig.df

    fig, axes = plt.subplots(1,1)
    axis = sns.boxplot(data=df, orient="h", showfliers=False)

    # Axis Labels
    axis.set_xlabel("Local Integrated Gradient")
    axis.set_ylabel("")

    # Y ticks 
    labels = [meta.sc_to_latex.get(label) for label in list(df.columns)]
    axis.set_yticklabels(labels)

    # Convert to LatexFigure to change font and figsize
    fig = LatexFigure(plt.gcf())
    fig.fit(scale=2)
    fig.save('latex/Figures/ig_boxplot.pdf')

    # ----------------------------------------------------------------------------------------- 
    # PLOT INTEGRATED GRADIENT SINGLE OBSERVATION
    # -----------------------------------------------------------------------------------------
    lig.plot_all_observations(lig.df[:1],xlabel="Local Integrated Gradient", p ="latex/Figures/local_ig_1.pdf")

