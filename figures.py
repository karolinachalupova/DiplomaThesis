"""
Utils for exploratory data analysis.
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

#==========================================================================================
#                                     On metadata
#==========================================================================================
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



#==========================================================================================
#                                     On not cleaned features
#==========================================================================================
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

#==========================================================================================
#                                     On cleaned features 
#==========================================================================================
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


#==========================================================================================
#                                     On cleaned returns
#==========================================================================================

def plot_returns_histogram(r, p=None):
    r_without_outliers = r[np.abs((r - r.mean())/r.std(ddof=0)<8)] # zscore more than 8 is considered outlier
    r_without_outliers.plot.hist(grid=True, bins=100, rwidth=0.9)
    plt.xlabel('Monthly Return')
    plt.ylabel('Number of Observations')
    fig = LatexFigure(plt.gcf())
    fig.fit()
    fig.save(p)


#==========================================================================================
#                                     On backtest predictions
#==========================================================================================
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


def tabulate_backtest_descriptives_models(path_to_backtests, HIDDEN_LAYERS, NN_DICT, p=None):
    df = backtest.get_metrics_all_models(path_to_backtests = path_to_backtests, 
                                    HIDDEN_LAYERS=HIDDEN_LAYERS, NN_DICT=NN_DICT).round(3)
    tab = LatexTable(df)
    tab.save(p)
    return df


def plot_backtest_cumreturns_models(path_to_backtests, HIDDEN_LAYERS, NN_DICT, p=None):
    df = backtest.get_cumulative_returns_all_models(path_to_backtests = path_to_backtests,
                                            HIDDEN_LAYERS=HIDDEN_LAYERS, NN_DICT=NN_DICT)
    axis = df.plot()
    axis.set_ylabel("Gross Cumulative Return")
    axis.set_xlabel("")
    fig = LatexFigure(plt.gcf())
    fig.fit()
    fig.save(p)

#==========================================================================================
#                                     On results
#==========================================================================================

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
            if self.pr is not None: 
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
    
    
    def tabulate_performance_single_model(self, p=None):
        df = self.pe.groupby([self.ar.ytrain])[["test_r_square", "test_mse", "test_mean_absolute_error", "test_root_mean_squared_error"]].apply(np.median)
        return df

    def tabulate_decile_performance(self, p=None):
        dp = pd.DataFrame(self.dp.median())
        dp["decile"] = [s.split("_")[0] for s in dp.index.values.tolist()]
        dp["measure"] = ["_".join(s.split("_")[1:]) for s in dp.index.values.tolist()]
        dp.set_index(["decile", "measure"],inplace=True)
        dp = dp.unstack(level=1)
        dp.columns = dp.columns.droplevel(0)
        dp.index = [int(s) for s in dp.index.values.tolist()]
        dp.sort_index(inplace=True)
        dp["r2"] = dp["r2"]*100
        dp.round(3).transpose()
        return dp
    
    def tabulate_performance(self, p=None):
        df = self.pe.groupby([self.ar.hidden_layers])[["test_r_square", "test_mse", "test_mean_absolute_error", "test_root_mean_squared_error"]].mean()
        df.columns = ["R Square", "Mean Squared Error", "Mean Absolute Error", "Root Mean Squared Error"]
        df.index = [self.NN_DICT.get(s) for s in list(df.index.values)]
        df = df.transpose()
        df = df*100
        df = df.round(2)
        tab = LatexTable(df)
        tab.save(p)
        return df        

    def plot_pr_ensemble(self, p=None, head=None, styling="blues", add_simulated=False):
        if styling == "blues":
            vmin = 0
        else: 
            vmin = None
        fig = self.style_plot_df(self.pr.transpose().head(head), styling=styling, mode='ensemble', NN_NAMES=self.NN_NAMES, vmin=vmin, add_simulated=add_simulated)
        fig.save(p)
    
    def plot_ig_ensemble(self, p=None, head=None, styling="blues", add_simulated=False):
        if styling == "relative":
            vmin = 0
        else: 
            vmin = None
        fig = self.style_plot_df(self.ig.transpose().head(head), styling=styling, mode='ensemble', NN_NAMES=self.NN_NAMES, vmin=vmin, add_simulated=add_simulated)
        fig.save(p)
    
    def plot_mr_ensemble(self, p=None, head=None, styling="blues", add_simulated=False):
        if styling == "relative":
            vmin = None
        else: 
            vmin = None
        fig = self.style_plot_df(self.mr.transpose().head(head), styling=styling, mode='ensemble', NN_NAMES=self.NN_NAMES, vmin=vmin, add_simulated=add_simulated)
        fig.save(p)
    
    def plot_ig_time(self, p=None, styling="order", add_simulated=False):
        if styling == "relative":
            vmin = 0
        else: 
            vmin = None
        fig = self.style_plot_df(self.ig.transpose(), styling=styling, mode='ensemble_time', NN_NAMES=self.NN_NAMES,vmin=vmin, add_simulated=add_simulated)
        fig.save(p)
    
    def plot_mr_time(self, p=None, styling="order", add_simulated=False):
        if styling == "relative":
            vmin = None
        else: 
            vmin = None
        fig = self.style_plot_df(self.mr.transpose(), styling=styling, mode='ensemble_time', NN_NAMES=self.NN_NAMES,vmin=vmin, add_simulated=add_simulated)
        fig.save(p)
    
    def plot_pr_time(self, p=None, styling="order", add_simulated=False):
        if styling == "relative":
            vmin = 0
        else: 
            vmin = None
        fig = self.style_plot_df(self.pr.transpose(), styling=styling, mode='ensemble_time', NN_NAMES=self.NN_NAMES,vmin=vmin, add_simulated=add_simulated)
        fig.save(p)
    
    def plot_ig_seeds(self, p=None, styling="order", add_simulated=False):
        if styling == "relative":
            vmin = 0
        else: 
            vmin = None
        fig = self.style_plot_df(self.ig.transpose(), styling=styling, mode='seeds', NN_NAMES=self.NN_NAMES,vmin=vmin, add_simulated=add_simulated)
        fig.save(p)
        print("Failed to plot ig seeds, styling={}".format(styling))
    
    def plot_pr_seeds(self, p=None, styling="order", add_simulated=False):
        if styling == "relative":
            vmin = 0
        else: 
            vmin = None
        fig = self.style_plot_df(self.pr.transpose(), styling=styling, mode='seeds', NN_NAMES=self.NN_NAMES,vmin=vmin, add_simulated=add_simulated)
        fig.save(p)
            
    def plot_mr_seeds(self, p=None, styling="order", add_simulated=False):
        if styling == "relative":
            vmin = None
        else: 
            vmin = None
        fig = self.style_plot_df(self.mr.transpose(), styling=styling, mode='seeds', NN_NAMES=self.NN_NAMES,vmin=vmin, add_simulated=add_simulated)
        fig.save(p)
        print("Failed to plot mr seeds, styling={}".format(styling))

    @staticmethod
    def plot_df_simple(df, scale=2, vmin=0, vmax=None, cmap=plt.cm.Blues, flip_cbar=False, show_cbar=True):
        fig, axis = plt.subplots() 
        heatmap = axis.pcolor(df, cmap=cmap, vmin=vmin, vmax=vmax)
        labels = [meta.sc_to_latex.get(label) for label in df.index.values.tolist()]
        plt.yticks(np.arange(0.5, len(df.index), 1), labels)
        plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
        axis.invert_yaxis()
        axis.xaxis.tick_top()
        if show_cbar:
            cbar = plt.colorbar(heatmap)
            if flip_cbar:
                cbar.ax.invert_yaxis() 
        fig = LatexFigure(plt.gcf())
        fig.fit(scale=scale)

    @staticmethod
    def style_plot_df(df, styling, mode, vmin=None, NN_NAMES=NN_NAMES, add_simulated=False):
        stylings = {
            'heatmap':Styling("heatmap", plt.cm.inferno_r, show_cbar=True, flip_cbar=True),
            'relative':Styling("relative", plt.cm.Greens, show_cbar=True, flip_cbar=False),
            'order':Styling("order", plt.cm.inferno_r, show_cbar=True, flip_cbar=True),
            'top':Styling("top10", plt.cm.Blues, show_cbar=False, flip_cbar=False),
            'bottom':Styling("bottom10", plt.cm.Blues, show_cbar=False, flip_cbar=False),
            'top10':Styling("top10", plt.cm.Blues, show_cbar=False, flip_cbar=False),
            'bottom10':Styling("bottom10", plt.cm.Blues, show_cbar=False, flip_cbar=False),
            'top5':Styling("top5", plt.cm.Blues, show_cbar=False, flip_cbar=False),
            'bottom5':Styling("bottom5", plt.cm.Blues, show_cbar=False, flip_cbar=False),
            'blues':Styling("identity", plt.cm.Blues, show_cbar=True, flip_cbar=False)
        }
        modes = {
            'ensemble': Mode([["LR"],NN_NAMES[1:]], False),
            'seeds': Mode(list(chunks(["{}-{}".format(n,s) for n,s in itertools.product(NN_NAMES,SEED_NAMES)],N_SEEDS)),True),
            'ensemble_time': Mode(list(chunks(["{}-{}".format(n,s) for n,s in itertools.product(NN_NAMES,YTRAIN_NAMES)],N_YTRAIN)),True),
            'single_model_time':Mode(list(YTRAIN_NAMES),False)
        }
        if type(styling) == str: 
            styling = stylings.get(styling)
        if type(mode) == str: 
            mode = modes.get(mode)
        
        df = styling.transform(df)
        if add_simulated: 
            df["True"] = [1]*3 + [0]*27 # First three features are important, rest are unimportant
            column_groups = [["True"]] + mode.column_groups
            convert_labels = False
        else: 
            column_groups = mode.column_groups
            convert_labels = True
        return _plot_df(
            df, column_groups=column_groups, group_xticks=mode.group_xticks, 
            cmap=styling.cmap, flip_cbar=styling.flip_cbar, show_cbar=styling.show_cbar, 
            vmin=vmin, convert_labels=convert_labels)


#==========================================================================================
#                                     On local integrated gradients
#==========================================================================================

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
    def __init__(self, transform, cmap, show_cbar, flip_cbar): 
        self.cmap = cmap
        self.show_cbar = show_cbar 
        self.flip_cbar = flip_cbar

        transforms = {
            'heatmap': self.heatmap,
            'order':self.order,
            'relative':self.relative,
            'top10': self.top10,
            'bottom10': self.bottom10,
            'top5': self.top5,
            'bottom5': self.bottom5, 
            'identity': self.identity  
        }
        self.transform = transforms.get(transform)
     
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
    def top10(df):
        return (df.apply(lambda x: get_orders(x), axis=0)>=20)
    
    @staticmethod
    def bottom10(df):
        return (df.apply(lambda x: get_orders(x), axis=0)<=10)
    
    @staticmethod
    def top5(df):
        return (df.apply(lambda x: get_orders(x), axis=0)>=25)
    
    @staticmethod
    def bottom5(df):
        return (df.apply(lambda x: get_orders(x), axis=0)<=5)
    
    @staticmethod
    def identity(df):
        return df 

class Mode():
    def __init__(self, column_groups, group_xticks):
        self.column_groups = column_groups 
        self.group_xticks = group_xticks 


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

    """
    #==========================================================================================
    #                                     On Meta data
    #==========================================================================================
    tabulate_meta(p=os.path.join(args.path_tables,"meta.tex"),SORTING_LATEX=SORTING_LATEX)
    tabulate_characteristics_motivation(p=os.path.join(args.path_tables,"characteristics_motivation.tex"),SORTING_LATEX=SORTING_LATEX)
    
    #==========================================================================================
    #                                     On not cleaned data 
    #==========================================================================================
    dt = Subset()
    dt.load()
    df = dt.features[SORTING]
    plot_missing_observations(df, p=os.path.join(args.path_figures,"missing_observations.pdf"))
    
    
    #==========================================================================================
    #                                     On cleaned data 
    #==========================================================================================
    dt = MinMaxed()
    dt.load()
    df = dt.features[SORTING]
    r = dt.targets["r"]
    
    # Figures
    plot_histograms(df, p=os.path.join(args.path_figures,"histograms.pdf"))
    plot_correlation_matrix(df, p=os.path.join(args.path_figures,"correlation_matrix.pdf"))
    plot_correlation_matrix_highest(df, p=os.path.join(args.path_figures,"correlation_matrix_highest.pdf"))
    plot_standard_deviation(df,p=os.path.join(args.path_figures,"standard_deviation.pdf"))
    
    # Tables
    tabulate_correlation_matrix(df, p=os.path.join(args.path_tables,"correlation_matrix.tex"))
    tabulate_most_correlated_pairs(df, p=os.path.join(args.path_tables,"most_correlated_pairs.tex"))
    tabulate_descriptives(df, p=os.path.join(args.path_tables,"descriptives.tex"))

    # Figures of Returns 
    plot_returns_histogram(r, p=os.path.join(args.path_figures,"returns_histogram.pdf"))
    
    """
    #==========================================================================================
    #                                     On alfa
    #==========================================================================================
    path_to_backtests = os.path.join("models", "alfa", "ensembles")
    
    # Single model tables
    tabulate_backtest_descriptives_ls(path_to_backtests,'32',
        p=os.path.join(args.path_tables,"backtest_descriptives_ls.tex"))
    # Single model figures 
    plot_backtest_cumreturns_ls(path_to_backtests,"32",
        p=os.path.join(args.path_figures,"backtest_cumreturns_ls.pdf"))
    plot_backtest_histogram(path_to_backtests,"32",
        p=os.path.join(args.path_figures,"backtest_histogram.pdf"))


    res = Results(os.path.join("results", "alfa", "ensembles"), SORTING=SORTING, NN_DICT={"32":"NN1"}, NN_NAMES=["NN1"])
    res.load()
    res.rename(['nn_name', "ytrain"])
    res.plot_ig_time(styling="relative", p=os.path.join(args.path_figures,"ig_time_relative.pdf"))
    res.plot_mr_time(styling="relative", p=os.path.join(args.path_figures,"mr_time_relative.pdf"))
    res.plot_pr_time(styling="relative", p=os.path.join(args.path_figures,"pr_time_relative.pdf"))

    res.plot_ig_time(styling="order",p=os.path.join(args.path_figures,"ig_time_order.pdf"))
    res.plot_mr_time(styling="order", p=os.path.join(args.path_figures,"mr_time_order.pdf"))
    res.plot_pr_time(styling="order", p=os.path.join(args.path_figures,"pr_time_order.pdf"))

    res.plot_ig_time(styling="blues",p=os.path.join(args.path_figures,"ig_time_blues.pdf"))
    res.plot_mr_time(styling="blues", p=os.path.join(args.path_figures,"mr_time_blues.pdf"))
    res.plot_pr_time(styling="blues", p=os.path.join(args.path_figures,"pr_time_blues.pdf"))

    #==========================================================================================
    #                                     On beta results
    #==========================================================================================    
    #res.load()
    #res.tabulate_performance(p=os.path.join(args.path_tables,"performance.tex"))
    
    res = Results(os.path.join("results", "beta", "ensembles"))
    res.load()
    res.rename(['nn_name'])
    
    res.plot_ig_ensemble(styling="relative", p=os.path.join(args.path_figures,"ig_relative.pdf"))
    res.plot_mr_ensemble(styling="relative", p=os.path.join(args.path_figures,"mr_relative.pdf"))
    res.plot_pr_ensemble(styling="relative", p=os.path.join(args.path_figures,"pr_relative.pdf"))

    res.plot_ig_ensemble(styling="blues", p=os.path.join(args.path_figures,"ig_blues.pdf"))
    res.plot_mr_ensemble(styling="blues", p=os.path.join(args.path_figures,"mr_blues.pdf"))
    res.plot_pr_ensemble(styling="blues", p=os.path.join(args.path_figures,"pr_blues.pdf"))

    res.plot_ig_ensemble(styling="order",p=os.path.join(args.path_figures,"ig_order.pdf"))
    res.plot_mr_ensemble(styling="order",p=os.path.join(args.path_figures,"mr_order.pdf"))
    res.plot_pr_ensemble(styling="order",p=os.path.join(args.path_figures,"pr_order.pdf"))

    res.plot_ig_ensemble(styling="relative", head=10, p=os.path.join(args.path_figures,"ig_relative_head.pdf"))
    res.plot_ig_ensemble(styling="blues", head=10, p=os.path.join(args.path_figures,"ig_blues_head.pdf"))


    #==========================================================================================
    #                                     On seeds results
    #==========================================================================================
    res = Results(os.path.join("results", "beta", "seeds"), SORTING=SORTING, NN_DICT=NN_DICT, NN_NAMES=NN_NAMES)
    
    res.load(suffix="_test")
    res.rename(['nn_name', "seed"])
    res.plot_ig_seeds(styling="relative", p=os.path.join(args.path_figures,"ig_seeds_relative.pdf"))
    res.plot_pr_seeds(styling="relative", p=os.path.join(args.path_figures,"pr_seeds_relative.pdf"))
    res.plot_mr_seeds(styling="relative", p=os.path.join(args.path_figures,"mr_seeds_relative.pdf"))

    res.plot_ig_seeds(styling="blues", p=os.path.join(args.path_figures,"ig_seeds_blues.pdf"))
    res.plot_pr_seeds(styling="blues", p=os.path.join(args.path_figures,"pr_seeds_blues.pdf"))
    res.plot_mr_seeds(styling="blues", p=os.path.join(args.path_figures,"mr_seeds_blues.pdf"))

    res.plot_ig_seeds(styling="order", p=os.path.join(args.path_figures,"ig_seeds_order.pdf"))
    res.plot_pr_seeds(styling="order", p=os.path.join(args.path_figures,"pr_seeds_order.pdf"))
    res.plot_mr_seeds(styling="order", p=os.path.join(args.path_figures,"mr_seeds_order.pdf"))
    
    """
    #==========================================================================================
    #                                     On ensemble simulation results
    #==========================================================================================
    res = Results(os.path.join("results", "simulated", "ensembles"), SORTING=SORTING, NN_DICT=NN_DICT, NN_NAMES=NN_NAMES)
    
    res.load(suffix="_test", sort_features=False)
    res.tabulate_performance(p=os.path.join(args.path_tables,"sim_performance.tex"))

    res.load(suffix="_test", sort_features=False)
    res.subset('ytrain',12)
    res.rename(['nn_name'])
    res.plot_ig_ensemble(styling="relative", add_simulated=True, p=os.path.join(args.path_figures,"sim_ig_ensemble_relative.pdf"))
    res.plot_mr_ensemble(styling="relative", add_simulated=True,  p=os.path.join(args.path_figures,"sim_mr_ensemble_relative.pdf"))
    res.plot_pr_ensemble(styling="relative", add_simulated=True, p=os.path.join(args.path_figures,"sim_pr_ensemble_relative.pdf"))


    res.load(suffix="_test", sort_features=False)
    res.rename(['nn_name', "ytrain"])
    res.plot_ig_time(styling="relative", add_simulated=True, p=os.path.join(args.path_figures,"sim_ig_time_relative.pdf"))
    res.plot_mr_time(styling="relative", add_simulated=True, p=os.path.join(args.path_figures,"sim_mr_time_relative.pdf"))
    res.plot_pr_time(styling="relative", add_simulated=True, p=os.path.join(args.path_figures,"sim_pr_time_relative.pdf"))
    

    #==========================================================================================
    #                                     On simulated seeds results
    #==========================================================================================
    # Remove LR (TODO Add)
    nn_names = NN_NAMES[1:]
    nn_dict = {k: v for k, v in NN_DICT.items() if k not in {"-1"}}
    hidden_layers = HIDDEN_LAYERS[1:]
    
    res = Results(os.path.join("results", "simulated", "seeds"), SORTING=SORTING, NN_DICT=nn_dict, NN_NAMES=nn_names)
    
    res.load(suffix="_test", sort_features=False)
    res.subset('ytrain',12)
    res.subset('hidden_layers',hidden_layers)
    res.rename(['nn_name', "seed"])
    res.plot_ig_seeds(styling="relative", add_simulated=True,  p=os.path.join(args.path_figures,"sim_ig_seeds_relative.pdf"))
    res.plot_pr_seeds(styling="relative", add_simulated=True, p=os.path.join(args.path_figures,"sim_pr_seeds_relative.pdf"))
    res.plot_mr_seeds(styling="relative", add_simulated=True, p=os.path.join(args.path_figures,"sim_mr_seeds_relative.pdf"))
    """

    #==========================================================================================
    #                                     On local integrated gradients - 
    #==========================================================================================
    locig = LocalIG(os.path.join("models", "minmaxed", "ensembles"))
    locig.load(model_name='y=16,y=12,y=1,hl=32,nm=9,o=adam')
    

    
    print("FINISHED")
    