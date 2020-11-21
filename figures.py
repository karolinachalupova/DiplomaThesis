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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import chunks, get_parts, get_orders
from data import Cleaned, Subset, Meta, FEATURES

meta = Meta()
meta.load()
mpl.use('pgf')
sns.set()


NN_DICT = {
        "-1":"LR",
        "32":"NN1",
        "32,16":"NN2",
        "32,16,8":"NN3",
        "32,16,8,4":"NN4",
        "32,16,8,4,2":"NN5"
        }
N_SEEDS = 9
SORTING = FEATURES
YTRAIN_NAMES = ["12","13","14", "15", "16"]


N_YTRAIN = len(YTRAIN_NAMES)
HIDDEN_LAYERS = list(NN_DICT.keys())
NN_NAMES = list(NN_DICT.values())
SEED_NAMES = [str(i) for i in list(range(1,N_SEEDS+1))]
SORTING_LATEX = [meta.sc_to_latex.get(s) for s in SORTING]


def plot_dummy(p=None):
    plt.plot([1, 2, 3, 4])
    plt.ylabel('Some Numbers')
    fig = LatexFigure(plt.gcf())
    fig.fit(square=True)
    if p is not None: 
        fig.save(p)

#==========================================================================================
#                                     On Meta
#==========================================================================================
def tabulate_meta(p=None):
    df = meta.signals[~meta.signals.important_otmh_global_liquid.isna()]
    df = df[["name_tex", "class", "class2", "tex_cite", "journal", "freq"]]
    df.set_index('name_tex', inplace=True)
    df.index.name = "Feature"
    df = df.loc[SORTING_LATEX]

    class_dict = {"frictions":"Market", "fund":"Accounting","IBES":"Analyst Forecasts"}
    class2_dict = {"other":"Other"}
    freq_dict = {"monthly":"M", "annual_july":"Y"}
    tex_cite_dict = {old: "\cite{" + old + "}" for old in list(df.tex_cite.values)}
    df.replace({"class": class_dict, "freq":freq_dict, "class2":class2_dict, "tex_cite":tex_cite_dict}, inplace=True)

    cdict = {
        'name_tex':"Feature", 
        "tex_cite":"Author", 
        "journal":"Journal", 
        "freq":"Frequency", 
        "class":"Category", 
        "class2":"Subcategory"}
    df = df.rename(columns = cdict)

    tab = df.to_latex(escape=False)
    if p is not None: 
        with open(p,'w') as tf:
            tf.write(tab)
    return df


#==========================================================================================
#                                     On not cleaned data
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
    if p is not None: 
        fig.save(p)

#==========================================================================================
#                                     On cleaned data 
#==========================================================================================
def plot_histograms(df, p=None):
    df.columns = [meta.sc_to_latex.get(s) for s in df.columns.tolist()]
    df.hist(sharex=True)
    fig = LatexFigure(plt.gcf())
    fig.fit(scale=5)
    if p is not None: 
        fig.save(p)

def plot_correlation_matrix(df, p=None):
    corr = df.corr()
    corr.rename(index=meta.sc_to_latex, inplace=True)
    corr.columns = corr.index.values
    _corrplot(corr, size_scale=30, legend=True)
    fig = LatexFigure(plt.gcf())
    fig.fit(square=True)
    if p is not None: 
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
    if p is not None: 
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
    if p is not None: 
        fig.save(p)
    
def tabulate_correlation_matrix(df, p=None):
    corr = df.corr()
    corr.rename(index=meta.sc_to_latex, inplace=True)
    corr.columns = corr.index.values
    corr = corr.round(3)
    tab = corr.to_latex()

    # Rotate header 
    break_one, break_two = "\\toprule\n{} &", "\\\\\n\\midrule"
    first, second = tab.split(break_one)
    second, third = second.split(break_two)
    second = "&".join(["\\rot{" +s + "}" for s in second.split("&")])
    tab = first +  break_one + second + break_two + third
    
    if p is not None: 
        with open(p,'w') as tf:
            tf.write(tab)
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
    tab = df.to_latex()
    if p is not None: 
        with open(p,'w') as tf:
            tf.write(tab)
    return df 

def tabulate_descriptives(df, p=None):
    df = df.describe().transpose().round(4)
    df.rename(index=meta.sc_to_latex, inplace=True)
    df = df[df.columns.tolist()[1:]] # Omit count
    df.columns = ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    if p is not None: 
        with open(p,'w') as tf:
            tf.write(df.to_latex())
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
    if p is not None:
        fig.save(p)





class LocalIG():
    def __init__(self, path_to_models):
        self.path_to_models = path_to_models 
    
    def load(self, model_name, sort_features=True, suffix="_test"):
        df = pd.read_csv(os.path.join(self.path_to_models, model_name, "integrated_gradients{}.csv".format(suffix)), index_col=[0,1])
        if sort_features: 
            df = df[SORTING]
        self.df = df
    
    def plot(self, nobs=100):
        df = self.df.iloc[:nobs]
        plot_all_observations(df, xlabel="Integrated Gradient")


def plot_all_observations(df, xlabel):
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
    plt.show()


class Results():
    def __init__(self, path):
        """
        """
        self.path = path
    
    def load(self, sort_features = True, suffix="_test"):
        """
        Models are in columns, features are in rows
        """
        self.ar = pd.read_csv(os.path.join(self.path, "args.csv"), index_col=0)
        self.pe = pd.read_csv(os.path.join(self.path, "performance.csv"), index_col=0)
        self.ig = pd.read_csv(os.path.join(self.path, "integrated_gradients_global{}.csv".format(suffix)), index_col=0)
        self.mr = pd.read_csv(os.path.join(self.path, "model_reliance{}.csv".format(suffix)), index_col=0)

        self.ar["nn_name"] = self.ar[["hidden_layers"]].replace(NN_DICT)
        self.ar["nn_name_short"] = [s[-1:] for s in self.ar.nn_name]

        if sort_features: 
            self.ig = self.ig[SORTING]
            self.mr = self.mr[SORTING]
    
    def subset(self, key_name, value):
        sub = (self.ar[key_name] == value) 
        self.ar = self.ar[sub]
        self.pe = self.pe[sub]
        self.ig = self.ig[sub]
        self.mr = self.mr[sub]
    
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
    
    def calculate_seed_corr(self, df):
        """
        Returns correlation of seeds
        """
        corr = dict()
        for ytrain in list(self.ar.ytrain.values.unique()):
            corr["{}".format(ytrain)] = dict()
            for hidden_layers in HIDDEN_LAYERS:
                sub = (self.ar["hidden_layers"]==hidden_layers) & (self.ar["ytrain"]==ytrain)
                corr["{}".format(ytrain)]["{}".format(hidden_layers)] = df[sub].transpose().corr().dropna(how="all").dropna(how="all", axis=1).values.mean()
        return corr


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
    plt.show()

class Styling(): 
    def __init__(self, transform, cmap, show_cbar, flip_cbar): 
        self.cmap = cmap
        self.show_cbar = show_cbar 
        self.flip_cbar = flip_cbar

        transforms = {
            'heatmap': self.heatmap,
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

def style_plot_df(df, styling, mode, vmin=None):
    stylings = {
        'heatmap':Styling("heatmap", plt.cm.inferno_r, show_cbar=True, flip_cbar=True),
        'top':Styling("top10", plt.cm.Blues, show_cbar=False, flip_cbar=False),
        'bottom':Styling("bottom10", plt.cm.Blues, show_cbar=False, flip_cbar=False),
        'top10':Styling("top10", plt.cm.Blues, show_cbar=False, flip_cbar=False),
        'bottom10':Styling("bottom10", plt.cm.Blues, show_cbar=False, flip_cbar=False),
        'top5':Styling("top5", plt.cm.Blues, show_cbar=False, flip_cbar=False),
        'bottom5':Styling("bottom5", plt.cm.Blues, show_cbar=False, flip_cbar=False),
        'blues':Styling("identity", plt.cm.Blues, show_cbar=True, flip_cbar=False)
    }
    modes = {
        'ensemble': Mode([["LR"],['NN1', 'NN2', 'NN3', 'NN4', 'NN5']], False),
        'seeds': Mode(list(chunks(["{}-{}".format(n,s) for n,s in itertools.product(NN_NAMES,SEED_NAMES)],N_SEEDS)),True),
        'ensemble_time': Mode(list(chunks(["{}-{}".format(n,s) for n,s in itertools.product(NN_NAMES,YTRAIN_NAMES)],N_YTRAIN)),True)
    }
    if type(styling) == str: 
        styling = stylings.get(styling)
    if type(mode) == str: 
        mode = modes.get(mode)
    
    df = styling.transform(df)
    plot_df(
        df, column_groups=mode.column_groups, group_xticks=mode.group_xticks, 
        cmap=styling.cmap, flip_cbar=styling.flip_cbar, show_cbar=styling.show_cbar, 
        vmin=vmin)

def plot_df(
        df, column_groups:list=None, cmap=plt.cm.Blues, 
        flip_cbar:bool=False, show_cbar:bool=True, 
        scale:int=2, vmin=None, group_xticks=False):
    
    # Create as many column subplots as there are column groups
    if column_groups == None: 
        column_groups = [df.columns.tolist()]
        
    fig, axes = plt.subplots(nrows=1, ncols=len(column_groups), figsize=(6.5,7),
                         gridspec_kw={"width_ratios":[len(group) for group in column_groups]})
    # fix axes if there is only one columns group from integer to list
    if len(column_groups) == 1: 
        axes = np.array(axes)
    
    # Populate all axes
    vmin = df.min().min() if vmin is None else vmin
    kw = {'cmap':cmap, 'vmin':vmin, 'vmax':df.max().max()}
    for i, group in enumerate(column_groups): 
        im = axes[i].pcolor(df[group], **kw)
    
    # Y labels for first group are the index of the df
    labels = [meta.sc_to_latex.get(label) for label in df.index.values.tolist()]
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
    plt.show()


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
    
    def save(self, path:str):
        self.fig.savefig(path, bbox_inches='tight')
        print("Figure saved to {}".format(path))
    

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
            "pgf.preamble": [
                r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
                r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
                ]
            }
        mpl.rcParams.update(pgf_with_latex)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_figures", default="latex/Figures", type=str, help="Folder where to save the figures.")
    parser.add_argument("--path_tables", default="latex/Tables", type=str, help="Folder where to save the figures.")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    mpl.use('pgf')
    sns.set()

    plot_dummy(p=os.path.join(args.path_figures,"dummy.pdf"))

    
    #==========================================================================================
    #                                     On Meta data
    #==========================================================================================
    tabulate_meta(p=os.path.join(args.path_tables,"meta.tex"))
    
    #==========================================================================================
    #                                     On not cleaned data 
    #==========================================================================================
    dt = Subset()
    dt.load()
    df = dt.features[SORTING]
    #plot_missing_observations(df, p=os.path.join(args.path_figures,"missing_observations.pdf"))

    #==========================================================================================
    #                                     On cleaned data 
    #==========================================================================================
    dt = Cleaned()
    dt.load()
    df = dt.features[SORTING]
    r = dt.targets["r"]
    
    # Figures
    #plot_histograms(df, p=os.path.join(args.path_figures,"histograms.pdf"))
    #plot_correlation_matrix(df, p=os.path.join(args.path_figures,"correlation_matrix.pdf"))
    #plot_correlation_matrix_highest(df, p=os.path.join(args.path_figures,"correlation_matrix_highest.pdf"))
    #plot_standard_deviation(df,p=os.path.join(args.path_figures,"standard_deviation.pdf"))
    
    # Tables
    #tabulate_correlation_matrix(df, p=os.path.join(args.path_tables,"correlation_matrix.tex"))
    #tabulate_most_correlated_pairs(df, p=os.path.join(args.path_tables,"most_correlated_pairs.tex"))
    #tabulate_descriptives(df, p=os.path.join(args.path_tables,"descriptives.tex"))

    # Figures of Returns 
    plot_returns_histogram(r, p=os.path.join(args.path_figures,"returns_histogram.pdf"))
    


