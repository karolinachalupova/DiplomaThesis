"""
Tools for making matplotlib figures latex-friendly. 
"""
import matplotlib as mpl
import numpy as np
mpl.use('pgf')


class Figure():
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