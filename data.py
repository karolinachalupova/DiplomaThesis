"""
Preprocesses data from raw to cleaned, saving intermediately.
Also defines utilities for loading any of the intermediate and final datasets into memory.
"""

import pandas as pd
import numpy as np
import pickle
import scipy
import sklearn 
import os

directory = os.path.dirname(os.path.abspath(__file__))

def run_pipeline(nrow=None):
    """
    Starts with raw data and applies the following changes: 
        1) filters them, 
        2) subsets them (only does something if `nrow` is not `None`)
        3) cleans them.
    Saves all the intermediate datasets and 
    finally loads the cleaned data into memory.
    """
    filtered = Filtered()
    filtered.calculate()
    subset = Subset()
    subset.calculate(nrow=nrow)
    cleaned = Cleaned()
    cleaned.calculate()
    selected = Selected() 
    selected.calculate()
    return selected

if __name__ == "__main__":
    run_pipeline()


class Data():

    def load(self):
        """
        Loads the data from path into working memory.
        """
        self.features = pd.read_pickle(self.paths.get("features"))
        self.targets = pd.read_pickle(self.paths.get("targets"))
    
    def save(self):
        """
        Saves features and targets as pickles to paths. 
        """
        for dataset, path in zip(
                [self.features, self.targets], 
                [self.paths.get("features"), self.paths.get("targets")]): 
                    dataset.to_pickle(path)
        

class Raw():
    PATHS = {
        "filter": os.path.join(directory, "data/raw/DST_universe_filter.gzip"),
        "features":  os.path.join(directory, "data/raw/DST_signals.gzip"),
        "targets":  os.path.join(directory, "data/raw/DST_returns.gzip")
    }
    def __init__(self, paths = PATHS):
        self.paths = paths
    
    def load(self):
        self.universe_filter = pd.read_parquet(self.paths.get("filter"))
        self.features = pd.read_parquet(self.paths.get("features"))
        self.targets = pd.read_parquet(self.paths.get("targets"))


class Filtered(Data):
    PATHS = {
        "features":  os.path.join(directory, 'data/filtered/features.pkl'), 
        "targets":  os.path.join(directory, 'data/filtered/targets.pkl')
    } 
    def __init__(self, paths = PATHS):
        self.paths = paths
        
    def calculate(self, ancestor_paths = Raw.PATHS):
        """
        Recalculates the data using ancestors. 
        """
        ancestor = Raw(paths = ancestor_paths)
        ancestor.load()
        # Removes examples that are not present in universe filter.
        features = ancestor.features[ancestor.features['DTID'].isin(ancestor.universe_filter.DTID.unique().tolist())]
        targets = ancestor.targets[ancestor.targets['DTID'].isin(ancestor.universe_filter.DTID.unique().tolist())]

        for dataset, path in zip([features, targets], [self.paths.get("features"), self.paths.get("targets")]): 
            dataset.set_index(['DTID', 'date'], inplace=True)
            dataset.sort_index(inplace=True)
        
        features.drop("FTID", axis=1, inplace=True)

        # Removes examples without target or without features.
        self.features = features.loc[targets.index]
        self.targets = targets.loc[features.index]
        assert (self.features.index == self.targets.index).all()

        self.save()


class Subset(Data):
    PATHS = {
        "features":  os.path.join(directory, 'data/subset/features.pkl'),
        "targets":  os.path.join(directory, 'data/subset/targets.pkl')
    }
    def __init__(self, paths = PATHS):
        self.paths = paths 

    def calculate(self, ancestor_paths=Filtered.PATHS, nrow=None):
        ancestor = Filtered(paths=ancestor_paths)
        ancestor.load()
        nrow = nrow if nrow is not None else len(ancestor.features)
        self.features = ancestor.features.take(nrow)
        self.targets = ancestor.targets.take(nrow)
        self.save()


class Cleaned(Data):
    PATHS = {
        "features":  os.path.join(directory, 'data/cleaned/features.pkl'),
        "targets":  os.path.join(directory, 'data/cleaned/targets.pkl')
    }
    def __init__(self, paths= PATHS):
        self.paths = paths

    def calculate(self, ancestor_paths = Subset.PATHS, freq="Y"):
        ancestor = Subset(paths = ancestor_paths)
        ancestor.load()
        self.features = self.groupby_clean(ancestor.features, freq=freq)
        self.targets = ancestor.targets
        self.save()
    
    @staticmethod
    def replace_inf(df):
        return df.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def winsorize(df):
        t = scipy.stats.mstats.winsorize(df, axis=0, limits=(0.01,0.01))
        return pd.DataFrame(t, columns=df.columns, index=df.index)

    @staticmethod
    def center(df):
        """
        removes the mean of the data, columnwise
        """
        scaler = sklearn.preprocessing.StandardScaler(with_std=False)
        t = scaler.fit_transform(df)
        return pd.DataFrame(t, columns=df.columns, index=df.index)
        
    @staticmethod
    def normalize(df):
        max_abs_scaler = sklearn.preprocessing.MaxAbsScaler()
        t = max_abs_scaler.fit_transform(df)
        return pd.DataFrame(t, columns=df.columns, index=df.index)

    @staticmethod
    def impute_nan(df):
        imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        t = imputer.fit_transform(df)
        return pd.DataFrame(t, columns=df.columns, index=df.index)

    def clean(self, df):
        return self.impute_nan(self.normalize(self.center(self.winsorize(self.replace_inf(df)))))

    def groupby_clean(self, df, freq = "Y"):
        # Clean data using yearly cross-sections
        df = df.groupby(pd.Grouper(level=1, freq="Y")).apply(self.clean)
        return df.sort_index()


class Selected(Data):
    PATHS = {
            "features":  os.path.join(directory, 'data/selected/features.pkl'),
            "targets":  os.path.join(directory, 'data/selected/targets.pkl')
        }
    N_FEATURES = 30
    def __init__(self, paths= PATHS):
        self.paths = paths

    def calculate(self, ancestor_paths = Cleaned.PATHS):
        ancestor = Cleaned(paths = ancestor_paths)
        ancestor.load()

        meta = Meta()
        meta.load()

        # Select N_FEATURES most important features based on 
        # Tobek and Hronec, 2020, JFM: Does it pay to follow anomalies research? 
        selected_cols = meta.signals[meta.signals["important_otmh_global_liquid"]<=self.N_FEATURES].sc.tolist()
        assert len(selected_cols) == self.N_FEATURES, "Number of selected features does not match number of columns"

        self.features = ancestor.features[selected_cols]
        self.targets = ancestor.targets
        self.save()


class Meta():
    PATHS = {
        "all":  os.path.join(directory, 'data/meta_KCHnotes.xlsx')
    }
    def __init__(self, paths = PATHS):
        self.paths = paths
    
    def load(self):
        self.signals = pd.read_excel(self.paths.get("all"), sheet_name="signals")

    @property
    def classification1(self):
        classification1 = dict()
        for c in self.signals['class'].unique().tolist(): 
            classification1[c.lower()] = self.signals[self.signals['class'] == c].name.tolist()
        return classification1
            
    @property
    def classification2(self):
        classification2 = dict()
        for c in self.signals.class2.unique().tolist(): 
            classification2[c.lower()] = self.signals[self.signals['class2'] == c].name.tolist()
        return classification2
            
    @property 
    def sc_to_name(self):
        return dict(zip(self.signals.name_sc, self.signals.name))
    
    @property 
    def name_to_sc(self):
        return dict(zip(self.signals.name, self.signals.name_sc))
    
    @property 
    def sc_to_latex(self):
        return dict(zip(self.signals.name_sc, self.signals.name_tex))