"""
Processing and loading of the data.

TODO: I'm not happy with replacing infty by np.nan in Clean.replace_inf, should be winsorization-like
"""

import pandas as pd
import pickle
from sklearn import preprocessing, impute
import numpy as np
import scipy


def run_pipeline(nrow=None):
    filtered = Filtered()
    filtered.calculate()
    subset = Subset()
    subset.calculate(nrow=nrow)
    cleaned = Cleaned()
    cleaned.calculate()
    cleaned.load()
    return cleaned

if __name__ == "__main__":
    run_pipeline()

class Dataset():

    def load(self):
        """
        Loads the data from path into working memory.
        """
        self.features = pd.read_pickle(self.paths.get("features"))
        self.targets = pd.read_pickle(self.paths.get("targets"))
        

class Meta(Dataset):
    PATHS = {
        "all": 'data/meta.xlsx'
    }
    def __init__(self, paths = PATHS):
        self.paths = paths
    
    def load(self):
        self.sheet0 = pd.read_excel(self.paths.get("all"), sheet_name=0)

    @property
    def classification1(self):
        classification1 = dict()
        for c in self.sheet0['class'].unique().tolist(): 
            classification1[c.lower()] = self.sheet0[self.sheet0['class'] == c].name.tolist()
        return classification1
            
    @property
    def classification2(self):
        classification2 = dict()
        for c in self.sheet0.class2.unique().tolist(): 
            classification2[c.lower()] = self.sheet0[self.sheet0['class2'] == c].name.tolist()
        return classification2
            
    @property 
    def sc_to_name(self):
        return dict(zip(self.sheet0.name_sc, self.sheet0.name))
    
    @property 
    def name_to_sc(self):
        return dict(zip(self.sheet0.name, self.sheet0.name_sc))


class Raw(Dataset):
    PATHS = {
        "filter": "data/raw/DST_universe_filter.gzip",
        "features": "data/raw/DST_signals.gzip",
        "targets": "data/raw/DST_returns.gzip"
    }
    def __init__(self, paths = PATHS):
        self.paths = paths
    
    def load(self):
        self.universe_filter = pd.read_parquet(self.paths.get("filter"))
        self.features = pd.read_parquet(self.paths.get("features"))
        self.targets = pd.read_parquet(self.paths.get("targets"))


class Filtered(Dataset):
    PATHS = {
        "features":'data/filtered/features.pkl', 
        "targets": 'data/filtered/targets.pkl'
    }
    def __init__(self, paths = PATHS):
        self.paths = paths
        
    def calculate(self, ancestor_paths = Raw.PATHS):
        """
        Removes examples that do not meat specified criteria.
        """
        ancestor = Raw(paths = ancestor_paths)
        ancestor.load()
        # Removes examples that are not present in universe filter.
        features = ancestor.features[ancestor.features['DTID'].isin(ancestor.universe_filter.DTID.unique().tolist())]
        targets = ancestor.targets[ancestor.targets['DTID'].isin(ancestor.universe_filter.DTID.unique().tolist())]

        # Removes examples where target is NaN.
        targets = targets[targets.r.isna()==False]

        # Set index and sort.
        for dataset in [features, targets]:
            dataset.set_index(['DTID', 'date'], inplace=True)
            dataset.sort_index(inplace=True)
        
        # Removes examples without target or without features.
        features = features.loc[targets.index]
        targets = targets.loc[features.index]
        assert features.index == targets.index

        features.drop("FTID", axis=1, inplace=True)

        for dataset, path in zip([features, targets], [self.paths.get("features"), self.paths.get("targets")]): 
            dataset.to_pickle(path)


class Subset(Dataset):
    PATHS = {
        "features": 'data/subset/features.pkl',
        "targets": 'data/subset/targets.pkl'
    }
    def __init__(self, paths = PATHS):
        self.paths = paths 

    def calculate(self, ancestor_paths=Filtered.PATHS, nrow=None):
        ancestor = Filtered(paths=ancestor_paths)
        ancestor.load()
        features = ancestor.features.head(nrow)
        targets = ancestor.targets.head(nrow)
        for dataset, path in zip([features, targets], [self.paths.get("features"), self.paths.get("targets")]): 
            dataset.to_pickle(path)


class Cleaned(Dataset):
    PATHS = {
        "features":'data/cleaned/features.pkl',
        "targets": 'data/cleaned/targets.pkl'
    }
    def __init__(self, paths= PATHS):
        self.paths = paths

    def calculate(self, ancestor_paths = Subset.PATHS, freq="Y"):
        ancestor = Subset(paths = ancestor_paths)
        ancestor.load()
        features = self.groupby_clean(ancestor.features, freq=freq)
        targets = ancestor.targets
        for dataset, path in zip([features, targets], [self.paths.get("features"), self.paths.get("targets")]): 
            dataset.to_pickle(path)
    
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
        scaler = preprocessing.StandardScaler(with_std=False)
        t = scaler.fit_transform(df)
        return pd.DataFrame(t, columns=df.columns, index=df.index)
        
    @staticmethod
    def normalize(df):
        max_abs_scaler = preprocessing.MaxAbsScaler()
        t = max_abs_scaler.fit_transform(df)
        return pd.DataFrame(t, columns=df.columns, index=df.index)

    @staticmethod
    def impute_nan(df):
        imputer = impute.SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        t = imputer.fit_transform(df)
        return pd.DataFrame(t, columns=df.columns, index=df.index)

    def clean(self, df):
        return self.impute_nan(self.normalize(self.center(self.winsorize(self.replace_inf(df)))))

    def groupby_clean(self, df, freq = "Y"):
        # Clean data using yearly cross-sections
        df = df.groupby(pd.Grouper(level=1, freq="Y")).apply(self.clean)
        return df.sort_index()


