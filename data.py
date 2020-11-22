"""
Preprocesses data from raw to cleaned, saving intermediately.
Also defines utilities for loading any of the intermediate and final datasets into memory.
"""

import pandas as pd
import numpy as np
import pickle
import scipy
from sklearn import preprocessing, impute
import os
import argparse
from scipy.stats import rankdata
import random
from datetime import date 

directory = os.path.dirname(os.path.abspath(__file__))


# Select 30 most important features based on liquid universe in
# Tobek and Hronec, 2020, JFM: Does it pay to follow anomalies research? 
FEATURES = [
    '52WH', 'STR', 'IdioRisk', 'VolMV', 'CVoST', 'Max', 'WWI',
    'Coskew', 'OPoA', 'MomLag', 'LB5', 'RDoMV', 'Seas6t10A',
    'Seas11t15N', 'Seas2t5N', 'MomRev', 'Amihud', 'NOA', 'Seas6t10N', 
    'Seas', 'Seas2t5A', 'Accr', 'DurE', 'dCE', 'PM', 'LB3', 'LiqShck',
    'LCoBP', 'EPred','EFoP']

FEATURES_OLD = ['52WH', 'STR', 'IdioRisk', 'VolTrend', 'CVoST', 'Max', 'WWI',
    'Coskew', 'OPoA', 'MomLag', 'LB5', 'RDoMV', 'Seas6t10A',
    'Seas11t15N', 'Seas2t5N', 'MomRev', 'Amihud', 'NOA', 'Seas6t10N', 
    'Seas', 'Seas2t5A', 'Accr', 'DurE', 'dCE', 'PM', 'LB3', 'LiqShck',
    'LCoBP', 'EPred','EFoP']

N_FEATURES = len(FEATURES)

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
        p = self.paths.get("features")
        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))
        p = self.paths.get("targets")
        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))
        print("Saving to {} and {}".format(self.paths.get("features"), self.paths.get("targets")))
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
        self.features = ancestor.features[FEATURES]
        self.targets = ancestor.targets
        self.save()


class Cleaned(Data):
    PATHS = {
        "features":  os.path.join(directory, 'data/cleaned/features.pkl'),
        "targets":  os.path.join(directory, 'data/cleaned/targets.pkl')
    }
    def __init__(self, paths= PATHS):
        self.paths = paths

    def calculate(self, ancestor_paths = Subset.PATHS):
        ancestor = Subset(paths = ancestor_paths)
        ancestor.load()
        self.features = self.clean(ancestor.features)
        self.targets = ancestor.targets
        self.save()
    
    @staticmethod
    def replace_inf(df):
        return df.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def mask_outliers(df):
        # Replaces all outside 2 stds by nan 
        return df.mask(df.sub(df.mean()).div(df.std()).abs().gt(2))
    
    @staticmethod
    def winsorize(df):
        return df.apply(scipy.stats.mstats.winsorize, axis=0, limits=(0.01,0.05))

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
        df = self.replace_inf(df)
        df =  df.groupby(pd.Grouper(level=1, freq="Y")).apply(self.mask_outliers)
        df = self.impute_nan(df)
        df = df.groupby(pd.Grouper(level=1, freq="Y")).apply(self.winsorize)
        df = df.groupby(pd.Grouper(level=1, freq="Y")).apply(self.center)
        df = df.groupby(pd.Grouper(level=1, freq="Y")).apply(self.normalize)
        return df


class Simulated(Data):
    PATHS = {
            "features":  os.path.join(directory, 'data/simulated/features.pkl'),
            "targets":  os.path.join(directory, 'data/simulated/targets.pkl')
        }
    def __init__(self, paths= PATHS):
        self.paths = paths

    def calculate(self, ancestor_paths = None, N=4600, T=190, Pc=N_FEATURES, save=True):
        print("Simulating data with N={}, T={}, Pc={}...".format(N, T, Pc))
        R, C = self.simulate_data(N=N, T=T, Pc=Pc)
        assert N == R.shape[0]
        assert T == R.shape[1]
        assert N == C.shape[0]
        assert T == C.shape[1]
        assert Pc == C.shape[2]

        # Create time index
        sdate = date(1990,1,1)
        edate = date(2018,12,31) 
        daterange = list(pd.date_range(sdate,edate,freq='m'))
        
        time_index = []
        for _ in range(N):
            s=random.sample(range(len(daterange)-T),1)[0]
            time_index = time_index + daterange[s:s+T]
        
        # Convert array R into  pd.DataFrame: targets
        N,T = R.shape
        out_array = np.column_stack((np.repeat(np.arange(N),T),R.reshape(N*T,-1)))
        targets = pd.DataFrame(out_array)
        targets.columns = ["DTID","r"]
        targets["date"] = time_index
        targets.set_index(["DTID", "date"],inplace=True)

        # Convert array C into pd.DataFrame: features
        
        N, T, Pc = C.shape
        out_arr = np.column_stack((np.repeat(np.arange(N),T),C.reshape(N*T,-1)))
        features = pd.DataFrame(out_arr)
        features.columns = ["DTID"]+["C{}".format(i) for i in range(1,Pc+1)]
        features["date"] = time_index
        features.set_index(["DTID", "date"],inplace=True)
        
        self.targets = targets
        self.features = features 
        if save: 
            self.save()
    
    @staticmethod
    def simulate_data(N, T, Pc):
        """
        Simulates matrix of returns R with shape (N,T) and matrix of features C with shape (N,T,Pc).
        
        The simulation follows excactly Internet Appendix A of Gu et al., 2018.
        """
        def crank(array):
            """
            Transforms array to its ranks. 
            Cross-Sectional Rank Function from Gu et al., 2018.  
            """
            b = np.empty(array.shape, dtype=int)
            for k, row in enumerate(array):
                b[k] = rankdata(-row, method='dense') - 1
            return b
        
        # Simulate factors V with shape T x 3
        V = np.random.normal(0,0.05**2, size=(T, 3)) 

        # Simulate all error terms 
        VAREPSILON = np.random.standard_t(5, size=(N,T))*0.05**2 # N x T
        U = np.random.normal(0,1-0.95**2, size=T) #T x 1
        RHO = np.random.uniform(0.9,1, size=Pc) #Pc x 1
        EPSILONS = np.zeros(shape=(N, T, Pc)) # N x T x Pc 
        for j in range(Pc):
            EPSILONS[:,:,j] = np.random.normal(0,1-RHO[j]**2, size=(N,T))

        # Simulate timeseries X with shape T x 1
        X = np.zeros(shape=(T)) 
        for t in range(1, len(X)):
            X[t] = 0.95*X[t-1] + U[t]

        # Simulate characteristics C with shape N x T x Pc 
        Cbar = np.zeros(shape=(N, T, Pc))
        for j in range(Pc):
            for t in range(1,T):
                Cbar[:,t,j] = RHO[j]*Cbar[:,t-1,j]  + EPSILONS[:,t,j]
        C = np.zeros(shape=(N, T, Pc)) # N x T x Pc         
        for i in range(N):        
            C[i,:,:] = 2/(N+1) * crank(Cbar[i,:,:]) - 1 

        # Simulate features G with shape N x T
        THETA = np.array([0.04,0.03,0.012])
        G = np.zeros(shape=(N,T))
        for i in range(N):
            for t in range(T):
                CH = np.array([
                    C[i,t,1]**2,
                    C[i,t,1]*C[i,t,2],
                    np.sign(C[i,t,3]*X[t])])
                G[i,t] = np.dot(CH,THETA)

        # Simulate noise E with shape N x T 
        E = np.zeros(shape=(N,T))
        for i in range(N):
            for t in range(1,T):
                E[i,t] = np.dot(np.array([C[i,t-1,1],C[i,t-1,2],C[i,t-1,3]]),V[t]) + VAREPSILON[i,t]

        # Simulate returns R with shape N x T
        R = np.zeros(shape=(N,T))
        for i in range(N):
            for t in range(1,T):
                R[i,t] = G[i,t-1] + E[i,t]

        return R, C


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calculate", default="cleaned", type=str, help="Which dataset to (re)calculate.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    name_map = {
        "filtered": Filtered,
        "subset": Subset, 
        "cleaned": Cleaned,
        "simulated": Simulated 
    }

    if args.calculate == "all":
        filtered = Filtered()
        filtered.calculate()
        subset = Subset()
        subset.calculate()
        cleaned = Cleaned()
        cleaned.calculate()
    else: 
        C = name_map.get(args.calculate)
        c = C()
        c.calculate()
