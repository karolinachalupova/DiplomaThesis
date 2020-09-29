"""
Loads data to memory
"""

import pandas as pd


class Anomalies():

    def __init__(self, 
            path_data='pipeline/data/01_anomalies_filtered/data.pkl', 
            path_meta='pipeline/data/00_anomalies_unfiltered/anomalies_meta.xlsx',
            nrows=None):

            # Load data, subset to nrows, set index
            data = pd.read_pickle(path_data)
            data = data.take(range(nrows if nrows is not None else len(data))) # subset to specified number of rows
            data = data.sort_index()

            # Get features and targets
            self.targets = data[["r"]]
            self.features = data.drop(["r", "FTID"], axis=1)
            
            # Load meta
            self.meta = pd.read_excel(path_meta, sheet_name=1)

            # check that meta corresponds one to one with features 
            assert set(self.meta.sc.tolist()).difference(set(self.features.columns.unique().tolist())) == set()
            assert set(self.features.columns.tolist()).difference(set(self.meta.sc.unique().tolist())) == set()

            # Sort the features on meta
            sorted_features = self.meta.sort_values(['class','class2']).sc.tolist()
            self.features = self.features[sorted_features]

    
    @property
    def classification1(self):
        classification1 = dict()
        for c in self.meta['class'].unique().tolist(): 
            classification1[c.lower()] = self.meta[self.meta['class'] == c].name.tolist()
        return classification1
            
    @property
    def classification2(self):
        classification2 = dict()
        for c in self.meta.class2.unique().tolist(): 
            classification2[c.lower()] = self.meta[self.meta['class2'] == c].name.tolist()
        return classification2
            
    @property 
    def sc_to_name(self):
        return dict(zip(self.meta.name_sc, self.meta.name))
    
    @property 
    def name_to_sc(self):
        return dict(zip(self.meta.name, self.meta.name_sc))
        


