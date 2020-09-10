"""
Loads data to memory
"""

import pandas as pd

class Anomalies():

    def __init__(self, 
            path_data='data/01_anomalies_filtered/data.pkl', 
            path_meta='data/anomalies_meta.xlsx'):

            self.data = pd.read_pickle(path_data)
            self.data.reset_index(inplace=True)
            self.meta = pd.read_excel('data/00_anomalies_unfiltered/anomalies_meta.xlsx', sheet_name=1)

            # Sort the columns based on meta
            sorted_signals = self.meta.sort_values(['class','class2']).sc.tolist()
            columns = ['DTID', 'FTID', 'date', 'r'] + sorted_signals
            self.data = self.data[columns]
            
            # check that meta corresponds one to one with signals 
            assert set(self.meta.sc.tolist()).difference(set(self.data.columns.unique().tolist())) == set()
            assert set(self.data.columns.tolist()).difference(set(self.meta.sc.unique().tolist())) == {'DTID', 'FTID', 'date', 'r'}    
    
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
        


