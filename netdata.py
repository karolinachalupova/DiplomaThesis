"""
Defines how data is split into training, validation and test sets. 
"""
import numpy as np
import pandas as pd

from data import Cleaned


class NetData():

    class Dataset():
        def __init__(self, data:dict, shuffle_batches:bool, seed:int=42, subset_percent:float=None):
            """
            Args: 
                data (dict): dictionary with keys "features" and "targets", each containing a np.array.
                shuffle_batches (bool): Bool indicating whether or not the batches should be shuffled. 
                seed (int): random seed used for random shuffling of batches 
                    (and, if subset_percent is not None, for random subsetting)
                subset_percent (float): float between 0 and 1, percentage of data to use. 
                    If not None, randomly drops 1-subset_percent of data. 
                    Use only during experimentation if you want to avoid training (validating, testing)
                    on too big data, otherwise keep the default `None` to preserve entire dataframe.
            """
            if subset_percent is not None:
                # Randomly drop `subset_percent` percent of rows from both features and targets
                
                num_rows_to_remove = round(len(data["features"])*subset_percent)
                np.random.seed(seed)
                drop_idx = np.random.choice(data["features"].index, num_rows_to_remove, replace=False)
                data = {
                    "features": data["features"].drop(drop_idx), 
                    "targets": data["targets"].drop(drop_idx)
                }

            self._data = data
            self._size = len(self._data["features"])
            
            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None 
    
        @property 
        def data(self):
            return self._data
        
        @property 
        def size(self):
            return self._size 
        
        def batches(self, size=None):
            """
            Splits data into batches. 
            Additonally, if `shuffle_batches` is True, shuffles data within each batch.
            """
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key in self._data:
                    batch[key] = self._data[key][batch_perm]
                yield batch

    def __init__(self, ytrain:int=9, yvalid:int=6, ytest:int=1):
        """
        Splits data into train, valid, and test sets.
        
        Following Gu et al. (2018), I split the data 
        by taking first `ytrain` years as train set, 
        the immediately following `yvalid` years as validation set, 
        and the `ytest` years after that as test set.

        Args: 
            ytrain (int): number of years in training set
            yvalid (int): number of years in validation set
            ytest (int): number of years in test set
        
        Examples: 
            >>> netdata = NetData()
            >>> netdata.train.data["features"], netdata.train.data["targets"]
            >>> netdata.valid.data["features"], netdata.valid.data["targets"]
            >>> netdata.test.data["features"], netdata.test.data["targets"]

        """
        self.ytrain = ytrain
        self.yvalid = yvalid
        self.ytest = ytest

        # Load entire dataset
        d = Cleaned()
        d.load()
        # d has attributes `targets` and `features`, each is a pd.DataFrame.

        # Create organizing masks 
        idx_year = d.targets.index.get_level_values('date').year
        splityear1 = idx_year.min() + self.ytrain
        splityear2 = splityear1 + self.yvalid
        splityear3 = splityear2 + self.ytest
        masks = {
            "train": (idx_year < splityear1),
            "valid": (idx_year >= splityear1) & (idx_year < splityear2),
            "test": (idx_year >= splityear2) & (idx_year < splityear3)
        }

        # Split data into train, valid and test
        for dataset in ["train", "valid", "test"]:
            data = {
                "features": np.asarray(d.features.loc[masks.get(dataset)]),
                "targets": np.asarray(d.targets.loc[masks.get(dataset)]),
            }
            setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset=="train"))