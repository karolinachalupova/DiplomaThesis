"""
Model performance evaluation
"""

import numpy as np
import os
import pickle
import argparse
import tensorflow as tf
import pandas as pd

from sklearn.metrics import r2_score
from ray import tune

from train_network import create_model, NetData, create_ensemble


class AModel():
    def evaluate(self, on="test", batch_size=10000000):
        netdata = self.netdata
        names = [self.model.loss] + [m.name for m in self.model.metrics] 
        if on=="test":
            values = self.model.evaluate(
                x=netdata.test.data["features"], 
                y=netdata.test.data["targets"],
                batch_size=batch_size)
        elif on=="valid": 
            values = self.model.evaluate(
                x=netdata.valid.data["features"], 
                y= netdata.valid.data["targets"],
                batch_size=batch_size)
        elif on=="train":
            values = self.model.evaluate(
                x=netdata.train.data["features"],
                y=netdata.train.data["targets"],
                batch_size=batch_size)
        return dict(zip(names, values))

class Networks():
    def __init__(self, logs:str, dataset):
        self.logs = logs 
        logdirs = [os.path.join(logs, x) for x in os.listdir(logs)]
        finished = [os.path.isfile(os.path.join(logdir, "args.pickle")) for logdir in logdirs]
        self.logdirs = [logdir for f, logdir in zip(finished, logdirs) if f]  # use finished logdirs only 
        self.nets = [Net(logdir, dataset) for logdir in self.logdirs]
    
    @property 
    def dataframe(self):
        """
        Returns a dataframe, where each net is a row and args are columns.
        There are instances of Net in the "index" column.
        """
        return pd.DataFrame(dict(zip(self.nets, [vars(net.args) for net in self.nets]))).transpose().reset_index()
    
    def evaluate_on_test(self):
        return pd.DataFrame(dict(zip(self.nets, [net.evaluate(on="test") for net in self.nets]))).transpose().reset_index()
    
    def evaluate_on_valid(self):
        return pd.DataFrame(dict(zip(self.nets, [net.evaluate(on="valid") for net in self.nets]))).transpose().reset_index()
    
    def create_ensembles(self, best_n:int=None, common_args=["hidden_layers", "ytrain", "yvalid", "ytest"]):
        """
        Groups self.nets into ensembles by grouping 
        together networks that have common arguments (`common_args`)
        Arguments: 
            common_args: list of args keys based on which to group.
        """
        # there are instances of Net in the "index" column
        groups = self.dataframe.groupby(common_args)["index"].apply(list)
        if best_n is not None:
            groups = [self.best_n(group,2) for group in groups]
        return [Ensemble(group) for group in groups]
    
    @staticmethod
    def best(models, metric="mse", mode="min"):
        """
        Returns `n` best models (best performance on validation set)
        n is chosen such that the ensemble of the models has best 
        performance on validation set
        """
        metrics = np.array([model.evaluate(on="valid").get(metric) for model in models])
        # models sorted in ascending order on metric
        sorted_models = [x for _,x in sorted(zip(metrics, models))]
        if mode == "min":
            # Taking models from the left side of the ascending list
            ensemble_metrics = np.array([Ensemble(sorted_models[:n]).evaluate(on="valid").get(metric) for n in range(2,len(models)+1)])
            return sorted_models[:np.argmin(ensemble_metrics)+2]
        elif mode =="max": 
            # Taking models from the right side of the ascending list
            ensemble_metrics = np.array([Ensemble(sorted_models[n:]).evaluate(on="valid").get(metric) for n in range(0,len(models)-1)])
            return sorted_models[np.argmax(ensemble_metrics):]
        else: 
            raise ValueError("Argument mode only takes values 'min' or 'max'")
    
    @staticmethod
    def best_n(models, n=2, metric="mse", mode="min"):
        """
        Returns `n` best models (best performance on validation set)
        n is chosen such that the ensemble of the models has best 
        performance on validation set
        """
        metrics = np.array([model.evaluate(on="valid").get(metric) for model in models])
        # models sorted in ascending order on metric
        sorted_models = [x for _,x in sorted(zip(metrics, models))]
        if mode == "min":
            # Taking models from the left side of the ascending list
            return sorted_models[:n]
        elif mode =="max": 
            # Taking models from the right side of the ascending list
            return sorted_models[-n:]
        else: 
            raise ValueError("Argument mode only takes values 'min' or 'max'")
    

class Ensemble(AModel): 
    def __init__(self, nets):
        self.nets = nets 
        # Check that all have the same yvalid ytest ytrain 
        assert len(set([network.args.ytrain for network in self.nets])) == 1, (
            "Networks in emsemble must have same args.ytrain")
        assert len(set([network.args.yvalid for network in self.nets])) == 1, (
            "Networks in emsemble must have same args.yvalid")
        assert len(set([network.args.ytest for network in self.nets])) == 1, (
            "Networks in emsemble must have same args.ytest")
    
        self.model = create_ensemble([network.model for network in self.nets])
    
    @property
    def netdata(self):
        return self.nets[0].netdata


class Net(AModel):
    def __init__(self, logdir:str, dataset):
        """
        Best tf.keras.model 
        (best: best performance on validation set out of all models in the logdir)
        used to choose optimum hyperparameters. 
        
        Arguments: 
            dataset: instance of data.Selected
            logdir(str): path to the logging folder of the model. 
                The logging folder of a trained model contains:
                1) folder "Training", witch contains logdirs 
                of individual hyperparameter searches
                2) pickle "args.pickle", which contains a dict of the network args. 
        """
        self.logdir = logdir
        self.dataset = dataset

        with open(os.path.join(self.logdir, "args.pickle"), "rb") as f: 
            self.args = argparse.Namespace(**pickle.load(f))  # loads dict and converts it to namespace

        self.analysis = tune.Analysis(os.path.join(self.logdir, "Training"))

        best_config = self.analysis.get_best_config(metric="valid_rmse", mode="min")
        best_logdir = self.analysis.get_best_logdir(metric="valid_rmse", mode="min")
        self.model = create_model(args=self.args, **best_config)
        checkpoint_folder_name = [s for s in os.listdir(best_logdir) if s.startswith("checkpoint")][0]
        self.model.load_weights(os.path.join(best_logdir, checkpoint_folder_name, "model.h5"))
    
    def __repr__(self):
        return "{}: hidden {}, seed {}, {}-{}-{}".format(
            self.__class__, self.args.hidden_layers, self.args.seed, 
            self.args.ytrain, self.args.yvalid, self.args.ytest)
    
    @property 
    def netdata(self):
        return NetData(
            ytrain=self.args.ytrain, 
            yvalid=self.args.yvalid, 
            ytest=self.args.ytest, 
            dataset=self.dataset)
    

    
