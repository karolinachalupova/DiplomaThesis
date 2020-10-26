"""
Model performance evaluation
"""
import re
import numpy as np
import os
import pickle
import argparse
import tensorflow as tf
import pandas as pd

from ray import tune

from train_network import create_model, NetData, create_ensemble


class Nets():
    def __init__(self, nets, dataset):
        self.nets = nets
        self.dataset = dataset
    
    @classmethod
    def from_logs(cls, logs, dataset):
        logdirs = [os.path.join(logs, x) for x in os.listdir(logs)]
        finished = [os.path.isfile(os.path.join(logdir, "args.pickle")) for logdir in logdirs]
        logdirs = [logdir for f, logdir in zip(finished, logdirs) if f]  # use finished logdirs only 
        nets = [Net.from_logdir(logdir, dataset) for logdir in logdirs]
        return cls(nets, dataset)
    
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
    
    def create_ensembles(self, common_args=["hidden_layers", "ytrain", "yvalid", "ytest"]):
        """
        Groups self.nets into ensembles by grouping 
        together networks that have common arguments (`common_args`)
        Arguments: 
            common_args: list of args keys based on which to group.
        """
        # there are instances of Net in the "index" column
        groups = self.dataframe.groupby(common_args)["index"].apply(list)
        models = [create_ensemble([net.model for net in group]) for group in groups]
        args_list = [argparse.Namespace(**{key: value for key, value in vars(group[0].args).items() if key in common_args}) for group in groups]
        return [Net(model, args, self.dataset) for model, args in zip(models, args_list)]


class Net():
    def __init__(self, model, args, dataset):
        self.model = model
        self.args = args
        self.dataset = dataset
    
    @classmethod
    def from_logdir(cls, logdir, dataset):
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
        with open(os.path.join(logdir, "args.pickle"), "rb") as f: 
            args = argparse.Namespace(**pickle.load(f))  # loads dict and converts it to namespace
        analysis = tune.Analysis(os.path.join(logdir, "Training"))
        best_config = analysis.get_best_config(metric="valid_rmse", mode="min")
        best_logdir = analysis.get_best_logdir(metric="valid_rmse", mode="min")
        model = create_model(args=args, **best_config)
        checkpoint_folder_name = [s for s in os.listdir(best_logdir) if s.startswith("checkpoint")][0]
        model.load_weights(os.path.join(best_logdir, checkpoint_folder_name, "model.h5"))
        return cls(model, args, dataset)

    def __repr__(self):
        return "{}: {}".format(
            self.__class__, 
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in vars(self.args).items())))
    

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
    
    @property 
    def netdata(self):
        return NetData(
            ytrain=self.args.ytrain, 
            yvalid=self.args.yvalid, 
            ytest=self.args.ytest, 
            dataset=self.dataset)
    

    

    
