"""
Defines calculations that can be performed on trained networks. 
"""
import re
import itertools
import json
import numpy as np
import os
import pickle
import argparse
import tensorflow as tf
import pandas as pd
import warnings
import pandas as pd

from alibi.explainers import IntegratedGradients

from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from train_network import RSquare

from train_network import create_model, NetData, create_ensemble


class PermutationImportanceMeasure():

   
    @staticmethod
    def _permute(x, y, index):
        # Fisher eq. 3.4,3.5
        adjustment = 1
        n = x.shape[0]
        if n % 2 == 1: 
            x = x[0:-1,:]
            y = y[0:-1]
            adjustment = n * (1/(2*((n-1)/2)))
        feature = x[:,index]
        halves = np.split(feature,2)
        perturbed = np.concatenate((halves[1], halves[0]))
        x = np.concatenate((x[:,0:index], perturbed[:,None], x[:,index+1:]),axis=1)
        return x, y, adjustment


class PortfolioReliance(PermutationImportanceMeasure):
    def __init__(self, percent_long:int, percent_short:int):
        self.percent_long = percent_long 
        self.percent_short = percent_short 
    
    @staticmethod 
    def _drop_last_observation_if_odd(x, y, df_index):
        n = x.shape[0]
        if n % 2 == 1: 
            x = x[0:-1,:]
            y = y[0:-1]
            df_index = df_index[0:-1]
        return x, y, df_index
    
    @staticmethod
    def _get_p_largest(df, percent):
        n = int(0.01* percent * len(df))
        return df.nlargest(n, columns="prediction")
    
    @staticmethod
    def _get_p_smallest(df, percent):
        n = int(0.01* percent * len(df))
        return df.nsmallest(n, columns="prediction")

    def _get_longshort_mean_return(self, f, x, y, df_index):
        predicted_return = f.predict(x)
        actual_return = y
        df = pd.DataFrame(actual_return, columns=["actual"], index=df_index)
        df["prediction"] = predicted_return
        srt = df.groupby(pd.Grouper(level=1, freq="M")).apply(self._get_p_smallest, percent=self.percent_short)
        lng = df.groupby(pd.Grouper(level=1, freq="M")).apply(self._get_p_largest, percent=self.percent_short)
        long_mean = lng.actual.mean()
        short_mean = srt.actual.mean()
        return long_mean - short_mean

    def _r_orig(self, f, x, y, df_index):
        return self._get_longshort_mean_return(f, x, y, df_index)
    
    def _r_div(self, f, x, y, index, df_index):
        x, y, df_index = self._drop_last_observation_if_odd(x, y, df_index)
        x, y, _ = self._permute(x, y, index)
        return self._get_longshort_mean_return(f, x, y, df_index)
    
    def fit_single_feature(self, f, x, y, index, df_index):
        """
        Calculates model reliance on a single feature (at position `index`.)

        Arguments: 
            f: tf.keras.Model instance
            x (np.array): inputs to the model
            y (np.array): true labels of the model
        """
        r_orig = self._r_orig(f, x, y, df_index)
        r_div = self._r_div(f, x, y, index, df_index)
        return r_orig - r_div  ## If the difference between orig and div is large, the feature is important
    
    def fit(self, f, x, y, df_index):
        return np.array([self.fit_single_feature(f, x, y, index, df_index) for index in range(x.shape[1])])


class ModelReliance(PermutationImportanceMeasure):
    def __init__(self, loss):
        """
        Arguments: 
            loss: a tf.keras.losses object (class, not instance)
        """
        self.loss = loss()
    
    def _e_orig(self, f, x, y):
        # Fisher eq. 3.2
        return self.loss(y,f.predict(x)).numpy()
    
    def _e_div(self, f, x, y, index):
        # Fisher eq. 3.4,3.5
        x, y, adjustment = self._permute(x, y, index)
        return adjustment * self.loss(y, f.predict(x)).numpy()
    
    def fit_single_feature(self, f, x, y, index):
        """
        Calculates model reliance on a single feature (at position `index`.)

        Arguments: 
            f: tf.keras.Model instance
            x (np.array): inputs to the model
            y (np.array): true labels of the model
        """
        e_orig = self._e_orig(f, x, y)
        e_div = self._e_div(f, x, y, index)
        return e_div/e_orig
    
    def fit(self, f, x, y):
        return np.array([self.fit_single_feature(f, x, y, index) for index in range(x.shape[1])])


class Nets():
    def __init__(self, nets):
        self.nets = nets
        # Check that elements in nets are instances of Net
        if not np.array([isinstance(net, Net) for net in self.nets]).all():
            raise ValueError("nets must be a list of instances of Net.")
    
    @classmethod 
    def from_saved(cls, path_to_models):
        dirs = [os.path.join(path_to_models, x) for x in os.listdir(path_to_models)]
        nets = [Net.from_saved(d) for d in dirs]
        return cls(nets)
    
    @property 
    def dataframe(self):
        """
        Returns a dataframe, where each net is a row and args are columns.
        There are instances of Net in the "index" column.
        """
        return pd.DataFrame(dict(zip(self.nets, [vars(net.args) for net in self.nets]))).transpose()
    
    def decile_rsquare(self):
        rsqrs = [net.decile_rsquare() for net in self.nets]
        df = pd.DataFrame(rsqrs)
        return df.median(axis=0)
        
    def performance(self):
        return pd.concat([net.performance() for net in self.nets])
    
    def decile_performance(self):
        return pd.concat([net.decile_performance() for net in self.nets])
    
    def model_reliance(self, on="test"):
        return pd.concat([net.model_reliance(on=on) for net in self.nets])
    
    def portfolio_reliance(self,  percent_long, percent_short):
        return pd.concat([net.portfolio_reliance(percent_long=percent_long, percent_short=percent_short) for net in self.nets])
    
    def integrated_gradients_global(self, on="test"):
        return pd.concat([net.integrated_gradients(on=on)[1] for net in self.nets])
    
    def create_ensembles(self, common_args=["hidden_layers", "ytrain", "yvalid", "ytest"]):
        """
        Groups self.nets into ensembles by grouping 
        together networks that have common arguments (`common_args`)
        Arguments: 
            common_args: list of args keys based on which to group.
        """
        # there are instances of Net in the "index" column
        df = self.dataframe.reset_index()
        groups = df.groupby(common_args)["index"].apply(list)
        models = [create_ensemble([net.model for net in group]) for group in groups]
        args_list = [argparse.Namespace(**{key: value for key, value in vars(group[0].args).items() if key in common_args}) for group in groups]
        return [Net(model, args) for model, args in zip(models, args_list)]
    
    def get_missing(self, 
                wanted_ytrain = [12,13,14,15,16], 
                wanted_hidden_layers = ["-1","16", "16,8", "16,8,4", "16,8,4,2"], 
                wanted_seeds = [1,2,3,4,5,6,7,8,9,10]):
        finished = self.dataframe.groupby(["ytrain", "hidden_layers"]).seed.apply(list)
        missing = dict()
        for ytrain in wanted_ytrain:
            try: 
                finished.loc[ytrain]
                for hidden_layers in wanted_hidden_layers: 
                    try: 
                        finished.loc[ytrain][hidden_layers]
                        for hidden_layers in wanted_hidden_layers:
                            missing_seeds = list(set(wanted_seeds).difference(set(finished.loc[ytrain][hidden_layers])))
                            if len(missing_seeds) > 0:
                                # Record cases where ytrain and hidden_layers is not missing, 
                                # but there are some missing seeds
                                missing["{}_{}".format(ytrain, hidden_layers)] = missing_seeds
                    except KeyError: 
                        # Record cases where there is ytrain, but 
                        # but some hidden layers are missing for that ytrain
                        missing["{}_{}".format(ytrain, hidden_layers)] = wanted_seeds
            except KeyError:
                # Record cases where whole ytrain is missing
                for hidden_layers in wanted_hidden_layers: 
                    missing["{}_{}".format(ytrain, hidden_layers)] = wanted_seeds
        return missing

class Net():
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.args.n_models = self.n_models
        # I know this is ugly, but I added the sgd arg only later so older networks 
        # do not have args.optimizer 
        try: 
            self.args.optimizer = args.optimizer 
        except AttributeError: 
            self.args.optimizer = "adam"
       
    
    @classmethod
    def from_saved(cls, folder):
        with open(os.path.join(folder, "args.pickle"), "rb") as f: 
            args = argparse.Namespace(**pickle.load(f))  # loads dict and converts it to namespace
        with open(os.path.join(folder,'model.json')) as f:
            json_string = json.load(f)
        model = tf.keras.models.model_from_json(json_string, custom_objects=None)
        model.load_weights(os.path.join(folder, 'weights.h5'))

        model.compile(
            loss ='mse',
            metrics = [RootMeanSquaredError(), MeanAbsoluteError(), RSquare()]
        )
        return cls(model, args)


    def __repr__(self):
        return "{}: {}".format(
            self.__class__, 
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in vars(self.args).items())))
    
    def set_dataset(self, dataset, ytest):
        self.dataset = dataset
        self.args.ytest = ytest
    
    def performance(self, batch_size=10000000):
        netdata = self.netdata
        names = [self.model.loss] + [m.name for m in self.model.metrics]
        train_perf = list(self.model.evaluate(
                x=netdata.train.data["features"],
                y=netdata.train.data["targets"],
                batch_size=batch_size))
        valid_perf = list(self.model.evaluate(
                x=netdata.valid.data["features"], 
                y= netdata.valid.data["targets"],
                batch_size=batch_size))
        test_perf = list(self.model.evaluate(
                x=netdata.test.data["features"], 
                y=netdata.test.data["targets"],
                batch_size=batch_size))

        return pd.DataFrame(
            train_perf+valid_perf+test_perf, 
            index = [s+"_"+n for s, n in itertools.product(["train", "valid", "test"],names)],
            columns=[self.__repr__()]).transpose()
    
    def decile_performance(self, batch_size=100000000):
        y_pred = pd.Series(self.model.predict(x=self.netdata.test.data["features"]).flatten())
        decile_names = [str(i) for i in range(10,110,10)]
        deciles = pd.qcut(y_pred,q=10, labels=decile_names)
        performances = []
        for s in decile_names: 
            features = np.array(pd.DataFrame(self.netdata.test.data["features"]).loc[deciles==s])
            targets = np.array(pd.DataFrame(self.netdata.test.data["targets"]).loc[deciles==s])
            mse, rmse, mae, r = self.model.evaluate(x=features, y=targets, batch_size=batch_size)
            m = targets.mean()
            mse_r, rmse_r, mae_r = mse/np.abs(m), rmse/np.abs(m), mae/np.abs(m)
            performances.append([mse, mse_r, rmse, rmse_r, mae, mae_r, r, m])
        performances_flat_list = [item for sublist in performances for item in sublist]
        names = ["mse", "mse_relative","rmse", "rmse_relative", "mae", "mae_relative", "r2", "mean_return"]  
        return pd.DataFrame(
            performances_flat_list,
            index = [s+"_"+n for s, n in itertools.product(decile_names,names)],
            columns=[self.__repr__()]).transpose()

    
    def backtest(self):
        prediction = self.model.predict(x=self.netdata.test.data["features"])
        backtest = pd.DataFrame(prediction, index=self.netdata.test.index, columns=['prediction'])
        backtest["actual"] = self.netdata.test.data["targets"]
        return backtest
    
    @staticmethod
    def rsquare(df): 
        y_true = df.y_true 
        y_pred = df.y_pred
        total = np.sum(np.square(y_true))
        unexplained = np.sum(np.square(np.subtract(y_true, y_pred))) # No demeaning - see Gu et al. (2020)
        return np.subtract(1.0, np.divide(unexplained,total)) *100
    
    def decile_rsquare(self):
        df = pd.DataFrame(self.model.predict(x=self.netdata.test.data["features"]),columns=["y_pred"])
        df['y_true'] =  self.netdata.test.data["targets"]
        rsq_overall = self.rsquare(df)
        df["q10"] = pd.qcut(df.y_pred,q=10)
        rsq = df.groupby("q10").apply(self.rsquare).round(3)
        rsq.index= [str(i) for i in range(10,110,10)]
        rsq["Overall"] = rsq_overall
        return rsq

    
    def portfolio_reliance(self, percent_long, percent_short):
        dataset = self.netdata.test
        data = dataset.data
        df_index = dataset.index

        pr = PortfolioReliance(percent_long=percent_long, percent_short=percent_short)
        array = pr.fit(self.model, x=data["features"], y=data["targets"], df_index=df_index)
        return pd.DataFrame(
            array, 
            index = dataset.columns['features'],
            columns=[self.__repr__()]).transpose()
    
    def model_reliance(self, on="test"):
        data_select = {
            "test": self.netdata.test,
            "train": self.netdata.train,
            "valid": self.netdata.valid
        }
        dataset = data_select.get(on)
        data = dataset.data

        loss = tf.keras.losses.MeanSquaredError if self.model.loss == "mse" else self.model.loss
        mr = ModelReliance(loss=loss)
        array = mr.fit(self.model, x=data["features"], y=data["targets"])
        return pd.DataFrame(
            array, 
            index = dataset.columns['features'],
            columns=[self.__repr__()]).transpose()
    
    def integrated_gradients(self, on="test"):
        data_select = {
            "test": self.netdata.test,
            "train": self.netdata.train,
            "valid": self.netdata.valid
        }
        dataset = data_select.get(on)
        explainer  = IntegratedGradients(self.model,
                          layer=None,
                          method="gausslegendre",
                          n_steps=50,
                          internal_batch_size=10000)
        explanation = explainer.explain(dataset.data["features"])
        attributions = explanation.attributions

        loc = pd.DataFrame(
            attributions, 
            columns=dataset.columns["features"], 
            index=dataset.index)
        
        # We need to take absolute value so that features that have both positive 
        # and negative impact do not seem unimportant 
        glob = pd.DataFrame(loc.abs().mean(), columns=[self.__repr__()]).transpose()

        return loc, glob
    
    def save(self, path):
        # make sure folder exists, if not create
        if not os.path.exists(path):
            os.makedirs(path)
        
        # save model architecture
        with open(os.path.join(path, 'model.json'),'w') as f: 
            json.dump(self.model.to_json(), f)

        # save model weights
        self.model.save_weights(os.path.join(path,'weights.h5'))
        
        # save model args
        with open(os.path.join(path, 'args.pickle'), 'wb') as f:
            pickle.dump(vars(self.args), f)
        
    
    @property 
    def netdata(self):
        return NetData(
            ytrain=self.args.ytrain, 
            yvalid=self.args.yvalid, 
            ytest=self.args.ytest, 
            dataset=self.dataset)
    
    @property 
    def n_models(self):
        return np.array([type(self.model.layers[i]) == tf.keras.Model for i in range(len(self.model.layers))]).sum()



        