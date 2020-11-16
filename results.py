"""
Model performance evaluation
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

from ray import tune
from alibi.explainers import IntegratedGradients

from train_network import create_model, NetData, create_ensemble
from data import Selected, Simulated
from utils import fix_folder_names



class ModelReliance():
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
    def from_logs(cls, logs):
        logdirs = [os.path.join(logs, x) for x in os.listdir(logs)]
        finished = [os.path.isfile(os.path.join(logdir, "args.pickle")) for logdir in logdirs]
        logdirs = [logdir for f, logdir in zip(finished, logdirs) if f]  # use finished logdirs only 
        nets = [Net.from_logdir(logdir) for logdir in logdirs]
        return cls(nets)
    
    @property 
    def dataframe(self):
        """
        Returns a dataframe, where each net is a row and args are columns.
        There are instances of Net in the "index" column.
        """
        return pd.DataFrame(dict(zip(self.nets, [vars(net.args) for net in self.nets]))).transpose()
    
    def performance(self):
        return pd.concat([net.performance() for net in self.nets])
    
    def model_reliance(self, on="test"):
        return pd.concat([net.model_reliance(on=on) for net in self.nets])
    
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
                wanted_hidden_layers = ["32", "32,16", "32,16,8", "32,16,8,4", "32,16,8,4,2"], 
                wanted_seeds = [1,2,3,4,5,6,7,8,9]):
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
    def from_logdir(cls, logdir):
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
        return cls(model, args)
    
    @classmethod
    def from_saved(cls, folder):
        with open(os.path.join(folder, "args.pickle"), "rb") as f: 
            args = argparse.Namespace(**pickle.load(f))  # loads dict and converts it to namespace
        with open(os.path.join(folder,'model.json')) as f:
            json_string = json.load(f)
        model = tf.keras.models.model_from_json(json_string, custom_objects=None)
        model.load_weights(os.path.join(folder, 'weights.h5'))
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
    
    @property 
    def folder_name(self):
        return self.__repr__().split(": ")[1]


    def save(self, directory_path):
        # make sure folder exists, if not create
        path = os.path.join(directory_path,self.folder_name)
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="C://Users//HP//Google Drive//DiplomaThesisGDrive/logs_simulated", type=str, help="Path to logdir.")
    parser.add_argument("--ensemble", default=True, action="store_true", help="Use ensembles instead of individual models.")
    parser.add_argument("--dataset", default="simulated", type=str, help="Which dataset class from data.py to use")
    parser.add_argument("--fix_folder_names", default=False, type=bool, help="Fix names of Training folders.")
    parser.add_argument('--calculate_on', default="train", type=str, help="Where to calculate interpretability measures. test, train or valid")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    dataset_name_map = {
        "selected":Selected,
        "simulated":Simulated
    }

    if args.fix_folder_names:
        fix_folder_names(args.logdir)

    C = dataset_name_map.get(args.dataset)
    dataset = C()
    dataset.load()
    nets = Nets.from_logs(args.logdir)
    
    if args.ensemble:
        models = Nets(nets.create_ensembles())
    else: 
        models = nets

    [net.set_dataset(dataset, ytest=1) for net in models.nets]
    
    path_generic =os.path.join("{}".format(args.dataset),"{}".format("ensembles" if args.ensemble else "individual"))
    path_results = os.path.join("results", path_generic)
    path_models = os.path.join("models", path_generic)
    if not os.path.exists(path_results):
            os.makedirs(path_results)
    if not os.path.exists(path_models):
            os.makedirs(path_models)
    
    for net in models.nets: 
        net.save(directory_path=path_models)
    
    for net in models.nets:
        loc, glob = net.integrated_gradients(on=args.calculate_on)
        loc.to_csv(os.path.join(path_models, net.folder_name, 'integrated_gradients_{}.csv'.format(args.calculate_on)))

    models.dataframe.to_csv(os.path.join(path_results, 'args.csv'))
    models.performance().to_csv(os.path.join(path_results, 'performance.csv'))
    models.model_reliance(on=args.calculate_on).to_csv(os.path.join(path_results, 'model_reliance_{}.csv'.format(args.calculate_on)))
    models.integrated_gradients_global(on=args.calculate_on).to_csv(os.path.join(path_results, 'integrated_gradients_global_{}.csv'.format(args.calculate_on)))