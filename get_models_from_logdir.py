"""
Extracts models out of tensorboard logs and saves them. 
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

from ray import tune
from alibi.explainers import IntegratedGradients

from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from train_network import RSquare

from train_network import create_model, NetData, create_ensemble
from utils import fix_folder_names


from nets import Net, Nets

def get_nets_from_logs(logs):
        logdirs = [os.path.join(logs, x) for x in os.listdir(logs)]
        finished = [os.path.isfile(os.path.join(logdir, "args.pickle")) for logdir in logdirs]
        logdirs = [logdir for f, logdir in zip(finished, logdirs) if f]  # use finished logdirs only 
        nets = [get_net_from_logs(lodir) for logdir in logdirs]
        return Nets(nets)

def get_net_from_logs(logs):
        """
        Best tf.keras.model 
        (best: best performance on validation set out of all models in the logdir)
        used to choose optimum hyperparameters. 
        
        Arguments: 
            dataset: instance of data.Cleaned or data.Simulated
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
        return Net(model, args)


if __name__ == "__main__":
    ## Extract models from logdir, save them.
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="C:/Users/HP/projects/DiplomaThesis/logs/logs_nn3", type=str, help="Path to logdir.")
    parser.add_argument("--name", default="nn3", type=str, help="Which dataset class from data.py to use")
    parser.add_argument("--ensemble", default=False, action="store_true", help="Whether to create ensemble models or seed models.")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    path_ensembles =os.path.join("models", "{}".format(args.name),"ensembles")
    path_seeds = os.path.join("models", "{}".format(args.name),"seeds")
    for p in [path_ensembles, path_seeds]:
        if not os.path.exists(p):
                os.makedirs(p)

    if args.ensemble:
        # Create ensembles from already extracted seed models
        seed_nets = Nets.from_saved(path_seeds)
        ensemble_models = Nets(seed_nets.create_ensembles())

        # Save the ensemble models
        for net in ensemble_models.nets: 
            path = net.__repr__().split(": ")[1]
            net.save(os.path.join(path_ensembles, path))
    
    else: 
        # Extract seed models from logdir
        #fix_folder_names(args.logdir)
        seed_nets = get_nets_from_logs(args.logdir)

        # Check if there are some missing models
        if args.name == "nn1":
            wanted_hidden_layers = ["32"]
            wanted_ytrain = list(range(6,25))
        elif args.name == "nn2":
            wanted_hidden_layers = ["32,16"]
            wanted_ytrain = list(range(6,25))
        elif args.name == "nn3":
            wanted_hidden_layers = ["32,16,8"]
            wanted_ytrain = list(range(6,25))
        elif args.name == "nn4":
            wanted_hidden_layers = ["32,16,8,4"]
            wanted_ytrain = list(range(6,25))
        elif args.name == "lr":
            wanted_hidden_layers = ["-1"]
            wanted_ytrain = list(range(6,25))
        elif args.name == "beta":
            wanted_hidden_layers = ["-1", "32", "32,16", "32,16,8", "32,16,8,4", "32,16,8,4,2"]
            wanted_ytrain = [24]
        missing = seed_nets.get_missing(
            wanted_hidden_layers = wanted_hidden_layers,
            wanted_ytrain = wanted_ytrain,
            wanted_seeds = list(range(1,11)))
        if missing:
            warnings.warn("There are missing models (ytrain_hidden: [seeds]): {}".format(missing))

        # Save the seed models
        for net in seed_nets.nets:
            path = net.__repr__().split(": ")[1]
            net.save(os.path.join(path_seeds, os.path.dirname(path)))