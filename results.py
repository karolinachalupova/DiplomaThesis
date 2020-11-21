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

from nets import Net, Nets
from data import Cleaned, Simulated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", default=True, action="store_true", help="Use ensembles instead of individual models.")
    parser.add_argument("--dataset", default="cleaned", type=str, help="Which dataset class from data.py to use")
    parser.add_argument('--calculate_on', default="test", type=str, help="Where to calculate interpretability measures. test, train or valid")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    ########################## MUST RUN AT ALL TIMES  ################################
     # Set paths for results and models, create if necessary 
    path_generic =os.path.join("{}".format(args.dataset),"{}".format("ensembles" if args.ensemble else "individual"))
    path_results = os.path.join("results", path_generic)
    path_models = os.path.join("models", path_generic)
    if not os.path.exists(path_results):
            os.makedirs(path_results)

    # Load necessary datasets
    dataset_name_map = {
        "cleaned":Cleaned,
        "simulated":Simulated
    }
    C = dataset_name_map.get(args.dataset)
    dataset = C()
    dataset.load()

    # Create models 
    models = Nets.from_saved(path_models)
   
    # Set datasets
    [net.set_dataset(dataset, ytest=1) for net in models.nets]

    # Calculate local integrated gradients
    for net in models.nets:
        loc, glob = net.integrated_gradients(on=args.calculate_on)
        loc.to_csv(os.path.join(path_models, net.folder_name, 'integrated_gradients_{}.csv'.format(args.calculate_on)))
    
    # Calculate backtest
    for net in models.nets: 
        backtest = net.backtest()
        backtest.to_csv(os.path.join(path_models, net.folder_name, 'backtest.csv'))

    # Calculate all other 
    models.dataframe.to_csv(os.path.join(path_results, 'args.csv'))
    models.performance().to_csv(os.path.join(path_results, 'performance.csv'))
    models.model_reliance(on=args.calculate_on).to_csv(os.path.join(path_results, 'model_reliance_{}.csv'.format(args.calculate_on)))
    models.integrated_gradients_global(on=args.calculate_on).to_csv(os.path.join(path_results, 'integrated_gradients_global_{}.csv'.format(args.calculate_on)))