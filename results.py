"""
Calculates all results.
"""
import re
import itertools
import json
import numpy as np
import os
import pickle
import argparse
import pandas as pd

from nets import Net, Nets
from data import MinMaxed, Normalized, Simulated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", default=False, action="store_true", help="Whether to use ensembles or of seed models.")
    parser.add_argument("--dataset", default="minmaxed", type=str, help="Which dataset class from data.py to use")
    parser.add_argument("--name", default="nn1", type=str, help="Name of the trial")
    parser.add_argument('--calculate_on', default="test", type=str, help="Where to calculate interpretability measures. test, train or valid")

    args = parser.parse_args([] if "__file__" not in globals() else None)

     # Set paths for results and models, create if necessary 
    path_generic =os.path.join("{}".format(args.name),"{}".format("ensembles" if args.ensemble else "seeds"))
    path_results = os.path.join("results", path_generic)
    path_models = os.path.join("models", path_generic)
    if not os.path.exists(path_results):
            os.makedirs(path_results)

    # Load necessary datasets
    dataset_name_map = {
        "simulated":Simulated,
        "minmaxed":MinMaxed, 
        "normalized":Normalized
    }
    C = dataset_name_map.get(args.dataset)
    dataset = C()
    dataset.load()

    # Create models 
    models = Nets.from_saved(path_models)
   
    # Set datasets
    [net.set_dataset(dataset, ytest=1) for net in models.nets]

    if args.ensemble: 
        # Calculate local integrated gradients

        for net in models.nets:
            loc, glob = net.integrated_gradients(on=args.calculate_on)
            path = net.__repr__().split(": ")[1]
            loc.to_csv(os.path.join(path_models, path, 'integrated_gradients_{}.csv'.format(args.calculate_on)))
        
        models.decile_performance().to_csv(os.path.join(path_results, 'decile_performance.csv'))

        # Calculate backtest
        for net in models.nets: 
            backtest = net.backtest()
            path = net.__repr__().split(": ")[1]
            backtest.to_csv(os.path.join(path_models, path, 'backtest.csv'))

    # Calculate all other 
    models.dataframe.to_csv(os.path.join(path_results, 'args.csv'))
    models.performance().to_csv(os.path.join(path_results, 'performance.csv'))
    models.model_reliance(on=args.calculate_on).to_csv(os.path.join(path_results, 'model_reliance_{}.csv'.format(args.calculate_on)))
    models.integrated_gradients_global(on=args.calculate_on).to_csv(os.path.join(path_results, 'integrated_gradients_global_{}.csv'.format(args.calculate_on)))
    models.portfolio_reliance(percent_long=10,percent_short=10).to_csv(os.path.join(path_results, 'portfolio_reliance.csv'))
    models.decile_performance().to_csv(os.path.join(p, 'decile_performance.csv'))
