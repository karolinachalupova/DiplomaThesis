"""
Trains a single network.

As of 10/12/2019: One caveat of using TF2.0 is that TF AutoGraph
functionality does not interact nicely with Ray actors. One way to get around
this is to `import tensorflow` inside the Tune Trainable.
"""
import argparse
import numpy as np
import pandas as pd
import os
import re
import datetime
import pickle

from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, average
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD 
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError, Metric
from tensorflow import reduce_sum, square, subtract, reduce_mean, divide, cast, float32

from ray import tune
import ray

from data import MinMaxed, Normalized, Simulated, N_FEATURES

class RSquare(Metric):
    def __init__(self, name="r_square", **kwargs):
        super(RSquare, self).__init__(name=name, **kwargs) 
        self.r_square = self.add_weight(name="rsq", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        total_error = reduce_sum(square(cast(y_true, float32)))  # No demeaning - see Gu et al. (2020) 
        unexplained_error = reduce_sum(square(subtract(cast(y_true, float32), y_pred)))
        rsq = subtract(1.0, divide(unexplained_error, total_error))
        self.r_square.assign(rsq)

    def result(self):
        return self.r_square

def create_model(args, learning_rate, l1):
    hidden_layers = [int(n) for n in args.hidden_layers.split(',')]
    inputs = Input(shape=[N_FEATURES])
    hidden = inputs
    if hidden_layers != [-1]:
        for size in hidden_layers: 
            hidden = Dense(
                size,  
                kernel_regularizer=L1L2(l1=l1), 
                bias_regularizer=L1L2(l1=l1))(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = ReLU()(hidden)
    outputs = Dense(1)(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    
    # I know this is ugly, but I added the sgd arg only later so older networks 
    # do not have args.optimizer (and were optimized with Adam)
    try: 
        if args.optimizer == "sgd":
            optimizer = SGD(learning_rate=learning_rate, momentum=0.99, nesterov=True)
        elif args.optimizer == "adam":
            optimizer = Adam(learning_rate=learning_rate)
    except AttributeError: 
        optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss ='mse',
        metrics = [RootMeanSquaredError(), MeanAbsoluteError(), RSquare()]
    )
    return model

def create_ensemble(models):
    if len(models) == 1:
        return models[0]
    else: 
        inputs = Input(shape=[N_FEATURES])
        predictions = [model(inputs) for model in models]
        outputs = average(predictions)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss = 'mse',
            metrics = [RootMeanSquaredError(), MeanAbsoluteError(), RSquare()]
        )
        return model
    
class Training(tune.Trainable):
    def setup(self, config): 
        import tensorflow as tf  # IMPORTANT: See the note above.

        # I use get_pinned_object so that data is not 
        # reloaded to memory for each trial
        self.data = tune.utils.get_pinned_object(data_id)  

        # Obtain model 
        self.model = create_model(
            args=args, 
            learning_rate=config.get("learning_rate"),
            l1=config.get("l1")
        )

        # Remember the lowest validation loss reached so far
        # as well as number of consecutive periods without 
        # improvement in this minumum 
        # (used for early stopping)
        self.min_valid_loss = np.inf
        self.consec_epochs_wo_impr = 0

    def step(self):
        # Train single epoch
        self.model.reset_metrics()
        for batch in self.data.train.batches(args.batch_size): 
            train_loss, train_rmse, train_mae, train_rsq = self.model.train_on_batch(batch["features"], batch["targets"])
        
        # Validate
        self.model.reset_metrics()
        for batch in self.data.valid.batches(args.batch_size):
            valid_loss, valid_rmse, valid_mae, valid_rsq = self.model.test_on_batch(batch["features"], batch["targets"])
        
        # Early stopping
        # If valid_loss does not improve (w. r. t. absolute minimum reached so far)
        # for `args.patience` consecutive periods, stop training
        if valid_loss > self.min_valid_loss: 
            self.consec_epochs_wo_impr += 1
        else: 
            self.consec_epochs_wo_impr = 0
            self.min_valid_loss = valid_loss 

        return {
            "epoch": self.iteration, 
            "train_loss": train_loss,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_rsq": train_rsq,
            "valid_loss": valid_loss, 
            "valid_rmse": valid_rmse,
            "valid_mae": valid_mae,
            "valid_rsq": valid_rsq,
            "stop_early": self.consec_epochs_wo_impr == args.patience
        }
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.h5")
        self.model.save(checkpoint_path)
        return tmp_checkpoint_dir
    

class NetData():

    class Dataset():
        def __init__(self, 
            data:dict, 
            shuffle_batches:bool, 
            seed:int=42):
            """
            Args: 
                data (dict): dictionary with keys "features" and "targets", 
                    each containing a pd.DataFrame
                shuffle_batches (bool): Bool indicating whether or not the batches 
                    should be shuffled. 
                seed (int): random seed used for random shuffling of batches 
                    (and, if subset_percent is not None, for random subsetting)
            """
            self._data = {
                "features": np.asarray(data["features"]),
                "targets": np.asarray(data["targets"])
            }
            self._size = len(self._data["features"])
            
            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

            self._columns = {
                "features": data["features"].columns.tolist(),
                "targets": data["targets"].columns.tolist()
            }

            self._index = data["features"].index
    
        @property 
        def data(self):
            return self._data
        
        @property 
        def size(self):
            return self._size 
        
        @property 
        def columns(self):
            return self._columns
        
        @property 
        def index(self):
            return self._index
        
        def batches(self, size=None):
            """
            Generator object splitting data into batches.
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

    def __init__(self, ytrain:int, yvalid:int, ytest:int, dataset):
        """
        Splits dataset into train, valid, and test sets.
        
        Following Gu et al. (2018), I split the data 
        by taking first `ytrain` years as train set, 
        the immediately following `yvalid` years as validation set, 
        and the `ytest` years after that as test set.

        Args: 
            ytrain (int): number of years in training set
            yvalid (int): number of years in validation set
            ytest (int): number of years in test set
            dataset (instance of loaded data.MinMaxed, data.Normalized or data.Simulated) 
                dataset has attributes `targets` and `features`, each is a pd.DataFrame.
        
        Examples: 
            >>> netdata = NetData(dataset)
            >>> netdata.train.data["features"], netdata.train.data["targets"]
            >>> netdata.valid.data["features"], netdata.valid.data["targets"]
            >>> netdata.test.data["features"], netdata.test.data["targets"]

        """
        self.ytrain = ytrain
        self.yvalid = yvalid
        self.ytest = ytest

        # Number of features per example 
        assert dataset.features.shape[1] == 30
        assert (dataset.features.index == dataset.targets.index).all()

        # Create organizing masks 
        idx_year = dataset.targets.index.get_level_values('date').year
        splityear1 = idx_year.min() + self.ytrain
        splityear2 = splityear1 + self.yvalid
        splityear3 = splityear2 + self.ytest
        masks = {
            "train": (idx_year < splityear1),
            "valid": (idx_year >= splityear1) & (idx_year < splityear2),
            "test": (idx_year >= splityear2) & (idx_year < splityear3)
        }

        # Split data into train, valid and test
        for name in ["train", "valid", "test"]:
            data = {
                "features": dataset.features.loc[masks.get(name)],
                "targets": dataset.targets.loc[masks.get(name)],
            }
            setattr(self, name, self.Dataset(data, shuffle_batches=name=="train"))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=5000, type=int, help="Batch size. Gu:10000")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs. Gu:100")
    parser.add_argument("--patience", default=5, type=int, help="Patience for early stopping. Gu:5")
    
    parser.add_argument("--ytrain", default=9, type=int, help="Number of years in train set.")
    parser.add_argument("--yvalid", default=3, type=int, help="Number of years in validation set.")
    parser.add_argument("--ytest", default=3, type=int, help="Number of years in test set.")
    parser.add_argument("--hidden_layers", default="32", type=str, help='Number of neurons in hidden layers. Gu:32,16,8,4,2')
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    
    parser.add_argument("--learning_rate_low", default=0.001, type=float, help="Initial learning rate, hyperparam lower bound. Gu:0.001")
    parser.add_argument("--learning_rate_high", default=0.01, type=float, help="Initial learning rate hyperparam upper bound. Gu:0.01")
    parser.add_argument("--l1_low", default=0.00001, type=float, help='L1 regularization term, hyperparam lower bound. Gu: 0.00001')
    parser.add_argument("--l1_high", default=0.001, type=float, help='L1 regularization term, hyperparam upper bound. Gu: 0.001')
    parser.add_argument("--num_samples", default=20, type=int, help='Number of trials in the hyperparameter search.')
    
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=True, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--optimizer", default="adam", type=str, help="Optimizer for gradient descent. Gu: adam")

    parser.add_argument("--dataset", default="minmaxed", type=str, help="Which dataset class from data.py to use")
    parser.add_argument("--name", default="minmax_long", type=str, help="Name of the trial")
    
    args = parser.parse_args([] if "__file__" not in globals() else None)

    dataset_name_map = {
        "minmaxed":MinMaxed,
        "simulated":Simulated,
        "normalized":Normalized
    }

    # Fix random seeds and threads
    np.random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs_{}".format(args.name), "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Initialize ray
    if ray.is_initialized(): 
        ray.shutdown()
    ray.init()

    # Load data
    C = dataset_name_map.get(args.dataset)
    dataset = C()
    dataset.load()
    # I use pin_in_object_store so that data is not 
    # reloaded to memory for each trial
    data_id = tune.utils.pin_in_object_store(
        NetData(ytrain=args.ytrain, yvalid=args.yvalid, ytest=args.ytest, dataset=dataset))

    # Run the training 
    analysis = tune.run(
        Training,
        stop={'training_iteration': args.epochs, 'stop_early': True},
        checkpoint_at_end=True,
        metric="valid_rmse",
        mode="min",
        local_dir=args.logdir,
        verbose=1,
        config = {
            "learning_rate": tune.loguniform(args.learning_rate_low, args.learning_rate_high),
            "l1": tune.loguniform(args.l1_low, args.l1_high),
        },
        num_samples=args.num_samples,
        resources_per_trial={"cpu":1, "gpu":0}
    )

    ray.shutdown()

    # Save args
    with open(os.path.join(args.logdir,"args.pickle"), 'wb') as f: 
        pickle.dump(vars(args), f)





