"""
Defines and trains models

As of 10/12/2019: One caveat of using TF2.0 is that TF AutoGraph
functionality does not interact nicely with Ray actors. One way to get around
this is to `import tensorflow` inside the Tune Trainable.
"""
import argparse
import numpy as np
import os
import re
import datetime

from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

from ray import tune
import ray

from netdata import NetData


class Network(): 
    def __init__(self, args, learning_rate, l1):
        hidden_layers = [int(n) for n in args.hidden_layers.split(',')]
        inputs = Input(shape=[153])
        hidden = inputs
        for size in hidden_layers: 
            hidden = Dense(
                size,  
                kernel_regularizer=L1L2(l1=l1), 
                bias_regularizer=L1L2(l1=l1))(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = ReLU()(hidden)
        outputs = Dense(1)(hidden)
        self.model = Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss ='mse',
            metrics = [RootMeanSquaredError(), MeanAbsoluteError()]
        )


class Training(tune.Trainable):
    def setup(self, config): 
        import tensorflow as tf  # IMPORTANT: See the note above.

        # I use get_pinned_object so that data is not 
        # reloaded to memory for each trial
        self.data = tune.utils.get_pinned_object(data_id)  

        # Obtain model 
        self.network = Network(
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
        self.network.model.reset_metrics()
        for batch in self.data.train.batches(args.batch_size): 
            train_loss, train_rmse, train_mae = self.network.model.train_on_batch(batch["features"], batch["targets"])
        
        # Validate
        self.network.model.reset_metrics()
        for batch in self.data.valid.batches(args.batch_size):
            valid_loss, valid_rmse, valid_mae = self.network.model.test_on_batch(batch["features"], batch["targets"])
        
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
            "valid_loss": valid_loss, 
            "valid_rmse": valid_rmse,
            "valid_mae": valid_mae, 
            "stop_early": self.consec_epochs_wo_impr == args.patience
        }
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.h5")
        self.network.model.save(checkpoint_path)
        return tmp_checkpoint_dir
    

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
    
    args = parser.parse_args([] if "__file__" not in globals() else None)

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
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Initialize ray
    if ray.is_initialized(): 
        ray.shutdown()
    ray.init()

    # Load data 
    data = NetData(ytrain=args.ytrain, yvalid=args.yvalid, ytest=args.ytest)
    # I use pin_in_object_store so that data is not 
    # reloaded to memory for each trial
    data_id = tune.utils.pin_in_object_store(data)

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
        resources_per_trial={"cpu":0, "gpu":1}
    )

    ray.shutdown()





