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
from tensorflow.keras.metrics import MeanSquaredError

from ray import tune

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
            metrics = [MeanSquaredError(name="mse")]
        )


class Training(tune.Trainable):
    def setup(self, config): 
        import tensorflow as tf  # IMPORTANT: See the note above.

        # Load training and valid data
        self.nd = NetData(ytrain=args.ytrain, yvalid=args.yvalid, ytest=args.ytest)

        # Obtain model 
        self.network = Network(
            args=args, 
            learning_rate=config.get("learning_rate"),
            l1=config.get("l1")
        )

    def step(self):
        # TODO: implement early stopping

        # Train single epoch
        self.network.model.reset_metrics()
        for batch in self.nd.train.batches(args.batch_size): 
            train_loss, train_mse = self.network.model.train_on_batch(batch["features"], batch["targets"])
        
        # Validate
        self.network.model.reset_metrics()
        for batch in self.nd.valid.batches(args.batch_size):
            valid_loss, valid_mse = self.network.model.test_on_batch(batch["features"], batch["targets"])

        return {
            "epoch": self.iteration, 
            "train_loss": train_loss,
            "train_mse": train_mse, 
            "valid_loss": valid_loss, 
            "valid_mse": valid_mse
        }

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size. Gu:10000")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs. Gu:100")
    parser.add_argument("--patience", default=5, type=int, help="Patience for early stopping. Gu:5")
    
    parser.add_argument("--ytrain", default=9, type=int, help="Number of years in train set.")
    parser.add_argument("--yvalid", default=6, type=int, help="Number of years in validation set.")
    parser.add_argument("--ytest", default=1, type=int, help="Number of years in test set.")
    parser.add_argument("--hidden_layers", default="32", type=str, help='Number of neurons in hidden layers. Gu:32,16,8,4,2')
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    
    parser.add_argument("--learning_rate_low", default=0.001, type=float, help="Initial learning rate, hyperparam lower bound. Gu:0.001")
    parser.add_argument("--learning_rate_high", default=0.01, type=float, help="Initial learning rate hyperparam upper bound. Gu:0.01")
    parser.add_argument("--l1_low", default=0.00001, type=float, help='L1 regularization term, hyperparam lower bound. Gu: 0.00001')
    parser.add_argument("--l1_high", default=0.001, type=float, help='L1 regularization term, hyperparam upper bound. Gu: 0.001')
    parser.add_argument("--num_samples", default=3, type=int, help='Number of trials in the hyperparameter search.')
    
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

    analysis = tune.run(
        Training, 
        stop={'training_iteration': args.epochs},
        metric="valid_mse",
        mode="min",
        local_dir=args.logdir,
        verbose=1,
        config = {
            "learning_rate": tune.loguniform(args.learning_rate_low, args.learning_rate_high),
            "l1": tune.loguniform(args.l1_low, args.l1_high),
        },
        num_samples=args.num_samples
    )


