"""
Defines and trains models, evaluates performance on test set.
"""
import argparse
import numpy as np
import tensorflow as tf
import os
import datetime

from netdata import NetData


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10000, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--ytrain", default=9, type=int, help="Number of years in train set.")
    parser.add_argument("--yvalid", default=6, type=int, help="Number of years in validation set.")
    parser.add_argument("--ytest", default=1, type=int, help="Number of years in test set.")
    parser.add_argument("--hidden_layers", default="32", type=str, help='Number of neurons in hidden layers.')
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--l1", default=0.001, type=float, help='L1 regularization term.')
    parser.add_argument("--patience", default=5, type=int, help="Patience for early stopping.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
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

    network = Network(args)
    network.train()

class Network(): 
    def __init__(self, args):
        hidden_layers = list(int(n for n in (args.hidden_layers.split(','))))
        regularizer = tf.keras.regularizers.L1L2(l1=args.l1)
        inputs = tf.keras.layers.Input()
        hidden = inputs
        for size in hidden_layers: 
            hidden = tf.keras.layers.Dense(
                size,  
                kernel_regularizer=regularizer, 
                bias_regularizer=regularizer)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
        outputs = tf.keras.layers.Dense(1)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss = tf.keras.losses.MeanSquaredError(),
        )
    
    def train(self):
        tb_callback = tf.keras.callbacks.TensorBoard(
            args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=1)

        nd = NetData(ytrain=args.ytrain, yvalid=args.yvalid, ytest=args.ytest)
        history = self.model.fit(
            x=nd.train.data["features"],
            y=nd.train.data["labels"],
            batch_size = args.batch_size, 
            epochs = args.epochs, 
            validation_data=(nd.valid.data["features"], nd.valid.data["labels"]),
            callbacks=[tb_callback, early_stopping_callback]
        )
        self.model.save(os.path.join(args.logdir, "model.h5"))
    
    def evaluate(self):
        nd = NetData(ytrain=args.ytrain, yvalid=args.yvalid, ytest=args.ytest)
        return self.model.evaluate(
            x=nd.test.data["features"],
            y=nd.test.data["labels"],
            batch_size=args.batch_size
        )
        
       

