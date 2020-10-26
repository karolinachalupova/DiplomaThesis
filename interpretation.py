"""
Calculation of Model Reliance and Model Class Reliance from Fisher et al. 2019
For Tensorflow models
"""

import numpy as np


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
        


class ModelClassReliance(): 
    def __init__(self, loss):
        """
        Arguments: 
            loss: a tf.keras.losses object (class, not instance)
        """
        self.mr = ModelReliance(loss)
        self.loss = loss()
    
    def fit(self, epsilon, F, x_test, y_test, x_valid, y_valid):
        epsilon_rashomon_set = self._get_epsilon_rashomon_set(
            epsilon, F, x_valid, y_valid)
        mrs = [self.mr.fit(f, x_test, y_test) for f in epsilon_rashomon_set]
        return np.min(mrs), np.max(mrs)
    
    def _get_epsilon_rashomon_set(self, epsilon, F, x_valid, y_valid):
        """
        Returns list with epsilon-Rashomon models. See Fisher, eq. 4.1.
        Arguments: 
            epsilon (float): epsilon
            f_ref: instance of tf.keras.Model
            F: list of tf.keras.Model instances
        """
        # I use best-in-class model from F as f_ref (
        # (==with smallest validation loss (argmin f from F: e_orig(f)))
        e_origs = np.array([self.mr._e_orig(f, x_valid, y_valid) for f in F])
        e_orig_f_ref = np.min(e_origs)
        return [f for index, f in enumerate(F) if e_origs[index] <= e_orig_f_ref + epsilon]
    

        