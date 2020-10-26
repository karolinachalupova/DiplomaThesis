    @staticmethod
    def best(models, metric="mse", mode="min"):
        """
        Returns `n` best models (best performance on validation set)
        n is chosen such that the ensemble of the models has best 
        performance on validation set
        """
        metrics = np.array([model.evaluate(on="valid").get(metric) for model in models])
        # models sorted in ascending order on metric
        sorted_models = [x for _,x in sorted(zip(metrics, models))]
        if mode == "min":
            # Taking models from the left side of the ascending list
            ensemble_metrics = np.array([Ensemble(sorted_models[:n]).evaluate(on="valid").get(metric) for n in range(2,len(models)+1)])
            return sorted_models[:np.argmin(ensemble_metrics)+2]
        elif mode =="max": 
            # Taking models from the right side of the ascending list
            ensemble_metrics = np.array([Ensemble(sorted_models[n:]).evaluate(on="valid").get(metric) for n in range(0,len(models)-1)])
            return sorted_models[np.argmax(ensemble_metrics):]
        else: 
            raise ValueError("Argument mode only takes values 'min' or 'max'")
    
    @staticmethod
    def best_n(models, n=2, metric="mse", mode="min"):
        """
        Returns `n` best models (best performance on validation set)
        n is chosen such that the ensemble of the models has best 
        performance on validation set
        """
        metrics = np.array([model.evaluate(on="valid").get(metric) for model in models])
        # models sorted in ascending order on metric
        sorted_models = [x for _,x in sorted(zip(metrics, models))]
        if mode == "min":
            # Taking models from the left side of the ascending list
            return sorted_models[:n]
        elif mode =="max": 
            # Taking models from the right side of the ascending list
            return sorted_models[-n:]
        else: 
            raise ValueError("Argument mode only takes values 'min' or 'max'")
