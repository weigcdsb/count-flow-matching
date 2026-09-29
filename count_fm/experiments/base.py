class UncondModel:
    name = "base"
    def fit(self, X_train_counts, **kwargs):
        raise NotImplementedError
    def sample(self, n_samples, **kwargs):
        raise NotImplementedError
