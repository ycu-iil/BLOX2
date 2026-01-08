import numpy as np
from sklearn.ensemble import RandomForestRegressor
from blox2 import Predictor

class RandomForestPredictor(Predictor):
    def __init__(self, n_estimators: int=200, max_depth: int=None, min_samples_split: int=2, min_samples_leaf: int=1, max_features: str | float | int = "sqrt", bootstrap: bool=True, oob_score: bool=False, n_jobs: int=-1, random_state: int=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self._model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, bootstrap=self.bootstrap, oob_score=self.oob_score, n_jobs=self.n_jobs, random_state=self.random_state)
        self._model.fit(X, Y)

    def pred(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before pred().")
        return self._model.predict(X)

    def pred_samples(self, X: np.ndarray, n_samples: int=None) -> np.ndarray:
        n_samples = n_samples or self.n_estimators
        
        if self._model is None:
            raise RuntimeError("Call fit() before pred_samples().")

        # collect per-tree predictions
        # each estimator predicts shape (m,) or (m, d_obj):  normalize to (m, d_obj)
        preds = []
        for est in self._model.estimators_:
            p = est.predict(X)
            if p.ndim == 1:
                p = p[:, None]
            preds.append(p)

        return np.stack(preds, axis=0)