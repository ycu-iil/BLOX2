import os
import numpy as np
from huggingface_hub import login, whoami
from huggingface_hub.utils import HfHubHTTPError
from tabpfn import TabPFNRegressor
from blox2 import Predictor

class TabPFNPredictor(Predictor):
    def __init__(self, token_path: str, n_estimators: int=4, device: str | list[str]="auto", n_jobs: int= None):
        self.token_path = token_path
        self.n_estimators = n_estimators
        self.device = device
        self.n_jobs = n_jobs
        self._models = None
        
        if token_path is not None:
            self._login_to_hf(token_path)

    def _login_to_hf(self, token_path: str):
        token_path = os.path.expanduser(token_path)
        if not os.path.isfile(token_path):
            raise FileNotFoundError(f"HF token file not found: {token_path}")

        with open(token_path, "r") as f:
            token = f.read().strip()

        if not token:
            raise ValueError("HF token file is empty.")

        # do nothing if already logged in
        try:
            whoami()
            return
        except Exception:
            pass

        login(token=token, add_to_git_credential=False)

        try:
            info = whoami()
        except HfHubHTTPError as e:
            raise RuntimeError("HF login failed.") from e

    def fit(self, X: np.ndarray, Y: np.ndarray):
        d_obj = Y.shape[1]

        if self._models is None:
            self._models = []
            for j in range(d_obj):
                yj = np.asarray(Y[:, j], float)
                if type(self.device) == list:
                    m = TabPFNRegressor(device=self.device[j], n_estimators=self.n_estimators, n_jobs=self.n_jobs)
                else:
                    m = TabPFNRegressor(device=self.device, n_estimators=self.n_estimators, n_jobs=self.n_jobs)
                m.fit(X, yj)
                self._models.append(m)
        else:
            for j in range(d_obj):
                yj = np.asarray(Y[:, j], float)
                m = self._models[j]
                m.fit(X, yj)

        return self

    def pred(self, X: np.ndarray) -> np.ndarray:
        if self._models is None:
            raise RuntimeError("Call fit() before pred().")

        preds = []
        for m in self._models:
            pj = np.asarray(m.predict(X), float) # (m,)
            preds.append(pj)

        Yhat = np.stack(preds, axis=1) # (m, d_obj)
        
        return Yhat