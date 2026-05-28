from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self) -> None:
        self._model = LogisticRegression(C=1e6, solver="lbfgs")
        self._fitted = False

    def fit(self, confidence: NDArray[np.float64], correct: NDArray[np.bool_]) -> PlattCalibrator:
        x = confidence.reshape(-1, 1)
        y = correct.astype(np.int64)
        if len(np.unique(y)) < 2:
            self._fitted = False
            return self
        self._model.fit(x, y)
        self._fitted = True
        return self

    def transform(self, confidence: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            return confidence.astype(np.float64)
        return self._model.predict_proba(confidence.reshape(-1, 1))[:, 1].astype(np.float64)
