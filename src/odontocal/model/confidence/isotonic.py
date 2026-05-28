from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False

    def fit(
        self, confidence: NDArray[np.float64], correct: NDArray[np.bool_]
    ) -> IsotonicCalibrator:
        if confidence.size == 0:
            self._fitted = False
            return self
        self._model.fit(confidence.astype(np.float64), correct.astype(np.float64))
        self._fitted = True
        return self

    def transform(self, confidence: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            return confidence.astype(np.float64)
        return np.asarray(self._model.transform(confidence.astype(np.float64)), dtype=np.float64)
