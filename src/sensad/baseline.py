from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

EPS = 1e-12

def _median(x: np.ndarray) -> float:
    return float(np.nanmedian(x))

def _mad(x: np.ndarray) -> float:
    # Median Absolute Deviation
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def _robust_z(x: np.ndarray, med: float, mad: float) -> np.ndarray:
    # 1.4826 makes MAD comparable to std for normal dist
    denom = 1.4826 * mad + EPS
    return (x - med) / denom

@dataclass
class BaselineModel:
    columns: List[str]
    med: Dict[str, float]
    mad: Dict[str, float]
    # Threshold on aggregated score
    threshold: float = 3.5
    # How to aggregate across sensors: "max" or "mean"
    agg: str = "max"

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        X shape: (n, d) for self.columns
        returns score shape: (n,), higher => more anomalous
        """
        zs = []
        for j, col in enumerate(self.columns):
            med = self.med[col]
            mad = self.mad[col]
            z = _robust_z(X[:, j], med, mad)
            zs.append(np.abs(z))
        Z = np.stack(zs, axis=1)  # (n,d)

        if self.agg == "mean":
            s = np.mean(Z, axis=1)
        else:
            s = np.max(Z, axis=1)
        return s

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        score = self.score(X)
        pred = (score >= self.threshold).astype(int)
        return score, pred

def fit_baseline(X: np.ndarray, columns: List[str], agg: str = "max") -> BaselineModel:
    med = {}
    mad = {}
    for j, col in enumerate(columns):
        v = X[:, j]
        med[col] = _median(v)
        mad[col] = _mad(v)
        # if mad is 0 (flat sensor), keep it tiny to avoid blowups
        if mad[col] < 1e-9:
            mad[col] = 1e-9
    return BaselineModel(columns=columns, med=med, mad=mad, threshold=3.5, agg=agg)

def choose_threshold_from_train(model: BaselineModel, X_train: np.ndarray, q: float = 0.995) -> float:
    """
    Sets threshold from high quantile of train scores.
    Default q=0.995 => about 0.5% false positives on train.
    """
    s = model.score(X_train)
    return float(np.quantile(s, q))
