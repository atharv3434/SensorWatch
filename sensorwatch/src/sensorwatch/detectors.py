"""Three detector families. Each returns an (n_machines, hours) anomaly-score matrix."""
from __future__ import annotations
import numpy as np
import polars as pl
from river import anomaly
from sklearn.ensemble import IsolationForest
from .data import CAL, CAL_WARM
from .features import FEATURES, S, scale01


def _mat(df: pl.DataFrame, v: np.ndarray) -> np.ndarray:
    M = df["machine_id"].n_unique()
    return v.reshape(M, -1)


def score_zscore(df: pl.DataFrame) -> np.ndarray:
    """Classic baseline: largest absolute deseasonalised z-score across sensors."""
    z = df.select([f"z_{s}" for s in S]).to_numpy()
    return _mat(df, np.abs(z).max(1))


def score_iforest(df: pl.DataFrame, seed: int = 0) -> np.ndarray:
    """Batch multivariate model trained once on pooled healthy data."""
    cal = df.filter((pl.col("t") >= 24) & (pl.col("t") < CAL)).select(FEATURES).to_numpy()
    model = IsolationForest(n_estimators=200, max_samples=512, random_state=seed).fit(cal)
    return _mat(df, -model.score_samples(df.select(FEATURES).to_numpy()))


def score_hst(df: pl.DataFrame, adaptive: bool, q: float = 0.995, seed: int = 0,
              n_trees: int = 25, height: int = 8, window: int = 72) -> np.ndarray:
    """Streaming Half-Space Trees (River), one model per machine.

    adaptive=True  : learns from every sample, so it can slowly absorb a degradation trend.
    adaptive=False : after calibration it stops learning from samples scored above the alarm
                     threshold (quarantine), so a developing fault cannot become 'normal'.
    """
    M = df["machine_id"].n_unique()
    X = scale01(df).reshape(M, -1, len(FEATURES))
    H = X.shape[1]
    models = [anomaly.HalfSpaceTrees(n_trees=n_trees, height=height, window_size=window, seed=seed + m)
              for m in range(M)]
    sc = np.zeros((M, H))
    as_dict = lambda v: {i: float(x) for i, x in enumerate(v)}
    for t in range(CAL):
        for m in range(M):
            x = as_dict(X[m, t])
            sc[m, t] = models[m].score_one(x)
            models[m].learn_one(x)
    thr = float(np.quantile(sc[:, CAL_WARM:CAL], q))
    for t in range(CAL, H):
        for m in range(M):
            x = as_dict(X[m, t])
            sc[m, t] = models[m].score_one(x)
            if adaptive or sc[m, t] <= thr:
                models[m].learn_one(x)
    return sc
