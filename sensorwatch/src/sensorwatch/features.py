"""Deseasonalise per machine (hour-of-day profile from the healthy period), then add rolling features."""
from __future__ import annotations
import numpy as np
import polars as pl
from .data import SENSORS, CAL

S = list(SENSORS)
FEATURES = ([f"z_{s}" for s in S] + [f"m12_{s}" for s in S]
            + [f"s12_{s}" for s in S] + [f"d12_{s}" for s in S])


def build_features(readings: pl.DataFrame) -> pl.DataFrame:
    df = readings.sort(["machine_id", "t"]).with_columns((pl.col("t") % 24).alias("hod"))
    cal = df.filter(pl.col("t") < CAL)
    prof = cal.group_by(["machine_id", "hod"]).agg([pl.col(s).mean().alias(f"{s}_mu") for s in S])
    df = df.join(prof, on=["machine_id", "hod"]).sort(["machine_id", "t"])
    df = df.with_columns([(pl.col(s) - pl.col(f"{s}_mu")).alias(f"r_{s}") for s in S])
    sd = df.filter(pl.col("t") < CAL).group_by("machine_id").agg([pl.col(f"r_{s}").std().alias(f"{s}_sd") for s in S])
    df = df.join(sd, on="machine_id").sort(["machine_id", "t"])
    df = df.with_columns([(pl.col(f"r_{s}") / pl.col(f"{s}_sd")).alias(f"z_{s}") for s in S])
    over = "machine_id"
    df = df.with_columns(
        [pl.col(f"z_{s}").rolling_mean(12).over(over).alias(f"m12_{s}") for s in S]
        + [pl.col(f"z_{s}").rolling_std(12).over(over).alias(f"s12_{s}") for s in S])
    df = df.with_columns([(pl.col(f"m12_{s}") - pl.col(f"m12_{s}").shift(12).over(over)).alias(f"d12_{s}") for s in S])
    return df.with_columns([pl.col(c).fill_null(0.0) for c in FEATURES])


def scale01(df: pl.DataFrame) -> np.ndarray:
    """Fixed-bound scaling to [0, 1] (streaming models cannot see the future to min-max)."""
    X = df.select(FEATURES).to_numpy().copy()
    for j, name in enumerate(FEATURES):
        X[:, j] = np.clip(X[:, j] / 3, 0, 1) if name.startswith("s12_") else np.clip((X[:, j] + 6) / 12, 0, 1)
    return X
