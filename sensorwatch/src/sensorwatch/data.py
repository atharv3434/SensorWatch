"""Simulated fleet of machines with 4 sensors, hourly readings.

Ground truth: each machine has 1-2 failures. Before each failure a 48h degradation ramp
(vibration/temperature/current up, pressure down) develops. The data also contains
*benign* load surges (temperature + current spike for a few hours) and random sensor
glitches, which are the traps that cause false alarms.
"""
from __future__ import annotations
import numpy as np
import polars as pl

CAL = 336          # first 14 days are healthy: used for normalisation and threshold calibration
CAL_WARM = 120     # ignore early calibration steps while rolling windows / streaming models warm up
RAMP = 48          # hours of visible degradation before failure
SENSORS = {  # name: (base level, daily-cycle amplitude, noise sd)
    "temperature": (70.0, 4.0, 0.8),
    "vibration": (2.0, 0.2, 0.15),
    "pressure": (30.0, 1.5, 0.5),
    "current": (12.0, 1.0, 0.3),
}


def simulate(n_machines: int = 24, hours: int = 1440, seed: int = 3) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    M, H = n_machines, hours
    t = np.arange(H)
    d: dict[str, np.ndarray] = {}
    for name, (base, amp, sd) in SENSORS.items():
        b = base * (1 + rng.normal(0, 0.05, (M, 1)))
        phase = rng.uniform(0, 2 * np.pi, (M, 1))
        e = rng.normal(size=(M, H))
        ar = np.empty_like(e)
        ar[:, 0] = e[:, 0]
        for i in range(1, H):
            ar[:, i] = 0.8 * ar[:, i - 1] + 0.6 * e[:, i]
        d[name] = b + amp * np.sin(2 * np.pi * t / 24 + phase) + sd * ar
    d["current"] += 0.15 * (d["temperature"] - d["temperature"].mean(1, keepdims=True))

    faults = []
    for m in range(M):
        t1 = int(rng.integers(400, max(401, int(H * 0.65)) + 1))
        times = [t1]
        if rng.random() < 0.5 and t1 + 200 < H - 1:
            times.append(int(rng.integers(t1 + 200, H)))
        for T in times:
            sev = float(rng.uniform(0.5, 1.3))
            p = np.clip((t - (T - RAMP)) / RAMP, 0, 1) * (t <= T)
            p2 = p ** 2
            d["vibration"][m] += sev * 6 * SENSORS["vibration"][2] * p2
            d["temperature"][m] += sev * 5 * SENSORS["temperature"][2] * p2
            d["current"][m] += sev * 4 * SENSORS["current"][2] * p2
            d["pressure"][m] -= sev * 3 * SENSORS["pressure"][2] * p2
            faults.append({"machine_id": m, "fail_t": T, "severity": round(sev, 3)})

        for _ in range(int(rng.integers(1, 4))):  # benign load surges
            s0, dur = int(rng.integers(CAL + 10, H - 10)), int(rng.integers(4, 9))
            d["temperature"][m, s0:s0 + dur] += 4 * SENSORS["temperature"][2]
            d["current"][m, s0:s0 + dur] += 5 * SENSORS["current"][2]

    for name, (_, _, sd) in SENSORS.items():  # sensor glitches
        mask = rng.random((M, H)) < 0.00025
        d[name] += mask * rng.choice([-8.0, 8.0], (M, H)) * sd

    readings = pl.DataFrame({
        "machine_id": np.repeat(np.arange(M), H), "t": np.tile(t, M),
        **{k: v.ravel() for k, v in d.items()}})
    return readings, pl.DataFrame(faults)
