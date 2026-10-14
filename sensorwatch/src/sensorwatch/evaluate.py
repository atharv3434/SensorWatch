"""Event-level evaluation: did we warn *before* the failure, how early, and at what false-alarm cost?"""
from __future__ import annotations
import numpy as np
import polars as pl
from .data import CAL, CAL_WARM, RAMP

MIN_ACTIONABLE_LEAD = 6  # hours of warning a maintenance crew needs


def alarms_from_scores(s: np.ndarray, thr: float, k: int = 3, cooldown: int = 24, start: int = CAL) -> list[int]:
    """Alarm when score exceeds thr for k consecutive hours; then stay silent for `cooldown` hours."""
    alarms, run, mute_until = [], 0, -1
    for t in range(start, len(s)):
        run = run + 1 if s[t] > thr else 0
        if run >= k and t > mute_until:
            alarms.append(t)
            mute_until = t + cooldown
    return alarms


def evaluate(scores: np.ndarray, faults: pl.DataFrame, q: float = 0.995, k: int = 3) -> dict:
    thr = float(np.quantile(scores[:, CAL_WARM:CAL], q))
    M, H = scores.shape
    by_m: dict[int, list[int]] = {}
    for r in faults.iter_rows(named=True):
        by_m.setdefault(r["machine_id"], []).append(r["fail_t"])

    detected, leads, false_alarms = 0, [], 0
    for m in range(M):
        alarms = alarms_from_scores(scores[m], thr, k)
        windows = [(T - RAMP, T) for T in by_m.get(m, [])]
        for a in alarms:
            if not any(w0 <= a <= w1 + 12 for w0, w1 in windows):
                false_alarms += 1
        for w0, w1 in windows:
            hit = [a for a in alarms if w0 <= a < w1]
            if hit:
                detected += 1
                leads.append(w1 - hit[0])

    n_f = faults.height
    months = M * (H - CAL) / 720
    leads_arr = np.array(leads) if leads else np.array([0])
    return {
        "threshold": round(thr, 4),
        "detection_rate": round(detected / n_f, 3),
        "actionable_rate": round(float((leads_arr[: len(leads)] >= MIN_ACTIONABLE_LEAD).sum() / n_f), 3) if leads else 0.0,
        "median_lead_h": float(np.median(leads_arr)) if leads else 0.0,
        "false_alarms_per_machine_month": round(false_alarms / months, 3),
        "precision": round(detected / max(detected + false_alarms, 1), 3),
        "n_faults": n_f, "false_alarms": false_alarms,
    }
