"""CLI: simulate fleet, score with 4 detectors, evaluate on early-warning metrics, plot."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import simulate, CAL, RAMP
from .features import build_features
from .detectors import score_zscore, score_iforest, score_hst
from .evaluate import evaluate, alarms_from_scores

OUT = Path(__file__).resolve().parents[2] / "outputs"
COLORS = {"z-score baseline": "#9ca3af", "Isolation Forest (batch)": "#f59e0b",
          "HST streaming (adaptive)": "#60a5fa", "HST streaming (quarantine)": "#4f46e5"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", type=int, default=24)
    ap.add_argument("--hours", type=int, default=1440)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)

    readings, faults = simulate(a.machines, a.hours, a.seed)
    readings.write_parquet(OUT / "readings.parquet")
    df = build_features(readings)

    scores = {
        "z-score baseline": score_zscore(df),
        "Isolation Forest (batch)": score_iforest(df),
        "HST streaming (adaptive)": score_hst(df, adaptive=True),
        "HST streaming (quarantine)": score_hst(df, adaptive=False),
    }
    results = {name: evaluate(s, faults) for name, s in scores.items()}
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    for n, r in results.items():
        print(f"{n:30s} detect={r['detection_rate']:.2f} actionable={r['actionable_rate']:.2f} "
              f"lead={r['median_lead_h']:>4.0f}h  FA/machine-month={r['false_alarms_per_machine_month']:.2f}")

    # Plot 1: comparison
    names = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for axi, key, title in zip(ax, ("actionable_rate", "median_lead_h", "false_alarms_per_machine_month"),
                               ("Faults caught with >=6h warning", "Median lead time (h)", "False alarms / machine-month")):
        axi.bar(range(4), [results[n][key] for n in names], color=[COLORS[n] for n in names])
        axi.set_xticks(range(4), ["z-score", "IForest", "HST\nadaptive", "HST\nquarantine"], fontsize=8)
        axi.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=140); plt.close(fig)

    # Plot 2: timeline of one failing machine
    f = faults.filter(faults["fail_t"] > 650).sort("severity", descending=True)
    m, T = (f["machine_id"][0], f["fail_t"][0]) if f.height else (faults["machine_id"][0], faults["fail_t"][0])
    lo, hi = max(CAL, T - 200), min(a.hours, T + 30)
    x = np.arange(lo, hi)
    mdf = df.filter(df["machine_id"] == m).sort("t")
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax[0].plot(x, mdf["z_vibration"].to_numpy()[lo:hi], color="#6b7280", lw=1, label="vibration (z)")
    ax[0].plot(x, mdf["z_temperature"].to_numpy()[lo:hi], color="#ef4444", lw=1, alpha=.7, label="temperature (z)")
    ax[0].set_ylabel("deseasonalised z"); ax[0].legend(fontsize=8, loc="upper left")
    for name in ("Isolation Forest (batch)", "HST streaming (quarantine)"):
        s = scores[name][m]; thr = results[name]["threshold"]
        ax[1].plot(x, s[lo:hi] / thr, color=COLORS[name], lw=1.2, label=name)
        for al in alarms_from_scores(s, thr):
            if lo <= al < hi:
                ax[1].scatter(al, s[al] / thr, color=COLORS[name], marker="v", s=40, zorder=3)
    ax[1].axhline(1, color="k", ls="--", lw=1); ax[1].set_ylabel("score / alarm threshold")
    ax[1].legend(fontsize=8, loc="upper left")
    for axi in ax:
        axi.axvspan(T - RAMP, T, color="#fecaca", alpha=.45); axi.axvline(T, color="#b91c1c")
    ax[1].set_xlabel("hour"); ax[0].set_title(f"Machine {m}: degradation window (red) and failure at t={T}")
    fig.tight_layout(); fig.savefig(OUT / "timeline.png", dpi=140); plt.close(fig)
    return results


if __name__ == "__main__":
    main()
