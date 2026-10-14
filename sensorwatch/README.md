# SensorWatch: Early-Warning Anomaly Detection for Machines

Predictive maintenance on a simulated fleet (24 machines, 60 days of hourly data,
4 sensors, 34 failures). Detectors are scored on what matters operationally:
**did we warn before the failure, how early, and how many false alarms did it cost?**

## What's inside
| Component | Tech |
|---|---|
| Fleet simulator with degradation ramps, benign load surges, sensor glitches | NumPy + **Polars** |
| Per-machine deseasonalisation (hour-of-day profile from the healthy period) and rolling features | **Polars** window expressions |
| Batch multivariate detector | **scikit-learn** Isolation Forest, trained once on healthy data |
| Streaming detector, one model per machine | **River** Half-Space Trees, with and without *quarantine learning* |
| Baseline | max |z-score| across sensors |
| Event-level evaluation | detection rate, lead time, false alarms per machine-month |

**Quarantine learning:** after calibration, the streaming model stops learning from samples
it scores above the alarm threshold, so a slowly developing fault cannot become the new normal.

Alarm rule (identical for all detectors): score above a threshold set at the 99.5th percentile of
healthy-period scores, for 3 consecutive hours, with a 24h cooldown.

## Run
```bash
pip install -r requirements.txt
export PYTHONPATH=src
make run    # prints a results table; writes outputs/{metrics.json,comparison.png,timeline.png}
make test
```

## Results (seed 3)
| Detector | Detected | Caught with 6h+ warning | Median lead | False alarms / machine-month |
|---|---|---|---|---|
| z-score baseline | 91% | 79% | 14h | **1.39** |
| Isolation Forest (batch) | **94%** | **91%** | 16h | 2.36 |
| HST streaming (adaptive) | 79% | 73% | 16h | 1.98 |
| HST streaming (quarantine) | 79% | 77% | 14h | 1.47 |

What this shows:
- Isolation Forest catches the most failures early, but alarms the most. Its extra false alarms are mostly
  the benign load surges.
- Quarantine learning cut the streaming model's false alarms and improved its early catches over the
  adaptive version, which supports the "don't learn from flagged data" idea.
- The streaming model did not beat the simple z-score baseline here. Streaming wins on memory and
  adaptation to genuinely changing machines, and this simulation has no long-term drift. Don't read this as
  "streaming is worse" in general.
- Precision is low for every detector (27 to 38%, counting each detected fault as one hit). The surges were
  designed as hard traps. Real use would add domain rules or a second-stage classifier.
- This is one simulated fleet and one seed; differences of a few points are within noise.

## Extend
- Multiple seeds with confidence intervals; sweep the threshold quantile to draw a lead-time vs false-alarm curve
- Add ADWIN drift detection (`river.drift`) and a drifting-baseline scenario where streaming should shine
- Second-stage classifier or a rule to suppress known load surges
- Real data: NASA C-MAPSS or CWRU bearing data (needs `machine_id`, `t`, sensor columns)
