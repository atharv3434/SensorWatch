import numpy as np
from sensorwatch.data import simulate, CAL
from sensorwatch.features import build_features, FEATURES
from sensorwatch.evaluate import alarms_from_scores, evaluate
from sensorwatch.detectors import score_zscore, score_iforest


def test_simulation_shapes_and_fault_timing():
    r, f = simulate(6, 720, seed=1)
    assert r.height == 6 * 720 and f.height >= 6
    assert f["fail_t"].min() >= 400 and f["fail_t"].max() < 720


def test_features_complete_and_healthy_period_normalised():
    r, _ = simulate(4, 720, seed=2)
    df = build_features(r)
    assert df.select(FEATURES).null_count().sum_horizontal()[0] == 0
    healthy = df.filter(df["t"] < CAL)
    assert abs(healthy["z_vibration"].mean()) < 0.05 and abs(healthy["z_vibration"].std() - 1) < 0.05


def test_alarm_logic_needs_k_consecutive_and_respects_cooldown():
    s = np.zeros(400); s[350] = 5                       # single spike -> no alarm
    assert alarms_from_scores(s, 1.0, k=3, start=336) == []
    s[360:380] = 5                                      # sustained -> exactly one alarm (cooldown)
    assert alarms_from_scores(s, 1.0, k=3, start=336) == [362]


def test_perfect_scorer_detects_all_with_no_false_alarms():
    r, f = simulate(5, 720, seed=4)
    sc = np.zeros((5, 720)) + 0.1
    for row in f.iter_rows(named=True):
        sc[row["machine_id"], row["fail_t"] - 30: row["fail_t"]] = 10
    m = evaluate(sc, f)
    assert m["detection_rate"] == 1.0 and m["false_alarms"] == 0


def test_detectors_return_matrix_and_detect_something():
    r, f = simulate(6, 720, seed=5)
    df = build_features(r)
    for fn in (score_zscore, score_iforest):
        sc = fn(df)
        assert sc.shape == (6, 720) and np.isfinite(sc).all()
        assert evaluate(sc, f)["detection_rate"] > 0
