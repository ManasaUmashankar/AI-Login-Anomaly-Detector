import pandas as pd
import pytest

from src.anomaly_detector import (
    CONTAMINATION,
    MODEL_FEATURES,
    apply_model,
    final_status_for,
    load_model,
    run_pipeline,
    save_model,
    train_model,
)
from src.detection_rules import RISK_SIGNALS, SIGNAL_REASONS, apply_risk_scores, apply_rules, risk_level_for
from src.feature_engineering import build_features
from src.preprocessing import REQUIRED_COLUMNS, demo_messy_data, preprocess_logs
from src.read_dataset import load_logs

# ---------- Helpers that build tiny, hand-made logs ----------
# A "place" is (IP, location, device)
USUAL = ("203.0.113.11", "Mumbai, India", "Windows Laptop")
ODD = ("198.51.100.23", "Moscow, Russia", "Unknown Device")
VPN_LIKE = ("198.51.100.99", "Frankfurt, Germany", "Windows Laptop")
FLAG_COLUMNS = ["is_new_ip", "is_new_device", "is_unusual_location", "is_unusual_hour"]


def make_row(time, place=USUAL, status="success", failed=0):
    ip, location, device = place
    return [time, "alice", ip, location, device, status, failed, "test-agent", "s1"]


def make_logs(rows):
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def baseline_rows():
    """10 ordinary 10:00 logins. They outnumber the odd rows, so 'usual' is clear."""
    return [make_row(f"2026-09-{day:02d} 10:00:00") for day in range(1, 11)]


def brute_force_rows():
    """6 fast failures from a strange place, then a success, at 14:00."""
    rows = [make_row(f"2026-09-15 14:00:{n:02d}", ODD, "failed", n) for n in range(1, 7)]
    rows.append(make_row("2026-09-15 14:00:07", ODD, "success", 6))
    return rows


def analyse(rows):
    """Run features, rules and risk scores on hand-made rows."""
    features = build_features(make_logs(rows))
    return apply_risk_scores(apply_rules(features))


def make_ml_frame():
    """200 very normal rows plus 1 extreme row (all numbers, for the model)."""
    normal = pd.DataFrame({
        "login_hour": [9 + i % 9 for i in range(200)],
        "failed_attempts": 0,
        "is_new_ip": 0,
        "is_new_device": 0,
        "is_unusual_hour": 0,
        "is_unusual_location": 0,
        "login_frequency": 1,
        "location_change": 0,
        "failed_logins_45m": 0,
        "successful_logins_5m": 1,
    })
    outlier = pd.DataFrame([{
        "login_hour": 3, "failed_attempts": 12, "is_new_ip": 1, "is_new_device": 1,
        "is_unusual_hour": 1, "is_unusual_location": 1, "login_frequency": 13,
        "location_change": 1, "failed_logins_45m": 12, "successful_logins_5m": 13,
    }])
    return pd.concat([normal, outlier], ignore_index=True)


@pytest.fixture
def pipeline_result(tmp_path, monkeypatch):
    """Run the pipeline with an isolated model artifact; never overwrite the user's model."""
    monkeypatch.setattr("src.anomaly_detector.MODEL_PATH", tmp_path / "pipeline_model.pkl")
    return run_pipeline()


# ---------- Dataset loading ----------
def test_load_logs_has_expected_columns_and_rows():
    df = load_logs()
    assert len(df) > 0
    assert list(df.columns) == REQUIRED_COLUMNS


def test_load_logs_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("src.read_dataset.DATA_PATH", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        load_logs()


# ---------- Preprocessing ----------
def test_preprocessing_fixes_messy_data():
    clean = preprocess_logs(demo_messy_data())
    assert len(clean) == 4                      # 7 rows in, 3 broken ones removed
    assert pd.api.types.is_datetime64_any_dtype(clean["timestamp"])
    assert clean["login_status"].isin(["success", "failed"]).all()
    assert clean["location"].notna().all()


def test_preprocessing_keeps_unusual_rows():
    clean = preprocess_logs(demo_messy_data())
    moscow = clean[clean["location"] == "Moscow, Russia"]
    assert len(moscow) == 1                     # unusual, but valid: must survive
    assert moscow.iloc[0]["failed_attempts"] == 12


# ---------- Feature engineering ----------
def test_features_flag_odd_login_but_not_normal_ones():
    features = build_features(make_logs(baseline_rows() + [make_row("2026-09-20 03:00:00", ODD)]))
    assert features.iloc[:10][FLAG_COLUMNS].to_numpy().sum() == 0
    assert (features.iloc[10][FLAG_COLUMNS] == 1).all()
    assert features.iloc[10]["login_hour"] == 3


def test_login_frequency_counts_recent_logins():
    rows = [
        make_row("2026-09-01 12:00:00"),
        make_row("2026-09-01 12:00:10"),
        make_row("2026-09-01 12:00:20"),
        make_row("2026-09-01 15:00:00"),
    ]
    features = build_features(make_logs(rows))
    assert list(features["login_frequency"]) == [1, 2, 3, 1]


def test_location_change_flags_the_switch():
    features = build_features(make_logs(baseline_rows() + [make_row("2026-09-20 03:00:00", ODD)]))
    assert list(features["location_change"]) == [0] * 10 + [1]


def test_rolling_login_status_windows_count_only_each_users_recent_events():
    rows = [
        make_row("2026-09-01 12:00:00", status="failed", failed=1),
        make_row("2026-09-01 12:07:00", status="failed", failed=2),
        make_row("2026-09-01 12:14:00", status="failed", failed=3),
        make_row("2026-09-01 12:21:00", status="failed", failed=4),
        make_row("2026-09-01 12:28:00", status="failed", failed=5),
        make_row("2026-09-01 12:35:00", status="failed", failed=6),
        make_row("2026-09-01 13:30:00", status="success"),
    ]
    features = build_features(make_logs(rows))
    assert list(features["failed_logins_45m"]) == [1, 2, 3, 4, 5, 6, 0]
    assert list(features["successful_logins_5m"]) == [0, 0, 0, 0, 0, 0, 1]


def test_rapid_success_burst_is_flagged_without_unfamiliar_context():
    rows = baseline_rows() + [
        make_row(f"2026-09-15 14:00:{second:02d}", status="success")
        for second in (1, 8, 15, 22, 29, 36)
    ]
    result = analyse(rows)
    burst = result.iloc[-1]
    assert burst["successful_logins_5m"] == 6
    assert "Rapid successful login burst" in burst["detection_reasons"]
    assert burst["risk_score"] >= 3

def test_cross_account_failure_window_detects_password_spray_pattern():
    rows = baseline_rows()
    for attempt, username in enumerate(("alice", "bob", "cara", "dan"), start=1):
        row = make_row(
            f"2026-09-15 14:00:{attempt * 8:02d}",
            status="failed",
            failed=1,
        )
        row[1] = username
        row[2] = "198.51.100.203"
        rows.append(row)

    result = analyse(rows)
    sequence = result.iloc[-4:]
    assert list(sequence["distinct_failed_users_per_ip_15m"]) == [1, 2, 3, 4]
    assert "Failed logins against multiple accounts from one source" in (
        sequence.iloc[-1]["detection_reasons"]
    )
    assert sequence.iloc[-1]["risk_score"] >= 3


def test_password_spray_rule_triggers_without_other_risk_signals():
    # Isolate the spray signal: no new IP/device/location, unusual hour, or burst.
    features = pd.DataFrame([{
        "failed_attempts": 1,
        "login_status": "failed",
        "is_new_ip": 0,
        "is_new_device": 0,
        "is_unusual_location": 0,
        "is_unusual_hour": 0,
        "login_frequency": 1,
        "failed_logins_45m": 1,
        "successful_logins_5m": 0,
        "distinct_failed_users_per_ip_15m": 4,
    }])

    ruled = apply_rules(features)
    scored = apply_risk_scores(features)

    assert ruled.iloc[0]["detection_reasons"] == (
        "Failed logins against multiple accounts from one source"
    )
    assert scored.iloc[0]["risk_score"] == 3
    assert scored.iloc[0]["risk_level"] == "Medium"


def test_cross_account_failure_window_does_not_mix_source_ips():
    rows = []
    for attempt, username in enumerate(("alice", "bob", "cara", "dan"), start=1):
        row = make_row(
            f"2026-09-15 14:00:{attempt * 8:02d}",
            status="failed",
            failed=1,
        )
        row[1] = username
        row[2] = f"198.51.100.{200 + attempt}"
        rows.append(row)

    features = build_features(make_logs(rows))
    assert list(features["distinct_failed_users_per_ip_15m"]) == [1, 1, 1, 1]


def test_low_and_slow_failures_trigger_before_the_sequence_ends():
    rows = baseline_rows() + [
        make_row(f"2026-09-15 14:{minute:02d}:00", status="failed", failed=attempt)
        for attempt, minute in enumerate((0, 7, 14, 21, 28, 35), start=1)
    ]
    result = analyse(rows)
    sequence = result.iloc[-6:]
    assert list(sequence["failed_logins_45m"]) == [1, 2, 3, 4, 5, 6]
    assert (sequence.iloc[-2:]["risk_score"] >= 3).all()
    assert sequence.iloc[-1]["risk_level"] == "Medium"


# ---------- Detection rules ----------
def test_normal_logins_trigger_no_rules():
    result = analyse(baseline_rows())
    assert (result["rules_triggered"] == 0).all()
    assert (result["detection_reasons"] == "").all()


def test_brute_force_then_success_triggers_expected_rules():
    result = analyse(baseline_rows() + brute_force_rows())
    last = result.iloc[-1]                      # the successful login
    assert last["login_status"] == "success"
    assert "Multiple failed login attempts" in last["detection_reasons"]
    assert "Successful login after repeated failures" in last["detection_reasons"]
    assert "Rapid login attempts" in last["detection_reasons"]
    assert last["rules_triggered"] == 5         # includes the 45-minute repeated-failure rule


# ---------- Risk scoring ----------
def test_normal_logins_score_zero_and_low():
    result = analyse(baseline_rows())
    assert (result["risk_score"] == 0).all()
    assert (result["risk_level"] == "Low").all()


def test_odd_night_login_scores_7_and_high():
    result = analyse(baseline_rows() + [make_row("2026-09-20 03:00:00", ODD)])
    last = result.iloc[-1]
    assert last["risk_score"] == 7              # new IP 2 + device 2 + location 2 + hour 1
    assert last["risk_level"] == "High"


def test_vpn_like_login_is_medium_not_high():
    result = analyse(baseline_rows() + [make_row("2026-09-20 14:00:00", VPN_LIKE)])
    last = result.iloc[-1]
    assert last["risk_score"] == 4              # new IP 2 + unusual location 2
    assert last["risk_level"] == "Medium"


def test_brute_force_success_scores_15_and_high():
    result = analyse(baseline_rows() + brute_force_rows())
    last = result.iloc[-1]
    assert last["risk_score"] == 15             # original six signals plus rolling repeated-failure signal
    assert last["risk_level"] == "High"


@pytest.mark.parametrize("score, expected", [
    (0, "Low"), (2, "Low"), (3, "Medium"), (5, "Medium"), (6, "High"), (13, "High"),
])
def test_risk_level_boundaries(score, expected):
    assert risk_level_for(score) == expected


# ---------- Combined status and ML ----------
@pytest.mark.parametrize("level, ml, expected", [
    ("High", "Anomalous", "Requires investigation"),
    ("High", "Normal", "Requires investigation"),
    ("Medium", "Anomalous", "Requires investigation"),
    ("Medium", "Normal", "Potentially suspicious"),
    ("Low", "Anomalous", "Potentially suspicious"),
    ("Low", "Normal", "Normal activity"),
])
def test_final_status_table(level, ml, expected):
    assert final_status_for(level, ml) == expected


def test_model_flags_an_obvious_outlier():
    frame = make_ml_frame()
    result = apply_model(train_model(frame), frame)
    assert set(result["ml_result"]) <= {"Anomalous", "Normal"}
    assert result.iloc[-1]["ml_result"] == "Anomalous"
    assert result.iloc[-1]["anomaly_score"] < 0
    assert (result.iloc[:200]["ml_result"] == "Normal").mean() > 0.9


def test_model_can_be_saved_and_loaded(tmp_path, monkeypatch):
    # Save into a temporary folder so the real model file is never touched
    monkeypatch.setattr("src.anomaly_detector.MODEL_PATH", tmp_path / "test_model.pkl")
    frame = make_ml_frame()
    model = train_model(frame)
    save_model(model)
    loaded = load_model()
    inputs = frame[MODEL_FEATURES]
    assert list(loaded.predict(inputs)) == list(model.predict(inputs))


# ---------- The whole real pipeline ----------
def test_pipeline_output_columns_and_row_count(pipeline_result):
    for column in ["risk_score", "risk_level", "ml_result", "final_status", "detection_reasons"]:
        assert column in pipeline_result.columns
    assert len(pipeline_result) == len(load_logs())


def test_pipeline_ml_share_matches_contamination(pipeline_result):
    share = (pipeline_result["ml_result"] == "Anomalous").mean()
    assert abs(share - CONTAMINATION) < 0.01


def test_pipeline_high_risk_rows_always_need_investigation(pipeline_result):
    high = pipeline_result[pipeline_result["risk_level"] == "High"]
    assert (high["final_status"] == "Requires investigation").all()


def test_pipeline_most_rows_are_normal_activity(pipeline_result):
    share = (pipeline_result["final_status"] == "Normal activity").mean()
    assert share > 0.9


def test_pipeline_top_risk_row_requires_investigation(pipeline_result):
    top = pipeline_result.loc[pipeline_result["risk_score"].idxmax()]
    assert top["final_status"] == "Requires investigation"

# ---------- Evaluation methodology ----------
def test_calculate_metrics_counts_confusion_matrix_and_false_positives():
    from src.evaluate_model import calculate_metrics

    actual = ["normal", "normal", "normal", "anomaly", "anomaly"]
    predicted = ["normal", "anomaly", "normal", "anomaly", "normal"]
    metrics = calculate_metrics(actual, predicted)

    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["precision"] == pytest.approx(1 / 2)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["f1"] == pytest.approx(1 / 2)


def test_calculate_metrics_rejects_mismatched_lengths():
    from src.evaluate_model import calculate_metrics

    with pytest.raises(ValueError, match="same length"):
        calculate_metrics(["normal"], ["normal", "anomaly"])


def test_calculate_metrics_rejects_empty_holdout():
    from src.evaluate_model import calculate_metrics

    with pytest.raises(ValueError, match="empty"):
        calculate_metrics([], [])



def test_multi_window_evaluation_uses_expanding_chronological_training(monkeypatch):
    from src.evaluate_model import evaluate_temporal_windows

    monkeypatch.setattr("src.evaluate_model.train_model", lambda frame: object())
    monkeypatch.setattr(
        "src.evaluate_model.apply_model",
        lambda model, frame: pd.DataFrame({"ml_result": ["Normal"] * len(frame)}),
    )

    rows = [
        make_row(f"2026-09-{day:02d} 10:00:00")
        for day in range(1, 13)
    ]
    frame = make_logs(rows)
    frame["true_label"] = ["normal", "anomaly"] * 6

    results = evaluate_temporal_windows(frame, train_fraction=0.50, windows=3)

    assert len(results) == 3
    assert [item["metadata"]["train_rows"] for item in results] == [6, 8, 10]
    assert [item["metadata"]["test_rows"] for item in results] == [2, 2, 2]
    assert all("baseline_metrics" in item for item in results)
    assert all(item["metrics"]["fn"] >= 0 for item in results)


def test_temporal_splits_never_separate_equal_timestamps(monkeypatch):
    from src.evaluate_model import evaluate_temporal_holdout, evaluate_temporal_windows

    monkeypatch.setattr("src.evaluate_model.train_model", lambda frame: object())
    monkeypatch.setattr(
        "src.evaluate_model.apply_model",
        lambda model, frame: pd.DataFrame({"ml_result": ["Normal"] * len(frame)}),
    )

    # Duplicate timestamp groups deliberately straddle the old row-based boundaries.
    timestamps = [
        "2026-09-01 10:00:00",
        "2026-09-02 10:00:00",
        "2026-09-03 10:00:00",
        "2026-09-03 10:00:00",
        "2026-09-04 10:00:00",
        "2026-09-05 10:00:00",
        "2026-09-05 10:00:00",
        "2026-09-06 10:00:00",
        "2026-09-07 10:00:00",
        "2026-09-08 10:00:00",
        "2026-09-08 10:00:00",
        "2026-09-09 10:00:00",
    ]
    frame = make_logs([make_row(timestamp) for timestamp in timestamps])
    frame["true_label"] = ["normal", "anomaly"] * 6

    _, holdout = evaluate_temporal_holdout(frame, train_fraction=0.60)
    assert holdout["train_end"] < holdout["test_start"]

    windows = evaluate_temporal_windows(frame, train_fraction=0.40, windows=3)
    assert len(windows) == 3
    for result in windows:
        metadata = result["metadata"]
        assert metadata["train_end"] < metadata["test_start"]
    for earlier, later in zip(windows, windows[1:]):
        assert earlier["metadata"]["test_end"] < later["metadata"]["test_start"]


def test_multi_window_evaluation_rejects_fewer_than_two_windows():
    from src.evaluate_model import evaluate_temporal_windows

    with pytest.raises(ValueError, match="at least 2"):
        evaluate_temporal_windows(pd.DataFrame(), windows=1)


# ---------- Explainability consistency ----------
def test_every_risk_signal_has_a_nonempty_human_readable_reason():
    signal_names = [name for name, _points, _function in RISK_SIGNALS]
    assert set(signal_names) == set(SIGNAL_REASONS)
    assert all(isinstance(SIGNAL_REASONS[name], str) and SIGNAL_REASONS[name].strip()
               for name in signal_names)


def test_signal_reasons_do_not_claim_an_attack_is_confirmed():
    forbidden_phrases = ("confirmed attack", "account is hacked", "definitely malicious")
    assert all(
        not any(phrase in reason.lower() for phrase in forbidden_phrases)
        for reason in SIGNAL_REASONS.values()
    )


# ---------- Multi-seed evaluation ----------
def test_seed_sweep_returns_mean_and_std_and_uses_requested_seeds(monkeypatch):
    from src.evaluate_model import evaluate_seed_sweep

    monkeypatch.setattr(
        "src.evaluate_model.train_model",
        lambda frame, random_state=42: object(),
    )
    monkeypatch.setattr(
        "src.evaluate_model.apply_model",
        lambda model, frame: pd.DataFrame({"ml_result": ["Normal"] * len(frame)}),
    )

    rows = [make_row(f"2026-09-{day:02d} 10:00:00") for day in range(1, 13)]
    frame = make_logs(rows)
    frame["true_label"] = ["normal", "anomaly"] * 6

    result = evaluate_seed_sweep(frame, seeds=[3, 7, 11], train_fraction=0.40, windows=3)

    assert result["seeds"] == [3, 7, 11]
    assert set(result["runs"]) == {3, 7, 11}
    assert result["summary"]["metrics"]["f1"]["mean"] == pytest.approx(0.0)
    assert result["summary"]["metrics"]["f1"]["std"] == pytest.approx(0.0)
    assert set(result["summary"]["baseline_metrics"]["f1"]) == {"mean", "std", "per_seed_mean"}


def test_seed_sweep_rejects_empty_or_duplicate_seeds():
    from src.evaluate_model import evaluate_seed_sweep

    with pytest.raises(ValueError, match="at least one|contain at least one"):
        evaluate_seed_sweep(pd.DataFrame(), seeds=[])
    with pytest.raises(ValueError, match="unique"):
        evaluate_seed_sweep(pd.DataFrame(), seeds=[4, 4])
