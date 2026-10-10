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
from src.detection_rules import apply_risk_scores, apply_rules, risk_level_for
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
    })
    outlier = pd.DataFrame([{
        "login_hour": 3, "failed_attempts": 12, "is_new_ip": 1, "is_new_device": 1,
        "is_unusual_hour": 1, "is_unusual_location": 1, "login_frequency": 13,
        "location_change": 1,
    }])
    return pd.concat([normal, outlier], ignore_index=True)


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the whole real pipeline once and share the result with several tests."""
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
    assert last["rules_triggered"] == 4         # (not at an unusual hour, so no rule 3)


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


def test_brute_force_success_scores_12_and_high():
    result = analyse(baseline_rows() + brute_force_rows())
    last = result.iloc[-1]
    assert last["risk_score"] == 12             # six signals x 2 points, daytime
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