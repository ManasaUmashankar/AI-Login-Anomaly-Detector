import pandas as pd

from src.detection_rules import apply_risk_scores, apply_rules


def test_every_positive_risk_score_has_an_explanation():
    rows = pd.DataFrame(
        {
            "failed_attempts": [0, 0, 0],
            "login_status": ["success", "success", "failed"],
            "is_new_ip": [1, 0, 0],
            "is_new_device": [0, 1, 0],
            "is_unusual_hour": [0, 0, 1],
            "is_unusual_location": [0, 0, 0],
            "login_frequency": [1, 1, 1],
            "failed_logins_45m": [0, 0, 0],
            "successful_logins_5m": [1, 1, 0],
            "distinct_failed_users_per_ip_15m": [0, 0, 0],
        }
    )

    result = apply_risk_scores(apply_rules(rows))
    positive = result["risk_score"] > 0

    assert positive.any()
    assert result.loc[positive, "detection_reasons"].str.strip().ne("").all()
