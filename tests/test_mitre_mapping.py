from src.mitre_mapping import map_detection_reasons


def test_password_spray_reason_maps_to_password_spraying():
    mapped = map_detection_reasons([
        "Failed logins against multiple accounts from one source",
    ])
    assert mapped == [{
        "technique_id": "T1110.003",
        "technique_name": "Password Spraying",
        "tactic": "Credential Access",
        "reference_type": "possible_behavioural_alignment",
    }]


def test_generic_novelty_and_ml_reasons_are_not_overmapped():
    mapped = map_detection_reasons([
        "New IP address at an unusual hour",
        "Statistically unusual pattern (ML)",
    ])
    assert mapped == []


def test_duplicate_reasons_return_one_technique():
    mapped = map_detection_reasons([
        "Multiple failed login attempts",
        "Repeated failed logins within 45 minutes",
    ])
    assert len(mapped) == 1
    assert mapped[0]["technique_id"] == "T1110.001"
