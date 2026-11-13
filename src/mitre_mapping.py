"""Conservative MITRE ATT&CK references for detection-rule explanations.

These mappings describe behavioural similarity, not proof that an ATT&CK
technique occurred. Contextual novelty and generic ML anomalies are not mapped
to techniques because they are not sufficiently specific.
"""

TECHNIQUE_BY_REASON = {
    "Multiple failed login attempts": {
        "technique_id": "T1110.001",
        "technique_name": "Password Guessing",
        "tactic": "Credential Access",
    },
    "Successful login after repeated failures": {
        "technique_id": "T1110.001",
        "technique_name": "Password Guessing",
        "tactic": "Credential Access",
    },
    "Repeated failed logins within 45 minutes": {
        "technique_id": "T1110.001",
        "technique_name": "Password Guessing",
        "tactic": "Credential Access",
    },
    "Failed logins against multiple accounts from one source": {
        "technique_id": "T1110.003",
        "technique_name": "Password Spraying",
        "tactic": "Credential Access",
    },
}


def map_detection_reasons(reasons):
    """Return unique, conservative ATT&CK mappings for known detection reasons."""
    mapped = []
    seen = set()
    for reason in reasons:
        technique = TECHNIQUE_BY_REASON.get(reason)
        if technique is None or technique["technique_id"] in seen:
            continue
        mapped.append({
            **technique,
            "reference_type": "possible_behavioural_alignment",
        })
        seen.add(technique["technique_id"])
    return mapped
