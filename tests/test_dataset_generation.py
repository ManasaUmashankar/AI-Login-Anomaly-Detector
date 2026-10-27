import random

from src import generate_dataset


def test_normal_generator_includes_benign_context_variation():
    random.seed(123)
    rows = generate_dataset.make_normal_rows(500)

    assert any(row["device"] in {"iPhone", "Linux Laptop"} for row in rows)
    assert all(row["true_label"] == "normal" for row in rows)


def test_attack_generator_can_reuse_a_users_familiar_context():
    random.seed(123)
    rows = []
    names = list(generate_dataset.USERS)

    for _ in range(100):
        rows.extend(
            generate_dataset.make_brute_force_rows(random.choice(names), False)
        )

    assert any(
        row["ip_address"] == generate_dataset.USERS[row["username"]]["ip"]
        and row["device"] == generate_dataset.USERS[row["username"]]["device"]
        for row in rows
    )
    assert all(row["true_label"] == "anomaly" for row in rows)
