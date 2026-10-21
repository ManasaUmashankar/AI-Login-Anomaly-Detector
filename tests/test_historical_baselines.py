import pandas as pd

from src.feature_engineering import build_features
from src.preprocessing import REQUIRED_COLUMNS


USUAL = ("203.0.113.11", "Mumbai, India", "Windows Laptop")
ODD = ("198.51.100.23", "Moscow, Russia", "Unknown Device")
DIFFERENCE_FLAGS = ["is_new_ip", "is_new_device", "is_unusual_location"]


def make_row(timestamp, place=USUAL, username="alice"):
    ip, location, device = place
    return [
        timestamp,
        username,
        ip,
        location,
        device,
        "success",
        0,
        "test-agent",
        f"{username}-{timestamp}",
    ]


def make_logs(rows):
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def test_future_events_cannot_change_historical_baselines():
    initial_rows = [
        make_row(f"2026-09-{day:02d} 10:00:00") for day in range(1, 11)
    ]
    initial_rows.append(make_row("2026-09-20 03:00:00", ODD))

    future_rows = [
        make_row(f"2026-10-{day:02d} 03:00:00", ODD) for day in range(1, 21)
    ]

    initial = build_features(make_logs(initial_rows))
    extended = build_features(make_logs(initial_rows + future_rows))

    # Ordinary historical logins remain ordinary even when later data changes
    # the overall dataset's most common IP, device, and location.
    for column in DIFFERENCE_FLAGS:
        assert initial.iloc[:10][column].tolist() == [0] * 10
        assert extended.iloc[:10][column].tolist() == [0] * 10
        assert initial.iloc[10][column] == 1
        assert extended.iloc[10][column] == 1


def test_same_timestamp_events_do_not_set_baselines_for_each_other():
    timestamp = "2026-09-21 10:00:00"
    rows = [
        make_row(timestamp, USUAL, username="new-user"),
        make_row(timestamp, ODD, username="new-user"),
    ]

    features = build_features(make_logs(rows))

    # Neither row has earlier history, so row ordering at the same timestamp
    # must not make one of the two values look established or anomalous.
    for column in DIFFERENCE_FLAGS:
        assert features[column].tolist() == [0, 0]



def test_simultaneous_events_have_order_independent_frequency_and_location_features():
    prior = make_row("2026-09-20 10:00:00", USUAL, username="alice")
    simultaneous = [
        make_row("2026-09-20 10:01:00", USUAL, username="alice"),
        make_row("2026-09-20 10:01:00", ODD, username="alice"),
    ]

    forward = build_features(make_logs([prior] + simultaneous))
    reverse = build_features(make_logs([prior] + list(reversed(simultaneous))))

    def by_event(frame):
        return {
            (row["timestamp"], row["ip_address"]): (
                row["login_frequency"], row["location_change"]
            )
            for _, row in frame.iterrows()
        }

    assert by_event(forward) == by_event(reverse)
    assert forward.loc[forward["timestamp"] == pd.Timestamp("2026-09-20 10:01:00"), "login_frequency"].tolist() == [3, 3]


def test_tied_historical_baselines_are_independent_of_input_row_order():
    timestamp = "2026-09-22 10:00:00"
    later_timestamp = "2026-09-22 10:01:00"
    tied_events = [
        make_row(timestamp, USUAL, username="alice"),
        make_row(timestamp, ODD, username="alice"),
    ]
    later_event = make_row(later_timestamp, USUAL, username="alice")

    forward = build_features(make_logs(tied_events + [later_event]))
    reverse = build_features(make_logs(list(reversed(tied_events)) + [later_event]))

    columns = DIFFERENCE_FLAGS + ["location_change"]
    def later_features(frame):
        row = frame.loc[frame["timestamp"] == pd.Timestamp(later_timestamp)].iloc[0]
        return tuple(int(row[column]) for column in columns)

    assert later_features(forward) == later_features(reverse)

