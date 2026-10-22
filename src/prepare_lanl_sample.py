"""Prepare a bounded LANL user-computer authentication sample for exploratory scoring.

The LANL source has timestamp-offset,user,computer columns and successful events only.
The fixed timestamp origin below is an arbitrary ordering anchor, not the collection date.
No labels, IPs, geolocation, or failed logins are invented.
"""
import argparse
from pathlib import Path
import pandas as pd

ORIGIN = pd.Timestamp("2000-01-01 00:00:00")

def prepare(input_path, output_path, rows=100000):
    if rows < 1: raise ValueError("rows must be positive")
    source = pd.read_csv(input_path, header=None, names=["time_offset", "username", "computer"], compression="infer", nrows=rows)
    source["time_offset"] = pd.to_numeric(source["time_offset"], errors="coerce")
    source = source.dropna(subset=["time_offset", "username", "computer"]).copy()
    source["timestamp"] = (ORIGIN + pd.to_timedelta(source["time_offset"], unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S")
    out = pd.DataFrame({
        "timestamp": source["timestamp"], "username": source["username"].astype(str),
        "ip_address": "unknown", "location": "unknown", "device": source["computer"].astype(str),
        "login_status": "success", "user_agent": "unknown",
        "session_id": [f"lanl-{i}" for i in range(len(source))],
        "latitude": float("nan"), "longitude": float("nan"), "scenario_id": "",
    })
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    split = max(1, int(len(out) * 0.7))
    history_path = output_path.with_name(output_path.stem + "_history.csv")
    holdout_path = output_path.with_name(output_path.stem + "_holdout.csv")
    out.iloc[:split].to_csv(history_path, index=False)
    out.iloc[split:].to_csv(holdout_path, index=False)
    out.to_csv(output_path, index=False)
    print(f"Prepared {len(out)} unlabeled successful authentication events: {output_path}")
    print(f"Chronological history: {history_path} ({split} rows)")
    print(f"Chronological holdout: {holdout_path} ({len(out) - split} rows)")
    print("This supports exploratory anomaly scoring only; the dataset has no attack labels and cannot validate attack-detection metrics.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the LANL .bz2 file or extracted CSV.")
    parser.add_argument("--output", default="data/lanl_auth_sample.csv")
    parser.add_argument("--rows", type=int, default=100000)
    args = parser.parse_args()
    prepare(args.input, args.output, args.rows)

if __name__ == "__main__": main()
