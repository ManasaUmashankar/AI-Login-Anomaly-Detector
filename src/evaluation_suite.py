"""Run synthetic-data audits and detector comparisons into reproducible reports."""
from pathlib import Path
import pandas as pd

from src.audit_label_alignment import audit_label_alignment
from src.evaluate_model import evaluate_seed_sweep
from src.read_dataset import load_logs
from src.stress_test_evaluation import evaluate_scenario_detection, evaluate_stress_test

METRICS = ("precision", "recall", "f1", "false_positive_rate", "accuracy")
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def _metric_rows(scope, detector_metrics):
    rows = []
    for detector, metrics in detector_metrics.items():
        for metric in METRICS:
            if metric in metrics:
                rows.append({
                    "evaluation_scope": scope, "detector": detector, "metric": metric,
                    "mean": metrics[metric], "std": None,
                    "tp": metrics.get("tp"), "fp": metrics.get("fp"),
                    "fn": metrics.get("fn"), "tn": metrics.get("tn),
                })
    return rows


def build_overview_rows(seed_result, stress_summary, scenario_metrics):
    """Flatten temporal, event-level, and scenario-level results into one table."""
    rows = []
    for key, detector in (("metrics", "Isolation Forest"), ("baseline_metrics", "Rule baseline")):
        for metric in METRICS:
            values = seed_result["summary"][key][metric]
            rows.append({
                "evaluation_scope": "temporal_multi_seed_window_mean",
                "detector": detector, "metric": metric,
                "mean": values["mean"], "std": values["std"],
                "tp": None, "fp": None, "fn": None, "tn": None,
            })
    for scope, metrics_by_name, names in (
        ("controlled_stress_event_level", stress_summary,
         {"ml": "Isolation Forest", "rules": "Rule baseline", "hybrid": "Hybrid (ML OR rules)"}),
        ("controlled_stress_scenario_level", scenario_metrics,
         {"ml": "Isolation Forest", "rule": "Rule baseline", "hybrid": "Hybrid (ML OR rules)"}),
    ):
        for key, detector in names.items():
            metrics = metrics_by_name[key]
            for metric in METRICS:
                rows.append({
                    "evaluation_scope": scope, "detector": detector, "metric": metric,
                    "mean": metrics[metric], "std": None,
                    "tp": metrics["tp"], "fp": metrics["fp"],
                    "fn": metrics["fn"], "tn": metrics["tn"],
                })
    return rows


def _markdown_table(frame, columns):
    selected = frame[columns].copy().fillna("")
    headers = [str(column) for column in columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in selected.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(format(value, ".4f"))
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_markdown_report(path, df, audit, overview):
    counts = df["true_label"].astype(str).str.strip().str.lower().value_counts().to_dict()
    temporal = overview[overview["evaluation_scope"] == "temporal_multi_seed_window_mean"]
    event = overview[overview["evaluation_scope"] == "controlled_stress_event_level"]
    scenario = overview[overview["evaluation_scope"] == "controlled_stress_scenario_level"]
    audit_frame = pd.DataFrame(audit["detectors"])
    content = [
        "# Synthetic Login Detector Evaluation Report", "",
        "> Synthetic-data engineering diagnostics only; not a real-world benchmark.", "",
        "## Dataset and label scope", "",
        "- Rows supplied: " + str(len(df)),
        "- Ground-truth label counts: " + str(counts),
        "- Rows in label/rule alignment audit: " + str(audit["rows_audited"]),
        "- The label audit measures agreement with synthetic labels, not whether labels represent real attacker behaviour.", "",
        "## Synthetic label / rule alignment", "",
        _markdown_table(audit_frame, ["detector", "precision", "recall", "f1", "false_positive_rate", "accuracy", "tp", "fp", "fn", "tn"]), "",
        "High agreement can indicate that the generator encodes signals used by the rules. Treat it as an overlap diagnostic, not an independent test.", "",
        "## Chronological multi-seed evaluation", "",
        "Mean and sample standard deviation are computed across seed-level averages of chronological holdout-window metrics. Seed variation on one dataset is not uncertainty across datasets.", "",
        _markdown_table(temporal, ["detector", "metric", "mean", "std"]), "",
        "## Controlled edge-case stress test", "",
        "The stress test is hand-designed. Event-level metrics score clearly labelled events only; ambiguous events are reported separately. Scenario-level metrics count each labelled scenario once.", "",
        "### Event-level metrics", "",
        _markdown_table(event, ["detector", "metric", "mean", "tp", "fp", "fn", "tn"]), "",
        "### Scenario-level metrics", "",
        _markdown_table(scenario, ["detector", "metric", "mean", "tp", "fp", "fn", "tn"]), "",
        "## Interpretation and limitations", "",
        "- The hybrid alerts if either ML or rules alerts; recall may rise while false positives also rise.",
        "- The rule baseline may score unusually well when synthetic attack labels are generated from signals the rules detect.",
        "- Small hand-designed scenarios expose blind spots but do not represent production prevalence, noise, or attack diversity.",
        "- Do not advertise these scores as real-world accuracy or SOC effectiveness.",
        "- Next step: use a separate, independently labelled dataset or documented anonymised authentication logs, with a time-based split and no tuning on the final test set.", "",
        "## Generated artifacts", "",
        "- label_alignment_detectors.csv", "- risk_score_distribution.csv",
        "- multi_seed_summary.csv", "- temporal_window_metrics.csv",
        "- stress_event_metrics.csv", "- stress_event_results.csv",
        "- stress_scenario_metrics.csv", "- stress_scenario_results.csv",
        "- evaluation_summary.csv", "- evaluation_report.md", "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def run_evaluation_suite(df=None, seeds=range(20), train_fraction=0.40, windows=4, report_dir=None):
    """Run all audits/evaluations and write machine-readable CSV plus a Markdown report."""
    if df is None:
        df = load_logs(include_labels=True)
    if "true_label" not in df.columns:
        raise ValueError("Dataset must contain true_label for the evaluation suite.")

    out_dir = Path(report_dir) if report_dir is not None else DEFAULT_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_label_alignment(df)
    audit_frame = pd.DataFrame(audit["detectors"])
    audit_frame.to_csv(out_dir / "label_alignment_detectors.csv", index=False)
    pd.DataFrame(audit["risk_score_distribution"]).to_csv(out_dir / "risk_score_distribution.csv", index=False)

    seed_result = evaluate_seed_sweep(df, seeds=seeds, train_fraction=train_fraction, windows=windows)
    seed_rows, temporal_rows = [], []
    for key, detector in (("metrics", "Isolation Forest"), ("baseline_metrics", "Rule baseline")):
        for metric in METRICS:
            values = seed_result["summary"][key][metric]
            seed_rows.append({"detector": detector, "metric": metric, "mean": values["mean"], "std": values["std"]})
        for seed, window_results in seed_result["runs"].items():
            for window in window_results:
                metric_key = key
                values = window[metric_key]
                temporal_rows.append({
                    "seed": seed, "window": window["window"], "detector": detector,
                    "train_rows": window["metadata"]["train_rows"],
                    "test_rows": window["metadata"]["test_rows"],
                    "actual_anomalies": window["metadata"]["actual_anomalies"],
                    "predicted_anomalies": window["metadata"]["predicted_anomalies"] if key == "metrics" else None,
                    **{metric: values[metric] for metric in METRICS},
                    **{name: values[name] for name in ("tp", "fp", "fn", "tn)},
                })
    pd.DataFrame(seed_rows).to_csv(out_dir / "multi_seed_summary.csv", index=False)
    pd.DataFrame(temporal_rows).to_csv(out_dir / "temporal_window_metrics.csv", index=False)

    stress_summary, event_rows = evaluate_stress_test(df)
    scenario_metrics, scenario_rows = evaluate_scenario_detection(event_rows)
    event_rows.to_csv(out_dir / "stress_event_results.csv", index=False)
    scenario_rows.to_csv(out_dir / "stress_scenario_results.csv", index=False)

    event_metric_rows = _metric_rows("event_level", {
        "Isolation Forest": stress_summary["ml"], "Rule baseline": stress_summary["rules"],
        "Hybrid (ML OR rules)": stress_summary["hybrid"],
    })
    scenario_metric_rows = _metric_rows("scenario_level", {
        "Isolation Forest": scenario_metrics["ml"], "Rule baseline": scenario_metrics["rule"],
        "Hybrid (ML OR rules)": scenario_metrics["hybrid"],
    })
    pd.DataFrame(event_metric_rows).to_csv(out_dir / "stress_event_metrics.csv", index=False)
    pd.DataFrame(scenario_metric_rows).to_csv(out_dir / "stress_scenario_metrics.csv", index=False)

    overview = pd.DataFrame(build_overview_rows(seed_result, stress_summary, scenario_metrics))
    overview.to_csv(out_dir / "evaluation_summary.csv", index=False)
    _write_markdown_report(out_dir / "evaluation_report.md", df, audit, overview)

    print("Evaluation suite completed. All results are synthetic-data diagnostics, not a real-world benchmark.")
    print("Reports written to: " + str(out_dir.resolve()))
    print("\nLabel counts:", df["true_label"].astype(str).str.strip().str.lower().value_counts().to_dict())
    print("\nTemporal multi-seed summary:")
    print(pd.DataFrame(seed_rows).to_string(index=False))
    print("\nControlled stress event metrics:")
    print(pd.DataFrame(event_metric_rows)[["detector", "metric", "mean", "tp", "fp", "fn", "tn"]].to_string(index=False))
    print("\nControlled stress scenario metrics:")
    print(pd.DataFrame(scenario_metric_rows)[["detector", "metric", "mean", "tp", "fp", "fn", "tn"]].to_string(index=False))
    print("\nLabel / rule alignment:")
    print(audit_frame[["detector", "precision", "recall", "f1", "false_positive_rate", "accuracy"]].to_string(index=False))
    print("\nReview reports/evaluation_report.md before updating headline project metrics.")
    return {"audit": audit, "seed_result": seed_result, "stress_summary": stress_summary,
            "scenario_metrics": scenario_metrics, "overview": overview, "report_dir": out_dir}


def main():
    run_evaluation_suite()


if __name__ == "__main__":
    main()
