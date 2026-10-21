# Evaluation Guide

This guide explains how to run and interpret the project's evaluation scripts. It deliberately avoids treating synthetic-data results as real-world security performance.

## Run the evaluations

Run from the repository root with the project's virtual environment active:

```powershell
python -m src.evaluate_model
python -m src.audit_label_alignment
python -m src.stress_test_evaluation
python -m pytest -v
```

- `src.evaluate_model` evaluates Isolation Forest across four chronological holdout windows and compares it with a rule/risk-score baseline.
- `src.audit_label_alignment` checks how individual signals and the combined risk-score baseline align with labels created by the synthetic data generator.
- `src.stress_test_evaluation` tests a separate set of controlled edge-case events, including benign, suspicious, and ambiguous scenarios.
- `pytest` runs the regression suite.

## What the metrics mean

| Metric | Interpretation |
|---|---|
| Precision | Of events flagged as anomalous, the fraction labelled anomalous in this evaluation set. |
| Recall | Of labelled anomalous events, the fraction detected. |
| F1-score | Harmonic mean of precision and recall. |
| False-positive rate | Fraction of labelled normal events incorrectly flagged. |
| Accuracy | Fraction of all scored events classified correctly; it can be misleading when classes are imbalanced. |
| Confusion matrix | Counts true negatives, false positives, false negatives, and true positives. |

Read precision and recall together. A detector that flags everything can have high recall but poor precision; a detector that flags very little can have high precision but miss many anomalies.

## Why chronological evaluation matters

The temporal evaluator fits a model on earlier events and evaluates later events. It expands the training history for each consecutive holdout window and keeps equal timestamps together at split boundaries. This is a better simulation of future-event scoring than randomly mixing past and future events.

The feature builder also compares historical baselines against strictly earlier timestamps, processing events with equal timestamps as a batch to reduce row-order-dependent leakage.

## Interpret results carefully

- The dataset is synthetic, and its suspicious examples were deliberately planted.
- The label-alignment audit is important because rules may overlap with the same characteristics used to create labels. High scores can reflect how the dataset was constructed.
- The controlled stress set is small and hand-designed. Ambiguous events are excluded from binary metrics rather than forced into normal/anomaly labels.
- A rule score is a project-specific triage heuristic, not a calibrated probability of attack.
- Isolation Forest detects unusual numeric patterns; it does not know whether an event is malicious.
- Do not report a single metric without naming the dataset, evaluation method, positive class, and limitations.

## Suggested report template

After running the scripts, record the actual output in a table such as this:

| Evaluation | Dataset | Precision | Recall | F1 | False-positive rate | Notes |
|---|---|---:|---:|---:|---:|---|
| Temporal window 1 | Synthetic chronological holdout | — | — | — | — | Copy actual output |
| Temporal window 2 | Synthetic chronological holdout | — | — | — | — | Copy actual output |
| Temporal window 3 | Synthetic chronological holdout | — | — | — | — | Copy actual output |
| Temporal window 4 | Synthetic chronological holdout | — | — | — | — | Copy actual output |
| Controlled stress test | 51 hand-designed events | — | — | — | — | Clearly labelled events only; ambiguous cases reported separately |

Replace the dashes with values from the current run. Do not reuse historical numbers after changing the dataset, feature logic, thresholds, or model artifact.

## Interview-ready explanation

> I evaluated the detector using chronological holdout windows rather than a random split, so each model was trained on earlier events and tested on later events. I compared Isolation Forest with an explainable risk-score baseline and added a separate controlled stress test for edge cases, including ambiguous shared-network patterns. Because the data and stress cases are synthetic and hand-designed, I treat the metrics as engineering diagnostics—not evidence of production detection accuracy.
