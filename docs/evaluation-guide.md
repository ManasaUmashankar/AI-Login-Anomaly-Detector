# Evaluation Guide

This guide explains how to run and interpret the project's evaluation scripts. It deliberately avoids treating synthetic-data results as real-world security performance.

## Run the evaluations

Run from the repository root with the project's virtual environment active:

```powershell
python -m src.evaluate_model
python -m src.audit_label_alignment
python -m src.stress_test_evaluation
python -m src.evaluation_suite
python -m pytest -v
```

- `src.evaluate_model` evaluates Isolation Forest across four chronological holdout windows and compares it with a rule/risk-score baseline. Its seed-sweep function repeats those chronological windows across 20 model seeds by default and reports mean and sample standard deviation across seed-level window averages.
- `src.audit_label_alignment` checks how individual signals and the combined risk-score baseline align with labels created by the synthetic data generator.
- `src.stress_test_evaluation` tests a separate set of controlled edge-case events, including benign, suspicious, and ambiguous scenarios.
- `src.evaluation_suite` runs the label-alignment audit, 20-seed chronological evaluation, controlled stress test, and writes consolidated CSV/Markdown reports under `reports/`.
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

## Latest verified CI run

The following results come from the GitHub Actions synthetic-evaluation workflow on 2026-11-14, commit `90d61523c773fb7fc3b96406a09a1dc7cd81d4a2` (PR #13 before merge). The workflow passed the test suite, built the API Docker image, generated the consolidated evaluation, validated its report outputs, and uploaded the reports. The dataset contained 1,588 synthetic events: 1,500 labelled normal and 88 labelled anomalous.

### Chronological multi-seed holdout

The evaluator used 20 Isolation Forest seeds and four chronological holdout windows per seed, with a 40% initial training fraction. Values below are unweighted window averages within each seed, then mean and sample standard deviation across seeds. This is sensitivity to model randomness on one synthetic dataset—not uncertainty across independent datasets.

| Detector | Metric | Mean | Std. dev. |
|---|---|---:|---:|
| Isolation Forest | Precision | 0.9906 | 0.0032 |
| Isolation Forest | Recall | 0.9071 | 0.0180 |
| Isolation Forest | F1 | 0.9376 | 0.0135 |
| Isolation Forest | False-positive rate | 0.0010 | 0.0004 |
| Isolation Forest | Accuracy | 0.9963 | 0.0006 |
| Rule baseline (risk score >= 3) | Precision | 0.9896 | 0.0000 |
| Rule baseline (risk score >= 3) | Recall | 1.0000 | 0.0000 |
| Rule baseline (risk score >= 3) | F1 | 0.9947 | 0.0000 |
| Rule baseline (risk score >= 3) | False-positive rate | 0.0012 | 0.0000 |
| Rule baseline (risk score >= 3) | Accuracy | 0.9990 | 0.0000 |

The rule baseline's unusually strong result must be interpreted alongside the label-alignment audit below: synthetic labels and detection rules share some of the same planted signals. The rule baseline is not independent evidence of real-world detection quality.

### Controlled edge-case stress test

The hand-designed stress test contains 62 events across 17 scenarios: 45 clearly labelled events and 17 ambiguous events. Ambiguous events are excluded from binary event-level metrics.

| Detector | Precision | Recall | F1 | False-positive rate | Accuracy |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 0.7000 | 0.2917 | 0.4118 | 0.1429 | 0.5556 |
| Rule baseline | 0.7500 | 0.3750 | 0.5000 | 0.1429 | 0.6000 |
| Hybrid (ML OR rules) | 0.7500 | 0.5000 | 0.6000 | 0.1905 | 0.6444 |

At scenario level, the rule baseline and hybrid each detected all six labelled attack scenarios, but each also flagged two of seven benign scenarios (scenario-level precision 0.7500 and false-positive rate 0.2857). Isolation Forest detected four of six attack scenarios and flagged one of seven benign scenarios. Scenario-level detection is not interchangeable with event-level recall; the hybrid's additional recall came with more false positives.

### Synthetic-label alignment audit

The combined risk score at threshold `>= 3` had precision 0.9355, recall 0.9886, F1 0.9613, false-positive rate 0.0040, and accuracy 0.9956 against the generator's labels. New-IP, new-device, and unusual-location signals each aligned strongly with those labels. The cross-account failed-login signal did not flag any of the 88 labelled anomalies in this audit.

**Interpretation warning:** the generator may encode the same signals used by the detector into its labels. High alignment can therefore reflect how the synthetic data was constructed, rather than independent evidence of real-world SOC effectiveness. The stress test also shows that event-level recall can be modest even when scenario-level detection is strong.

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
