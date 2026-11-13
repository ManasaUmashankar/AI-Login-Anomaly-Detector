# Latest Synthetic Evaluation Snapshot

**Run date:** 2026-11-13  
**GitHub Actions run:** [Synthetic evaluation report #43377124172](https://github.com/ManasaUmashankar/AI-Login-Anomaly-Detector/actions/runs/43377124172)  
**Commit:** `eaf14a594b88be37e712c96b09d194247554dcfb`  
**Dataset:** 1,588 synthetic login events (1,500 normal; 88 anomaly-labelled)  
**CI result:** 114 tests passed; Docker image build, consolidated evaluation, and artifact upload completed successfully. The test run emitted three warnings; none failed the workflow.

> These results are synthetic engineering diagnostics. They are not a real-world benchmark and do not establish production SOC effectiveness.

## Chronological multi-seed evaluation

The evaluation uses 20 random seeds and four chronological holdout windows. The values below are means of seed-level averages across windows; the standard deviation reflects seed variation on this one generated dataset, not variation across independent datasets.

| Detector | Precision (mean) | Recall (mean) | F1 (mean) | False-positive rate (mean) | Accuracy (mean) |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 0.9906 | 0.9071 | 0.9376 | 0.0010 | 0.9963 |
| Rule baseline | 0.9896 | 1.0000 | 0.9947 | 0.0012 | 0.9990 |

The rule baseline's high score must be interpreted in light of the generator: some planted labels are defined by behavioural patterns that the rules explicitly target. This is an overlap diagnostic, not independent evidence of generalisation.

## Controlled edge-case stress test

The hand-designed stress set reports event-level metrics only for clearly labelled events. Ambiguous events are excluded from binary metrics and assessed separately.

| Detector | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.7000 | 0.2917 | 0.4118 | 7 | 3 | 17 | 18 |
| Rule baseline | 0.7500 | 0.3750 | 0.5000 | 9 | 3 | 15 | 18 |
| Hybrid (ML OR rules) | 0.7500 | 0.5000 | 0.6000 | 12 | 4 | 12 | 17 |

Scenario-level results are different because a scenario is counted as detected if at least one event in it triggers an alert:

| Detector | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.8000 | 0.6667 | 0.7273 | 4 | 1 | 2 | 6 |
| Rule baseline | 0.7500 | 1.0000 | 0.8571 | 6 | 2 | 0 | 5 |
| Hybrid (ML OR rules) | 0.7500 | 1.0000 | 0.8571 | 6 | 2 | 0 | 5 |

The stress test contains **24 clearly anomaly-labelled events, 21 clearly benign-labelled events, and 17 ambiguous events**. The 45 clearly labelled events are the population used for the event-level binary metrics. The scenario-level recall should not be mistaken for event-level recall: catching one event in a sequence does not mean every suspicious event was caught.

## Label/rule alignment

On the primary synthetic dataset, the combined risk score threshold (score >= 3) had precision 0.9355, recall 0.9886, F1 0.9613, and false-positive rate 0.0040 (87 TP, 6 FP, 1 FN, 1,494 TN). Because the labels and rules share generator-defined patterns, this result is not an independent validation.

## Interpretation

- The controlled stress set exposes meaningful misses: the hybrid missed 12 of 24 clearly labelled suspicious events at event level and produced four false positives.
- The rule baseline performs very strongly on the generated chronological evaluation, but that is partly explained by overlap between generator scenarios and rule conditions.
- Event-level and scenario-level metrics answer different questions and must be reported separately.
- A representative, independently labelled authentication dataset is still required before making real-world performance claims.

## Reproduce

From the repository root:

```bash
python -m pytest -q
python -m src.evaluation_suite
```

The full CSV and Markdown outputs are retained in the GitHub Actions artifact named `synthetic-evaluation-report`. See the linked workflow run for the exact CI execution and artifact.
