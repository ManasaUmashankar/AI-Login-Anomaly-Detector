# AI Login Anomaly Detector

An educational **Blue Team** project that analyses synthetic login logs and flags *potentially suspicious* authentication behaviour using explainable rules, risk scoring, and machine learning (Isolation Forest).

> **Important:** an anomaly is a signal to investigate, not proof of an attack. This project uses synthetic data only and is not a production security tool.

**Author:** Manasa Umashankar |  **Repository:** https://github.com/ManasaUmashankar/AI-Login-Anomaly-Detector

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Cybersecurity Relevance](#cybersecurity-relevance)
5. [Features](#features)
6. [Architecture](#architecture)
7. [Technologies Used](#technologies-used)
8. [Dataset](#dataset)
9. [Detection Methodology](#detection-methodology)
10. [Project Structure](#project-structure)
11. [Installation](#installation)
12. [Usage](#usage)
13. [Dashboard](#dashboard)
14. [Dashboard Screenshots](#dashboard-screenshots)
15. [Example Output](#example-output)
16. [Model Evaluation](#model-evaluation)
17. [Project Documentation](#project-documentation)
18. [Testing](#testing)
19. [Limitations](#limitations)
20. [Future Improvements](#future-improvements)
21. [Blue Team / SOC Relevance](#blue-team--soc-relevance)
22. [Disclaimer](#disclaimer)
23. [Synthetic Label / Rule Alignment Audit](#synthetic-label--rule-alignment-audit)
24. [Controlled Edge-Case Stress Test](#controlled-edge-case-stress-test)

---

## Project Overview

The AI Login Anomaly Detector reads authentication (login) logs and answers one question: **which logins look unusual enough to deserve an analyst's attention?**

It combines two approaches:

- **Explainable rules and a risk score**, which say *why* a login was flagged.
- **An Isolation Forest model**, which finds logins that are statistically unusual, even when no rule was written for them.

The results are shown in a Streamlit dashboard with metrics, charts, filters, and a table of suspicious logins.

## Problem Statement

Organisations generate thousands of login events every day. Attacks such as password guessing hide inside that volume, and analysts cannot read every line. A monitoring system must highlight the small number of events that matter, **explain why**, and **rank them by urgency**, while avoiding false claims such as "the account is hacked".

## Objectives

- Build a complete log-analysis pipeline in Python, from raw logs to a dashboard.
- Clean data **without deleting unusual records**, because unusual records may be the evidence.
- Create meaningful features from raw log fields.
- Implement explainable detection rules and a project-specific risk score.
- Apply unsupervised machine learning (Isolation Forest) for anomaly detection.
- Combine rules, scoring, and ML into one final result.
- Test the project and document its limitations honestly.

## Cybersecurity Relevance

Authentication logs are one of the most important data sources in security monitoring. Common warning signs include:

- Many failed logins in a short time (possible password guessing or brute-force behaviour)
- Failures followed by a success (the guessing may have worked)
- Logins from new IP addresses, new devices, or unusual locations
- Logins at unusual hours or in rapid bursts

Each of these is a **signal**, not proof. New IPs, new devices, VPNs, and travel can all be legitimate.

## Features

- Synthetic login dataset with normal and suspicious behaviour
- Data preprocessing and validation (missing values, duplicates, timestamps, inconsistent values)
- Feature engineering (hour, unusual hour, new IP, new device, unusual location, location change, login frequency, rolling failed/successful login windows)
- Explainable point-in-time rules plus sequence-aware detection for repeated failures and rapid successful-login bursts
- Project-specific risk score with Low, Medium, and High levels
- Isolation Forest anomaly detection with a saved model
- Combined final status for every login
- Interactive Streamlit dashboard with filters
- Automated tests with pytest

## Architecture

```text
data/login_logs.csv
        |
        v
  Preprocessing  ->  Feature engineering
                            |
              +-------------+--------------+
              |                            |
              v                            v
   Rules + Risk scoring             Isolation Forest
   (explainable)                    (behavioural anomaly)
              |                            |
              +-------------+--------------+
                            v
               Combined result (final status)
                            |
                            v
                  Streamlit dashboard
```

## Technologies Used

| Purpose | Tool |
|---|---|
| Language | Python 3 |
| Data analysis | Pandas, NumPy |
| Machine learning | scikit-learn (Isolation Forest) |
| Model saving | Joblib |
| Dashboard and charts | Streamlit, Plotly |
| Testing | pytest |
| Version control | Git and GitHub |

## Dataset

The dataset is **synthetic**. No real passwords, credentials, or personal data are used, and the IP addresses come from ranges reserved for documentation (`203.0.113.x`, `198.51.100.x`, `192.0.2.x`).

`src/generate_dataset.py` creates `data/login_logs.csv` with **1,588 records** over 30 days for four users in the current seeded generation: **1,500 normal rows and 88 anomaly-labelled rows** (about 5.5% anomalies). The fixed random seed makes the generated dataset reproducible. Re-run the generator after changing its scenarios or counts, and treat the resulting row/label counts as version-specific.

| Column | Meaning |
|---|---|
| `timestamp` | Date and time of the login attempt |
| `username` | Account used |
| `ip_address` | Source IP address |
| `location` | Approximate location of the IP |
| `device` | Device type |
| `login_status` | `success` or `failed` |
| `failed_attempts` | Number of failures in the current sequence |
| `user_agent` | Browser or program identifier |
| `session_id` | Unique ID for the attempt |

**Normal behaviour (1,500 records):** mostly familiar context and working-hour logins, with occasional failed attempts. A small share of benign logins deliberately use a changed IP/device context to ensure novelty alone does not define the label.

**Anomaly-labelled behaviour (88 records in the current seeded generation):** brute-force-style sequences, unusual-hour successful logins, and rapid-login bursts. Some anomalous scenarios deliberately reuse familiar IP/device/location context, so a single categorical field should not be treated as proof of an attack. Exact scenario row counts vary with the seeded random sequence and generator version.

## Detection Methodology

1. **Preprocessing:** fix types, clean text, remove exact duplicates and unreadable rows. Unusual but valid rows are always kept.
2. **Feature engineering:** turn raw fields into signals such as "is this IP different from the user's usual IP?".
3. **Rules:** check each login against eight explainable rules.
4. **Risk scoring:** add up points for each signal to get a score and a level.
5. **Machine learning:** Isolation Forest scores how unusual each login is.
6. **Combination:** merge the risk level and the ML result into one final status.

### Rule-based detection

| Rule | Condition (project thresholds) |
|---|---|
| Multiple failed login attempts | 5 or more failures in a row |
| Successful login after repeated failures | A success after 3 or more failures |
| New IP address at an unusual hour | IP differs from the user's usual, and the hour is before 06:00 or from 22:00 |
| New device from an unusual location | Device and location both differ from the user's usual |
| Rapid login attempts | 5 or more attempts by one user within 5 minutes |
| Repeated failed logins within 45 minutes | 5 or more failed events in the trailing 45-minute window |
| Rapid successful login burst | 5 or more successful events in the trailing 5-minute window |
| Failed logins against multiple accounts from one source | Failures against 4 or more distinct accounts from one IP within 15 minutes |

Each rule that fires adds a readable reason to the row's `detection_reasons`. "Usual" means the most common IP, device, and location in earlier login history only. Events with the same timestamp share one baseline, and a user's first observed timestamp has no historical baseline yet.

### Risk scoring

| Signal | Points |
|---|---|
| Multiple failed login attempts | +2 |
| Successful login after repeated failures | +2 |
| New IP address | +2 |
| New device | +2 |
| Unusual login location | +2 |
| Rapid login attempts | +2 |
| Unusual login hour | +1 |
| Repeated failures within 45 minutes | +3 |
| Rapid successful logins within 5 minutes | +3 |
| Multiple accounts failed from one source within 15 minutes | +3 |

| Total score | Level |
|---|---|
| 0 to 2 | Low |
| 3 to 5 | Medium |
| 6 and above | High |

The highest possible score is 22. The rolling-window signals overlap intentionally with existing signals; they add sequence context rather than representing independent evidence.

> **These points and levels are project-specific design choices, not cybersecurity standards.** They exist to rank alerts. SOC analysts have limited time, and not every alert has the same severity, so scoring helps them look at the most serious events first.

### Machine learning

The project uses **unsupervised learning**: the model is trained without labels, because in real environments labelled attacks are rare and attack patterns keep changing. The planted suspicious rows are used only afterwards to sanity-check results.

The model sees ten numeric features: `login_hour`, `failed_attempts`, `is_new_ip`, `is_new_device`, `is_unusual_hour`, `is_unusual_location`, `login_frequency`, `location_change`, `failed_logins_45m`, and `successful_logins_5m`. The last two expose recent failure and success bursts directly to the model, rather than leaving those patterns to the rule layer alone. They are project-specific rolling windows, not proof of malicious intent.

The trained model is saved with Joblib to `models/anomaly_model_v3.pkl`, so the dashboard can load it instead of retraining every time. The artifact includes a feature-version marker; the pipeline retrains an unversioned or outdated model automatically.

> `.pkl` (pickle) files can run code when loaded. Only load model files from sources you trust.

### Isolation Forest

Isolation Forest builds many random decision trees. Each tree repeatedly picks a **random feature** and a **random cut point** to split the data. A login that is very different from the others is separated after **few cuts** (a short path), while a typical login needs **many cuts**. Shorter average paths mean more anomalous.

Settings used: 100 trees, `contamination = 0.05` (our guess that about 5% of rows are anomalous), and a fixed random seed. Rows with a `decision_function` score below 0 are labelled `Anomalous`.

The model does **not understand hacking**. It only finds numerically unusual patterns in the features it is given. That is why the number of anomalies is largely set by the `contamination` guess.

### Combining rules, scoring, and ML

| Risk level | ML result | Final status |
|---|---|---|
| High | Any | Requires investigation |
| Medium | Anomalous | Requires investigation |
| Medium | Normal | Potentially suspicious |
| Low | Anomalous | Potentially suspicious |
| Low | Normal | Normal activity |

Rules are explainable but only catch patterns we thought of. The model can flag unusual rows no rule covers, but it cannot explain itself. Combining them gives a more useful prototype than either alone. It is **not** claimed to be production-grade.

## Project Structure

```text
AI-Login-Anomaly-Detector/
├── data/
│   └── login_logs.csv
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py       # creates the synthetic logs
│   ├── read_dataset.py           # loads and explores logs
│   ├── preprocessing.py          # cleaning and validation
│   ├── feature_engineering.py    # point-in-time behavioural features
│   ├── detection_rules.py        # rules, risk scoring, signal explanations
│   ├── anomaly_detector.py       # Isolation Forest and combined pipeline
│   ├── evaluate_model.py         # temporal and multi-seed evaluation
│   ├── robust_evaluation.py      # robustness and prediction-overlap experiment
│   ├── audit_label_alignment.py  # audits synthetic label/rule overlap
│   ├── stress_test_evaluation.py # controlled edge-case stress evaluation
│   ├── evaluation_suite.py       # combined synthetic evaluation reports
│   ├── real_world_evaluation.py  # external independently labelled CSV harness
│   ├── model_comparison.py       # chronological unsupervised model comparison
│   ├── openssh_ingest.py         # OpenSSH auth.log parsing
│   ├── api.py                    # FastAPI inference endpoints
│   ├── api_security.py           # API key and request-rate controls
│   ├── siem_export.py            # optional Splunk/Elasticsearch forwarding
│   ├── mitre_mapping.py          # behavioural references to ATT&CK techniques
│   └── utils.py                  # shared file paths
├── models/
│   └── anomaly_model_v3.pkl
├── dashboard/
│   └── app.py
├── tests/
│   ├── test_api.py
│   ├── test_api_security.py
│   ├── test_dataset_generation.py
│   ├── test_detector.py
│   ├── test_mitre_mapping.py
│   ├── test_model_comparison.py
│   ├── test_openssh_ingest.py
│   ├── test_real_world_evaluation.py
│   ├── test_siem_export.py
│   └── test_stress_test_evaluation.py
├── Dockerfile
├── docker-compose.yml
├── Caddyfile
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Requires Python 3.10 or newer and Git.

```powershell
git clone https://github.com/ManasaUmashankar/AI-Login-Anomaly-Detector.git
cd AI-Login-Anomaly-Detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate` instead.

## Usage

Run these from the project folder with the virtual environment active:

```powershell
python -m src.generate_dataset             # 1. create the synthetic logs
python -m src.read_dataset                 # 2. (optional) explore the data
python -m src.anomaly_detector             # 3. train, save the model, run the full pipeline
python -m src.evaluate_model                # 4. run 20-seed chronological evaluation
python -m src.stress_test_evaluation        # 5. stress-test difficult synthetic edge cases
python -m src.evaluation_suite              # 6. write combined CSV and Markdown reports
python -m streamlit run dashboard/app.py   # 7. open the dashboard
```

Each stage can also be run on its own, for example `python -m src.preprocessing`, `python -m src.feature_engineering`, or `python -m src.detection_rules`.

## Dashboard

After starting Streamlit, open `http://localhost:8501`.

The dashboard is designed as a small SOC analyst workspace. It includes:

- **Overview metrics:** login-event volume, failed logins, ML outliers, high-severity events, and events requiring review.
- **Activity and risk visualisations:** login activity over time and a breakdown of risk levels.
- **Alert filtering:** use the available filters to focus on events relevant to an investigation.
- **Alert queue:** review suspicious events with risk scores, severity, and disposition/status information.
- **Investigation context:** inspect a selected event, its detection reasons, and suggested analyst checks.
- **Pipeline refresh control:** request refreshed pipeline results from the dashboard.

Exact labels and counts can change with the dataset and pipeline output. The dashboard is a local prototype; it does not ingest live authentication events or send production alerts.

## Dashboard Screenshots

The screenshot files referenced in an earlier version of this README are not currently present in the repository. To avoid broken image links or implying that screenshots have been verified, this section will be populated after fresh screenshots are captured from the current Streamlit dashboard.

Recommended screenshots to add under `docs/images/`:

- `dashboard-overview.jpg` — overview metrics and authentication trends
- `alert-queue.jpg` — alert queue with risk scores and severity
- `severity-filter.jpg` — the alert table with the relevant severity filter visibly applied
- `alert-investigation.jpg` — selected event details and detection reasons

Only include screenshots that actually match the described view. Use synthetic data and avoid exposing credentials, tokens, or personal information.

## Example Output

Exact counts depend on the dataset and the current feature/model version. The pipeline prints final-status counts, a risk-level breakdown, and the highest-priority events. Run `python -m src.evaluate_model` to calculate precision, recall, and F1 against the synthetic dataset's `true_label` column.

An illustrative alert row may look like this (values are an example, not a performance benchmark):

```text
2026-09-03 22:22:24 | diana | 198.51.100.23 | Sao Paulo, Brazil | Unknown Device
score 13 (High) | ML: Anomalous | status: Requires investigation
reasons: Multiple failed login attempts; Successful login after repeated failures;
New IP address at an unusual hour; New device from an unusual location;
Rapid login attempts; Statistically unusual pattern (ML)
```

The dataset contains planted synthetic scenarios. Performance on those scenarios is useful for testing the code, but it does not predict performance on real authentication logs.

## Model Evaluation

Run `python -m src.evaluate_model` to perform a **20-seed temporal evaluation** across four chronological holdout windows per seed. Each seed keeps the same time-ordered splits but varies Isolation Forest's random state. The report summarizes mean and sample standard deviation of each metric across seed-level averages, compares Isolation Forest with the rule-based risk-score baseline (score >= 3), and prints detailed window metrics for seed 0 (the first seed in the default sweep). This measures sensitivity to model randomness; it does not measure uncertainty across different datasets or real environments.

The separate `src/robust_evaluation.py` experiment also includes one-field category-prior baselines. These are **supervised baselines**, not unsupervised detectors: each estimates category anomaly rates from labelled training rows only, then applies those rates to later holdout rows. They are now named `supervised_category_prior_*` to make that distinction explicit. They must not be described as independent unsupervised anomaly detectors or compared without noting their access to training labels.

Run the robustness experiment with `python -m src.robust_evaluation --seeds 20 --windows 4`. It writes per-window/per-seed results to `reports/robust-evaluation.csv`, the metric summary to `reports/robust-evaluation-summary.csv`, and explicit hybrid/ML/rule prediction-overlap diagnostics to `reports/robust-evaluation-overlap.csv` and `reports/robust-evaluation-overlap-summary.csv`. The overlap report counts rule-only alerts added by the hybrid, ML-only alerts, shared alerts, and windows where both prediction arrays are identical. This distinguishes genuinely redundant rule alerts from coincidentally similar aggregate metrics.

The model is fitted only on rows earlier than each holdout window; ground-truth labels are used only to score predictions. Features are built chronologically so later events do not define earlier user baselines. The per-window mean is an unweighted average of window metrics, not a pooled metric. Some windows may contain very few planted anomalies, so interpret each metric alongside its support and confusion matrix. All results describe this synthetic dataset and are not a real-world benchmark.

## Historical Evaluation Snapshot (2026-10-27)

The following figures are retained as a historical record only. They were produced from an earlier 1,579-row dataset (1,500 normal; 79 anomaly-labelled rows) and do not describe the currently restored 1,588-row dataset. Do not use these values as the current project metrics. Regenerate and review `reports/evaluation_report.md` and the robustness reports before publishing a new snapshot.

| Metric | Isolation Forest | Rule baseline |
|---|---:|---:|
| Mean F1-score | 0.7782 | 0.4813 |
| Mean precision | 0.7421 | 0.3596 |
| Mean recall | 0.8952 | 0.9306 |
| Mean false-positive rate | 0.0078 | 0.0641 |
| Mean accuracy | 0.9873 | 0.9335 |

Isolation Forest F1 standard deviation was **0.0240**; the rule baseline is deterministic on this dataset. These are averages across seed-level chronological-window metrics, not pooled metrics. Seed variation on one generated dataset does not measure uncertainty across different datasets or real environments.

In the controlled edge-case stress test, the hybrid (ML OR rules) reached event-level **F1 0.7027 / recall 0.5417** (13 TP, 11 FN, 0 FP); Isolation Forest reached **F1 0.6286 / recall 0.4583** (11 TP, 13 FN, 0 FP); and the rule baseline reached **F1 0.5455 / recall 0.3750** (9 TP, 15 FN, 0 FP). There were 24 clearly anomaly-labelled events and 10 clearly benign-labelled events; 17 ambiguous events were assessed separately and excluded from binary event metrics.

At scenario level, the rule baseline and hybrid detected all six labelled anomaly scenarios and raised no alerts on the five benign scenarios; Isolation Forest detected four of six anomaly scenarios. A scenario counts as detected if at least one event in it triggers an alert, so scenario-level recall can be higher than event-level recall.

The label/rule alignment audit reported **F1 0.5892** for the combined risk-score threshold (score >= 3), with precision **0.4383**, recall **0.8987**, and false-positive rate **0.0607**. The password-spray signal had no true positives in the primary dataset, so that pattern still needs stronger coverage in the main generator; it is separately represented in the controlled stress set.

**Interpretation:** these are engineering diagnostics on synthetic, deliberately designed data—not a real-world benchmark. The updated dataset includes more overlap between benign and anomaly contexts, but it remains small and generated by known rules. The results should not be presented as production detection performance. Review `reports/evaluation_report.md` after each run and preserve the run's dataset version and metrics together.
## Project Documentation

- [Architecture and component flow](docs/architecture.md)
- [Evaluation guide and metric interpretation](docs/evaluation-guide.md)
- [Demo walkthrough](docs/demo-walkthrough.md)
- [SOC Analyst interview notes](docs/interview-notes.md)
- [Latest verified synthetic evaluation snapshot](docs/latest-synthetic-evaluation.md)

## Testing

```powershell
python -m pytest -v
```

The tests use small hand-made logs with known correct answers, plus checks on the full pipeline. They cover dataset loading, preprocessing, feature generation, detection rules, risk scoring, ML output, normal cases, and suspicious cases.

## Combined Evaluation Suite

Run `python -m src.evaluation_suite` to execute the label/rule alignment audit, 20-seed chronological evaluation, and controlled stress test in one reproducible command. It compares Isolation Forest, the rule baseline, and a diagnostic hybrid on the stress cases, with both event-level and scenario-level metrics. Ambiguous stress cases remain separately reported rather than being silently treated as benign or malicious.

The command writes CSV artifacts and `reports/evaluation_report.md` under `reports/`. `temporal_window_metrics.csv` preserves seed, window, train/test row counts, chronological boundaries, anomaly support, predictions, and confusion counts. `multi_seed_summary.csv` records seed count, split settings, and aggregation. Inspect label alignment for overlap between synthetic label-generation patterns and detector rules. The report does not overwrite headline metrics in this README; update published snapshots only after reviewing what each metric measures. All scores are synthetic-data diagnostics, not real-world benchmarks.

## Automated Evaluation (GitHub Actions)

The `.github/workflows/synthetic-evaluation.yml` workflow runs the full pytest suite and regenerates the consolidated synthetic evaluation when relevant project files change in a pull request. It can also be started manually from the repository's **Actions** tab using **Synthetic evaluation report**.

The generated `reports/` directory is uploaded as a workflow artifact named `synthetic-evaluation-report` and is retained by GitHub Actions for 30 days. Download that artifact to inspect the report and CSV files from the exact CI run. The workflow does not commit generated reports or publish performance numbers automatically; review the dataset, run configuration, and limitations before quoting any metric.

## Limitations

| Limitation | Notes |
|---|---|
| Synthetic dataset | The planted attacks are easy to separate from normal data, so results look better than real-world results would. |
| Limited features | Only a handful of simple signals are used. Real systems use many more. |
| Sequence windows | The 45-minute failure and 5-minute success windows are project-specific choices and need tuning against broader datasets. |
| False positives | VPNs, travel, new phones, and late work can all look suspicious. |
| False negatives | A slow attacker ("low and slow") can stay below the thresholds. The model can miss real attacks too. |
| VPN and proxy effects | They can hide or fake the IP and location. |
| Shared devices | They create unusual patterns for legitimate reasons. |
| IP and location accuracy | IP geolocation can be wrong. |
| Changing user behaviour and model drift | Habits change over time, so the baseline and model need regular updates. |
| Project-specific thresholds | Scores, levels, and thresholds are design choices, not standards. |
| Guessed `contamination` | The share of anomalies is assumed, not discovered. |
| Cumulative historical baseline | Future events no longer influence past features, but repeated suspicious events can eventually shift the most-common value; production systems need trusted or rolling baselines. |
| Generated failure counter | `failed_attempts` is recorded by our generator. Real logs require the system to count recent failures itself. |
| `location_change` is noisy | It also fires when a user returns to their normal location, so it is not used as a rule on its own. |
| No live authentication source | The project does not connect to an organisation's identity provider or production-scale log stream. OpenSSH parsing and a prediction API are provided, but they are not a managed ingestion pipeline. |

A production system would improve these with real log ingestion, per-user baselines built from history, many more features, tuned thresholds, analyst feedback, and regular model retraining.

## Future Improvements

- Broader authentication-log ingestion beyond the included OpenSSH parser
- End-to-end validation against test Splunk and Elasticsearch instances
- Real-time monitoring
- Email or chat alert notifications
- Tune sequence windows against diverse benign and malicious scenarios
- Validate cross-account password-spraying correlation against more benign shared-network cases
- Better behavioural baselines per user
- More advanced anomaly detection methods
- Scheduled model retraining
- Cloud security log integration
- Authentication risk scoring on live logins
- Role-based dashboards
- Incident response integration

## Blue Team / SOC Relevance

This project is a small model of a real monitoring workflow:

```text
Authentication logs -> Log collection -> Preprocessing / normalisation
-> Detection rules -> Behavioural anomaly detection
-> Risk / alert prioritisation -> Investigation -> Incident response
```

| Project stage | SOC equivalent |
|---|---|
| Synthetic login logs | Authentication logs from servers and identity systems |
| Preprocessing | Parsing and normalising logs |
| Rules | SIEM detection and correlation rules |
| Isolation Forest | Behavioural analytics (often called UEBA) |
| Risk score and final status | Alert severity and triage |
| Dashboard | Analyst monitoring console |

It is **much simpler** than a real SIEM such as Splunk. It does not collect live logs, has no alert workflow, and is **not** a complete SIEM.

## Disclaimer

This is an educational cybersecurity project focused on **defensive security**: detection, monitoring, and analysis. It uses synthetic data only. It does not contain instructions for attacking real systems, stealing credentials, or bypassing authentication. Results must not be treated as proof of malicious activity, and the project is not suitable for production use.

## Synthetic Label / Rule Alignment Audit

Run the audit from the repository root:

```bash
python -m src.audit_label_alignment
```

This report compares each individual risk signal and the combined risk-score baseline against `true_label`, including precision, recall, F1, false-positive rate, accuracy, and anomaly rate by risk score.

**Why this matters:** the dataset generator creates anomalies through specific scenarios (repeated failures, unusual-hour logins from unfamiliar locations/devices, and rapid-login bursts). Several of the detector rules target those same scenario characteristics. Strong rule performance on this dataset may therefore reflect how its labels were constructed, rather than independent evidence that the rules will generalise to real authentication logs.

Treat the report as a **synthetic-data diagnostic**, not a real-world benchmark. A stronger future evaluation should include independently designed scenarios, benign edge cases that resemble suspicious activity, and—when available—appropriately anonymised real or public authentication data. Do not tune the rules to achieve a perfect score on the same synthetic labels and then present that score as proof of production effectiveness.


## Controlled Edge-Case Stress Test

Run a second, deliberately difficult evaluation separately from the primary synthetic dataset:

```bash
python -m src.stress_test_evaluation
```

The stress test fits Isolation Forest on the historical synthetic logs only, then scores 62 controlled events across 17 scenarios: 21 clearly benign-labelled events across seven scenarios, 24 suspicious-labelled events across six scenarios, and 17 ambiguous events across four scenarios. It includes a cross-account password-spraying pattern (one source IP failing against four accounts) and a separate ambiguous shared-network pattern. Clearly benign cases include a new IP, new device, new location, or unusual login hour in isolation; six successful logins spaced six minutes apart; five mistyped-password failures followed by a successful login; and five familiar-context successful logins spaced 45 seconds apart. The last two are hard negatives intended to test whether ordinary user behaviour is over-alerted. Suspicious cases include brute-force-style failures from familiar context, rapid successful logins from familiar context, low-and-slow failures, daytime logins involving a new IP/location or new device/location, and failures against multiple accounts from one source IP. Four overlapping scenarios—rapid successful logins, low-and-slow failures, a new device plus location during daytime, and failures against multiple accounts from a shared network—are marked ambiguous rather than assumed benign or malicious.

Feature engineering also counts distinct accounts with failed logins from the same source IP in an inclusive 15-minute window. Four distinct accounts is a project-specific trigger threshold, not a universal security standard. This signal can help expose password spraying, but shared office/NAT/VPN egress can produce similar patterns; therefore, an alert is evidence for investigation, not proof of an attack. The script reports event-level and scenario-level precision, recall, F1, false-positive rate, accuracy, and confusion counts using only clearly labelled normal/anomaly cases. Ambiguous events and scenarios are excluded from binary metrics and listed separately with each detector's alert outcome; they are not silently counted as false positives or true positives. It also reports the event position of the first alert in each scenario, so you can compare detection delay and identify completely missed sequences. It compares three approaches: Isolation Forest alone, the risk-score baseline (score >= 3), and a diagnostic hybrid that alerts when either ML or the rules flag an event. The hybrid is a layered stress-test comparison; it does not replace the dashboard's existing final-status logic. Combining detectors can improve recall but may also increase false positives, so compare both metrics rather than assuming the hybrid is automatically better. The run writes event-level outcomes, scenario-level outcomes, and a focused false-positive/false-negative CSV under `reports/` (`stress-test-event-outcomes.csv`, `stress-test-scenario-outcomes.csv`, and `stress-test-errors.csv`). The error audit includes the detector, error type, scenario, event context, and relevant rolling-window/risk diagnostics. The tests cover ambiguous-label exclusion, hard-negative scenario construction, error-audit classification, scenario detection, first-alert positions, and hybrid prediction logic.

**Important evaluation caveat:** the stress set is small and hand-designed, and its labels are assumptions for testing, not objective truth. It is intended to expose blind spots and help guide further test design—not to establish real-world detection performance. It is separate from the original dataset and does not change the baseline labels or its reported results.

## Hybrid Threshold Comparison

The robust evaluation now compares the existing automatic-contamination Isolation Forest baseline with the rule alerts, and also evaluates two diagnostic OR-hybrids using Isolation Forest contamination settings of 0.03 and 0.05. Each comparison uses the same chronological test windows. The overlap CSV records how many alerts are shared, ML-only, or rule-only; the summary groups those counts by comparison and reports the mean and standard deviation across seeds.

The candidate methods are named `hybrid_contamination_0.03_or_rules` and `hybrid_contamination_0.05_or_rules` in the main evaluation CSV. Compare their precision, recall, and F1 against the corresponding ML-only and rules-only methods. A lower contamination setting does not guarantee better detection: it changes the model's alert volume, and the OR combination can increase false positives. Keep the current detector and dashboard logic unchanged until results are reviewed. These are synthetic-data experiments, not production performance claims.


## REST API (FastAPI)

The optional API scores a submitted login event against the local historical CSV and returns a risk level, Isolation Forest result, anomaly score, final disposition, and human-readable detection reasons.

Install dependencies and start the service:

```bash
uvicorn src.api:app --reload
```

- Health: `GET /health`
- Model readiness: `GET /ready`
- Score one event: `POST /predict`
- Interactive API documentation: `/docs`

Example request:

```json
{
  "timestamp": "2026-11-10T10:00:00",
  "username": "alice",
  "ip_address": "203.0.113.25",
  "location": "Bengaluru, India",
  "device": "Windows Laptop",
  "login_status": "failed",
  "failed_attempts": 1
}
```

The service uses the repository's CSV as its historical context. It is a local prototype, not a multi-user production service: deploy with a persistent event store, authenticated access, tenant isolation, rate limiting, and operational monitoring before exposing it to a network.

### Docker

Build and start the API:

```bash
docker compose up --build
```

The container exposes port 8000 and mounts the local data and model folders read-only. Visit `http://localhost:8000/docs` for the interactive API docs.


### Detection explanations and MITRE ATT&CK

The API includes conservative ATT&CK references for a small set of specific authentication patterns:

- Repeated failed attempts: possible alignment with **T1110.001 – Password Guessing**.
- Failures against multiple accounts from one source: possible alignment with **T1110.003 – Password Spraying**.

These are investigation hints, not confirmed technique attribution. Generic novelty signals and Isolation Forest anomalies are deliberately left unmapped because they are not specific enough.

The API also emits structured request metadata (method, path, status, duration) without writing usernames, IP addresses, or request bodies to its access log.

**Deployment boundary:** Docker Compose binds the API to `127.0.0.1` by default. Do not expose this prototype directly to the internet. A network deployment needs authentication, TLS, rate limiting, tenant isolation, a persistent event store, and monitoring.


### Unsupervised model comparison

Run the chronological model comparison:

```bash
python -m src.model_comparison
```

This compares **Isolation Forest**, **Local Outlier Factor (novelty mode)**, and **One-Class SVM** across future holdout windows. Each model is trained only on earlier rows; labels are used only after prediction to calculate metrics. The output is written to `reports/model_comparison.csv`.

Do not select a model from accuracy alone. Compare recall, false-positive rate, precision, F1, anomaly support, and consistency across windows. All three models are still being evaluated on the same synthetic generator, so this comparison does not establish real-world generalisation.


### OpenSSH authentication-log ingestion

The project can now parse explicit SSH authentication events from Linux `auth.log` files into the same canonical schema used by the detector. It recognizes accepted and failed password/public-key/keyboard-interactive events, counts consecutive failures per account and source IP, and skips unrelated daemon messages.

Example:

```python
from src.openssh_ingest import load_openssh_auth_log
from src.preprocessing import preprocess_logs
from src.feature_engineering import build_features

events = load_openssh_auth_log("/var/log/auth.log", year=2026)
clean = preprocess_logs(events, verbose=False)
features = build_features(clean)
print(features[["timestamp", "username", "ip_address", "login_status"]].tail())
```

The parser deliberately sets location and device to `unknown`; it does not invent geolocation or endpoint identity. Syslog timestamps omit the year, so pass the correct year for archived files. This is an ingestion adapter, not a claim that the saved synthetic-data model is calibrated for your server. For meaningful ML scoring, build a representative benign historical baseline from the target environment and evaluate it separately before deployment.

### API authentication, rate limiting and SIEM forwarding

The prediction endpoint now requires an API key in the `X-API-Key` header. If `DETECTOR_API_KEY` is not configured, prediction requests fail closed with HTTP 503. A per-client in-memory sliding-window limiter returns HTTP 429 when the configured threshold is reached. The limiter is per process and keys off the immediate TCP peer. When all requests pass through one reverse proxy, they can share that proxy's bucket; use a trusted gateway that applies client-aware rate limiting or a shared Redis-backed limiter for multi-worker/multi-instance deployments.

Optional event forwarding supports:

- **Splunk HEC:** `SPLUNK_HEC_URL` and `SPLUNK_HEC_TOKEN`
- **Elasticsearch:** `ELASTICSEARCH_URL`, `ELASTICSEARCH_API_KEY`, and optional `ELASTICSEARCH_INDEX`

Forwarding is disabled when credentials are absent. TLS certificate verification remains enabled. Delivery failures are logged without logging secrets or event payloads, and do not turn a successful detection into a failed prediction response. For reliable delivery at scale, use a durable queue/outbox and retry worker; this prototype makes bounded synchronous HTTP attempts.

Copy `.env.example` to `.env`, generate a long random API key, and fill only the SIEM settings you actually use. Never commit `.env`. For local-only use, start the API with `docker compose up --build -d detector-api`.

For public TLS termination with Caddy, configure `DOMAIN` and run:

```bash
docker compose --profile tls up --build -d
```

Caddy obtains and renews certificates automatically when DNS points to this host and inbound ports 80/443 are reachable. Keep the API service's host port bound to loopback. The API key is still required behind the proxy.

### Evaluation on independently labelled logs

The external evaluation harness accepts a CSV containing the canonical login columns plus an independently reviewed `true_label` column (`normal` or `anomaly`). It uses a chronological split, trains Isolation Forest on the earlier portion of that same dataset, and reports later-holdout metrics for ML, rules and hybrid detection.

```bash
python -m src.real_world_evaluation /path/to/labelled_auth_logs.csv \
  --label-provenance "Analyst-reviewed incident tickets and confirmed benign events"
```

The command does not download or upload your data. Do not use labels generated by the same synthetic generator as evidence of real-world performance. The model comparison is meaningful only when labels are independent, the source environment is representative, and the holdout contains enough confirmed attack and benign examples.
