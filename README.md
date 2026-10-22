# AI Login Anomaly Detector

A defensive Blue Team prototype for triaging authentication events.
It combines explainable detection rules, risk scoring, and Isolation Forest.
The Streamlit dashboard helps an analyst inspect why an event was flagged.
Evaluation uses synthetic scenarios; results are diagnostic, not production evidence.

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

The primary dataset is synthetic and contains no real credentials or personal data. `src/generate_dataset.py` creates a reproducible dataset with 36 users over 60 days, benign VPN-like IP changes, new devices, travel, late-night work, typo bursts, and multiple attack scenarios. Each scenario has a `scenario_id`; latitude and longitude support impossible-travel experiments.

```powershell
python -m src.generate_dataset --seed 42
```

| Column | Meaning |
|---|---|
| `timestamp`, `username` | Event time and account |
| `ip_address`, `location`, `latitude`, `longitude` | Source network and coarse location context |
| `device`, `user_agent` | Client context |
| `login_status` | `success` or `failed` |
| `session_id` | Event identifier |
| `true_label` | Synthetic evaluation label; never a model input |
| `scenario_id` | Groups events belonging to a designed scenario |

Labels are synthetic assumptions, not independent ground truth. The generator is designed to reduce easy single-field shortcuts, but that must be verified by the evaluation report before claiming the baseline requirement is met.

## Detection Methodology

1. **Preprocessing:** fix types, clean text, remove exact duplicates and unreadable rows. Unusual but valid rows are always kept.
2. **Feature engineering:** turn raw fields into signals such as "is this IP different from the user's usual IP?".
3. **Rules:** check each login against explainable rules.
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
| Password spraying | Failures against 4 or more distinct accounts from one source IP within 15 minutes |
| Slow brute force | 8 or more failures accumulated over 6 hours |
| Impossible travel | Successful logins imply travel faster than 900 km/h between known coordinates |

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
| Repeated failures within six hours | +3 |
| Impossible travel between successful logins | +3 |

| Total score | Level |
|---|---|
| 0 to 2 | Low |
| 3 to 5 | Medium |
| 6 and above | High |

With the two added +3 signals (slow brute force and impossible travel), the current configured maximum is 28 points. The original ten signals alone sum to 22. The rolling-window signals overlap intentionally with existing signals; they add sequence context rather than representing independent evidence.

> **These points and levels are project-specific design choices, not cybersecurity standards.** They exist to rank alerts. SOC analysts have limited time, and not every alert has the same severity, so scoring helps them look at the most serious events first.

### Machine learning

The project uses **unsupervised learning**: the model is trained without labels, because in real environments labelled attacks are rare and attack patterns keep changing. The planted suspicious rows are used only afterwards to sanity-check results.

The model uses eleven numeric features: `login_hour`, `is_new_ip`, `is_new_device`, `failed_logins_6h`, `impossible_travel`, `is_unusual_hour`, `is_unusual_location`, `login_frequency`, `location_change`, `failed_logins_45m`, and `successful_logins_5m`. It does not use `true_label` or a generator-supplied `failed_attempts` counter. The rolling windows and travel estimate are project-specific signals, not proof of malicious intent.

The trained model is saved with Joblib to `models/anomaly_model_v4.pkl`, so the dashboard can load it instead of retraining every time. The artifact includes a feature-version marker; the pipeline retrains an unversioned or outdated model automatically.

> `.pkl` (pickle) files can run code when loaded. Only load model files from sources you trust.

### Isolation Forest

Isolation Forest builds many random decision trees. Each tree repeatedly picks a **random feature** and a **random cut point** to split the data. A login that is very different from the others is separated after **few cuts** (a short path), while a typical login needs **many cuts**. Shorter average paths mean more anomalous.

Settings used: 100 trees, `contamination = "auto"` (a threshold setting, not a prevalence estimate), and an explicit random seed for reproducibility. The robust evaluator also reports PR-AUC from continuous anomaly scores. Rows with a `decision_function` score below 0 are labelled `Anomalous`.

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
│   ├── generate_dataset.py      # creates the synthetic logs
│   ├── read_dataset.py          # loads and explores logs (Pandas)
│   ├── preprocessing.py         # cleaning and validation
│   ├── feature_engineering.py   # turns logs into features
│   ├── detection_rules.py       # rules and risk scoring
│   ├── anomaly_detector.py      # Isolation Forest and combined pipeline
│   ├── audit_label_alignment.py # audits rule/label overlap
│   ├── stress_test_evaluation.py # controlled edge-case stress evaluation
│   └── utils.py                 # shared file paths
├── models/
│   └── anomaly_model_v3.pkl
├── dashboard/
│   └── app.py
├── notebooks/                   # reserved for exploration notebooks
├── tests/
│   ├── test_detector.py
│   └── test_stress_test_evaluation.py
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
python -m src.evaluate_model                # 4. original chronological evaluation
python -m src.robust_evaluation --seeds 20   # 5. multi-seed baseline / PR-AUC / scenario report
python -m src.anomaly_detector --input data/new_events.csv  # score incoming events
# Optional: add --history path/to/history.csv to override the default history
python -m src.stress_test_evaluation        # 6. stress-test difficult synthetic edge cases
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

Capture the running dashboard and save the four screenshots at these exact paths before presenting the project: `docs/images/dashboard-overview.jpg`, `docs/images/alert-queue.jpg`, `docs/images/severity-filter.jpg`, and `docs/images/alert-investigation.jpg`. If these files are absent from a checkout, the screenshot links will not render.


The Streamlit dashboard provides a compact SOC-style workflow: review authentication trends, prioritize the alert queue, and inspect the context behind an individual event.

### Security overview

![AI Login Anomaly Detector dashboard overview](docs/images/dashboard-overview.jpg)

*Overview metrics and authentication telemetry from the synthetic login dataset.*

### Analyst alert queue

![Analyst alert queue with risk scores and severity](docs/images/alert-queue.jpg)

*Alerts are ordered by risk score so higher-priority events can be reviewed first.*

### Alert table detail

![Alert table showing event timestamps, accounts, source IPs, risk scores, and detection reasons](docs/images/severity-filter.jpg)

*The event table surfaces the fields an analyst can use to triage suspicious activity. This screenshot illustrates the queue; it should only be described as a severity-filter demo if the High-only filter was actually applied.*

### Event investigation

![Selected login event investigation showing risk score, severity, ML signal, event context, and detection reasons](docs/images/alert-investigation.jpg)

*Inspect the selected event's context and the reasons it was flagged. An alert is a lead for investigation, not proof of compromise.*

> Screenshots show a local prototype using synthetic data, not a live production SOC.

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

Run `python -m src.evaluate_model` to perform **four chronological holdout windows**. The model starts with the earliest 40% of valid login events, then evaluates consecutive later windows while expanding the training history. Each window reports precision, recall, F1-score, false-positive rate, accuracy, and a confusion matrix. The report also compares Isolation Forest with a simple explainable rule-based baseline that flags rows with a risk score of 3 or higher.

The model is fitted only on rows earlier than each holdout window; ground-truth labels are used only to score predictions. Features are built chronologically so later events do not define earlier user baselines. The per-window mean is an unweighted average of window metrics, not a pooled metric. Some windows may contain very few planted anomalies, so interpret each metric alongside its support and confusion matrix. All results describe this synthetic dataset and are not a real-world benchmark.

## Latest Evaluation Snapshot

Latest local run (2026-10-21) on synthetic login data: across four chronological holdout windows, Isolation Forest achieved an unweighted mean F1-score of **0.9583** and recall of **0.9286**; the rule-based risk-score baseline achieved mean F1 of **0.9947** and recall of **1.0000**. In the separate controlled stress test, the hybrid detector achieved event-level F1 of **0.667** and recall of **0.500** across 34 clearly labelled events; 17 ambiguous events were reviewed separately.

These are results on synthetic, deliberately designed data, not a real-world benchmark. The generator may encode signals also used by the detector, so high scores are not independent evidence of production effectiveness. See the [full evaluation guide](docs/evaluation-guide.md) for per-window metrics and interpretation.

## Project Documentation

- [Architecture and component flow](docs/architecture.md)
- [Evaluation guide and metric interpretation](docs/evaluation-guide.md)
- [Demo walkthrough](docs/demo-walkthrough.md)
- [SOC Analyst interview notes](docs/interview-notes.md)

## Testing

```powershell
python -m pytest -v
```

The tests use small hand-made logs with known correct answers, plus checks on the full pipeline. They cover dataset loading, preprocessing, feature generation, detection rules, risk scoring, ML output, normal cases, and suspicious cases.

## Limitations

| Limitation | Notes |
|---|---|
| Synthetic dataset | The original dataset is too easy: `device == "Unknown Device"` alone achieves F1 = 1.0 on the current labels. This is label leakage by construction, not evidence of useful detection. |
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
| No real infrastructure | There is no real authentication system and no production-scale log ingestion. |

A production system would improve these with real log ingestion, per-user baselines built from history, many more features, tuned thresholds, analyst feedback, and regular model retraining.

## Future Improvements

- Real authentication log ingestion
- Splunk or other SIEM integration
- Real-time monitoring
- Email or chat alert notifications
- Tune sequence windows against diverse benign and malicious scenarios
- Add cross-account password-spraying correlation while reviewing shared-network false positives
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

## MITRE ATT&CK Mapping

This is a conceptual mapping of defensive detections to ATT&CK techniques, not a claim of complete technique coverage:

| Detection | ATT&CK technique | Rationale and caveat |
|---|---|---|
| Repeated authentication failures, password guessing and slow brute force | [T1110 — Brute Force](https://attack.mitre.org/techniques/T1110/) | Failure patterns may be consistent with brute force; legitimate typos and shared egress can also trigger alerts. |
| Successful login from a new context or after repeated failures | [T1078 — Valid Accounts](https://attack.mitre.org/techniques/T1078/) | Unusual successful authentication may warrant investigation; this detector cannot prove credential compromise. |

Impossible travel and unusual device/location are behavioural analytics signals, not standalone proof of a specific ATT&CK technique.

## What I Found

The first synthetic dataset was too easy: the single field `device == "Unknown Device"` achieved F1 = 1.0 against its generated labels. That meant a trivial shortcut could appear perfect without learning meaningful authentication behaviour.

I rebuilt the generator to mix legitimate VPN-like IP changes, travel, new devices, late-night work and typo bursts with attacks that can use familiar devices or normal-looking IPs. The new design adds 36 users, 60 days, and scenario IDs. **The rebuild is a hypothesis, not a result:** run `python -m src.robust_evaluation --seeds 20` and inspect the single-field baselines before claiming the shortcut has been eliminated. The results are not yet a public-data or production benchmark.

## Public Dataset Validation

A candidate for external exploration is the [LANL User-Computer Authentication Associations in Time dataset](https://csr.lanl.gov/data/auth/). The publisher states that it contains anonymized successful user-to-computer authentication events over nine continuous months and waives copyright to the extent possible under law. It contains successful events only, so it does **not** provide a labelled attack test set or failed-login sequences; it can test behaviour-feature compatibility but cannot validate attack detection metrics. Cite the dataset if you use it.


## Disclaimer

This is an educational cybersecurity project focused on **defensive security**: detection, monitoring, and analysis. It uses synthetic data only. It does not contain instructions for attacking real systems, stealing credentials, or bypassing authentication. Results must not be treated as proof of malicious activity, and the project is not suitable for production use.

## Synthetic Label / Rule Alignment Audit

Run the audit from the repository root:

```bash
python -m src.audit_label_alignment
```

This report compares each individual risk signal and the combined risk-score baseline against `true_label`, including precision, recall, F1, false-positive rate, accuracy, and anomaly rate by risk score.

**Why this matters:** the dataset generator creates anomalies through specific scenarios (repeated failures, unusual-hour logins from unfamiliar locations/devices, and rapid-login bursts). Several of the detector rules target those same scenario characteristics. Strong rule performance on this dataset may therefore reflect how its labels were constructed, rather than independent evidence that the rules will generalise to real authentication logs.

Treat the report as a **synthetic-data diagnostic**, not a real-world benchmark. A stronger evaluation should include independently designed scenarios, benign edge cases that resemble suspicious activity, and—when available—appropriately anonymised real or public authentication data. Do not tune the rules to achieve a perfect score on the same synthetic labels and then present that score as proof of production effectiveness.


## Controlled Edge-Case Stress Test

The following stress test is a legacy diagnostic against the original 1,588-row dataset; it does not evaluate the rebuilt generator. Run it separately:

```bash
python -m src.stress_test_evaluation
```

The stress test fits Isolation Forest on the historical synthetic logs only, then scores 51 controlled events across 15 scenarios: 10 clearly benign-labelled events across five scenarios, 24 suspicious-labelled events across six scenarios, and 17 ambiguous events across four scenarios. It includes a cross-account password-spraying pattern (one source IP failing against four accounts) and a separate ambiguous shared-network pattern. Clearly benign cases include a new IP, new device, new location, or unusual login hour in isolation, plus six successful logins spaced six minutes apart. Suspicious cases include brute-force-style failures from familiar context, rapid successful logins from familiar context, low-and-slow failures, daytime logins involving a new IP/location or new device/location, and failures against multiple accounts from one source IP. Four overlapping scenarios—rapid successful logins, low-and-slow failures, a new device plus location during daytime, and failures against multiple accounts from a shared network—are marked ambiguous rather than assumed benign or malicious.

Feature engineering also counts distinct accounts with failed logins from the same source IP in an inclusive 15-minute window. Four distinct accounts is a project-specific trigger threshold, not a universal security standard. This signal can help expose password spraying, but shared office/NAT/VPN egress can produce similar patterns; therefore, an alert is evidence for investigation, not proof of an attack. The script reports event-level and scenario-level precision, recall, F1, false-positive rate, accuracy, and confusion counts using only clearly labelled normal/anomaly cases. Ambiguous events and scenarios are excluded from binary metrics and listed separately with each detector's alert outcome; they are not silently counted as false positives or true positives. It also reports the event position of the first alert in each scenario, so you can compare detection delay and identify completely missed sequences. It compares three approaches: Isolation Forest alone, the risk-score baseline (score >= 3), and a diagnostic hybrid that alerts when either ML or the rules flag an event. The hybrid is a layered stress-test comparison; it does not replace the dashboard's existing final-status logic. Combining detectors can improve recall but may also increase false positives, so compare both metrics rather than assuming the hybrid is automatically better. The tests cover ambiguous-label exclusion, scenario detection, first-alert positions, and hybrid prediction logic.

**Important evaluation caveat:** the stress set is small and hand-designed, and its labels are assumptions for testing, not objective truth. It is intended to expose blind spots and help guide further test design—not to establish real-world detection performance. It is separate from the original dataset and does not change the baseline labels or its reported results.
