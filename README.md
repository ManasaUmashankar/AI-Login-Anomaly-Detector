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

`src/generate_dataset.py` creates `data/login_logs.csv` with **1,588 records** over 30 days for four users. The random seed is fixed, so the data is identical on every run.

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

**Normal behaviour (1,500 records):** familiar IP, device, and location, working hours, mostly successful logins, with an occasional typo.

**Suspicious behaviour planted (88 records):**

- 5 brute-force-style runs (6 to 12 fast failures, 3 of them ending in a success)
- 8 unusual logins (new IP, new device, foreign location, 1 to 4 AM)
- 4 rapid-login bursts (6 logins in about a minute)

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
│   ├── generate_dataset.py      # creates the synthetic logs
│   ├── read_dataset.py          # loads and explores logs (Pandas)
│   ├── preprocessing.py         # cleaning and validation
│   ├── feature_engineering.py   # turns logs into features
│   ├── detection_rules.py       # rules, risk scoring, signal explanations
│   ├── anomaly_detector.py      # Isolation Forest and combined pipeline
│   ├── evaluate_model.py       # temporal holdout and multi-seed evaluation
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
python -m src.evaluate_model                # 4. run 20-seed chronological evaluation
python -m src.stress_test_evaluation        # 5. stress-test difficult synthetic edge cases
python -m streamlit run dashboard/app.py   # 6. open the dashboard
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

The model is fitted only on rows earlier than each holdout window; ground-truth labels are used only to score predictions. Features are built chronologically so later events do not define earlier user baselines. The per-window mean is an unweighted average of window metrics, not a pooled metric. Some windows may contain very few planted anomalies, so interpret each metric alongside its support and confusion matrix. All results describe this synthetic dataset and are not a real-world benchmark.

## Latest Evaluation Snapshot

Historical baseline snapshot (2026-10-21; before the multi-seed evaluation was added): across four chronological holdout windows, Isolation Forest achieved an unweighted mean F1-score of **0.9583** and recall of **0.9286**; the rule-based risk-score baseline achieved mean F1 of **0.9947** and recall of **1.0000**. In the separate controlled stress test, the hybrid detector achieved event-level F1 of **0.667** and recall of **0.500** across 34 clearly labelled events; 17 ambiguous events were reviewed separately. These figures are historical and have **not** been reproduced with the new 20-seed evaluator. Do not present them as the new evaluator's results.

These are results on synthetic, deliberately designed data, not a real-world benchmark. The generator may encode signals also used by the detector, so high scores are not independent evidence of production effectiveness. Run the evaluator and update this section only after the new results have been produced and reviewed. See the [full evaluation guide](docs/evaluation-guide.md) for metric interpretation.

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
| No real infrastructure | There is no real authentication system and no production-scale log ingestion. |

A production system would improve these with real log ingestion, per-user baselines built from history, many more features, tuned thresholds, analyst feedback, and regular model retraining.

## Future Improvements

- Real authentication log ingestion
- Splunk or other SIEM integration
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

Run a second, deliberately difficult evaluation separately from the original 1,588-row baseline:

```bash
python -m src.stress_test_evaluation
```

The stress test fits Isolation Forest on the historical synthetic logs only, then scores 51 controlled events across 15 scenarios: 10 clearly benign-labelled events across five scenarios, 24 suspicious-labelled events across six scenarios, and 17 ambiguous events across four scenarios. It includes a cross-account password-spraying pattern (one source IP failing against four accounts) and a separate ambiguous shared-network pattern. Clearly benign cases include a new IP, new device, new location, or unusual login hour in isolation, plus six successful logins spaced six minutes apart. Suspicious cases include brute-force-style failures from familiar context, rapid successful logins from familiar context, low-and-slow failures, daytime logins involving a new IP/location or new device/location, and failures against multiple accounts from one source IP. Four overlapping scenarios—rapid successful logins, low-and-slow failures, a new device plus location during daytime, and failures against multiple accounts from a shared network—are marked ambiguous rather than assumed benign or malicious.

Feature engineering also counts distinct accounts with failed logins from the same source IP in an inclusive 15-minute window. Four distinct accounts is a project-specific trigger threshold, not a universal security standard. This signal can help expose password spraying, but shared office/NAT/VPN egress can produce similar patterns; therefore, an alert is evidence for investigation, not proof of an attack. The script reports event-level and scenario-level precision, recall, F1, false-positive rate, accuracy, and confusion counts using only clearly labelled normal/anomaly cases. Ambiguous events and scenarios are excluded from binary metrics and listed separately with each detector's alert outcome; they are not silently counted as false positives or true positives. It also reports the event position of the first alert in each scenario, so you can compare detection delay and identify completely missed sequences. It compares three approaches: Isolation Forest alone, the risk-score baseline (score >= 3), and a diagnostic hybrid that alerts when either ML or the rules flag an event. The hybrid is a layered stress-test comparison; it does not replace the dashboard's existing final-status logic. Combining detectors can improve recall but may also increase false positives, so compare both metrics rather than assuming the hybrid is automatically better. The tests cover ambiguous-label exclusion, scenario detection, first-alert positions, and hybrid prediction logic.

**Important evaluation caveat:** the stress set is small and hand-designed, and its labels are assumptions for testing, not objective truth. It is intended to expose blind spots and help guide further test design—not to establish real-world detection performance. It is separate from the original dataset and does not change the baseline labels or its reported results.
