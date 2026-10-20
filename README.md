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
14. [Example Output](#example-output)
15. [Testing](#testing)
16. [Limitations](#limitations)
17. [Future Improvements](#future-improvements)
18. [Blue Team / SOC Relevance](#blue-team--soc-relevance)
19. [Disclaimer](#disclaimer)

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
- Feature engineering (hour, unusual hour, new IP, new device, unusual location, location change, login frequency)
- Five explainable detection rules with readable reasons
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
   (explainable)                    (behavioural anomaly)F
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
3. **Rules:** check each login against five explainable rules.
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

| Total score | Level |
|---|---|
| 0 to 2 | Low |
| 3 to 5 | Medium |
| 6 and above | High |

The highest possible score is 13.

> **These points and levels are project-specific design choices, not cybersecurity standards.** They exist to rank alerts. SOC analysts have limited time, and not every alert has the same severity, so scoring helps them look at the most serious events first.

### Machine learning

The project uses **unsupervised learning**: the model is trained without labels, because in real environments labelled attacks are rare and attack patterns keep changing. The planted suspicious rows are used only afterwards to sanity-check results.

The model only sees these eight numeric features: `login_hour`, `failed_attempts`, `is_new_ip`, `is_new_device`, `is_unusual_hour`, `is_unusual_location`, `login_frequency`, and `location_change`.

The trained model is saved with Joblib to `models/anomaly_model.pkl`, so the dashboard can load it instead of retraining every time. The artifact includes a feature-version marker; the pipeline retrains an unversioned or outdated model automatically.

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
│   ├── detection_rules.py       # rules and risk scoring
│   ├── anomaly_detector.py      # Isolation Forest and combined pipeline
│   └── utils.py                 # shared file paths
├── models/
│   └── anomaly_model.pkl
├── dashboard/
│   └── app.py
├── notebooks/                   # reserved for exploration notebooks
├── tests/
│   └── test_detector.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Requires Python 3.10 or newer and Git.

```powershell
git clone https://github.com/your-username/AI-Login-Anomaly-Detector.git
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
python -m streamlit run dashboard/app.py   # 4. open the dashboard
```

Each stage can also be run on its own, for example `python -m src.preprocessing`, `python -m src.feature_engineering`, or `python -m src.detection_rules`.

## Dashboard

After step 4 above, open `http://localhost:8501`.

- **Headline numbers:** total login attempts, successful logins, failed logins, anomalies detected (ML), and high-risk events.
- **Charts:** login activity over time, successful vs failed logins, suspicious events by user, suspicious events by location, and risk level counts.
- **Filters:** username, location, risk level, status, and date range.
- **Suspicious login table:** timestamp, username, IP, location, device, risk score, ML result, status, and detection reason, sorted by risk score.

## Example Output

```text
Final status counts
  Normal activity            1495
  Requires investigation       88
  Potentially suspicious        5
```

An example row from the results:

```text
2026-09-03 22:22:24 | diana | 198.51.100.23 | Sao Paulo, Brazil | Unknown Device
score 13 (High) | ML: Anomalous | status: Requires investigation
reasons: Multiple failed login attempts; Successful login after repeated failures;
New IP address at an unusual hour; New device from an unusual location;
Rapid login attempts; Statistically unusual pattern (ML)
```

Of the 88 rule-flagged rows, the model also flagged 75 and missed 13. It flagged 5 other rows that no rule caught, which become "Potentially suspicious". This shows why the two methods complement each other.

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