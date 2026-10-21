# Interview Notes: AI Login Anomaly Detector

Use these notes to practise explaining the project in your own words. The goal is to understand the implementation, not memorise claims.

## 60-second project pitch

“I built an educational Blue Team login-anomaly detection pipeline in Python. It cleans synthetic authentication logs, derives time-aware and rolling-window features, applies explainable detection rules and risk scoring, and uses Isolation Forest to flag statistically unusual events. A Streamlit dashboard lets an analyst filter the alert queue and inspect why an event was flagged. I evaluate the model with chronological holdout windows, compare it with a rule-based baseline, and run controlled edge-case stress tests. The data is synthetic, so I present the results as engineering diagnostics rather than real-world security performance.”

## Questions to practise

**Why use both rules and machine learning?**  
Rules provide readable, explicit reasons for known patterns. Isolation Forest can flag unusual combinations not directly encoded by a rule. Their outputs are complementary, but neither proves malicious activity.

**Why Isolation Forest?**  
It is an unsupervised anomaly-detection method that works with numeric features and does not require attack labels during fitting. Its output depends on the feature design and configuration, including the contamination assumption.

**What does contamination mean here?**  
It is a model setting that influences the expected fraction of observations treated as outliers. The project's 0.05 value is a design assumption, not a measured real-world attack rate.

**Why chronological evaluation instead of a random split?**  
A random split can mix future and past events. Chronological evaluation better represents training on historical data and scoring later events.

**How do you reduce temporal leakage?**  
Historical baselines use strictly earlier timestamps. Events sharing a timestamp are evaluated as a batch, and temporal split boundaries do not separate equal timestamps.

**Why include a rule-based baseline?**  
It gives a simple, explainable comparison point. A more complex model should be evaluated against a meaningful baseline rather than presented without context.

**What are the biggest limitations?**  
Synthetic labels can overlap with detector rules; the data is small and hand-designed; thresholds and risk scores are project-specific; IP/location fields can be noisy; and the project does not ingest live logs or integrate with a production SIEM.

**What would you do next for production use?**  
Validate on appropriately governed real or public authentication logs, add benign and malicious scenarios with independent labels, calibrate thresholds against operational needs, monitor false positives and false negatives, add safe log ingestion and access controls, and integrate with a SIEM only after testing and review.

## Claims to avoid

- “It detects real attacks with 98% accuracy” unless you have a clearly described, appropriate evaluation that supports that claim.
- “The model understands hacking.”
- “A high-risk event means the account is compromised.”
- “The dashboard is a production SIEM.”
- “Synthetic-data performance guarantees real-world performance.”

## Resume bullet (adapt after confirming your own work)

- Built a Python-based login-anomaly detection prototype combining Isolation Forest, explainable detection rules, risk scoring, temporal/rolling-window features, and a Streamlit SOC-style investigation dashboard; added chronological evaluation, a rule-based baseline, synthetic edge-case stress tests, and regression tests.

Be ready to explain every component in this bullet and show the relevant code.
