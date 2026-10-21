# Demo Walkthrough

Use this short walkthrough when presenting the AI Login Anomaly Detector to a recruiter, mentor, or interviewer.

## Before recording

1. Create and activate the virtual environment.
2. Install the dependencies from `requirements.txt`.
3. Run the dataset generator if the expected dataset is absent.
4. Run the pipeline and evaluation scripts if you want fresh output.
5. Start the dashboard:

   ```powershell
   python -m streamlit run dashboard/app.py
   ```

6. Open the local Streamlit URL printed in the terminal.

## Suggested 2–3 minute walkthrough

### 1. Explain the goal

“This is an educational Blue Team project that ranks potentially suspicious authentication events using explainable rules, risk scoring, and Isolation Forest.”

### 2. Show the overview

Point out the login-event totals, failed logins, ML outliers, severity breakdown, and activity chart. Explain that the values describe the current synthetic dataset and pipeline output.

### 3. Filter the alert queue

Change a severity or disposition filter and show that the visible results update. Mention that filtering helps narrow an investigation; it is not a new model evaluation.

### 4. Investigate one event

Select an alert and explain its risk score, ML signal, final disposition, and detection reasons. Describe these as investigation clues, not proof that an account is compromised.

### 5. Explain the detection pipeline

Briefly describe preprocessing, temporal feature engineering, rule-based scoring, Isolation Forest, and the combined final status. Highlight that equal-timestamp events are processed in batches to avoid input-row-order-dependent historical baselines.

### 6. Close with evaluation and limitations

Explain the chronological holdout evaluation, rule-based baseline, and controlled edge-case stress tests. Be explicit that the data is synthetic and the project is not connected to live authentication infrastructure or a production SIEM.

## Suggested screen recording

- Capture the overview first.
- Show one filter change.
- Select an alert and pause on its investigation reasons.
- Briefly show the architecture diagram or evaluation report.
- End on the repository README.

Avoid showing secrets, personal files, access tokens, or unrelated browser tabs. Do not claim that the dashboard sends live alerts or that the model has been validated against real production traffic.
