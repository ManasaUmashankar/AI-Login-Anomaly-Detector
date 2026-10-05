import sys
from pathlib import Path

# Lets Python find the 'src' folder when we run: streamlit run dashboard/app.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from src.anomaly_detector import run_pipeline

NORMAL_STATUS = "Normal activity"
RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#2e9e5b", "Medium": "#f0a030", "High": "#d64545"}
STATUS_COLORS = {"success": "#2e9e5b", "failed": "#d64545"}

# Column name in the data -> heading shown in the table
TABLE_COLUMNS = {
    "timestamp": "Timestamp",
    "username": "Username",
    "ip_address": "IP",
    "location": "Location",
    "device": "Device",
    "risk_score": "Risk Score",
    "ml_result": "ML Result",
    "final_status": "Status",
    "detection_reasons": "Detection Reason",
}

st.set_page_config(page_title="AI Login Anomaly Detector", layout="wide")


@st.cache_data
def load_results():
    """Run the full pipeline once and remember the answer."""
    return run_pipeline()


def make_count_chart(df, column, title):
    """Bar chart of suspicious events, split by final status."""
    counts = df.groupby([column, "final_status"]).size().reset_index(name="count")
    return px.bar(counts, x=column, y="count", color="final_status", title=title)


results = load_results()

st.title("AI Login Anomaly Detector")
st.caption(
    "Educational prototype using synthetic data. "
    "An anomaly is a signal to investigate, not proof of an attack."
)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
all_users = sorted(results["username"].unique())
all_locations = sorted(results["location"].unique())
all_statuses = sorted(results["final_status"].unique())

users = st.sidebar.multiselect("Username", all_users, default=all_users)
locations = st.sidebar.multiselect("Location", all_locations, default=all_locations)
levels = st.sidebar.multiselect("Risk level", RISK_ORDER, default=RISK_ORDER)
statuses = st.sidebar.multiselect("Status", all_statuses, default=all_statuses)

min_date = results["timestamp"].dt.date.min()
max_date = results["timestamp"].dt.date.max()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if len(date_range) != 2:
    st.info("Pick an end date to apply the date filter.")
    st.stop()
start_date, end_date = date_range

# ---------- Apply filters ----------
mask = (
    results["username"].isin(users)
    & results["location"].isin(locations)
    & results["risk_level"].isin(levels)
    & results["final_status"].isin(statuses)
    & (results["timestamp"].dt.date >= start_date)
    & (results["timestamp"].dt.date <= end_date)
)
data = results[mask]
if data.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

suspicious = data[data["final_status"] != NORMAL_STATUS]

# ---------- Headline numbers ----------
total = len(data)
successful = int((data["login_status"] == "success").sum())
failed = int((data["login_status"] == "failed").sum())
anomalies = int((data["ml_result"] == "Anomalous").sum())
high_risk = int((data["risk_level"] == "High").sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total login attempts", f"{total:,}")
c2.metric("Successful logins", f"{successful:,}")
c3.metric("Failed logins", f"{failed:,}")
c4.metric("Anomalies detected (ML)", f"{anomalies:,}")
c5.metric("High-risk events", f"{high_risk:,}")

# ---------- Charts: activity ----------
left, right = st.columns(2)
with left:
    daily = (
        data.assign(date=data["timestamp"].dt.date)
        .groupby(["date", "login_status"]).size().reset_index(name="count")
    )
    st.plotly_chart(px.line(daily, x="date", y="count", color="login_status",
                            color_discrete_map=STATUS_COLORS,
                            title="Login activity over time"))
with right:
    status_counts = data["login_status"].value_counts().reset_index()
    status_counts.columns = ["login_status", "count"]
    st.plotly_chart(px.bar(status_counts, x="login_status", y="count",
                           color="login_status", color_discrete_map=STATUS_COLORS,
                           title="Successful vs failed logins"))

# ---------- Charts: suspicious events ----------
st.subheader("Suspicious events")
st.caption("Suspicious = final status is not 'Normal activity' (rules and ML combined).")
if suspicious.empty:
    st.info("No suspicious events for the selected filters.")
else:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(make_count_chart(suspicious, "username", "Suspicious events by user"))
    with right:
        st.plotly_chart(make_count_chart(suspicious, "location", "Suspicious events by location"))

# ---------- Risk distribution ----------
st.subheader("Risk distribution")
left, right = st.columns(2)
with left:
    risk_counts = data["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
    risk_counts.columns = ["risk_level", "count"]
    st.plotly_chart(px.bar(risk_counts, x="risk_level", y="count", color="risk_level",
                           color_discrete_map=RISK_COLORS, text_auto=True,
                           title="Risk level counts"))
with right:
    st.info(
        "**How to read this dashboard**\n\n"
        "- A high score or an anomaly means *investigate*, not *hacked*.\n"
        "- New IPs, new devices and VPNs can be legitimate.\n"
        "- The model can raise false alarms and miss real attacks.\n"
        "- Scores and thresholds are this project's own design choices."
    )

# ---------- Suspicious login table ----------
st.subheader("Suspicious login table")
if suspicious.empty:
    st.info("No suspicious logins for the selected filters.")
else:
    table = suspicious.sort_values(["risk_score", "timestamp"], ascending=[False, True])
    table = table[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)
    st.dataframe(table, hide_index=True)