import sys
from pathlib import Path

# Lets Python find the 'src' folder when we run: streamlit run dashboard/app.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from src.anomaly_detector import FEATURE_VERSION, run_pipeline

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

st.set_page_config(page_title="AI Login Anomaly Detector", page_icon="🛡️", layout="wide")

# ---------- SOC visual theme ----------
st.markdown(
    """
    <style>
    :root {
        --soc-bg: #0b0e12;
        --soc-panel: #12171e;
        --soc-panel-raised: #171e27;
        --soc-border: #28313c;
        --soc-text: #edf2f7;
        --soc-muted: #9aa7b5;
        --soc-orange: #ff8a3d;
        --soc-orange-soft: rgba(255, 138, 61, 0.12);
    }
    .stApp {
        background:
            radial-gradient(ellipse at 80% 0%, rgba(255, 138, 61, 0.075), transparent 34%),
            var(--soc-bg);
        color: var(--soc-text);
    }
    [data-testid="stHeader"] { background: rgba(11, 14, 18, 0.88); }
    [data-testid="stSidebar"] {
        background: #0e1319;
        border-right: 1px solid var(--soc-border);
    }
    [data-testid="stSidebar"] > div { background: transparent; }
    h1, h2, h3 { color: var(--soc-text) !important; letter-spacing: -0.025em; }
    h1 { font-weight: 750 !important; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--soc-muted); }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #171e27, #11161d);
        border: 1px solid var(--soc-border);
        border-top: 2px solid var(--soc-orange);
        padding: 17px 18px;
        border-radius: 13px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
    }
    [data-testid="stMetricLabel"] { color: var(--soc-muted) !important; }
    [data-testid="stMetricValue"] { color: var(--soc-text) !important; }
    [data-testid="stMetricDelta"] { color: var(--soc-orange) !important; }
    [data-testid="stPlotlyChart"] {
        background: var(--soc-panel);
        border: 1px solid var(--soc-border);
        border-radius: 13px;
        padding: 8px 8px 0;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--soc-border);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: var(--soc-panel-raised);
        border-color: var(--soc-border);
    }
    .soc-hero {
        display: flex; justify-content: space-between; align-items: center;
        gap: 20px; padding: 25px 27px; margin: 5px 0 24px;
        border: 1px solid #34302c; border-radius: 17px;
        background: linear-gradient(115deg, #171b21 0%, #1c1a18 65%, #241910 100%);
    }
    .soc-kicker {
        color: var(--soc-orange); text-transform: uppercase; font-size: 0.72rem;
        font-weight: 800; letter-spacing: 0.16em; margin-bottom: 8px;
    }
    .soc-title { font-size: clamp(1.65rem, 3vw, 2.35rem); font-weight: 800;
        color: #f5f7fa; letter-spacing: -0.045em; line-height: 1.1; }
    .soc-subtitle { color: #a9b3be; font-size: 0.94rem; margin-top: 9px; max-width: 650px; }
    .soc-status {
        white-space: nowrap; border: 1px solid rgba(255, 138, 61, .42);
        color: #ffb17d; background: var(--soc-orange-soft); border-radius: 999px;
        padding: 9px 13px; font-size: .76rem; font-weight: 750; letter-spacing: .04em;
    }
    .soc-section {
        color: var(--soc-orange); text-transform: uppercase; letter-spacing: .12em;
        font-size: .72rem; font-weight: 800; margin: 8px 0 5px;
    }
    div.stButton > button {
        border: 1px solid var(--soc-border); border-radius: 9px;
        background: var(--soc-panel-raised); color: var(--soc-text);
    }
    div.stButton > button:hover { border-color: var(--soc-orange); color: var(--soc-orange); }
    hr { border-color: var(--soc-border); }
    @media (max-width: 700px) {
        .soc-hero { align-items: flex-start; flex-direction: column; padding: 20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_results(feature_version=FEATURE_VERSION):
    """Cache results per feature version so old dashboard results are invalidated."""
    return run_pipeline()


def make_count_chart(df, column, title):
    """Bar chart of suspicious events, split by final status."""
    counts = df.groupby([column, "final_status"]).size().reset_index(name="count")
    return px.bar(counts, x=column, y="count", color="final_status", title=title)


results = load_results(FEATURE_VERSION)

st.markdown(
    """
    <div class="soc-hero">
      <div>
        <div class="soc-kicker">Security Operations · Authentication Analytics</div>
        <div class="soc-title">Login Threat Monitor</div>
        <div class="soc-subtitle">
          AI-assisted visibility into authentication patterns, risk signals, and events
          that may deserve analyst review.
        </div>
      </div>
      <div class="soc-status">● PROTOTYPE · MONITORING VIEW</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Educational prototype using synthetic data. An anomaly is a signal to investigate, "
    "not proof of an attack."
)

st.markdown('<div class="soc-section">01 / Scope & filters</div>', unsafe_allow_html=True)\n\n# ---------- Sidebar filters ----------
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

st.markdown('<div class="soc-section">02 / Security overview</div>', unsafe_allow_html=True)\n\n# ---------- Headline numbers ----------
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

st.markdown('<div class="soc-section">03 / Authentication telemetry</div>', unsafe_allow_html=True)\n\n# ---------- Charts: activity ----------
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

st.markdown('<div class="soc-section">04 / Threat patterns</div>', unsafe_allow_html=True)\n\n# ---------- Charts: suspicious events ----------
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

st.markdown('<div class="soc-section">05 / Risk posture</div>', unsafe_allow_html=True)\n\n# ---------- Risk distribution ----------
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

st.markdown('<div class="soc-section">06 / Alert queue</div>', unsafe_allow_html=True)\n\n# ---------- Suspicious login table ----------
st.subheader("Suspicious login table")
if suspicious.empty:
    st.info("No suspicious logins for the selected filters.")
else:
    table = suspicious.sort_values(["risk_score", "timestamp"], ascending=[False, True])
    table = table[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)
    st.dataframe(table, hide_index=True)

    
st.markdown('<div class="soc-section">07 / Analyst investigation</div>', unsafe_allow_html=True)\n\n# ---------- Alert investigation view ----------
st.subheader("Alert investigation")

if suspicious.empty:
    st.info("No suspicious events available to investigate.")
else:
    investigation_data = suspicious.sort_values(
        ["risk_score", "timestamp"],
        ascending=[False, True]
    ).copy()

    investigation_data["alert_label"] = (
        investigation_data["username"].astype(str)
        + " | "
        + investigation_data["timestamp"].astype(str)
        + " | Risk: "
        + investigation_data["risk_level"].astype(str)
    )

    selected_alert = st.selectbox(
        "Select a login event",
        investigation_data.index,
        format_func=lambda idx: investigation_data.loc[idx, "alert_label"]
    )

    alert = investigation_data.loc[selected_alert]

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score", alert["risk_score"])
    c2.metric("Risk level", alert["risk_level"])
    c3.metric("ML result", alert["ml_result"])

    st.markdown("### Login details")
    st.write("**Timestamp:**", alert["timestamp"])
    st.write("**Username:**", alert["username"])
    st.write("**IP address:**", alert["ip_address"])
    st.write("**Location:**", alert["location"])
    st.write("**Device:**", alert["device"])
    st.write("**Login status:**", alert["login_status"])
    st.write("**Final status:**", alert["final_status"])

    st.markdown("### Why was this flagged?")
    st.write(alert["detection_reasons"])

    st.markdown("### Recommended investigation steps")

    reasons = str(alert["detection_reasons"]).lower()
    recommendations = []

    if "failed login" in reasons or "repeated failures" in reasons:
        recommendations.append(
            "Review failed login attempts and check whether a successful login followed them."
        )

    if "new ip" in reasons or "ip address" in reasons:
        recommendations.append(
            "Check whether the IP address is expected for this user."
        )

    if "new device" in reasons or "device" in reasons:
        recommendations.append(
            "Verify whether the device is recognised by the account owner."
        )

    if "unusual hour" in reasons or "unusual pattern" in reasons:
        recommendations.append(
            "Compare the login time with the user's normal activity."
        )

    if recommendations:
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")
    else:
        st.info(
            "Review the event details and related authentication logs manually."
        )

    st.caption(
        "These are investigation suggestions, not proof of malicious activity."
    )