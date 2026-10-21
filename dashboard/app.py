import sys
from pathlib import Path

# Lets Python find the "src" folder when running Streamlit from the project root.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detector import FEATURE_VERSION, run_pipeline

NORMAL_STATUS = "Normal activity"
RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#36c98f", "Medium": "#ffb547", "High": "#ff5d67"}
STATUS_COLORS = {"success": "#36c98f", "failed": "#ff5d67"}
ACCENT = "#ff8a3d"

TABLE_COLUMNS = {
    "timestamp": "Timestamp",
    "username": "Username",
    "ip_address": "Source IP",
    "location": "Location",
    "device": "Device",
    "risk_score": "Risk score",
    "risk_level": "Severity",
    "ml_result": "ML signal",
    "final_status": "Disposition",
    "detection_reasons": "Detection reason",
}

st.set_page_config(
    page_title="AI Login Anomaly Detector | Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A distinct charcoal / amber SOC workspace. Charts and values below are built
# from the project's actual pipeline output; no simulated live status is shown.
st.markdown(
    """
    <style>
    :root {
      --bg: #0a0d11; --panel: #111720; --panel2: #161e28;
      --line: #27313e; --text: #eef3f8; --muted: #8d9bab;
      --amber: #ff8a3d; --green: #36c98f; --red: #ff5d67;
    }
    .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stHeader"] { background: rgba(10,13,17,.92); }
    [data-testid="stSidebar"] { background: #0e131a; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div { background: transparent; }
    h1,h2,h3 { color: var(--text) !important; letter-spacing: -.025em; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    .block-container { padding: 1.25rem 2.25rem 3rem; max-width: 1660px; }
    .brand { display:flex; gap:11px; align-items:center; padding: 7px 0 22px; }
    .brand-mark { display:flex; align-items:center; justify-content:center; width:42px;height:42px;
      border:1px solid rgba(255,138,61,.48); border-radius:12px; color:var(--amber);
      background:rgba(255,138,61,.10); font-size:21px; }
    .brand-name { color:var(--text); font-weight:850; font-size:.84rem; letter-spacing:.035em; line-height:1.35; max-width:180px; }
    .brand-sub { color:var(--muted); font-size:.68rem; letter-spacing:.13em; text-transform:uppercase; margin-top:3px; }
    .eyebrow { color:var(--amber); font-size:.7rem; font-weight:850; letter-spacing:.16em; text-transform:uppercase; }
    .hero { display:flex; justify-content:space-between; align-items:center; gap:22px; padding:25px 28px;
      border:1px solid #34302c; border-radius:17px; margin:4px 0 18px;
      background:linear-gradient(115deg,#151b23 0%,#191a1b 62%,#281b13 100%); }
    .hero-title { font-size:clamp(1.8rem,3vw,2.65rem); line-height:1.1; font-weight:850;
      letter-spacing:-.05em; color:#f6f8fb; margin-top:8px; }
    .hero-copy { max-width:720px; color:#a8b3c0; font-size:.92rem; margin-top:10px; line-height:1.55; }
    .scope-pill { border:1px solid rgba(255,138,61,.4); background:rgba(255,138,61,.09);
      border-radius:999px; padding:9px 12px; color:#ffb27e; font-size:.72rem; font-weight:800; white-space:nowrap; }
    .section-head { display:flex; align-items:center; gap:9px; margin:25px 0 11px; color:#e9eef5;
      font-size:.98rem; font-weight:800; }
    .section-num { color:var(--amber); font-size:.69rem; letter-spacing:.1em; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px 18px; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,#171f29,#111720); border:1px solid var(--line);
      border-top:2px solid var(--amber); border-radius:13px; padding:16px 17px; min-height:105px; }
    [data-testid="stMetricLabel"] { color:var(--muted) !important; font-size:.78rem; }
    [data-testid="stMetricValue"] { color:var(--text) !important; font-weight:800; }
    [data-testid="stPlotlyChart"] { background:var(--panel); border:1px solid var(--line);
      border-radius:14px; padding:5px; overflow:hidden; }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] { background:#33251c !important; border:1px solid #75401f !important; color:#ffe0c7 !important; }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] span { color:#ffe0c7 !important; }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg { fill:#ffb27e !important; }
    [data-testid="stSidebar"] h3 { font-size:1rem !important; }
    [data-testid="stMetric"] { box-shadow:0 7px 22px rgba(0,0,0,.16); }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
      background:var(--panel2); border-color:var(--line); border-radius:9px; }
    div.stButton > button { border:1px solid var(--line); border-radius:9px; background:var(--panel2); color:var(--text); }
    div.stButton > button:hover { border-color:var(--amber); color:var(--amber); }
    hr { border-color:var(--line); }
    .footnote { color:var(--muted); font-size:.78rem; line-height:1.5; }
    @media(max-width:760px) { .hero { align-items:flex-start; flex-direction:column; padding:20px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_results(feature_version=FEATURE_VERSION):
    """Cache pipeline output per feature version to invalidate stale results."""
    return run_pipeline()


def section(number, label):
    st.markdown(
        f'<div class="section-head"><span class="section-num">{number}</span>{label}</div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=310):
    """Apply the same clean, dark chart styling to Plotly figures."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aab6c4", family="Inter, sans-serif", size=11),
        title=dict(font=dict(color="#edf2f7", size=13), x=0.03, xanchor="left", y=0.97),
        margin=dict(l=18, r=18, t=64, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        hoverlabel=dict(bgcolor="#18212c", font_color="#f3f6fa"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#27313e")
    fig.update_yaxes(showgrid=True, gridcolor="#222c37", zeroline=False)
    return fig


def parse_detection_reasons(series):
    reasons = []
    for value in series.fillna("").astype(str):
        reasons.extend(reason.strip() for reason in value.split(";") if reason.strip())
    return pd.Series(reasons, dtype="object")


def make_alert_table(frame):
    columns = [column for column in TABLE_COLUMNS if column in frame.columns]
    table = frame.sort_values(["risk_score", "timestamp"], ascending=[False, False]).copy()
    table = table[columns].rename(columns=TABLE_COLUMNS)
    if "Timestamp" in table:
        table["Timestamp"] = table["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return table


results = load_results(FEATURE_VERSION).copy()
results["timestamp"] = pd.to_datetime(results["timestamp"], errors="coerce")
results = results.dropna(subset=["timestamp"])

# ---------- Sidebar / analyst controls ----------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">⌁</div>
          <div><div class="brand-name">AI LOGIN ANOMALY DETECTOR</div>
          <div class="brand-sub">Authentication defense</div></div>
        </div>
        <div class="eyebrow">Workspace</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Investigation scope")
    all_users = sorted(results["username"].dropna().astype(str).unique())
    all_locations = sorted(results["location"].dropna().astype(str).unique())
    all_statuses = sorted(results["final_status"].dropna().astype(str).unique())
    users = st.multiselect("Accounts", all_users, default=all_users)
    locations = st.multiselect("Locations", all_locations, default=all_locations)
    levels = st.multiselect("Severity", RISK_ORDER, default=RISK_ORDER)
    statuses = st.multiselect("Disposition", all_statuses, default=all_statuses)

    min_date = results["timestamp"].dt.date.min()
    max_date = results["timestamp"].dt.date.max()
    date_range = st.date_input(
        "Time window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    st.divider()
    st.markdown("**Signal legend**")
    st.markdown("🟢 Low · 🟠 Medium · 🔴 High")
    st.caption("Severity is based on this project's risk-score thresholds.")
    if st.button("Refresh pipeline results", use_container_width=True):
        load_results.clear()
        st.rerun()

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Choose both a start and end date to apply the time filter.")
    st.stop()

start_date, end_date = date_range
mask = (
    results["username"].astype(str).isin(users)
    & results["location"].astype(str).isin(locations)
    & results["risk_level"].isin(levels)
    & results["final_status"].isin(statuses)
    & (results["timestamp"].dt.date >= start_date)
    & (results["timestamp"].dt.date <= end_date)
)
data = results.loc[mask].copy()
if data.empty:
    st.markdown(
        '<div class="hero"><div><div class="eyebrow">No matching telemetry</div>'
        '<div class="hero-title">Nothing in this scope</div><div class="hero-copy">'
        'Adjust the account, location, severity, disposition, or date filters to inspect available events.</div></div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

suspicious = data[data["final_status"] != NORMAL_STATUS].copy()
high_risk = data[data["risk_level"] == "High"]
medium_risk = data[data["risk_level"] == "Medium"]
failed = data[data["login_status"] == "failed"]
ml_anomalies = data[data["ml_result"] == "Anomalous"]
needs_review = data[data["final_status"] == "Requires investigation"]

# ---------- Command header ----------
st.markdown(
    f"""
    <div class="hero">
      <div>
        <div class="eyebrow">Machine learning · Explainable security rules · Authentication telemetry</div>
        <div class="hero-title">AI Login Anomaly Detector</div>
        <div class="hero-copy">A focused view of authentication behaviour, rule-based risk signals,
        and ML outliers. Use the queue to decide which events deserve closer review.</div>
      </div>
      <div class="scope-pill">● AI + RULE-BASED DETECTION</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="footnote">AI Login Anomaly Detector · Prototype data only. An anomaly is a signal to investigate, not proof of compromise. '
    'This dashboard does not represent a live production SOC.</div>',
    unsafe_allow_html=True,
)

# ---------- Topline operational metrics ----------
section("01 /", "Security overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Events in scope", f"{len(data):,}")
m2.metric("Failed logins", f"{len(failed):,}", help="Login attempts whose login_status is failed.")
m3.metric("ML outliers", f"{len(ml_anomalies):,}", help="Rows marked Anomalous by Isolation Forest.")
m4.metric("High severity", f"{len(high_risk):,}")
m5.metric("Requires review", f"{len(needs_review):,}")

# ---------- Analytics ----------
section("02 /", "Authentication telemetry")
left, right = st.columns([1.55, 1])
with left:
    daily = (
        data.assign(day=data["timestamp"].dt.floor("D"))
        .groupby(["day", "login_status"]).size().reset_index(name="events")
    )
    activity = px.area(
        daily, x="day", y="events", color="login_status",
        color_discrete_map=STATUS_COLORS, title="Login activity by outcome",
        labels={"day": "Time", "events": "Events", "login_status": "Result"},
    )
    activity.update_traces(line=dict(width=2))
    st.plotly_chart(chart_layout(activity, 335), use_container_width=True)

with right:
    risk_counts = data["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
    donut = go.Figure(
        data=[go.Pie(
            labels=risk_counts.index.tolist(), values=risk_counts.values.tolist(),
            hole=0.72, sort=False, marker=dict(colors=[RISK_COLORS[x] for x in risk_counts.index]),
            textinfo="label+value", textfont=dict(size=11),
            hovertemplate="%{label}: %{value} events<extra></extra>",
        )]
    )
    donut.update_layout(title="Risk level breakdown", showlegend=False, annotations=[
        dict(text=f"<b>{len(data):,}</b><br><span style='font-size:10px'>EVENTS</span>",
             x=0.5, y=0.5, showarrow=False, font=dict(color="#edf2f7", size=20))
    ])
    st.plotly_chart(chart_layout(donut, 335), use_container_width=True)

section("03 /", "Detection signals")
left, right = st.columns([1.15, 1])
with left:
    reason_counts = parse_detection_reasons(suspicious.get("detection_reasons", pd.Series(dtype=str))).value_counts().head(8)
    if reason_counts.empty:
        st.info("No detection reasons are present in the current filter scope.")
    else:
        reason_frame = reason_counts.sort_values().reset_index()
        reason_frame.columns = ["reason", "events"]
        reason_fig = px.bar(
            reason_frame, x="events", y="reason", orientation="h",
            title="Top detection reasons",
            labels={"events": "Event count", "reason": ""},
            text="events",
            color_discrete_sequence=[ACCENT],
        )
        reason_fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(chart_layout(reason_fig, 330), use_container_width=True)

with right:
    if suspicious.empty:
        st.info("No suspicious events in this scope.")
    else:
        ip_counts = suspicious.groupby("ip_address").size().sort_values(ascending=False).head(7)
        ip_frame = ip_counts.reset_index(name="events")
        ip_frame["ip_address"] = ip_frame["ip_address"].astype(str)
        ip_fig = px.bar(
            ip_frame.sort_values("events"), x="events", y="ip_address", orientation="h",
            title="Most-flagged source IPs",
            labels={"events": "Flagged events", "ip_address": "Source IP"},
            text="events",
            color_discrete_sequence=["#ff6b55"],
        )
        ip_fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(chart_layout(ip_fig, 330), use_container_width=True)

# ---------- Alert queue ----------
section("04 /", "Analyst alert queue")
if suspicious.empty:
    st.success("No suspicious events match the selected filters.")
else:
    q1, q2, q3 = st.columns(3)
    q1.metric("Potentially suspicious", int((suspicious["final_status"] == "Potentially suspicious").sum()))
    q2.metric("Requires investigation", int((suspicious["final_status"] == "Requires investigation").sum()))
    q3.metric("Medium severity", len(medium_risk))
    st.caption("Sorted by risk score first. Treat each row as a lead for investigation, not a confirmed incident.")
    table = make_alert_table(suspicious)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=360,
        column_config={
            "Risk score": st.column_config.NumberColumn("Risk score", format="%d"),
            "Severity": st.column_config.TextColumn("Severity"),
            "ML signal": st.column_config.TextColumn("ML signal"),
            "Disposition": st.column_config.TextColumn("Disposition"),
            "Detection reason": st.column_config.TextColumn("Detection reason", width="large"),
        },
    )

    investigation = suspicious.sort_values(
        ["risk_score", "timestamp"], ascending=[False, False]
    ).copy()
    investigation["event_key"] = (
        investigation["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        + " · " + investigation["username"].astype(str)
        + " · " + investigation["ip_address"].astype(str)
    )
    section("05 /", "Event investigation")
    selected_key = st.selectbox(
        "Choose an event to inspect",
        investigation["event_key"].tolist(),
        help="Events are ordered by risk score, then newest timestamp.",
    )
    alert = investigation.loc[investigation["event_key"] == selected_key].iloc[0]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Risk score", int(alert["risk_score"]))
    d2.metric("Severity", str(alert["risk_level"]))
    d3.metric("ML signal", str(alert["ml_result"]))
    d4.metric("Login result", str(alert["login_status"]).title())

    detail_left, detail_right = st.columns([1, 1])
    with detail_left:
        with st.container(border=True):
            st.markdown("**EVENT CONTEXT**")
            st.write("**Timestamp:**", alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S"))
            st.write("**Account:**", alert["username"])
            st.write("**Source IP:**", alert["ip_address"])
            st.write("**Location:**", alert["location"])
            st.write("**Device:**", alert["device"])
            st.write("**Disposition:**", alert["final_status"])
    with detail_right:
        with st.container(border=True):
            st.markdown("**WHY IT WAS FLAGGED**")
            reasons = str(alert.get("detection_reasons", "")).strip()
            if reasons:
                for reason in [part.strip() for part in reasons.split(";") if part.strip()]:
                    st.markdown(f"- {reason}")
            else:
                st.write("No explicit rule reason recorded for this event.")

    reasons_lower = str(alert.get("detection_reasons", "")).lower()
    recommendations = []
    if "failed" in reasons_lower or "failures" in reasons_lower:
        recommendations.append("Review nearby authentication failures and check whether a success followed them.")
    if "source" in reasons_lower or "multiple accounts" in reasons_lower:
        recommendations.append("Compare the source IP across affected accounts; shared networks can also produce this pattern.")
    if "ip address" in reasons_lower or "new ip" in reasons_lower:
        recommendations.append("Verify whether this source IP is expected for the account and environment.")
    if "device" in reasons_lower:
        recommendations.append("Confirm whether the device is recognised by the account owner.")
    if "unusual hour" in reasons_lower or "location" in reasons_lower:
        recommendations.append("Compare time and location with the user's usual activity and known travel.")
    with st.expander("Suggested analyst checks", expanded=True):
        if recommendations:
            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")
        else:
            st.markdown("- Correlate this event with surrounding login records and other available telemetry.")
        st.caption("Suggestions are triage prompts, not conclusions about user intent.")

st.divider()
st.markdown(
    '<div class="footnote">AI Login Anomaly Detector · Sentinel analyst workspace · '
    'Rule thresholds and model outputs are project-specific. Validate against labelled, representative data before operational use.</div>',
    unsafe_allow_html=True,
)
