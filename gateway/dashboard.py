"""
gateway/dashboard.py
====================
Industrial e-Nose — Streamlit Real-Time Monitoring Dashboard

Polls the gateway REST API every 2 seconds and renders:
  - Live sensor matrix status (R_ratio, health, drift)
  - Identified gas + concentration + confidence badge
  - 60-second rolling R_ratio time-series per channel
  - Anomaly feed

Launch (standalone dev):
    streamlit run gateway/dashboard.py

Launch (via docker-compose):
    docker-compose up dashboard
"""

import os
import time
from datetime import datetime

import requests
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GATEWAY_URL   = os.getenv("GATEWAY_API_URL", "http://localhost:8000")
POLL_INTERVAL = 2.0   # seconds
HISTORY_N     = 300   # ~60 s at 200 ms intervals

GAS_EMOJI = {
    "clean_air": "✅",
    "ethanol":   "🍷",
    "ammonia":   "⚗️",
    "co":        "☠️",
    "acetone":   "🧪",
    "unknown":   "❓",
}

HEALTH_COLOR = {
    "ok":       "#2ecc71",
    "warn":     "#f39c12",
    "critical": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title     = "e-Nose Industrial Monitor",
    page_icon      = "🔬",
    layout         = "wide",
    initial_sidebar_state = "collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .metric-card {
        background: #1c1f2e;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #4a90e2;
        margin-bottom: 8px;
    }
    .gas-badge {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .confidence-bar {
        background: #2a2d3e;
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        margin-top: 6px;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #4a90e2, #7ed321);
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    .anomaly-tag {
        display: inline-block;
        background: #c0392b22;
        border: 1px solid #c0392b;
        color: #e74c3c;
        border-radius: 6px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.78rem;
    }
    .healthy { color: #2ecc71; }
    .warn    { color: #f39c12; }
    .critical{ color: #e74c3c; }
    h1, h2, h3 { color: #e0e0e0 !important; }
    .stDataFrame { background: #1c1f2e; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1)
def fetch_latest() -> dict | None:
    try:
        r = requests.get(f"{GATEWAY_URL}/api/v1/latest", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=2)
def fetch_history(n: int = HISTORY_N) -> list[dict]:
    try:
        r = requests.get(f"{GATEWAY_URL}/api/v1/history", params={"n": n}, timeout=3)
        r.raise_for_status()
        return r.json().get("snapshots", [])
    except Exception:
        return []


@st.cache_data(ttl=5)
def fetch_health() -> dict:
    try:
        r = requests.get(f"{GATEWAY_URL}/api/v1/health", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_gas_panel(snap: dict):
    gas    = snap.get("predicted_gas", "unknown")
    ppm    = snap.get("concentration_ppm", 0.0)
    conf   = snap.get("model_confidence", 0.0)
    emoji  = GAS_EMOJI.get(gas, "❓")
    locked = snap.get("all_baselines_locked", False)

    col1, col2, col3 = st.columns([2, 1.5, 1.5])

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:1px;">
                Identified Gas
            </div>
            <div class="gas-badge" style="margin-top:8px;">
                {emoji} &nbsp; {gas.upper().replace('_', ' ')}
            </div>
            <div style="font-size:0.8rem; color:#888; margin-top:6px;">
                Confidence
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{conf*100:.0f}%;"></div>
            </div>
            <div style="font-size:0.9rem; color:#aaa; margin-top:4px;">{conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.75rem; color:#888; text-transform:uppercase;">Concentration</div>
            <div style="font-size:2.8rem; font-weight:700; color:#7ed321; margin-top:4px;">
                {ppm:.1f}
            </div>
            <div style="color:#666; font-size:0.9rem;">ppm</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        baseline_color = "#2ecc71" if locked else "#f39c12"
        baseline_text  = "LOCKED" if locked else "CALIBRATING"
        ts = snap.get("timestamp_utc", "")[:19].replace("T", " ")
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.75rem; color:#888; text-transform:uppercase;">Baseline</div>
            <div style="font-size:1.4rem; font-weight:700; color:{baseline_color}; margin-top:8px;">
                {baseline_text}
            </div>
            <div style="color:#555; font-size:0.75rem; margin-top:8px;">{ts} UTC</div>
        </div>
        """, unsafe_allow_html=True)


def render_anomaly_feed(snap: dict):
    flags = snap.get("anomaly_flags", [])
    if not flags:
        st.markdown('<span style="color:#2ecc71; font-size:0.9rem;">✓ No anomalies detected</span>',
                    unsafe_allow_html=True)
    else:
        tags = "".join(f'<span class="anomaly-tag">{f}</span>' for f in flags)
        st.markdown(f'<div style="margin-top:4px;">{tags}</div>', unsafe_allow_html=True)


def render_sensor_matrix(snap: dict, health: dict):
    channels = snap.get("channels", [])
    if not channels:
        st.warning("No channel data available")
        return

    cols = st.columns(len(channels))
    for ch_data in channels:
        ch = ch_data.get("ch", 0)
        with cols[ch]:
            r_ratio    = ch_data.get("r_ratio", 1.0)
            r_ohm      = ch_data.get("resistance_ohm", 0.0)
            phase      = ch_data.get("heater_phase", "—")
            drift_stat = ch_data.get("drift_status", "ok")
            drift_mag  = ch_data.get("drift_magnitude", 0.0)
            h_score    = ch_data.get("health_score", 1.0)
            locked     = ch_data.get("baseline_locked", False)

            drift_color = HEALTH_COLOR.get(drift_stat, "#aaa")
            health_pct  = h_score * 100

            # R_ratio gauge color
            if r_ratio > 0.9:
                ratio_color = "#2ecc71"
            elif r_ratio > 0.6:
                ratio_color = "#f39c12"
            else:
                ratio_color = "#e74c3c"

            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{drift_color};">
                <div style="font-size:0.7rem; color:#888; text-transform:uppercase;">
                    Channel {ch}
                </div>
                <div style="font-size:2rem; font-weight:700; color:{ratio_color}; margin:6px 0;">
                    {r_ratio:.4f}
                </div>
                <div style="font-size:0.7rem; color:#666;">R_gas / R_air</div>
                <hr style="border-color:#2a2d3e; margin:8px 0;">
                <table style="width:100%; font-size:0.72rem; color:#aaa;">
                    <tr><td>R_gas</td><td style="text-align:right;">{r_ohm/1000:.1f} kΩ</td></tr>
                    <tr><td>Heater</td><td style="text-align:right;">{phase}</td></tr>
                    <tr><td>Drift</td>
                        <td style="text-align:right; color:{drift_color};">
                            {drift_stat.upper()} ({drift_mag*100:.1f}%)
                        </td>
                    </tr>
                    <tr><td>Health</td>
                        <td style="text-align:right; color:{drift_color};">
                            {health_pct:.0f}%
                        </td>
                    </tr>
                    <tr><td>Baseline</td>
                        <td style="text-align:right; color:{'#2ecc71' if locked else '#f39c12'};">
                            {'✓' if locked else '…'}
                        </td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)


def render_timeseries(history: list[dict]):
    if not history:
        st.info("Waiting for historical data...")
        return

    rows = []
    for snap in history:
        ts = snap.get("timestamp_utc", "")
        for ch_data in snap.get("channels", []):
            rows.append({
                "timestamp": ts,
                "channel":   f"Ch{ch_data['ch']}",
                "r_ratio":   ch_data.get("r_ratio", 1.0),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df_pivot = df.pivot_table(index="timestamp", columns="channel", values="r_ratio", aggfunc="last")

    st.line_chart(df_pivot, height=240, use_container_width=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main():
    st.markdown("## 🔬 e-Nose Industrial Monitor")
    st.markdown(
        '<span style="color:#555; font-size:0.85rem;">Real-time gas detection and quantification</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Controls")
        if st.button("🔄 Force Recalibration", use_container_width=True):
            try:
                r = requests.post(f"{GATEWAY_URL}/api/v1/recalibrate", timeout=3)
                st.success("Recalibration command sent!")
            except Exception as e:
                st.error(f"Failed: {e}")

        st.markdown("---")
        st.markdown("### Info")
        health = fetch_health()
        stats  = health.get("gateway_stats", {})
        st.metric("Messages received",  stats.get("message_count", 0))
        st.metric("Throughput",         f"{stats.get('msg_per_sec', 0)} msg/s")
        st.metric("Uptime",             f"{stats.get('uptime_s', 0):.0f} s")
        st.markdown(f"**API:** [{GATEWAY_URL}]({GATEWAY_URL}/docs)")

    # Fetch data
    snap    = fetch_latest()
    history = fetch_history(HISTORY_N)
    health  = fetch_health()

    if snap is None:
        st.error("⚠️ Cannot reach Gateway API. Check that the service is running.")
        st.code(f"Expected: GET {GATEWAY_URL}/api/v1/latest")
        time.sleep(3)
        st.rerun()
        return

    # ---- Row 1: Gas identification panel ----
    render_gas_panel(snap)

    # ---- Row 2: Anomaly feed ----
    st.markdown("**Anomaly Feed**")
    render_anomaly_feed(snap)
    st.markdown("---")

    # ---- Row 3: Sensor matrix ----
    st.markdown("**Sensor Array — R_gas / R_air Matrix**")
    render_sensor_matrix(snap, health)
    st.markdown("---")

    # ---- Row 4: Time-series ----
    st.markdown("**R_ratio Time Series — Last 60 s**")
    render_timeseries(history)

    # ---- Row 5: Raw data expander ----
    with st.expander("📋 Raw JSON Snapshot"):
        st.json(snap)

    # Auto-refresh
    time.sleep(POLL_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
