"""Streamlit dashboard: professional energy forecasting analytics interface."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ── Environment & Config ──────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")
DATA_PATH = os.environ.get("PROCESSED_DATA_PATH", "data/processed/energy_clean.csv")

st.set_page_config(
    page_title="Energy Demand Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Professional Styling (Clean, Modern Engineering / Analytics) ──────
st.markdown(
    """
    <style>
    /* Global font & clean canvas */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main container spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Top banner */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.1rem 1.4rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .header-title-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .header-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        color: #64748B;
        font-size: 0.88rem;
        margin: 0.2rem 0 0 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    .status-pill-offline {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }
    .chip-link {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #334155 !important;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        text-decoration: none !important;
        transition: all 0.15s ease;
    }
    .chip-link:hover {
        background: #E2E8F0;
        color: #0F172A !important;
    }

    /* Cards */
    .dashboard-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .dashboard-card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0F172A;
        margin: 0 0 0.85rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Metric highlights */
    .kpi-tile {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        text-align: left;
    }
    .kpi-label {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.2rem;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] {
        background-color: #FAFAFC;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* Button styling */
    div.stButton > button {
        background: #2563EB;
        color: #FFFFFF;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        border: none;
        padding: 0.55rem 1.2rem;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:hover {
        background: #1D4ED8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    /* Clean expander */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #334155 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Fetch Model Information ──────────────────────────────────────────────────
@st.cache_data(ttl=15)
def get_model_info() -> dict[str, Any] | None:
    try:
        resp = requests.get(f"{API_URL}/model/info", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


model_info = get_model_info()
is_api_healthy = model_info is not None

# ── Top Application Header ────────────────────────────────────────────────────
status_badge_html = (
    '<span class="status-pill">● Model Online</span>'
    if is_api_healthy
    else '<span class="status-pill-offline">● API Offline</span>'
)

st.markdown(
    f"""
    <div class="header-container">
        <div>
            <div class="header-title-row">
                <span style="font-size: 1.6rem;">⚡</span>
                <h1 class="header-title">Home Energy Demand Forecaster</h1>
                {status_badge_html}
            </div>
            <p class="header-subtitle">
                Recursive multi-step electricity consumption forecasting with XGBoost quantile regression
            </p>
        </div>
        <div style="display: flex; gap: 0.5rem; align-items: center;">
            <a href="http://localhost:8000/docs" target="_blank" class="chip-link">
                <span>📘</span> API Docs
            </a>
            <a href="http://localhost:5050" target="_blank" class="chip-link">
                <span>🧪</span> MLflow UI
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar: Model Specs & Telemetry ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Telemetry")

    if is_api_healthy and model_info:
        # Key Evaluation Metrics
        mae = model_info.get("test_mae", "—")
        rmse = model_info.get("test_rmse", "—")
        skill = model_info.get("skill_vs_daily_baseline")
        skill_str = f"{skill:.1%}" if isinstance(skill, (int, float)) else "—"

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Test MAE", f"{mae} kW")
        with col_m2:
            st.metric("Test RMSE", f"{rmse} kW")

        st.metric("Skill vs Daily Baseline", skill_str, help="Forecast accuracy improvement over naive daily baseline")

        st.markdown("---")

        with st.expander("📌 Benchmark Baselines"):
            st.markdown(
                f"""
                - **Naive Last-Hour MAE:** `{model_info.get('baseline_naive_last_mae', '—')}` kW
                - **Naive Daily MAE:** `{model_info.get('baseline_naive_daily_mae', '—')}` kW
                - **Naive Weekly MAE:** `{model_info.get('baseline_naive_weekly_mae', '—')}` kW
                """
            )

        with st.expander("⚙️ Architecture & Splits"):
            features_list = model_info.get("features", [])
            st.markdown(
                f"""
                - **Model:** `XGBoost Quantile Regressor`
                - **Quantiles:** P10 · P50 (median) · P90
                - **Train / Val / Test:** `{model_info.get('train_size', '—'):,}` / `{model_info.get('val_size', '—'):,}` / `{model_info.get('test_size', '—'):,}` hrs
                - **Engineered Features ({len(features_list)}):**
                """
            )
            st.caption(", ".join(features_list))

    else:
        st.warning("⚠️ FastAPI server not reachable at `http://localhost:8000`.")
        st.info("Check container logs: `docker compose logs -f`")


# ── Forecast Controls & Visualization ─────────────────────────────────────────
if "horizon" not in st.session_state:
    st.session_state.horizon = 24

st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
st.markdown('<div class="dashboard-card-title">📈 Generate Forward Forecast</div>', unsafe_allow_html=True)

# Quick preset buttons & fine-tuning slider
col_preset, col_slider = st.columns([1, 2])

with col_preset:
    st.markdown("<p style='font-size: 0.82rem; font-weight: 600; color: #475569; margin-bottom: 0.35rem;'>Quick Presets</p>", unsafe_allow_html=True)
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("24h (1 Day)", use_container_width=True):
            st.session_state.horizon = 24
        if st.button("72h (3 Days)", use_container_width=True):
            st.session_state.horizon = 72
    with p_col2:
        if st.button("48h (2 Days)", use_container_width=True):
            st.session_state.horizon = 48
        if st.button("168h (1 Wk)", use_container_width=True):
            st.session_state.horizon = 168

with col_slider:
    st.markdown("<p style='font-size: 0.82rem; font-weight: 600; color: #475569; margin-bottom: 0.35rem;'>Custom Horizon (Hours)</p>", unsafe_allow_html=True)
    selected_horizon = st.slider(
        label="Select forecast horizon",
        min_value=1,
        max_value=168,
        value=st.session_state.horizon,
        label_visibility="collapsed",
    )
    st.session_state.horizon = selected_horizon

btn_col1, btn_col2 = st.columns([1, 4])
with btn_col1:
    run_clicked = st.button("Generate Forecast", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# Fetch or render forecast
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None
    st.session_state.forecast_meta = None

# If user clicked button or initial run
if run_clicked or st.session_state.forecast_data is None:
    if is_api_healthy:
        with st.spinner("Generating recursive multi-step predictions…"):
            try:
                resp = requests.get(
                    f"{API_URL}/forecast",
                    params={"horizon": st.session_state.horizon},
                    timeout=30,
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    df_fc = pd.DataFrame(payload["forecast"])
                    df_fc["timestamp"] = pd.to_datetime(df_fc["timestamp"])
                    st.session_state.forecast_data = df_fc
                    st.session_state.forecast_meta = payload
                else:
                    st.error(f"API returned error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Failed to communicate with prediction service: {e}")

# Render Forecast Section if data is available
if st.session_state.forecast_data is not None:
    fc_df = st.session_state.forecast_data
    meta = st.session_state.forecast_meta

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    
    # Header summary row
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;">
            <div class="dashboard-card-title" style="margin-bottom: 0;">
                <span>🔮</span> Forecast Trajectory ({len(fc_df)} Hours Ahead)
            </div>
            <span style="font-size: 0.8rem; color: #64748B;">
                Forecast Origin: <b>{pd.to_datetime(meta['from_timestamp']).strftime('%Y-%m-%d %H:00')}</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key statistics cards
    peak_val = fc_df["p50"].max()
    peak_time = fc_df.loc[fc_df["p50"].idxmax(), "timestamp"].strftime("%a %H:00")
    avg_val = fc_df["p50"].mean()
    spread_avg = (fc_df["p90"] - fc_df["p10"]).mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-tile">
                <div class="kpi-label">Peak Predicted Load</div>
                <div class="kpi-value">{peak_val:.3f} <span style="font-size: 0.85rem; font-weight: normal; color: #64748B;">kW</span></div>
                <div class="kpi-sub">Expected at {peak_time}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-tile">
                <div class="kpi-label">Average Demand</div>
                <div class="kpi-value">{avg_val:.3f} <span style="font-size: 0.85rem; font-weight: normal; color: #64748B;">kW</span></div>
                <div class="kpi-sub">Over {len(fc_df)} hours</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-tile">
                <div class="kpi-label">Min Predicted Load</div>
                <div class="kpi-value">{fc_df['p50'].min():.3f} <span style="font-size: 0.85rem; font-weight: normal; color: #64748B;">kW</span></div>
                <div class="kpi-sub">Baseline minimum</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-tile">
                <div class="kpi-label">80% Uncertainty Band</div>
                <div class="kpi-value">±{spread_avg / 2:.3f} <span style="font-size: 0.85rem; font-weight: normal; color: #64748B;">kW</span></div>
                <div class="kpi-sub">Average P10–P90 spread</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Altair Forecast Visualization
    band = (
        alt.Chart(fc_df)
        .mark_area(opacity=0.22, color="#2563EB")
        .encode(
            x=alt.X(
                "timestamp:T",
                title="Timeline",
                axis=alt.Axis(
                    format="%b %d %H:00",
                    gridColor="#F1F5F9",
                    labelColor="#64748B",
                    titleColor="#475569",
                ),
            ),
            y=alt.Y(
                "p10:Q",
                title="Active Power (kW)",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(
                    gridColor="#F1F5F9",
                    labelColor="#64748B",
                    titleColor="#475569",
                ),
            ),
            y2="p90:Q",
        )
    )

    median_line = (
        alt.Chart(fc_df)
        .mark_line(color="#1D4ED8", strokeWidth=2.6, interpolate="monotone")
        .encode(
            x="timestamp:T",
            y="p50:Q",
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%a, %b %d %H:00"),
                alt.Tooltip("p50:Q", title="Predicted (P50)", format=".3f"),
                alt.Tooltip("p90:Q", title="Upper 80% (P90)", format=".3f"),
                alt.Tooltip("p10:Q", title="Lower 80% (P10)", format=".3f"),
            ],
        )
    )

    forecast_chart = (
        (band + median_line)
        .properties(height=320)
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(forecast_chart, use_container_width=True)

    st.markdown(
        """
        <div style="display: flex; gap: 1.5rem; justify-content: center; font-size: 0.8rem; color: #64748B; margin-top: -0.5rem; margin-bottom: 0.5rem;">
            <span><b style="color: #1D4ED8;">―</b> P50 Median Forecast</span>
            <span><span style="display: inline-block; width: 12px; height: 10px; background: rgba(37, 99, 235, 0.25); border-radius: 2px;"></span> 80% Prediction Interval (P10–P90)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 View & Export Tabular Forecast Data"):
        export_col1, export_col2 = st.columns([3, 1])
        with export_col2:
            csv_data = fc_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"energy_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.dataframe(
            fc_df.rename(
                columns={
                    "timestamp": "Timestamp",
                    "p10": "P10 Lower (kW)",
                    "p50": "P50 Median (kW)",
                    "p90": "P90 Upper (kW)",
                }
            ),
            use_container_width=True,
            height=200,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ── Recent Actual Consumption ────────────────────────────────────────────────
st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
st.markdown('<div class="dashboard-card-title"><span>🕒</span> Recent Actual Consumption (Past 3 Days)</div>', unsafe_allow_html=True)

try:
    if os.path.exists(DATA_PATH):
        df_hist = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")
        cutoff = df_hist.index.max() - pd.Timedelta(days=3)
        recent_series = df_hist.loc[df_hist.index >= cutoff, "Global_active_power"].reset_index()

        # Summary KPIs for recent actuals
        latest_obs = recent_series["Global_active_power"].iloc[-1]
        recent_peak = recent_series["Global_active_power"].max()
        recent_avg = recent_series["Global_active_power"].mean()

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="kpi-label">Latest Observed Reading</div>
                    <div class="kpi-value">{latest_obs:.3f} <span style="font-size: 0.85rem; font-weight: normal; color: #64748B;">kW</span></div>
                    <div class="kpi-sub">{recent_series['datetime'].iloc[-1].strftime('%a, %b %d %H:00')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rc2:
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="kpi-label">3-Day Peak Demand</div>
                    <div class="kpi-value">{recent_peak:.3f} <span style="font-size: 0.85rem; font-weight: normal; color: #64748B;">kW</span></div>
                    <div class="kpi-sub">Maximum observed power</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rc3:
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="kpi-label">3-Day Average Demand</div>
                    <div class="kpi-value">{recent_avg:.3f} <span style="font-size: 0.85rem; font-weight: normal; color: #64748B;">kW</span></div>
                    <div class="kpi-sub">Historical baseline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Altair chart for actuals (warm slate/amber palette)
        hist_area = (
            alt.Chart(recent_series)
            .mark_area(opacity=0.12, color="#0F766E")
            .encode(
                x=alt.X(
                    "datetime:T",
                    title="Timeline",
                    axis=alt.Axis(
                        format="%b %d %H:00",
                        gridColor="#F1F5F9",
                        labelColor="#64748B",
                        titleColor="#475569",
                    ),
                ),
                y=alt.Y(
                    "Global_active_power:Q",
                    title="Active Power (kW)",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(
                        gridColor="#F1F5F9",
                        labelColor="#64748B",
                        titleColor="#475569",
                    ),
                ),
            )
        )

        hist_line = (
            alt.Chart(recent_series)
            .mark_line(color="#0F766E", strokeWidth=2.2, interpolate="monotone")
            .encode(
                x="datetime:T",
                y="Global_active_power:Q",
                tooltip=[
                    alt.Tooltip("datetime:T", title="Timestamp", format="%a, %b %d %H:00"),
                    alt.Tooltip("Global_active_power:Q", title="Observed Power", format=".3f"),
                ],
            )
        )

        recent_chart = (
            (hist_area + hist_line)
            .properties(height=240)
            .configure_view(strokeWidth=0)
        )

        st.altair_chart(recent_chart, use_container_width=True)

        st.markdown(
            f"""
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.35rem; display: flex; justify-content: space-between;">
                <span>Historical series: {df_hist.index.min().strftime('%Y-%m-%d')} → {df_hist.index.max().strftime('%Y-%m-%d')}</span>
                <span><b>{len(df_hist):,}</b> hourly clean observations</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Processed dataset file not found at specified path.")
except Exception as e:
    st.error(f"Error reading historical dataset: {e}")

st.markdown("</div>", unsafe_allow_html=True)
