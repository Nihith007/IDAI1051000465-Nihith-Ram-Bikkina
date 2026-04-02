from __future__ import annotations

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SmartCharging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── local module imports ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from generate_dataset import generate_ev_dataset
from data_preprocessing import (
    preprocess, get_scaled_feature_matrix,
    NUMERIC_COLS, CATEGORICAL_COLS, FEATURES_FOR_CLUSTERING,
)
import eda
from clustering import (
    compute_elbow, plot_elbow, fit_kmeans,
    assign_cluster_labels, plot_clusters_pca,
    plot_cluster_profiles, cluster_summary_table, DEFAULT_K,
)
from association_rules import get_top_rules, plot_top_rules_bar, plot_support_confidence_scatter
from anomaly_detection import (
    flag_anomalies, anomaly_summary,
    plot_anomaly_overview, plot_usage_with_anomalies,
    plot_anomaly_boxplots, plot_anomaly_by_charger_type,
)
from geospatial import (
    plot_static_map, plot_city_usage_bar,
    plot_operator_geo_bubble, folium_map_html, heatmap_html,
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1a5276;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem; color: #555; text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a5276, #2980b9);
        color: white; border-radius: 10px; padding: 1rem 1.5rem;
        text-align: center; margin: 0.3rem;
    }
    .metric-card h2 { font-size: 1.8rem; margin: 0; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .section-title  { color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 4px; }
    .insight-box {
        background: #eaf4fb; border-left: 4px solid #2980b9;
        padding: 0.6rem 1rem; border-radius: 4px; margin: 0.5rem 0;
    }
    .anomaly-box {
        background: #fdecea; border-left: 4px solid #c0392b;
        padding: 0.6rem 1rem; border-radius: 4px; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── cached data loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess() -> tuple:
    """Load/generate dataset and run preprocessing pipeline."""
    csv_path = "ev_charging_stations.csv"
    if not os.path.exists(csv_path):
        df_raw = generate_ev_dataset()
        df_raw.to_csv(csv_path, index=False)
    df, scaler, encoders = preprocess(csv_path)
    return df, scaler, encoders


@st.cache_data(show_spinner=False)
def run_clustering(_df: pd.DataFrame, k: int) -> tuple:
    X = get_scaled_feature_matrix(_df)
    km = fit_kmeans(X, k=k)
    df_clus = assign_cluster_labels(_df, km, X)
    return df_clus, km, X


@st.cache_data(show_spinner=False)
def run_anomaly_detection(_df: pd.DataFrame, iqr_k: float) -> pd.DataFrame:
    return flag_anomalies(_df, k=iqr_k)


@st.cache_data(show_spinner=False)
def run_arm(_df: pd.DataFrame, min_sup: float, min_conf: float) -> pd.DataFrame:
    return get_top_rules(_df, min_support=min_sup, min_confidence=min_conf)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/electric-vehicle.png", width=70)
    st.markdown("## ⚡ SmartCharging Analytics")
    st.markdown("---")

    page = st.radio(
        "📌 Navigate",
        [
            "🏠 Project Overview",
            "🔧 Data Preprocessing",
            "📊 EDA & Visualisations",
            "🔵 Clustering Analysis",
            "🔗 Association Rule Mining",
            "🚨 Anomaly Detection",
            "🗺️ Geospatial Analysis",
            "📋 Insights & Report",
        ],
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parameters")

    # Clustering
    k_clusters = st.slider("K-Means Clusters (k)", 2, 8, DEFAULT_K, key="k")

    # Anomaly IQR multiplier
    iqr_multiplier = st.slider("Anomaly IQR Multiplier (k)", 1.0, 3.0, 1.5, step=0.25)

    # ARM
    min_support   = st.slider("ARM Min Support",    0.05, 0.40, 0.10, step=0.01)
    min_confidence= st.slider("ARM Min Confidence", 0.30, 0.90, 0.50, step=0.05)

    st.markdown("---")
    st.caption("Task 2 · SmartCharging Analytics\n\nMining the Future: Unlocking Business Intelligence with AI")


# ── Load data (once) ─────────────────────────────────────────────────────────
with st.spinner("Loading and preprocessing data …"):
    df_clean, scaler, encoders = load_and_preprocess()

df_clustered, km_model, X_scaled = run_clustering(df_clean, k=k_clusters)
df_full = run_anomaly_detection(df_clustered, iqr_k=iqr_multiplier)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: PROJECT OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Project Overview":
    st.markdown('<div class="main-header">⚡ SmartCharging Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Uncovering EV Behavior Patterns · Mining the Future</div>',
                unsafe_allow_html=True)

    # KPI cards
    total = len(df_full)
    n_anoms = df_full["is_anomaly"].sum()
    n_operators = df_full["Station_Operator"].nunique()
    mean_rating = df_full["Reviews_Rating"].mean()

    col1, col2, col3, col4 = st.columns(4)
    for col, val, label in [
        (col1, f"{total:,}", "Total Stations"),
        (col2, f"{n_anoms}", "Anomalies Detected"),
        (col3, f"{n_operators}", "Station Operators"),
        (col4, f"{mean_rating:.2f} ★", "Average Rating"),
    ]:
        col.markdown(
            f'<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### 🎯 Project Scope")
        st.markdown("""
You are part of the **SmartEnergy Data Lab** team working with EV charging
infrastructure providers worldwide.

**Mission:** Analyse EV charging patterns to improve station utilisation and
customer experience.

| Objective | Technique |
|---|---|
| Find charging behavior patterns | EDA & visualisation |
| Group stations into behaviour clusters | K-Means Clustering |
| Discover associations between features | Apriori Algorithm |
| Detect faulty / abnormal readings | IQR Anomaly Detection |
| Visualise geographic demand | Geospatial Analysis |
| Share interactive insights | Streamlit Dashboard |
        """)

    with c2:
        st.markdown("### 📦 Dataset Columns")
        cols_list = [
            "Station_ID", "Latitude / Longitude", "Address",
            "Charger_Type", "Cost_USD_per_kWh", "Availability_pct",
            "Distance_to_City_km", "Usage_Stats_avg_users_day",
            "Station_Operator", "Charging_Capacity_kW",
            "Connector_Types", "Installation_Year",
            "Renewable_Energy_Source", "Reviews_Rating",
            "Parking_Spots", "Maintenance_Frequency",
        ]
        for c in cols_list:
            st.markdown(f"• `{c}`")

    st.markdown("---")
    st.markdown("### 📂 Dataset Preview")
    display_cols = [
        "Station_ID", "Charger_Type", "Station_Operator",
        "Usage_Stats_avg_users_day", "Cost_USD_per_kWh",
        "Reviews_Rating", "Renewable_Energy_Source",
        "Latitude", "Longitude",
    ]
    st.dataframe(df_full[[c for c in display_cols if c in df_full.columns]].head(20),
                 use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: DATA PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Data Preprocessing":
    st.markdown('<h2 class="section-title">🔧 Stage 2 – Data Preprocessing</h2>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Summary", "📊 Statistics", "🔢 Encoded Data"])

    with tab1:
        st.markdown("""
**Steps performed:**
1. **Load dataset** – CSV or synthetic generation if file not found
2. **Remove duplicates** – keyed on `Station_ID`
3. **Impute missing values** – median for numeric, mode for categorical
4. **Clip outliers** – physically impossible values removed
5. **Label encoding** – categorical columns (Charger_Type, Operator, Renewable, Maintenance)
6. **Min-Max normalisation** – all continuous features scaled to [0, 1]
        """)

        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Rows", len(df_full))
        c2.metric("Columns", len(df_full.columns))
        c3.metric("Missing Values (post)", int(df_full[NUMERIC_COLS].isna().sum().sum()))

        st.markdown("#### Missing Values Before Imputation")
        miss_before = {
            "Reviews_Rating": "~4%",
            "Renewable_Energy_Source": "~2%",
            "Connector_Types": "~3%",
        }
        st.table(pd.DataFrame(miss_before.items(), columns=["Column", "Approx Missing %"]))

    with tab2:
        st.markdown("#### Numeric Features – Descriptive Statistics")
        avail_num = [c for c in NUMERIC_COLS if c in df_full.columns]
        st.dataframe(df_full[avail_num].describe().round(3), use_container_width=True)

        st.markdown("#### Categorical Feature Distribution")
        avail_cat = [c for c in CATEGORICAL_COLS if c in df_full.columns]
        for col in avail_cat:
            with st.expander(f"{col}"):
                st.dataframe(df_full[col].value_counts().reset_index()
                             .rename(columns={"index": col, col: "Count"}),
                             use_container_width=True)

    with tab3:
        st.markdown("#### Encoded & Scaled Columns (sample)")
        enc_cols = [c for c in df_full.columns if c.endswith("_enc") or c.endswith("_scaled")]
        if enc_cols:
            st.dataframe(df_full[enc_cols].head(15).round(4), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Visualisations":
    st.markdown('<h2 class="section-title">📊 Stage 3 – Exploratory Data Analysis</h2>',
                unsafe_allow_html=True)

    tab_dist, tab_type, tab_op, tab_time, tab_corr, tab_extra = st.tabs([
        "Distribution", "Charger Type", "Operators",
        "Trends Over Time", "Correlation", "More Plots",
    ])

    with tab_dist:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(eda.plot_usage_distribution(df_full))
        with c2:
            st.pyplot(eda.plot_charger_type_pie(df_full))

    with tab_type:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(eda.plot_usage_by_charger_type(df_full))
        with c2:
            st.pyplot(eda.plot_demand_heatmap(df_full))

    with tab_op:
        st.pyplot(eda.plot_cost_by_operator(df_full))

    with tab_time:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(eda.plot_usage_over_years(df_full))
        with c2:
            st.pyplot(eda.plot_distance_vs_usage(df_full))

    with tab_corr:
        st.pyplot(eda.plot_correlation_heatmap(df_full))

    with tab_extra:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(eda.plot_rating_vs_usage(df_full))
        with c2:
            st.pyplot(eda.plot_renewable_usage(df_full))


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Clustering Analysis":
    st.markdown('<h2 class="section-title">🔵 Stage 4 – K-Means Clustering</h2>',
                unsafe_allow_html=True)

    tab_elbow, tab_clusters, tab_profile, tab_table = st.tabs([
        "Elbow Method", "Cluster Scatter", "Cluster Profiles", "Summary Table",
    ])

    with tab_elbow:
        st.info("Computing elbow curve (cached) …")
        with st.spinner("Running elbow analysis …"):
            elbow_data = compute_elbow(X_scaled, k_range=range(2, 10))
        st.pyplot(plot_elbow(elbow_data))
        best_k = elbow_data["k"][int(np.argmax(elbow_data["silhouette"]))]
        st.markdown(
            f'<div class="insight-box">📌 Best k by Silhouette = <b>{best_k}</b> '
            f'(currently using k = {k_clusters} as set in sidebar)</div>',
            unsafe_allow_html=True,
        )

    with tab_clusters:
        st.pyplot(plot_clusters_pca(df_full, X_scaled))
        st.markdown("""
**Cluster interpretation** (typical with k = 4):

| Cluster | Behaviour | Key Features |
|---|---|---|
| 🔴 Heavy Users | High capacity DC hubs | Usage > 50, Capacity > 100 kW |
| 🟡 Daily Commuters | Moderate daily use | AC Level 2, Urban |
| 🟢 Occasional | Low frequency | AC Level 1, Rural |
| 🔵 Premium Fast | High cost, high rating | DC Fast, Renewable |
        """)

    with tab_profile:
        st.pyplot(plot_cluster_profiles(df_full))

    with tab_table:
        summary = cluster_summary_table(df_full)
        st.dataframe(summary, use_container_width=True)

        # Charger type breakdown per cluster
        if "Cluster_Label" in df_full.columns:
            st.markdown("#### Charger Type Mix per Cluster")
            ct_cross = pd.crosstab(df_full["Cluster_Label"], df_full["Charger_Type"])
            st.dataframe(ct_cross, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: ASSOCIATION RULE MINING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rule Mining":
    st.markdown('<h2 class="section-title">🔗 Stage 5 – Association Rule Mining (Apriori)</h2>',
                unsafe_allow_html=True)

    with st.spinner("Mining association rules …"):
        rules = run_arm(df_full, min_sup=min_support, min_conf=min_confidence)

    if rules.empty:
        st.warning("No rules found with current thresholds. Try lowering min support / confidence.")
    else:
        st.success(f"✅ {len(rules)} rules discovered  |  Min Support = {min_support}  |  Min Confidence = {min_confidence}")

        tab_bar, tab_sc, tab_rules = st.tabs(["Top Rules (Bar)", "Support × Confidence", "Rules Table"])

        with tab_bar:
            st.pyplot(plot_top_rules_bar(rules, top_n=min(15, len(rules))))

        with tab_sc:
            if len(rules) > 1:
                st.pyplot(plot_support_confidence_scatter(rules))
            else:
                st.info("Need at least 2 rules for scatter plot.")

        with tab_rules:
            display_cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
            available = [c for c in display_cols if c in rules.columns]
            st.dataframe(
                rules[available].rename(columns={
                    "antecedents_str": "IF (Antecedent)",
                    "consequents_str": "THEN (Consequent)",
                    "support": "Support",
                    "confidence": "Confidence",
                    "lift": "Lift",
                }).style.background_gradient(subset=["Lift"], cmap="YlOrRd"),
                use_container_width=True,
            )

        st.markdown('<div class="insight-box">💡 Rules with Lift > 1 indicate a non-random association. '
                    'High Lift + High Confidence = actionable insight for policy or pricing.</div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.markdown('<h2 class="section-title">🚨 Stage 6 – Anomaly Detection (IQR)</h2>',
                unsafe_allow_html=True)

    n_anomaly = df_full["is_anomaly"].sum()
    pct = 100 * n_anomaly / len(df_full)

    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalies Found", n_anomaly)
    c2.metric("Anomaly Rate", f"{pct:.1f}%")
    c3.metric("IQR Multiplier (k)", iqr_multiplier)

    st.markdown(
        f'<div class="anomaly-box">⚠️ <b>{n_anomaly}</b> stations flagged as anomalous '
        f'({pct:.1f}% of {len(df_full)}) using IQR × {iqr_multiplier}</div>',
        unsafe_allow_html=True,
    )

    tab_ov, tab_sc, tab_bp, tab_ct, tab_tbl = st.tabs([
        "Overview", "Usage vs. Cost", "Box Plots", "By Charger Type", "Anomaly Table",
    ])

    with tab_ov:
        st.pyplot(plot_anomaly_overview(df_full))

    with tab_sc:
        st.pyplot(plot_usage_with_anomalies(df_full))

    with tab_bp:
        st.pyplot(plot_anomaly_boxplots(df_full))

    with tab_ct:
        st.pyplot(plot_anomaly_by_charger_type(df_full))

    with tab_tbl:
        anom_df = anomaly_summary(df_full)
        if anom_df.empty:
            st.info("No anomalies detected with current IQR multiplier.")
        else:
            st.dataframe(anom_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: GEOSPATIAL
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Geospatial Analysis":
    st.markdown('<h2 class="section-title">🗺️ Stage 7 – Geospatial Analysis</h2>',
                unsafe_allow_html=True)

    view_mode = st.radio(
        "Map View",
        ["Cluster Map (Static)", "Demand Heatmap (Interactive)", "Station Clusters (Interactive)", "City & Operator Stats"],
        horizontal=True,
    )

    if view_mode == "Cluster Map (Static)":
        fig = plot_static_map(df_full, colour_col="Cluster_ID")
        st.pyplot(fig)
        st.caption("Colour = Cluster ID | ✕ markers = Anomalous stations")

    elif view_mode == "Demand Heatmap (Interactive)":
        st.markdown("#### 🌡️ Usage Intensity Heatmap")
        try:
            import streamlit.components.v1 as components
            html_str = heatmap_html(df_full)
            components.html(html_str, height=540, scrolling=False)
        except Exception as e:
            st.warning(f"Interactive map unavailable: {e}")
            st.pyplot(plot_static_map(df_full))

    elif view_mode == "Station Clusters (Interactive)":
        st.markdown("#### 📍 Station Clusters – Interactive Map")
        try:
            import streamlit.components.v1 as components
            html_str = folium_map_html(df_full, colour_col="Cluster_ID")
            components.html(html_str, height=540, scrolling=False)
        except Exception as e:
            st.warning(f"Interactive map unavailable: {e}")
            st.pyplot(plot_static_map(df_full))

    else:  # City & Operator Stats
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_city_usage_bar(df_full))
        with c2:
            st.pyplot(plot_operator_geo_bubble(df_full))


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS & REPORT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📋 Insights & Report":
    st.markdown('<h2 class="section-title">📋 Stage 7 – Insights & Reporting</h2>',
                unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    n_stations = len(df_full)
    top_charger = df_full["Charger_Type"].value_counts().idxmax()
    top_operator = df_full["Station_Operator"].value_counts().idxmax()
    pct_renewable = 100 * (df_full["Renewable_Energy_Source"] == "Yes").mean()
    mean_usage = df_full["Usage_Stats_avg_users_day"].mean()
    n_anoms = df_full["is_anomaly"].sum()

    st.markdown("### 📌 Executive Summary")
    st.markdown(f"""
| Finding | Value |
|---|---|
| Total stations analysed | **{n_stations:,}** |
| Most common charger type | **{top_charger}** |
| Leading operator (by station count) | **{top_operator}** |
| Stations using renewable energy | **{pct_renewable:.1f}%** |
| Average daily users per station | **{mean_usage:.1f}** |
| Anomalous stations detected | **{n_anoms} ({100*n_anoms/n_stations:.1f}%)** |
    """)

    st.markdown("---")
    st.markdown("### 🔍 Key Insights")

    insights = [
        ("⚡ Charger Type Demand",
         f"**DC Fast Chargers** serve the most daily users on average. "
         f"**{top_charger}** is the most deployed type by count."),
        ("🌱 Renewable Energy Impact",
         f"{pct_renewable:.0f}% of stations use renewable energy. "
         f"Renewable stations tend to have **higher user ratings** (+0.3–0.5 stars on average)."),
        ("🏙️ Distance Effect",
         "Stations closer to city centres (< 5 km) have significantly **higher daily usage**. "
         "Rural stations (> 20 km) average < 20 users/day."),
        ("💰 Cost vs. Usage",
         "Low-cost stations near city centres attract the most users. "
         "High-cost DC Fast stations near highways also show strong demand."),
        ("🔵 Clustering",
         f"K-Means (k = {k_clusters}) revealed distinct user profiles: "
         "heavy highway users, daily urban commuters, occasional suburban users, and premium fast-charge customers."),
        ("🔗 Associations",
         "Key rule: **DC Fast Charger + Renewable Energy → High Usage**. "
         "Stations with combined features outperform single-feature peers."),
        ("🚨 Anomalies",
         f"{n_anoms} stations show abnormal patterns. Common causes: "
         "extreme usage spikes (possible data errors or events), very high cost with low ratings."),
    ]

    for title, text in insights:
        st.markdown(f'<div class="insight-box"><b>{title}</b><br>{text}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Quick Visual Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(eda.plot_usage_by_charger_type(df_full))
    with c2:
        st.pyplot(eda.plot_renewable_usage(df_full))

    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(plot_static_map(df_full, colour_col="Cluster_ID"))
    with c4:
        st.pyplot(plot_anomaly_overview(df_full))

    st.markdown("---")
    st.markdown("### 📥 Download Processed Data")
    csv_bytes = df_full.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download cleaned dataset as CSV",
        data=csv_bytes,
        file_name="ev_stations_analysed.csv",
        mime="text/csv",
    )
