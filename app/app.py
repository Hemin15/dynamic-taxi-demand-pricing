import sys
import os
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.model import prepare_model_data, ALL_FEATURES
from src.pricing_engine import DynamicPricingEngine, optimize_for_row


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "outputs" / "data" / "demand_panel.csv"
MODEL_PATH = BASE_DIR / "models" / "demand_model.pkl"
ARTIFACTS_DIR = BASE_DIR / "outputs" / "artifacts"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"
FINAL_DIR = BASE_DIR / "outputs" / "final"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"


# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Pricing AI Dashboard",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(25,25,35,0.98), rgba(30,41,59,0.96));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        }
        .hero-title {
            font-size: 2.0rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
            color: white;
        }
        .hero-subtitle {
            font-size: 0.98rem;
            color: rgba(255,255,255,0.78);
        }
        .small-note {
            font-size: 0.88rem;
            color: rgba(255,255,255,0.72);
        }
        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(148,163,184,0.18);
            padding: 10px 12px;
            border-radius: 14px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(148,163,184,0.15);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Small utilities
# -------------------------------------------------------------------
def _safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _fmt_number(value, digits=2):
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "n/a"


def _file_text(path: Path):
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _json_default(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, default=_json_default)


# -------------------------------------------------------------------
# Cached loaders
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_optional_csv(path_str: str):
    path = Path(path_str)
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_optional_json(path_str: str):
    path = Path(path_str)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# -------------------------------------------------------------------
# Demo fallback data
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_demo_data(n_rows: int = 600):
    rng = np.random.default_rng(42)
    zones = [
        "40.64_-73.78",
        "40.71_-73.98",
        "40.73_-73.98",
        "40.76_-73.97",
        "40.77_-73.93",
        "40.79_-73.95",
        "40.81_-73.97",
        "40.83_-73.95",
    ]

    rows = []
    for i in range(n_rows):
        hour = int(rng.integers(0, 24))
        dow = int(rng.integers(0, 7))
        month = int(rng.integers(1, 13))
        zone = zones[int(rng.integers(0, len(zones)))]
        base = float(rng.lognormal(mean=2.2, sigma=0.7))
        demand = max(1.0, base * (1.0 + 0.35 * (hour in [7, 8, 9, 16, 17, 18, 19])))

        row = {
            "pickup_date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 31))),
            "hour": hour,
            "day_of_week": dow,
            "month": month,
            "week_of_year": int(rng.integers(1, 53)),
            "is_weekend": int(dow >= 5),
            "is_peak_hour": int(hour in [7, 8, 9, 16, 17, 18, 19]),
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
            "pickup_lat_bin": float(rng.uniform(40.60, 40.85)),
            "pickup_lon_bin": float(rng.uniform(-74.05, -73.75)),
            "zone_trip_volume": float(rng.integers(20, 500)),
            "date_ordinal": float(pd.Timestamp("2015-01-01").toordinal() + int(rng.integers(0, 31))),
            "lag_1": float(max(1.0, demand * rng.uniform(0.85, 1.10))),
            "lag_2": float(max(1.0, demand * rng.uniform(0.80, 1.05))),
            "lag_24": float(max(1.0, demand * rng.uniform(0.75, 1.20))),
            "rolling_mean_3": float(max(1.0, demand * rng.uniform(0.90, 1.08))),
            "rolling_mean_24": float(max(1.0, demand * rng.uniform(0.85, 1.15))),
            "rolling_std_24": float(rng.uniform(1.0, 18.0)),
            "expanding_mean_zone": float(max(1.0, demand * rng.uniform(0.90, 1.12))),
            "pickup_zone": zone,
            "demand": float(demand),
            "avg_fare_amount": float(rng.uniform(8, 45)),
            "avg_trip_distance": float(rng.uniform(1, 12)),
            "avg_duration_min": float(rng.uniform(4, 45)),
            "avg_speed_kph": float(rng.uniform(8, 45)),
        }
        rows.append(row)

    demo_df = pd.DataFrame(rows)
    return prepare_model_data(demo_df)


def build_context_row(prepared_df: pd.DataFrame, zone: str, hour: int) -> pd.Series:
    """
    Build a realistic context row for a chosen zone and hour.
    """
    subset = prepared_df[(prepared_df["pickup_zone"] == zone) & (prepared_df["hour"] == hour)].copy()

    if subset.empty:
        subset = prepared_df[prepared_df["pickup_zone"] == zone].copy()
    if subset.empty:
        subset = prepared_df[prepared_df["hour"] == hour].copy()
    if subset.empty:
        subset = prepared_df.copy()

    row = {}

    for col in ALL_FEATURES:
        if col == "hour":
            row[col] = hour
        elif col == "hour_sin":
            row[col] = float(np.sin(2 * np.pi * hour / 24))
        elif col == "hour_cos":
            row[col] = float(np.cos(2 * np.pi * hour / 24))
        elif col == "is_peak_hour":
            row[col] = int(hour in [7, 8, 9, 16, 17, 18, 19])
        elif col == "dow_sin":
            dow = int(subset["day_of_week"].median()) if "day_of_week" in subset.columns else 0
            row[col] = float(np.sin(2 * np.pi * dow / 7))
        elif col == "dow_cos":
            dow = int(subset["day_of_week"].median()) if "day_of_week" in subset.columns else 0
            row[col] = float(np.cos(2 * np.pi * dow / 7))
        elif col in [
            "pickup_lat_bin", "pickup_lon_bin", "zone_trip_volume", "date_ordinal",
            "lag_1", "lag_2", "lag_24", "rolling_mean_3", "rolling_mean_24",
            "rolling_std_24", "expanding_mean_zone"
        ]:
            row[col] = _safe_float(subset[col].median()) if col in subset.columns else 0.0
        elif col in ["day_of_week", "month", "week_of_year", "is_weekend"]:
            row[col] = int(subset[col].median()) if col in subset.columns else 0
        else:
            if col in subset.columns and pd.api.types.is_numeric_dtype(subset[col]):
                row[col] = _safe_float(subset[col].median())
            else:
                row[col] = 0.0

    return pd.Series(row)


def predict_no_warnings(model, row_df):
    try:
        # keep dataframe columns intact
        return float(model.predict(row_df)[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0


def safe_demo_predict(prepared_df: pd.DataFrame, context_row: pd.Series, model=None) -> float:
    """
    Use the real model when available; otherwise use a stable heuristic fallback.
    """
    if model is not None:
        row_df = pd.DataFrame([context_row[ALL_FEATURES].to_dict()])
        return predict_no_warnings(model, row_df)

    base = float(prepared_df["demand"].median()) if "demand" in prepared_df.columns else 25.0
    hour = int(context_row["hour"])
    peak_boost = 1.25 if hour in [7, 8, 9, 16, 17, 18, 19] else 1.0
    weekend_adj = 0.92 if int(context_row.get("is_weekend", 0)) == 1 else 1.0
    zone_adj = 1.0 + min(float(context_row.get("zone_trip_volume", 0)) / 1200.0, 0.35)
    return max(1.0, base * peak_boost * weekend_adj * zone_adj)


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------
def plot_revenue_curve(curve_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(curve_df["price"], curve_df["revenue"], linewidth=2)
    ax.set_title("Revenue Curve Across Price Levels")
    ax.set_xlabel("Price")
    ax.set_ylabel("Expected Revenue")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def plot_sensitivity_curve(sensitivity_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sensitivity_df["elasticity"], sensitivity_df["optimal_price"], marker="o", label="Optimal price")
    ax.plot(sensitivity_df["elasticity"], sensitivity_df["expected_revenue"], marker="o", label="Expected revenue")
    ax.set_title("Pricing Sensitivity Across Elasticity")
    ax.set_xlabel("Elasticity")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def plot_comparison_bars(model_metrics: dict):
    if not model_metrics or not model_metrics.get("comparison"):
        return
    comp = pd.DataFrame(model_metrics["comparison"]).sort_values("val_mae")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(comp["model_name"], comp["val_mae"])
    ax.set_title("Validation MAE by Model")
    ax.set_ylabel("Validation MAE")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def plot_model_quality(model_metrics: dict):
    if not model_metrics:
        return
    train_mae = _safe_float(model_metrics.get("selected_model_train_metrics", {}).get("mae"))
    val_mae = _safe_float(model_metrics.get("selected_model_validation_metrics", {}).get("mae"))
    test_mae = _safe_float(model_metrics.get("selected_model_test_metrics", {}).get("mae"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["train", "validation", "test"], [train_mae, val_mae, test_mae])
    ax.set_title("Selected Model MAE Across Splits")
    ax.set_ylabel("MAE")
    ax.grid(True, axis="y", alpha=0.25)
    st.pyplot(fig, clear_figure=True)


# -------------------------------------------------------------------
# UI helpers
# -------------------------------------------------------------------
def make_scenario_table(pricing_result):
    return pd.DataFrame([
        {
            "scenario": "Static pricing",
            "price": pricing_result.static_price,
            "revenue": pricing_result.static_revenue,
        },
        {
            "scenario": "Dynamic pricing",
            "price": pricing_result.optimal_price,
            "revenue": pricing_result.expected_revenue,
        },
    ])


def make_kpi_cards(predicted_demand, ref_price, pricing_result):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted demand", _fmt_number(predicted_demand))
    c2.metric("Reference price", _fmt_number(ref_price))
    c3.metric("Optimal price", _fmt_number(pricing_result.optimal_price))
    c4.metric("Revenue uplift", f"{_fmt_number(pricing_result.uplift_percent)}%")


def build_fallback_final_report(best_model_name: str, pricing_result, ref_price, predicted_demand):
    return {
        "project": {
            "name": "dynamic taxi demand forecasting and pricing optimization",
            "model_type": best_model_name,
            "notes": "hybrid ai system combining machine learning demand prediction with rule-based pricing optimization",
        },
        "demo_case": {
            "pickup_date": "demo",
            "hour": int(st.session_state.get("hour", 11)),
            "pickup_zone": str(st.session_state.get("zone", "demo-zone")),
            "actual_demand": None,
            "predicted_demand": float(predicted_demand),
            "reference_price": float(ref_price),
            "static_price": float(pricing_result.static_price),
            "static_revenue": float(pricing_result.static_revenue),
            "optimal_price": float(pricing_result.optimal_price),
            "optimal_demand": float(pricing_result.expected_demand),
            "optimal_revenue": float(pricing_result.expected_revenue),
            "uplift_percent": float(pricing_result.uplift_percent),
        },
        "pricing_sensitivity": [],
        "batch_summary": {
            "sample_size": 0,
            "average_predicted_demand": float(predicted_demand),
            "average_static_revenue": float(pricing_result.static_revenue),
            "average_optimal_revenue": float(pricing_result.expected_revenue),
            "average_uplift_percent": float(pricing_result.uplift_percent),
        },
        "top_5_uplift_cases": [],
        "model_summary": {},
    }


# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🚖 AI-driven Dynamic Pricing Optimization System</div>
        <div class="hero-subtitle">
            Demand forecasting + pricing intelligence + neural-network benchmark + walk-forward evaluation
        </div>
        <div class="small-note">
            Built to stay usable both locally and on Streamlit Cloud, even when training artifacts are not deployed.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

data_exists = DATA_PATH.exists()
model_exists = MODEL_PATH.exists()

if not data_exists or not model_exists:
    st.warning(
        "Running in demo mode because the full data/model bundle is not available in this deployment. "
        "The dashboard remains interactive using a synthetic fallback dataset."
    )

with st.spinner("Loading data and artifacts..."):
    if data_exists:
        try:
            df = load_data()
            prepared_df = prepare_model_data(df)
            prepared_df = prepared_df.dropna(subset=ALL_FEATURES + ["demand"]).copy()
        except Exception:
            prepared_df = build_demo_data()
    else:
        prepared_df = build_demo_data()

    try:
        model = load_model() if model_exists else None
    except Exception:
        model = None

    model_metrics = load_optional_json(str(ARTIFACTS_DIR / "model_metrics.json"))
    cv_summary_df = load_optional_csv(str(ARTIFACTS_DIR / "cross_validation_summary.csv"))
    selection_df = load_optional_csv(str(ARTIFACTS_DIR / "model_selection_summary.csv"))
    nn_history_df = load_optional_csv(str(ARTIFACTS_DIR / "nn_training_history.csv"))
    feature_importance_df = load_optional_csv(str(ARTIFACTS_DIR / "feature_importance.csv"))
    test_predictions_df = load_optional_csv(str(PREDICTIONS_DIR / "test_predictions.csv"))
    sensitivity_df = load_optional_csv(str(FINAL_DIR / "sensitivity_curve.csv"))
    final_report = load_optional_json(str(FINAL_DIR / "final_report.json"))
    run_summary_text = _file_text(FINAL_DIR / "run_summary.txt")

zones = sorted(prepared_df["pickup_zone"].dropna().astype(str).unique().tolist())
if not zones:
    zones = ["demo-zone"]

# Sidebar
with st.sidebar:
    st.header("Scenario controls")
    zone = st.selectbox("Pickup zone", zones)
    hour = st.slider("Hour of day", 0, 23, 11)
    elasticity = st.slider("Elasticity assumption", 0.05, 1.00, 0.35, 0.05)
    st.caption("Higher elasticity means demand falls faster when price rises.")
    st.divider()
    st.subheader("Project status")
    st.write("Live pricing demo, model comparison, cross-validation, feature importance, and final outputs are included below.")

st.session_state["zone"] = zone
st.session_state["hour"] = hour

# Core scenario
context_row = build_context_row(prepared_df, zone, hour)
context_row_df = pd.DataFrame([context_row[ALL_FEATURES].to_dict()])
predicted_demand = safe_demo_predict(prepared_df, context_row, model)

pricing_engine = DynamicPricingEngine(
    min_price=5.0,
    max_price=100.0,
    price_step=1.0,
    elasticity=elasticity,
)

ref_price = pricing_engine._get_reference_price(context_row)
pricing_result, revenue_curve = optimize_for_row(
    row=context_row,
    predicted_demand=predicted_demand,
    min_price=5.0,
    max_price=100.0,
    price_step=1.0,
    elasticity=elasticity,
)

if final_report is None:
    final_report = build_fallback_final_report(
        best_model_name=(model_metrics or {}).get("best_model", "demo-mode"),
        pricing_result=pricing_result,
        ref_price=ref_price,
        predicted_demand=predicted_demand,
    )

# Hero KPI section
make_kpi_cards(predicted_demand, ref_price, pricing_result)

col_left, col_right = st.columns([1.15, 0.85])
with col_left:
    st.subheader("Pricing recommendation")
    st.json(pricing_result.__dict__)

    st.subheader("Scenario comparison")
    st.dataframe(make_scenario_table(pricing_result), use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Revenue curve")
    plot_revenue_curve(revenue_curve)

st.subheader("Revenue curve data")
st.dataframe(revenue_curve.head(15), use_container_width=True, hide_index=True)

# Executive summary
st.divider()
st.subheader("Project overview")
summary_cols = st.columns(3)
summary_cols[0].metric("Train rows", f"{model_metrics.get('train_rows', 0):,}" if model_metrics else "n/a")
summary_cols[1].metric("Validation rows", f"{model_metrics.get('val_rows', 0):,}" if model_metrics else "n/a")
summary_cols[2].metric("Test rows", f"{model_metrics.get('test_rows', 0):,}" if model_metrics else "n/a")

left_summary, right_summary = st.columns([1.2, 0.8])
with left_summary:
    st.write(
        "This dashboard combines a demand forecasting model with a revenue-maximizing pricing engine. "
        "The pipeline evaluates multiple classical regressors, gradient boosting models, and a neural-network benchmark, "
        "then surfaces the best-performing model for pricing decisions."
    )
    st.info(
        f"Best model: {final_report.get('project', {}).get('model_type', 'unknown')} | "
        f"Demo uplift: {final_report.get('demo_case', {}).get('uplift_percent', 0):.2f}% | "
        f"Batch uplift: {final_report.get('batch_summary', {}).get('average_uplift_percent', 0):.2f}%"
    )
with right_summary:
    st.write("Model selection snapshot")
    if model_metrics and model_metrics.get("best_model"):
        st.success(f"Selected model: {model_metrics['best_model']}")
        st.caption("Tree boosting won on validation; the neural network was tested as a benchmark.")
    else:
        st.info("Demo mode or model metadata not found.")

# Context explorer
st.divider()
st.subheader("Top demand context")
show_cols = [
    "pickup_date", "hour", "day_of_week", "pickup_zone", "demand",
    "avg_fare_amount", "avg_trip_distance", "avg_duration_min", "avg_speed_kph"
]
available_cols = [c for c in show_cols if c in prepared_df.columns]
zone_history = prepared_df[prepared_df["pickup_zone"] == zone].sort_values(["hour", "pickup_date"])
if available_cols:
    st.dataframe(zone_history[available_cols].head(10), use_container_width=True, hide_index=True)
else:
    st.info("Context table not available in this deployment.")

# Model quality
st.divider()
st.subheader("Model quality summary")
if model_metrics:
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Best model", model_metrics.get("best_model", "unknown"))
    q2.metric("Train MAE", _fmt_number(model_metrics.get("selected_model_train_metrics", {}).get("mae")))
    q3.metric("Validation MAE", _fmt_number(model_metrics.get("selected_model_validation_metrics", {}).get("mae")))
    q4.metric("Test MAE", _fmt_number(model_metrics.get("selected_model_test_metrics", {}).get("mae")))

    quality_left, quality_right = st.columns([1.05, 0.95])
    with quality_left:
        selected_metrics = pd.DataFrame([
            {
                "train_mae": model_metrics.get("selected_model_train_metrics", {}).get("mae"),
                "val_mae": model_metrics.get("selected_model_validation_metrics", {}).get("mae"),
                "test_mae": model_metrics.get("selected_model_test_metrics", {}).get("mae"),
                "train_rmse": model_metrics.get("selected_model_train_metrics", {}).get("rmse"),
                "val_rmse": model_metrics.get("selected_model_validation_metrics", {}).get("rmse"),
                "test_rmse": model_metrics.get("selected_model_test_metrics", {}).get("rmse"),
                "train_r2": model_metrics.get("selected_model_train_metrics", {}).get("r2"),
                "val_r2": model_metrics.get("selected_model_validation_metrics", {}).get("r2"),
                "test_r2": model_metrics.get("selected_model_test_metrics", {}).get("r2"),
                "generalization_gap_mae": model_metrics.get("generalization_gap_mae"),
                "generalization_gap_rmse": model_metrics.get("generalization_gap_rmse"),
                "overfitting_suspected": model_metrics.get("overfitting_suspected"),
            }
        ])
        st.dataframe(selected_metrics, use_container_width=True, hide_index=True)

    with quality_right:
        plot_model_quality(model_metrics)
else:
    st.info("Model metrics not available in this deployment.")

with st.expander("Model comparison", expanded=False):
    if model_metrics and model_metrics.get("comparison"):
        comp_df = pd.DataFrame(model_metrics["comparison"]).sort_values("val_mae")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        plot_comparison_bars(model_metrics)
    else:
        st.info("Comparison table not found.")

with st.expander("Cross-validation summary", expanded=False):
    if cv_summary_df is not None and len(cv_summary_df) > 0:
        st.dataframe(cv_summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("Cross-validation summary not found.")

with st.expander("Model selection comparison", expanded=False):
    if selection_df is not None and len(selection_df) > 0:
        st.dataframe(selection_df, use_container_width=True, hide_index=True)
    else:
        st.info("Model selection summary not found.")

with st.expander("Feature importance", expanded=False):
    if feature_importance_df is not None and len(feature_importance_df) > 0:
        st.dataframe(feature_importance_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("Feature importance file not found.")

with st.expander("Neural network training history", expanded=False):
    if nn_history_df is not None and len(nn_history_df) > 0:
        st.dataframe(nn_history_df, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(nn_history_df["epoch"], nn_history_df["train_loss"], label="train_loss")
        if "val_rmse" in nn_history_df.columns:
            ax.plot(nn_history_df["epoch"], nn_history_df["val_rmse"], label="val_rmse")
        ax.set_title("Neural Network Training History")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Neural-network history not available for the selected training run.")

with st.expander("Pricing sensitivity", expanded=False):
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)
        plot_sensitivity_curve(sensitivity_df)
    else:
        st.info("Sensitivity analysis file not found.")

with st.expander("Test-set predictions sample", expanded=False):
    if test_predictions_df is not None and len(test_predictions_df) > 0:
        st.dataframe(test_predictions_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("Test prediction file not found.")

with st.expander("Final report snapshot", expanded=False):
    if final_report:
        st.json(final_report)
    else:
        st.info("Final report not found.")

with st.expander("Run summary", expanded=False):
    if run_summary_text:
        st.text(run_summary_text)
    else:
        st.info("Run summary not found.")

# Downloads and artifacts
st.divider()
st.subheader("Artifacts")
art_cols = st.columns(3)
with art_cols[0]:
    if (FINAL_DIR / "final_report.json").exists():
        st.download_button(
            "Download final report",
            data=(FINAL_DIR / "final_report.json").read_bytes(),
            file_name="final_report.json",
            mime="application/json",
            use_container_width=True,
        )
with art_cols[1]:
    if (FINAL_DIR / "pricing_simulation_summary.csv").exists():
        st.download_button(
            "Download pricing summary",
            data=(FINAL_DIR / "pricing_simulation_summary.csv").read_bytes(),
            file_name="pricing_simulation_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
with art_cols[2]:
    if (FINAL_DIR / "sensitivity_curve.csv").exists():
        st.download_button(
            "Download sensitivity curve",
            data=(FINAL_DIR / "sensitivity_curve.csv").read_bytes(),
            file_name="sensitivity_curve.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.caption(
    "This dashboard is built from the full training pipeline: preprocessing, EDA, time-series cross-validation, "
    "classical model comparison, neural-network benchmark, and rule-based revenue optimization."
)