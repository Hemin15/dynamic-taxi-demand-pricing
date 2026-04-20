import sys
import os
import json
from pathlib import Path

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


# -------------------------------------------------------------------
# Small utilities
# -------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


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
# Prediction helpers
# -------------------------------------------------------------------
def predict_no_warnings(model, row_df: pd.DataFrame) -> float:
    """
    Predict using a NumPy array to avoid feature-name warnings from sklearn/lightgbm.
    """
    try:
        return float(model.predict(row_df.to_numpy(dtype=float))[0])
    except Exception:
        return float(model.predict(row_df)[0])


def build_context_row(prepared_df: pd.DataFrame, zone: str, hour: int) -> pd.Series:
    """
    Build a realistic context row for a chosen zone and hour.
    The row is populated with robust median-based fallbacks so the dashboard
    works even when some features are sparse in a local slice.
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
    comp = pd.DataFrame(model_metrics["comparison"])  # full comparison table
    comp = comp.sort_values("val_mae")

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


def human_summary(final_report, model_metrics, pricing_result):
    if not final_report:
        return "No final report found yet."
    best_model = final_report.get("project", {}).get("model_type", "unknown")
    avg_uplift = final_report.get("batch_summary", {}).get("average_uplift_percent", None)
    demo_uplift = final_report.get("demo_case", {}).get("uplift_percent", None)
    nn_used = bool(model_metrics.get("has_neural_network_branch", False)) if model_metrics else False
    return (
        f"Best model: {best_model}. "
        f"Demo uplift: {demo_uplift:.2f}% if available. "
        f"Batch uplift: {avg_uplift:.2f}% if available. "
        f"Neural network branch tested: {'yes' if nn_used else 'no'}."
    )


# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
st.title("🚖 AI-driven Dynamic Pricing Optimization System")
st.caption("Demand forecasting + pricing intelligence + neural-network benchmark + walk-forward evaluation")

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Required files are missing. Run preprocessing, model training, and src/main.py first.")
    st.stop()

# Load artifacts
with st.spinner("Loading data and artifacts..."):
    df = load_data()
    prepared_df = prepare_model_data(df)
    prepared_df = prepared_df.dropna(subset=ALL_FEATURES + ["demand"]).copy()
    model = load_model()

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

# Sidebar
with st.sidebar:
    st.header("Scenario controls")
    zone = st.selectbox("Pickup zone", zones)
    hour = st.slider("Hour of day", 0, 23, 11)
    elasticity = st.slider("Elasticity assumption", 0.05, 1.00, 0.35, 0.05)
    st.caption("Higher elasticity means demand falls faster when price rises.")

    st.divider()
    st.subheader("Quick links")
    st.write("Live pricing demo, model comparison, cross-validation, feature importance, and final outputs are below.")

# Core scenario
context_row = build_context_row(prepared_df, zone, hour)
context_row_df = pd.DataFrame([context_row[ALL_FEATURES].to_dict()])
predicted_demand = predict_no_warnings(model, context_row_df)

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

# Executive summary / methodology
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
        "The project evaluates multiple classical regressors, gradient boosting models, and a neural-network branch, "
        "then surfaces the best-performing model in a real scenario."
    )
    if final_report:
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
        st.info("Model metadata not found.")

# Context explorer
st.divider()
st.subheader("Top demand context")
show_cols = ["pickup_date", "hour", "day_of_week", "pickup_zone", "demand", "avg_fare_amount", "avg_trip_distance", "avg_duration_min", "avg_speed_kph"]
available_cols = [c for c in show_cols if c in prepared_df.columns]
zone_history = prepared_df[prepared_df["pickup_zone"] == zone].sort_values(["hour", "pickup_date"])
st.dataframe(zone_history[available_cols].head(10), use_container_width=True, hide_index=True)

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
    st.info("Model metrics not available.")

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
