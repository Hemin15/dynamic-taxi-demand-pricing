# visualization.py

import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _base_theme():
    sns.set_theme(style="whitegrid")


def _safe_sample(df, n=5000, random_state=42):
    return df.sample(min(n, len(df)), random_state=random_state) if len(df) > 0 else df


def _load_optional_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


# -------------------------------------------------------------------
# EDA plots
# -------------------------------------------------------------------
def plot_demand_eda(demand_df):
    _base_theme()

    # ------------------------------------------------------------------
    # 1) Demand distribution
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(demand_df["demand"], bins=50, kde=True)
    plt.title("Distribution of Demand")
    plt.xlabel("Demand")
    plt.ylabel("Frequency")
    _save_fig(os.path.join(PLOTS_DIR, "demand_distribution.png"))

    # ------------------------------------------------------------------
    # 2) Average demand by hour
    # ------------------------------------------------------------------
    hourly_demand = demand_df.groupby("hour", as_index=False)["demand"].mean().sort_values("hour")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hourly_demand, x="hour", y="demand", marker="o")
    plt.title("Average Demand by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Demand")
    plt.xticks(range(0, 24))
    _save_fig(os.path.join(PLOTS_DIR, "demand_by_hour.png"))

    # ------------------------------------------------------------------
    # 3) Average demand by weekday
    # ------------------------------------------------------------------
    weekday_order = [0, 1, 2, 3, 4, 5, 6]
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_demand = demand_df.groupby("day_of_week", as_index=False)["demand"].mean().sort_values("day_of_week")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=weekday_demand, x="day_of_week", y="demand", order=weekday_order)
    plt.title("Average Demand by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Demand")
    plt.xticks(ticks=range(7), labels=weekday_labels)
    _save_fig(os.path.join(PLOTS_DIR, "demand_by_weekday.png"))

    # ------------------------------------------------------------------
    # 4) Weekday vs weekend
    # ------------------------------------------------------------------
    weekend_compare = demand_df.groupby("is_weekend", as_index=False)["demand"].mean().sort_values("is_weekend")
    plt.figure(figsize=(7, 6))
    sns.barplot(data=weekend_compare, x="is_weekend", y="demand")
    plt.title("Average Demand: Weekday vs Weekend")
    plt.xlabel("Is Weekend (0 = No, 1 = Yes)")
    plt.ylabel("Average Demand")
    _save_fig(os.path.join(PLOTS_DIR, "weekday_vs_weekend.png"))

    # ------------------------------------------------------------------
    # 5) Peak hour vs non-peak hour
    # ------------------------------------------------------------------
    peak_compare = demand_df.groupby("is_peak_hour", as_index=False)["demand"].mean().sort_values("is_peak_hour")
    plt.figure(figsize=(7, 6))
    sns.barplot(data=peak_compare, x="is_peak_hour", y="demand")
    plt.title("Average Demand: Peak Hour vs Non-Peak Hour")
    plt.xlabel("Is Peak Hour (0 = No, 1 = Yes)")
    plt.ylabel("Average Demand")
    _save_fig(os.path.join(PLOTS_DIR, "peak_vs_nonpeak.png"))

    # ------------------------------------------------------------------
    # 6) Monthly trend
    # ------------------------------------------------------------------
    monthly_demand = demand_df.groupby("month", as_index=False)["demand"].mean().sort_values("month")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_demand, x="month", y="demand", marker="o")
    plt.title("Average Demand by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Demand")
    plt.xticks(range(1, 13))
    _save_fig(os.path.join(PLOTS_DIR, "demand_by_month.png"))

    # ------------------------------------------------------------------
    # 7) Month start / month end / travel season
    # ------------------------------------------------------------------
    if "is_month_start" in demand_df.columns:
        month_start = demand_df.groupby("is_month_start", as_index=False)["demand"].mean().sort_values("is_month_start")
        plt.figure(figsize=(7, 6))
        sns.barplot(data=month_start, x="is_month_start", y="demand")
        plt.title("Average Demand: Month Start vs Others")
        plt.xlabel("Is Month Start")
        plt.ylabel("Average Demand")
        _save_fig(os.path.join(PLOTS_DIR, "month_start_vs_other.png"))

    if "is_month_end" in demand_df.columns:
        month_end = demand_df.groupby("is_month_end", as_index=False)["demand"].mean().sort_values("is_month_end")
        plt.figure(figsize=(7, 6))
        sns.barplot(data=month_end, x="is_month_end", y="demand")
        plt.title("Average Demand: Month End vs Others")
        plt.xlabel("Is Month End")
        plt.ylabel("Average Demand")
        _save_fig(os.path.join(PLOTS_DIR, "month_end_vs_other.png"))

    if "is_high_travel_period" in demand_df.columns:
        travel_period = demand_df.groupby("is_high_travel_period", as_index=False)["demand"].mean().sort_values("is_high_travel_period")
        plt.figure(figsize=(7, 6))
        sns.barplot(data=travel_period, x="is_high_travel_period", y="demand")
        plt.title("Average Demand: High Travel Period vs Others")
        plt.xlabel("Is High Travel Period")
        plt.ylabel("Average Demand")
        _save_fig(os.path.join(PLOTS_DIR, "high_travel_period.png"))

    # ------------------------------------------------------------------
    # 8) Top zones
    # ------------------------------------------------------------------
    top_zones = (
        demand_df.groupby("pickup_zone", as_index=False)["demand"]
        .sum()
        .sort_values("demand", ascending=False)
        .head(15)
    )
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_zones, y="pickup_zone", x="demand")
    plt.title("Top 15 Pickup Zones by Total Demand")
    plt.xlabel("Total Demand")
    plt.ylabel("Pickup Zone")
    _save_fig(os.path.join(PLOTS_DIR, "top_zones.png"))

    # ------------------------------------------------------------------
    # 9) Correlation heatmap
    # ------------------------------------------------------------------
    numeric_cols = demand_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    corr = demand_df[numeric_cols].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Features")
    _save_fig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))

    # ------------------------------------------------------------------
    # 10) Demand vs trip distance
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=_safe_sample(demand_df, 5000),
        x="avg_trip_distance",
        y="demand",
        alpha=0.4
    )
    plt.title("Demand vs Average Trip Distance")
    plt.xlabel("Average Trip Distance")
    plt.ylabel("Demand")
    _save_fig(os.path.join(PLOTS_DIR, "demand_vs_trip_distance.png"))

    # ------------------------------------------------------------------
    # 11) Demand vs fare amount
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=_safe_sample(demand_df, 5000),
        x="avg_fare_amount",
        y="demand",
        alpha=0.4
    )
    plt.title("Demand vs Average Fare Amount")
    plt.xlabel("Average Fare Amount")
    plt.ylabel("Demand")
    _save_fig(os.path.join(PLOTS_DIR, "demand_vs_fare.png"))

    # ------------------------------------------------------------------
    # 12) Demand vs duration and speed
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=_safe_sample(demand_df, 5000),
        x="avg_duration_min",
        y="demand",
        alpha=0.4
    )
    plt.title("Demand vs Average Trip Duration")
    plt.xlabel("Average Trip Duration (min)")
    plt.ylabel("Demand")
    _save_fig(os.path.join(PLOTS_DIR, "demand_vs_duration.png"))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=_safe_sample(demand_df, 5000),
        x="avg_speed_kph",
        y="demand",
        alpha=0.4
    )
    plt.title("Demand vs Average Speed")
    plt.xlabel("Average Speed (kph)")
    plt.ylabel("Demand")
    _save_fig(os.path.join(PLOTS_DIR, "demand_vs_speed.png"))

    # ------------------------------------------------------------------
    # 13) Hour x weekday heatmap
    # ------------------------------------------------------------------
    hour_weekday = demand_df.pivot_table(values="demand", index="day_of_week", columns="hour", aggfunc="mean")
    plt.figure(figsize=(14, 6))
    sns.heatmap(hour_weekday, cmap="viridis")
    plt.title("Average Demand Heatmap by Hour and Day of Week")
    plt.xlabel("Hour")
    plt.ylabel("Day of Week")
    _save_fig(os.path.join(PLOTS_DIR, "hour_weekday_heatmap.png"))

    # ------------------------------------------------------------------
    # 14) EDA summary
    # ------------------------------------------------------------------
    summary_text = [
        "EDA SUMMARY",
        "=" * 50,
        f"Rows: {len(demand_df)}",
        f"Columns: {len(demand_df.columns)}",
        "",
        "Demand Statistics:",
        str(demand_df["demand"].describe()),
        "",
        "Top 5 Zones by Total Demand:",
        str(top_zones.head(5)),
    ]

    with open(os.path.join(REPORTS_DIR, "eda_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_text))

    print("✅ All EDA plots saved to outputs/plots/")
    print("✅ EDA summary saved to outputs/reports/eda_summary.txt")


# -------------------------------------------------------------------
# Model diagnostics
# -------------------------------------------------------------------
def plot_model_diagnostics(
    comparison_df: pd.DataFrame,
    train_pred_df: pd.DataFrame,
    val_pred_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame = None
):
    _base_theme()

    # ------------------------------------------------------------------
    # 1) Model comparison by MAE
    # ------------------------------------------------------------------
    comp = comparison_df.copy()
    comp_melt = comp.melt(
        id_vars=["model_name"],
        value_vars=["train_mae", "val_mae", "test_mae"],
        var_name="split",
        value_name="mae"
    )
    comp_melt["split"] = comp_melt["split"].str.replace("_mae", "", regex=False).str.title()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=comp_melt, x="model_name", y="mae", hue="split")
    plt.title("Model Comparison by MAE")
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.xticks(rotation=15)
    _save_fig(os.path.join(PLOTS_DIR, "model_comparison_mae.png"))

    # ------------------------------------------------------------------
    # 2) Model comparison by RMSE
    # ------------------------------------------------------------------
    comp_melt_rmse = comp.melt(
        id_vars=["model_name"],
        value_vars=["train_rmse", "val_rmse", "test_rmse"],
        var_name="split",
        value_name="rmse"
    )
    comp_melt_rmse["split"] = comp_melt_rmse["split"].str.replace("_rmse", "", regex=False).str.title()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=comp_melt_rmse, x="model_name", y="rmse", hue="split")
    plt.title("Model Comparison by RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=15)
    _save_fig(os.path.join(PLOTS_DIR, "model_comparison_rmse.png"))

    # ------------------------------------------------------------------
    # 3) Cross-validation summary if available
    # ------------------------------------------------------------------
    cv_summary_path = os.path.join(ARTIFACTS_DIR, "cross_validation_summary.csv")
    cv_summary_df = _load_optional_csv(cv_summary_path)
    if cv_summary_df is not None and len(cv_summary_df) > 0:
        plt.figure(figsize=(11, 6))
        sns.barplot(data=cv_summary_df, x="model_name", y="cv_val_mae_mean")
        plt.title("Cross-Validation Mean Validation MAE")
        plt.xlabel("Model")
        plt.ylabel("CV Validation MAE")
        plt.xticks(rotation=15)
        _save_fig(os.path.join(PLOTS_DIR, "cross_validation_mae.png"))

    # ------------------------------------------------------------------
    # 4) Train vs validation gap
    # ------------------------------------------------------------------
    gap_df = comparison_df.copy()
    gap_df["mae_gap"] = gap_df["val_mae"] - gap_df["train_mae"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=gap_df, x="model_name", y="mae_gap")
    plt.title("Validation vs Training MAE Gap")
    plt.xlabel("Model")
    plt.ylabel("MAE Gap")
    plt.xticks(rotation=15)
    _save_fig(os.path.join(PLOTS_DIR, "train_val_gap.png"))

    # ------------------------------------------------------------------
    # 5) Actual vs predicted on test
    # ------------------------------------------------------------------
    sample_test = test_pred_df.sample(min(8000, len(test_pred_df)), random_state=42)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=sample_test, x="actual_demand", y="predicted_demand", alpha=0.35)
    max_val = max(sample_test["actual_demand"].max(), sample_test["predicted_demand"].max())
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.title("Actual vs Predicted Demand (Test Set)")
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    _save_fig(os.path.join(PLOTS_DIR, "actual_vs_predicted_test.png"))

    # ------------------------------------------------------------------
    # 6) Residual distribution
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(test_pred_df["error"], bins=50, kde=True)
    plt.title("Residual Distribution (Test Set)")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    _save_fig(os.path.join(PLOTS_DIR, "residual_distribution.png"))

    # ------------------------------------------------------------------
    # 7) Residuals vs predicted
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sample_test, x="predicted_demand", y="error", alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.title("Residuals vs Predicted Demand")
    plt.xlabel("Predicted Demand")
    plt.ylabel("Residual")
    _save_fig(os.path.join(PLOTS_DIR, "residual_vs_predicted.png"))

    # ------------------------------------------------------------------
    # 8) Error by hour
    # ------------------------------------------------------------------
    hour_error = (
        test_pred_df.groupby("hour", as_index=False)["abs_error"]
        .mean()
        .sort_values("hour")
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hour_error, x="hour", y="abs_error", marker="o")
    plt.title("Mean Absolute Error by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(range(0, 24))
    _save_fig(os.path.join(PLOTS_DIR, "error_by_hour.png"))

    # ------------------------------------------------------------------
    # 9) Error by zone
    # ------------------------------------------------------------------
    zone_error = (
        test_pred_df.groupby("pickup_zone", as_index=False)["abs_error"]
        .mean()
        .sort_values("abs_error", ascending=False)
        .head(15)
    )
    plt.figure(figsize=(12, 7))
    sns.barplot(data=zone_error, y="pickup_zone", x="abs_error")
    plt.title("Top 15 Zones by Mean Absolute Error")
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Pickup Zone")
    _save_fig(os.path.join(PLOTS_DIR, "error_by_zone.png"))

    # ------------------------------------------------------------------
    # 10) Peak vs non-peak error
    # ------------------------------------------------------------------
    peak_error = (
        test_pred_df.groupby("is_peak_hour", as_index=False)["abs_error"]
        .mean()
        .sort_values("is_peak_hour")
    )
    plt.figure(figsize=(7, 6))
    sns.barplot(data=peak_error, x="is_peak_hour", y="abs_error")
    plt.title("Mean Absolute Error: Peak vs Non-Peak Hours")
    plt.xlabel("Is Peak Hour (0 = No, 1 = Yes)")
    plt.ylabel("Mean Absolute Error")
    _save_fig(os.path.join(PLOTS_DIR, "error_peak_vs_nonpeak.png"))

    # ------------------------------------------------------------------
    # 11) Calibration plot
    # ------------------------------------------------------------------
    temp = test_pred_df[["actual_demand", "predicted_demand"]].copy()
    temp["bin"] = pd.qcut(temp["predicted_demand"], q=10, duplicates="drop")
    calib = temp.groupby("bin", as_index=False, observed=False).agg(
        mean_predicted=("predicted_demand", "mean"),
        mean_actual=("actual_demand", "mean")
    )

    plt.figure(figsize=(9, 7))
    sns.lineplot(data=calib, x="mean_predicted", y="mean_actual", marker="o")
    max_val = max(calib["mean_predicted"].max(), calib["mean_actual"].max())
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.title("Calibration Plot")
    plt.xlabel("Mean Predicted Demand")
    plt.ylabel("Mean Actual Demand")
    _save_fig(os.path.join(PLOTS_DIR, "calibration_plot.png"))

    # ------------------------------------------------------------------
    # 12) Feature importance
    # ------------------------------------------------------------------
    if feature_importance_df is not None and len(feature_importance_df) > 0:
        top_fi = feature_importance_df.head(15).sort_values("importance")
        plt.figure(figsize=(10, 7))
        sns.barplot(data=top_fi, x="importance", y="feature")
        plt.title("Top Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        _save_fig(os.path.join(PLOTS_DIR, "feature_importance.png"))

    # ------------------------------------------------------------------
    # 13) Neural-network training history if available
    # ------------------------------------------------------------------
    nn_history_path = os.path.join(ARTIFACTS_DIR, "nn_training_history.csv")
    nn_history_df = _load_optional_csv(nn_history_path)
    if nn_history_df is not None and len(nn_history_df) > 0:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=nn_history_df, x="epoch", y="val_rmse", marker="o")
        plt.title("Neural Network Validation RMSE by Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Validation RMSE")
        _save_fig(os.path.join(PLOTS_DIR, "nn_val_rmse.png"))

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=nn_history_df, x="epoch", y="train_loss", marker="o")
        plt.title("Neural Network Training Loss by Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        _save_fig(os.path.join(PLOTS_DIR, "nn_train_loss.png"))

    print("✅ All model diagnostic plots saved to outputs/plots/")


if __name__ == "__main__":
    demand_df = pd.read_csv(os.path.join(OUTPUT_DIR, "data", "demand_panel.csv"))
    plot_demand_eda(demand_df)