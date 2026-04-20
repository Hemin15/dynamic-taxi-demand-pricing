# main.py

import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from model import prepare_model_data, time_based_split, ALL_FEATURES
from pricing_engine import DynamicPricingEngine, optimize_for_row, save_curve, save_result


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "outputs" / "data" / "demand_panel.csv"
MODEL_PATH = BASE_DIR / "models" / "demand_model.pkl"
ARTIFACTS_DIR = BASE_DIR / "outputs" / "artifacts"
FINAL_DIR = BASE_DIR / "outputs" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# JSON helper
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


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, default=_json_default)


def load_optional_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)


def predict_no_warnings(model, row_df: pd.DataFrame) -> float:
    """
    Predict using a NumPy array to avoid LightGBM / sklearn feature-name warnings.
    """
    return float(model.predict(row_df.to_numpy(dtype=float))[0])


def pick_representative_row(test_df: pd.DataFrame) -> pd.Series:
    """
    Pick a row close to the median demand so the demo is representative.
    """
    median_demand = test_df["demand"].median()
    idx = (test_df["demand"] - median_demand).abs().idxmin()
    return test_df.loc[idx]


def plot_revenue_curve(curve_df: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(curve_df["price"], curve_df["revenue"], linewidth=2)
    plt.title("Revenue Curve Across Price Levels")
    plt.xlabel("Price")
    plt.ylabel("Expected Revenue")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sensitivity_curve(sensitivity_df: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(
        sensitivity_df["elasticity"],
        sensitivity_df["optimal_price"],
        marker="o",
        label="Optimal price",
    )
    plt.plot(
        sensitivity_df["elasticity"],
        sensitivity_df["expected_revenue"],
        marker="o",
        label="Expected revenue",
    )
    plt.title("Pricing Sensitivity Across Elasticity Assumptions")
    plt.xlabel("Elasticity")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_pricing_simulation(model, test_df: pd.DataFrame, n_samples: int = 100):
    sample_df = test_df.sample(min(n_samples, len(test_df)), random_state=42).copy()
    records = []

    for _, row in sample_df.iterrows():
        row_df = pd.DataFrame([row[ALL_FEATURES].to_dict()])
        predicted_demand = predict_no_warnings(model, row_df)

        result, _ = optimize_for_row(
            row=row,
            predicted_demand=predicted_demand,
            min_price=5.0,
            max_price=100.0,
            price_step=1.0,
            elasticity=0.35
        )

        records.append({
            "pickup_date": row["pickup_date"].isoformat() if hasattr(row["pickup_date"], "isoformat") else str(row["pickup_date"]),
            "hour": int(row["hour"]),
            "pickup_zone": str(row["pickup_zone"]),
            "actual_demand": float(row["demand"]),
            "predicted_demand": float(predicted_demand),
            "static_price": float(result.static_price),
            "static_revenue": float(result.static_revenue),
            "optimal_price": float(result.optimal_price),
            "optimal_revenue": float(result.expected_revenue),
            "uplift_percent": float(result.uplift_percent),
        })

    sim_df = pd.DataFrame(records)
    sim_df.to_csv(FINAL_DIR / "pricing_simulation_summary.csv", index=False)

    avg_uplift = sim_df["uplift_percent"].mean()
    avg_static_revenue = sim_df["static_revenue"].mean()
    avg_optimal_revenue = sim_df["optimal_revenue"].mean()
    avg_predicted_demand = sim_df["predicted_demand"].mean()

    top_uplift_cases = (
        sim_df.sort_values("uplift_percent", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )

    return (
        sim_df,
        avg_uplift,
        avg_static_revenue,
        avg_optimal_revenue,
        avg_predicted_demand,
        top_uplift_cases,
    )


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def main():
    print("Loading demand panel...")
    demand_df = pd.read_csv(RAW_DATA_PATH)

    print("Preparing model data...")
    prepared_df = prepare_model_data(demand_df)
    prepared_df = prepared_df.dropna(subset=ALL_FEATURES + ["demand"]).copy()

    train_df, val_df, test_df = time_based_split(prepared_df)

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows  : {len(val_df)}")
    print(f"Test rows : {len(test_df)}")

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    # Optional metadata from model training
    model_metrics = load_optional_json(ARTIFACTS_DIR / "model_metrics.json")
    cv_summary_df = load_optional_csv(ARTIFACTS_DIR / "cross_validation_summary.csv")
    comparison_df = load_optional_csv(ARTIFACTS_DIR / "model_selection_summary.csv")

    # --------------------------------------------------------
    # Demo scenario
    # --------------------------------------------------------
    demo_row = pick_representative_row(test_df)
    demo_row_df = pd.DataFrame([demo_row[ALL_FEATURES].to_dict()])

    predicted_demand = predict_no_warnings(model, demo_row_df)

    # Create one pricing engine instance so we can reuse it
    pricing_engine = DynamicPricingEngine(
        min_price=5.0,
        max_price=100.0,
        price_step=1.0,
        elasticity=0.35,
    )

    ref_price = pricing_engine._get_reference_price(demo_row)

    pricing_result = pricing_engine.optimize_price(
        base_demand=predicted_demand,
        reference_price=ref_price,
    )
    revenue_curve = pricing_engine.revenue_curve(
        base_demand=predicted_demand,
        reference_price=ref_price,
    )
    sensitivity_df = pricing_engine.sensitivity_curve(
        base_demand=predicted_demand,
        reference_price=ref_price,
    )

    save_curve(revenue_curve, FINAL_DIR / "revenue_curve.csv")
    save_result(pricing_result, FINAL_DIR / "best_pricing_result.json")
    sensitivity_df.to_csv(FINAL_DIR / "sensitivity_curve.csv", index=False)

    plot_revenue_curve(revenue_curve, FINAL_DIR / "revenue_curve.png")
    plot_sensitivity_curve(sensitivity_df, FINAL_DIR / "sensitivity_curve.png")

    # Save a small scenario table for reporting
    scenario_table = pd.DataFrame([
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
    scenario_table.to_csv(FINAL_DIR / "scenario_comparison.csv", index=False)

    # --------------------------------------------------------
    # Batch simulation
    # --------------------------------------------------------
    (
        sim_df,
        avg_uplift,
        avg_static_revenue,
        avg_optimal_revenue,
        avg_predicted_demand,
        top_uplift_cases,
    ) = run_pricing_simulation(
        model=model,
        test_df=test_df,
        n_samples=100
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    final_report = {
        "project": {
            "name": "dynamic taxi demand forecasting and pricing optimization",
            "model_type": model_metrics.get("best_model", "unknown") if model_metrics else "unknown",
            "notes": "hybrid ai system combining machine learning demand prediction with rule-based pricing optimization",
        },
        "demo_case": {
            "pickup_date": demo_row["pickup_date"].isoformat() if hasattr(demo_row["pickup_date"], "isoformat") else str(demo_row["pickup_date"]),
            "hour": int(demo_row["hour"]),
            "pickup_zone": str(demo_row["pickup_zone"]),
            "actual_demand": float(demo_row["demand"]),
            "predicted_demand": float(predicted_demand),
            "reference_price": float(ref_price),
            "static_price": float(pricing_result.static_price),
            "static_revenue": float(pricing_result.static_revenue),
            "optimal_price": float(pricing_result.optimal_price),
            "optimal_demand": float(pricing_result.expected_demand),
            "optimal_revenue": float(pricing_result.expected_revenue),
            "uplift_percent": float(pricing_result.uplift_percent),
        },
        "pricing_sensitivity": sensitivity_df.to_dict(orient="records"),
        "batch_summary": {
            "sample_size": int(len(sim_df)),
            "average_predicted_demand": float(avg_predicted_demand),
            "average_static_revenue": float(avg_static_revenue),
            "average_optimal_revenue": float(avg_optimal_revenue),
            "average_uplift_percent": float(avg_uplift),
        },
        "top_5_uplift_cases": top_uplift_cases,
        "model_summary": model_metrics if model_metrics else {},
    }

    save_json(final_report, FINAL_DIR / "final_report.json")

    # Optional run summary text file
    run_summary = [
        "RUN SUMMARY",
        "=" * 50,
        f"Best model: {final_report['project']['model_type']}",
        f"Demo uplift: {final_report['demo_case']['uplift_percent']:.2f}%",
        f"Average uplift: {avg_uplift:.2f}%",
        f"Average static revenue: {avg_static_revenue:.2f}",
        f"Average optimal revenue: {avg_optimal_revenue:.2f}",
    ]
    if cv_summary_df is not None and len(cv_summary_df) > 0:
        run_summary.append("")
        run_summary.append("Top CV models:")
        run_summary.append(str(cv_summary_df.head(5)))

    with open(FINAL_DIR / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(run_summary))

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------
    print("\n=== DEMO SCENARIO ===")
    print(f"Pickup date       : {final_report['demo_case']['pickup_date']}")
    print(f"Hour              : {final_report['demo_case']['hour']}")
    print(f"Pickup zone       : {final_report['demo_case']['pickup_zone']}")
    print(f"Actual demand     : {final_report['demo_case']['actual_demand']:.2f}")
    print(f"Predicted demand  : {final_report['demo_case']['predicted_demand']:.2f}")
    print(f"Reference price   : {final_report['demo_case']['reference_price']:.2f}")
    print(f"Static price      : {final_report['demo_case']['static_price']:.2f}")
    print(f"Static revenue    : {final_report['demo_case']['static_revenue']:.2f}")
    print(f"Optimal price     : {final_report['demo_case']['optimal_price']:.2f}")
    print(f"Optimal demand    : {final_report['demo_case']['optimal_demand']:.2f}")
    print(f"Optimal revenue   : {final_report['demo_case']['optimal_revenue']:.2f}")
    print(f"Uplift            : {final_report['demo_case']['uplift_percent']:.2f}%")

    print("\n=== PRICING SENSITIVITY ===")
    print(sensitivity_df.head())

    print("\n=== BATCH SUMMARY (100 TEST SAMPLES) ===")
    print(f"Average predicted demand : {avg_predicted_demand:.2f}")
    print(f"Average static revenue   : {avg_static_revenue:.2f}")
    print(f"Average optimal revenue   : {avg_optimal_revenue:.2f}")
    print(f"Average uplift            : {avg_uplift:.2f}%")

    print("\n=== TOP 5 UPLIFT CASES ===")
    for i, case in enumerate(top_uplift_cases, 1):
        print(
            f"{i}. zone={case['pickup_zone']} hour={case['hour']} "
            f"uplift={case['uplift_percent']:.2f}% "
            f"static={case['static_revenue']:.2f} optimal={case['optimal_revenue']:.2f}"
        )

    print("\nSaved outputs to outputs/final/")
    print("- final_report.json")
    print("- run_summary.txt")
    print("- best_pricing_result.json")
    print("- revenue_curve.csv")
    print("- revenue_curve.png")
    print("- sensitivity_curve.csv")
    print("- sensitivity_curve.png")
    print("- scenario_comparison.csv")
    print("- pricing_simulation_summary.csv")


if __name__ == "__main__":
    main()