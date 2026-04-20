# preprocessing.py

import os
import json
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
RAW_USECOLS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "store_and_fwd_flag",
]

NUMERIC_COLS = [
    "VendorID",
    "passenger_count",
    "trip_distance",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
]

# Approximate NYC bounds to remove obvious invalid coordinates
NYC_LAT_MIN, NYC_LAT_MAX = 40.40, 41.20
NYC_LON_MIN, NYC_LON_MAX = -75.20, -72.70


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def _memory_optimize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns where possible to reduce memory usage.
    This is useful because the raw dataset is very large.
    """
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar and cyclical time features.
    These are important for learning daily and weekly demand patterns.
    """
    pickup_dt = df["tpep_pickup_datetime"]

    df["pickup_date"] = pickup_dt.dt.floor("D")
    df["hour"] = pickup_dt.dt.hour
    df["day"] = pickup_dt.dt.day
    df["month"] = pickup_dt.dt.month
    df["day_of_week"] = pickup_dt.dt.dayofweek  # Monday=0, Sunday=6
    df["week_of_year"] = pickup_dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # Cyclical encoding for periodic patterns
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def _add_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trip-level behavioral features.
    These help validate data quality and capture operational behavior.
    """
    trip_duration_min = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    )
    df["trip_duration_min"] = trip_duration_min

    # Speed is useful for cleaning abnormal trips
    df["speed_kph"] = (
        df["trip_distance"] / (df["trip_duration_min"] / 60)
    ).replace([np.inf, -np.inf], np.nan)

    # Fare efficiency
    df["fare_per_mile"] = (
        df["fare_amount"] / df["trip_distance"]
    ).replace([np.inf, -np.inf], np.nan)

    df["total_per_mile"] = (
        df["total_amount"] / df["trip_distance"]
    ).replace([np.inf, -np.inf], np.nan)

    return df


def _add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple spatial bucket from pickup coordinates.
    This converts raw taxi records into a demand-by-region style dataset.
    """
    df["pickup_lat_bin"] = df["pickup_latitude"].round(2)
    df["pickup_lon_bin"] = df["pickup_longitude"].round(2)
    df["pickup_zone"] = df["pickup_lat_bin"].astype(str) + "_" + df["pickup_lon_bin"].astype(str)

    df["dropoff_lat_bin"] = df["dropoff_latitude"].round(2)
    df["dropoff_lon_bin"] = df["dropoff_longitude"].round(2)
    df["dropoff_zone"] = df["dropoff_lat_bin"].astype(str) + "_" + df["dropoff_lon_bin"].astype(str)

    return df


def _add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional but useful calendar-based flags for richer demand analysis.
    These add realism and can be useful later if you expand the model.
    """
    df["is_month_start"] = (df["day"] <= 3).astype(int)
    df["is_month_end"] = (df["day"] >= 28).astype(int)

    # Simple demand-sensitive holiday-like proxy flags
    # These are not exact holidays, but they can help explore seasonality.
    df["is_high_travel_period"] = df["month"].isin([11, 12]).astype(int)

    return df


def _clean_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean invalid values and keep only realistic trips.
    Returns the cleaned dataframe and a summary dictionary.
    """
    summary = {}
    summary["raw_rows"] = int(len(df))

    # Standardize missing values
    df["store_and_fwd_flag"] = (
        df["store_and_fwd_flag"]
        .fillna("N")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Drop rows with missing critical values
    critical_cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
        "total_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    df = df.dropna(subset=critical_cols).copy()
    summary["after_dropna"] = int(len(df))

    # Ensure numeric columns are numeric
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC_COLS).copy()
    summary["after_numeric_coercion"] = int(len(df))

    # Remove impossible timestamps
    df = df[df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"]].copy()

    # Add trip features before filtering
    df = _add_trip_features(df)

    # Sanity checks
    df = df[df["trip_duration_min"].between(1, 240)].copy()
    df = df[df["trip_distance"].between(0.1, 100)].copy()
    df = df[df["fare_amount"].between(0, 1000)].copy()
    df = df[df["total_amount"].between(0, 1500)].copy()
    df = df[df["passenger_count"].between(1, 8)].copy()

    # Coordinate sanity checks
    pickup_ok = (
        df["pickup_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df["pickup_longitude"].between(NYC_LON_MIN, NYC_LON_MAX)
    )
    dropoff_ok = (
        df["dropoff_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df["dropoff_longitude"].between(NYC_LON_MIN, NYC_LON_MAX)
    )
    df = df[pickup_ok & dropoff_ok].copy()

    # Remove extreme speeds caused by bad records
    df = df[df["speed_kph"].between(1, 140)].copy()

    summary["clean_rows"] = int(len(df))
    summary["removed_rows"] = int(summary["raw_rows"] - summary["clean_rows"])

    return df, summary


def aggregate_demand_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trip records into a spatiotemporal demand panel.
    Each row represents demand in a pickup region during a date-hour window.
    """
    group_cols = [
        "pickup_date",
        "hour",
        "day_of_week",
        "month",
        "week_of_year",
        "is_weekend",
        "is_peak_hour",
        "is_month_start",
        "is_month_end",
        "is_high_travel_period",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "pickup_zone",
        "pickup_lat_bin",
        "pickup_lon_bin",
    ]

    agg_df = (
        df.groupby(group_cols, as_index=False)
        .agg(
            demand=("VendorID", "size"),
            avg_trip_distance=("trip_distance", "mean"),
            avg_duration_min=("trip_duration_min", "mean"),
            avg_passenger_count=("passenger_count", "mean"),
            avg_fare_amount=("fare_amount", "mean"),
            avg_total_amount=("total_amount", "mean"),
            avg_tip_amount=("tip_amount", "mean"),
            avg_tolls_amount=("tolls_amount", "mean"),
            avg_speed_kph=("speed_kph", "mean"),
            vendor_diversity=("VendorID", "nunique"),
            payment_mode=("payment_type", "mean"),
        )
        .sort_values(["pickup_date", "hour", "pickup_zone"])
        .reset_index(drop=True)
    )

    # Demand transforms for modeling
    agg_df["log_demand"] = np.log1p(agg_df["demand"])
    agg_df["demand_norm"] = agg_df["demand"] / agg_df["demand"].max()

    # Zone-level density indicator
    zone_counts = agg_df["pickup_zone"].value_counts()
    agg_df["zone_trip_volume"] = agg_df["pickup_zone"].map(zone_counts)

    return agg_df


def load_and_preprocess(
    path: str,
    save_output: bool = True,
    output_dir: str = "outputs/data"
):
    """
    Main preprocessing pipeline:
    1. Load raw taxi data
    2. Clean invalid rows
    3. Add time, trip, spatial, and calendar features
    4. Aggregate into a demand panel
    5. Save outputs
    """
    df = pd.read_csv(
        path,
        usecols=RAW_USECOLS,
        low_memory=False
    )

    # Parse datetime columns
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

    # Clean raw data
    df, summary = _clean_raw_data(df)

    # Feature engineering
    df = _add_time_features(df)
    df = _add_spatial_features(df)
    df = _add_calendar_flags(df)

    # Keep a clean trip-level dataset for exploration and debugging
    clean_trip_df = _memory_optimize(df.copy())

    # Create demand panel for modeling
    demand_df = aggregate_demand_panel(df)
    demand_df = _memory_optimize(demand_df)

    # Add summary stats
    summary["clean_trip_columns"] = list(clean_trip_df.columns)
    summary["demand_panel_rows"] = int(len(demand_df))
    summary["demand_panel_columns"] = list(demand_df.columns)

    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        clean_trip_df.to_csv(os.path.join(output_dir, "clean_trips.csv"), index=False)
        demand_df.to_csv(os.path.join(output_dir, "demand_panel.csv"), index=False)

        with open(os.path.join(output_dir, "preprocess_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

    return clean_trip_df, demand_df, summary


if __name__ == "__main__":
    raw_path = "data/yellow_tripdata_2015-01.csv"

    clean_trip_df, demand_df, summary = load_and_preprocess(raw_path, save_output=True)

    print("\n=== PREPROCESSING SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, list):
            print(f"{k}: {len(v)} columns")
        else:
            print(f"{k}: {v}")

    print("\n=== CLEAN TRIP DATA ===")
    print(clean_trip_df.head())
    print(clean_trip_df.shape)

    print("\n=== DEMAND PANEL ===")
    print(demand_df.head())
    print(demand_df.shape)

    print("\nSaved files to: outputs/data/")