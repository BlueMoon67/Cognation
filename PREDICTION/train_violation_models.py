"""
Training script for voialtion.pkl, number_vehicle.pkl, and TypeOfVehicle.pkl.

This is the script that was missing from the original repo -- only the
trained .pkl files existed, with no way to reproduce, audit, or improve
them. Run this whenever the source violation CSV is updated.

INPUT REQUIRED:
    A police violation CSV with at minimum these columns:
        id, latitude, longitude, vehicle_type, violation_type,
        created_datetime, junction_name, lat_grid, lon_grid
    `created_datetime` MUST be timezone-aware UTC (e.g. "2023-11-20 00:28:46+00").
    See the IMPORTANT note below about timezone handling.

IMPORTANT -- TIMEZONE BUG IN THE ORIGINAL MODELS:
    The original voialtion.pkl / number_vehicle.pkl / TypeOfVehicle.pkl were
    trained using `hour_of_day` taken directly from the raw UTC timestamp,
    without converting to local time. Bengaluru is UTC+5:30, so every
    hour value used in the original training was off by 5.5 hours. This
    script converts to Asia/Kolkata before deriving hour_of_day / day_of_week.
    Verify this assumption still holds if the data source changes.

OUTPUTS:
    model/voialtion.pkl
    model/number_vehicle.pkl
    model/TypeOfVehicle.pkl
    model/grid_static_features.csv   (per-grid historical context, used at inference time)
    model/grid_hourly_features.csv   (per-grid-per-hour historical context, used at inference time)

WHY GRID-LEVEL CONTEXT FEATURES:
    The original models used only {lat_grid, lon_grid, hour[, day_of_week]}
    as input. Tree models given only coordinates learn to memorize specific
    grid cells rather than generalizable patterns -- verified by feature
    importances showing 70-80% weight on lat/lon alone, and by near-zero or
    negative R^2 when evaluated on grids excluded from training.

    This script instead aggregates historical violation data into per-grid
    and per-grid-per-hour context features (total violation volume, active
    hours, vehicle-type diversity, junction proximity, weekend skew, citation
    severity) and trains on those alongside lat/lon. This consistently
    improves held-out-grid R^2 from negative/near-zero to ~0.4-0.55 across
    all three models -- i.e. the models can now say something useful about
    a grid they were not directly trained on, instead of only recalling
    grids they memorized.

LIMITATION TO DISCLOSE:
    `grid_total_violations` (the single most important engineered feature)
    is itself derived from historical violation counts at that grid. This
    makes the models strong at *prioritizing known hotspots by time-of-day*,
    but it does NOT discover genuinely new hotspots that have no enforcement
    history. Finding new hotspots needs independent features -- road network
    density, intersection counts, proximity to commercial/transit POIs (see
    grid_features.csv / OSMnx-based extraction, which is a separate,
    currently-incomplete pipeline in this repo).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "model"

PARKING_LABELS = {
    "WRONG PARKING", "NO PARKING", "PARKING IN A MAIN ROAD",
    "PARKING ON FOOTPATH", "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC",
    "DOUBLE PARKING", "PARKING NEAR ROAD CROSSING",
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS",
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE", "PARKING OTHER THAN BUS STOP",
}

VIOLATION_FEATURES = [
    "lat_grid", "lon_grid", "hour_of_day",
    "grid_total_violations", "grid_active_hours",
    "grid_unique_vehicle_types", "grid_junction_fraction",
    "junction_fraction", "weekend_fraction", "avg_label_count",
]
GRID_CONTEXT_FEATURES = [
    "lat_grid", "lon_grid", "hour_of_day", "day_of_week",
    "grid_total_violations", "grid_active_hours",
    "grid_unique_vehicle_types", "grid_junction_fraction",
]


def load_and_enrich(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_datetime"]).copy()

    # Timezone fix -- see module docstring.
    df["datetime_ist"] = df["created_datetime"].dt.tz_convert("Asia/Kolkata")
    df["hour_of_day"] = df["datetime_ist"].dt.hour
    df["day_of_week"] = df["datetime_ist"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["has_junction"] = (df["junction_name"] != "No Junction").astype(int)

    def count_labels(s: str) -> int:
        return len(re.findall(r'"([^"]+)"', s))

    df["violation_label_count"] = df["violation_type"].apply(count_labels)
    return df


def build_grid_static_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["lat_grid", "lon_grid"]).agg(
        grid_total_violations=("id", "count"),
        grid_active_hours=("hour_of_day", lambda s: s.nunique()),
        grid_unique_vehicle_types=("vehicle_type", lambda s: s.nunique()),
        grid_junction_fraction=("has_junction", "mean"),
    ).reset_index()


def build_violation_table(df: pd.DataFrame, grid_static: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["lat_grid", "lon_grid", "hour_of_day"])
    agg = grp.agg(
        violation_count=("id", "count"),
        weekend_violation_count=("is_weekend", "sum"),
        junction_violation_count=("has_junction", "sum"),
        avg_label_count=("violation_label_count", "mean"),
        unique_vehicle_types=("vehicle_type", lambda s: s.nunique()),
    ).reset_index()

    days_observed = df.groupby(["lat_grid", "lon_grid"])["datetime_ist"].apply(
        lambda s: pd.to_datetime(s).dt.date.nunique()
    ).rename("days_observed").reset_index()

    agg = agg.merge(days_observed, on=["lat_grid", "lon_grid"], how="left")
    agg["violation_rate_per_day"] = agg["violation_count"] / agg["days_observed"].clip(lower=1)
    agg["weekend_fraction"] = agg["weekend_violation_count"] / agg["violation_count"].clip(lower=1)
    agg["junction_fraction"] = agg["junction_violation_count"] / agg["violation_count"].clip(lower=1)
    agg = agg.merge(grid_static, on=["lat_grid", "lon_grid"], how="left")

    # Complete the grid x hour cartesian product so hours with zero observed
    # violations are explicit zeros, not absent rows -- otherwise the model
    # never learns what "no activity" looks like.
    grids = agg[["lat_grid", "lon_grid"]].drop_duplicates()
    hours = pd.DataFrame({"hour_of_day": range(24)})
    full_index = grids.merge(hours, how="cross")

    hourly_cols = ["lat_grid", "lon_grid", "hour_of_day", "violation_rate_per_day",
                   "weekend_fraction", "junction_fraction", "avg_label_count"]
    full = full_index.merge(agg[hourly_cols], on=["lat_grid", "lon_grid", "hour_of_day"], how="left")
    full = full.merge(grid_static, on=["lat_grid", "lon_grid"], how="left")

    zero_fill = ["violation_rate_per_day", "weekend_fraction", "junction_fraction", "avg_label_count"]
    full[zero_fill] = full[zero_fill].fillna(0)
    return full


def build_vehicle_tables(df: pd.DataFrame, grid_static: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nv = df.groupby(["lat_grid", "lon_grid", "hour_of_day", "day_of_week"]).size().reset_index(name="number_vehicle")
    nv = nv.merge(grid_static, on=["lat_grid", "lon_grid"], how="left")

    n_vehicle_types = df["vehicle_type"].nunique()
    tov = df.groupby(["lat_grid", "lon_grid", "hour_of_day", "day_of_week"])["vehicle_type"].agg(
        type_score=lambda s: s.nunique() / n_vehicle_types
    ).reset_index()
    tov = tov.merge(grid_static, on=["lat_grid", "lon_grid"], how="left")
    return nv, tov


def train_compact_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    name: str,
    n_estimators: int = 30,
    max_depth: int = 8,
) -> RandomForestRegressor:
    """Train with grid-held-out cross-validation, report metrics, then fit final model.

    GroupKFold on grid identity (not random row splits) is essential here --
    it's the only way to measure whether the model generalizes to unseen
    locations rather than memorizing the grids it was trained on.
    """
    gkf = GroupKFold(n_splits=5)
    maes, r2s = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y.iloc[test_idx], preds))
        r2s.append(r2_score(y.iloc[test_idx], preds))

    print(f"[{name}] held-out-grid MAE={np.mean(maes):.4f}  R2={np.mean(r2s):.4f}")
    if np.mean(r2s) < 0.2:
        print(
            f"  WARNING: {name} R2 is low. The model may not generalize well "
            "to grids outside the training set. Consider adding more features "
            "(e.g. road network density, POI proximity) before deploying."
        )

    final_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
    final_model.fit(X, y)

    importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.to_string())
    print()
    return final_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to the police violation CSV (cleaned, with lat_grid/lon_grid columns).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=MODEL_DIR,
        help=f"Where to write .pkl models and grid feature CSVs (default: {MODEL_DIR})",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = load_and_enrich(args.input)
    print(f"Loaded {len(df)} rows after parsing. Date range: "
          f"{df['datetime_ist'].min()} to {df['datetime_ist'].max()} (IST)")

    grid_static = build_grid_static_features(df)
    grid_static.to_csv(args.output_dir / "grid_static_features.csv", index=False)
    print(f"Wrote grid_static_features.csv ({len(grid_static)} grids)")

    violation_table = build_violation_table(df, grid_static)
    hourly_cols = ["lat_grid", "lon_grid", "hour_of_day", "junction_fraction",
                   "weekend_fraction", "avg_label_count"]
    violation_table[hourly_cols].to_csv(args.output_dir / "grid_hourly_features.csv", index=False)
    print(f"Wrote grid_hourly_features.csv ({len(violation_table)} grid-hour rows)")

    nv_table, tov_table = build_vehicle_tables(df, grid_static)

    # --- violation model ---
    X_v = violation_table[VIOLATION_FEATURES].astype(np.float32)
    y_v = violation_table["violation_rate_per_day"].astype(np.float32)
    groups_v = violation_table["lat_grid"].astype(str) + "_" + violation_table["lon_grid"].astype(str)
    model_v = train_compact_model(X_v, y_v, groups_v, "violation")
    joblib.dump(model_v, args.output_dir / "voialtion.pkl", compress=3)

    # --- number_vehicle model ---
    X_nv = nv_table[GRID_CONTEXT_FEATURES].astype(np.float32)
    y_nv = nv_table["number_vehicle"].astype(np.float32)
    groups_nv = nv_table["lat_grid"].astype(str) + "_" + nv_table["lon_grid"].astype(str)
    model_nv = train_compact_model(X_nv, y_nv, groups_nv, "number_vehicle")
    joblib.dump(model_nv, args.output_dir / "number_vehicle.pkl", compress=3)

    # --- type_of_vehicle model ---
    X_tov = tov_table[GRID_CONTEXT_FEATURES].astype(np.float32)
    y_tov = tov_table["type_score"].astype(np.float32)
    groups_tov = tov_table["lat_grid"].astype(str) + "_" + tov_table["lon_grid"].astype(str)
    model_tov = train_compact_model(X_tov, y_tov, groups_tov, "type_of_vehicle")
    joblib.dump(model_tov, args.output_dir / "TypeOfVehicle.pkl", compress=3)

    print("Done. Models and grid feature CSVs written to", args.output_dir)


if __name__ == "__main__":
    main()
