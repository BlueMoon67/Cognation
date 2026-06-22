import pickle
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def locate_model_dir() -> Path:
    current = Path(__file__).resolve().parent
    candidates = [
        current / "model",
        current.parent / "model",
        current.parent.parent / "model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate model directory. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


MODEL_DIR = locate_model_dir()

# These models now take grid-level context features (computed once from the
# Jan-May police violation dataset) rather than just lat/lon/hour. Static
# features describe the grid itself; hourly features vary by hour-of-day.
GRID_STATIC_FEATURES_FILE = MODEL_DIR / "grid_static_features.csv"
GRID_HOURLY_FEATURES_FILE = MODEL_DIR / "grid_hourly_features.csv"

_GRID_STATIC_DF: pd.DataFrame | None = None
_GRID_HOURLY_DF: pd.DataFrame | None = None

# Defaults used when a grid has no historical violation data on record
# (e.g. a newly added grid). These represent "no observed activity" rather
# than "average activity" so the model doesn't overstate risk for unknowns.
STATIC_DEFAULTS = {
    "grid_total_violations": 0,
    "grid_active_hours": 0,
    "grid_unique_vehicle_types": 0,
    "grid_junction_fraction": 0.0,
}
HOURLY_DEFAULTS = {
    "junction_fraction": 0.0,
    "weekend_fraction": 0.0,
    "avg_label_count": 0.0,
}


def _load_grid_static() -> pd.DataFrame:
    global _GRID_STATIC_DF
    if _GRID_STATIC_DF is None:
        if not GRID_STATIC_FEATURES_FILE.exists():
            raise FileNotFoundError(
                f"Grid static features file not found: {GRID_STATIC_FEATURES_FILE}. "
                "Run the training pipeline to regenerate it."
            )
        _GRID_STATIC_DF = pd.read_csv(GRID_STATIC_FEATURES_FILE)
    return _GRID_STATIC_DF


def _load_grid_hourly() -> pd.DataFrame:
    global _GRID_HOURLY_DF
    if _GRID_HOURLY_DF is None:
        if not GRID_HOURLY_FEATURES_FILE.exists():
            raise FileNotFoundError(
                f"Grid hourly features file not found: {GRID_HOURLY_FEATURES_FILE}. "
                "Run the training pipeline to regenerate it."
            )
        _GRID_HOURLY_DF = pd.read_csv(GRID_HOURLY_FEATURES_FILE)
    return _GRID_HOURLY_DF


def get_grid_context(lat_grid: float, lon_grid: float, hour_of_day: int) -> dict[str, Any]:
    """Look up historical grid-level + hour-level features for a grid cell.

    Falls back to safe defaults (zero activity) if the grid has no
    historical record, rather than raising — a grid outside the training
    set should not crash the whole prediction cycle for that location.
    """
    static_df = _load_grid_static()
    hourly_df = _load_grid_hourly()

    context = dict(STATIC_DEFAULTS)
    static_match = static_df[
        (static_df["lat_grid"] == lat_grid) & (static_df["lon_grid"] == lon_grid)
    ]
    if not static_match.empty:
        row = static_match.iloc[0]
        context["grid_total_violations"] = row["grid_total_violations"]
        context["grid_active_hours"] = row["grid_active_hours"]
        context["grid_unique_vehicle_types"] = row["grid_unique_vehicle_types"]
        context["grid_junction_fraction"] = row["grid_junction_fraction"]

    context.update(HOURLY_DEFAULTS)
    hourly_match = hourly_df[
        (hourly_df["lat_grid"] == lat_grid)
        & (hourly_df["lon_grid"] == lon_grid)
        & (hourly_df["hour_of_day"] == hour_of_day)
    ]
    if not hourly_match.empty:
        row = hourly_match.iloc[0]
        context["junction_fraction"] = row["junction_fraction"]
        context["weekend_fraction"] = row["weekend_fraction"]
        context["avg_label_count"] = row["avg_label_count"]

    return context


MODEL_METADATA = {
    "number_vehicle": {
        "filename": "number_vehicle.pkl",
        "features": [
            "lat_grid", "lon_grid", "hour_of_day", "day_of_week",
            "grid_total_violations", "grid_active_hours",
            "grid_unique_vehicle_types", "grid_junction_fraction",
        ],
        "description": "Predict vehicle/citation count for a grid+hour+day-of-week bucket.",
    },
    "typeofvehicle": {
        "filename": "TypeOfVehicle.pkl",
        "features": [
            "lat_grid", "lon_grid", "hour_of_day", "day_of_week",
            "grid_total_violations", "grid_active_hours",
            "grid_unique_vehicle_types", "grid_junction_fraction",
        ],
        "description": "Predict vehicle-type diversity score for a grid+hour+day-of-week bucket.",
    },
    "violation": {
        "filename": "voialtion.pkl",
        "features": [
            "lat_grid", "lon_grid", "hour_of_day",
            "grid_total_violations", "grid_active_hours",
            "grid_unique_vehicle_types", "grid_junction_fraction",
            "junction_fraction", "weekend_fraction", "avg_label_count",
        ],
        "description": "Predict exposure-normalized parking violation rate per day for a grid+hour.",
    },
}

_MODEL_CACHE: dict[str, Any] = {}


def load_model(filename: str) -> Any:
    if filename in _MODEL_CACHE:
        return _MODEL_CACHE[filename]

    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        model = joblib.load(path)
    except Exception as joblib_error:
        try:
            with path.open("rb") as f:
                model = pickle.load(f)
        except Exception as pickle_error:
            raise RuntimeError(
                f"Failed to load model '{filename}': joblib error={joblib_error}; pickle error={pickle_error}"
            ) from pickle_error

    _MODEL_CACHE[filename] = model
    return model


def predict(model_key: str, values: dict[str, Any]) -> Any:
    """Predict using the named model.

    `values` only needs to supply the caller-known inputs (lat_grid,
    lon_grid, hour_of_day, day_of_week) — grid-level context features are
    looked up automatically via get_grid_context() and merged in, so callers
    don't need to know the full enriched feature schema.
    """
    metadata = MODEL_METADATA[model_key]
    model = load_model(metadata["filename"])
    features = metadata["features"]

    if "lat_grid" not in values or "lon_grid" not in values:
        raise ValueError(f"Missing lat_grid/lon_grid for model '{model_key}'")

    hour_of_day = values.get("hour_of_day", values.get("hour", 0))
    context = get_grid_context(values["lat_grid"], values["lon_grid"], hour_of_day)
    merged = {**context, **values}
    # normalize hour key name across callers
    merged.setdefault("hour_of_day", hour_of_day)

    try:
        X = pd.DataFrame([[merged[feature] for feature in features]], columns=features)
    except KeyError as exc:
        raise ValueError(f"Missing feature for model '{model_key}': {exc.args[0]}") from exc

    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded object for '{model_key}' does not support predict()")

    prediction = model.predict(X)
    return prediction[0]


if __name__ == "__main__":
    print(f"Model directory: {MODEL_DIR}")
    print("Available models:")
    for key, metadata in MODEL_METADATA.items():
        print(f" - {key}: {metadata['filename']} ({metadata['description']})")

    example_inputs = {
        "number_vehicle": {"lat_grid": 12.975, "lon_grid": 77.575, "hour_of_day": 9, "day_of_week": 2},
        "typeofvehicle": {"lat_grid": 12.975, "lon_grid": 77.575, "hour_of_day": 9, "day_of_week": 2},
        "violation": {"lat_grid": 12.975, "lon_grid": 77.575, "hour_of_day": 9},
    }

    for key, example in example_inputs.items():
        print(f"\nPredicting with '{key}'...")
        try:
            prediction = predict(key, example)
            print(f"Prediction ({key}): {prediction}")
        except Exception as exc:
            print(f"Failed to predict for '{key}': {exc}")
