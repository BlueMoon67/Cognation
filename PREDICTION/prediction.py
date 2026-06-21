import pickle
from pathlib import Path
from typing import Any

import joblib

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
MODEL_METADATA = {
    "number_vehicle": {
        "filename": "number_vehicle.pkl",
        "features": ["lat_grid", "lon_grid", "hour", "day_of_week"],
        "description": "Predict number of vehicles.",
    },
    "typeofvehicle": {
        "filename": "TypeOfVehicle.pkl",
        "features": ["lat_grid", "lon_grid", "year", "month", "day", "hour", "minute", "dayofweek"],
        "description": "Predict vehicle type target.",
    },
    "violation": {
        "filename": "voialtion.pkl",
        "features": ["lat_grid", "lon_grid", "hour_of_day"],
        "description": "Predict violation score.",
    },
}


def load_model(filename: str) -> Any:
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        return joblib.load(path)
    except Exception as joblib_error:
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as pickle_error:
            raise RuntimeError(
                f"Failed to load model '{filename}': joblib error={joblib_error}; pickle error={pickle_error}"
            ) from pickle_error


def predict(model_key: str, values: dict[str, Any]) -> Any:
    metadata = MODEL_METADATA[model_key]
    model = load_model(metadata["filename"])
    features = metadata["features"]

    try:
        X = [[values[feature] for feature in features]]
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
        "number_vehicle": {"lat_grid": 12.9, "lon_grid": 77.6, "hour": 10, "day_of_week": 2},
        "typeofvehicle": {"lat_grid": 12.9, "lon_grid": 77.6, "year": 2026, "month": 6, "day": 17, "hour": 10, "minute": 0, "dayofweek": 0},
        "violation": {"lat_grid": 12.9, "lon_grid": 77.6, "hour_of_day": 10},
    }

    for key, example in example_inputs.items():
        print(f"\nPredicting with '{key}'...")
        try:
            prediction = predict(key, example)
            print(f"Prediction ({key}): {prediction}")
        except Exception as exc:
            print(f"Failed to predict for '{key}': {exc}")

