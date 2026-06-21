import os
import pickle
from pathlib import Path
from typing import Any

import joblib
import requests
import pandas as pd
import sklearn.compose._column_transformer as _column_transformer
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("WEATHER_API")


def get_weather_condition(lat, lon, default: str = "Clear"):
    if not API_KEY:
        return default

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}"
        f"&appid={API_KEY}"
    )

    data = requests.get(url).json()

    weather = data["weather"][0]["main"]

    mapping = {
        "Clear": "Clear",
        "Clouds": "Cloudy",
        "Rain": "Rainy",
        "Drizzle": "Rainy",
        "Thunderstorm": "Storm",
        "Mist": "Foggy",
        "Fog": "Foggy",
        "Haze": "Foggy",
        "Smoke": "Foggy",
        "Dust": "Dusty",
        "Sand": "Dusty",
    }

    return mapping.get(weather, weather)


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
TRAFFIC_MODEL_FILE = "traffic_volume_model.pkl"


def _ensure_rmainder_cols_list() -> None:
    if not hasattr(_column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _column_transformer._RemainderColsList = _RemainderColsList


def load_traffic_model() -> Any:
    path = MODEL_DIR / TRAFFIC_MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"Traffic volume model not found at {path}")

    _ensure_rmainder_cols_list()

    try:
        return joblib.load(path)
    except Exception:
        _ensure_rmainder_cols_list()
        try:
            return joblib.load(path)
        except Exception:
            with path.open("rb") as f:
                return pickle.load(f)


def predict_traffic_volume(
    lat: float,
    lon: float,
    year: int,
    month: int,
    day_of_week: int,
    hour: int,
    weather_condition: str | None = None,
) -> Any:
    model = load_traffic_model()
    if weather_condition is None:
        weather_condition = get_weather_condition(lat, lon)

    values = {
        "Year": year,
        "Month": month,
        "DayOfWeek": day_of_week,
        "Hour": hour,
        "latitude": lat,
        "longitude": lon,
        "Weather Conditions": weather_condition,
    }
    X = pd.DataFrame([values])

    if not hasattr(model, "predict"):
        raise TypeError("Loaded traffic model does not support predict()")

    prediction = model.predict(X)
    return prediction[0]


if __name__ == "__main__":
    condition = get_weather_condition(12.9716, 77.5946)
    print(condition)
    print("Traffic volume prediction:", predict_traffic_volume(12.9716, 77.5946, 2026, 6, 0, 10))
