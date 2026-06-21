from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Iterable

from Weather import predict_traffic_volume
from prediction import predict
from TrafficLive import get_congestion
from insert import save_prediction
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

def locate_grid_file() -> Path:
    local_grid = BASE_DIR / "model" / "unique_grids.csv"
    root_grid = BASE_DIR.parent / "model" / "unique_grids.csv"
    if local_grid.exists():
        return local_grid
    if root_grid.exists():
        return root_grid
    raise FileNotFoundError(
        "Could not find unique_grids.csv in PREDICTION/model or ../model"
    )

GRID_FILE = locate_grid_file()
OUTPUT_FILE = ROOT_DIR / "traffic_updates.csv"


def read_grid_blocks(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Grid file not found: {path}")

    grid_blocks_set: set[tuple[float, float]] = set()
    with path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if "lat_grid" not in reader.fieldnames or "lon_grid" not in reader.fieldnames:
            raise ValueError("CSV must contain lat_grid and lon_grid columns")

        for row in reader:
            try:
                lat = float(row["lat_grid"])
                lon = float(row["lon_grid"])
            except (TypeError, ValueError):
                continue
            grid_blocks_set.add((lat, lon))

    return sorted(grid_blocks_set)


def current_time_parts(dt: datetime.datetime) -> dict[str, int]:
    return {
        "year": dt.year,
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "hour": dt.hour,
    }


def write_update_header(path: Path) -> None:
    if path.exists():
        return

    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp",
            "lat_grid",
            "lon_grid",
            "traffic_volume",
            "number_vehicle",
            "type_score",
            "violation_score",
            "TrafficLive_score",
            "final_score",
            "score_count",
        ])


def append_update(path: Path, row: Iterable[object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


def predict_for_all_grids(grid_blocks: list[tuple[float, float]]) -> None:
    now = datetime.datetime.now()
    time_parts = current_time_parts(now)

    print(f"[{now.isoformat()}] Predicting traffic for {len(grid_blocks)} grid blocks...")

    write_update_header(OUTPUT_FILE)

    for lat, lon in grid_blocks:
        try:
            traffic_volume = predict_traffic_volume(
                lat=lat,
                lon=lon,
                year=time_parts["year"],
                month=time_parts["month"],
                day_of_week=time_parts["day_of_week"],
                hour=time_parts["hour"],
                weather_condition="Clear",
            )
            number_vehicle = predict(
                "number_vehicle",
                {
                    "lat_grid": lat,
                    "lon_grid": lon,
                    "hour": time_parts["hour"],
                    "day_of_week": time_parts["day_of_week"],
                },
            )
            type_score = predict(
                "typeofvehicle",
                {
                    "lat_grid": lat,
                    "lon_grid": lon,
                    "year": time_parts["year"],
                    "month": time_parts["month"],
                    "day": datetime.datetime.now().day,
                    "hour": time_parts["hour"],
                    "minute": 0,
                    "dayofweek": time_parts["day_of_week"],
                },
            )
            violation_score = predict(
                "violation",
                {
                    "lat_grid": lat,
                    "lon_grid": lon,
                    "hour_of_day": time_parts["hour"],
                },
            )
            TrafficLive_score = get_congestion(lat, lon)
            print(f"  {lat:.6f},{lon:.6f} -> traffic_volume={traffic_volume}, number_vehicle={number_vehicle}, type_score={type_score}, violation_score={violation_score}, TrafficLive_score={TrafficLive_score}")
            final_score = (
                 (0.05 / 40000) * traffic_volume
                + 0.2 * number_vehicle 
                + 0.2 * type_score
                + 0.1 * violation_score 
                + 0.55 * TrafficLive_score
            )
            score_count = 5
            append_update(
                OUTPUT_FILE,
                [
                    now.isoformat(),
                    lat,
                    lon,
                    traffic_volume,
                    number_vehicle,
                    type_score,
                    violation_score,
                    TrafficLive_score,
                    final_score,
                    score_count,
                ],
            )
            grid_id = f"{lat:.6f}_{lon:.6f}"
            save_prediction(
            grid_id=grid_id,
            timestamp=now.isoformat(),
            lat_grid=lat,
            lon_grid=lon,
            traffic_volume=traffic_volume,
            number_vehicle=number_vehicle,
            type_score=type_score,
            violation_score=violation_score,
            traffic_live_score=TrafficLive_score,
            final_score=final_score
            )
            print(f"  {lat:.6f},{lon:.6f} -> final_score={final_score}")
        except Exception as exc:
            print(f"Skipping grid {lat:.6f},{lon:.6f}: {exc}")
            continue


if __name__ == "__main__":
    grid_blocks = read_grid_blocks(GRID_FILE)
    print(f"Loaded {len(grid_blocks)} grid blocks from {GRID_FILE}")
    predict_for_all_grids(grid_blocks)
