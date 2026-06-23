# Cognation — AI-Driven Illegal Parking Enforcement & Traffic Intelligence

## Overview

Cognation is an AI-driven system that detects illegal parking hotspots in Bengaluru and quantifies their impact on traffic flow, so traffic enforcement can be prioritized by location and time of day instead of relying on reactive, patrol-based policing.

On-street illegal parking and spillover parking near commercial areas, metro stations, and event venues choke carriageways and intersections. Enforcement today is patrol-based and reactive, with no heatmap of parking violations against congestion impact and no data-driven way to prioritize where to send enforcement resources. Cognation addresses this by combining historical police violation data, live traffic congestion, weather, and vehicle/type patterns into a single per-grid risk score, refreshed continuously and shown on an interactive map.

The project consists of:

* **Backend (Flask + scikit-learn RandomForest models)** for grid-level violation and congestion scoring
* **Frontend (React + Leaflet)** for map visualization, hotspot ranking, and nearest-station lookup
* **PostgreSQL (Supabase)** for storing live grid predictions
* **Dockerized Infrastructure**, deployed as a single container (worker + API) on Railway

---

# Features

### Parking Violation Intelligence

* Trained on ~298,000 real police violation records (Jan–May, Bengaluru), covering vehicle type, violation type, junction, and timestamp.
* Per-grid historical violation density, junction proximity, vehicle-type diversity, and weekend/weekday skew.
* Hour-of-day patterns corrected to IST — the source timestamps are UTC, and earlier iterations of this model used the raw UTC hour, which skewed predicted enforcement activity by 5.5 hours.

### Real-Time Traffic Visualization

* Interactive map interface built with Leaflet.
* Grid-based hotspot visualization with a ranked top-violation panel.
* Nearest-police-station lookup per grid.

### AI-Based Composite Risk Scoring

* Three RandomForestRegressor models, retrained on grid-level historical context rather than raw coordinates alone, so predictions generalize to grids not seen during training (held-out-grid R² of 0.40–0.55, evaluated via grid-level GroupKFold, not random row splits):
  * **Violation score** — exposure-normalized parking violation rate per grid/hour
  * **Vehicle count** — predicted citation/vehicle volume per grid/hour/day-of-week
  * **Vehicle-type diversity** — mix of vehicle types cited per grid/hour/day-of-week
* A live traffic congestion signal (TomTom Traffic API) and live weather signal (OpenWeather), combined into a single `final_score` per grid.
* All inputs are percentile-normalized to a comparable 0–1 scale before weighting, so `final_score` means the same thing across every grid regardless of how skewed the underlying raw counts are.

### Multi-Source Data Integration

* TomTom Traffic API (live congestion)
* OpenWeather API (live weather conditions)
* Historical police violation dataset (Jan–May, Bengaluru)
* Bengaluru police station locations

### Risk Categorization

`final_score` is a 0–1 composite. Suggested bands (tune against your own deployment's score distribution before treating these as fixed):

| final_score | Risk Level |
| ----------- | ---------- |
| 0.00 – 0.25 | Low        |
| 0.25 – 0.50 | Moderate   |
| 0.50 – 0.75 | High       |
| 0.75 – 1.00 | Critical   |

### Docker Support

* Single-container deployment (worker + API) via Dockerfile, suited to Railway's single-service model.
* Worker loop re-scores all grids on an interval; Flask API serves the latest scores.

---

# System Architecture

```text
                   +------------------+      +------------------+
                   | TomTom Traffic   |      | OpenWeather API  |
                   | API (live)       |      | (live)           |
                   +------------------+      +------------------+
                            |                        |
                            v                        v
       +-----------------------------------------------------------+
       |               Prediction Worker (main.py)                 |
       |                                                            |
       |  Grid-level historical context (grid_static_features.csv, |
       |  grid_hourly_features.csv) joined with live signals        |
       |                                                            |
       |  3x RandomForestRegressor models:                         |
       |   - voialtion.pkl       (violation rate per grid/hour)    |
       |   - number_vehicle.pkl  (vehicle/citation count)          |
       |   - TypeOfVehicle.pkl   (vehicle-type diversity)           |
       |                                                            |
       |  Percentile normalization -> weighted final_score          |
       +-----------------------------------------------------------+
                            |
                            v
                   +------------------+
                   | PostgreSQL       |
                   | (Supabase)       |
                   +------------------+
                            |
                            v
                   +------------------+
                   | Flask Backend    |
                   | (server.py)      |
                   +------------------+
                            |
                            v
                   +------------------+
                   | React Frontend   |
                   | Leaflet Maps     |
                   +------------------+
```

---
<img width="680" height="760" alt="cognation_full_pipeline_flow" src="https://github.com/user-attachments/assets/10082a02-08ad-4c33-95ba-64f84fec9212" />

# Technology Stack

## Backend

* Python
* Flask
* Pandas, NumPy
* scikit-learn (RandomForestRegressor)
* joblib
* psycopg2 (PostgreSQL / Supabase)

## Frontend

* React
* Vite
* React Leaflet, Leaflet.js

## DevOps

* Docker
* Railway (single-container worker + API)

## APIs

* OpenWeather API
* TomTom Traffic API

---

# Project Structure

```text
Cognation-main/
│
├── PREDICTION/
│   ├── main.py                  # Scoring loop: joins live + historical signals, writes final_score
│   ├── server.py                # Flask API serving /traffic
│   ├── prediction.py            # Model loading, grid-context joins, percentile normalization
│   ├── Weather.py                # Live weather lookup + traffic_volume model
│   ├── TrafficLive.py            # TomTom congestion lookup
│   ├── database.py               # Postgres/Supabase connection
│   ├── insert.py                 # Upsert predictions into Postgres
│   ├── checkdb.py                # Quick DB introspection utility
│   ├── entrypoint.sh             # Runs worker loop + Flask server in one container
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── training/
│   │   └── train_violation_models.py   # Reproducible training script for all 3 RF models
│   └── model/
│       ├── voialtion.pkl
│       ├── number_vehicle.pkl
│       ├── TypeOfVehicle.pkl
│       ├── traffic_volume_model.pkl
│       ├── grid_static_features.csv     # Per-grid historical context
│       ├── grid_hourly_features.csv     # Per-grid-per-hour historical context
│       ├── number_vehicle_percentiles.csv
│       ├── violation_score_percentiles.csv
│       └── unique_grids.csv             # 1,409 grid cells across Bengaluru
│
└── traffic-map/
    ├── src/
    │   └── App.jsx
    ├── public/
    ├── package.json
    └── Dockerfile
```

---

# Model Training

The three RandomForest models are trained from the raw police violation CSV using `PREDICTION/training/train_violation_models.py`. This is a from-scratch, reproducible pipeline — earlier model versions existed only as `.pkl` files with no training code, which made them impossible to audit or retrain.

```bash
cd PREDICTION/training
python train_violation_models.py --input /path/to/police_violation_cleaned.csv
```

This regenerates `voialtion.pkl`, `number_vehicle.pkl`, `TypeOfVehicle.pkl`, and the grid context CSVs the models depend on at inference time.

**Key design decisions:**

* **Timezone correction**: source timestamps are UTC; Bengaluru is UTC+5:30. All hour-of-day features are converted to IST before training. Verify this still holds if the data source changes.
* **Grid-level context over raw coordinates**: earlier models took only `{lat_grid, lon_grid, hour}` as input, so lat/lon alone carried 70–80% of feature importance — the models were memorizing specific grid cells rather than learning generalizable patterns, and had no way to predict for any grid outside the training set. The current models add historical violation density, junction proximity, vehicle-type diversity, and weekend skew as features, dropping lat/lon's combined importance to roughly 5% and improving held-out-grid R² from near-zero/negative to 0.40–0.55.
* **Known limitation**: the strongest engineered feature, historical violation count per grid, is itself derived from past violations at that grid. This makes the models strong at prioritizing known hotspots by time-of-day, but it does not independently discover new hotspots with no enforcement history. Identifying genuinely new hotspots would need independent features such as road network density or proximity to commercial/transit points of interest — partially scaffolded in `grid_features.csv` but not yet complete for all 1,409 grids.

---

# Docker Deployment

## Prerequisites

* Docker

Verify installation:

```bash
docker --version
```

## Environment Variables

Create `PREDICTION/.env`:

```env
WEATHER_API=your_openweather_api_key
TRAFFIC_API=your_tomtom_api_key
DATABASE_URL=your_postgres_connection_string
PREDICT_INTERVAL_SECONDS=120
```

## Build and Run

```bash
cd PREDICTION
docker build -t cognation .
docker run -p 8080:8080 --env-file .env cognation
```

`entrypoint.sh` runs the prediction worker on a loop (default every 120 seconds) in the background, and the Flask API in the foreground.

---

# API Documentation

## Get Latest Grid Predictions

```http
GET /traffic
```

### Response

```json
[
  {
    "grid_id": "12.975000_77.575000",
    "lat_grid": 12.975,
    "lon_grid": 77.575,
    "traffic_volume": 24863.98,
    "number_vehicle": 335.78,
    "type_score": 0.6924,
    "violation_score": 13.588,
    "traffic_live_score": 0.62,
    "final_score": 0.82,
    "timestamp": "2026-06-22T09:00:00"
  }
]
```

---

# Frontend Configuration

```javascript
fetch(
  `${import.meta.env.VITE_API_URL || "http://localhost:8080"}/traffic`
)
```

---

# Database Persistence

Grid predictions are stored in PostgreSQL (Supabase), upserted on each scoring cycle so the latest score per grid is always available without growing the table unbounded.

---

# Future Improvements

* Complete road-network feature extraction (intersection density, signal proximity) for all 1,409 grids, to enable genuine new-hotspot discovery rather than only ranking known ones
* Bounding-box and top-N query parameters on `/traffic`, so the frontend and any enforcement-facing client can ask for a jurisdiction's top hotspots directly instead of fetching the full table
* Caching/batching for TomTom and OpenWeather calls to reduce per-cycle API usage as grid count grows
* Multi-city deployment
* Real-time incident/anomaly detection layered on top of the existing congestion signal

---

# Troubleshooting

## TomTom Traffic API returning 403 on every grid

This is not a per-grid issue — if every grid fails identically, check:

1. Whether the key has the **Traffic API product explicitly enabled** in the TomTom developer dashboard (a key scoped only to Maps/Search will 403 on Traffic calls regardless of quota)
2. Daily quota — TomTom's free tier returns **403** (not 429) when the 2,500 request/day limit is exhausted
3. Test the key directly in a browser, isolated from the app, to rule out anything code-side:
   ```
   https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=12.9716,77.5946&key=YOUR_KEY
   ```

## Frontend cannot fetch data

Verify the backend is reachable:

```bash
curl http://localhost:8080/traffic
```

---

# License

This project is intended for educational, research, and traffic enforcement analytics purposes.

---

# Author

**Mohammad Arham Reza**
