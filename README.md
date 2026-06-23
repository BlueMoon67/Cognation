# Cognation — AI-Driven Illegal Parking Enforcement & Traffic Intelligence

## Overview

Cognation is an AI-driven system that detects illegal parking hotspots in Bengaluru and quantifies their impact on traffic flow, so traffic enforcement can be prioritized by location and time of day instead of relying on reactive, patrol-based policing.

On-street illegal parking and spillover parking near commercial areas, metro stations, and event venues choke carriageways and intersections. Enforcement today is patrol-based and reactive, with no heatmap of parking violations against congestion impact and no data-driven way to prioritize where to send enforcement resources. Cognation addresses this by combining historical police violation data, live traffic congestion, weather, and vehicle/type patterns into a single per-grid risk score, refreshed continuously and shown on an interactive map.

The project consists of:

* **Backend (Flask + scikit-learn RandomForest models)** for grid-level violation and congestion scoring
* **Frontend (React + Leaflet)** for map visualization, hotspot ranking, and nearest-station lookup
* **SQLite** for storing live grid predictions (zero-config, embedded in the container)
* **Dockerized infrastructure**, deployed as a single container (worker + API) on Railway

---

## Features

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
* `final_score` weights are tuned to reflect the problem statement — violation-related signals carry the majority of the weight:

```
final_score = 0.05 × norm_volume
            + 0.20 × number_vehicle
            + 0.15 × type_score
            + 0.10 × violation_score
            + 0.50 × traffic_live_score
```

### Multi-Source Data Integration

* TomTom Traffic API (live congestion)
* OpenWeather API (live weather conditions)
* Historical police violation dataset (Jan–May, Bengaluru)
* Bengaluru police station locations

### Risk Categorization

`final_score` is a 0–1 composite. Suggested bands:

| final\_score | Risk Level |
| ------------ | ---------- |
| 0.00 – 0.25  | Low        |
| 0.25 – 0.50  | Moderate   |
| 0.50 – 0.75  | High       |
| 0.75 – 1.00  | Critical   |

---

## System Architecture

```
+------------------+      +------------------+
| TomTom Traffic   |      | OpenWeather API  |
| API (live)       |      | (live)           |
+------------------+      +------------------+
         |                        |
         v                        v
+-----------------------------------------------------------+
|               Prediction Worker (main.py)                 |
|                                                           |
|  Grid-level historical context (grid_features.csv)       |
|  joined with live signals                                 |
|                                                           |
|  3× RandomForestRegressor models:                        |
|   - voialtion.pkl       (violation rate per grid/hour)   |
|   - number_vehicle.pkl  (vehicle/citation count)         |
|   - TypeOfVehicle.pkl   (vehicle-type diversity)         |
|                                                           |
|  Min-max normalization → weighted final_score             |
+-----------------------------------------------------------+
         |
         v
+------------------+
| SQLite           |
| (traffic.db)     |
+------------------+
         |
         v
+------------------+
| Flask Backend    |
| (server.py)      |
| PORT from env    |
+------------------+
         |
         v
+------------------+
| React Frontend   |
| Leaflet Maps     |
+------------------+
```

---

## Technology Stack

**Backend:** Python, Flask, flask-cors, pandas, scikit-learn, joblib, python-dotenv

**Frontend:** React, Vite, React Leaflet, Leaflet.js

**DevOps:** Docker, Railway (single-container worker + API)

**APIs:** OpenWeather API, TomTom Traffic API

---

## Project Structure

```
Cognation/
│
├── PREDICTION/
│   ├── main.py              # Scoring loop: joins live + historical signals, writes final_score
│   ├── server.py            # Flask API — /health and /traffic endpoints
│   ├── prediction.py        # Model loading, grid-context joins, normalization
│   ├── Weather.py           # Live weather lookup + traffic_volume model
│   ├── TrafficLive.py       # TomTom congestion lookup
│   ├── database.py          # Creates SQLite schema (traffic_predictions table)
│   ├── insert.py            # Upserts predictions into SQLite
│   ├── entrypoint.sh        # Runs worker loop + Flask in one container
│   ├── Dockerfile
│   ├── requirements.txt
│   └── model/
│       ├── voialtion.pkl
│       ├── number_vehicle.pkl
│       ├── TypeOfVehicle.pkl
│       ├── traffic_volume_model.pkl
│       ├── grid_features.csv        # Per-grid historical context (1,409 grids)
│       └── unique_grids.csv         # Bengaluru grid cell coordinates
│
└── traffic-map/
    ├── src/
    │   └── App.jsx
    ├── package.json
    └── Dockerfile
```

---

## Getting Started

### Prerequisites

| Tool       | Version  | Install                                     |
| ---------- | -------- | ------------------------------------------- |
| Docker     | 20+      | https://docs.docker.com/get-docker/         |
| Node.js    | 18+      | https://nodejs.org/ (frontend dev only)     |
| Python     | 3.11+    | https://python.org (local dev only)         |

---

### Option A — Run with Docker (recommended)

This is the same path used on Railway. The single container runs both the prediction worker and the Flask API.

**1. Clone the repo**

```bash
git clone https://github.com/BlueMoon67/Future_Analyzer.git
cd Future_Analyzer
```

**2. Create your environment file**

```bash
cp PREDICTION/.env.example PREDICTION/.env   # if example exists, otherwise:
```

Create `PREDICTION/.env`:

```env
WEATHER_API=your_openweather_api_key
TRAFFIC_API=your_tomtom_api_key
PREDICT_INTERVAL_SECONDS=120
PORT=5000
```

> Get a free OpenWeather key at https://openweathermap.org/api  
> Get a free TomTom key at https://developer.tomtom.com — make sure **Traffic API** is enabled on the key (Maps/Search-only keys return 403 on every traffic call).

**3. Build the Docker image**

```bash
cd PREDICTION
docker build -t cognation-backend .
```

**4. Run the container**

```bash
docker run -p 5000:5000 --env-file .env cognation-backend
```

**5. Verify it's running**

```bash
# Health check
curl http://localhost:5000/health
# → {"status": "ok"}

# Traffic data (empty list until first prediction cycle completes ~2 min)
curl http://localhost:5000/traffic
# → [] initially, then [{grid_id, lat_grid, lon_grid, final_score, ...}, ...]
```

**6. Run the frontend**

```bash
cd ../traffic-map
npm install
VITE_API_URL=http://localhost:5000 npm run dev
```

Open http://localhost:5173 in your browser.

---

### Option B — Run locally without Docker

**Backend**

```bash
cd PREDICTION
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create traffic.db schema
python database.py

# Run one prediction cycle manually
python main.py

# Start the Flask API
python server.py
```

**Worker loop (separate terminal)**

```bash
source venv/bin/activate
while true; do python main.py; sleep 120; done
```

**Frontend**

```bash
cd traffic-map
npm install
npm run dev
```

---

### Option C — Deploy to Railway

**Backend service**

1. In Railway, create a new service and connect your GitHub repo.
2. Set the **Root Directory** to `PREDICTION`.
3. Railway auto-detects the `Dockerfile` — no build command needed.
4. Add these environment variables in Railway's Variables tab:

```
WEATHER_API      = your_openweather_api_key
TRAFFIC_API      = your_tomtom_api_key
PREDICT_INTERVAL_SECONDS = 120
PORT             = (Railway sets this automatically — do not hardcode)
```

5. Set **Health Check Path** to `/health` in Railway's service settings.
6. Deploy. The first prediction cycle takes ~10 minutes to score all 1,409 grids.

**Frontend service**

1. Create a second Railway service, set Root Directory to `traffic-map`.
2. Railway auto-detects the `Dockerfile`.
3. Add one environment variable:

```
VITE_API_URL = https://your-backend-service.up.railway.app
```

4. Deploy.

---

## API Reference

### `GET /health`

Always returns 200. Used by Railway's health check. Safe to call before the first prediction cycle.

```json
{"status": "ok"}
```

### `GET /traffic`

Returns all latest grid predictions. Returns `[]` before the first cycle completes.

```json
[
  {
    "grid_id": "12.975000_77.575000",
    "lat_grid": 12.975,
    "lon_grid": 77.575,
    "traffic_volume": 24863.98,
    "number_vehicle": 0.42,
    "type_score": 0.69,
    "violation_score": 0.58,
    "traffic_live_score": 0.62,
    "final_score": 0.57,
    "timestamp": "2026-06-22T09:00:00"
  }
]
```

All score fields are in `[0, 1]`. `final_score` is the weighted composite used for map coloring and enforcement ranking.

---

## Model Training

The three RandomForest models are reproducible from the raw police violation CSV:

```bash
cd PREDICTION/training
python train_violation_models.py --input /path/to/police_violation_cleaned.csv
```

This regenerates `voialtion.pkl`, `number_vehicle.pkl`, `TypeOfVehicle.pkl`, and the grid context CSVs.

**Key design decisions:**

* **Timezone correction** — source timestamps are UTC; all hour-of-day features are converted to IST (UTC+5:30) before training.
* **Grid-level context over raw coordinates** — earlier models used only `{lat_grid, lon_grid, hour}`, making lat/lon carry 70–80% of feature importance and memorizing specific grids. Current models add historical violation density, junction proximity, vehicle-type diversity, and weekend skew, dropping lat/lon importance to ~5% and improving held-out-grid R² from near-zero to 0.40–0.55.
* **Known limitation** — the strongest feature, historical violation count per grid, is derived from past enforcement at that grid. The models prioritize known hotspots by time-of-day well, but cannot independently discover new hotspots with no enforcement history.

---

## Troubleshooting

### Server returns 502 on Railway

Check that `/health` is set as the Health Check Path in Railway service settings. The container needs ~30 seconds to start Flask before Railway's check fires.

### `GET /traffic` returns `[]`

Normal on first boot. The prediction worker runs its first full cycle ~120 seconds after startup, scoring all 1,409 grids. Check Railway deploy logs for `[worker] cycle complete` to confirm it finished.

### TomTom returning 403 on every grid

Not a code issue. Check:

1. **Traffic API is explicitly enabled** on your key in the TomTom developer dashboard — a key scoped to Maps/Search only will 403 on Traffic calls regardless of quota.
2. **Daily quota** — TomTom free tier returns 403 (not 429) when the 2,500 req/day limit is exhausted.
3. Test the key directly in a browser:

```
https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=12.9716,77.5946&key=YOUR_KEY
```

When TomTom is down or quota-exceeded, `traffic_live_score` falls back to `0.0` for all grids — predictions still write to the DB, they just lack the live congestion signal.

### sklearn `UserWarning` flooding logs

If you see thousands of `X does not have valid feature names` warnings, they are harmless but can fill Railway's log buffer. Suppress them by adding to the top of `prediction.py` and `main.py`:

```python
import warnings
warnings.filterwarnings("ignore")
```

### Frontend cannot reach backend

```bash
curl https://your-backend.up.railway.app/health
```

If that fails, the backend isn't up. Check Railway deploy logs. If it succeeds but the frontend still fails, verify `VITE_API_URL` is set correctly in the frontend Railway service and that CORS is not blocking your frontend's origin.

---

## Future Improvements

* Complete road-network feature extraction (intersection density, signal proximity) for all 1,409 grids, to enable genuine new-hotspot discovery
* Bounding-box and top-N query parameters on `/traffic`
* Caching/batching for TomTom and OpenWeather calls to reduce per-cycle API usage
* Multi-city deployment
* Real-time anomaly detection layered on the congestion signal

---

## License

This project is intended for educational, research, and traffic enforcement analytics purposes.

---

## Author

**Mohammad Arham Reza**
