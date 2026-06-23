# Cognation — AI-Driven Illegal Parking Enforcement & Traffic Intelligence

> **Live demo:** [cognation-red.vercel.app](https://cognation-red.vercel.app)  
> **Backend API:** [futureanaluzer-production.up.railway.app](https://futureanaluzer-production.up.railway.app/health)

---

## What it does

Illegal parking on Bengaluru's streets chokes intersections and triggers cascading congestion. Enforcement today is reactive — officers patrol and respond, with no data on *where* and *when* violations cluster or how badly they impact flow.

**Cognation** fixes this by combining:

- **~298,000 real Bengaluru police violation records** (Jan–May, anonymised)
- **Live traffic congestion** from TomTom's Flow API
- **Live weather** from OpenWeather
- **Three trained RandomForest models** (violation rate, vehicle count, vehicle-type mix)

…into a single `final_score` per 300 m × 300 m grid cell, refreshed every 2 minutes, shown on an interactive Leaflet map with a ranked hotspot panel and nearest-police-station lookup.

---

## Screenshots

| Heatmap view | Hotspot panel + station lookup |
|---|---|
| *(map with colour-coded grids from red → blue)* | *(right panel showing top grids + nearest station distance)* |

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐
│  TomTom Traffic  │     │  OpenWeather API  │
│  (live flow)     │     │  (live weather)   │
└────────┬─────────┘     └────────┬──────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────┐
│         Prediction Worker  (main.py)        │
│                                             │
│  1,409 Bengaluru grid cells                 │
│  ├── voialtion.pkl      → violation_score   │
│  ├── number_vehicle.pkl → number_vehicle    │
│  └── TypeOfVehicle.pkl  → type_score        │
│                                             │
│  final_score = 0.05 × norm_volume           │
│              + 0.20 × number_vehicle        │
│              + 0.15 × type_score            │
│              + 0.10 × violation_score       │
│              + 0.50 × traffic_live_score    │
└──────────────┬──────────────────────────────┘
               │
               ▼
        ┌─────────────┐
        │  SQLite DB  │  (traffic.db — embedded, zero config)
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  Flask API  │  GET /health  GET /traffic
        │  server.py  │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   React +   │
        │   Leaflet   │  colour-coded grid map, ranked panel,
        │  Frontend   │  police-station overlay
        └─────────────┘
```

The backend runs as a **single Docker container** — `entrypoint.sh` starts the prediction worker loop in the background and Flask in the foreground. Both processes share one `traffic.db` file inside the container.
<img width="680" height="760" alt="cognation_full_pipeline_flow" src="https://github.com/user-attachments/assets/009b1632-9ffa-4b1d-829f-a66155b3e244" />

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask, flask-cors |
| ML | scikit-learn RandomForestRegressor, joblib, pandas, numpy |
| Database | SQLite (embedded — no external DB needed) |
| Frontend | React 18, Vite, React-Leaflet, Leaflet.js, lucide-react |
| APIs | TomTom Traffic Flow API, OpenWeather API |
| Deploy | Docker, Railway (backend), Vercel (frontend) |

---

## Repository structure

```
Cognation/
├── PREDICTION/                    ← backend (build root for Railway)
│   ├── main.py                    # scoring loop — runs all grids, writes DB
│   ├── server.py                  # Flask API — /health and /traffic
│   ├── prediction.py              # loads .pkl models, runs inference
│   ├── Weather.py                 # OpenWeather lookup + traffic_volume model
│   ├── TrafficLive.py             # TomTom Flow API → congestion [0,1]
│   ├── database.py                # creates SQLite schema on first run
│   ├── insert.py                  # upserts one row per grid per cycle
│   ├── entrypoint.sh              # starts worker loop + Flask in one container
│   ├── Dockerfile
│   ├── requirements.txt
│   └── model/
│       ├── voialtion.pkl          # violation rate model
│       ├── number_vehicle.pkl     # vehicle count model
│       ├── TypeOfVehicle.pkl      # vehicle-type diversity model
│       ├── traffic_volume_model.pkl
│       ├── grid_features.csv      # per-grid historical context
│       └── unique_grids.csv       # 1,409 grid coordinates across Bengaluru
│
└── traffic-map/                   ← frontend (build root for Vercel)
    ├── src/
    │   └── App.jsx                # entire frontend in one file
    ├── public/
    │   └── bangalore_police_stations.csv
    ├── package.json
    ├── vite.config.js
    └── Dockerfile
```

---

## API reference

### `GET /health`

Returns 200 immediately, even before the first prediction cycle. Used by Railway's health check.

```json
{"status": "ok"}
```

### `GET /traffic`

Returns all latest grid predictions (one row per grid, upserted each cycle).  
Returns `[]` for the first ~10 minutes while the initial cycle runs.

```json
[
  {
    "grid_id":           "12.975000_77.575000",
    "lat_grid":          12.975,
    "lon_grid":          77.575,
    "traffic_volume":    24863.98,
    "number_vehicle":    0.42,
    "type_score":        0.69,
    "violation_score":   0.58,
    "traffic_live_score": 0.62,
    "final_score":       0.57,
    "timestamp":         "2026-06-23T14:00:00"
  }
]
```

All score fields are `[0, 1]`. `final_score` is the weighted composite used for map colouring and enforcement ranking.

### Risk bands

| final\_score | Colour on map | Label    |
|-------------|---------------|----------|
| 0.70 – 1.00 | Red           | Critical |
| 0.50 – 0.70 | Orange-red    | High     |
| 0.30 – 0.50 | Amber         | Medium   |
| 0.20 – 0.30 | Green         | Low      |
| 0.00 – 0.20 | Blue          | Clear    |

---

## ML models

### Training data

~298,000 Bengaluru police violation records (Jan–May). Each record contains grid coordinates, timestamp, vehicle type, violation type.

### Key design decisions

**Timezone correction** — source timestamps are UTC. All hour-of-day features are converted to IST (UTC+5:30) before training. An earlier version skipped this, shifting every predicted peak by 5.5 hours.

**Grid-level context, not raw coordinates** — early models used only `{lat, lon, hour}` as features. Lat/lon carried ~75% of feature importance — the models memorised specific cells and could not generalise to unseen grids. Current models add historical violation density, junction proximity, vehicle-type diversity, and weekend skew. Lat/lon importance dropped to ~5%. Held-out-grid R² improved from near-zero to 0.40–0.55 (evaluated via `GroupKFold` on grid IDs, not random row splits).

**Known limitation** — the strongest feature is historical violation count per grid. Models are strong at ranking *known* hotspots by time-of-day pattern, but cannot independently discover grids with zero enforcement history. Completing OSM-derived road-network features (intersection density, signal proximity) for all 1,409 grids would address this.

---

## `final_score` formula

```python
norm_volume = (traffic_volume - 10_000) / (40_000 - 10_000)  # clip to [0,1]

final_score = (
    0.05 * norm_volume          # background traffic volume
  + 0.20 * number_vehicle       # predicted citation volume
  + 0.15 * type_score           # vehicle-type diversity
  + 0.10 * violation_score      # historical violation rate
  + 0.50 * traffic_live_score   # live TomTom congestion
)
```

All inputs are `[0, 1]`, weights sum to 1.0, so `final_score` is always `[0, 1]`.

When TomTom is unavailable (403, quota, timeout), `traffic_live_score` falls back to `0.0` — predictions still write to the DB, they just lack the live signal.

---

---

# Instructions to Run

---

## Prerequisites

| Tool | Min version | Install |
|---|---|---|
| **Docker** | 20+ | https://docs.docker.com/get-docker/ |
| **Git** | any | https://git-scm.com/ |
| **Node.js** | 18+ | https://nodejs.org/ — LTS version (frontend only) |

You also need two free API keys (see Step 2).

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/BlueMoon67/Cognation.git
cd Cognation
```

---

## Step 2 — Get API keys

**OpenWeather** (free, instant)
1. Go to https://openweathermap.org/api
2. Sign up → My API Keys → copy the default key

**TomTom** (free, instant)
1. Go to https://developer.tomtom.com
2. Sign up → Dashboard → Create App
3. ⚠️ In the app settings, **enable the Traffic API product** — a Maps-only key returns 403 on every traffic call
4. Copy the API key

---

## Step 3 — Create the environment file

```bash
cd PREDICTION
```

Create a file named `.env` with this exact content:

```env
WEATHER_API=your_openweather_key_here
TRAFFIC_API=your_tomtom_key_here
PREDICT_INTERVAL_SECONDS=120
PORT=5000
```

Replace the placeholder values with your real keys. Do not add quotes around the values.

---

## Step 4 — Build the Docker image

```bash
# Still inside PREDICTION/
docker build -t cognation-backend .
```

First build takes **2–4 minutes** (downloads Python 3.11-slim, installs packages).  
Subsequent builds use the layer cache and are much faster.

You should see it end with something like:

```
Database ready at /app/traffic.db
Successfully built <image-id>
Successfully tagged cognation-backend:latest
```

---

## Step 5 — Start the backend

```bash
docker run -p 5000:5000 --env-file .env cognation-backend
```

Expected output in the first 30 seconds:

```
[entrypoint] Starting Flask backend on port 5000...
[worker] starting prediction cycle...
[2026-06-23T...] Predicting for 1409 grid blocks...
  12.970000,77.715000 → vol=24249.7 veh=0.311 type=1.158 viol=0.062 live=0.000 final=0.095
  12.970000,77.725000 → ...
```

Flask starts **immediately** — the API is available before the first cycle finishes.

---

## Step 6 — Verify the backend

Open a second terminal:

```bash
# Should return {"status": "ok"} within 1 second
curl http://localhost:5000/health

# Returns [] for the first ~10 minutes, then returns all grid scores
curl http://localhost:5000/traffic
```

The first full cycle scores all 1,409 grids and takes **8–12 minutes**.  
Watch the terminal from Step 5 for `[worker] cycle complete` to know when data is ready.

---

## Step 7 — Run the frontend

Open a third terminal:

```bash
cd traffic-map
npm install
```

**Mac / Linux:**
```bash
VITE_API_URL=http://localhost:5000 npm run dev
```

**Windows (Command Prompt):**
```cmd
set VITE_API_URL=http://localhost:5000 && npm run dev
```

**Windows (PowerShell):**
```powershell
$env:VITE_API_URL="http://localhost:5000"; npm run dev
```

Then open **http://localhost:5173** in your browser.

The map loads immediately with an empty grid. After the first cycle completes (Step 6), refresh the page — the heatmap will populate with colour-coded enforcement hotspots.

---

## Using the live deployment instead

The backend and frontend are already deployed. To skip Docker entirely and just run the frontend against the live API:

```bash
cd traffic-map
npm install
VITE_API_URL=https://futureanaluzer-production.up.railway.app npm run dev
```

Or just open: **https://cognation-red.vercel.app**

---

## Stopping everything

```bash
# List running containers
docker ps

# Stop the backend container
docker stop <CONTAINER_ID>

# Stop the frontend dev server
Ctrl+C  (in the terminal running npm run dev)
```

---

## Deploying to Railway (backend)

1. Push repo to GitHub
2. In Railway → New Project → Deploy from GitHub repo
3. Set **Root Directory** = `PREDICTION`
4. Railway auto-detects the `Dockerfile`
5. In Railway → Variables, add:

```
WEATHER_API      = your_openweather_key
TRAFFIC_API      = your_tomtom_key
PREDICT_INTERVAL_SECONDS = 120
```

Do **not** set `PORT` — Railway injects it automatically. If you hardcode it, the healthcheck fails with 502.

6. In Railway → Settings → Health Check Path = `/health`
7. Deploy. First cycle takes ~10 minutes.

---

## Deploying to Vercel (frontend)

1. In Vercel → New Project → import the repo
2. Set **Root Directory** = `traffic-map`
3. Framework preset = **Vite**
4. In Environment Variables, add:

```
VITE_API_URL = https://your-backend-service.up.railway.app
```

5. Deploy.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/traffic` returns `[]` | First cycle still running | Wait ~10 min, watch logs for `cycle complete` |
| Every grid shows `live=0.000` | TomTom 403 — Traffic API not enabled or quota exceeded | Enable Traffic API in TomTom dashboard, or test key at `https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=12.9716,77.5946&key=YOUR_KEY` |
| Railway returns 502 | PORT hardcoded or health check path not set | Remove `PORT` from Variables; set Health Check Path to `/health` |
| `docker: command not found` | Docker not installed | Install from https://docs.docker.com/get-docker/ |
| `npm: command not found` | Node not installed | Install LTS from https://nodejs.org/ |
| Port 5000 already in use | Another process on 5000 | Run with `-p 5001:5000` and use `http://localhost:5001` everywhere |
| Map is blank / CORS error | `VITE_API_URL` wrong or missing | Check browser console (F12); confirm URL matches exactly what backend is running on |
| `sklearn UserWarning` flooding logs | Model was trained with named DataFrame, inference passes plain list | Add `import warnings; warnings.filterwarnings("ignore")` at top of `prediction.py` and `main.py` |

---

## Future improvements

- Complete OSM road-network features (intersection density, signal proximity) for all 1,409 grids — enables discovery of new hotspots with no enforcement history
- Top-N and bounding-box query params on `/traffic` for jurisdiction-scoped enforcement dashboards
- Batch TomTom + OpenWeather calls per cycle to reduce API usage as grid count grows
- Anomaly detection layer on top of the congestion signal
- Multi-city support

---

## Author

**Mohammad Arham Reza**  
B.Tech ECE, NIT Patna (2023–2027)  
GitHub: [BlueMoon67](https://github.com/BlueMoon67)
