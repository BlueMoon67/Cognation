# 🚦 Cognation — AI Traffic Congestion Analyzer

![Python](https://img.shields.io/badge/Python-52.5%25-3776AB?style=flat&logo=python&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-37.9%25-F7DF1E?style=flat&logo=javascript&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=flat&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=flat&logo=react&logoColor=black)
![Live Demo](https://img.shields.io/badge/Live%20Demo-cognation--red.vercel.app-brightgreen?style=flat)

> An AI-powered traffic congestion prediction and visualization platform for city planners, traffic authorities, and commuters.

---

## 📌 Overview

Cognation combines historical traffic patterns, weather conditions, road network data, and live traffic feeds to generate congestion predictions across city grids. Results are rendered on an interactive map dashboard, letting users identify high-risk zones and make data-driven decisions in real time.

---

## ✨ Features

### 🗺️ Real-Time Traffic Visualization
- Interactive map interface powered by Leaflet
- Grid-based congestion heatmap with dynamic marker updates
- Color-coded risk levels updated from backend predictions

### 🤖 AI-Based Congestion Prediction
Machine learning models trained on:
- Historical traffic data
- Weather conditions
- Temporal features (hour of day, day of week)
- Road network information

### 🔗 Multi-Source Data Integration
- Traffic APIs
- Weather APIs (OpenWeather)
- Parking and congestion datasets
- Geographic location datasets

### 📊 Risk Categorization

| Congestion Score | Risk Level |
|:---:|:---:|
| 0 – 25 | 🟢 Low |
| 26 – 50 | 🟡 Moderate |
| 51 – 75 | 🟠 High |
| 76 – 100 | 🔴 Critical |

### 🐳 Docker Support
- One-command deployment via Docker Compose
- Separate backend and frontend containers
- Persistent SQLite database volume

---

## 🏗️ System Architecture

```
  Traffic APIs ──┐
                 ├──▶  Traffic Prediction Engine
  Weather APIs ──┘      (Feature Engineering +
                         Random Forest / XGBoost / Ensemble)
                                    │
                                    ▼
                            SQLite Database
                                    │
                                    ▼
                            Flask Backend
                          (REST API: /traffic)
                                    │
                                    ▼
                          React Frontend
                          (Leaflet Map UI)
```
<img width="680" height="760" alt="cognation_full_pipeline_flow" src="https://github.com/user-attachments/assets/33cb5840-6ae7-4df8-9c90-584fec4783e1" />

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python, Flask, Pandas, NumPy, Scikit-Learn, SQLite |
| **Frontend** | React, Vite, React Leaflet, Leaflet.js |
| **DevOps** | Docker, Docker Compose, Nginx |
| **APIs** | OpenWeather API, Traffic API |

---

## 📁 Project Structure

```
Cognation/
├── docker-compose.yml
├── PREDICTION/               # Flask backend + ML engine
│   ├── server.py
│   ├── main.py
│   ├── requirements.txt
│   ├── traffic.db
│   ├── Dockerfile
│   ├── .dockerignore
│   └── .env
└── traffic-map/              # React frontend
    ├── src/
    │   └── App.jsx
    ├── public/
    ├── package.json
    ├── nginx.conf
    ├── Dockerfile
    └── .dockerignore
```

---

## 🚀 Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Docker Compose

Verify your installation:

```bash
docker --version
docker compose version
```

### 1. Configure Environment Variables

Create a `.env` file inside the `PREDICTION/` directory:

```bash
# PREDICTION/.env
WEATHER_API=your_openweather_api_key
TRAFFIC_API=your_traffic_api_key
```

### 2. Build and Run

From the project root:

```bash
docker compose up --build
```

### 3. Access the Application

| Service | URL |
|---|---|
| Frontend (React Map) | http://localhost:5173 |
| Backend (Flask API) | http://localhost:5000 |
| Traffic Endpoint | http://localhost:5000/traffic |

---

## 📡 API Reference

### `GET /traffic`

Returns congestion predictions for all tracked grid zones.

**Response:**
```json
[
  {
    "grid_id": "GRID_001",
    "latitude": 12.9716,
    "longitude": 77.5946,
    "congestion_score": 68.5
  }
]
```

---

## ⚙️ Frontend Configuration

The frontend dynamically resolves the backend URL:

```javascript
fetch(`${import.meta.env.VITE_API_URL || "http://localhost:5000"}/traffic`)
```

For local development without Docker:
```bash
npm run dev   # uses http://localhost:5000 by default
```

For Docker deployment, the `VITE_API_URL` is set via Docker Compose automatically.

---

## 🗄️ Database Persistence

Traffic predictions are stored in `traffic.db` (SQLite) and mounted as a Docker volume, ensuring:
- Data survives container restarts
- Easy local inspection
- Persistent storage across deployments

---

## 🧠 ML Model Workflow

```
Data Collection
    │  Historical traffic, weather, road network, temporal data
    ▼
Feature Engineering
    │  Hour of day · Day of week · Weather · Traffic density · Coordinates
    ▼
ML Prediction
    │  Congestion Score (0–100) per grid cell
    ▼
Visualization
    │  Map markers · Heat zones · Color-coded levels
```

---

## 🔧 Troubleshooting

**Containers not running?**
```bash
docker ps
```
Start Docker Desktop if no containers appear.

**Backend not responding?**
```bash
docker compose logs backend
```
Then verify `http://localhost:5000/traffic` is accessible.

**Frontend can't fetch data?**
Check that the backend is running and the `/traffic` endpoint returns data.

**Rebuild from scratch:**
```bash
docker compose down
docker compose up --build
```

---

## 🔮 Future Improvements

- Real-time streaming traffic updates
- Advanced deep learning models (LSTM / Transformer)
- Route optimization engine
- Smart parking integration
- Incident detection & traffic anomaly alerts
- Multi-city deployment
- Live analytics dashboard

---

## 👤 Author

**Mohammad Arham Reza**

🌐 [cognation-red.vercel.app](https://cognation-red.vercel.app)

---

## 📄 License

This project is intended for educational, research, and traffic analytics purposes.
