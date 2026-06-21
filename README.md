# Cognition Analyzer

## Overview

Cognition Analyzer is an AI-powered traffic congestion prediction and visualization platform designed to help city planners, traffic authorities, and commuters understand real-time and future traffic conditions.

The system combines historical traffic patterns, weather conditions, road network information, and live traffic data to generate congestion predictions across different city grids. Predictions are displayed through an interactive map-based dashboard, enabling users to identify high-risk congestion zones and make informed decisions.

The project consists of:

* **Backend (Flask + Machine Learning)** for traffic prediction
* **Frontend (React + Leaflet)** for map visualization
* **SQLite Database** for storing congestion predictions
* **Dockerized Infrastructure** for easy deployment

---
<img width="1536" height="1024" alt="ChatGPT Image Jun 20, 2026, 07_52_59 PM" src="https://github.com/user-attachments/assets/411b4472-9bb1-43ce-b43b-dd336b65cd46" />

# Features

### Real-Time Traffic Visualization

* Interactive map interface built using Leaflet.
* Grid-based congestion visualization.
* Dynamic marker updates from backend predictions.

### AI-Based Congestion Prediction

* Machine Learning models trained on:

  * Historical traffic data
  * Weather conditions
  * Temporal features
  * Road network information

### Multi-Source Data Integration

* Traffic APIs
* Weather APIs
* Parking and congestion datasets
* Geographic location datasets

### Risk Categorization

| Congestion Score | Risk Level |
| ---------------- | ---------- |
| 0 - 25           | Low        |
| 26 - 50          | Moderate   |
| 51 - 75          | High       |
| 76 - 100         | Critical   |

### Docker Support

* One-command deployment using Docker Compose.
* Backend and frontend containers.
* Persistent database storage.

---

# System Architecture

```text
                   +------------------+
                   | Traffic APIs     |
                   +------------------+
                            |
                            v
                   +------------------+
                   | Weather APIs     |
                   +------------------+
                            |
                            v
+------------------------------------------------+
|            Traffic Prediction Engine           |
|                                                |
|  Feature Engineering + ML Models               |
|  (Random Forest / XGBoost / Ensemble)          |
+------------------------------------------------+
                            |
                            v
                   +------------------+
                   | SQLite Database  |
                   +------------------+
                            |
                            v
                   +------------------+
                   | Flask Backend    |
                   +------------------+
                            |
                            v
                   +------------------+
                   | React Frontend   |
                   | Leaflet Maps     |
                   +------------------+
```

---

# Technology Stack

## Backend

* Python
* Flask
* Pandas
* NumPy
* Scikit-Learn
* SQLite

## Frontend

* React
* Vite
* React Leaflet
* Leaflet.js

## DevOps

* Docker
* Docker Compose
* Nginx

## APIs

* OpenWeather API
* Traffic API

---

# Project Structure

```text
basic/
│
├── docker-compose.yml
│
├── PREDICTION/
│   ├── server.py
│   ├── main.py
│   ├── requirements.txt
│   ├── traffic.db
│   ├── Dockerfile
│   ├── .dockerignore
│   └── .env
│
└── traffic-map/
    ├── src/
    │   └── App.jsx
    ├── public/
    ├── package.json
    ├── nginx.conf
    ├── Dockerfile
    └── .dockerignore
```

---

# Docker Deployment

## Prerequisites

* Docker Desktop
* Docker Compose

Verify installation:

```bash
docker --version
docker compose version
```

---

## Environment Variables

Create:

```text
PREDICTION/.env
```

Example:

```env
WEATHER_API=your_openweather_api_key
TRAFFIC_API=your_traffic_api_key
```

---

## Build and Run

From the project root:

```bash
cd basic

docker compose up --build
```

---

## Services

### Backend

Runs Flask API:

```text
http://localhost:5000
```

Traffic endpoint:

```text
http://localhost:5000/traffic
```

---

### Frontend

Runs React application:

```text
http://localhost:5173
```

---

# API Documentation

## Get Traffic Predictions

### Request

```http
GET /traffic
```

### Response

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

# Frontend Configuration

The frontend dynamically reads the backend URL using:

```javascript
fetch(
  `${import.meta.env.VITE_API_URL || "http://localhost:5000"}/traffic`
)
```

This allows:

### Local Development

```bash
npm run dev
```

Uses:

```text
http://localhost:5000
```

### Docker Deployment

Uses:

```env
VITE_API_URL=http://localhost:5000
```

configured through Docker Compose.

---

# Database Persistence

Traffic predictions are stored in:

```text
traffic.db
```

The database is mounted as a Docker volume to ensure:

* Predictions survive container restarts
* Easy local inspection
* Persistent storage across deployments

---

# Model Workflow

## Data Collection

Sources:

* Historical traffic data
* Weather data
* Road network features
* Temporal information

## Feature Engineering

Generated features:

* Hour of day
* Day of week
* Weather conditions
* Traffic density
* Geographic coordinates

## Prediction

Machine learning model generates:

```text
Congestion Score (0-100)
```

for each traffic grid.

## Visualization

Predictions are displayed as:

* Map markers
* Heat zones
* Color-coded congestion levels

---

# Future Improvements

* Real-time streaming traffic updates
* Advanced deep learning models
* Route optimization
* Smart parking integration
* Incident detection
* Traffic anomaly alerts
* Multi-city deployment
* Live dashboard analytics

---

# Performance

### Backend

* Fast prediction retrieval
* Lightweight SQLite storage
* Dockerized deployment

### Frontend

* Responsive map rendering
* Real-time marker updates
* Mobile-friendly UI

---

# Troubleshooting

## Docker Not Running

```bash
docker ps
```

Start Docker Desktop if containers are not running.

---

## API Not Accessible

Check backend logs:

```bash
docker compose logs backend
```

---

## Frontend Cannot Fetch Data

Verify:

```bash
http://localhost:5000/traffic
```

is accessible.

---

## Rebuild Containers

```bash
docker compose down

docker compose up --build
```

---

# License

This project is intended for educational, research, and traffic analytics purposes.

---

# Author

**Mohammad Arham Reza**
