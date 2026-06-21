import { useEffect, useMemo, useState, useCallback } from "react";
import { MapContainer, TileLayer, Rectangle, Popup, Marker, useMap, useMapEvents } from "react-leaflet";
import { Navigation, RefreshCw, AlertCircle, Wifi, WifiOff, MapPin, X } from "lucide-react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const DEFAULT_GRID_SIZE_METERS = 300;
const REFRESH_INTERVAL_MS = 30000;

// ── Haversine distance (meters) ────────────────────────────────────────────
function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ── Shield icon (lucide) for police markers ────────────────────────────────
const SHIELD_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"
     fill="#1e40af" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>
</svg>`;

const SHIELD_SVG_ACTIVE = `
<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"
     fill="#f97316" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>
</svg>`;

function makeShieldIcon(active = false) {
  const bg = active ? "#7c2d12" : "#1e3a8a";
  const border = active ? "#fb923c" : "#93c5fd";
  const glow = active ? "rgba(251,146,60,0.8)" : "rgba(59,130,246,0.7)";
  const svg = active ? SHIELD_SVG_ACTIVE : SHIELD_SVG;
  return L.divIcon({
    className: "",
    html: `<div style="
      width:28px;height:28px;background:${bg};
      border:2px solid ${border};
      border-radius:50% 50% 50% 50%/60% 60% 40% 40%;
      display:flex;align-items:center;justify-content:center;
      box-shadow:0 0 10px ${glow};cursor:pointer;
    ">${svg}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    popupAnchor: [0, -16],
  });
}

// ── Map helpers ────────────────────────────────────────────────────────────
function FitBounds({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds.length > 0) map.fitBounds(bounds, { padding: [40, 40] });
  }, [map, bounds]);
  return null;
}

function RecenterButton({ bounds }) {
  const map = useMap();
  return (
    <button
      onClick={() => bounds?.length && map.fitBounds(bounds, { padding: [50, 50], animate: true })}
      style={{
        position: "fixed", top: "90px", left: "410px", zIndex: 600,
        padding: "10px", backgroundColor: "#007bff", color: "white",
        border: "none", borderRadius: "6px", width: "40px", height: "40px",
        cursor: "pointer", boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
        display: "flex", alignItems: "center", justifyContent: "center",
        transition: "background-color 0.2s",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#0056b3")}
      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#007bff")}
      title="Recenter map"
    >
      <Navigation size={24} />
    </button>
  );
}

// ── Parsers ────────────────────────────────────────────────────────────────
function parsePoint(item) {
  return {
    lat: parseFloat(item.lat_grid ?? item.latitude ?? 0),
    lon: parseFloat(item.lon_grid ?? item.longitude ?? 0),
    traffic_volume: parseFloat(item.traffic_volume ?? 0),
    number_vehicle: parseFloat(item.number_vehicle ?? 0),
    type_score: parseFloat(item.type_score ?? 0),
    violation_score: parseFloat(item.violation_score ?? 0),
    final_score: (parseFloat(item.final_score ?? 0))*100,
    raw: item,
  };
}

function parseCSV(text) {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
  return lines.slice(1).map((line) => {
    const vals = line.split(",").map((v) => v.trim().replace(/^"|"$/g, ""));
    return Object.fromEntries(headers.map((h, i) => [h, vals[i] ?? ""]));
  });
}

function detectLatLon(row) {
  const find = (...candidates) =>
    Object.keys(row).find((k) => candidates.includes(k.toLowerCase()));
  return {
    latKey: find("latitude", "lat", "y"),
    lonKey: find("longitude", "lon", "long", "lng", "x"),
  };
}

function getColorForScore(score) {
  if (score >= 90) return "#7f0000"; // Dark Red
  if (score >= 80) return "#b30000"; // Red
  if (score >= 60) return "#e34a33"; // Orange-Red
  if (score >= 40) return "#fdbb2d"; // Orange
  if (score >= 20) return "#fed976"; // Yellow
  return "#31a354";                  // Green
}

function formatDist(m) {
  return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`;
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [points, setPoints]           = useState([]);
  const [policeStations, setPolice]   = useState([]);
  const [showPolice, setShowPolice]   = useState(true);
  const [gridSize]                    = useState(DEFAULT_GRID_SIZE_METERS);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isOnline, setIsOnline]       = useState(true);

  // selected police station → nearest grid panel
  const [selectedStation, setSelectedStation] = useState(null); // { name, lat, lon }
  const [nearestGrids, setNearestGrids]       = useState([]);   // top-10 sorted by dist
  console.log("API URL:", import.meta.env.VITE_API_URL);
  // ── Fetch traffic data ──────────────────────────────────────────────────
  console.log("API URL:", import.meta.env.VITE_API_URL);
  const fetchData = useCallback(() => {
    fetch(`${import.meta.env.VITE_API_URL || "http://localhost:5000"}/traffic`)
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => {
        setPoints(Array.isArray(data) ? data.map(parsePoint) : []);
        setLastUpdated(new Date());
        setError(null); setIsOnline(true); setLoading(false);
      })
      .catch((err) => { setError(err.message); setIsOnline(false); setLoading(false); });
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, REFRESH_INTERVAL_MS);
    return () => clearInterval(id);
  }, [fetchData]);

  // ── Fetch police stations CSV ───────────────────────────────────────────
  useEffect(() => {
    fetch("/bangalore_police_stations.csv")
      .then((res) => { if (!res.ok) throw new Error("CSV not found"); return res.text(); })
      .then((text) => {
        const rows = parseCSV(text);
        if (!rows.length) return;
        const { latKey, lonKey } = detectLatLon(rows[0]);
        if (!latKey || !lonKey) return;
        const nameKey = Object.keys(rows[0]).find((k) =>
          ["name","station","station_name","ps_name","police_station"].includes(k.toLowerCase())
        );
        setPolice(
          rows
            .map((r) => ({ lat: parseFloat(r[latKey]), lon: parseFloat(r[lonKey]), name: nameKey ? r[nameKey] : "Police Station", raw: r }))
            .filter((s) => Number.isFinite(s.lat) && Number.isFinite(s.lon))
        );
      })
      .catch((err) => console.warn("Police CSV error:", err.message));
  }, []);

  // ── When station selected, compute nearest 10 grids ────────────────────
  const handleStationClick = useCallback((station) => {
    setSelectedStation(station);
    const withDist = points
      .filter((p) => Number.isFinite(p.lat) && Number.isFinite(p.lon))
      .map((p) => ({ ...p, dist: haversine(station.lat, station.lon, p.lat, p.lon) }))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, 10);
    setNearestGrids(withDist);
  }, [points]);

  // Set of nearest grid keys for fast lookup in rectangle renderer
  const nearestKeys = useMemo(
    () => new Set(nearestGrids.map((p) => `${p.lat}_${p.lon}`)),
    [nearestGrids]
  );

  // ── Derived ────────────────────────────────────────────────────────────
  const center = useMemo(() =>
    points.length > 0 ? [points[0].lat, points[0].lon] : [12.9716, 77.5946],
  [points]);

  const rectangles = useMemo(() =>
    points
      .filter((p) => Number.isFinite(p.lat) && Number.isFinite(p.lon))
      .map((point) => {
        const latOff = gridSize / 111320;
        const lonOff = gridSize / (111320 * Math.cos((point.lat * Math.PI) / 180));
        return {
          bounds: [
            [point.lat - latOff / 2, point.lon - lonOff / 2],
            [point.lat + latOff / 2, point.lon + lonOff / 2],
          ],
          point,
          isNearest: nearestKeys.has(`${point.lat}_${point.lon}`),
        };
      }),
  [points, gridSize, nearestKeys]);

  const mapBounds = useMemo(() => rectangles.flatMap((r) => r.bounds), [rectangles]);

  const maxBounds = useMemo(() => {
    if (!mapBounds.length) return null;
    const lats = mapBounds.map(([lat]) => lat);
    const lons  = mapBounds.map(([, lon]) => lon);
    return [[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]];
  }, [mapBounds]);

  const stats = useMemo(() => {
    if (!points.length) return { total: 0, maxScore: 0, minScore: 0, avgScore: "0" };
    const scores = points.map((p) => p.final_score);
    return {
      total: points.length,
      maxScore: Math.max(...scores),
      minScore: Math.min(...scores),
      avgScore: (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(4),
    };
  }, [points]);

  const sortedTop100 = useMemo(
    () => [...points].sort((a, b) => b.final_score - a.final_score).slice(0, 100),
    [points]
  );

  const shieldIcon       = useMemo(() => makeShieldIcon(false), []);
  const shieldIconActive = useMemo(() => makeShieldIcon(true),  []);

  const rightPanelOpen = !!selectedStation;

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div style={{ display: "flex", height: "100vh", width: "100vw", overflow: "hidden" }}>

      {/* ── Map ── */}
      <div style={{ flex: 1, position: "relative" }}>
        {loading && (
          <div style={{
            position: "absolute", inset: 0, zIndex: 700,
            display: "flex", alignItems: "center", justifyContent: "center",
            backgroundColor: "rgba(255,255,255,0.7)", fontSize: "14px", color: "#555", gap: "8px",
          }}>
            <RefreshCw size={16} style={{ animation: "spin 1s linear infinite" }} />
            Loading traffic data…
            <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
          </div>
        )}

        <MapContainer
          center={center} zoom={13} dragging touchZoom scrollWheelZoom doubleClickZoom
          boxZoom={false} keyboard={false} zoomControl={false}
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer attribution="© OpenStreetMap" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

          {/* Traffic grid rectangles */}
          {rectangles.map(({ bounds, point, isNearest }, i) => (
            <Rectangle
              key={`${point.lat}-${point.lon}-${i}`}
              bounds={bounds}
              pathOptions={
                isNearest
                  ? { color: "#b30000", fillColor: "#ff0000", weight: 2, fillOpacity: 0.55 }
                  : { color: getColorForScore(point.final_score), fillColor: getColorForScore(point.final_score), weight: 1, fillOpacity: 0.3 }
              }
            >
              <Popup>
                <div>
                  {isNearest && (
                    <div style={{ color: "#b30000", fontWeight: "bold", marginBottom: "4px", fontSize: "11px" }}>
                      📍 Near {selectedStation?.name}
                    </div>
                  )}
                  <strong>Grid Point</strong><br />
                  {point.lat.toFixed(6)}, {point.lon.toFixed(6)}<br />
                  <strong>Traffic:</strong> {point.traffic_volume}<br />
                  <strong>Vehicles:</strong> {point.number_vehicle}<br />
                  <strong>Type Score:</strong> {point.type_score.toFixed(4)}<br />
                  <strong>Violation Score:</strong> {point.violation_score.toFixed(4)}<br />
                  <strong>Final Score:</strong> {point.final_score.toFixed(4)}
                </div>
              </Popup>
            </Rectangle>
          ))}

          {/* Police markers */}
          {showPolice && policeStations.map((s, i) => {
            const isActive = selectedStation?.name === s.name && selectedStation?.lat === s.lat;
            return (
              <Marker
                key={`ps-${i}`}
                position={[s.lat, s.lon]}
                icon={isActive ? shieldIconActive : shieldIcon}
                eventHandlers={{ click: () => handleStationClick(s) }}
              >
                <Popup>
                  <div style={{ minWidth: "160px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
                      <span style={{ fontSize: "16px" }}>🛡️</span>
                      <strong style={{ fontSize: "13px" }}>{s.name}</strong>
                    </div>
                    <div style={{ fontSize: "11px", color: "#666", marginBottom: "6px" }}>
                      {s.lat.toFixed(5)}, {s.lon.toFixed(5)}
                    </div>
                    <button
                      onClick={() => handleStationClick(s)}
                      style={{
                        background: "#1e3a8a", color: "white", border: "none",
                        borderRadius: "4px", padding: "4px 10px", fontSize: "11px",
                        cursor: "pointer", width: "100%",
                      }}
                    >
                      Show 10 nearest grids →
                    </button>
                  </div>
                </Popup>
              </Marker>
            );
          })}

          {maxBounds && <RecenterButton bounds={maxBounds} />}
          {maxBounds && points.length > 0 && <FitBounds bounds={maxBounds} />}
        </MapContainer>
      </div>

      {/* ── Left Panel ── */}
      <div style={{
        position: "fixed", bottom: 0, left: 0, width: "400px", height: "100vh",
        backgroundColor: "white", borderRight: "1px solid #ddd",
        boxShadow: "2px 0 8px rgba(0,0,0,0.1)",
        display: "flex", flexDirection: "column", zIndex: 500,
      }}>
        <div style={{
          padding: "12px", borderBottom: "1px solid #eee", backgroundColor: "#f5f5f5",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <span style={{ fontWeight: "bold", fontSize: "14px" }}>Top 100 Scores</span>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button
              onClick={() => setShowPolice((v) => !v)}
              style={{
                background: showPolice ? "#1e3a8a" : "#e5e7eb",
                border: "none", borderRadius: "4px", cursor: "pointer",
                padding: "3px 7px", fontSize: "11px", color: showPolice ? "white" : "#555",
                display: "flex", alignItems: "center", gap: "4px",
              }}
            >
              <span>🛡️</span>{showPolice ? "Hide PS" : "Show PS"}
            </button>
            {isOnline ? <Wifi size={14} color="#22c55e" /> : <WifiOff size={14} color="#ef4444" />}
            <button onClick={fetchData} style={{ background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center", color: "#666" }}>
              <RefreshCw size={14} />
            </button>
          </div>
        </div>

        {policeStations.length > 0 && (
          <div style={{
            padding: "6px 12px", fontSize: "11px", color: "#1e40af",
            backgroundColor: "#eff6ff", borderBottom: "1px solid #bfdbfe",
            display: "flex", alignItems: "center", gap: "5px",
          }}>
            🛡️ {policeStations.length} police stations — click a marker to inspect
          </div>
        )}

        {lastUpdated && (
          <div style={{ padding: "6px 12px", fontSize: "11px", color: "#999", borderBottom: "1px solid #f0f0f0" }}>
            Updated {lastUpdated.toLocaleTimeString()}
          </div>
        )}

        {error && (
          <div style={{
            padding: "8px 12px", backgroundColor: "#fff0f0", borderBottom: "1px solid #ffc0c0",
            fontSize: "12px", color: "#c00", display: "flex", alignItems: "center", gap: "6px",
          }}>
            <AlertCircle size={13} />{error}
          </div>
        )}

        <div style={{ flex: 1, overflowY: "auto" }}>
          {points.length === 0 && !loading ? (
            <div style={{ padding: "12px", fontSize: "12px", color: "#999" }}>No data loaded</div>
          ) : (
            sortedTop100.map((point, index) => (
              <div
                key={`${point.lat}-${point.lon}-${index}`}
                style={{
                  padding: "10px 12px", borderBottom: "1px solid #f0f0f0",
                  fontSize: "12px", backgroundColor: index % 2 === 0 ? "#fafafa" : "white",
                }}
              >
                <div style={{ fontWeight: "bold", color: "#333", marginBottom: "3px", display: "flex", justifyContent: "space-between" }}>
                  <span>#{index + 1} • {point.final_score.toFixed(4)}</span>
                  <span style={{
                    width: "10px", height: "10px", borderRadius: "50%",
                    backgroundColor: getColorForScore(point.final_score),
                    display: "inline-block", marginTop: "2px",
                  }} />
                </div>
                <div style={{ color: "#666", fontSize: "11px" }}>{point.lat.toFixed(4)}, {point.lon.toFixed(4)}</div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* ── RIGHT PANEL — Nearest Grids ── */}
      {rightPanelOpen && (
        <div style={{
          position: "fixed", top: 0, right: 0, width: "400px", height: "100vh",
          backgroundColor: "white",
          borderLeft: "1px solid #ddd",
          boxShadow: "-2px 0 10px rgba(0,0,0,0.08)",
          display: "flex", flexDirection: "column", zIndex: 500,
        }}>
          {/* Header */}
          <div style={{
            padding: "14px 16px",
            backgroundColor: "#f5f7fb",
            borderBottom: "1px solid #e5e7eb",
            display: "flex", alignItems: "flex-start", justifyContent: "space-between",
          }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
                <span style={{ fontSize: "18px" }}>🛡️</span>
                <div>
                  <div style={{ color: "#111827", fontWeight: "700", fontSize: "14px" }}>
                    {selectedStation.name}
                  </div>
                  <div style={{ color: "#475569", fontSize: "12px" }}>
                    {selectedStation.lat.toFixed(5)}, {selectedStation.lon.toFixed(5)}
                  </div>
                </div>
              </div>
              <div style={{
                marginTop: "8px", display: "inline-flex", alignItems: "center", gap: "6px",
                background: "rgba(219,234,254,0.8)", border: "1px solid #bfdbfe",
                borderRadius: "6px", padding: "5px 10px",
              }}>
                <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "#2563eb", display: "inline-block" }} />
                <span style={{ color: "#1d4ed8", fontSize: "12px", fontWeight: "600" }}>Nearest 10 grids</span>
              </div>
            </div>
            <button
              onClick={() => { setSelectedStation(null); setNearestGrids([]); }}
              style={{
                background: "#e5e7eb", border: "none", borderRadius: "6px",
                color: "#334155", cursor: "pointer", padding: "6px",
                display: "flex", alignItems: "center", justifyContent: "center",
              }}
              title="Close"
            >
              <X size={18} />
            </button>
          </div>

          {/* Grid list */}
          <div style={{ flex: 1, overflowY: "auto", backgroundColor: "#ffffff" }}>
            {nearestGrids.length === 0 ? (
              <div style={{ padding: "18px 16px", color: "#64748b", fontSize: "13px" }}>
                Select a police station marker to view nearest grid zones.
              </div>
            ) : nearestGrids.map((point, i) => (
              <div
                key={`nearest-${i}`}
                style={{
                  padding: "14px 16px",
                  borderBottom: "1px solid #f0f0f0",
                  backgroundColor: i % 2 === 0 ? "#fafafa" : "#ffffff",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "10px", marginBottom: "8px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <div style={{
                      width: "26px", height: "26px",
                      backgroundColor: "#2563eb",
                      borderRadius: "8px",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: "white", fontSize: "12px", fontWeight: "700",
                    }}>
                      {i + 1}
                    </div>
                    <div>
                      <div style={{ color: "#0f172a", fontWeight: "700", fontSize: "13px" }}>
                        {point.final_score.toFixed(4)}
                      </div>
                      <div style={{ color: "#64748b", fontSize: "11px" }}>Traffic risk score</div>
                    </div>
                  </div>
                  <span style={{
                    backgroundColor: getColorForScore(point.final_score),
                    color: "white", fontSize: "10px", fontWeight: "700",
                    padding: "4px 10px", borderRadius: "999px",
                  }}>
                    {point.final_score >= 1.6 ? "Critical"
                      : point.final_score >= 1.2 ? "High"
                      : point.final_score >= 1 ? "Medium"
                      : point.final_score >= 0.4 ? "Low"
                      : "Clear"}
                  </span>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "8px" }}>
                  <MapPin size={14} color="#fb923c" />
                  <span style={{ color: "#334155", fontSize: "12px", fontWeight: "600" }}>
                    {formatDist(point.dist)}
                  </span>
                  <span style={{ color: "#94a3b8", fontSize: "11px" }}>from station</span>
                </div>

                <div style={{ color: "#64748b", fontSize: "12px", marginBottom: "10px" }}>
                  {point.lat.toFixed(5)}, {point.lon.toFixed(5)}
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "8px" }}>
                  {[
                    { label: "Vol", value: point.traffic_volume },
                    { label: "Veh", value: point.number_vehicle },
                    { label: "Viol", value: point.violation_score.toFixed(3) },
                  ].map(({ label, value }) => (
                    <div key={label} style={{
                      background: "#f8fafc", border: "1px solid #e2e8f0",
                      borderRadius: "8px", padding: "8px 10px", fontSize: "11px",
                    }}>
                      <div style={{ color: "#64748b", marginBottom: "2px" }}>{label}</div>
                      <div style={{ color: "#0f172a", fontWeight: "700" }}>{value}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Footer summary */}
          <div style={{
            padding: "12px 16px",
            borderTop: "1px solid #e5e7eb",
            backgroundColor: "#f8fafc",
            fontSize: "12px", color: "#475569",
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <span>Average distance</span>
            <span style={{ color: "#0f172a", fontWeight: "700" }}>
              {nearestGrids.length > 0
                ? formatDist(nearestGrids.reduce((s, p) => s + p.dist, 0) / nearestGrids.length)
                : "—"}
            </span>
          </div>
        </div>
      )}

      {/* ── Stats Cards ── */}
      <div style={{
        position: "fixed", top: "10px", left: "10px",
        display: "grid", gridTemplateColumns: "repeat(2,1fr)",
        gap: "10px", zIndex: 400, width: "280px",
      }}>
        {[
          { label: "Total",  value: stats.total,                   bg: "#f0f4f8", color: "#333"    },
          { label: "Avg",    value: stats.avgScore,                 bg: "#d1ecf1", color: "#0c5460" },
          { label: "Min",    value: stats.minScore.toFixed?.(3) ?? stats.minScore, bg: "#d4edda", color: "#155724" },
          { label: "Max",    value: stats.maxScore.toFixed?.(3) ?? stats.maxScore, bg: "#fff3cd", color: "#856404" },
        ].map(({ label, value, bg, color }) => (
          <div key={label} style={{ backgroundColor: bg, padding: "12px", borderRadius: "6px", boxShadow: "0 2px 4px rgba(0,0,0,0.1)" }}>
            <div style={{ fontSize: "11px", color: "#666", marginBottom: "3px" }}>{label}</div>
            <div style={{ fontSize: "20px", fontWeight: "bold", color }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
