import React from "react";

function formatFeatureName(name) {
  const names = {
    dem: "Elevation (DEM)",
    aspect: "Aspect",
    slope: "Slope",
    slope_main: "Slope (Main)",
    drainage: "Drainage Dir.",
    hillshade: "Hillshade",
    twi: "TWI",
    dist_river: "Dist. to River",
    soil_type: "Soil Type",
    land_use: "Land Use",
  };
  return names[name] || name;
}

function getZoneColor(level) {
  const colors = {
    1: "#22c55e",
    2: "#84cc16",
    3: "#eab308",
    4: "#f97316",
    5: "#ef4444",
  };
  return colors[level] || "#6366f1";
}

export default function PredictionPanel({ active, onClose, prediction, loading, error }) {
  const susceptibility = prediction?.susceptibility ?? null;
  const zone = prediction?.zone ?? "—";
  const zone_level = prediction?.zone_level ?? 1;

  const circumference = 2 * Math.PI * 52;
  const arc = susceptibility == null ? 0 : susceptibility * circumference;

  return (
    <div id="prediction-panel" className={`side-panel prediction-panel ${active ? "" : "hidden"}`}>
      <div className="panel-header">
        <h2>Prediction Result</h2>
        <button className="close-btn" onClick={onClose}>&times;</button>
      </div>

      <div className="panel-body">
        <div className="risk-gauge">
          <div className="gauge-ring">
            <svg viewBox="0 0 120 120">
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke="rgba(255,255,255,0.1)"
                strokeWidth="8"
              />
              <circle
                id="gauge-arc"
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke={getZoneColor(zone_level)}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${arc} ${circumference}`}
                transform="rotate(-90 60 60)"
              />
            </svg>

            <div className="gauge-value">
              <span id="risk-percent">
                {loading ? "…" : susceptibility == null ? "—" : Math.round(susceptibility * 100)}
              </span>
              <small>%</small>
            </div>
          </div>

          <div
            id="risk-label"
            className={`risk-label zone-${loading ? zone_level : zone_level}`}
          >
            {error ? error : loading ? "Predicting..." : zone}
          </div>
        </div>

        <div className="info-row">
          <span className="info-label">Latitude</span>
          <span id="info-lat" className="info-value">
            {loading || prediction ? (prediction.lat ?? "—") : "—"}
          </span>
        </div>
        <div className="info-row">
          <span className="info-label">Longitude</span>
          <span id="info-lng" className="info-value">
            {loading || prediction ? (prediction.lng ?? "—") : "—"}
          </span>
        </div>
        <div className="info-row">
          <span className="info-label">Risk Zone</span>
          <span id="info-zone" className={`info-value zone-badge risk-label zone-${zone_level}`}>
            {loading ? "—" : zone}
          </span>
        </div>

        <h3 className="section-title">Feature Values</h3>
        <div id="feature-list" className="feature-list">
          {loading || !prediction ? (
            <p className="muted">{loading ? "Computing features..." : "Click map to predict"}</p>
          ) : (
            Object.entries(prediction.features || {}).map(([key, val]) => (
              <div key={key} className="feature-item">
                <span className="fname">{formatFeatureName(key)}</span>
                <span className="fval">
                  {typeof val === "number" ? val.toFixed(2) : val ?? "—"}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

