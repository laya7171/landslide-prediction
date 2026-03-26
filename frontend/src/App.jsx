import { useEffect, useMemo, useState } from "react";
import MapCanvas from "./components/MapCanvas.jsx";
import PredictionPanel from "./components/PredictionPanel.jsx";
import StatsPanel from "./components/StatsPanel.jsx";
import LayersPanel from "./components/LayersPanel.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function togglePanel(current, panelId) {
  return current === panelId ? null : panelId;
}

export default function App() {
  const [activePanel, setActivePanel] = useState(null);
  const [hasClicked, setHasClicked] = useState(false);

  const [loadingStage, setLoadingStage] = useState("loading"); // loading | fading | hidden

  const [susceptibilityData, setSusceptibilityData] = useState(null);
  const [landslidesGeojson, setLandslidesGeojson] = useState(null);
  const [nepalGeojson, setNepalGeojson] = useState(null);
  const [sindhupalchowkGeojson, setSindhupalchowkGeojson] = useState(null);
  const [stats, setStats] = useState(null);

  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState(null);
  const [prediction, setPrediction] = useState(null); // can hold partial draft during loading

  const [layerToggles, setLayerToggles] = useState({
    susceptibility: true,
    landslides: true,
    satellite: false,
    nepal: false,
    sindhupalchowk: true,
  });

  const showLoadingOverlay = loadingStage !== "hidden";
  const loadingFade = loadingStage === "fading";

  useEffect(() => {
    let cancelled = false;

    async function loadInitial() {
      try {
        setLoadingStage("loading");

        const [suscRes, landsRes, statsRes, nepalRes, sindRes] = await Promise.all([
          fetch(`${API_BASE}/api/susceptibility-image`),
          fetch(`${API_BASE}/api/landslides`),
          fetch(`${API_BASE}/api/stats`),
          fetch(`${API_BASE}/api/nepal-boundary`),
          fetch(`${API_BASE}/api/sindhupalchowk-boundary`),
        ]);

        if (!suscRes.ok) throw new Error("Failed to load susceptibility map");
        if (!landsRes.ok) throw new Error("Failed to load landslides");
        if (!statsRes.ok) throw new Error("Failed to load stats");

        const susc = await suscRes.json().catch(() => null);
        const lands = await landsRes.json().catch(() => null);
        const st = await statsRes.json().catch(() => null);
        const nep = await nepalRes.json().catch(() => null);
        const sin = await sindRes.json().catch(() => null);

        if (cancelled) return;
        setSusceptibilityData(susc);
        setLandslidesGeojson(lands);
        setNepalGeojson(nep);
        setSindhupalchowkGeojson(sin);
        setStats(st);
      } catch (err) {
        console.error(err);
      } finally {
        if (cancelled) return;
        setLoadingStage("fading");
        setTimeout(() => {
          if (!cancelled) setLoadingStage("hidden");
        }, 500);
      }
    }

    loadInitial();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleFirstClick = () => {
    setHasClicked(true);
  };

  const handlePredictionStart = ({ lat, lng }) => {
    setPredictionError(null);
    setPredictionLoading(true);
    setActivePanel("prediction-panel");
    setPrediction({ lat, lng });
  };

  const handlePredictionComplete = (data) => {
    setPredictionError(null);
    setPredictionLoading(false);
    setPrediction(data);
    setActivePanel("prediction-panel");
  };

  const handlePredictionError = (message) => {
    setPredictionError(message);
    setPredictionLoading(false);
    setActivePanel("prediction-panel");
  };

  const handleToggleLayer = (layerName) => {
    setLayerToggles((prev) => {
      if (layerName === "susceptibility") return { ...prev, susceptibility: !prev.susceptibility };
      if (layerName === "landslides") return { ...prev, landslides: !prev.landslides };
      if (layerName === "satellite") return { ...prev, satellite: !prev.satellite };
      if (layerName === "nepal") return { ...prev, nepal: !prev.nepal };
      if (layerName === "sindhupalchowk") return { ...prev, sindhupalchowk: !prev.sindhupalchowk };
      return prev;
    });
  };

  const loadingOverlayClass = useMemo(() => {
    return `loading-overlay ${loadingFade ? "fade-out" : ""}`;
  }, [loadingFade]);

  return (
    <>
      <header id="app-header">
        <div className="header-left">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 22h20L12 2z" />
              <path d="M12 9v5" />
              <circle cx="12" cy="17" r="1" />
            </svg>
          </div>
          <div>
            <h1>Landslide Susceptibility</h1>
            <p className="subtitle">Sindhupalchowk District, Nepal</p>
          </div>
        </div>

        <div className="header-right">
          <button
            id="btn-stats"
            className="header-btn"
            onClick={() => setActivePanel((cur) => togglePanel(cur, "stats-panel"))}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7" />
              <rect x="14" y="3" width="7" height="7" />
              <rect x="3" y="14" width="7" height="7" />
              <rect x="14" y="14" width="7" height="7" />
            </svg>
            <span>Dashboard</span>
          </button>

          <button
            id="btn-layers"
            className="header-btn"
            onClick={() => setActivePanel((cur) => togglePanel(cur, "layers-panel"))}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
            </svg>
            <span>Layers</span>
          </button>
        </div>
      </header>

      <MapCanvas
        apiBase={API_BASE}
        susceptibilityData={susceptibilityData}
        landslidesGeojson={landslidesGeojson}
        nepalGeojson={nepalGeojson}
        sindhupalchowkGeojson={sindhupalchowkGeojson}
        layerToggles={layerToggles}
        onFirstPredictionClick={handleFirstClick}
        onPredictionStart={handlePredictionStart}
        onPredictionComplete={handlePredictionComplete}
        onPredictionError={handlePredictionError}
      />

      <div
        id="click-hint"
        className={`click-hint ${hasClicked ? "fade-out" : ""}`}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M15 15l-2 5L9 9l11 4-5 2z" />
        </svg>
        Click anywhere on the map to predict landslide susceptibility
      </div>

      <PredictionPanel
        active={activePanel === "prediction-panel"}
        onClose={() => setActivePanel(null)}
        prediction={prediction}
        loading={predictionLoading}
        error={predictionError}
      />

      <StatsPanel active={activePanel === "stats-panel"} onClose={() => setActivePanel(null)} stats={stats} />
      <LayersPanel
        active={activePanel === "layers-panel"}
        onClose={() => setActivePanel(null)}
        layerToggles={layerToggles}
        onToggleLayer={handleToggleLayer}
      />

      {showLoadingOverlay && (
        <div
          id="loading-overlay"
          className={loadingOverlayClass}
          style={{ display: showLoadingOverlay ? "flex" : "none" }}
        >
          <div className="spinner"></div>
          <p>Loading susceptibility map...</p>
        </div>
      )}
    </>
  );
}

