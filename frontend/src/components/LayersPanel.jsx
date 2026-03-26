export default function LayersPanel({ active, onClose, layerToggles, onToggleLayer }) {
  return (
    <div id="layers-panel" className={`side-panel layers-panel ${active ? "" : "hidden"}`}>
      <div className="panel-header">
        <h2>Map Layers</h2>
        <button className="close-btn" onClick={onClose}>&times;</button>
      </div>

      <div className="panel-body">
        <div className="layer-item">
          <label className="toggle-switch">
            <input
              type="checkbox"
              id="toggle-susceptibility"
              checked={!!layerToggles.susceptibility}
              onChange={() => onToggleLayer?.("susceptibility")}
            />
            <span className="slider"></span>
          </label>
          <span>Susceptibility Overlay</span>
        </div>

        <div className="layer-item">
          <label className="toggle-switch">
            <input
              type="checkbox"
              id="toggle-landslides"
              checked={!!layerToggles.landslides}
              onChange={() => onToggleLayer?.("landslides")}
            />
            <span className="slider"></span>
          </label>
          <span>Past Landslides</span>
        </div>

        <div className="layer-item">
          <label className="toggle-switch">
            <input
              type="checkbox"
              id="toggle-nepal"
              checked={!!layerToggles.nepal}
              onChange={() => onToggleLayer?.("nepal")}
            />
            <span className="slider"></span>
          </label>
          <span>Nepal Boundary</span>
        </div>

        <div className="layer-item">
          <label className="toggle-switch">
            <input
              type="checkbox"
              id="toggle-sindhupalchowk"
              checked={!!layerToggles.sindhupalchowk}
              onChange={() => onToggleLayer?.("sindhupalchowk")}
            />
            <span className="slider"></span>
          </label>
          <span>Sindhupalchowk Boundary</span>
        </div>

        <div className="layer-item">
          <label className="toggle-switch">
            <input
              type="checkbox"
              id="toggle-satellite"
              checked={!!layerToggles.satellite}
              onChange={() => onToggleLayer?.("satellite")}
            />
            <span className="slider"></span>
          </label>
          <span>Satellite Imagery</span>
        </div>

        <h3 className="section-title" style={{ marginTop: 24 }}>
          Susceptibility Legend
        </h3>
        <div className="legend">
          <div className="legend-item">
            <span className="legend-color" style={{ background: "#228B22" }}></span> Very Low (0–20%)
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: "#9ACD32" }}></span> Low (20–40%)
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: "#FFD700" }}></span> Moderate (40–60%)
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: "#FF8C00" }}></span> High (60–80%)
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: "#DC143C" }}></span> Very High (80–100%)
          </div>
        </div>
        <div className="legend" style={{ marginTop: 16 }}>
          <div className="legend-item">
            <span
              className="legend-color"
              style={{ background: "rgba(255, 80, 80, 0.5)", border: "2px solid #ff5050" }}
            ></span>{" "}
            Past Landslide
          </div>
        </div>
      </div>
    </div>
  );
}

