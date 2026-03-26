import { useEffect, useMemo, useRef } from "react";
import Chart from "chart.js/auto";

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

export default function StatsPanel({ active, onClose, stats }) {
  const chartCanvasRef = useRef(null);
  const chartContainerRef = useRef(null);
  const chartInstanceRef = useRef(null);

  const metrics = stats?.metrics || null;

  const featureImportance = useMemo(() => {
    return metrics?.feature_importance || {};
  }, [metrics]);

  const hasFeatureImportance = useMemo(() => {
    return Object.keys(featureImportance).length > 0;
  }, [featureImportance]);

  useEffect(() => {
    if (!metrics) return;

    const importance = featureImportance || {};
    const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]);
    const labels = sorted.map(([k]) => formatFeatureName(k));
    const values = sorted.map(([, v]) => (v * 100).toFixed(1));

    if (labels.length === 0) {
      return;
    }

    const colors = [
      "#6366f1",
      "#818cf8",
      "#a5b4fc",
      "#c7d2fe",
      "#e0e7ff",
      "#93c5fd",
      "#60a5fa",
      "#3b82f6",
      "#2563eb",
    ];

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }

    if (!chartCanvasRef.current) return;

    const ctx = chartCanvasRef.current.getContext("2d");

    // Set canvas height based on number of features.
    if (chartContainerRef.current) {
      chartContainerRef.current.style.height = `${Math.max(200, labels.length * 32)}px`;
    }

    chartInstanceRef.current = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Importance (%)",
            data: values,
            backgroundColor: colors.slice(0, labels.length),
            borderRadius: 6,
            borderSkipped: false,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#1a2235",
            titleColor: "#f1f5f9",
            bodyColor: "#94a3b8",
            borderColor: "rgba(255,255,255,0.1)",
            borderWidth: 1,
            cornerRadius: 8,
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: { color: "#64748b", font: { size: 11 } },
          },
          y: {
            grid: { display: false },
            ticks: { color: "#94a3b8", font: { size: 11, weight: 500 } },
          },
        },
      },
    });

    return () => {
      // Do not destroy here; we handle it before recreation.
    };
  }, [metrics, featureImportance]);

  return (
    <div id="stats-panel" className={`side-panel stats-panel ${active ? "" : "hidden"}`}>
      <div className="panel-header">
        <h2>Model Dashboard</h2>
        <button className="close-btn" onClick={onClose}>&times;</button>
      </div>

      <div className="panel-body">
        {!metrics ? (
          <p className="muted">Loading model dashboard...</p>
        ) : (
          <>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-value">{(metrics.accuracy * 100).toFixed(1)}%</div>
                <div className="metric-label">Accuracy</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{(metrics.f1_score * 100).toFixed(1)}%</div>
                <div className="metric-label">F1 Score</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{(metrics.roc_auc * 100).toFixed(1)}%</div>
                <div className="metric-label">ROC AUC</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{(metrics.precision * 100).toFixed(1)}%</div>
                <div className="metric-label">Precision</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{(metrics.recall * 100).toFixed(1)}%</div>
                <div className="metric-label">Recall</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{(metrics.cv_auc_mean * 100).toFixed(1)}%</div>
                <div className="metric-label">CV AUC</div>
              </div>
            </div>

            <h3 className="section-title">Confusion Matrix</h3>
            <div className="confusion-matrix">
              <table>
                <thead>
                  <tr>
                    <th></th>
                    <th>Pred: No</th>
                    <th>Pred: Yes</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      <strong>Actual: No</strong>
                    </td>
                    <td>{metrics.confusion_matrix.TN}</td>
                    <td>{metrics.confusion_matrix.FP}</td>
                  </tr>
                  <tr>
                    <td>
                      <strong>Actual: Yes</strong>
                    </td>
                    <td>{metrics.confusion_matrix.FN}</td>
                    <td>{metrics.confusion_matrix.TP}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h3 className="section-title">Feature Importance</h3>
            {hasFeatureImportance ? (
              <div className="chart-container" ref={chartContainerRef}>
                <canvas ref={chartCanvasRef} />
              </div>
            ) : (
              <p className="muted">Not available for this model type.</p>
            )}

            <h3 className="section-title">Training Info</h3>
            <div className="info-row">
              <span className="info-label">Algorithm</span>
              <span className="info-value">{metrics.model_name ?? "Random Forest"}</span>
            </div>
            {metrics.n_estimators != null && (
              <div className="info-row">
                <span className="info-label">Trees</span>
                <span className="info-value">{metrics.n_estimators}</span>
              </div>
            )}
            {metrics.hidden_layers && (
              <div className="info-row">
                <span className="info-label">Hidden Layers</span>
                <span className="info-value">{metrics.hidden_layers.join(" - ")}</span>
              </div>
            )}
            <div className="info-row">
              <span className="info-label">Training Samples</span>
              <span className="info-value">{metrics.n_train ?? "—"}</span>
            </div>
            <div className="info-row">
              <span className="info-label">Test Samples</span>
              <span className="info-value">{metrics.n_test ?? "—"}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

