/**
 * Landslide Susceptibility Prediction — Frontend App
 * ===================================================
 * Interactive map with click-to-predict, susceptibility overlay,
 * and model dashboard.
 */

const API_BASE = "http://localhost:8000";

// ====================================================================
// Map Setup
// ====================================================================

const map = L.map("map", {
    center: [27.85, 85.7],
    zoom: 11,
    zoomControl: true,
    attributionControl: true,
});

// Base layers
const osmTile = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
});

const satelliteTile = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
        maxZoom: 18,
        attribution: "&copy; Esri",
    }
);

osmTile.addTo(map);

// Layer references
let susceptibilityLayer = null;
let landslideLayer = null;
let predictionMarker = null;
let satelliteActive = false;

// ====================================================================
// Load Data
// ====================================================================

async function init() {
    try {
        // Load susceptibility overlay
        await loadSusceptibilityMap();
        // Load past landslides
        await loadLandslides();
        // Load stats
        await loadStats();
        // Hide loading
        hideLoading();
    } catch (err) {
        console.error("Init error:", err);
        hideLoading();
    }
}

async function loadSusceptibilityMap() {
    try {
        const res = await fetch(`${API_BASE}/api/susceptibility-image`);
        const data = await res.json();

        const bounds = [
            [data.bounds.south, data.bounds.west],
            [data.bounds.north, data.bounds.east],
        ];

        susceptibilityLayer = L.imageOverlay(data.image, bounds, {
            opacity: 0.65,
            interactive: false,
        });
        susceptibilityLayer.addTo(map);

        // Fit map to data bounds
        map.fitBounds(bounds, { padding: [20, 20] });
    } catch (err) {
        console.error("Failed to load susceptibility map:", err);
    }
}

async function loadLandslides() {
    try {
        const res = await fetch(`${API_BASE}/api/landslides`);
        const geojson = await res.json();

        landslideLayer = L.geoJSON(geojson, {
            style: {
                fillColor: "#ff5050",
                fillOpacity: 0.35,
                color: "#ff5050",
                weight: 1.5,
                opacity: 0.7,
            },
            onEachFeature: (feature, layer) => {
                const props = feature.properties;
                let popup = "<div style='font-family:Inter,sans-serif;'>";
                popup += "<b style='color:#ff5050;'>Past Landslide</b><br>";
                if (props.year) popup += `Year: ${props.year}<br>`;
                if (props.area_geome) popup += `Area: ${props.area_geome.toFixed(4)}<br>`;
                popup += "</div>";
                layer.bindPopup(popup);
            },
        });
        landslideLayer.addTo(map);
    } catch (err) {
        console.error("Failed to load landslides:", err);
    }
}

async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/api/stats`);
        const data = await res.json();
        const m = data.metrics;

        // Populate metric cards
        document.getElementById("stat-accuracy").textContent = (m.accuracy * 100).toFixed(1) + "%";
        document.getElementById("stat-f1").textContent = (m.f1_score * 100).toFixed(1) + "%";
        document.getElementById("stat-auc").textContent = (m.roc_auc * 100).toFixed(1) + "%";
        document.getElementById("stat-precision").textContent = (m.precision * 100).toFixed(1) + "%";
        document.getElementById("stat-recall").textContent = (m.recall * 100).toFixed(1) + "%";
        document.getElementById("stat-cv").textContent = (m.cv_auc_mean * 100).toFixed(1) + "%";

        // Confusion matrix
        document.getElementById("cm-tn").textContent = m.confusion_matrix.TN;
        document.getElementById("cm-fp").textContent = m.confusion_matrix.FP;
        document.getElementById("cm-fn").textContent = m.confusion_matrix.FN;
        document.getElementById("cm-tp").textContent = m.confusion_matrix.TP;

        // Training info
        document.getElementById("stat-trees").textContent = m.n_estimators;
        document.getElementById("stat-train").textContent = m.n_train;
        document.getElementById("stat-test").textContent = m.n_test;

        // Feature importance chart
        renderImportanceChart(m.feature_importance);
    } catch (err) {
        console.error("Failed to load stats:", err);
    }
}

function renderImportanceChart(importance) {
    const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]);
    const labels = sorted.map(([k]) => formatFeatureName(k));
    const values = sorted.map(([, v]) => (v * 100).toFixed(1));

    const colors = [
        "#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe",
        "#e0e7ff", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb",
    ];

    const ctx = document.getElementById("importance-chart").getContext("2d");
    new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Importance (%)",
                data: values,
                backgroundColor: colors.slice(0, labels.length),
                borderRadius: 6,
                borderSkipped: false,
            }],
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

    // Set canvas height based on number of features
    document.getElementById("importance-chart").parentElement.style.height =
        Math.max(200, labels.length * 32) + "px";
}

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
    };
    return names[name] || name;
}

// ====================================================================
// Map Click → Predict
// ====================================================================

map.on("click", async (e) => {
    const { lat, lng } = e.latlng;

    // Hide hint
    const hint = document.getElementById("click-hint");
    hint.classList.add("fade-out");

    // Show prediction panel
    openPanel("prediction-panel");

    // Set loading state
    document.getElementById("risk-percent").textContent = "...";
    document.getElementById("risk-label").textContent = "Predicting...";
    document.getElementById("risk-label").className = "risk-label";

    // Place marker
    if (predictionMarker) {
        map.removeLayer(predictionMarker);
    }
    predictionMarker = L.marker([lat, lng], {
        icon: L.divIcon({
            className: "prediction-marker",
            iconSize: [16, 16],
            iconAnchor: [8, 8],
        }),
    }).addTo(map);

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lat, lng }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Prediction failed");
        }

        const data = await res.json();
        displayPrediction(data);
    } catch (err) {
        document.getElementById("risk-percent").textContent = "!";
        document.getElementById("risk-label").textContent = err.message;
        document.getElementById("risk-label").className = "risk-label";
        console.error("Prediction error:", err);
    }
});

function displayPrediction(data) {
    const pct = (data.susceptibility * 100).toFixed(1);

    // Update gauge
    document.getElementById("risk-percent").textContent = Math.round(data.susceptibility * 100);
    const circumference = 2 * Math.PI * 52;
    const arc = (data.susceptibility * circumference).toFixed(1);
    const gaugeArc = document.getElementById("gauge-arc");
    gaugeArc.style.strokeDasharray = `${arc} ${circumference}`;
    gaugeArc.style.stroke = getZoneColor(data.zone_level);

    // Update label
    const riskLabel = document.getElementById("risk-label");
    riskLabel.textContent = data.zone;
    riskLabel.className = `risk-label zone-${data.zone_level}`;

    // Coordinates
    document.getElementById("info-lat").textContent = data.lat.toFixed(6);
    document.getElementById("info-lng").textContent = data.lng.toFixed(6);

    // Zone badge
    const zoneBadge = document.getElementById("info-zone");
    zoneBadge.textContent = data.zone;
    zoneBadge.className = `info-value zone-badge risk-label zone-${data.zone_level}`;

    // Feature values
    const featureList = document.getElementById("feature-list");
    featureList.innerHTML = "";
    for (const [key, val] of Object.entries(data.features)) {
        const item = document.createElement("div");
        item.className = "feature-item";
        item.innerHTML = `
            <span class="fname">${formatFeatureName(key)}</span>
            <span class="fval">${typeof val === "number" ? val.toFixed(2) : val}</span>
        `;
        featureList.appendChild(item);
    }
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

// ====================================================================
// Panel Controls
// ====================================================================

function togglePanel(panelId) {
    const panel = document.getElementById(panelId);
    const isHidden = panel.classList.contains("hidden");

    // Close all panels
    document.querySelectorAll(".side-panel").forEach((p) => p.classList.add("hidden"));

    // Toggle the requested one
    if (isHidden) {
        panel.classList.remove("hidden");
    }
}

function openPanel(panelId) {
    document.querySelectorAll(".side-panel").forEach((p) => p.classList.add("hidden"));
    document.getElementById(panelId).classList.remove("hidden");
}

function closePanel(panelId) {
    document.getElementById(panelId).classList.add("hidden");
}

// ====================================================================
// Layer Toggles
// ====================================================================

function toggleLayer(layerName) {
    switch (layerName) {
        case "susceptibility":
            if (susceptibilityLayer) {
                if (map.hasLayer(susceptibilityLayer)) {
                    map.removeLayer(susceptibilityLayer);
                } else {
                    susceptibilityLayer.addTo(map);
                }
            }
            break;
        case "landslides":
            if (landslideLayer) {
                if (map.hasLayer(landslideLayer)) {
                    map.removeLayer(landslideLayer);
                } else {
                    landslideLayer.addTo(map);
                }
            }
            break;
        case "satellite":
            if (satelliteActive) {
                map.removeLayer(satelliteTile);
                osmTile.addTo(map);
                satelliteActive = false;
            } else {
                map.removeLayer(osmTile);
                satelliteTile.addTo(map);
                satelliteActive = true;
            }
            break;
    }
}

// ====================================================================
// Loading
// ====================================================================

function hideLoading() {
    const overlay = document.getElementById("loading-overlay");
    overlay.classList.add("fade-out");
    setTimeout(() => (overlay.style.display = "none"), 500);
}

// ====================================================================
// Init
// ====================================================================
init();
