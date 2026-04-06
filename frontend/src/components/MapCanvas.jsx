import { useEffect, useRef } from "react";
import * as L from "leaflet";

const API_PREFIX = "/api";

export default function MapCanvas({
  apiBase,
  susceptibilityData,
  landslidesGeojson,
  nepalGeojson,
  sindhupalchowkGeojson,
  layerToggles,
  onFirstPredictionClick,
  onPredictionStart,
  onPredictionComplete,
  onPredictionError,
}) {
  const mapHostRef = useRef(null);

  const mapRef = useRef(null);
  const osmTileRef = useRef(null);
  const satelliteTileRef = useRef(null);
  const susceptibilityLayerRef = useRef(null);
  const landslideLayerRef = useRef(null);
  const predictionMarkerRef = useRef(null);
  const nepalLayerRef = useRef(null);
  const sindhupalchowkLayerRef = useRef(null);

  // Initialize map once.
  useEffect(() => {
    const map = L.map(mapHostRef.current, {
      center: [27.85, 85.7],
      zoom: 11,
      zoomControl: true,
      attributionControl: true,
    });
    mapRef.current = map;

    const osmTile = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    });

    const satelliteTile = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        maxZoom: 18,
        attribution: "&copy; Esri",
      },
    );

    osmTile.addTo(map);
    osmTileRef.current = osmTile;
    satelliteTileRef.current = satelliteTile;

    map.on("click", async (e) => {
      const { lat, lng } = e.latlng;

      onFirstPredictionClick?.();
      onPredictionStart?.({ lat, lng });

      // Marker placement
      if (predictionMarkerRef.current) {
        predictionMarkerRef.current.remove();
      }
      predictionMarkerRef.current = L.marker([lat, lng], {
        icon: L.divIcon({
          className: "prediction-marker",
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        }),
      }).addTo(map);

      try {
        const res = await fetch(`${apiBase}${API_PREFIX}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ lat, lng }),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => null);
          throw new Error(err?.detail || "Prediction failed");
        }

        const data = await res.json();
        onPredictionComplete?.(data);
      } catch (err) {
        onPredictionError?.(err?.message || "Prediction error");
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Satellite toggle
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const satellite = satelliteTileRef.current;
    const osm = osmTileRef.current;
    if (!satellite || !osm) return;

    const shouldShowSatellite = !!layerToggles.satellite;

    if (shouldShowSatellite) {
      map.removeLayer(osm);
      satellite.addTo(map);
    } else {
      map.removeLayer(satellite);
      osm.addTo(map);
    }
  }, [layerToggles.satellite]);

  // Susceptibility overlay toggle
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Remove old layer first if present.
    if (susceptibilityLayerRef.current) {
      susceptibilityLayerRef.current.remove();
      susceptibilityLayerRef.current = null;
    }

    if (!layerToggles.susceptibility) return;
    if (!susceptibilityData?.image || !susceptibilityData?.bounds) return;

    const bounds = [
      [susceptibilityData.bounds.south, susceptibilityData.bounds.west],
      [susceptibilityData.bounds.north, susceptibilityData.bounds.east],
    ];

    const overlay = L.imageOverlay(susceptibilityData.image, bounds, {
      opacity: 0.8,
      interactive: false,
    });

    overlay.addTo(map);
    susceptibilityLayerRef.current = overlay;
    map.fitBounds(bounds, { padding: [20, 20] });
  }, [layerToggles.susceptibility, susceptibilityData]);

  // Landslides toggle
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (landslideLayerRef.current) {
      landslideLayerRef.current.remove();
      landslideLayerRef.current = null;
    }

    if (!layerToggles.landslides) return;
    if (!landsLidesAreValid(landslidesGeojson)) return;

    const geoLayer = L.geoJSON(landslidesGeojson, {
      style: {
        fillColor: "#ff5050",
        fillOpacity: 0.35,
        color: "#ff5050",
        weight: 1.5,
        opacity: 0.7,
      },
      onEachFeature: (feature, layer) => {
        const props = feature.properties || {};
        let popup = "<div style='font-family:Inter,sans-serif;'>";
        popup += "<b style='color:#ff5050;'>Past Landslide</b><br>";
        if (props.year) popup += `Year: ${props.year}<br>`;
        if (props.area_geome) popup += `Area: ${props.area_geome.toFixed(4)}<br>`;
        popup += "</div>";
        layer.bindPopup(popup);
      },
    });

    geoLayer.addTo(map);
    landslideLayerRef.current = geoLayer;
  }, [layerToggles.landslides, landslidesGeojson]);

  // Nepal boundary toggle
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (nepalLayerRef.current) {
      nepalLayerRef.current.remove();
      nepalLayerRef.current = null;
    }

    if (!layerToggles.nepal || !nepalGeojson) return;

    const layer = L.geoJSON(nepalGeojson, {
      style: {
        fillOpacity: 0,
        color: "#111111",
        weight: 2,
        dashArray: "5, 5",
      },
    });
    layer.addTo(map);
    nepalLayerRef.current = layer;
  }, [layerToggles.nepal, nepalGeojson]);

  // Sindhupalchowk boundary toggle
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (sindhupalchowkLayerRef.current) {
      sindhupalchowkLayerRef.current.remove();
      sindhupalchowkLayerRef.current = null;
    }

    if (!layerToggles.sindhupalchowk || !sindhupalchowkGeojson) return;

    const layer = L.geoJSON(sindhupalchowkGeojson, {
      style: {
        fillOpacity: 0,
        color: "#3b82f6",
        weight: 3,
      },
    });
    layer.addTo(map);
    sindhupalchowkLayerRef.current = layer;
  }, [layerToggles.sindhupalchowk, sindhupalchowkGeojson]);

  function landsLidesAreValid(gj) {
    return gj && typeof gj === "object" && gj.type && gj.type.toLowerCase() === "featurecollection";
  }

  return <div id="map" ref={mapHostRef} />;
}

