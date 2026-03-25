"""
FastAPI Backend for Landslide Susceptibility Prediction
=======================================================
Serves the prediction model, susceptibility map tiles, and
geospatial data to the frontend.
"""

import os
import json
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.transform import rowcol
from pyproj import Transformer
import geopandas as gpd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from io import BytesIO
import base64

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
META_PATH = os.path.join(BASE_DIR, "feature_metadata.json")
SUSC_MAP_PATH = os.path.join(BASE_DIR, "susceptibility_map.tif")
SUSC_ZONES_PATH = os.path.join(BASE_DIR, "susceptibility_zones.tif")
LANDSLIDE_PATH = os.path.join(BASE_DIR, "landslide.gpkg")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ---------------------------------------------------------------------------
# Load resources at startup
# ---------------------------------------------------------------------------
print("Loading model and data...")
model = joblib.load(MODEL_PATH)

with open(META_PATH) as f:
    feature_meta = json.load(f)

with open(METRICS_PATH) as f:
    model_metrics = json.load(f)

feature_names = feature_meta["features"]
raster_files = feature_meta["raster_files"]

# Load susceptibility map raster
susc_ds = rasterio.open(SUSC_MAP_PATH)
susc_data = susc_ds.read(1)
susc_nodata = susc_ds.nodata

# Load all feature rasters for point-level prediction
print("Loading feature rasters...")
feature_datasets = {}
feature_data_arrays = {}

# We need to load the reference raster (slope) and reproject others
ref_path = os.path.join(BASE_DIR, raster_files["slope"])
ref_ds = rasterio.open(ref_path)

for name, fname in raster_files.items():
    fpath = os.path.join(BASE_DIR, fname)
    ds = rasterio.open(fpath)
    # Read raw data
    if ds.crs == ref_ds.crs and ds.shape == (ref_ds.height, ref_ds.width):
        data = ds.read(1).astype(np.float32)
    else:
        from rasterio.warp import reproject, Resampling
        data = np.empty((ref_ds.height, ref_ds.width), dtype=np.float32)
        reproject(
            source=rasterio.band(ds, 1),
            destination=data,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            resampling=Resampling.bilinear,
        )
    nodata = ds.nodata
    if nodata is not None:
        data[data == np.float32(nodata)] = np.nan
    data[np.abs(data) > 1e30] = np.nan
    feature_data_arrays[name] = data
    ds.close()

# Load precomputed derived features (dist_river, soil_type) from the full map
# We'll recompute them at startup
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

print("Computing distance to river...")
river_gdf = gpd.read_file(os.path.join(BASE_DIR, "river line.gpkg"))
river_proj = river_gdf.to_crs(ref_ds.crs)
shapes = [(geom, 1) for geom in river_proj.geometry if geom is not None]
river_raster = rasterize(shapes, out_shape=(ref_ds.height, ref_ds.width),
                         transform=ref_ds.transform, fill=0, dtype=np.uint8)
dist_river = distance_transform_edt(river_raster == 0).astype(np.float32)
dist_river *= abs(ref_ds.res[0])
feature_data_arrays["dist_river"] = dist_river

print("Rasterizing soil data...")
soil_gdf = gpd.read_file(os.path.join(BASE_DIR, "soil Parent.gpkg"))
soil_proj = soil_gdf.to_crs(ref_ds.crs)
cat_col = None
for col in soil_proj.columns:
    if soil_proj[col].dtype == object and col != "geometry":
        cat_col = col
        break
if cat_col is None:
    cat_col = soil_proj.columns[0]
cat_map = feature_meta.get("soil_categories", {})
soil_proj["_code"] = soil_proj[cat_col].map(cat_map).fillna(0).astype(int)
shapes = [(geom, code) for geom, code in zip(soil_proj.geometry, soil_proj["_code"]) if geom is not None]
soil_raster = rasterize(shapes, out_shape=(ref_ds.height, ref_ds.width),
                        transform=ref_ds.transform, fill=0, dtype=np.int16).astype(np.float32)
feature_data_arrays["soil_type"] = soil_raster

# Load landslide GeoJSON
print("Loading landslide data...")
landslide_gdf = gpd.read_file(LANDSLIDE_PATH)
landslide_geojson = json.loads(landslide_gdf.to_json())

# CRS transformer: WGS84 (EPSG:4326) <-> UTM (EPSG:32645)
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)

# Raster bounds in WGS84
transformer_to_wgs = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
bounds_utm = ref_ds.bounds
x_min_wgs, y_min_wgs = transformer_to_wgs.transform(bounds_utm.left, bounds_utm.bottom)
x_max_wgs, y_max_wgs = transformer_to_wgs.transform(bounds_utm.right, bounds_utm.top)

print("Startup complete!")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Landslide Susceptibility Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    lat: float
    lng: float


class PredictResponse(BaseModel):
    lat: float
    lng: float
    susceptibility: float
    zone: str
    zone_level: int
    features: dict


def _get_pixel(lat: float, lng: float):
    """Convert lat/lng → UTM → pixel row/col in reference raster."""
    x_utm, y_utm = transformer_to_utm.transform(lng, lat)
    try:
        row, col = rowcol(ref_ds.transform, x_utm, y_utm)
    except Exception:
        raise HTTPException(status_code=400, detail="Coordinates outside raster extent")
    if row < 0 or row >= ref_ds.height or col < 0 or col >= ref_ds.width:
        raise HTTPException(status_code=400, detail="Coordinates outside raster extent")
    return int(row), int(col)


def _classify_zone(prob: float):
    if prob < 0.2:
        return "Very Low", 1
    elif prob < 0.4:
        return "Low", 2
    elif prob < 0.6:
        return "Moderate", 3
    elif prob < 0.8:
        return "High", 4
    else:
        return "Very High", 5


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Landslide Susceptibility Prediction API", "status": "running"}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict landslide susceptibility at given lat/lng."""
    row, col = _get_pixel(req.lat, req.lng)

    # Extract features
    feature_values = {}
    X = []
    for fname in feature_names:
        val = float(feature_data_arrays[fname][row, col])
        feature_values[fname] = val
        X.append(val)

    if any(np.isnan(v) for v in X):
        raise HTTPException(status_code=400, detail="No data at this location (ocean or outside mapped area)")

    X_arr = np.array([X])
    prob = float(model.predict_proba(X_arr)[0, 1])
    zone_name, zone_level = _classify_zone(prob)

    return PredictResponse(
        lat=req.lat,
        lng=req.lng,
        susceptibility=round(prob, 4),
        zone=zone_name,
        zone_level=zone_level,
        features=feature_values,
    )


@app.get("/api/features/{lat}/{lng}")
def get_features(lat: float, lng: float):
    """Get raw feature values at a point."""
    row, col = _get_pixel(lat, lng)
    features = {}
    for fname in feature_names:
        val = float(feature_data_arrays[fname][row, col])
        features[fname] = round(val, 4) if not np.isnan(val) else None
    return {"lat": lat, "lng": lng, "features": features}


@app.get("/api/stats")
def get_stats():
    """Return model performance metrics and feature importance."""
    return {
        "metrics": model_metrics,
        "features": feature_names,
        "n_samples": feature_meta["n_samples"],
        "raster_bounds_wgs84": {
            "south": round(y_min_wgs, 6),
            "north": round(y_max_wgs, 6),
            "west": round(x_min_wgs, 6),
            "east": round(x_max_wgs, 6),
        },
    }


@app.get("/api/landslides")
def get_landslides():
    """Return past landslide locations as GeoJSON."""
    return JSONResponse(content=landslide_geojson)


@app.get("/api/susceptibility-tile/{z}/{x}/{y}.png")
def get_tile(z: int, x: int, y: int):
    """
    Serve susceptibility map as PNG overlay tiles.
    Uses a simple on-the-fly tile renderer.
    """
    import math
    from PIL import Image

    tile_size = 256

    # Tile bounds in WGS84
    n = 2 ** z
    lng_min = x / n * 360.0 - 180.0
    lng_max = (x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))

    # Create tile image
    img = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    pixels = img.load()

    # Color map: Very Low=green, Low=yellow-green, Moderate=yellow, High=orange, Very High=red
    def prob_to_color(p):
        if p < 0.2:
            return (34, 139, 34, 120)      # green
        elif p < 0.4:
            return (154, 205, 50, 140)     # yellow-green
        elif p < 0.6:
            return (255, 215, 0, 160)      # yellow/gold
        elif p < 0.8:
            return (255, 140, 0, 180)      # orange
        else:
            return (220, 20, 60, 200)      # crimson red

    for py in range(tile_size):
        lat = lat_max - (lat_max - lat_min) * py / tile_size
        for px in range(tile_size):
            lng = lng_min + (lng_max - lng_min) * px / tile_size
            try:
                x_utm, y_utm = transformer_to_utm.transform(lng, lat)
                row, col = rowcol(ref_ds.transform, x_utm, y_utm)
                if 0 <= row < ref_ds.height and 0 <= col < ref_ds.width:
                    val = float(susc_data[row, col])
                    if val != susc_nodata and not np.isnan(val) and val >= 0:
                        pixels[px, py] = prob_to_color(val)
            except Exception:
                pass

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return FileResponse(
        path=None,
        media_type="image/png",
        content=buf.read(),
    )


@app.get("/api/susceptibility-image")
def get_susceptibility_image():
    """
    Serve the entire susceptibility map as a single PNG image
    with bounds info for Leaflet overlay.
    """
    from PIL import Image

    h, w = susc_data.shape
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pixels = img.load()

    def prob_to_color(p):
        if p < 0.2:
            return (34, 139, 34, 100)
        elif p < 0.4:
            return (154, 205, 50, 120)
        elif p < 0.6:
            return (255, 215, 0, 140)
        elif p < 0.8:
            return (255, 140, 0, 160)
        else:
            return (220, 20, 60, 180)

    for r in range(h):
        for c in range(w):
            val = float(susc_data[r, c])
            if val != susc_nodata and not np.isnan(val) and val >= 0:
                pixels[c, r] = prob_to_color(val)

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "image": f"data:image/png;base64,{img_b64}",
        "bounds": {
            "south": round(y_min_wgs, 6),
            "north": round(y_max_wgs, 6),
            "west": round(x_min_wgs, 6),
            "east": round(x_max_wgs, 6),
        },
    }


# Mount frontend static files
if os.path.exists(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
