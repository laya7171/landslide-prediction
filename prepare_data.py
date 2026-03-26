"""
prepare_data.py
===============
Data preprocessing pipeline for Landslide Susceptibility Prediction.
- Loads rasters & vector data
- Reprojects everything to a common CRS (EPSG:32645)
- Samples balanced landslide / non-landslide points
- Extracts feature values at each point
- Saves training CSV + metadata
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import rowcol
from shapely.geometry import Point, box
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_CRS = "EPSG:32645"  # UTM Zone 45N — matches most rasters

RASTER_FILES = {
    "dem":        "Dem_sind.tif",
    "aspect":     "aspect.tif",
    "slope":      "slope_modified.tif",
    "slope_main": "Slope_main.tif",
    "drainage":   "drainage_direction.tif",
    "hillshade":  "hillshade2.tif",
    "twi":        "twi.tif",
}

VECTOR_FILES = {
    "landslide":     "landslide.gpkg",
    "soil":          "soil Parent.gpkg",
    "river_line":    "river line.gpkg",
    "river_polygon": "river polygon.gpkg",
    "landuse":       "sind_land_use.gpkg",
}

OUTPUT_CSV = os.path.join(BASE_DIR, "training_data.csv")
OUTPUT_META = os.path.join(BASE_DIR, "feature_metadata.json")

N_SAMPLES_PER_CLASS = 2000  # balanced: 2000 landslide + 2000 non-landslide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reproject_raster_to_match(src_path, ref_ds):
    """Reproject a raster to match the reference dataset's CRS, transform, and shape."""
    with rasterio.open(src_path) as src:
        if src.crs == ref_ds.crs and src.shape == (ref_ds.height, ref_ds.width):
            return src.read(1), src.nodata

        dst_data = np.empty((ref_ds.height, ref_ds.width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            resampling=Resampling.bilinear,
        )
        return dst_data, src.nodata


def compute_distance_to_river(river_gdf, ref_ds):
    """Rasterize river lines and compute Euclidean distance (in pixels, scaled by res)."""
    river_proj = river_gdf.to_crs(ref_ds.crs)
    shapes = [(geom, 1) for geom in river_proj.geometry if geom is not None]
    if not shapes:
        print("  WARNING: No river geometries found.")
        return np.zeros((ref_ds.height, ref_ds.width), dtype=np.float32)
    river_raster = rasterize(
        shapes,
        out_shape=(ref_ds.height, ref_ds.width),
        transform=ref_ds.transform,
        fill=0,
        dtype=np.uint8,
    )
    # Euclidean distance in pixel units, scale by resolution
    dist = distance_transform_edt(river_raster == 0).astype(np.float32)
    pixel_size = abs(ref_ds.res[0])
    dist *= pixel_size  # metres
    return dist


def rasterize_soil(soil_gdf, ref_ds):
    """Rasterize soil parent material as integer-coded categories."""
    soil_proj = soil_gdf.to_crs(ref_ds.crs)
    # Use a text column for categories — find the best one
    cat_col = None
    for col in soil_proj.columns:
        if soil_proj[col].dtype == object and col != "geometry":
            cat_col = col
            break
    if cat_col is None:
        # Fall back to OBJECTID
        cat_col = soil_proj.columns[0]

    categories = sorted(soil_proj[cat_col].dropna().unique())
    cat_map = {c: i + 1 for i, c in enumerate(categories)}
    soil_proj["_code"] = soil_proj[cat_col].map(cat_map).fillna(0).astype(int)

    shapes = [(geom, code) for geom, code in zip(soil_proj.geometry, soil_proj["_code"]) if geom is not None]
    soil_raster = rasterize(
        shapes,
        out_shape=(ref_ds.height, ref_ds.width),
        transform=ref_ds.transform,
        fill=0,
        dtype=np.int16,
    ).astype(np.float32)
    return soil_raster, cat_map


def rasterize_landuse(landuse_gdf, ref_ds):
    """Rasterize landuse as integer-coded categories."""
    landuse_proj = landuse_gdf.to_crs(ref_ds.crs)
    cat_col = None
    for col in landuse_proj.columns:
        if landuse_proj[col].dtype == object and col != "geometry":
            cat_col = col
            break
    if cat_col is None:
        cat_col = landuse_proj.columns[0]

    categories = sorted(landuse_proj[cat_col].dropna().astype(str).unique())
    cat_map = {c: i + 1 for i, c in enumerate(categories)}
    landuse_proj["_code"] = landuse_proj[cat_col].astype(str).map(cat_map).fillna(0).astype(int)

    shapes = [(geom, code) for geom, code in zip(landuse_proj.geometry, landuse_proj["_code"]) if geom is not None]
    lu_raster = rasterize(
        shapes,
        out_shape=(ref_ds.height, ref_ds.width),
        transform=ref_ds.transform,
        fill=0,
        dtype=np.int16,
    ).astype(np.float32)
    return lu_raster, cat_map


def sample_points_in_polygons(gdf, n, ref_ds):
    """Sample n random points inside the polygon geometries, clipped to raster extent."""
    raster_bounds = box(*ref_ds.bounds)
    gdf_proj = gdf.to_crs(ref_ds.crs)
    gdf_clipped = gdf_proj.clip(raster_bounds)
    if gdf_clipped.empty:
        raise ValueError("No landslide polygons overlap the raster extent!")

    merged = gdf_clipped.union_all()
    minx, miny, maxx, maxy = merged.bounds
    points = []
    attempts = 0
    max_attempts = n * 100
    while len(points) < n and attempts < max_attempts:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        p = Point(x, y)
        if merged.contains(p):
            # Check it falls within the raster grid
            try:
                r, c = rowcol(ref_ds.transform, x, y)
                if 0 <= r < ref_ds.height and 0 <= c < ref_ds.width:
                    points.append((x, y))
            except Exception:
                pass
        attempts += 1
    print(f"  Sampled {len(points)} landslide points (attempts: {attempts})")
    return points


def sample_non_landslide_points(landslide_gdf, n, ref_ds):
    """Sample n random points outside all landslide polygons but within raster extent."""
    raster_bounds = box(*ref_ds.bounds)
    ls_proj = landslide_gdf.to_crs(ref_ds.crs)
    ls_clipped = ls_proj.clip(raster_bounds)
    merged = ls_clipped.union_all() if not ls_clipped.empty else None

    minx, miny, maxx, maxy = ref_ds.bounds
    points = []
    attempts = 0
    max_attempts = n * 50
    while len(points) < n and attempts < max_attempts:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        p = Point(x, y)
        try:
            r, c = rowcol(ref_ds.transform, x, y)
        except Exception:
            attempts += 1
            continue
        if 0 <= r < ref_ds.height and 0 <= c < ref_ds.width:
            if merged is None or not merged.contains(p):
                points.append((x, y))
        attempts += 1
    print(f"  Sampled {len(points)} non-landslide points (attempts: {attempts})")
    return points


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    print("=" * 60)
    print("LANDSLIDE SUSCEPTIBILITY — DATA PREPARATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Open reference raster (slope_modified — EPSG:32645)
    # ------------------------------------------------------------------
    ref_path = os.path.join(BASE_DIR, RASTER_FILES["slope"])
    ref_ds = rasterio.open(ref_path)
    print(f"\nReference raster: {RASTER_FILES['slope']}")
    print(f"  Shape: ({ref_ds.height}, {ref_ds.width}), CRS: {ref_ds.crs}")

    # ------------------------------------------------------------------
    # 2. Load / reproject all rasters to match reference
    # ------------------------------------------------------------------
    print("\n--- Loading rasters ---")
    feature_arrays = {}
    for name, fname in RASTER_FILES.items():
        fpath = os.path.join(BASE_DIR, fname)
        print(f"  {name} ({fname})...", end=" ")
        data, nodata = reproject_raster_to_match(fpath, ref_ds)
        # Ensure float32 for NaN masking
        data = data.astype(np.float32)
        # Mask nodata
        if nodata is not None:
            data[data == np.float32(nodata)] = np.nan
        # Also mask extreme float32 sentinels
        data[np.abs(data) > 1e30] = np.nan
        feature_arrays[name] = data
        valid = np.count_nonzero(~np.isnan(data))
        print(f"OK  valid={valid}/{data.size}")

    # ------------------------------------------------------------------
    # 3. Compute distance-to-river
    # ------------------------------------------------------------------
    print("\n--- Computing distance to river ---")
    river_gdf = gpd.read_file(os.path.join(BASE_DIR, VECTOR_FILES["river_line"]))
    dist_river = compute_distance_to_river(river_gdf, ref_ds)
    feature_arrays["dist_river"] = dist_river
    print(f"  Range: {np.nanmin(dist_river):.1f} — {np.nanmax(dist_river):.1f} m")

    # ------------------------------------------------------------------
    # 4. Rasterize soil type
    # ------------------------------------------------------------------
    print("\n--- Rasterizing soil parent material ---")
    soil_gdf = gpd.read_file(os.path.join(BASE_DIR, VECTOR_FILES["soil"]))
    soil_raster, soil_cat_map = rasterize_soil(soil_gdf, ref_ds)
    feature_arrays["soil_type"] = soil_raster
    print(f"  Categories: {soil_cat_map}")

    # ------------------------------------------------------------------
    # 4.5. Rasterize landuse
    # ------------------------------------------------------------------
    print("\n--- Rasterizing landuse ---")
    landuse_gdf = gpd.read_file(os.path.join(BASE_DIR, VECTOR_FILES["landuse"]))
    lu_raster, lu_cat_map = rasterize_landuse(landuse_gdf, ref_ds)
    feature_arrays["landuse"] = lu_raster
    print(f"  Categories: {lu_cat_map}")

    # ------------------------------------------------------------------
    # 5. Sample balanced points
    # ------------------------------------------------------------------
    print("\n--- Sampling balanced training points ---")
    landslide_gdf = gpd.read_file(os.path.join(BASE_DIR, VECTOR_FILES["landslide"]))

    ls_points = sample_points_in_polygons(landslide_gdf, N_SAMPLES_PER_CLASS, ref_ds)
    nls_points = sample_non_landslide_points(landslide_gdf, N_SAMPLES_PER_CLASS, ref_ds)

    actual_ls = len(ls_points)
    actual_nls = len(nls_points)
    # Match the smaller count to ensure balance
    n_final = min(actual_ls, actual_nls)
    ls_points = ls_points[:n_final]
    nls_points = nls_points[:n_final]
    print(f"  Final balanced count: {n_final} per class ({n_final*2} total)")

    # ------------------------------------------------------------------
    # 6. Extract features at sample points
    # ------------------------------------------------------------------
    print("\n--- Extracting features ---")
    feature_names = list(feature_arrays.keys())
    all_points = ls_points + nls_points
    labels = [1] * n_final + [0] * n_final

    rows = []
    for x, y in all_points:
        r, c = rowcol(ref_ds.transform, x, y)
        row = {"x": x, "y": y}
        for fname in feature_names:
            row[fname] = feature_arrays[fname][r, c]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["label"] = labels

    # Drop rows with any NaN features
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with NaN values")
        # Re-balance after dropping
        n_ls = (df["label"] == 1).sum()
        n_nls = (df["label"] == 0).sum()
        n_bal = min(n_ls, n_nls)
        df = pd.concat([
            df[df["label"] == 1].sample(n=n_bal, random_state=42),
            df[df["label"] == 0].sample(n=n_bal, random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  Re-balanced to {n_bal} per class")

    print(f"\n  Final dataset: {len(df)} rows, {len(feature_names)} features")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved training data to: {OUTPUT_CSV}")

    # Feature metadata
    meta = {
        "features": feature_names,
        "n_samples": len(df),
        "n_per_class": int(df["label"].value_counts().min()),
        "reference_raster": RASTER_FILES["slope"],
        "target_crs": TARGET_CRS,
        "raster_shape": [ref_ds.height, ref_ds.width],
        "raster_bounds": list(ref_ds.bounds),
        "raster_transform": list(ref_ds.transform),
        "soil_categories": soil_cat_map,
        "landuse_categories": lu_cat_map,
        "raster_files": RASTER_FILES,
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to: {OUTPUT_META}")

    ref_ds.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
