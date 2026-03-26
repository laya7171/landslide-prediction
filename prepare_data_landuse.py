"""
prepare_data_landuse.py
========================
Creates new training datasets for landslide susceptibility prediction by:
- Loading the existing raster feature stack (DEM/aspect/slope/etc.)
- Rasterizing the added `sind_land_use.gpkg` polygons onto the same reference grid
- Sampling balanced landslide / non-landslide points
- Extracting features at sampled point locations

Outputs:
- `training_data_landuse.csv` (full feature set)
- `feature_metadata_landuse.json` (metadata for the dataset)

This script is intentionally additive: it does not modify the original
`training_data.csv` pipeline.
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import rowcol
from shapely.geometry import Point, box
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_CRS = "EPSG:32645"  # UTM Zone 45N — matches most rasters

# Existing raster features
RASTER_FILES = {
    "dem": "Dem_sind.tif",
    "aspect": "aspect.tif",
    # Keep both slope variants in the "full" dataset
    "slope": "slope_modified.tif",
    "slope_main": "Slope_main.tif",
    "drainage": "drainage_direction.tif",
    "hillshade": "hillshade2.tif",
    "twi": "twi.tif",
}

# Existing vector layers
VECTOR_FILES = {
    "landslide": "landslide.gpkg",
    "soil": "soil Parent.gpkg",
    "river_line": "river line.gpkg",
}

# Added feature vector layer
LAND_USE_GPKG = "sind_land_use.gpkg"


OUTPUT_CSV_FULL = os.path.join(BASE_DIR, "training_data_landuse.csv")
OUTPUT_META_FULL = os.path.join(BASE_DIR, "feature_metadata_landuse.json")

OUTPUT_CSV_REDUCED = os.path.join(BASE_DIR, "training_data_landuse_reduced.csv")
OUTPUT_META_REDUCED = os.path.join(BASE_DIR, "feature_metadata_landuse_reduced.json")

N_SAMPLES_PER_CLASS_DEFAULT = 2000


def reproject_raster_to_match(src_path: str, ref_ds: rasterio.DatasetReader):
    """Reproject a raster to match the reference dataset's grid."""
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


def compute_distance_to_river(river_gdf: gpd.GeoDataFrame, ref_ds: rasterio.DatasetReader):
    """Rasterize river lines and compute Euclidean distance (metres)."""
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

    dist = distance_transform_edt(river_raster == 0).astype(np.float32)
    pixel_size = abs(ref_ds.res[0])  # metres per pixel
    dist *= pixel_size
    return dist


def rasterize_soil(soil_gdf: gpd.GeoDataFrame, ref_ds: rasterio.DatasetReader):
    """Rasterize soil parent material as integer-coded categories."""
    soil_proj = soil_gdf.to_crs(ref_ds.crs)

    cat_col = None
    for col in soil_proj.columns:
        if soil_proj[col].dtype == object and col != "geometry":
            cat_col = col
            break
    if cat_col is None:
        # Fall back to first column if no obvious string category exists
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


def rasterize_land_use(land_use_gdf: gpd.GeoDataFrame, ref_ds: rasterio.DatasetReader, class_col: str = "class"):
    """Rasterize land-use polygons and encode them into integer categories."""
    land_use_proj = land_use_gdf.to_crs(ref_ds.crs)
    if class_col not in land_use_proj.columns:
        raise ValueError(
            f"Expected column `{class_col}` in sind_land_use.gpkg. Found: {land_use_proj.columns.tolist()}"
        )

    categories = sorted(land_use_proj[class_col].dropna().unique())
    # Integer codes starting at 1 (0 = background / no polygon)
    cat_map = {c: i + 1 for i, c in enumerate(categories)}
    land_use_proj["_code"] = land_use_proj[class_col].map(cat_map).fillna(0).astype(int)

    shapes = [(geom, code) for geom, code in zip(land_use_proj.geometry, land_use_proj["_code"]) if geom is not None]
    land_use_raster = rasterize(
        shapes,
        out_shape=(ref_ds.height, ref_ds.width),
        transform=ref_ds.transform,
        fill=0,
        dtype=np.int16,
    ).astype(np.float32)
    return land_use_raster, cat_map


def sample_points_in_polygons(gdf: gpd.GeoDataFrame, n: int, ref_ds: rasterio.DatasetReader):
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
            try:
                r, c = rowcol(ref_ds.transform, x, y)
                if 0 <= r < ref_ds.height and 0 <= c < ref_ds.width:
                    points.append((x, y))
            except Exception:
                pass
        attempts += 1
    print(f"  Sampled {len(points)} landslide points (attempts: {attempts})")
    return points


def sample_non_landslide_points(landslide_gdf: gpd.GeoDataFrame, n: int, ref_ds: rasterio.DatasetReader):
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


def build_dataset(n_samples_per_class: int, reduced: bool):
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1) Open reference raster (slope_modified)
    # ------------------------------------------------------------------
    ref_path = os.path.join(BASE_DIR, RASTER_FILES["slope"])
    ref_ds = rasterio.open(ref_path)
    print(f"Reference raster: {RASTER_FILES['slope']} ({ref_ds.height}x{ref_ds.width})")

    # ------------------------------------------------------------------
    # 2) Load / reproject rasters into the reference grid
    # ------------------------------------------------------------------
    print("\n--- Loading rasters ---")
    feature_arrays = {}
    for name, fname in RASTER_FILES.items():
        fpath = os.path.join(BASE_DIR, fname)
        print(f"  {name} ({fname})...", end=" ")
        data, nodata = reproject_raster_to_match(fpath, ref_ds)
        data = data.astype(np.float32)
        if nodata is not None:
            data[data == np.float32(nodata)] = np.nan
        data[np.abs(data) > 1e30] = np.nan
        feature_arrays[name] = data
        valid = np.count_nonzero(~np.isnan(data))
        print(f"OK valid={valid}/{data.size}")

    # ------------------------------------------------------------------
    # 3) Derive dist_river + soil_type
    # ------------------------------------------------------------------
    print("\n--- Computing distance to river ---")
    river_gdf = gpd.read_file(os.path.join(BASE_DIR, VECTOR_FILES["river_line"]))
    dist_river = compute_distance_to_river(river_gdf, ref_ds)
    feature_arrays["dist_river"] = dist_river

    print("\n--- Rasterizing soil parent material ---")
    soil_gdf = gpd.read_file(os.path.join(BASE_DIR, VECTOR_FILES["soil"]))
    soil_raster, soil_cat_map = rasterize_soil(soil_gdf, ref_ds)
    feature_arrays["soil_type"] = soil_raster

    # ------------------------------------------------------------------
    # 4) Rasterize land-use polygons (added feature)
    # ------------------------------------------------------------------
    print("\n--- Rasterizing land-use polygons ---")
    land_use_gdf = gpd.read_file(os.path.join(BASE_DIR, LAND_USE_GPKG))
    land_use_raster, land_use_cat_map = rasterize_land_use(land_use_gdf, ref_ds, class_col="class")
    feature_arrays["land_use"] = land_use_raster

    # ------------------------------------------------------------------
    # 5) Sample balanced points
    # ------------------------------------------------------------------
    print("\n--- Sampling balanced training points ---")
    landslide_gdf = gpd.read_file(os.path.join(BASE_DIR, VECTOR_FILES["landslide"]))

    ls_points = sample_points_in_polygons(landslide_gdf, n_samples_per_class, ref_ds)
    nls_points = sample_non_landslide_points(landslide_gdf, n_samples_per_class, ref_ds)

    actual_ls = len(ls_points)
    actual_nls = len(nls_points)

    n_final = min(actual_ls, actual_nls)
    ls_points = ls_points[:n_final]
    nls_points = nls_points[:n_final]
    print(f"Final balanced count: {n_final} per class ({n_final*2} total)")

    # ------------------------------------------------------------------
    # 6) Extract features at sample points
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

    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before-after} rows with NaN values")
        # Re-balance after dropping (mirrors prepare_data.py behavior)
        n_ls = (df["label"] == 1).sum()
        n_nls = (df["label"] == 0).sum()
        n_bal = min(n_ls, n_nls)
        df = pd.concat(
            [
                df[df["label"] == 1].sample(n=n_bal, random_state=42),
                df[df["label"] == 0].sample(n=n_bal, random_state=42),
            ]
        ).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Re-balanced to {n_bal} per class")

    print(f"Final dataset: {len(df)} rows, {len(feature_names)} features")
    print("Label distribution:")
    print(df["label"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 7) Optional reduction (drop redundant slope)
    # ------------------------------------------------------------------
    if reduced:
        # Keep only one slope variant for redundancy testing:
        # drop `slope` (slope_modified.tif) and keep `slope_main`.
        if "slope" in df.columns:
            df = df.drop(columns=["slope"])

    # ------------------------------------------------------------------
    # 8) Save outputs
    # ------------------------------------------------------------------
    if reduced:
        output_csv = OUTPUT_CSV_REDUCED
        output_meta = OUTPUT_META_REDUCED
    else:
        output_csv = OUTPUT_CSV_FULL
        output_meta = OUTPUT_META_FULL

    df.to_csv(output_csv, index=False)

    saved_feature_names = [c for c in df.columns if c not in ("x", "y", "label")]

    meta = {
        "features": saved_feature_names,
        "n_samples": len(df),
        "n_per_class": int(df["label"].value_counts().min()),
        "reference_raster": RASTER_FILES["slope"],
        "target_crs": TARGET_CRS,
        "raster_shape": [ref_ds.height, ref_ds.width],
        "raster_bounds": list(ref_ds.bounds),
        "raster_transform": list(ref_ds.transform),
        "soil_categories": soil_cat_map,
        "land_use_categories": land_use_cat_map,
        "raster_files": RASTER_FILES,
        "land_use_vector_layer": LAND_USE_GPKG,
        "land_use_class_column": "class",
        "reduced": reduced,
    }
    with open(output_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved dataset: {output_csv}")
    print(f"Saved metadata: {output_meta}")
    ref_ds.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_SAMPLES_PER_CLASS_DEFAULT, help="Samples per class (balanced).")
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Drop redundant slope feature(s) for experiments (drops `slope`, keeps `slope_main`).",
    )
    args = parser.parse_args()

    build_dataset(n_samples_per_class=args.n, reduced=args.reduced)


if __name__ == "__main__":
    main()

