"""
generate_susceptibility_map.py
===============================
Apply trained model across the entire raster extent to produce a
susceptibility probability map (GeoTIFF).
"""

import os
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def reproject_raster_to_match(src_path, ref_ds):
    with rasterio.open(src_path) as src:
        if src.crs == ref_ds.crs and src.shape == (ref_ds.height, ref_ds.width):
            data = src.read(1).astype(np.float32)
            return data, src.nodata
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


def main():
    print("=" * 60)
    print("GENERATING SUSCEPTIBILITY MAP")
    print("=" * 60)

    with open(os.path.join(BASE_DIR, "feature_metadata.json")) as f:
        meta = json.load(f)

    feature_names = meta["features"]
    raster_files = meta["raster_files"]

    # Load reference raster
    ref_path = os.path.join(BASE_DIR, raster_files["slope"])
    ref_ds = rasterio.open(ref_path)
    h, w = ref_ds.height, ref_ds.width
    print(f"Reference: ({h}, {w}), CRS: {ref_ds.crs}")

    # Load model
    model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
    print("Model loaded.")

    # Load all raster features
    print("\n--- Loading rasters ---")
    feature_arrays = {}
    for name, fname in raster_files.items():
        fpath = os.path.join(BASE_DIR, fname)
        print(f"  {name}...", end=" ")
        data, nodata = reproject_raster_to_match(fpath, ref_ds)
        data = data.astype(np.float32)
        if nodata is not None:
            data[data == np.float32(nodata)] = np.nan
        data[np.abs(data) > 1e30] = np.nan
        feature_arrays[name] = data
        print("OK")

    # Distance to river
    print("  dist_river...", end=" ")
    river_gdf = gpd.read_file(os.path.join(BASE_DIR, "river line.gpkg"))
    river_proj = river_gdf.to_crs(ref_ds.crs)
    shapes = [(geom, 1) for geom in river_proj.geometry if geom is not None]
    river_raster = rasterize(shapes, out_shape=(h, w), transform=ref_ds.transform, fill=0, dtype=np.uint8)
    dist_river = distance_transform_edt(river_raster == 0).astype(np.float32)
    dist_river *= abs(ref_ds.res[0])
    feature_arrays["dist_river"] = dist_river
    print("OK")

    # Soil type
    print("  soil_type...", end=" ")
    soil_gdf = gpd.read_file(os.path.join(BASE_DIR, "soil Parent.gpkg"))
    soil_proj = soil_gdf.to_crs(ref_ds.crs)
    cat_col = None
    for col in soil_proj.columns:
        if soil_proj[col].dtype == object and col != "geometry":
            cat_col = col
            break
    if cat_col is None:
        cat_col = soil_proj.columns[0]
    cat_map = meta.get("soil_categories", {})
    if not cat_map:
        categories = sorted(soil_proj[cat_col].dropna().unique())
        cat_map = {c: i + 1 for i, c in enumerate(categories)}
    soil_proj["_code"] = soil_proj[cat_col].map(cat_map).fillna(0).astype(int)
    shapes = [(geom, code) for geom, code in zip(soil_proj.geometry, soil_proj["_code"]) if geom is not None]
    soil_raster = rasterize(shapes, out_shape=(h, w), transform=ref_ds.transform, fill=0, dtype=np.int16).astype(np.float32)
    feature_arrays["soil_type"] = soil_raster
    print("OK")

    # Stack features in correct order
    print("\n--- Building feature stack ---")
    stack = np.stack([feature_arrays[fn] for fn in feature_names], axis=-1)  # (H, W, F)

    # Create valid mask (no NaN in any feature)
    valid_mask = np.all(np.isfinite(stack), axis=-1)  # (H, W)
    n_valid = np.sum(valid_mask)
    print(f"  Valid pixels: {n_valid}/{h*w} ({100*n_valid/(h*w):.1f}%)")

    # Predict in chunks for memory efficiency
    print("\n--- Predicting ---")
    susc_map = np.full((h, w), np.nan, dtype=np.float32)
    valid_coords = np.where(valid_mask)
    X_valid = stack[valid_coords[0], valid_coords[1], :]

    CHUNK = 100000
    for i in range(0, len(X_valid), CHUNK):
        chunk = X_valid[i:i+CHUNK]
        probs = model.predict_proba(chunk)[:, 1]
        susc_map[valid_coords[0][i:i+CHUNK], valid_coords[1][i:i+CHUNK]] = probs
        pct = min(100, (i + CHUNK) / len(X_valid) * 100)
        print(f"  {pct:.0f}% ...", end="\r")

    print(f"\n  Susceptibility range: {np.nanmin(susc_map):.4f} — {np.nanmax(susc_map):.4f}")

    # Save as GeoTIFF
    out_path = os.path.join(BASE_DIR, "susceptibility_map.tif")
    profile = ref_ds.profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999, compress="lzw")
    susc_map_out = np.where(np.isnan(susc_map), -9999, susc_map)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(susc_map_out[np.newaxis, :, :])
    print(f"\nSaved: {out_path}")

    # Also save a classified version (5 zones)
    print("\n--- Classifying into 5 zones ---")
    zones = np.full((h, w), 0, dtype=np.uint8)
    zones[susc_map < 0.2] = 1   # Very Low
    zones[(susc_map >= 0.2) & (susc_map < 0.4)] = 2  # Low
    zones[(susc_map >= 0.4) & (susc_map < 0.6)] = 3  # Moderate
    zones[(susc_map >= 0.6) & (susc_map < 0.8)] = 4  # High
    zones[susc_map >= 0.8] = 5   # Very High
    zones[np.isnan(susc_map)] = 0

    for z, label in [(1,"Very Low"),(2,"Low"),(3,"Moderate"),(4,"High"),(5,"Very High")]:
        cnt = np.sum(zones == z)
        print(f"  Zone {z} ({label}): {cnt} pixels ({100*cnt/n_valid:.1f}%)")

    zones_path = os.path.join(BASE_DIR, "susceptibility_zones.tif")
    profile.update(dtype="uint8", nodata=0)
    with rasterio.open(zones_path, "w", **profile) as dst:
        dst.write(zones[np.newaxis, :, :])
    print(f"Saved: {zones_path}")

    ref_ds.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
