import rasterio
import geopandas as gpd
import os
import numpy as np

print("=" * 60)
print("RASTER FILES")
print("=" * 60)
files = sorted([f for f in os.listdir('.') if f.endswith('.tif')])
for f in files:
    try:
        ds = rasterio.open(f)
        data = ds.read(1)
        valid = data[data != ds.nodata] if ds.nodata is not None else data.flatten()
        print(f"\n{f}:")
        print(f"  Shape: ({ds.height}, {ds.width}), Bands: {ds.count}")
        print(f"  CRS: {ds.crs}")
        print(f"  Dtype: {ds.dtypes[0]}, Nodata: {ds.nodata}")
        print(f"  Resolution: {ds.res}")
        print(f"  Bounds: {ds.bounds}")
        print(f"  Valid pixels: {len(valid)}/{data.size}")
        if len(valid) > 0:
            print(f"  Min: {np.nanmin(valid):.4f}, Max: {np.nanmax(valid):.4f}, Mean: {np.nanmean(valid):.4f}")
        ds.close()
    except Exception as e:
        sz = os.path.getsize(f)
        print(f"\n{f}: FAILED (size={sz} bytes)")
        print(f"  Error: {e}")

print("\n" + "=" * 60)
print("VECTOR FILES")
print("=" * 60)

for gpkg in ['landslide.gpkg', 'river line.gpkg', 'river polygon.gpkg', 'soil Parent.gpkg']:
    try:
        gdf = gpd.read_file(gpkg)
        print(f"\n{gpkg}:")
        print(f"  Rows: {len(gdf)}, Columns: {len(gdf.columns)}")
        print(f"  CRS: {gdf.crs}")
        print(f"  Column names: {gdf.columns.tolist()}")
        print(f"  Geom types: {gdf.geometry.type.value_counts().to_dict()}")
        print(f"  Total bounds: {gdf.total_bounds}")
        for c in gdf.columns:
            if c != 'geometry':
                print(f"    {c}: dtype={gdf[c].dtype}, nunique={gdf[c].nunique()}, nulls={gdf[c].isna().sum()}, sample={gdf[c].head(2).tolist()}")
    except Exception as e:
        print(f"\n{gpkg}: FAILED - {e}")
