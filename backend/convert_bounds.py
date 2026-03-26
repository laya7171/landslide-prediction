import geopandas as gpd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

nepal_path = os.path.join(BASE_DIR, "Nepal.shp")
sind_path = os.path.join(BASE_DIR, "Sindhupalchowk_boundry.gpkg")

with open(os.path.join(BASE_DIR, "convert_out_utf8.txt"), "w", encoding="utf-8") as f:
    try:
        nepal = gpd.read_file(nepal_path)
        nepal = nepal.to_crs("EPSG:4326")
        nepal.to_file(os.path.join(BASE_DIR, "backend", "nepal.geojson"), driver="GeoJSON")
        f.write("Nepal boundary converted successfully.\n")
    except Exception as e:
        f.write(f"Failed to convert Nepal.shp: {e}\n")

    try:
        sind = gpd.read_file(sind_path)
        sind = sind.to_crs("EPSG:4326")
        sind.to_file(os.path.join(BASE_DIR, "backend", "sindhupalchowk.geojson"), driver="GeoJSON")
        f.write("Sindhupalchowk boundary converted successfully.\n")
    except Exception as e:
        f.write(f"Failed to convert Sindhupalchowk_boundry.gpkg: {e}\n")
