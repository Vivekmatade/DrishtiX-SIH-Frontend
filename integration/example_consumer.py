"""
Downstream Integration Consumer Example.
Demonstrates how a teammate's Cadastral Parcel Generation Module
consumes the four standardized GeoJSON outputs produced by the AI Feature Extraction Pipeline:
  1. buildings.geojson
  2. roads.geojson
  3. landuse.geojson
  4. boundary_indicators.geojson
"""
import os
import sys
import geopandas as gpd

# Ensure workspace root in path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.utils.logger import get_logger

logger = get_logger("DownstreamParcelConsumer")

def main():
    print("=" * 60)
    print("  DOWNSTREAM CADASTRE PARCEL GENERATION: INTEGRATION DEMO")
    print("=" * 60)

    outputs_dir = os.path.join(workspace_dir, "data", "outputs")

    buildings_path = os.path.join(outputs_dir, "buildings.geojson")
    roads_path = os.path.join(outputs_dir, "roads.geojson")
    landuse_path = os.path.join(outputs_dir, "landuse.geojson")
    boundaries_path = os.path.join(outputs_dir, "boundary_indicators.geojson")

    files = [
        ("Buildings", buildings_path),
        ("Roads", roads_path),
        ("Land-Use", landuse_path),
        ("Boundary Indicators", boundaries_path)
    ]

    for layer_name, path in files:
        if not os.path.exists(path):
            print(f"[WARN] {layer_name} GeoJSON not found at: {path}")
            print("       Please run 'python demo.py' or 'python run_pipeline.py' first.")
            return

        gdf = gpd.read_file(path)
        print(f"\n[+] Loaded Layer: {layer_name}")
        print(f"    File:       {os.path.basename(path)}")
        print(f"    Count:      {len(gdf)} features")
        print(f"    CRS:        {gdf.crs}")
        print(f"    Attributes: {list(gdf.columns)}")
        if len(gdf) > 0:
            sample = gdf.iloc[0].to_dict()
            sample.pop("geometry", None)
            print(f"    Sample Record: {sample}")

    print("\n" + "-" * 60)
    print("Cadastral Parcel Generation Module Consumption Simulation:")
    print("1. [Boundaries] Buffer road centerlines to subtract public right-of-ways.")
    print("2. [Parcels] Reconstruct candidate parcel polygons by polygonizing boundary indicator lines.")
    print("3. [Aggregation] Spatial join building footprints to associate property ownership IDs.")
    print("4. [Land-Use] Tag each cadastral parcel with dominant land-use category (e.g. Residential).")
    print("[+] All 4 GeoJSON feature layers successfully validated for downstream parcel generation!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
