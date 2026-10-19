import os
import sys
import math
import ssl
import json
import urllib.request
import urllib.parse
import numpy as np
import cv2
import pyproj
from rasterio.transform import Affine

# Add workspace to path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.geospatial.raster_io import RasterIO
from src.inference.pipeline import FeatureExtractionPipeline
from src.utils.visualizer import Visualizer

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_and_generate_sit_tumkur(output_dir="data/sit_tumkur"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print("   SIDDAGANGA INSTITUTE OF TECHNOLOGY (SIT), TUMKUR MAP PIPELINE")
    print("=" * 65)

    # 1. Download satellite tiles centered on SIT Tumkur
    ctx = ssl._create_unverified_context()
    z = 18
    center_x = 187233
    center_y = 121278

    x_tiles = [center_x - 1, center_x, center_x + 1, center_x + 2]
    y_tiles = [center_y - 1, center_y, center_y + 1, center_y + 2]

    print(f"[Step 1/5] Fetching high-resolution aerial imagery tiles (Zoom {z})...")
    stitched_bgr = np.zeros((len(y_tiles) * 256, len(x_tiles) * 256, 3), dtype=np.uint8)

    for r, y in enumerate(y_tiles):
        for c, x in enumerate(x_tiles):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                tile_bytes = resp.read()
                arr = np.frombuffer(tile_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                stitched_bgr[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = img

    stitched_rgb = cv2.cvtColor(stitched_bgr, cv2.COLOR_BGR2RGB)
    H, W = stitched_rgb.shape[:2]
    print(f"  [+] Stitched SIT Tumkur image: {W}x{H} pixels")

    # Save raw photo copy
    raw_jpg_path = os.path.join(output_dir, "sit_tumkur_satellite.jpg")
    cv2.imwrite(raw_jpg_path, stitched_bgr)
    print(f"  [+] Saved satellite photo: {raw_jpg_path}")

    # 2. Compute exact geospatial bounds and affine transform in UTM Zone 43N (Tumkur, Karnataka)
    print("\n[Step 2/5] Calculating exact geospatial projection (UTM Zone 43N)...")
    top_lat, left_lon = num2deg(x_tiles[0], y_tiles[0], z)
    bottom_lat, right_lon = num2deg(x_tiles[-1] + 1, y_tiles[-1] + 1, z)

    trans = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    left_utm, top_utm = trans.transform(left_lon, top_lat)
    right_utm, bottom_utm = trans.transform(right_lon, bottom_lat)

    res_x = (right_utm - left_utm) / float(W)
    res_y = (top_utm - bottom_utm) / float(H)

    transform = Affine(res_x, 0.0, left_utm, 0.0, -res_y, top_utm)
    crs = "EPSG:32643"

    ortho_path = os.path.join(output_dir, "orthophoto.tif")
    RasterIO.write_geotiff(
        file_path=ortho_path,
        data=stitched_rgb,
        crs=crs,
        transform=transform
    )
    print(f"  [+] Georeferenced GeoTIFF orthophoto created: {ortho_path}")
    print(f"  [+] Extent: [{left_utm:.1f}, {bottom_utm:.1f}, {right_utm:.1f}, {top_utm:.1f}]")
    print(f"  [+] Spatial Resolution: {res_x:.3f} m/pixel")

    # 3. Run AI Feature Extraction Pipeline
    print("\n[Step 3/5] Executing AI Models on SIT Tumkur Imagery...")
    pipeline = FeatureExtractionPipeline(config_path="config.yaml")
    results = pipeline.run(input_orthophoto=ortho_path, output_dir=output_dir)

    # 4. Generate all features overlay PNG
    overlay_path = os.path.join(output_dir, "all_features_overlay.png")
    sit_overlay_path = os.path.join(output_dir, "sit_tumkur_all_features_overlay.png")

    import shutil
    if os.path.exists(overlay_path):
        shutil.copyfile(overlay_path, sit_overlay_path)

    # Also copy to data/outputs so web UI displays it
    web_out_dir = os.path.join("data", "outputs", "uploaded_run")
    os.makedirs(web_out_dir, exist_ok=True)
    for fname in ["orthophoto.tif", "all_features_overlay.png", "buildings.geojson", "roads.geojson", "landuse.geojson", "boundary_indicators.geojson"]:
        src_f = os.path.join(output_dir, fname)
        if os.path.exists(src_f):
            shutil.copyfile(src_f, os.path.join(web_out_dir, fname))

    # Also save original photo in uploaded_run
    shutil.copyfile(raw_jpg_path, os.path.join(web_out_dir, "original_input.jpg"))

    summary = results["summary"]
    print("\n" + "=" * 65)
    print("      SIDDAGANGA INSTITUTE OF TECHNOLOGY (SIT) EXTRACTION RESULTS")
    print("=" * 65)
    print(f"Location:                 Tumkur (Tumakuru), Karnataka, India")
    print(f"Geographic BBox:          Lat {bottom_lat:.5f}° - {top_lat:.5f}° N, Lon {left_lon:.5f}° - {right_lon:.5f}° E")
    print(f"Projected CRS:            {summary['crs']} (UTM Zone 43N)")
    print(f"Dimensions:               {summary['dimensions']} pixels (~600m x ~586m)")
    print(f"Inference Time:           {summary['processing_time_s']}s")
    print("-" * 65)
    print(f"1. [Buildings] Detected:  {summary['buildings_count']}")
    print(f"   --> {results['buildings']}")
    print(f"2. [Roads] Corridors:     {summary['roads_count']}")
    print(f"   --> {results['roads']}")
    print(f"3. [Land-Use] Zones:      {summary['landuse_regions_count']}")
    print(f"   --> {results['landuse']}")
    print(f"4. [Boundaries] Demarc:   {summary['boundaries_count']}")
    print(f"   --> {results['boundaries']}")
    print("-" * 65)
    print(f"5. [ALL FEATURES PNG]     {sit_overlay_path}")
    print(f"                          (also at {overlay_path})")
    print("=" * 65 + "\n")

    return results

if __name__ == "__main__":
    fetch_and_generate_sit_tumkur()
