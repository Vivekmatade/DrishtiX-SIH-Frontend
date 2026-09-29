#!/usr/bin/env python3
"""
Upload / Input Image to Cadastral Map Generator
Takes any input image (JPG, PNG, TIFF, or drone orthophoto),
extracts all 4 cadastral layers (buildings, roads, land-use, boundary indicators),
and outputs the integrated cadastral overlay image (all_features_overlay.png) and GeoJSONs.

Usage:
    python upload_and_generate.py path/to/image.jpg
    python upload_and_generate.py path/to/drone_aerial.png --output data/outputs/my_run
    python upload_and_generate.py data/sample_drone_aerial.jpg --open
"""

import sys
import os
import argparse
import subprocess

# Ensure workspace is on sys.path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.geospatial.raster_io import RasterIO
from src.geospatial.camera_adapter import CameraAdapter
from src.inference.pipeline import FeatureExtractionPipeline
from src.utils.logger import get_logger

logger = get_logger("UploadAndGenerate")

def process_uploaded_image(
    image_path: str,
    output_dir: str = "data/outputs/uploaded_run",
    config_path: str = "config.yaml",
    open_overlay: bool = False
):
    if not os.path.exists(image_path):
        print(f"[!] Error: File not found: {image_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 62)
    print("      IMAGE TO CADASTRAL MAP GENERATOR (AI PIPELINE)")
    print("=" * 62)
    print(f"Input image:        {image_path}")
    print(f"Output directory:   {output_dir}")

    # Check image format
    ext = os.path.splitext(image_path)[1].lower()
    ortho_path = os.path.join(output_dir, "orthophoto.tif")

    if ext in [".tif", ".tiff"]:
        # If it's already a TIFF, check if it's georeferenced
        try:
            img, meta = RasterIO.read_orthophoto(image_path)
            # Write standardized copy to output directory
            RasterIO.write_geotiff(
                file_path=ortho_path,
                data=img,
                crs=meta["crs"],
                transform=meta["transform"]
            )
            print(f"[+] Loaded existing GeoTIFF ({img.shape[1]}x{img.shape[0]}, CRS: {meta['crs']})")
        except Exception as e:
            logger.warning(f"GeoTIFF read warning: {e}. Converting with CameraAdapter...")
            adapter = CameraAdapter()
            import cv2
            img_bgr = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            adapter.convert_to_orthophoto(img_rgb, ortho_path)
    else:
        # Standard image (JPG, PNG, etc.): georeference it into standard orthophoto.tif
        print(f"[+] Georeferencing standard image ({ext}) into metric orthophoto.tif...")
        adapter = CameraAdapter(
            default_crs="EPSG:32643",
            pixel_resolution=0.1,  # 10 cm ground sampling distance
            origin_x=700000.0,
            origin_y=3100000.0
        )
        import cv2
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"[!] Error: Could not decode image {image_path}")
            sys.exit(1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        meta = adapter.convert_to_orthophoto(img_rgb, ortho_path)
        print(f"  [+] Orthophoto created: {ortho_path}")
        print(f"  [+] Projected CRS:      {meta['crs']}")
        print(f"  [+] Extent (UTM):       {meta['bounds_utm']}")

    # Run AI feature extraction pipeline
    print("\n[+] Running AI Deep Learning Feature Extraction Models...")
    pipeline = FeatureExtractionPipeline(config_path=config_path)
    results = pipeline.run(input_orthophoto=ortho_path, output_dir=output_dir)

    summary = results["summary"]
    overlay_img = results.get("overlay_image", os.path.join(output_dir, "all_features_overlay.png"))

    print("\n" + "=" * 62)
    print("               CADASTRAL GENERATION RESULTS")
    print("=" * 62)
    print(f"Input dimensions:     {summary['dimensions']}")
    print(f"Projected CRS:        {summary['crs']}")
    print(f"Inference time:       {summary['processing_time_s']}s")
    print("-" * 62)
    print(f"1. [Buildings]        Detected: {summary['buildings_count']}")
    print(f"   --> {results['buildings']}")
    print(f"2. [Roads]            Corridors: {summary['roads_count']}")
    print(f"   --> {results['roads']}")
    print(f"3. [Land-Use]         Zones: {summary['landuse_regions_count']}")
    print(f"   --> {results['landuse']}")
    print(f"4. [Boundaries]       Indicators: {summary['boundaries_count']}")
    print(f"   --> {results['boundaries']}")
    print(f"5. [Cadastral Image]  Generated Overlay:")
    print(f"   --> {overlay_img}")
    print("-" * 62)
    print(f"6. [Photogrammetry]   Digital Elevation Models:")
    print(f"   [DEM] Earth Surface Elevation Grid:  {results.get('dem_png')}")
    print(f"   [DSM] Top Reflective Surface Model:  {results.get('dsm_png')}")
    print(f"   [DTM] Bare-Earth Terrain & Contours: {results.get('dtm_png')}")
    print(f"   [Contours Vector]:                   {results.get('contours')}")
    print("=" * 62 + "\n")

    if open_overlay and os.path.exists(overlay_img):
        try:
            if os.name == "nt":
                os.startfile(overlay_img)
            else:
                subprocess.run(["xdg-open", overlay_img])
            print(f"[+] Opened cadastral image in default viewer.")
        except Exception as e:
            print(f"[*] Could not auto-open viewer: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Upload/provide an aerial image and generate cadastral map overlay & GeoJSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python upload_and_generate.py data/sample_drone_aerial.jpg
    python upload_and_generate.py data/sample_drone_aerial_2.png --open
    python upload_and_generate.py data/synthetic/test/scene_001/orthophoto.tif
        """
    )
    parser.add_argument("image", type=str, help="Path to input image file (JPG, PNG, or TIF)")
    parser.add_argument("--output", "-o", type=str, default="data/outputs/uploaded_run", help="Output directory")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--open", action="store_true", help="Automatically open the generated cadastral overlay image")
    args = parser.parse_args()

    process_uploaded_image(
        image_path=args.image,
        output_dir=args.output,
        config_path=args.config,
        open_overlay=args.open
    )

if __name__ == "__main__":
    main()
