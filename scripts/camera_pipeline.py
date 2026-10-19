#!/usr/bin/env python3
"""
Camera-to-Cadastre AI Feature Extraction Pipeline
Captures an image directly from your camera, converts it to a georeferenced orthophoto (orthophoto.tif),
and runs the full AI pipeline producing all four GeoJSON outputs:
  1. buildings.geojson 🏠
  2. roads.geojson 🛣️
  3. landuse.geojson 🗺️
  4. boundary_indicators.geojson 📐
and generates the integrated cadastral overlay: all_features_overlay.png
"""
import os
import sys
import argparse
import cv2

# Ensure workspace root in sys.path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.geospatial.camera_adapter import CameraAdapter
from src.inference.pipeline import FeatureExtractionPipeline
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("CameraPipeline")

def main():
    parser = argparse.ArgumentParser(
        description="Capture image from camera and extract buildings, roads, land-use, and parcel-boundary indicators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Automatic snapshot from default camera:
    python scripts/camera_pipeline.py

    # Interactive live camera preview (press SPACE to snap photo):
    python scripts/camera_pipeline.py --interactive

    # Use a pre-captured photo or image:
    python scripts/camera_pipeline.py --input-image my_photo.jpg
        """
    )
    parser.add_argument("--device", "-d", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--interactive", action="store_true", help="Show live camera preview window and press SPACE to snap photo")
    parser.add_argument("--input-image", "-i", type=str, default=None, help="Path to an existing image file instead of live camera")
    parser.add_argument("--output", "-o", type=str, default="data/outputs/camera_run", help="Output directory for GeoJSON and overlay files")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resolution", "-r", type=float, default=0.1, help="Ground sampling distance in meters per pixel (default: 0.1m)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = load_config(args.config)
    crs = config.get("geospatial", {}).get("default_crs", "EPSG:32643")

    adapter = CameraAdapter(
        default_crs=crs,
        pixel_resolution=args.resolution,
        origin_x=700000.0,
        origin_y=3100000.0
    )

    print("\n" + "=" * 60)
    print("      CAMERA-TO-CADASTRE AI FEATURE EXTRACTION PIPELINE")
    print("=" * 60)

    # Step 1: Obtain Image (Camera or File)
    if args.input_image:
        if not os.path.exists(args.input_image):
            print(f"[!] Error: Image file not found at '{args.input_image}'")
            sys.exit(1)
        print(f"\n[Step 1/5] Loading image from file: {args.input_image}")
        img_bgr = cv2.imread(args.input_image)
        if img_bgr is None:
            print(f"[!] Error: Could not read image at '{args.input_image}'")
            sys.exit(1)
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        mode_str = "Interactive preview (Press SPACE to capture)" if args.interactive else "Automatic snapshot"
        print(f"\n[Step 1/5] Capturing frame from Camera #{args.device} ({mode_str})...")
        try:
            image_rgb = adapter.capture_frame(
                device_index=args.device,
                interactive=args.interactive,
                warmup_frames=10
            )
            print(f"  [+] Captured camera frame: {image_rgb.shape[1]}x{image_rgb.shape[0]} pixels")
        except Exception as e:
            print(f"[!] Camera capture failed: {e}")
            print("  Falling back to test orthophoto...")
            fallback_path = os.path.join("data", "synthetic", "test", "scene_001", "orthophoto.tif")
            if os.path.exists(fallback_path):
                from src.geospatial.raster_io import RasterIO
                image_rgb, _ = RasterIO.read_orthophoto(fallback_path)
            else:
                sys.exit(1)

    # Step 2: Convert to Standardized GeoTIFF orthophoto.tif
    ortho_out_path = os.path.join(args.output, "orthophoto.tif")
    print(f"\n[Step 2/5] Georeferencing into standard Cloud-Optimized GeoTIFF...")
    meta = adapter.convert_to_orthophoto(
        image_rgb=image_rgb,
        output_path=ortho_out_path,
        resolution=args.resolution,
        crs=crs
    )
    print(f"  [+] Created: {ortho_out_path}")
    print(f"  [+] CRS:     {meta['crs']}")
    print(f"  [+] Extent:  {meta['bounds_utm']}")

    # Step 3: Run AI Feature Extraction Pipeline
    print(f"\n[Step 3/5] Executing AI Feature Extraction Models...")
    pipeline = FeatureExtractionPipeline(config_path=args.config)
    results = pipeline.run(input_orthophoto=ortho_out_path, output_dir=args.output)

    summary = results["summary"]

    # Step 4: Summary of Extracted Layers
    print("\n" + "=" * 60)
    print("                    EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Input:                {summary['input_orthophoto']}")
    print(f"Raster Dimensions:    {summary['dimensions']} pixels")
    print(f"Projected CRS:        {summary['crs']}")
    print(f"Time Elapsed:         {summary['processing_time_s']}s\n")

    print(f"1. [Buildings] Detected:      {summary['buildings_count']}")
    print(f"   Output: {results['buildings']}")

    print(f"2. [Roads] Corridors:          {summary['roads_count']}")
    print(f"   Output: {results['roads']}")

    print(f"3. [Land-Use] Regions:         {summary['landuse_regions_count']}")
    print(f"   Output: {results['landuse']}")

    print(f"4. [Boundary] Indicators:      {summary['boundaries_count']}")
    print(f"   Output: {results['boundaries']}")

    overlay_path = os.path.join(args.output, "all_features_overlay.png")
    print(f"\n5. [Overlay] Cadastral Map:    {overlay_path}")
    print("=" * 60 + "\n")

    return results

if __name__ == "__main__":
    main()
