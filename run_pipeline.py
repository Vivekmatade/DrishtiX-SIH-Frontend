#!/usr/bin/env python3
"""
Master CLI script for the AI Feature Extraction Pipeline.
Usage:
    python run_pipeline.py --input path/to/orthophoto.tif --output outputs/
"""
import sys
import os
import argparse

# Ensure current workspace is on sys.path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.inference.pipeline import FeatureExtractionPipeline
from src.utils.logger import get_logger

logger = get_logger("MasterPipeline")

def main():
    parser = argparse.ArgumentParser(
        description="Master CLI for Drone-based AI Feature Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python run_pipeline.py --input data/synthetic/test/scene_001/orthophoto.tif --output data/outputs/
        """
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input orthophoto.tif")
    parser.add_argument("--output", "-o", type=str, default="data/outputs", help="Output directory for GeoJSON files")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file does not exist: {args.input}")
        sys.exit(1)

    input_file = args.input
    ext = os.path.splitext(input_file)[1].lower()
    os.makedirs(args.output, exist_ok=True)

    if ext not in [".tif", ".tiff"]:
        from src.geospatial.camera_adapter import CameraAdapter
        import cv2
        logger.info(f"Converting input image ({ext}) to standardized orthophoto.tif...")
        adapter = CameraAdapter()
        img_bgr = cv2.imread(input_file)
        if img_bgr is None:
            logger.error(f"Could not read image: {input_file}")
            sys.exit(1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ortho_path = os.path.join(args.output, "orthophoto.tif")
        adapter.convert_to_orthophoto(img_rgb, ortho_path)
        input_file = ortho_path

    logger.info(f"Executing Feature Extraction Pipeline...")
    pipeline = FeatureExtractionPipeline(config_path=args.config)
    results = pipeline.run(input_orthophoto=input_file, output_dir=args.output)

    summary = results["summary"]
    print("\n" + "="*50)
    print("           AI FEATURE EXTRACTION PIPELINE")
    print("="*50)
    print(f"Input:               {summary['input_orthophoto']}")
    print(f"Projected CRS:       {summary['crs']}")
    print(f"Dimensions:          {summary['dimensions']}")
    print(f"Buildings detected:  {summary['buildings_count']}")
    print(f"Roads detected:      {summary['roads_count']}")
    print(f"Land-use regions:    {summary['landuse_regions_count']}")
    print(f"Boundary indicators: {summary['boundaries_count']}")
    print(f"Total time elapsed:  {summary['processing_time_s']}s")
    print("\nOutputs Generated:")
    print(f"  [+] {results['buildings']}")
    print(f"  [+] {results['roads']}")
    print(f"  [+] {results['landuse']}")
    print(f"  [+] {results['boundaries']}")
    if "overlay_image" in results:
        print(f"  [+] {results['overlay_image']} (Cadastral Map Overlay)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
