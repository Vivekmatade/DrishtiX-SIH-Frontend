import os
import sys
import argparse
import json

# Ensure workspace root is in sys.path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.inference.pipeline import FeatureExtractionPipeline
from src.utils.logger import get_logger

logger = get_logger("RunInference")

def main():
    parser = argparse.ArgumentParser(description="Run AI Feature Extraction Pipeline on an Orthophoto")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input orthophoto.tif")
    parser.add_argument("--output", "-o", type=str, default="data/outputs", help="Directory to save GeoJSON outputs")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    pipeline = FeatureExtractionPipeline(config_path=args.config)
    results = pipeline.run(input_orthophoto=args.input, output_dir=args.output)

    print("\n" + "="*50)
    print("INFERENCE EXECUTION SUMMARY")
    print("="*50)
    print(f"Input Orthophoto:    {results['summary']['input_orthophoto']}")
    print(f"Projected CRS:       {results['summary']['crs']}")
    print(f"Raster Dimensions:   {results['summary']['dimensions']}")
    print(f"Buildings Detected:  {results['summary']['buildings_count']}")
    print(f"Roads Detected:      {results['summary']['roads_count']}")
    print(f"Land-use Regions:    {results['summary']['landuse_regions_count']}")
    print(f"Boundary Indicators: {results['summary']['boundaries_count']}")
    print(f"Inference Time:      {results['summary']['processing_time_s']} seconds")
    print("\nGenerated GeoJSON Outputs:")
    print(f"  ✓ Buildings: {results['buildings']}")
    print(f"  ✓ Roads:     {results['roads']}")
    print(f"  ✓ Land-Use:  {results['landuse']}")
    print(f"  ✓ Boundary:  {results['boundaries']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
