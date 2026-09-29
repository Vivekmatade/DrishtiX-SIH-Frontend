#!/usr/bin/env python3
"""
End-to-end Demonstration Script for the AI Feature Extraction Pipeline.
Usage:
    python demo.py
"""
import os
import sys
import glob

# Ensure workspace root is in sys.path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.inference.pipeline import FeatureExtractionPipeline
from src.evaluation.evaluator import ModelEvaluator
from src.geospatial.synthetic_generator import SyntheticDroneGenerator
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("Demo")

def main():
    print("=" * 50)
    print("      AI FEATURE EXTRACTION PIPELINE DEMO")
    print("=" * 50)

    config = load_config("config.yaml")
    test_dir = config["data"]["test_dir"]
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Check if synthetic dataset exists; generate if missing
    sample_ortho = os.path.join(test_dir, "scene_001", "orthophoto.tif")
    if not os.path.exists(sample_ortho):
        print("\n[Step 1/4] Synthetic drone dataset not detected. Generating scenes...")
        generator = SyntheticDroneGenerator(
            tile_size=config["geospatial"].get("tile_size", 512),
            resolution=config["geospatial"].get("pixel_resolution", 0.1),
            crs=config["geospatial"].get("default_crs", "EPSG:32643")
        )
        for split, count in [("train", 4), ("val", 2), ("test", 2)]:
            sdir = os.path.join(config["data"]["synthetic_dir"], split)
            for i in range(count):
                generator.generate_scene(f"scene_{i+1:03d}", sdir, seed=100 + i * 10)
        print("[+] Synthetic drone scenes generated successfully.")
    else:
        print("\n[Step 1/4] Synthetic drone dataset verified.")

    # 2. Run Feature Extraction Pipeline on sample scene
    print(f"\n[Step 2/4] Executing AI Feature Extraction on {sample_ortho}...")
    pipeline = FeatureExtractionPipeline(config_path="config.yaml")
    results = pipeline.run(input_orthophoto=sample_ortho, output_dir=output_dir)

    # 3. Run Evaluation across test scenes
    print("\n[Step 3/4] Evaluating models dynamically on test dataset...")
    evaluator = ModelEvaluator(config_path="config.yaml")
    eval_metrics = evaluator.evaluate(
        test_dir=test_dir,
        output_json_path="evaluation_results.json",
        report_md_path="evaluation_report.md",
        vis_output_dir=output_dir
    )

    # 4. Display Concise Summary
    b = eval_metrics["building_extraction"]
    r = eval_metrics["road_extraction"]
    lu = eval_metrics["landuse_classification"]
    bd = eval_metrics["boundary_extraction"]
    summary = results["summary"]

    print("\n" + "=" * 50)
    print("        AI FEATURE EXTRACTION PIPELINE")
    print("=" * 50)
    print(f"Input:\n{os.path.basename(sample_ortho)} (CRS: {summary['crs']}, {summary['dimensions']})\n")
    print(f"Buildings detected:     {summary['buildings_count']}")
    print(f"Road features detected: {summary['roads_count']}")
    print(f"Land-use regions:       {summary['landuse_regions_count']}")
    print(f"Boundary indicators:    {summary['boundaries_count']}\n")
    print("Outputs:")
    print(f"[+] buildings.geojson           ({results['buildings']})")
    print(f"[+] roads.geojson               ({results['roads']})")
    print(f"[+] landuse.geojson             ({results['landuse']})")
    print(f"[+] boundary_indicators.geojson ({results['boundaries']})")
    print(f"[+] all_features_overlay.png    ({os.path.join(output_dir, 'all_features_overlay.png')})\n")
    print("Evaluation (Genuine Test Dataset Metrics):")
    print(f"Building IoU:       {b['iou']:.4f}")
    print(f"Building Precision: {b['precision']:.4f}")
    print(f"Building Recall:    {b['recall']:.4f}")
    print(f"Building F1:        {b['f1']:.4f}")
    print(f"Road IoU:           {r['iou']:.4f}")
    print(f"Road Precision:     {r['precision']:.4f}")
    print(f"Road Recall:        {r['recall']:.4f}")
    print(f"Road F1:            {r['f1']:.4f}")
    print(f"Boundary IoU:       {bd['iou']:.4f}")
    print(f"Boundary F1:        {bd['f1']:.4f}")
    print(f"Land-Use Accuracy:  {lu['overall_accuracy']*100:.2f}%\n")
    print("Pipeline completed successfully.")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
