import os
import sys
import argparse

# Ensure workspace root is in sys.path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger("EvaluateModels")

def main():
    parser = argparse.ArgumentParser(description="Evaluate AI Feature Extraction Models on Test Dataset")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--test_dir", "-t", type=str, default="data/synthetic/test", help="Test dataset directory")
    parser.add_argument("--output_json", type=str, default="evaluation_results.json", help="Path for JSON metrics")
    parser.add_argument("--report_md", type=str, default="evaluation_report.md", help="Path for Markdown report")
    parser.add_argument("--vis_dir", type=str, default="data/outputs", help="Directory for comparison plots")
    args = parser.parse_args()

    evaluator = ModelEvaluator(config_path=args.config)
    results = evaluator.evaluate(
        test_dir=args.test_dir,
        output_json_path=args.output_json,
        report_md_path=args.report_md,
        vis_output_dir=args.vis_dir
    )

    b = results["building_extraction"]
    r = results["road_extraction"]
    lu = results["landuse_classification"]
    bd = results["boundary_extraction"]

    print("\n" + "="*55)
    print("AI MODEL EVALUATION BENCHMARK RESULTS")
    print("="*55)
    print(f"Dataset:            {results['metadata']['dataset']}")
    print(f"Test Scenes:        {results['metadata']['number_of_scenes']}")
    print(f"Mean Inference:     {results['metadata']['mean_inference_time_seconds']}s")
    print("-" * 55)
    print(f"Building Extraction:  IoU={b['iou']:.4f} | Prec={b['precision']:.4f} | Rec={b['recall']:.4f} | F1={b['f1']:.4f}")
    print(f"Road Extraction:      IoU={r['iou']:.4f} | Prec={r['precision']:.4f} | Rec={r['recall']:.4f} | F1={r['f1']:.4f}")
    print(f"Boundary Indicators:  IoU={bd['iou']:.4f} | Prec={bd['precision']:.4f} | Rec={bd['recall']:.4f} | F1={bd['f1']:.4f}")
    print(f"Land-Use Overall Acc: {lu['overall_accuracy']*100:.2f}% (Mean IoU={lu['mean_iou']:.4f})")
    print("="*55)
    print(f"Detailed JSON results saved to: {args.output_json}")
    print(f"Human-readable report saved to: {args.report_md}")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
