import os
import sys
import argparse

# Ensure workspace root is in sys.path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.geospatial.synthetic_generator import SyntheticDroneGenerator
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("GenerateDummyData")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic drone orthophotos and ground truth labels.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--num_train", type=int, default=6, help="Number of training scenes")
    parser.add_argument("--num_val", type=int, default=2, help="Number of validation scenes")
    parser.add_argument("--num_test", type=int, default=3, help="Number of test scenes")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    geo_cfg = config.get("geospatial", {})

    tile_size = geo_cfg.get("tile_size", 512)
    resolution = geo_cfg.get("pixel_resolution", 0.1)
    crs = geo_cfg.get("default_crs", "EPSG:32643")

    train_dir = data_cfg.get("train_dir", "data/synthetic/train")
    val_dir = data_cfg.get("val_dir", "data/synthetic/val")
    test_dir = data_cfg.get("test_dir", "data/synthetic/test")

    generator = SyntheticDroneGenerator(
        tile_size=tile_size,
        resolution=resolution,
        crs=crs,
        origin_x=700000.0,
        origin_y=3100000.0
    )

    logger.info(f"Generating synthetic drone dataset in Projected CRS: {crs} (Resolution: {resolution}m/px)...")

    # Generate Train Scenes
    logger.info(f"Generating {args.num_train} training scenes into '{train_dir}'...")
    for i in range(args.num_train):
        scene_id = f"scene_{i+1:03d}"
        # Stagger origin coordinates slightly for realistic spatial variation
        generator.origin_x = 700000.0 + (i * 100.0)
        generator.origin_y = 3100000.0 - (i * 100.0)
        meta = generator.generate_scene(scene_id, train_dir, seed=100 + i)
        logger.info(f"  [Train] {scene_id} generated: {meta['buildings_count']} buildings")

    # Generate Val Scenes
    logger.info(f"Generating {args.num_val} validation scenes into '{val_dir}'...")
    for i in range(args.num_val):
        scene_id = f"scene_{i+1:03d}"
        generator.origin_x = 701000.0 + (i * 100.0)
        generator.origin_y = 3101000.0 - (i * 100.0)
        meta = generator.generate_scene(scene_id, val_dir, seed=200 + i)
        logger.info(f"  [Val] {scene_id} generated: {meta['buildings_count']} buildings")

    # Generate Test Scenes
    logger.info(f"Generating {args.num_test} test scenes into '{test_dir}'...")
    for i in range(args.num_test):
        scene_id = f"scene_{i+1:03d}"
        generator.origin_x = 702000.0 + (i * 100.0)
        generator.origin_y = 3102000.0 - (i * 100.0)
        meta = generator.generate_scene(scene_id, test_dir, seed=300 + i)
        logger.info(f"  [Test] {scene_id} generated: {meta['buildings_count']} buildings")

    logger.info("Synthetic drone dataset generation completed successfully!")

if __name__ == "__main__":
    main()
