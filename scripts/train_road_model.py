import os
import sys
import argparse

# Ensure workspace root is in sys.path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.road_extraction.trainer import RoadTrainer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("TrainRoadModel")

def main():
    parser = argparse.ArgumentParser(description="Train Road & Corridor Extraction Model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = RoadTrainer(config)

    train_dir = config["data"]["train_dir"]
    val_dir = config["data"]["val_dir"]

    logger.info("Initializing Road Extraction Model Training...")
    best_path = trainer.train(train_dir, val_dir, epochs=args.epochs)
    logger.info(f"Road Model training finished. Best checkpoint: {best_path}")

if __name__ == "__main__":
    main()
