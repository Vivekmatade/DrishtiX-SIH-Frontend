import os
import glob
from typing import Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.road_extraction.model import RoadSegmentationNet
from src.building_extraction.trainer import DiceBCELoss
from src.preprocessing.normalizer import ImageNormalizer
from src.geospatial.raster_io import RasterIO
from src.utils.logger import get_logger

logger = get_logger("RoadTrainer")

class RoadSegmentationDataset(Dataset):
    """PyTorch Dataset loading drone scenes and binary road masks."""

    def __init__(self, data_dir: str, augment: bool = False):
        self.data_dir = data_dir
        self.augment = augment
        self.normalizer = ImageNormalizer()
        self.samples = []

        scene_dirs = sorted(glob.glob(os.path.join(data_dir, "scene_*")))
        for sdir in scene_dirs:
            ortho = os.path.join(sdir, "orthophoto.tif")
            mask = os.path.join(sdir, "road_mask.png")
            if os.path.exists(ortho) and os.path.exists(mask):
                self.samples.append((ortho, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ortho_path, mask_path = self.samples[idx]
        image, _ = RasterIO.read_orthophoto(ortho_path)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            if np.random.rand() > 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                image = np.rot90(image, rot_k)
                mask = np.rot90(mask, rot_k)

        image_tensor = self.normalizer.to_tensor(image)
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0)

        return image_tensor, mask_tensor


class RoadTrainer:
    """Orchestrates training and validation for the Road Extraction model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config["training"].get("device") == "cuda" else "cpu")
        self.model = RoadSegmentationNet(in_channels=3, num_classes=1).to(self.device)
        self.criterion = DiceBCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["training"].get("learning_rate", 1e-3))
        self.checkpoint_dir = config["models"]["road"]["model_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, train_dir: str, val_dir: str, epochs: Optional[int] = None) -> str:
        """Trains the road model and saves best checkpoint based on validation loss."""
        epochs = epochs or self.config["training"].get("epochs", 8)
        batch_size = self.config["training"].get("batch_size", 4)

        train_dataset = RoadSegmentationDataset(train_dir, augment=True)
        val_dataset = RoadSegmentationDataset(val_dir, augment=False)

        if len(train_dataset) == 0:
            raise ValueError(f"No training samples found in {train_dir}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None

        best_loss = float("inf")
        best_checkpoint = os.path.join(self.checkpoint_dir, "best_road_model.pth")

        logger.info(f"Starting Road Model training on {self.device} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0

            for images, targets in train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_dataset)

            # Validation
            val_loss = train_loss
            if val_loader:
                self.model.eval()
                val_total = 0.0
                with torch.no_grad():
                    for images, targets in val_loader:
                        images = images.to(self.device)
                        targets = targets.to(self.device)
                        logits = self.model(images)
                        loss = self.criterion(logits, targets)
                        val_total += loss.item() * images.size(0)
                val_loss = val_total / len(val_dataset)

            logger.info(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss
                }, best_checkpoint)
                logger.info(f"-> Saved new best road checkpoint: {best_checkpoint} (Val Loss: {val_loss:.4f})")

        return best_checkpoint
