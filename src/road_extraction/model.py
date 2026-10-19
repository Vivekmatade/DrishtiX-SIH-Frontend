import torch
import torch.nn as nn
from src.building_extraction.model import DroneUNet

class RoadSegmentationNet(nn.Module):
    """
    Dedicated road network extraction model designed for continuous linear corridors.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.backbone = DroneUNet(in_channels=in_channels, num_classes=num_classes, base_features=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
