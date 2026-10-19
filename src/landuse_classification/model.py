import torch
import torch.nn as nn
from src.building_extraction.model import DroneUNet

class LandUseSegmentationNet(nn.Module):
    """
    Multi-class semantic segmentation network for categorizing drone imagery
    into 7 cadastral land-use classes:
    0: Residential, 1: Commercial, 2: Institutional, 3: Open Space, 4: Road, 5: Water, 6: Other
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 7):
        super().__init__()
        self.backbone = DroneUNet(in_channels=in_channels, num_classes=num_classes, base_features=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
