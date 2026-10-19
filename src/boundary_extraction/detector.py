from typing import Tuple, Optional
import numpy as np
import cv2
import torch
from src.boundary_extraction.model import BoundarySegmentationNet
from src.preprocessing.normalizer import ImageNormalizer

class BoundaryDetector:
    """
    Extracts boundary indicators using a hybrid approach combining:
    1. Deep learning segmentation (BoundarySegmentationNet) when model weights are available
    2. Computer vision edge processing (Canny edge detection + morphological gradient)
    3. Structural boundary edges from roadsides and building perimeters
    """

    def __init__(
        self,
        canny_low: int = 40,
        canny_high: int = 120,
        device: str = "cpu"
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.device = torch.device(device)
        self.normalizer = ImageNormalizer()
        self.model: Optional[BoundarySegmentationNet] = None

    def load_model(self, checkpoint_path: str) -> bool:
        """Loads trained boundary indicator weights if available."""
        try:
            self.model = BoundarySegmentationNet(in_channels=3, num_classes=1).to(self.device)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            return True
        except Exception:
            self.model = None
            return False

    def detect_boundaries(
        self,
        image_rgb: np.ndarray,
        building_mask: Optional[np.ndarray] = None,
        road_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Extracts fused boundary indicator mask.
        Returns:
            fused_mask: (H, W) uint8 binary mask (0 or 255)
            source_tag: string explaining method ('deep_learning_segmentation', 'image_processing_edge_detection', or 'hybrid_boundary_fusion')
        """
        H, W = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Classical Edge Detection (Canny + Sobel morphological gradient)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny_edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Morphological gradient to enhance structural boundary lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        grad_thresh = (gradient > 35).astype(np.uint8) * 255

        classical_edges = cv2.bitwise_or(canny_edges, grad_thresh)

        # 2. Deep Learning Boundary Model (if weights loaded)
        dl_mask = None
        if self.model is not None:
            tensor = self.normalizer.to_tensor(image_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            dl_mask = (probs >= 0.45).astype(np.uint8) * 255

        # 3. Combine signals honestly
        if dl_mask is not None:
            # Fused ML + Edge Detection
            fused_mask = cv2.bitwise_or(dl_mask, classical_edges)
            source_tag = "hybrid_boundary_fusion"
        else:
            fused_mask = classical_edges
            source_tag = "image_processing_edge_detection"

        # Suppress edges within building interiors and road centerlines
        if building_mask is not None and np.any(building_mask > 0):
            b_eroded = cv2.erode(building_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
            fused_mask[b_eroded > 0] = 0

        if road_mask is not None and np.any(road_mask > 0):
            r_eroded = cv2.erode(road_mask.astype(np.uint8), np.ones((7, 7), np.uint8))
            fused_mask[r_eroded > 0] = 0

        return fused_mask, source_tag
