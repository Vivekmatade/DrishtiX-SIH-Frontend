import numpy as np
from typing import List, Tuple

class OrthoTiler:
    """
    Slices large orthophoto rasters into overlapping patches for inference,
    and merges prediction tiles back into full-resolution seamless probability maps.
    """

    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def generate_tiles(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Extracts overlapping patches from an image (H, W, C).
        Returns list of tile arrays and list of bounding coordinates (y1, y2, x1, x2).
        """
        h, w = image.shape[:2]

        # If image is smaller than or equal to tile size, pad if needed or return single tile
        if h <= self.tile_size and w <= self.tile_size:
            pad_h = self.tile_size - h
            pad_w = self.tile_size - w
            if pad_h > 0 or pad_w > 0:
                padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                return [padded], [(0, h, 0, w)]
            return [image], [(0, h, 0, w)]

        tiles = []
        coords = []

        y_points = list(range(0, h - self.tile_size + 1, self.stride))
        if len(y_points) == 0 or y_points[-1] + self.tile_size < h:
            y_points.append(max(0, h - self.tile_size))

        x_points = list(range(0, w - self.tile_size + 1, self.stride))
        if len(x_points) == 0 or x_points[-1] + self.tile_size < w:
            x_points.append(max(0, w - self.tile_size))

        for y in y_points:
            for x in x_points:
                tile = image[y:y + self.tile_size, x:x + self.tile_size]
                tiles.append(tile)
                coords.append((y, y + self.tile_size, x, x + self.tile_size))

        return tiles, coords

    def merge_probability_maps(
        self,
        tile_preds: List[np.ndarray],
        coords: List[Tuple[int, int, int, int]],
        output_shape: Tuple[int, int],
        num_classes: int = 1
    ) -> np.ndarray:
        """
        Blends overlapping prediction tiles using distance-weighted averaging
        to prevent seam artifacts at patch borders.
        """
        h, w = output_shape
        if num_classes == 1:
            full_map = np.zeros((h, w), dtype=np.float32)
            weight_map = np.zeros((h, w), dtype=np.float32)
        else:
            full_map = np.zeros((num_classes, h, w), dtype=np.float32)
            weight_map = np.zeros((h, w), dtype=np.float32)

        # Create 2D linear ramp weight mask for blending
        ramp = np.ones((self.tile_size, self.tile_size), dtype=np.float32)
        edge = max(1, self.overlap // 2)
        for i in range(edge):
            weight = (i + 1) / edge
            ramp[i, :] *= weight
            ramp[-1 - i, :] *= weight
            ramp[:, i] *= weight
            ramp[:, -1 - i] *= weight

        for pred, (y1, y2, x1, x2) in zip(tile_preds, coords):
            actual_h = y2 - y1
            actual_w = x2 - x1
            current_pred = pred[..., :actual_h, :actual_w] if pred.ndim >= 2 else pred
            current_weight = ramp[:actual_h, :actual_w]

            if num_classes == 1:
                # Ensure 2D
                if current_pred.ndim == 3:
                    current_pred = current_pred[0]
                full_map[y1:y2, x1:x2] += current_pred * current_weight
                weight_map[y1:y2, x1:x2] += current_weight
            else:
                for c in range(num_classes):
                    full_map[c, y1:y2, x1:x2] += current_pred[c] * current_weight
                weight_map[y1:y2, x1:x2] += current_weight

        # Avoid division by zero
        weight_map = np.maximum(weight_map, 1e-6)
        if num_classes == 1:
            return full_map / weight_map
        else:
            return full_map / weight_map[np.newaxis, ...]
