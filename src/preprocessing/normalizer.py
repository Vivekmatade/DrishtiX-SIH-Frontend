import numpy as np
import torch

class ImageNormalizer:
    """Normalizes aerial/drone imagery for AI model ingestion."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def normalize_numpy(self, img: np.ndarray) -> np.ndarray:
        """
        Normalizes an RGB image array (H, W, C) with values [0, 255]
        to standardized zero-mean, unit-variance float32.
        """
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        if img.max() > 1.0:
            img = img / 255.0

        if img.shape[-1] > 3:
            img = img[:, :, :3]  # Take RGB if NIR is present
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        norm_img = (img - self.mean) / self.std
        return norm_img.astype(np.float32)

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Converts (H, W, C) normalized numpy array to (C, H, W) torch Tensor."""
        norm_img = self.normalize_numpy(img)
        tensor = torch.from_numpy(norm_img).permute(2, 0, 1)
        return tensor

    def denormalize(self, norm_img: np.ndarray) -> np.ndarray:
        """Denormalizes back to standard [0, 255] uint8 RGB."""
        img = (norm_img * self.std) + self.mean
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img
