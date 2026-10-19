import os
from typing import Dict, Tuple, Any
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

class RasterIO:
    """Handles georeferenced raster input/output using rasterio."""

    @staticmethod
    def read_orthophoto(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reads a GeoTIFF orthophoto.
        Returns:
            image: uint8 numpy array with shape (H, W, C)
            metadata: dict containing crs, transform, bounds, width, height, count, etc.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Orthophoto file not found: {file_path}")

        try:
            with rasterio.open(file_path) as src:
                # Read bands (bands, H, W)
                bands = src.read()
                if bands.shape[0] >= 3:
                    # Transpose to (H, W, C)
                    image = np.transpose(bands[:3], (1, 2, 0))
                elif bands.shape[0] == 1:
                    # Single band grayscale, replicate to 3 channels
                    image = np.repeat(bands[0][:, :, np.newaxis], 3, axis=-1)
                else:
                    image = np.transpose(bands, (1, 2, 0))

                # Ensure uint8 [0, 255]
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255.0).astype(np.uint8)
                    else:
                        image = np.clip(image, 0, 255).astype(np.uint8)

                crs_str = src.crs.to_string() if src.crs else "EPSG:32643"
                transform = src.transform
                if src.crs is None or transform.is_identity:
                    from rasterio.transform import from_origin
                    transform = from_origin(700000.0, 3100000.0, 0.1, 0.1)
                    crs_str = "EPSG:32643"

                metadata = {
                    "crs": crs_str,
                    "transform": transform,
                    "bounds": src.bounds,
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "driver": src.driver,
                    "nodata": src.nodata
                }
            return image, metadata
        except Exception:
            # Fallback to OpenCV/PIL for non-rasterio standard images
            import cv2
            from rasterio.transform import from_origin
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                raise ValueError(f"Could not read image at {file_path}")
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H, W = image.shape[:2]
            transform = from_origin(700000.0, 3100000.0, 0.1, 0.1)
            metadata = {
                "crs": "EPSG:32643",
                "transform": transform,
                "bounds": (700000.0, 3100000.0 - (H * 0.1), 700000.0 + (W * 0.1), 3100000.0),
                "width": W,
                "height": H,
                "count": 3,
                "driver": "ImageIO",
                "nodata": None
            }
            return image, metadata

    @staticmethod
    def write_geotiff(
        file_path: str,
        data: np.ndarray,
        crs: str,
        transform: Affine,
        nodata: Any = None,
        dtype: str = "uint8"
    ) -> None:
        """
        Writes a numpy array to a georeferenced GeoTIFF file.
        Accepts:
            data: (H, W) or (H, W, C) or (C, H, W)
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        if data.ndim == 2:
            count = 1
            h, w = data.shape
            bands = data[np.newaxis, ...]
        elif data.ndim == 3:
            if data.shape[2] in [1, 3, 4] and data.shape[0] not in [1, 3, 4]:
                # (H, W, C) -> (C, H, W)
                bands = np.transpose(data, (2, 0, 1))
            else:
                bands = data
            count, h, w = bands.shape
        else:
            raise ValueError(f"Unsupported array dimension: {data.ndim}")

        raster_crs = CRS.from_user_input(crs) if crs else CRS.from_epsg(32643)

        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=count,
            dtype=dtype,
            crs=raster_crs,
            transform=transform,
            nodata=nodata,
            compress="deflate"
        ) as dst:
            dst.write(bands.astype(dtype))
