import os
import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from rasterio.transform import from_origin
from src.geospatial.raster_io import RasterIO
from src.utils.logger import get_logger

logger = get_logger("CameraAdapter")

class CameraAdapter:
    """
    Captures live drone/aerial camera frames or webcam images,
    and georeferences them into standardized GeoTIFF orthophotos (RFC compliant)
    suitable for the AI Feature Extraction Pipeline.
    """

    def __init__(
        self,
        default_crs: str = "EPSG:32643",
        pixel_resolution: float = 0.1,  # 10 cm per pixel
        origin_x: float = 700000.0,
        origin_y: float = 3100000.0
    ):
        self.default_crs = default_crs
        self.pixel_resolution = pixel_resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

    def capture_frame(
        self,
        device_index: int = 0,
        interactive: bool = False,
        warmup_frames: int = 10
    ) -> np.ndarray:
        """
        Captures a single high-quality frame from the camera.
        If interactive=True, displays a preview window where user can press SPACE to snap.
        Returns:
            image_rgb: (H, W, 3) uint8 numpy array in RGB format.
        """
        logger.info(f"Connecting to camera device #{device_index}...")
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)

        if not cap.isOpened():
            # Fallback to default backend
            cap = cv2.VideoCapture(device_index)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera device index {device_index}. Verify camera is connected and not in use by another application.")

        # Request high resolution from camera sensor if supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Allow sensor auto-exposure and auto-white-balance to stabilize
        logger.info(f"Stabilizing camera sensor ({warmup_frames} frames)...")
        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)

        captured_frame = None

        if interactive:
            logger.info("Opening camera preview window. Press SPACE to capture, or ESC / Q to cancel.")
            window_name = "Camera Stream - Press SPACE to Snap Orthophoto"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                display_frame = frame.copy()
                # Display HUD overlay
                cv2.putText(
                    display_frame,
                    "AI FEATURE EXTRACTION: PRESS [SPACE] TO CAPTURE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == 32:  # SPACE
                    captured_frame = frame
                    logger.info("Frame captured via user trigger (SPACE).")
                    break
                elif key in [27, ord('q'), ord('Q')]:  # ESC or Q
                    logger.warning("Camera capture canceled by user.")
                    break

            cv2.destroyWindow(window_name)
        else:
            # Automatic snapshot mode
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError("Failed to read image frame from camera.")
            captured_frame = frame
            logger.info("Snapshot frame captured successfully.")

        cap.release()

        if captured_frame is None:
            raise RuntimeError("No frame was captured from camera.")

        # Convert from BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
        return image_rgb

    def convert_to_orthophoto(
        self,
        image_rgb: np.ndarray,
        output_path: str,
        resolution: Optional[float] = None,
        crs: Optional[str] = None,
        origin_x: Optional[float] = None,
        origin_y: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Embeds geospatial metadata (Affine transform, projected CRS) and
        writes the image to a standardized GeoTIFF orthophoto.
        """
        resolution = resolution or self.pixel_resolution
        crs = crs or self.default_crs
        origin_x = origin_x or self.origin_x
        origin_y = origin_y or self.origin_y

        H, W = image_rgb.shape[:2]

        transform = from_origin(
            west=origin_x,
            north=origin_y,
            xsize=resolution,
            ysize=resolution
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        RasterIO.write_geotiff(
            file_path=output_path,
            data=image_rgb,
            crs=crs,
            transform=transform,
            dtype="uint8"
        )

        bounds = (
            origin_x,
            origin_y - (H * resolution),
            origin_x + (W * resolution),
            origin_y
        )

        meta = {
            "orthophoto_path": output_path,
            "crs": crs,
            "width": W,
            "height": H,
            "resolution_m": resolution,
            "bounds_utm": bounds,
            "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]
        }

        logger.info(f"Georeferenced orthophoto created at: {output_path} ({W}x{H}, CRS: {crs})")
        return meta
