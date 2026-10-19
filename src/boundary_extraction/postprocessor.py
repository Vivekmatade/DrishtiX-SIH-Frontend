from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import shapely.geometry
from shapely.validation import make_valid
from rasterio.transform import Affine
from src.geospatial.vector_io import VectorIO
from src.geospatial.coordinate_transform import CoordinateTransformer

DEFAULT_DISCLAIMER = (
    "AI-derived boundary indicators intended for downstream parcel generation. "
    "Not legally valid cadastral boundaries."
)

class BoundaryPostprocessor:
    """
    Vectorizes boundary indicator masks into georeferenced LineStrings,
    filters micro-segments, applies Douglas-Peucker simplification,
    and attaches standard parcel-indicator metadata.
    """

    def __init__(
        self,
        min_line_length_m: float = 5.0,
        simplify_tolerance: float = 0.3,
        crs: str = "EPSG:32643",
        disclaimer: str = DEFAULT_DISCLAIMER
    ):
        self.min_line_length_m = min_line_length_m
        self.simplify_tolerance = simplify_tolerance
        self.crs = crs
        self.disclaimer = disclaimer
        self.coord_transformer = CoordinateTransformer(src_crs=crs, dst_crs=crs)

    def process_boundary_mask(
        self,
        boundary_mask: np.ndarray,
        affine_transform: Affine,
        source_tag: str = "hybrid_boundary_fusion"
    ) -> Tuple[List[shapely.geometry.LineString], List[Dict[str, Any]]]:
        """
        Extracts LineStrings from the binary boundary mask.
        """
        # Morphological skeleton to clean thickness
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_OPEN, kernel)

        # Extract contours
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        properties = []
        counter = 1

        for cnt in contours:
            if len(cnt) < 2:
                continue

            coords = []
            for pt in cnt:
                px, py = pt[0][0], pt[0][1]
                gx, gy = CoordinateTransformer.pixel_to_geo(px, py, affine_transform)
                coords.append((gx, gy))

            line = shapely.geometry.LineString(coords)
            if not line.is_valid:
                line = make_valid(line)

            if line.is_empty:
                continue

            if self.simplify_tolerance > 0:
                line = line.simplify(self.simplify_tolerance, preserve_topology=True)

            length_m = self.coord_transformer.compute_length_m(line)
            if length_m < self.min_line_length_m:
                continue

            # Confidence based on source
            conf = 0.85 if "hybrid" in source_tag else (0.80 if "segmentation" in source_tag else 0.72)

            prop = {
                "id": f"boundary_{counter:03d}",
                "type": "boundary_indicator",
                "source": source_tag,
                "confidence": conf,
                "length_m": length_m,
                "disclaimer": self.disclaimer
            }

            lines.append(line)
            properties.append(prop)
            counter += 1

        return lines, properties

    def export_geojson(
        self,
        geometries: List[shapely.geometry.base.BaseGeometry],
        properties: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """Exports boundary indicators into a standardized GeoJSON file."""
        return VectorIO.export_geojson(
            geometries=geometries,
            properties_list=properties,
            output_path=output_path,
            crs=self.crs
        )
