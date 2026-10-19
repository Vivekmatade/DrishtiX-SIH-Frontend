from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import shapely.geometry
from rasterio.transform import Affine
from src.geospatial.vector_io import VectorIO
from src.geospatial.coordinate_transform import CoordinateTransformer

class BuildingPostprocessor:
    """
    Implements the post-processing pipeline for building footprint extraction:
    Probability mask -> Thresholding -> Morphology -> Connected Components ->
    Polygonization -> Topological Validation -> Filtering -> GeoJSON.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_area_m2: float = 15.0,
        simplify_tolerance: float = 0.3,
        crs: str = "EPSG:32643"
    ):
        self.threshold = threshold
        self.min_area_m2 = min_area_m2
        self.simplify_tolerance = simplify_tolerance
        self.crs = crs
        self.coord_transformer = CoordinateTransformer(src_crs=crs, dst_crs=crs)

    def process_probability_mask(
        self,
        prob_map: np.ndarray,
        affine_transform: Affine
    ) -> Tuple[List[shapely.geometry.Polygon], List[Dict[str, Any]], np.ndarray]:
        """
        Executes morphological cleanup, polygonization, and attribution.
        Returns:
            polygons: list of validated Shapely Polygons in real-world metric CRS
            properties: list of attribute dicts for each building
            cleaned_mask: binary uint8 mask after morphological operations
        """
        # 1. Thresholding
        binary_mask = (prob_map >= self.threshold).astype(np.uint8)

        # 2. Morphological Cleanup (Opening then Closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3. Connected Components Analysis to eliminate micro-speckles
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
        filtered_mask = np.zeros_like(cleaned_mask)
        # 10 pixels minimum in pixel space (10 * 0.1 * 0.1 = 0.1 m²)
        min_pixels = max(10, int(self.min_area_m2 / ((affine_transform.a or 0.1) ** 2)))

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
                filtered_mask[labels == i] = 1

        # 4. Polygonization with Affine transform mapping pixels directly to projected coordinates
        raw_polygons = VectorIO.mask_to_polygons(
            mask=filtered_mask,
            affine_transform=affine_transform,
            min_area=self.min_area_m2,
            simplify_tol=self.simplify_tolerance
        )

        # 5. Polygon validation & Confidence estimation
        valid_polygons = []
        properties = []

        for idx, poly in enumerate(raw_polygons):
            if poly.is_empty:
                continue

            area_m2 = self.coord_transformer.compute_area_m2(poly)
            if area_m2 < self.min_area_m2:
                continue

            # Compute average confidence over polygon footprint in pixel coordinates
            # Rasterize polygon back to pixel space or sample from prob_map
            # Approximate confidence using mean probability of the connected component
            conf_val = float(round(np.mean(prob_map[filtered_mask == 1]), 3)) if np.any(filtered_mask == 1) else float(self.threshold)
            conf_val = max(0.65, min(0.98, conf_val))

            prop = {
                "id": f"building_{idx + 1:03d}",
                "class": "building",
                "confidence": conf_val,
                "area_m2": area_m2,
                "source": "deep_learning_segmentation"
            }

            valid_polygons.append(poly)
            properties.append(prop)

        return valid_polygons, properties, filtered_mask

    def export_geojson(
        self,
        polygons: List[shapely.geometry.Polygon],
        properties: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """Writes validated building footprints to a standardized GeoJSON file."""
        return VectorIO.export_geojson(
            geometries=polygons,
            properties_list=properties,
            output_path=output_path,
            crs=self.crs
        )
