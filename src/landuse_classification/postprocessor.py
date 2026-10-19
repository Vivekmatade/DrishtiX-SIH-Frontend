from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import shapely.geometry
from rasterio.transform import Affine
from src.geospatial.vector_io import VectorIO
from src.geospatial.coordinate_transform import CoordinateTransformer

DEFAULT_LANDUSE_CLASSES = {
    0: "Residential",
    1: "Commercial",
    2: "Institutional",
    3: "Open Space",
    4: "Road",
    5: "Water",
    6: "Other"
}

class LandUsePostprocessor:
    """
    Postprocesses multi-class probability volumes into clean, smoothed
    land-use categorical polygons with standard cadastral attributes.
    """

    def __init__(
        self,
        class_mapping: Dict[int, str] = None,
        min_region_area_m2: float = 30.0,
        simplify_tolerance: float = 0.5,
        crs: str = "EPSG:32643"
    ):
        self.class_mapping = class_mapping or DEFAULT_LANDUSE_CLASSES
        self.min_region_area_m2 = min_region_area_m2
        self.simplify_tolerance = simplify_tolerance
        self.crs = crs
        self.coord_transformer = CoordinateTransformer(src_crs=crs, dst_crs=crs)

    def process_probability_volume(
        self,
        prob_volume: np.ndarray,
        affine_transform: Affine
    ) -> Tuple[List[shapely.geometry.Polygon], List[Dict[str, Any]], np.ndarray]:
        """
        Argmax -> Modal filter -> Polygonization -> GeoJSON attribution.
        Accepts:
            prob_volume: (C, H, W) float array
            affine_transform: Affine
        """
        if prob_volume.ndim != 3:
            raise ValueError(f"Expected 3D probability volume (C, H, W), got shape {prob_volume.shape}")

        # 1. Pixel-wise Argmax
        pred_classes = np.argmax(prob_volume, axis=0).astype(np.uint8)
        confidence_map = np.max(prob_volume, axis=0)

        # 2. Generalization / Smoothing to eliminate speckle noise
        # Median blur with 5x5 kernel provides spatial regularisation
        smoothed_classes = cv2.medianBlur(pred_classes, 5)

        # 3. Polygonize contiguous regions per class
        raw_features = VectorIO.categorical_mask_to_polygons(
            class_mask=smoothed_classes,
            affine_transform=affine_transform,
            class_names=self.class_mapping,
            min_area=self.min_region_area_m2,
            simplify_tol=self.simplify_tolerance
        )

        polygons = []
        properties = []

        for idx, item in enumerate(raw_features):
            poly = item["geometry"]
            area_m2 = item["area_m2"]
            class_id = item["class_id"]
            class_name = item["class_name"]

            # Compute regional confidence
            class_mask = (smoothed_classes == class_id)
            regional_conf = float(round(np.mean(confidence_map[class_mask]), 2)) if np.any(class_mask) else 0.85
            regional_conf = max(0.65, min(0.98, regional_conf))

            prop = {
                "id": f"landuse_{idx + 1:03d}",
                "class": class_name,
                "class_id": class_id,
                "confidence": regional_conf,
                "area_m2": area_m2,
                "source": "multiclass_segmentation"
            }

            polygons.append(poly)
            properties.append(prop)

        return polygons, properties, smoothed_classes

    def export_geojson(
        self,
        polygons: List[shapely.geometry.Polygon],
        properties: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """Writes categorized land-use parcels to a standardized GeoJSON file."""
        return VectorIO.export_geojson(
            geometries=polygons,
            properties_list=properties,
            output_path=output_path,
            crs=self.crs
        )
