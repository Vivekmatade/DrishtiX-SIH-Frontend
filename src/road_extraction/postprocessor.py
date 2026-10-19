from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import shapely.geometry
from shapely.validation import make_valid
from rasterio.transform import Affine
from src.geospatial.vector_io import VectorIO
from src.geospatial.coordinate_transform import CoordinateTransformer

class RoadPostprocessor:
    """
    Postprocessor for extracting road networks, centerlines, and access corridors.
    Implements thresholding, morphological reconnection, thinning/skeletonization,
    length measurement in meters, and GeoJSON vector generation.
    """

    def __init__(
        self,
        threshold: float = 0.45,
        min_length_m: float = 10.0,
        simplify_tolerance: float = 0.4,
        crs: str = "EPSG:32643"
    ):
        self.threshold = threshold
        self.min_length_m = min_length_m
        self.simplify_tolerance = simplify_tolerance
        self.crs = crs
        self.coord_transformer = CoordinateTransformer(src_crs=crs, dst_crs=crs)

    def _skeletonize(self, binary_mask: np.ndarray) -> np.ndarray:
        """Morphological thinning to obtain single-pixel width road centerlines."""
        try:
            from skimage.morphology import skeletonize
            skel_bool = skeletonize(binary_mask > 0)
            return (skel_bool.astype(np.uint8)) * 255
        except ImportError:
            skel = np.zeros(binary_mask.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            img = ((binary_mask > 0).astype(np.uint8)) * 255
            for _ in range(128):
                eroded = cv2.erode(img, element, borderType=cv2.BORDER_CONSTANT, borderValue=0)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(img, temp)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded
                if cv2.countNonZero(img) == 0:
                    break
            return skel

    def process_probability_mask(
        self,
        prob_map: np.ndarray,
        affine_transform: Affine
    ) -> Tuple[List[shapely.geometry.base.BaseGeometry], List[Dict[str, Any]], np.ndarray]:
        """
        Extracts road centerlines and corridors.
        Returns:
            geometries: list of Shapely LineStrings (or Polygons) in metric CRS
            properties: list of road attribute dicts
            cleaned_mask: morphological road mask
        """
        # 1. Thresholding
        binary_mask = (prob_map >= self.threshold).astype(np.uint8)

        # 2. Morphological closing to bridge small occlusions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. Skeletonize to extract clean centerlines
        skeleton = self._skeletonize(closed_mask)

        # 4. Extract LineStrings from skeleton using contour tracing
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        geometries = []
        properties = []
        road_counter = 1

        for cnt in contours:
            if len(cnt) < 3:
                continue

            # Convert pixel coordinates to real-world coordinates via Affine transform
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
            if length_m < self.min_length_m:
                continue

            # Classify into Road, Pathway, or Access Corridor
            if length_m >= 35.0:
                road_class = "road"
            elif length_m >= 18.0:
                road_class = "pathway"
            else:
                road_class = "access_corridor"

            conf_val = float(round(np.mean(prob_map[closed_mask == 1]), 3)) if np.any(closed_mask == 1) else float(self.threshold)
            conf_val = max(0.60, min(0.96, conf_val))

            prop = {
                "id": f"road_{road_counter:03d}",
                "class": road_class,
                "confidence": conf_val,
                "length_m": length_m,
                "source": "deep_learning_segmentation"
            }

            geometries.append(line)
            properties.append(prop)
            road_counter += 1

        # Fallback to corridor polygons if lines are too short or sparse
        if len(geometries) == 0:
            polygons = VectorIO.mask_to_polygons(
                mask=closed_mask,
                affine_transform=affine_transform,
                min_area=self.min_length_m * 2.0,
                simplify_tol=self.simplify_tolerance
            )
            for idx, poly in enumerate(polygons):
                length_m = round(poly.length / 2.0, 2)
                geometries.append(poly)
                properties.append({
                    "id": f"road_{idx+1:03d}",
                    "class": "road",
                    "confidence": float(round(self.threshold + 0.35, 2)),
                    "length_m": length_m,
                    "source": "deep_learning_segmentation"
                })

        return geometries, properties, closed_mask

    def export_geojson(
        self,
        geometries: List[shapely.geometry.base.BaseGeometry],
        properties: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """Writes validated road features to a standardized GeoJSON file."""
        return VectorIO.export_geojson(
            geometries=geometries,
            properties_list=properties,
            output_path=output_path,
            crs=self.crs
        )
