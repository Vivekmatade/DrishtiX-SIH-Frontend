import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import shapely.geometry
from shapely.validation import make_valid
import rasterio.features
from rasterio.transform import Affine
import geopandas as gpd

class VectorIO:
    """Converts pixel masks to georeferenced Shapely geometries and standard GeoJSON outputs."""

    @staticmethod
    def mask_to_polygons(
        mask: np.ndarray,
        affine_transform: Affine,
        min_area: float = 0.0,
        simplify_tol: float = 0.0
    ) -> List[shapely.geometry.Polygon]:
        """
        Converts a binary mask (uint8, 0 or 1/255) into real-world georeferenced polygons.
        Uses rasterio.features.shapes with affine transform applied directly to coordinates.
        """
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Ensure binary 0 or 1
        binary_mask = (mask > 0).astype(np.uint8)

        polygons = []
        # rasterio.features.shapes yields (geom_dict, value)
        for geom_dict, val in rasterio.features.shapes(binary_mask, mask=binary_mask, transform=affine_transform):
            if val == 1:
                poly = shapely.geometry.shape(geom_dict)
                # Topological repair
                if not poly.is_valid:
                    poly = make_valid(poly)

                if poly.is_empty:
                    continue

                if simplify_tol > 0:
                    poly = poly.simplify(simplify_tol, preserve_topology=True)

                if isinstance(poly, shapely.geometry.Polygon):
                    if poly.area >= min_area:
                        polygons.append(poly)
                elif isinstance(poly, shapely.geometry.MultiPolygon):
                    for sub_poly in poly.geoms:
                        if sub_poly.area >= min_area:
                            polygons.append(sub_poly)

        return polygons

    @staticmethod
    def categorical_mask_to_polygons(
        class_mask: np.ndarray,
        affine_transform: Affine,
        class_names: Dict[int, str],
        min_area: float = 0.0,
        simplify_tol: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Converts a multi-class segmentation mask into labeled real-world polygons.
        Returns list of dicts with {'geometry': poly, 'class_id': int, 'class_name': str, 'area_m2': float}.
        """
        features = []
        mask_uint8 = class_mask.astype(np.uint8)

        for geom_dict, class_id in rasterio.features.shapes(mask_uint8, transform=affine_transform):
            class_id = int(class_id)
            if class_id not in class_names:
                continue

            poly = shapely.geometry.shape(geom_dict)
            if not poly.is_valid:
                poly = make_valid(poly)

            if poly.is_empty:
                continue

            if simplify_tol > 0:
                poly = poly.simplify(simplify_tol, preserve_topology=True)

            if isinstance(poly, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
                area_m2 = round(float(poly.area), 2)
                if area_m2 >= min_area:
                    features.append({
                        "geometry": poly,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "area_m2": area_m2
                    })

        return features

    @staticmethod
    def export_geojson(
        geometries: List[shapely.geometry.base.BaseGeometry],
        properties_list: List[Dict[str, Any]],
        output_path: str,
        crs: str = "EPSG:32643"
    ) -> str:
        """
        Exports geometries and their corresponding attributes into a standardized RFC 7946 GeoJSON file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if not geometries:
            # Create valid empty GeoJSON feature collection
            empty_fc = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": crs}
                },
                "features": []
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(empty_fc, f, indent=2)
            return output_path

        gdf = gpd.GeoDataFrame(properties_list, geometry=geometries, crs=crs)
        gdf.to_file(output_path, driver="GeoJSON")
        return output_path
