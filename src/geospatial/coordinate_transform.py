from typing import Union
import shapely.geometry
from shapely.ops import transform
import pyproj
from rasterio.transform import Affine

class CoordinateTransformer:
    """Provides coordinate transformation, area in m², and length in meters."""

    def __init__(self, src_crs: str = "EPSG:32643", dst_crs: str = "EPSG:32643"):
        self.src_crs_str = src_crs
        self.dst_crs_str = dst_crs
        self.is_projected = "326" in src_crs or "3857" in src_crs or "utm" in src_crs.lower()

        if src_crs != dst_crs:
            self.transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        else:
            self.transformer = None

    @staticmethod
    def pixel_to_geo(col: float, row: float, affine_transform: Affine) -> tuple:
        """Converts pixel column and row into projected geospatial coordinates (x, y)."""
        x, y = affine_transform @ (col, row)
        return x, y

    def compute_area_m2(self, geom: shapely.geometry.base.BaseGeometry) -> float:
        """
        Calculates area in square meters.
        If CRS is already projected (e.g. UTM), returns geom.area directly.
        If geographic (e.g. EPSG:4326), reprojects to UTM or computes geodetic area.
        """
        if geom is None or geom.is_empty:
            return 0.0

        if self.is_projected:
            return float(round(geom.area, 2))
        else:
            # Reproject to UTM 43N or appropriate metric zone
            project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True).transform
            projected_geom = transform(project, geom)
            return float(round(projected_geom.area, 2))

    def compute_length_m(self, geom: shapely.geometry.base.BaseGeometry) -> float:
        """
        Calculates length in meters for linear features.
        """
        if geom is None or geom.is_empty:
            return 0.0

        if self.is_projected:
            return float(round(geom.length, 2))
        else:
            project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True).transform
            projected_geom = transform(project, geom)
            return float(round(projected_geom.length, 2))
