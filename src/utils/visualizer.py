import os
from typing import List, Optional, Any, Tuple
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapely.geometry

class Visualizer:
    """Generates comparison figures and cadastral feature overlays for presentations and demos."""

    @staticmethod
    def generate_comparison(
        image_rgb: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        title: str,
        output_path: str,
        is_categorical: bool = False
    ) -> str:
        """
        Creates a side-by-side 3-panel plot: [Original Orthophoto | Ground Truth | Prediction].
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Orthophoto", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        cmap = "tab10" if is_categorical else "magma"
        vmax = 6 if is_categorical else 1.0

        axes[1].imshow(gt_mask, cmap=cmap, vmin=0, vmax=vmax)
        axes[1].set_title("Ground Truth Mask", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=vmax)
        axes[2].set_title("AI Model Prediction", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    @staticmethod
    def generate_overlay(
        image_rgb: np.ndarray,
        buildings: List[shapely.geometry.Polygon],
        roads: List[shapely.geometry.base.BaseGeometry],
        landuse_features: List[shapely.geometry.Polygon],
        boundaries: List[shapely.geometry.LineString],
        output_path: str,
        affine_transform: Optional[Any] = None
    ) -> str:
        """
        Renders a composite cadastral overlay of all 4 extracted layers onto the orthophoto.
        Correctly aligns pixel coordinates and metric UTM coordinates using affine_transform.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        H, W = image_rgb.shape[:2]

        fig, ax = plt.subplots(figsize=(12, 12))

        # Calculate geospatial extent if affine_transform is provided
        extent = None
        if affine_transform is not None:
            left = affine_transform.c
            top = affine_transform.f
            right = left + W * affine_transform.a
            bottom = top + H * affine_transform.e
            # In projected CRS with north-up imagery: bottom < top
            min_y = min(bottom, top)
            max_y = max(bottom, top)
            extent = [left, right, min_y, max_y]
            ax.imshow(image_rgb, extent=extent, origin="upper")
        else:
            # Fallback: check if geometries are projected (coords > 1000)
            all_geoms = list(buildings) + list(roads) + list(landuse_features) + list(boundaries)
            has_proj = False
            for g in all_geoms:
                if g and not g.is_empty:
                    minx, miny, maxx, maxy = g.bounds
                    if minx > 1000:
                        has_proj = True
                        break
            if has_proj:
                # Approximate extent from valid geometries
                bounds_list = [g.bounds for g in all_geoms if g and not g.is_empty]
                if bounds_list:
                    min_x = min(b[0] for b in bounds_list)
                    min_y = min(b[1] for b in bounds_list)
                    max_x = max(b[2] for b in bounds_list)
                    max_y = max(b[3] for b in bounds_list)
                    extent = [min_x, max_x, min_y, max_y]
                    ax.imshow(image_rgb, extent=extent, origin="upper")
            if extent is None:
                ax.imshow(image_rgb)

        def _get_polygons(geom):
            if isinstance(geom, shapely.geometry.Polygon):
                return [geom]
            elif isinstance(geom, shapely.geometry.MultiPolygon):
                return list(geom.geoms)
            return []

        def _get_lines(geom):
            if isinstance(geom, shapely.geometry.LineString):
                return [geom]
            elif isinstance(geom, shapely.geometry.MultiLineString):
                return list(geom.geoms)
            return []

        # 1. Plot Land-use (Subtle transparent categorical fills)
        for item in landuse_features:
            for poly in _get_polygons(item):
                if poly.is_empty or not poly.exterior:
                    continue
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.20, color="#00E5FF")
                ax.plot(x, y, color="#00B0FF", linewidth=1.0, alpha=0.4)

        # 2. Plot Roads (Vibrant amber/gold lines and corridors)
        for item in roads:
            for line in _get_lines(item):
                if line.is_empty:
                    continue
                x, y = line.xy
                ax.plot(x, y, color="#FFD700", linewidth=3.0, alpha=0.95)
            for poly in _get_polygons(item):
                if poly.is_empty or not poly.exterior:
                    continue
                x, y = poly.exterior.xy
                ax.fill(x, y, color="#FFD700", alpha=0.40)
                ax.plot(x, y, color="#FFA000", linewidth=2.0)

        # 3. Plot Boundary Indicators (High-contrast magenta dashed lines)
        for item in boundaries:
            for line in _get_lines(item):
                if line.is_empty:
                    continue
                x, y = line.xy
                ax.plot(x, y, color="#E040FB", linestyle="--", linewidth=2.0, alpha=0.9)

        # 4. Plot Buildings (Crimson red outlines and transparent red fill)
        for item in buildings:
            for poly in _get_polygons(item):
                if poly.is_empty or not poly.exterior:
                    continue
                x, y = poly.exterior.xy
                ax.fill(x, y, color="#FF1744", alpha=0.45)
                ax.plot(x, y, color="#D50000", linewidth=2.2)

        # Lock axis limits strictly to the orthophoto extent
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        else:
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        ax.set_title("AI Feature Extraction: Integrated Cadastral Overlay", fontsize=15, fontweight="bold", pad=12)
        ax.axis("off")

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor="#FF1744", edgecolor="#D50000", alpha=0.5, label="Buildings (Footprints)"),
            Line2D([0], [0], color="#FFD700", lw=3.0, label="Roads & Corridors"),
            Line2D([0], [0], color="#E040FB", lw=2.0, linestyle="--", label="Boundary Indicators"),
            Patch(facecolor="#00E5FF", edgecolor="#00B0FF", alpha=0.3, label="Land-Use Zones")
        ]
        ax.legend(handles=legend_elements, loc="lower right", framealpha=0.92, facecolor="white", edgecolor="#CCCCCC", fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path
