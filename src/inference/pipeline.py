import os
import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import cv2

from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.preprocessing.normalizer import ImageNormalizer
from src.preprocessing.tiler import OrthoTiler
from src.geospatial.raster_io import RasterIO
from src.building_extraction.model import DroneUNet
from src.building_extraction.postprocessor import BuildingPostprocessor
from src.road_extraction.model import RoadSegmentationNet
from src.road_extraction.postprocessor import RoadPostprocessor
from src.landuse_classification.model import LandUseSegmentationNet
from src.landuse_classification.postprocessor import LandUsePostprocessor
from src.boundary_extraction.detector import BoundaryDetector
from src.boundary_extraction.postprocessor import BoundaryPostprocessor
from src.utils.visualizer import Visualizer

logger = get_logger("FeatureExtractionPipeline")

class FeatureExtractionPipeline:
    """
    Standardized, production-grade AI Feature Extraction Pipeline for drone imagery.
    Hides all preprocessing, deep learning inference, morphological processing,
    and polygonization behind a single intuitive entrypoint: `run(input_orthophoto)`.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config["training"].get("device") == "cuda" else "cpu")
        self.normalizer = ImageNormalizer()
        self.tile_size = self.config["geospatial"].get("tile_size", 512)
        self.tiler = OrthoTiler(tile_size=self.tile_size, overlap=64)

        # 1. Initialize Building Module
        self.building_model = DroneUNet(in_channels=3, num_classes=1).to(self.device)
        self._load_checkpoint(self.building_model, self.config["models"]["building"]["checkpoint_path"], "Building")
        self.building_model.eval()
        self.building_post = BuildingPostprocessor(
            threshold=self.config["models"]["building"].get("threshold", 0.5),
            min_area_m2=self.config["models"]["building"].get("min_area_m2", 15.0),
            simplify_tolerance=self.config["models"]["building"].get("simplify_tolerance", 0.3),
            crs=self.config["geospatial"].get("default_crs", "EPSG:32643")
        )

        # 2. Initialize Road Module
        self.road_model = RoadSegmentationNet(in_channels=3, num_classes=1).to(self.device)
        self._load_checkpoint(self.road_model, self.config["models"]["road"]["checkpoint_path"], "Road")
        self.road_model.eval()
        self.road_post = RoadPostprocessor(
            threshold=self.config["models"]["road"].get("threshold", 0.45),
            min_length_m=self.config["models"]["road"].get("min_length_m", 10.0),
            simplify_tolerance=self.config["models"]["road"].get("simplify_tolerance", 0.4),
            crs=self.config["geospatial"].get("default_crs", "EPSG:32643")
        )

        # 3. Initialize Land-Use Module
        num_classes = self.config["models"]["landuse"].get("num_classes", 7)
        self.landuse_model = LandUseSegmentationNet(in_channels=3, num_classes=num_classes).to(self.device)
        self._load_checkpoint(self.landuse_model, self.config["models"]["landuse"]["checkpoint_path"], "LandUse")
        self.landuse_model.eval()
        self.landuse_post = LandUsePostprocessor(
            class_mapping=self.config["models"]["landuse"].get("classes"),
            min_region_area_m2=self.config["models"]["landuse"].get("min_region_area_m2", 30.0),
            simplify_tolerance=self.config["models"]["landuse"].get("simplify_tolerance", 0.5),
            crs=self.config["geospatial"].get("default_crs", "EPSG:32643")
        )

        # 4. Initialize Boundary Indicator Module
        self.boundary_detector = BoundaryDetector(
            canny_low=self.config["models"]["boundary"].get("canny_low", 40),
            canny_high=self.config["models"]["boundary"].get("canny_high", 120),
            device=str(self.device)
        )
        bound_ckpt = self.config["models"]["boundary"]["checkpoint_path"]
        if os.path.exists(bound_ckpt):
            self.boundary_detector.load_model(bound_ckpt)
        self.boundary_post = BoundaryPostprocessor(
            min_line_length_m=self.config["models"]["boundary"].get("min_line_length_m", 5.0),
            simplify_tolerance=self.config["models"]["boundary"].get("simplify_tolerance", 0.3),
            crs=self.config["geospatial"].get("default_crs", "EPSG:32643"),
            disclaimer=self.config["models"]["boundary"].get("disclaimer")
        )

        self.visualizer = Visualizer()

    def _load_checkpoint(self, model: torch.nn.Module, path: str, name: str) -> None:
        """Helper to safely load model checkpoint if present."""
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                else:
                    model.load_state_dict(ckpt)
                logger.info(f"Loaded {name} model weights from {path}")
            except Exception as e:
                logger.warning(f"Could not load {name} checkpoint ({e}). Initialized with default weights.")
        else:
            logger.info(f"{name} checkpoint not found at '{path}'. Running with current baseline initialization.")

    def run(
        self,
        input_orthophoto: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs the complete end-to-end feature extraction pipeline on a drone orthophoto.
        Returns:
            Dict containing output file paths and detection summary metrics.
        """
        start_time = time.time()
        logger.info(f"==================================================")
        logger.info(f"Loading orthophoto: {input_orthophoto}")

        if not os.path.exists(input_orthophoto):
            raise FileNotFoundError(f"Input orthophoto does not exist: {input_orthophoto}")

        output_dir = output_dir or self.config["data"].get("output_dir", "data/outputs")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Read Orthophoto & Extract Geospatial Metadata
        image_rgb, meta = RasterIO.read_orthophoto(input_orthophoto)
        H, W = image_rgb.shape[:2]
        affine_transform = meta["transform"]
        crs = meta["crs"]
        logger.info(f"Orthophoto loaded: {W}x{H} pixels, CRS: {crs}")

        # 2. Patch-based Tiling and Inference
        tiles, coords = self.tiler.generate_tiles(image_rgb)
        logger.info(f"Image partitioned into {len(tiles)} tiles (tile_size={self.tile_size})")

        b_tile_preds = []
        r_tile_preds = []
        lu_tile_preds = []

        with torch.no_grad():
            for tile in tiles:
                tensor = self.normalizer.to_tensor(tile).unsqueeze(0).to(self.device)

                # Building prediction
                b_logits = self.building_model(tensor)
                b_prob = torch.sigmoid(b_logits).squeeze().cpu().numpy()
                b_tile_preds.append(b_prob)

                # Road prediction
                r_logits = self.road_model(tensor)
                r_prob = torch.sigmoid(r_logits).squeeze().cpu().numpy()
                r_tile_preds.append(r_prob)

                # Land-use prediction
                lu_logits = self.landuse_model(tensor)
                lu_probs = torch.softmax(lu_logits, dim=1).squeeze().cpu().numpy()  # (C, H, W)
                lu_tile_preds.append(lu_probs)

        # 3. Merge Tile Predictions
        building_prob_map = self.tiler.merge_probability_maps(b_tile_preds, coords, (H, W), num_classes=1)
        road_prob_map = self.tiler.merge_probability_maps(r_tile_preds, coords, (H, W), num_classes=1)
        num_lu_classes = self.config["models"]["landuse"].get("num_classes", 7)
        landuse_prob_vol = self.tiler.merge_probability_maps(lu_tile_preds, coords, (H, W), num_classes=num_lu_classes)

        # 4. Postprocess Buildings
        logger.info("Running building footprint extraction...")
        b_polygons, b_props, b_clean_mask = self.building_post.process_probability_mask(building_prob_map, affine_transform)
        buildings_path = os.path.join(output_dir, "buildings.geojson")
        self.building_post.export_geojson(b_polygons, b_props, buildings_path)
        logger.info(f"Buildings detected: {len(b_polygons)} (saved to {buildings_path})")

        # 5. Postprocess Roads
        logger.info("Running road network extraction...")
        r_geoms, r_props, r_clean_mask = self.road_post.process_probability_mask(road_prob_map, affine_transform)
        roads_path = os.path.join(output_dir, "roads.geojson")
        self.road_post.export_geojson(r_geoms, r_props, roads_path)
        logger.info(f"Road features detected: {len(r_geoms)} (saved to {roads_path})")

        # 6. Postprocess Land-Use
        logger.info("Running land-use classification...")
        lu_polygons, lu_props, lu_clean_mask = self.landuse_post.process_probability_volume(landuse_prob_vol, affine_transform)
        landuse_path = os.path.join(output_dir, "landuse.geojson")
        self.landuse_post.export_geojson(lu_polygons, lu_props, landuse_path)
        logger.info(f"Land-use zones identified: {len(lu_polygons)} (saved to {landuse_path})")

        # 7. Postprocess Boundary Indicators
        logger.info("Extracting parcel-boundary indicators...")
        bound_mask, bound_source = self.boundary_detector.detect_boundaries(
            image_rgb=image_rgb,
            building_mask=b_clean_mask,
            road_mask=r_clean_mask
        )
        bound_lines, bound_props = self.boundary_post.process_boundary_mask(
            boundary_mask=bound_mask,
            affine_transform=affine_transform,
            source_tag=bound_source
        )
        boundaries_path = os.path.join(output_dir, "boundary_indicators.geojson")
        self.boundary_post.export_geojson(bound_lines, bound_props, boundaries_path)
        logger.info(f"Boundary indicators extracted: {len(bound_lines)} (saved to {boundaries_path})")

        # 8. Visualizations & Overlays
        elapsed_sec = round(time.time() - start_time, 2)
        overlay_path = os.path.join(output_dir, "all_features_overlay.png")
        if self.config["inference"].get("save_visualizations", True):
            logger.info("Generating demonstration visualizations and cadastral overlay...")
            self.visualizer.generate_overlay(
                image_rgb=image_rgb,
                buildings=b_polygons,
                roads=r_geoms,
                landuse_features=lu_polygons,
                boundaries=bound_lines,
                output_path=overlay_path,
                affine_transform=affine_transform
            )

        # 9. Photogrammetric Elevation Modeling (DEM, DSM, DTM, Contours, Breaklines)
        elevation_results = {}
        try:
            from src.elevation.elevation_generator import ElevationGenerator
            logger.info("Deriving photogrammetric DEM, DSM, and DTM models...")
            elevation_gen = ElevationGenerator(default_crs=crs)
            elevation_results = elevation_gen.generate_all(
                image_rgb=image_rgb,
                affine_transform=affine_transform,
                crs=crs,
                output_dir=output_dir,
                buildings=b_polygons,
                roads=r_geoms,
                landuse_features=lu_polygons,
                boundaries=bound_lines
            )
        except Exception as e:
            logger.warning(f"Elevation generation warning: {e}")

        logger.info(f"Feature Extraction Pipeline completed in {elapsed_sec}s")
        logger.info(f"==================================================")

        return {
            "buildings": buildings_path,
            "roads": roads_path,
            "landuse": landuse_path,
            "boundaries": boundaries_path,
            "overlay_image": overlay_path,
            "dem_png": elevation_results.get("dem_png"),
            "dsm_png": elevation_results.get("dsm_png"),
            "dtm_png": elevation_results.get("dtm_png"),
            "dem_tif": elevation_results.get("dem_tif"),
            "dsm_tif": elevation_results.get("dsm_tif"),
            "dtm_tif": elevation_results.get("dtm_tif"),
            "contours": elevation_results.get("contours_geojson"),
            "breaklines": elevation_results.get("breaklines_geojson"),
            "elevation_stats": elevation_results.get("elevation_stats", {}),
            "summary": {
                "input_orthophoto": input_orthophoto,
                "crs": crs,
                "dimensions": f"{W}x{H}",
                "buildings_count": len(b_polygons),
                "roads_count": len(r_geoms),
                "landuse_regions_count": len(lu_polygons),
                "boundaries_count": len(bound_lines),
                "processing_time_s": elapsed_sec
            }
        }
