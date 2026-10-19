import os
import shutil
import tempfile
import pytest
import rasterio
from src.geospatial.synthetic_generator import SyntheticDroneGenerator

def test_synthetic_scene_generation():
    temp_dir = tempfile.mkdtemp(prefix="test_synthetic_")
    try:
        generator = SyntheticDroneGenerator(
            tile_size=256,
            resolution=0.1,
            crs="EPSG:32643"
        )
        meta = generator.generate_scene("scene_test", temp_dir, seed=42)

        assert meta["scene_id"] == "scene_test"
        assert meta["crs"] == "EPSG:32643"
        assert meta["width"] == 256
        assert meta["height"] == 256

        # Check files exist
        ortho_path = meta["files"]["orthophoto"]
        assert os.path.exists(ortho_path)
        assert os.path.exists(meta["files"]["building_mask"])
        assert os.path.exists(meta["files"]["road_mask"])
        assert os.path.exists(meta["files"]["landuse_raster"])
        assert os.path.exists(meta["files"]["boundary_mask"])

        # Validate GeoTIFF metadata
        with rasterio.open(ortho_path) as src:
            assert src.crs.to_string() == "EPSG:32643"
            assert src.width == 256
            assert src.height == 256
            assert src.count == 3
            assert src.transform.a == 0.1
            assert src.transform.e == -0.1

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
