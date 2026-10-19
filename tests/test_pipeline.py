import os
import shutil
import tempfile
import pytest
import geopandas as gpd

from src.geospatial.synthetic_generator import SyntheticDroneGenerator
from src.inference.pipeline import FeatureExtractionPipeline

def test_pipeline_end_to_end():
    temp_dir = tempfile.mkdtemp(prefix="test_pipeline_")
    try:
        scene_dir = os.path.join(temp_dir, "scene_001")
        generator = SyntheticDroneGenerator(tile_size=256, resolution=0.1, crs="EPSG:32643")
        meta = generator.generate_scene("scene_001", temp_dir, seed=42)
        ortho_path = meta["files"]["orthophoto"]

        out_dir = os.path.join(temp_dir, "outputs")
        pipeline = FeatureExtractionPipeline(config_path="config.yaml")
        results = pipeline.run(input_orthophoto=ortho_path, output_dir=out_dir)

        # Verify all 4 files returned and created
        assert "buildings" in results
        assert "roads" in results
        assert "landuse" in results
        assert "boundaries" in results

        b_file = results["buildings"]
        r_file = results["roads"]
        lu_file = results["landuse"]
        bound_file = results["boundaries"]

        assert os.path.exists(b_file)
        assert os.path.exists(r_file)
        assert os.path.exists(lu_file)
        assert os.path.exists(bound_file)

        # Validate GeoJSON geometries
        gdf_b = gpd.read_file(b_file)
        assert gdf_b.crs is not None
        for geom in gdf_b.geometry:
            assert geom.is_valid
            assert geom.geom_type in ["Polygon", "MultiPolygon"]

        gdf_r = gpd.read_file(r_file)
        assert gdf_r.crs is not None
        for geom in gdf_r.geometry:
            assert geom.is_valid

        gdf_lu = gpd.read_file(lu_file)
        assert gdf_lu.crs is not None
        for geom in gdf_lu.geometry:
            assert geom.is_valid

        gdf_bd = gpd.read_file(bound_file)
        assert gdf_bd.crs is not None
        for geom in gdf_bd.geometry:
            assert geom.is_valid

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
