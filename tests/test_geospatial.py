import os
import shutil
import tempfile
import numpy as np
import shapely.geometry
from rasterio.transform import from_origin

from src.geospatial.raster_io import RasterIO
from src.geospatial.vector_io import VectorIO
from src.geospatial.coordinate_transform import CoordinateTransformer

def test_raster_io_and_crs():
    temp_dir = tempfile.mkdtemp(prefix="test_geo_")
    try:
        t_path = os.path.join(temp_dir, "test.tif")
        data = np.ones((100, 100, 3), dtype=np.uint8) * 128
        transform = from_origin(500000, 3000000, 0.1, 0.1)

        RasterIO.write_geotiff(t_path, data, crs="EPSG:32643", transform=transform)
        assert os.path.exists(t_path)

        read_img, meta = RasterIO.read_orthophoto(t_path)
        assert read_img.shape == (100, 100, 3)
        assert meta["crs"] == "EPSG:32643"
        assert meta["width"] == 100
        assert meta["height"] == 100
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_coordinate_transformer_area_and_length():
    trans = CoordinateTransformer(src_crs="EPSG:32643", dst_crs="EPSG:32643")
    # 10m x 10m square = 100 m²
    poly = shapely.geometry.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert trans.compute_area_m2(poly) == 100.0

    # 50m line
    line = shapely.geometry.LineString([(0, 0), (50, 0)])
    assert trans.compute_length_m(line) == 50.0

def test_vector_io_polygonization():
    temp_dir = tempfile.mkdtemp(prefix="test_vec_")
    try:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 20:50] = 1  # 30x30 pixels
        transform = from_origin(500000, 3000000, 0.1, 0.1)  # 0.1m/px -> 3m x 3m = 9m²

        polys = VectorIO.mask_to_polygons(mask, transform, min_area=5.0)
        assert len(polys) == 1
        assert polys[0].is_valid
        assert abs(polys[0].area - 9.0) < 0.5

        # Test GeoJSON Export
        geojson_path = os.path.join(temp_dir, "test.geojson")
        props = [{"id": "b1", "class": "building", "area_m2": 9.0}]
        VectorIO.export_geojson(polys, props, geojson_path, crs="EPSG:32643")
        assert os.path.exists(geojson_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
