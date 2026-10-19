import pytest
import torch
import numpy as np
from rasterio.transform import from_origin

from src.building_extraction.model import DroneUNet
from src.building_extraction.postprocessor import BuildingPostprocessor
from src.road_extraction.model import RoadSegmentationNet
from src.road_extraction.postprocessor import RoadPostprocessor
from src.landuse_classification.model import LandUseSegmentationNet
from src.landuse_classification.postprocessor import LandUsePostprocessor
from src.boundary_extraction.detector import BoundaryDetector
from src.boundary_extraction.postprocessor import BoundaryPostprocessor

def test_drone_unet_forward():
    model = DroneUNet(in_channels=3, num_classes=1, base_features=16)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)

def test_road_segmentation_net_forward():
    model = RoadSegmentationNet(in_channels=3, num_classes=1)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)

def test_landuse_net_forward():
    model = LandUseSegmentationNet(in_channels=3, num_classes=7)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 7, 64, 64)

def test_building_postprocessor():
    post = BuildingPostprocessor(threshold=0.5, min_area_m2=10.0)
    prob_map = np.zeros((100, 100), dtype=np.float32)
    prob_map[20:60, 20:60] = 0.9  # 40x40 px = 4m x 4m = 16 m² at 0.1m/px
    transform = from_origin(700000, 3100000, 0.1, 0.1)

    polys, props, clean_mask = post.process_probability_mask(prob_map, transform)
    assert len(polys) == 1
    assert polys[0].is_valid
    assert props[0]["class"] == "building"
    assert props[0]["area_m2"] >= 10.0

def test_boundary_detector_and_postprocessor():
    detector = BoundaryDetector(canny_low=30, canny_high=100)
    dummy_img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    mask, source = detector.detect_boundaries(dummy_img)
    assert mask.shape == (100, 100)
    assert "edge" in source or "hybrid" in source

    post = BoundaryPostprocessor(min_line_length_m=1.0)
    transform = from_origin(700000, 3100000, 0.1, 0.1)
    lines, props = post.process_boundary_mask(mask, transform, source_tag=source)
    assert isinstance(lines, list)
    assert isinstance(props, list)
