import os
import json
import random
from typing import Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image
from rasterio.transform import from_origin
from src.geospatial.raster_io import RasterIO

class SyntheticDroneGenerator:
    """
    Generates realistic synthetic high-resolution aerial/drone scenes
    complete with georeferenced GeoTIFF orthophotos, ground truth masks,
    and metadata for training and testing.
    """

    def __init__(
        self,
        tile_size: int = 512,
        resolution: float = 0.1,  # 0.1 meter per pixel (10cm GSD)
        crs: str = "EPSG:32643",  # UTM Zone 43N (Metric coordinates)
        origin_x: float = 700000.0,
        origin_y: float = 3100000.0
    ):
        self.tile_size = tile_size
        self.resolution = resolution
        self.crs = crs
        self.origin_x = origin_x
        self.origin_y = origin_y

    def generate_scene(
        self,
        scene_id: str,
        output_dir: str,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Generates a complete synthetic drone scene with orthophoto and 4 ground truth masks.
        """
        random.seed(seed)
        np.random.seed(seed)

        H, W = self.tile_size, self.tile_size
        # Canvas layers
        ortho_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        building_mask = np.zeros((H, W), dtype=np.uint8)
        road_mask = np.zeros((H, W), dtype=np.uint8)
        landuse_mask = np.zeros((H, W), dtype=np.uint8)  # 0:Res, 1:Comm, 2:Inst, 3:Open, 4:Road, 5:Water, 6:Other
        boundary_mask = np.zeros((H, W), dtype=np.uint8)

        # 1. Base Terrain Texture (Open space by default: class 3)
        base_noise = np.random.normal(loc=0, scale=12, size=(H, W, 3))
        base_green = np.array([55, 125, 45], dtype=np.float32)  # Vegetation / lawn green
        terrain = np.clip(base_green + base_noise, 30, 200).astype(np.uint8)
        ortho_rgb[:] = terrain
        landuse_mask[:] = 3  # Open Space

        # 2. Add Water Body (e.g. pond or canal in corner/edge) - Class 5
        has_water = random.random() > 0.4
        if has_water:
            water_center = (random.randint(40, W // 3), random.randint(40, H // 3))
            water_axes = (random.randint(30, 70), random.randint(25, 55))
            water_angle = random.randint(0, 180)
            water_mask_temp = np.zeros((H, W), dtype=np.uint8)
            cv2.ellipse(water_mask_temp, water_center, water_axes, water_angle, 0, 360, 255, -1)
            # Water color: deep blue-green
            water_color = np.array([120, 85, 35], dtype=np.uint8)  # BGR order for CV2 or RGB: [35, 85, 120]
            water_noise = np.random.normal(0, 5, (H, W, 3)).astype(np.int16)
            water_rgb = np.clip(np.array([35, 95, 125]) + water_noise, 0, 255).astype(np.uint8)
            indices = water_mask_temp == 255
            ortho_rgb[indices] = water_rgb[indices]
            landuse_mask[indices] = 5  # Water

        # 3. Roads and Access Corridors - Class 4
        # Main road (spanning vertically or horizontally)
        road_coords = []
        is_vertical = random.random() > 0.5
        road_width = random.randint(14, 22)

        if is_vertical:
            rx = random.randint(W // 3, 2 * W // 3)
            pt1 = (rx, 0)
            pt2 = (rx + random.randint(-20, 20), H)
            road_coords.append((pt1, pt2, road_width))
            # Secondary branch road
            ry = random.randint(H // 4, 3 * H // 4)
            pt3 = (rx, ry)
            pt4 = (W if random.random() > 0.5 else 0, ry + random.randint(-15, 15))
            road_coords.append((pt3, pt4, int(road_width * 0.75)))
        else:
            ry = random.randint(H // 3, 2 * H // 3)
            pt1 = (0, ry)
            pt2 = (W, ry + random.randint(-20, 20))
            road_coords.append((pt1, pt2, road_width))
            rx = random.randint(W // 4, 3 * W // 4)
            pt3 = (rx, ry)
            pt4 = (rx + random.randint(-15, 15), H if random.random() > 0.5 else 0)
            road_coords.append((pt3, pt4, int(road_width * 0.75)))

        for (p1, p2, rw) in road_coords:
            # Draw on road mask
            cv2.line(road_mask, p1, p2, 255, rw)
            # Draw asphalt on ortho (dark grey ~70-90)
            cv2.line(ortho_rgb, p1, p2, (80, 80, 82), rw)
            # Lane markings (dashed white/yellow line in center)
            cv2.line(ortho_rgb, p1, p2, (220, 220, 215), 1)
            # Mark land-use as road (class 4)
            cv2.line(landuse_mask, p1, p2, 4, rw)

        # 4. Land-use Sectors (Residential = 0, Commercial = 1, Institutional = 2)
        # Assign quadrants or sectors
        sector_mask_res = np.zeros((H, W), dtype=bool)
        sector_mask_comm = np.zeros((H, W), dtype=bool)
        sector_mask_inst = np.zeros((H, W), dtype=bool)

        # Residential zone typically occupies one major sector
        res_x1, res_y1, res_x2, res_y2 = 10, 10, W // 2 - 20, H // 2 - 20
        sector_mask_res[res_y1:res_y2, res_x1:res_x2] = True

        # Commercial zone
        comm_x1, comm_y1, comm_x2, comm_y2 = W // 2 + 20, 10, W - 10, H // 2 - 20
        sector_mask_comm[comm_y1:comm_y2, comm_x1:comm_x2] = True

        # Institutional zone
        inst_x1, inst_y1, inst_x2, inst_y2 = 10, H // 2 + 20, W // 2 - 20, H - 10
        sector_mask_inst[inst_y1:inst_y2, inst_x1:inst_x2] = True

        # Apply base landuse colors and labels where road/water not present
        non_road_water = (road_mask == 0) & (landuse_mask != 5)
        res_idx = sector_mask_res & non_road_water
        comm_idx = sector_mask_comm & non_road_water
        inst_idx = sector_mask_inst & non_road_water

        landuse_mask[res_idx] = 0  # Residential
        landuse_mask[comm_idx] = 1 # Commercial
        landuse_mask[inst_idx] = 2 # Institutional

        # Light ground color tints for sectors
        ortho_rgb[res_idx] = np.clip(ortho_rgb[res_idx].astype(np.int16) + np.array([10, -5, -10]), 0, 255).astype(np.uint8)
        ortho_rgb[comm_idx] = np.clip(ortho_rgb[comm_idx].astype(np.int16) + np.array([20, 20, 15]), 0, 255).astype(np.uint8)
        ortho_rgb[inst_idx] = np.clip(ortho_rgb[inst_idx].astype(np.int16) + np.array([5, 15, 20]), 0, 255).astype(np.uint8)

        # 5. Boundary Indicators (Walls, Fences, Field Borders)
        # Create parcel boundary outlines around residential plots and roadsides
        boundary_candidates = []
        # Roadside boundaries
        road_dilated = cv2.dilate(road_mask, np.ones((5, 5), np.uint8), iterations=2)
        road_edge = cv2.Canny(road_dilated, 100, 200)
        boundary_mask[road_edge > 0] = 255

        # Cadastral parcel grid dividers in residential/agricultural areas
        grid_step = random.randint(70, 95)
        for gx in range(20, W - 20, grid_step):
            p_top = (gx, random.randint(10, 30))
            p_bot = (gx + random.randint(-5, 5), H - random.randint(10, 30))
            cv2.line(boundary_mask, p_top, p_bot, 255, 1)
        for gy in range(20, H - 20, grid_step):
            p_left = (random.randint(10, 30), gy)
            p_right = (W - random.randint(10, 30), gy + random.randint(-5, 5))
            cv2.line(boundary_mask, p_left, p_right, 255, 1)

        # Remove boundary indicators where water or road centers are
        boundary_mask[road_mask > 0] = 0
        boundary_mask[landuse_mask == 5] = 0

        # Draw visible fences/compound walls on ortho (light grey or reddish brown line)
        wall_indices = boundary_mask == 255
        ortho_rgb[wall_indices] = np.array([190, 185, 175], dtype=np.uint8)

        # 6. Buildings Footprints & Rooftops
        # Place realistic buildings in residential, commercial, institutional, and open zones
        building_palettes = [
            (195, 75, 55),   # Terracotta red roof
            (215, 190, 160), # Light cement/sand
            (130, 140, 150), # Grey concrete
            (70, 110, 165),  # Blue industrial tin
            (225, 215, 200), # White plastered flat roof
            (160, 115, 80)   # Brown timber/asphalt shingle
        ]

        num_buildings = random.randint(10, 18)
        placed_buildings = []

        for b_idx in range(num_buildings):
            # Size in pixels (e.g. 25-60 pixels = 2.5m to 6.0m)
            bw = random.randint(26, 54)
            bh = random.randint(24, 50)
            bx = random.randint(15, W - bw - 15)
            by = random.randint(15, H - bh - 15)

            # Check collision with roads and water
            sub_road = road_mask[by:by+bh, bx:bx+bw]
            sub_water = (landuse_mask[by:by+bh, bx:bx+bw] == 5)
            if np.any(sub_road > 0) or np.any(sub_water):
                continue

            angle = random.choice([0, 15, 30, 45, 90])
            rect = ((bx + bw / 2, by + bh / 2), (bw, bh), angle)
            box_points = cv2.boxPoints(rect).astype(np.int32)

            # Draw building shadow (offset north-east or south-east)
            shadow_offset = (random.randint(4, 8), random.randint(4, 8))
            shadow_points = box_points + np.array(shadow_offset)
            shadow_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(shadow_mask, [shadow_points], 255)
            # Darken ortho for shadow
            shadow_idx = (shadow_mask == 255) & (road_mask == 0)
            ortho_rgb[shadow_idx] = (ortho_rgb[shadow_idx] * 0.55).astype(np.uint8)

            # Draw building rooftop
            b_color = random.choice(building_palettes)
            # Add rooftop structure/ridge line
            cv2.fillPoly(ortho_rgb, [box_points], b_color)
            cv2.polylines(ortho_rgb, [box_points], True, (40, 40, 40), 1)  # Parapet line

            # Mark building mask
            cv2.fillPoly(building_mask, [box_points], 255)
            placed_buildings.append(box_points)

        # Ensure buildings don't overlap boundary lines
        boundary_mask[building_mask > 0] = 0

        # 7. Add High-Resolution Drone Sensor Noise and Blur
        drone_noise = np.random.normal(0, 4, ortho_rgb.shape).astype(np.int16)
        ortho_rgb = np.clip(ortho_rgb.astype(np.int16) + drone_noise, 0, 255).astype(np.uint8)

        # Geospatial Metadata: Affine transform and Projected CRS
        # Calculate real-world coordinates: origin_x, origin_y with pixel size
        transform = from_origin(
            west=self.origin_x,
            north=self.origin_y,
            xsize=self.resolution,
            ysize=self.resolution
        )

        scene_dir = os.path.join(output_dir, scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        # Write GeoTIFF orthophoto
        ortho_path = os.path.join(scene_dir, "orthophoto.tif")
        RasterIO.write_geotiff(
            file_path=ortho_path,
            data=ortho_rgb,
            crs=self.crs,
            transform=transform,
            dtype="uint8"
        )

        # Write Ground Truth Masks as PNG & GeoTIFF for convenience
        b_mask_path = os.path.join(scene_dir, "building_mask.png")
        Image.fromarray(building_mask).save(b_mask_path)

        r_mask_path = os.path.join(scene_dir, "road_mask.png")
        Image.fromarray(road_mask).save(r_mask_path)

        lu_mask_path = os.path.join(scene_dir, "landuse_raster.png")
        Image.fromarray(landuse_mask).save(lu_mask_path)

        bound_mask_path = os.path.join(scene_dir, "boundary_mask.png")
        Image.fromarray(boundary_mask).save(bound_mask_path)

        # Save metadata
        bounds = (
            self.origin_x,
            self.origin_y - (H * self.resolution),
            self.origin_x + (W * self.resolution),
            self.origin_y
        )
        metadata = {
            "scene_id": scene_id,
            "crs": self.crs,
            "width": W,
            "height": H,
            "resolution_m": self.resolution,
            "bounds_utm": bounds,
            "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
            "buildings_count": len(placed_buildings),
            "files": {
                "orthophoto": ortho_path,
                "building_mask": b_mask_path,
                "road_mask": r_mask_path,
                "landuse_raster": lu_mask_path,
                "boundary_mask": bound_mask_path
            }
        }

        meta_path = os.path.join(scene_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata
