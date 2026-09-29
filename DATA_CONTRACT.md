# Geospatial Data Contract: AI Feature Extraction Pipeline

## Overview
This document defines the strict data interface specification between the **AI Feature Extraction Module** and all upstream data sources (drone surveys) as well as downstream consumers (cadastral parcel generation engines).

---

## 1. Input Specification: Drone Orthophoto

- **File Name**: `orthophoto.tif` (or any valid GeoTIFF)
- **Data Format**: Cloud-Optimized GeoTIFF (COG) or standard GeoTIFF (`.tif`, `.tiff`)
- **Color Bands**: 3-band RGB (Red, Green, Blue) or 4-band RGB-NIR. Pixel values must be scaled 0–255 (`uint8`).
- **Coordinate Reference System (CRS)**: Must contain an embedded projected CRS (e.g. UTM Zone `EPSG:32643` or local state plane projection) with units in meters.
- **Pixel Resolution (GSD)**: Recommended Ground Sampling Distance between `0.02 m` and `0.20 m` per pixel.
- **Georeferencing**: Affine geotransform matrix embedded in GeoTIFF tags (`rasterio.transform.Affine`).
- **Nodata Value**: Valid nodata mask or 0-padded boundary.

---

## 2. Output Specification: GeoJSON Feature Collections

The pipeline outputs four standard GeoJSON files compliant with **RFC 7946**. All geometries use native metric coordinates matching the orthophoto's projected CRS (or optionally WGS84 EPSG:4326).

### A. `buildings.geojson`
Represents digitized structural footprints.

- **Geometry Type**: `Polygon` or `MultiPolygon`
- **Topological Constraints**: Simple, non-self-intersecting (`is_valid == True`), hole-closed exterior rings.
- **Attributes / Properties**:
  ```json
  {
    "id": "building_001",
    "class": "building",
    "confidence": 0.92,
    "area_m2": 142.6,
    "source": "deep_learning_segmentation"
  }
  ```

### B. `roads.geojson`
Represents vehicular roadways, internal streets, walkways, and access corridors.

- **Geometry Type**: `LineString` (centerline) or `Polygon` (road surface area)
- **Topological Constraints**: Continuous path vectors, minimal self-intersections.
- **Attributes / Properties**:
  ```json
  {
    "id": "road_001",
    "class": "road",
    "confidence": 0.88,
    "length_m": 84.3,
    "source": "deep_learning_segmentation"
  }
  ```

### C. `landuse.geojson`
Represents mutually exclusive land utilization zones categorized across standard cadastral types.

- **Geometry Type**: `Polygon` or `MultiPolygon`
- **Categories**:
  - `0`: `Residential`
  - `1`: `Commercial`
  - `2`: `Institutional`
  - `3`: `Open Space`
  - `4`: `Road`
  - `5`: `Water`
  - `6`: `Other`
- **Attributes / Properties**:
  ```json
  {
    "id": "landuse_001",
    "class": "Residential",
    "class_id": 0,
    "confidence": 0.90,
    "area_m2": 2410.8,
    "source": "multiclass_segmentation"
  }
  ```

### D. `boundary_indicators.geojson`
Spatial delineation indicators (compound walls, fences, hedge rows, road edges, building alignment boundaries).

- **Geometry Type**: `LineString` or `MultiLineString`
- **Legal Disclaimer**:
  > [!IMPORTANT]
  > These features are AI-derived spatial indicators intended to assist downstream parcel generation algorithms. They do NOT constitute legally verified cadastral boundaries.
- **Attributes / Properties**:
  ```json
  {
    "id": "boundary_001",
    "type": "boundary_indicator",
    "confidence": 0.81,
    "length_m": 45.2,
    "source": "hybrid_edge_segmentation",
    "disclaimer": "AI-derived boundary indicators intended for downstream parcel generation. Not legally valid cadastral boundaries."
  }
  ```
