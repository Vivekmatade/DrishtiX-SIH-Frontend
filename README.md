# DrishtiX 🛰️
### AI-Powered Cadastral Intelligence & Photogrammetric Terrain Analysis

> **Smart India Hackathon (SIH)** — Automated Cadastral Feature Extraction, Land-Use Classification, Boundary Delineation, and Digital Elevation Modeling (DEM / DSM / DTM) from Drone & Aerial Orthophotos.

---

## 🌟 Overview

**DrishtiX** is an end-to-end geospatial artificial intelligence platform designed to automate large-scale cadastral mapping from high-resolution drone orthophotos and aerial imagery. 

By combining deep semantic segmentation, morphological postprocessing, topological vectorization, and photogrammetric terrain reconstruction, DrishtiX transforms raw aerial imagery into survey-grade vector datasets and elevation models in seconds.

---

## 🚀 Key Capabilities

### 1. 🏠 Automated Building Footprint Extraction (`buildings.geojson`)
- High-precision deep learning segmentation (U-Net / ResNet architecture).
- Polygon regularization, Douglas-Peucker simplification, and orthogonal squaring.
- Per-building geometric metrics: polygon geometry, confidence score, and footprint area ($m^2$).

### 2. 🛣️ Road & Access Corridor Tracing (`roads.geojson`)
- Continuous road network extraction, pathway tracing, and corridor classification.
- Centerline skeletonization and graph vectorization.
- Multi-scale connectivity preservation with segment length calculations ($m$).

### 3. 🗺️ Multiclass Land-Use Classification (`landuse.geojson`)
- Pixel-to-parcel multi-class segmentation covering:
  - **Institutional** (academic, research, administrative blocks)
  - **Residential** (hostels, housing quarters)
  - **Commercial** (amenities, utility zones)
  - **Open Space / Vegetation** (parks, open grounds)
  - **Road & Transportation**
  - **Water Bodies**
  - **Other / Transition**
- GeoJSON export with categorical metadata and zone area computations ($m^2$ / ha).

### 4. 📐 Boundary Indicators (`boundary_indicators.geojson`)
- Hybrid edge and semantic boundary detection (walls, fences, hedgerows, parcel separators).
- Provides foundational delineation cues for downstream legal parcel partitioning.
- Includes confidence metrics and survey disclaimers.

### 5. ⛰️ Photogrammetric Elevation Modeling (DEM / DSM / DTM)
- **DEM (Digital Elevation Model)**: Ground surface elevation grid essential for hydrology, flood risk analysis, and earthwork engineering.
- **DSM (Digital Surface Model)**: Full surface capture including tree canopies, building roofs, infrastructure, and vegetative structures.
- **DTM (Digital Terrain Model)**: Bare-earth elevation model with filtered structures, contour intervals (1m / 2m), and breaklines.
- Exported as both 32-bit floating-point GeoTIFFs (`dem.tif`, `dsm.tif`, `dtm.tif`) and color-ramp visual analytics (`dem.png`, `dsm.png`, `dtm.png`).

### 6. 💻 Interactive Web Dashboard & Real-Time Upload
- Modern UI powered by Leaflet.js, Alpine.js, and TailwindCSS.
- Live drone image upload with instantaneous processing pipeline.
- Interactive multi-layer toggles (Orthophoto, Buildings, Roads, Land Use, Boundaries, DEM, DSM, DTM).
- One-click GeoJSON and raster downloads.

---

## 📂 Project Structure

```
DrustiX/
├── app/                        # FastAPI Backend & Web Application
│   ├── api.py                  # REST API endpoints (/predict, /upload, /layers, /elevation)
│   └── static/
│       └── index.html          # Interactive Web Dashboard with Leaflet & DEM/DSM/DTM
├── data/                       # Spatial Data Repository
│   ├── raw/                    # Raw aerial images & orthophotos
│   ├── processed/              # Normalized tiles & intermediate rasters
│   ├── outputs/                # Generated GeoJSON vectors, elevation models & overlays
│   └── sit_tumkur/             # Real-world benchmark dataset (SIT Tumkur Campus)
├── models/                     # Deep Learning Model Checkpoints
│   ├── boundary/               # Boundary detection weights
│   ├── building/               # Building extraction weights
│   ├── landuse/                # Land-use multiclass weights
│   └── road/                   # Road network weights
├── scripts/                    # Training, Evaluation & Benchmarking
│   ├── camera_pipeline.py      # Camera/Drone feed ingestion
│   ├── evaluate_models.py      # Quantitative model validation
│   ├── generate_dummy_data.py  # Synthetic training data generator
│   └── render_sit_annotated_map.py # Campus annotation renderer
├── src/                        # Core Python Engine
│   ├── boundary_extraction/    # Boundary detection & fusion logic
│   ├── building_extraction/    # Building model & polygon postprocessor
│   ├── elevation/              # DEM, DSM, DTM & contour generator
│   ├── evaluation/             # IoU, F1, Precision, Recall metrics
│   ├── geospatial/             # RasterIO, GeoPandas, Affine coordinate transforms
│   ├── inference/              # End-to-end integrated inference pipeline
│   ├── landuse_classification/ # Multiclass segmentation module
│   ├── preprocessing/          # Sliding-window tiler & normalizer
│   ├── road_extraction/        # Road centerline & vectorizer
│   └── utils/                  # Visualizer, Logger & Config loader
├── tests/                      # Pytest Unit & Integration Test Suite
├── config.yaml                 # System & Model Hyperparameters
├── DATA_CONTRACT.md            # GeoJSON Schema & CRS Data Contract
├── demo.py                     # CLI End-to-end Demonstration
├── requirements.txt            # Python Dependencies
├── run_pipeline.py             # Pipeline execution entrypoint
└── upload_and_generate.py      # Programmatic upload & elevation test script
```

---

## ⚡ Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Vivekmatade/DrishtiX-SIH-Frontend.git
cd DrishtiX-SIH-Frontend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard

Start the FastAPI application server:

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`** to access the live dashboard:
- Upload any drone orthophoto (TIFF, PNG, JPG).
- View instant extraction of buildings, roads, land use, and boundaries.
- Inspect 3D photogrammetric elevation models (DEM, DSM, DTM).
- Download exported GeoJSON layers.

### 3. Run Pipeline via CLI

Run the full inference pipeline on a sample or orthophoto:

```bash
python run_pipeline.py --input data/sit_tumkur/orthophoto.tif --output data/outputs/
```

Or run the self-contained demonstration:

```bash
python demo.py
```

### 4. Run Automated Tests

Execute the unit test suite:

```bash
pytest tests/ -v
```

---

## 📋 Data Contract & Standards

All vector outputs strictly adhere to the OGC GeoJSON format in **WGS84 (EPSG:4326)** with real-world spatial coordinates:

- **`buildings.geojson`**: `Polygon` / `MultiPolygon` with properties `id`, `class`, `confidence`, `area_m2`.
- **`roads.geojson`**: `LineString` / `MultiLineString` with properties `id`, `class`, `confidence`, `length_m`.
- **`landuse.geojson`**: `Polygon` / `MultiPolygon` with properties `id`, `class`, `confidence`, `area_m2`.
- **`boundary_indicators.geojson`**: `LineString` with properties `id`, `type`, `confidence`, `length_m`.

See [`DATA_CONTRACT.md`](./DATA_CONTRACT.md) for full schema specifications.

---

## 👥 Authors & Team

Developed for **Smart India Hackathon (SIH)**.
- **Team DrishtiX**
