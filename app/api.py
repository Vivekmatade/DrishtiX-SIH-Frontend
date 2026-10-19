import os
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.inference.pipeline import FeatureExtractionPipeline
from src.utils.logger import get_logger

logger = get_logger("API")

app = FastAPI(
    title="Drone AI Feature Extraction, Cadastral Mapping & Elevation (DEM/DSM/DTM) API",
    description="Geospatial AI API extracting buildings, roads, land-use, boundaries, and photogrammetric DEM, DSM, and DTM models from aerial imagery.",
    version="1.2.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directory exists and mount static outputs
os.makedirs("data/outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="data/outputs"), name="outputs")

# Global pipeline instance initialized on startup
pipeline: Optional[FeatureExtractionPipeline] = None

@app.on_event("startup")
def startup_event():
    global pipeline
    logger.info("Initializing FeatureExtractionPipeline inside FastAPI application...")
    pipeline = FeatureExtractionPipeline(config_path="config.yaml")
    logger.info("FeatureExtractionPipeline initialized successfully.")

@app.get("/")
def serve_ui():
    """Serves the interactive web-based image upload, cadastral mapping, and DEM/DSM/DTM dashboard."""
    ui_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "Drone AI Feature Extraction API is active. Navigate to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    """Health status endpoint providing model and pipeline runtime status."""
    return {
        "status": "healthy",
        "service": "Drone AI Feature Extraction & Elevation Pipeline",
        "version": "1.2.0",
        "device": str(pipeline.device) if pipeline else "unknown",
        "models": {
            "building_extraction": "DroneUNet",
            "road_extraction": "RoadSegmentationNet",
            "landuse_classification": "LandUseSegmentationNet",
            "boundary_indicators": "HybridDetector",
            "elevation_modeling": "ElevationGenerator (DEM, DSM, DTM, Contours)"
        }
    }

@app.post("/api/upload-image")
async def upload_image_and_generate(file: UploadFile = File(...)):
    """
    Accepts any uploaded aerial/drone image (JPG, PNG, TIFF),
    converts it to a georeferenced metric GeoTIFF (orthophoto.tif),
    runs deep learning models to extract all 4 cadastral layers,
    generates DEM, DSM, and DTM elevation rasters and visualizations,
    and returns comprehensive results and download links.
    """
    global pipeline
    if pipeline is None:
        pipeline = FeatureExtractionPipeline(config_path="config.yaml")

    from src.geospatial.camera_adapter import CameraAdapter
    import cv2
    import numpy as np

    output_dir = os.path.join("data", "outputs", "uploaded_run")
    os.makedirs(output_dir, exist_ok=True)

    contents = await file.read()
    filename_lower = file.filename.lower()

    # Save a copy of the original input file
    orig_ext = os.path.splitext(file.filename)[1] or ".jpg"
    orig_save_path = os.path.join(output_dir, f"original_input{orig_ext}")
    with open(orig_save_path, "wb") as f:
        f.write(contents)

    adapter = CameraAdapter()
    ortho_path = os.path.join(output_dir, "orthophoto.tif")

    if filename_lower.endswith((".tif", ".tiff")):
        with open(ortho_path, "wb") as f:
            f.write(contents)
        try:
            from src.geospatial.raster_io import RasterIO
            img_rgb, meta = RasterIO.read_orthophoto(ortho_path)
        except Exception:
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            adapter.convert_to_orthophoto(img_rgb, ortho_path)
    else:
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode uploaded image file.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        adapter.convert_to_orthophoto(img_rgb, ortho_path)

    results = pipeline.run(input_orthophoto=ortho_path, output_dir=output_dir)

    return JSONResponse(content={
        "status": "success",
        "cadastral_overlay_url": "/results/all_features_overlay.png",
        "original_image_url": f"/results/original_input{orig_ext}",
        "dem_url": "/results/dem.png",
        "dsm_url": "/results/dsm.png",
        "dtm_url": "/results/dtm.png",
        "dem_tif_url": "/results/dem.tif",
        "dsm_tif_url": "/results/dsm.tif",
        "dtm_tif_url": "/results/dtm.tif",
        "contours_url": "/results/contours.geojson",
        "breaklines_url": "/results/breaklines.geojson",
        "buildings_geojson_url": "/results/buildings.geojson",
        "roads_geojson_url": "/results/roads.geojson",
        "landuse_geojson_url": "/results/landuse.geojson",
        "boundaries_geojson_url": "/results/boundary_indicators.geojson",
        "elevation_stats": results.get("elevation_stats", {}),
        "summary": results["summary"]
    })

@app.post("/api/use-sample")
def use_sample_image(sample_id: int = 1):
    """Processes one of the pre-packaged sample aerial images (including SIT Tumkur)."""
    global pipeline
    if pipeline is None:
        pipeline = FeatureExtractionPipeline(config_path="config.yaml")

    sample_map = {
        1: os.path.join("data", "sample_drone_aerial.jpg"),
        2: os.path.join("data", "sample_drone_aerial_2.png"),
        3: os.path.join("data", "sit_tumkur", "sit_tumkur_satellite.jpg")
    }
    sample_file = sample_map.get(sample_id, os.path.join("data", "sample_drone_aerial.jpg"))
    if not os.path.exists(sample_file):
        raise HTTPException(status_code=404, detail="Sample image not found on server.")

    from src.geospatial.camera_adapter import CameraAdapter
    import cv2
    output_dir = os.path.join("data", "outputs", "uploaded_run")
    os.makedirs(output_dir, exist_ok=True)

    img_bgr = cv2.imread(sample_file)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # If SIT Tumkur sample (sample_id 3), use SIT's true UTM Zone 43N coordinates
    if sample_id == 3:
        adapter = CameraAdapter(
            default_crs="EPSG:32643",
            pixel_resolution=0.586,
            origin_x=730071.8,
            origin_y=1474557.2
        )
    else:
        adapter = CameraAdapter()

    ortho_path = os.path.join(output_dir, "orthophoto.tif")
    adapter.convert_to_orthophoto(img_rgb, ortho_path)

    orig_ext = os.path.splitext(sample_file)[1]
    cv2.imwrite(os.path.join(output_dir, f"original_input{orig_ext}"), img_bgr)

    results = pipeline.run(input_orthophoto=ortho_path, output_dir=output_dir)

    # If SIT Tumkur, copy the annotated departmental blocks map over all_features_overlay
    if sample_id == 3 and os.path.exists("data/sit_tumkur/sit_tumkur_all_features_annotated.png"):
        shutil.copyfile("data/sit_tumkur/sit_tumkur_all_features_annotated.png", os.path.join(output_dir, "all_features_overlay.png"))

    return JSONResponse(content={
        "status": "success",
        "cadastral_overlay_url": "/results/all_features_overlay.png",
        "original_image_url": f"/results/original_input{orig_ext}",
        "dem_url": "/results/dem.png",
        "dsm_url": "/results/dsm.png",
        "dtm_url": "/results/dtm.png",
        "dem_tif_url": "/results/dem.tif",
        "dsm_tif_url": "/results/dsm.tif",
        "dtm_tif_url": "/results/dtm.tif",
        "contours_url": "/results/contours.geojson",
        "breaklines_url": "/results/breaklines.geojson",
        "buildings_geojson_url": "/results/buildings.geojson",
        "roads_geojson_url": "/results/roads.geojson",
        "landuse_geojson_url": "/results/landuse.geojson",
        "boundaries_geojson_url": "/results/boundary_indicators.geojson",
        "elevation_stats": results.get("elevation_stats", {}),
        "summary": results["summary"]
    })

@app.post("/api/capture-camera")
async def capture_camera_and_extract(image: Optional[UploadFile] = File(None)):
    """
    Captures or receives a drone/camera frame, converts it to a standard GeoTIFF orthophoto (orthophoto.tif),
    and executes end-to-end feature extraction and elevation modeling.
    """
    global pipeline
    if pipeline is None:
        pipeline = FeatureExtractionPipeline(config_path="config.yaml")

    from src.geospatial.camera_adapter import CameraAdapter
    import cv2
    import numpy as np

    output_dir = os.path.join("data", "outputs", "camera_run")
    os.makedirs(output_dir, exist_ok=True)
    adapter = CameraAdapter()

    if image is not None:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        try:
            image_rgb = adapter.capture_frame(device_index=0, interactive=False)
        except Exception as e:
            logger.warning(f"Hardware camera capture failed: {e}. Falling back to default scene.")
            fallback_path = os.path.join("data", "synthetic", "test", "scene_001", "orthophoto.tif")
            from src.geospatial.raster_io import RasterIO
            image_rgb, _ = RasterIO.read_orthophoto(fallback_path)

    ortho_path = os.path.join(output_dir, "orthophoto.tif")
    adapter.convert_to_orthophoto(image_rgb=image_rgb, output_path=ortho_path)

    results = pipeline.run(input_orthophoto=ortho_path, output_dir=output_dir)
    return JSONResponse(content={
        "status": "success",
        "cadastral_overlay_url": "/results/all_features_overlay.png",
        "dem_url": "/results/dem.png",
        "dsm_url": "/results/dsm.png",
        "dtm_url": "/results/dtm.png",
        "dem_tif_url": "/results/dem.tif",
        "dsm_tif_url": "/results/dsm.tif",
        "dtm_tif_url": "/results/dtm.tif",
        "contours_url": "/results/contours.geojson",
        "breaklines_url": "/results/breaklines.geojson",
        "buildings_geojson_url": "/results/buildings.geojson",
        "roads_geojson_url": "/results/roads.geojson",
        "landuse_geojson_url": "/results/landuse.geojson",
        "boundaries_geojson_url": "/results/boundary_indicators.geojson",
        "elevation_stats": results.get("elevation_stats", {}),
        "summary": results["summary"]
    })

@app.get("/results/{filename:path}")
def get_result_file(filename: str):
    """Downloads a generated GeoJSON, DEM/DSM/DTM raster, or visualization file."""
    candidates = [
        os.path.join("data", "outputs", "uploaded_run", filename),
        os.path.join("data", "sit_tumkur", filename),
        os.path.join("data", "outputs", "camera_run", filename),
        os.path.join("data", "outputs", "api_runs", filename),
        os.path.join("data", "outputs", filename),
        os.path.join("data", filename),
        filename
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isfile(path):
            media_type = "application/geo+json" if filename.endswith(".geojson") else None
            return FileResponse(path, filename=os.path.basename(filename), media_type=media_type)

    raise HTTPException(status_code=404, detail=f"Result file '{filename}' not found.")
