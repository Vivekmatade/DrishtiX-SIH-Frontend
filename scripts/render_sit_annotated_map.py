import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import geopandas as gpd
import rasterio
import pyproj
from rasterio.transform import Affine

def render_annotated_sit_map():
    print("=" * 65)
    print("  RENDERING ANNOTATED SIT TUMKUR CADASTRAL MAP WITH DEPARTMENT BLOCKS")
    print("=" * 65)

    ortho_path = "data/sit_tumkur/orthophoto.tif"
    buildings_path = "data/sit_tumkur/buildings.geojson"
    roads_path = "data/sit_tumkur/roads.geojson"
    landuse_path = "data/sit_tumkur/landuse.geojson"
    boundaries_path = "data/sit_tumkur/boundary_indicators.geojson"
    output_png = "data/sit_tumkur/sit_tumkur_all_features_annotated.png"

    # Read orthophoto
    with rasterio.open(ortho_path) as src:
        image = src.read()[:3].transpose(1, 2, 0)
        transform = src.transform
        crs = src.crs
        inv_transform = ~transform
        bounds = src.bounds
        W, H = src.width, src.height

    # Extent for imshow in metric UTM coordinates
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16), dpi=150)
    ax.imshow(image, extent=extent, origin="upper")

    # Load and plot vector layers
    if os.path.exists(landuse_path):
        try:
            gdf_lu = gpd.read_file(landuse_path)
            if not gdf_lu.empty:
                gdf_lu.plot(ax=ax, facecolor="#06b6d4", edgecolor="#0891b2", alpha=0.25, linewidth=0.8)
        except Exception as e:
            print(f"Warning plotting landuse: {e}")

    if os.path.exists(buildings_path):
        try:
            gdf_b = gpd.read_file(buildings_path)
            if not gdf_b.empty:
                gdf_b.plot(ax=ax, facecolor="#ef4444", edgecolor="#b91c1c", alpha=0.55, linewidth=1.2)
        except Exception as e:
            print(f"Warning plotting buildings: {e}")

    if os.path.exists(roads_path):
        try:
            gdf_r = gpd.read_file(roads_path)
            if not gdf_r.empty:
                gdf_r.plot(ax=ax, color="#eab308", linewidth=2.8, alpha=0.9)
        except Exception as e:
            print(f"Warning plotting roads: {e}")

    if os.path.exists(boundaries_path):
        try:
            gdf_bd = gpd.read_file(boundaries_path)
            if not gdf_bd.empty:
                gdf_bd.plot(ax=ax, color="#d946ef", linewidth=2.2, linestyle="--", alpha=0.85)
        except Exception as e:
            print(f"Warning plotting boundaries: {e}")

    # Transform coordinates from WGS84 to UTM
    trans = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # Specific SIT Tumkur Departmental Blocks and Landmarks
    landmarks = [
        {
            "name": "AD BLOCK\n(Administrative)",
            "lat": 13.326913,
            "lon": 77.126183,
            "color": "#1d4ed8",  # Royal Blue
            "dx": -75, "dy": -50,
            "category": "Admin"
        },
        {
            "name": "CS BLOCK\n(Computer Science)",
            "lat": 13.326097,
            "lon": 77.126531,
            "color": "#0284c7",  # Sky Blue / Cyan
            "dx": 45, "dy": 50,
            "category": "Dept"
        },
        {
            "name": "GOLDEN JUBILEE\n(CS / IS Labs)",
            "lat": 13.326357,
            "lon": 77.125299,
            "color": "#0369a1",  # Deep Sky Blue
            "dx": -65, "dy": 45,
            "category": "Dept"
        },
        {
            "name": "CIVIL BLOCK\n(Civil Engg)",
            "lat": 13.326457,
            "lon": 77.125698,
            "color": "#059669",  # Emerald Green
            "dx": 40, "dy": -55,
            "category": "Dept"
        },
        {
            "name": "EC BLOCK\n(Electronics & Comm)",
            "lat": 13.325929,
            "lon": 77.124800,
            "color": "#ea580c",  # Vivid Orange
            "dx": -65, "dy": 35,
            "category": "Dept"
        },
        {
            "name": "MECH BLOCK\n& WORKSHOPS",
            "lat": 13.326171,
            "lon": 77.124462,
            "color": "#dc2626",  # Crimson Red
            "dx": -85, "dy": -30,
            "category": "Dept"
        },
        {
            "name": "ELECTRICAL BLOCK\n(EEE)",
            "lat": 13.326922,
            "lon": 77.124488,
            "color": "#ca8a04",  # Golden Yellow
            "dx": -75, "dy": 35,
            "category": "Dept"
        },
        {
            "name": "KC CENTRAL\nLIBRARY",
            "lat": 13.327214,
            "lon": 77.125315,
            "color": "#7c3aed",  # Purple
            "dx": -60, "dy": 45,
            "category": "Facility"
        },
        {
            "name": "BIRLA AUDITORIUM\n(Silver Jubilee)",
            "lat": 13.326841,
            "lon": 77.126619,
            "color": "#9333ea",  # Purple
            "dx": 75, "dy": 30,
            "category": "Facility"
        },
        {
            "name": "BIOTECH BLOCK",
            "lat": 13.327424,
            "lon": 77.124461,
            "color": "#0d9488",  # Teal
            "dx": -60, "dy": -35,
            "category": "Dept"
        },
        {
            "name": "CHEMISTRY BLOCK",
            "lat": 13.326357,
            "lon": 77.126398,
            "color": "#475569",  # Slate
            "dx": 65, "dy": -40,
            "category": "Dept"
        },
        {
            "name": "CAMPUS CANTEEN",
            "lat": 13.325887,
            "lon": 77.126988,
            "color": "#b45309",  # Amber
            "dx": 75, "dy": -25,
            "category": "Facility"
        },
        {
            "name": "MAIN ENTRANCE\n(B.H. Road Gate)",
            "lat": 13.324850,
            "lon": 77.125600,
            "color": "#15803d",  # Forest Green
            "dx": 45, "dy": 35,
            "category": "Gate"
        }
    ]

    # Draw callouts on map
    for item in landmarks:
        utm_x, utm_y = trans.transform(item["lon"], item["lat"])

        # Check if point falls within bounds
        if not (bounds.left <= utm_x <= bounds.right and bounds.bottom <= utm_y <= bounds.top):
            continue

        # Draw pinpoint dot on the actual building
        ax.plot(utm_x, utm_y, marker="o", markersize=7, color=item["color"], markeredgecolor="white", markeredgewidth=1.5, zorder=10)

        # Draw annotation box with connecting arrow
        text_x = utm_x + item["dx"]
        text_y = utm_y + item["dy"]

        ax.annotate(
            item["name"],
            xy=(utm_x, utm_y),
            xytext=(text_x, text_y),
            arrowprops=dict(
                arrowstyle="->",
                color="white",
                lw=1.8,
                shrinkA=3,
                shrinkB=3,
                connectionstyle="arc3,rad=0.1"
            ),
            fontsize=8.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=12,
            bbox=dict(
                boxstyle="round,pad=0.45,rounding_size=0.3",
                facecolor=item["color"],
                edgecolor="white",
                linewidth=1.2,
                alpha=0.92
            )
        )

    # Lock view coordinates exactly to orthophoto boundaries
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_axis_off()

    # Title Banner
    fig.text(
        0.5, 0.965,
        "SIT Tumkur - AI Cadastral Map with Departmental Blocks",
        ha="center", va="top",
        fontsize=18, fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cbd5e1", alpha=0.95)
    )

    fig.text(
        0.5, 0.940,
        "Siddaganga Institute of Technology, B.H. Road, Tumakuru, Karnataka | Projected CRS: EPSG:32643 (UTM Zone 43N)",
        ha="center", va="top",
        fontsize=10.5, color="#334155"
    )

    # Legend in bottom-right corner
    legend_elements = [
        mpatches.Patch(facecolor="#ef4444", edgecolor="#b91c1c", alpha=0.7, label="Building Footprints (AI Detected)"),
        Line2D([0], [0], color="#eab308", lw=3, label="Roads & Corridors (RoadNet)"),
        Line2D([0], [0], color="#d946ef", lw=2.2, linestyle="--", label="Boundary Indicators (Walls/Demarcations)"),
        mpatches.Patch(facecolor="#06b6d4", edgecolor="#0891b2", alpha=0.4, label="Land-Use Zones (Institutional / Open)"),
        mpatches.Patch(facecolor="#1d4ed8", edgecolor="white", label="Admin / Central Academic Blocks"),
        mpatches.Patch(facecolor="#0284c7", edgecolor="white", label="CS Block / Golden Jubilee (CSE/ISE)"),
        mpatches.Patch(facecolor="#ea580c", edgecolor="white", label="EC Block (Electronics & Comm)"),
        mpatches.Patch(facecolor="#059669", edgecolor="white", label="Civil Engineering Block"),
        mpatches.Patch(facecolor="#dc2626", edgecolor="white", label="Mechanical Engineering & Workshops"),
        mpatches.Patch(facecolor="#7c3aed", edgecolor="white", label="Auditorium & Central Library")
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#94a3b8",
        title="Cadastral & Campus Legend",
        title_fontsize=10.5
    )
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_png, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"[+] Successfully generated annotated map: {output_png}")

    # Copy to uploaded_run and brain artifacts for web access
    import shutil
    shutil.copyfile(output_png, "data/outputs/uploaded_run/all_features_overlay.png")
    shutil.copyfile(output_png, "data/outputs/uploaded_run/sit_tumkur_all_features_annotated.png")
    
    artifact_path = r"C:\Users\Admin\.gemini\antigravity\brain\eec5b2d0-8cfb-46c5-8cbb-1c5f94262172\sit_tumkur_annotated_blocks.png"
    shutil.copyfile(output_png, artifact_path)
    print(f"[+] Copied to artifact directory: {artifact_path}")

if __name__ == "__main__":
    render_annotated_sit_map()
