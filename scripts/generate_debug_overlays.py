import os
import sys
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Ensure workspace root in path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

def main():
    ortho_path = os.path.join(workspace_dir, "data", "synthetic", "test", "scene_001", "orthophoto.tif")
    outputs_dir = os.path.join(workspace_dir, "data", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    buildings_path = os.path.join(outputs_dir, "buildings.geojson")
    roads_path = os.path.join(outputs_dir, "roads.geojson")
    landuse_path = os.path.join(outputs_dir, "landuse.geojson")
    boundaries_path = os.path.join(outputs_dir, "boundary_indicators.geojson")

    # Read Orthophoto
    with rasterio.open(ortho_path) as src:
        image_rgb = src.read().transpose(1, 2, 0)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        crs = src.crs

    print("Raster Extent:", extent)

    # Read GeoJSONs
    gdf_b = gpd.read_file(buildings_path) if os.path.exists(buildings_path) else None
    gdf_r = gpd.read_file(roads_path) if os.path.exists(roads_path) else None
    gdf_lu = gpd.read_file(landuse_path) if os.path.exists(landuse_path) else None
    gdf_bd = gpd.read_file(boundaries_path) if os.path.exists(boundaries_path) else None

    def _plot_base(ax, title):
        ax.imshow(image_rgb, extent=extent, origin="upper")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")

    # 1. Debug Orthophoto
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_base(ax, "Debug 1: Drone Orthophoto (EPSG:32643)")
    out_ortho = os.path.join(outputs_dir, "debug_orthophoto.png")
    plt.tight_layout()
    plt.savefig(out_ortho, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {out_ortho}")

    # 2. Debug Buildings
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_base(ax, f"Debug 2: Extracted Buildings ({len(gdf_b)} Footprints)")
    if gdf_b is not None and len(gdf_b) > 0:
        gdf_b.plot(ax=ax, facecolor="#FF1744", edgecolor="#D50000", linewidth=2.0, alpha=0.5)
    b_patch = Patch(facecolor="#FF1744", edgecolor="#D50000", alpha=0.5, label=f"Buildings ({len(gdf_b)})")
    ax.legend(handles=[b_patch], loc="lower right", framealpha=0.9, facecolor="white", fontsize=11)
    out_b = os.path.join(outputs_dir, "debug_buildings.png")
    plt.tight_layout()
    plt.savefig(out_b, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {out_b}")

    # 3. Debug Roads
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_base(ax, f"Debug 3: Extracted Road Features ({len(gdf_r)} Features)")
    if gdf_r is not None and len(gdf_r) > 0:
        for _, row in gdf_r.iterrows():
            geom = row.geometry
            if geom.geom_type in ["LineString", "MultiLineString"]:
                ax.plot(*geom.xy if geom.geom_type == "LineString" else geom.geoms[0].xy,
                        color="#FFD700", linewidth=3.0, alpha=0.95)
            elif geom.geom_type in ["Polygon", "MultiPolygon"]:
                gdf_r.loc[[_]].plot(ax=ax, facecolor="#FFD700", edgecolor="#FFA000", linewidth=2.0, alpha=0.45)
    r_line = Line2D([0], [0], color="#FFD700", lw=3.0, label=f"Roads & Corridors ({len(gdf_r)})")
    ax.legend(handles=[r_line], loc="lower right", framealpha=0.9, facecolor="white", fontsize=11)
    out_r = os.path.join(outputs_dir, "debug_roads.png")
    plt.tight_layout()
    plt.savefig(out_r, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {out_r}")

    # 4. Debug Landuse
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_base(ax, f"Debug 4: Extracted Land-Use Zones ({len(gdf_lu)} Regions)")
    if gdf_lu is not None and len(gdf_lu) > 0:
        gdf_lu.plot(ax=ax, facecolor="#00E5FF", edgecolor="#00B0FF", linewidth=1.5, alpha=0.25)
    lu_patch = Patch(facecolor="#00E5FF", edgecolor="#00B0FF", alpha=0.3, label=f"Land-Use Zones ({len(gdf_lu)})")
    ax.legend(handles=[lu_patch], loc="lower right", framealpha=0.9, facecolor="white", fontsize=11)
    out_lu = os.path.join(outputs_dir, "debug_landuse.png")
    plt.tight_layout()
    plt.savefig(out_lu, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {out_lu}")

    # 5. Debug Boundaries
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_base(ax, f"Debug 5: Extracted Boundary Indicators ({len(gdf_bd)} Lines)")
    if gdf_bd is not None and len(gdf_bd) > 0:
        for _, row in gdf_bd.iterrows():
            geom = row.geometry
            if geom.geom_type == "LineString":
                ax.plot(*geom.xy, color="#E040FB", linestyle="--", linewidth=2.0, alpha=0.95)
            elif geom.geom_type == "MultiLineString":
                for sub in geom.geoms:
                    ax.plot(*sub.xy, color="#E040FB", linestyle="--", linewidth=2.0, alpha=0.95)
    bd_line = Line2D([0], [0], color="#E040FB", lw=2.0, linestyle="--", label=f"Boundary Indicators ({len(gdf_bd)})")
    ax.legend(handles=[bd_line], loc="lower right", framealpha=0.9, facecolor="white", fontsize=11)
    out_bd = os.path.join(outputs_dir, "debug_boundaries.png")
    plt.tight_layout()
    plt.savefig(out_bd, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {out_bd}")

    # 6. Combined all_features_overlay.png
    fig, ax = plt.subplots(figsize=(12, 12))
    _plot_base(ax, "AI Feature Extraction: Integrated Cadastral Overlay")

    # Layer 1: Land-use
    if gdf_lu is not None and len(gdf_lu) > 0:
        gdf_lu.plot(ax=ax, facecolor="#00E5FF", edgecolor="#00B0FF", linewidth=1.2, alpha=0.22)

    # Layer 2: Roads
    if gdf_r is not None and len(gdf_r) > 0:
        for _, row in gdf_r.iterrows():
            geom = row.geometry
            if geom.geom_type in ["LineString", "MultiLineString"]:
                lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
                for l in lines:
                    ax.plot(*l.xy, color="#FFD700", linewidth=3.0, alpha=0.95)
            elif geom.geom_type in ["Polygon", "MultiPolygon"]:
                gdf_r.loc[[_]].plot(ax=ax, facecolor="#FFD700", edgecolor="#FFA000", linewidth=1.5, alpha=0.40)

    # Layer 3: Boundary Indicators
    if gdf_bd is not None and len(gdf_bd) > 0:
        for _, row in gdf_bd.iterrows():
            geom = row.geometry
            lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
            for l in lines:
                ax.plot(*l.xy, color="#E040FB", linestyle="--", linewidth=2.0, alpha=0.95)

    # Layer 4: Buildings
    if gdf_b is not None and len(gdf_b) > 0:
        gdf_b.plot(ax=ax, facecolor="#FF1744", edgecolor="#D50000", linewidth=2.2, alpha=0.50)

    # Final axis bounds
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    legend_elements = [
        Patch(facecolor="#FF1744", edgecolor="#D50000", alpha=0.5, label=f"Buildings ({len(gdf_b)})"),
        Line2D([0], [0], color="#FFD700", lw=3.0, label=f"Roads & Corridors ({len(gdf_r)})"),
        Line2D([0], [0], color="#E040FB", lw=2.0, linestyle="--", label=f"Boundary Indicators ({len(gdf_bd)})"),
        Patch(facecolor="#00E5FF", edgecolor="#00B0FF", alpha=0.25, label=f"Land-Use Zones ({len(gdf_lu)})")
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.92, facecolor="white", edgecolor="#CCCCCC", fontsize=11)

    out_combined = os.path.join(outputs_dir, "all_features_overlay.png")
    plt.tight_layout()
    plt.savefig(out_combined, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {out_combined}")
    print("[+] All debug and combined overlays generated successfully!")

if __name__ == "__main__":
    main()
