import os
import json
import geopandas as gpd

datasets = [
    ("SIT Tumkur Campus Map", "data/sit_tumkur"),
    ("Current Web / Uploaded Run", "data/outputs/uploaded_run"),
    ("Default Pipeline Output", "data/outputs")
]

for title, folder in datasets:
    print("=" * 65)
    print(f"  {title.upper()} ({folder})")
    print("=" * 65)

    # 1. Buildings
    b_path = os.path.join(folder, "buildings.geojson")
    if os.path.exists(b_path):
        b_gdf = gpd.read_file(b_path)
        total_b = len(b_gdf)
        total_b_area = b_gdf["area_m2"].sum() if "area_m2" in b_gdf.columns else b_gdf.geometry.area.sum()
        avg_b_area = b_gdf["area_m2"].mean() if "area_m2" in b_gdf.columns else b_gdf.geometry.area.mean()
        print(f"1. BUILDINGS:")
        print(f"   - Total Count:        {total_b} buildings detected")
        print(f"   - Total Footprint:    {total_b_area:,.2f} m2 ({total_b_area/10000:,.2f} hectares)")
        print(f"   - Average Size:       {avg_b_area:,.2f} m2 per building")
    else:
        print("1. BUILDINGS: Not found")

    # 2. Roads
    r_path = os.path.join(folder, "roads.geojson")
    if os.path.exists(r_path):
        r_gdf = gpd.read_file(r_path)
        total_r = len(r_gdf)
        total_r_len = r_gdf["length_m"].sum() if "length_m" in r_gdf.columns else r_gdf.geometry.length.sum()
        print(f"\n2. ROADS & CORRIDORS:")
        print(f"   - Total Count:        {total_r} road corridors/segments")
        print(f"   - Total Network:      {total_r_len:,.2f} meters ({total_r_len/1000:,.2f} km)")
    else:
        print("\n2. ROADS & CORRIDORS: Not found")

    # 3. Land Use
    lu_path = os.path.join(folder, "landuse.geojson")
    if os.path.exists(lu_path):
        lu_gdf = gpd.read_file(lu_path)
        total_lu = len(lu_gdf)
        total_lu_area = lu_gdf["area_m2"].sum() if "area_m2" in lu_gdf.columns else lu_gdf.geometry.area.sum()
        print(f"\n3. LAND-USE (LAND USED):")
        print(f"   - Total Zones:        {total_lu} classified zones")
        print(f"   - Total Classified:   {total_lu_area:,.2f} m2 ({total_lu_area/10000:,.2f} hectares)")
        if "class" in lu_gdf.columns:
            print("   - Breakdown by Category:")
            for cat, count in lu_gdf["class"].value_counts().items():
                cat_area = lu_gdf[lu_gdf["class"] == cat]["area_m2"].sum() if "area_m2" in lu_gdf.columns else lu_gdf[lu_gdf["class"] == cat].geometry.area.sum()
                pct = (cat_area / total_lu_area * 100) if total_lu_area > 0 else 0
                print(f"     * {cat:16s}: {count:3d} zones | {cat_area:10,.2f} m2 ({pct:5.1f}%)")
    else:
        print("\n3. LAND-USE: Not found")

    # 4. Boundaries
    bd_path = os.path.join(folder, "boundary_indicators.geojson")
    if os.path.exists(bd_path):
        bd_gdf = gpd.read_file(bd_path)
        total_bd = len(bd_gdf)
        total_bd_len = bd_gdf["length_m"].sum() if "length_m" in bd_gdf.columns else bd_gdf.geometry.length.sum()
        print(f"\n4. BOUNDARY INDICATORS (BOUNDARIES FILLED):")
        print(f"   - Total Count:        {total_bd} boundary indicator lines")
        print(f"   - Total Demarcation:  {total_bd_len:,.2f} meters ({total_bd_len/1000:,.2f} km)")
    else:
        print("\n4. BOUNDARY INDICATORS: Not found")
    print("\n")
