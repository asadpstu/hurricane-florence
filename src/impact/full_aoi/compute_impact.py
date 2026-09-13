#!/usr/bin/env python3
"""
Compute Florence flood impacts over the ORIGINAL Copernicus Goldsboro AOI.

This workflow is deliberately independent of the NOAA FIM footprint.  The
hazard comes from one or more model-generated flood rasters supplied with
--hazard.  Every hazard must cover the original AOI unless
--allow-partial-hazard-coverage is explicitly supplied.

Exposure outputs
----------------
For each hazard/source the script computes:
  * inundated area (km2)
  * estimated affected WorldPop 2018 population
  * impacted OSM building count
  * flooded OSM road length (km)
  * impacted OSM critical-facility count
  * flooded NLCD area by class (fractional 30 m cell weighting)
  * equivalent impervious flooded area (km2)

It also writes map-ready GeoTIFFs:
  * flood_extent.tif
  * impacted_buildings.tif
  * impacted_roads.tif
  * impacted_critical_facilities.tif
  * impacted_assets_combined.tif
  * flood_fraction_worldpop.tif
  * affected_population_estimate.tif
  * flood_fraction_nlcd.tif
  * impacted_landcover.tif
  * impacted_impervious_percent.tif

Combined asset classes
----------------------
  0   = dry / no impacted asset inside AOI
  1   = flooded
  2   = impacted road
  3   = impacted building
  4   = impacted critical facility
  255 = outside AOI / nodata

Notes
-----
* OSM exposure is a current snapshot unless replaced with a historical extract.
* Affected population uses the fraction of each WorldPop cell covered by the
  model flood raster, rather than simple centre-cell inclusion.
* NLCD class areas and impervious exposure use the same fractional flood
  weighting on the 30 m NLCD grid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize, shapes
from rasterio.warp import Resampling, reproject
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union


NLCD_CLASSES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}


DEFAULT_AOI = Path(
    "output/spatial/goldsboro_florence_reference_aoi/"
    "goldsboro_florence_analysis_aoi.geojson"
)
DEFAULT_INPUT_DIR = Path("input/impact/florence_2018/full_aoi")
DEFAULT_OUT_DIR = Path("output/impact/florence_2018/full_aoi")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute model-based flood exposure over the original Copernicus "
            "Goldsboro AOI. NOAA FIM is not used."
        )
    )
    p.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--hazard",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Model flood raster. Repeat for multiple sources, e.g. "
            "--hazard 'Physics=path.tif' --hazard 'ML=path.tif'. "
            "Flooded cells are values > --flood-threshold."
        ),
    )
    p.add_argument(
        "--flood-threshold",
        type=float,
        default=0.0,
        help="Raster values strictly greater than this are flooded (default 0).",
    )
    p.add_argument(
        "--exclude-value",
        action="append",
        type=float,
        default=[],
        help=(
            "Extra raster value to treat as nodata. Repeat as needed. "
            "Binary 0/1/255 rasters with missing nodata metadata auto-detect 255."
        ),
    )
    p.add_argument(
        "--min-hazard-aoi-coverage",
        type=float,
        default=0.995,
        help="Minimum fraction of original AOI covered by each hazard (default 0.995).",
    )
    p.add_argument(
        "--allow-partial-hazard-coverage",
        action="store_true",
        help=(
            "Allow hazard rasters that cover less than --min-hazard-aoi-coverage. "
            "Use only when partial-domain impacts are intentionally desired."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "source"


def parse_hazards(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --hazard {raw!r}; expected NAME=PATH")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = Path(path.strip())
        if not name or not str(path):
            raise ValueError(f"Invalid --hazard {raw!r}; expected NAME=PATH")
        if name in seen:
            raise ValueError(f"Duplicate hazard source name: {name}")
        seen.add(name)
        out.append((name, path))
    return out


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def read_aoi(path: Path) -> gpd.GeoDataFrame:
    aoi = gpd.read_file(path)
    if aoi.empty:
        raise RuntimeError(f"AOI is empty: {path}")
    if aoi.crs is None:
        raise RuntimeError(f"AOI has no CRS: {path}")
    aoi = aoi[["geometry"]].copy()
    aoi["geometry"] = aoi.geometry.make_valid()
    aoi = aoi[~aoi.geometry.is_empty]
    if aoi.empty:
        raise RuntimeError(f"AOI has no valid geometry: {path}")
    return aoi


def vector_layer_or_empty(path: Path, layer: str, crs: Any) -> gpd.GeoDataFrame:
    try:
        available = set(gpd.list_layers(path)["name"].astype(str))
    except Exception:
        available = set()
    if layer not in available:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    gdf = gpd.read_file(path, layer=layer)
    if gdf.crs is None:
        raise RuntimeError(f"OSM layer {layer!r} has no CRS: {path}")
    return gdf.to_crs(crs)


def write_array(path: Path, array: np.ndarray, profile: dict[str, Any], *, dtype: str, nodata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    p = profile.copy()
    width = int(p["width"])
    height = int(p["height"])
    use_tiles = width >= 16 and height >= 16
    p.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=use_tiles,
        BIGTIFF="IF_SAFER",
    )
    if use_tiles:
        p["blockxsize"] = max(16, min(256, (width // 16) * 16))
        p["blockysize"] = max(16, min(256, (height // 16) * 16))
    else:
        p.pop("blockxsize", None)
        p.pop("blockysize", None)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype(dtype, copy=False), 1)


def pixel_area_m2(transform: rasterio.Affine) -> float:
    return abs(transform.a * transform.e - transform.b * transform.d)


def polygonize_flood(flood: np.ndarray, transform: rasterio.Affine):
    geoms = [
        shape(geom)
        for geom, value in shapes(
            flood.astype(np.uint8),
            mask=flood,
            transform=transform,
            connectivity=8,
        )
        if int(value) == 1
    ]
    if not geoms:
        return None
    geom = unary_union(geoms)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def reproject_fraction(
    flood: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: Any,
    dst_profile: dict[str, Any],
) -> np.ndarray:
    dest = np.zeros(
        (int(dst_profile["height"]), int(dst_profile["width"])),
        dtype=np.float32,
    )
    reproject(
        source=flood.astype(np.float32),
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=0.0,
        resampling=Resampling.average,
    )
    return np.clip(dest, 0.0, 1.0)


def aoi_mask_on_grid(aoi: gpd.GeoDataFrame, crs: Any, height: int, width: int, transform: rasterio.Affine) -> np.ndarray:
    geom = aoi.to_crs(crs).geometry.union_all()
    return geometry_mask(
        [mapping(geom)],
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=False,
    )


def raster_footprint_coverage(aoi: gpd.GeoDataFrame, raster_crs: Any, bounds: rasterio.coords.BoundingBox) -> float:
    aoi_r = aoi.to_crs(raster_crs)
    geom = aoi_r.geometry.union_all()
    footprint = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    if geom.is_empty:
        return 0.0
    # If source CRS is geographic, transform both geometries to UTM 18N for area.
    frame = gpd.GeoDataFrame(
        {"kind": ["aoi", "footprint"]},
        geometry=[geom, footprint],
        crs=raster_crs,
    ).to_crs("EPSG:32618")
    aoi_m = frame.geometry.iloc[0]
    fp_m = frame.geometry.iloc[1]
    if aoi_m.area <= 0:
        return 0.0
    return float(aoi_m.intersection(fp_m).area / aoi_m.area)


def detect_extra_nodata(data: np.ndarray, src_nodata: float | None, user_excludes: list[float]) -> list[float]:
    excludes = list(user_excludes)
    if src_nodata is None:
        sample = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()
        if sample.size:
            unique = np.unique(sample)
            # Common project mask encoding: 0=dry, 1=flood, 255=outside-domain.
            if unique.size <= 4 and 255 in unique and 1 in unique:
                excludes.append(255.0)
    return excludes


def model_flood_from_raster(
    path: Path,
    aoi: gpd.GeoDataFrame,
    threshold: float,
    user_excludes: list[float],
) -> dict[str, Any]:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise RuntimeError(f"Hazard raster has no CRS: {path}")
        data = src.read(1)
        profile = src.profile.copy()
        aoi_mask = aoi_mask_on_grid(aoi, src.crs, src.height, src.width, src.transform)
        valid = np.ones(data.shape, dtype=bool)
        if np.issubdtype(data.dtype, np.floating):
            valid &= np.isfinite(data)
        if src.nodata is not None:
            valid &= data != src.nodata
        excludes = detect_extra_nodata(data, src.nodata, user_excludes)
        for value in excludes:
            valid &= data != value
        flood = aoi_mask & valid & (data > threshold)
        coverage = raster_footprint_coverage(aoi, src.crs, src.bounds)
        aoi_pixels = int(aoi_mask.sum())
        valid_inside = int((aoi_mask & valid).sum())
        valid_pixel_fraction = float(valid_inside / aoi_pixels) if aoi_pixels else 0.0
        flood_area_km2 = float(flood.sum() * pixel_area_m2(src.transform) / 1e6)
        return {
            "data": data,
            "profile": profile,
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
            "aoi_mask": aoi_mask,
            "valid": valid,
            "flood": flood,
            "coverage": coverage,
            "valid_pixel_fraction": valid_pixel_fraction,
            "flood_area_km2": flood_area_km2,
            "extra_nodata": excludes,
        }


def vector_impacts(
    flood_geom,
    hazard_crs: Any,
    osm_path: Path,
) -> tuple[dict[str, Any], dict[str, gpd.GeoDataFrame]]:
    buildings = vector_layer_or_empty(osm_path, "buildings", hazard_crs)
    roads = vector_layer_or_empty(osm_path, "roads", hazard_crs)
    critical = vector_layer_or_empty(osm_path, "critical_facilities", hazard_crs)

    if flood_geom is None:
        empty_b = buildings.iloc[0:0].copy()
        empty_r = roads.iloc[0:0].copy()
        empty_c = critical.iloc[0:0].copy()
        return {
            "flooded_buildings": 0,
            "flooded_road_length_km": 0.0,
            "affected_critical_facilities": 0,
        }, {"buildings": empty_b, "roads": empty_r, "critical": empty_c}

    bmask = buildings.geometry.intersects(flood_geom) if not buildings.empty else np.array([], dtype=bool)
    rmask = roads.geometry.intersects(flood_geom) if not roads.empty else np.array([], dtype=bool)
    cmask = critical.geometry.intersects(flood_geom) if not critical.empty else np.array([], dtype=bool)

    b = buildings.loc[bmask].copy() if len(bmask) else buildings.iloc[0:0].copy()
    r = roads.loc[rmask].copy() if len(rmask) else roads.iloc[0:0].copy()
    c = critical.loc[cmask].copy() if len(cmask) else critical.iloc[0:0].copy()

    road_length_km = 0.0
    if not r.empty:
        road_length_km = float(r.geometry.intersection(flood_geom).length.sum() / 1000.0)

    return {
        "flooded_buildings": int(len(b)),
        "flooded_road_length_km": road_length_km,
        "affected_critical_facilities": int(len(c)),
    }, {"buildings": b, "roads": r, "critical": c}



def write_impacted_roads_geojson(
    roads: gpd.GeoDataFrame,
    flood_geom,
    out_path: Path,
    source_name: str,
) -> tuple[int, float]:
    """Write exact flood-clipped OSM road segments for browser/QGIS rendering.

    The impact summary already measures road length using
    ``roads.geometry.intersection(flood_geom)``.  This helper persists that same
    clipped geometry as EPSG:4326 GeoJSON, so the dashboard can draw true line
    features rather than a faint rasterized road mask.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write_empty() -> tuple[int, float]:
        out_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": []}, indent=2),
            encoding="utf-8",
        )
        return 0, 0.0

    if flood_geom is None or roads.empty:
        return write_empty()

    clipped = roads.copy()
    clipped["geometry"] = clipped.geometry.intersection(flood_geom)
    nonempty = clipped.geometry.map(lambda geom: geom is not None and not geom.is_empty)
    clipped = clipped[nonempty].copy()
    if clipped.empty:
        return write_empty()

    # Intersection can return MultiLineString/GeometryCollection. Explode first,
    # then retain only line geometry that represents an actual road segment.
    clipped = clipped.explode(index_parts=False, ignore_index=True)
    clipped = clipped[
        clipped.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ].copy()
    if clipped.empty:
        return write_empty()

    clipped["segment_length_km"] = clipped.geometry.length.astype(float) / 1000.0
    clipped["source"] = source_name

    # Keep useful road attributes without shipping the full OSM tag JSON to the
    # browser.  Missing columns are simply omitted.
    keep = [
        "source",
        "osm_type",
        "osm_id",
        "name",
        "highway",
        "surface",
        "lanes",
        "bridge",
        "tunnel",
        "segment_length_km",
        "geometry",
    ]
    keep = [c for c in keep if c in clipped.columns]
    clipped = clipped[keep].copy()

    total_km = float(clipped["segment_length_km"].sum())
    clipped = clipped.to_crs("EPSG:4326")
    clipped.to_file(out_path, driver="GeoJSON")
    return int(len(clipped)), total_km


def rasterize_impacted_assets(
    flood: np.ndarray,
    aoi_mask: np.ndarray,
    profile: dict[str, Any],
    impacted: dict[str, gpd.GeoDataFrame],
) -> dict[str, np.ndarray]:
    h = int(profile["height"])
    w = int(profile["width"])
    transform = profile["transform"]

    def burn(gdf: gpd.GeoDataFrame) -> np.ndarray:
        if gdf.empty:
            return np.zeros((h, w), dtype=np.uint8)
        geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros((h, w), dtype=np.uint8)
        return rasterize(
            [(mapping(g), 1) for g in geoms],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )

    roads = burn(impacted["roads"]) & flood.astype(np.uint8)
    buildings = burn(impacted["buildings"]) & flood.astype(np.uint8)
    critical = burn(impacted["critical"]) & flood.astype(np.uint8)

    flood_out = np.full((h, w), 255, dtype=np.uint8)
    flood_out[aoi_mask] = 0
    flood_out[flood] = 1

    combined = flood_out.copy()
    combined[roads == 1] = 2
    combined[buildings == 1] = 3
    combined[critical == 1] = 4

    def mask_out(arr: np.ndarray) -> np.ndarray:
        out = np.full((h, w), 255, dtype=np.uint8)
        out[aoi_mask] = arr[aoi_mask]
        return out

    return {
        "flood": flood_out,
        "buildings": mask_out(buildings),
        "roads": mask_out(roads),
        "critical": mask_out(critical),
        "combined": combined,
    }


def population_impacts(
    flood: np.ndarray,
    hazard_profile: dict[str, Any],
    aoi: gpd.GeoDataFrame,
    worldpop_path: Path,
    source_dir: Path,
) -> float:
    with rasterio.open(worldpop_path) as src:
        profile = src.profile.copy()
        pop = src.read(1).astype(np.float64)
        flood_fraction = reproject_fraction(
            flood,
            hazard_profile["transform"],
            hazard_profile["crs"],
            profile,
        )
        aoi_mask = aoi_mask_on_grid(aoi, src.crs, src.height, src.width, src.transform)
        valid = aoi_mask & np.isfinite(pop)
        if src.nodata is not None:
            valid &= pop != src.nodata
        valid &= pop >= 0
        affected = np.zeros(pop.shape, dtype=np.float32)
        affected[valid] = (pop[valid] * flood_fraction[valid]).astype(np.float32)
        total = float(affected[valid].sum())

        fraction_out = np.full(pop.shape, -9999.0, dtype=np.float32)
        fraction_out[aoi_mask] = flood_fraction[aoi_mask]
        affected_out = np.full(pop.shape, -9999.0, dtype=np.float32)
        affected_out[aoi_mask] = affected[aoi_mask]

        write_array(
            source_dir / "flood_fraction_worldpop.tif",
            fraction_out,
            profile,
            dtype="float32",
            nodata=-9999.0,
        )
        write_array(
            source_dir / "affected_population_estimate.tif",
            affected_out,
            profile,
            dtype="float32",
            nodata=-9999.0,
        )
        return total


def nlcd_impacts(
    flood: np.ndarray,
    hazard_profile: dict[str, Any],
    aoi: gpd.GeoDataFrame,
    landcover_path: Path,
    impervious_path: Path,
    source_dir: Path,
) -> tuple[dict[int, float], float]:
    with rasterio.open(landcover_path) as lc_src:
        lc_profile = lc_src.profile.copy()
        lc = lc_src.read(1)
        flood_fraction = reproject_fraction(
            flood,
            hazard_profile["transform"],
            hazard_profile["crs"],
            lc_profile,
        )
        aoi_mask = aoi_mask_on_grid(aoi, lc_src.crs, lc_src.height, lc_src.width, lc_src.transform)
        valid_lc = aoi_mask.copy()
        if lc_src.nodata is not None:
            valid_lc &= lc != lc_src.nodata
        valid_lc &= lc != 0
        pixel_km2 = pixel_area_m2(lc_src.transform) / 1e6

        class_areas: dict[int, float] = {}
        for code in np.unique(lc[valid_lc]):
            code_i = int(code)
            sel = valid_lc & (lc == code_i)
            area = float(np.sum(flood_fraction[sel]) * pixel_km2)
            if area > 0:
                class_areas[code_i] = area

        fraction_out = np.full(lc.shape, -9999.0, dtype=np.float32)
        fraction_out[aoi_mask] = flood_fraction[aoi_mask]
        impacted_lc = np.zeros(lc.shape, dtype=lc.dtype)
        impacted_lc[valid_lc & (flood_fraction > 0)] = lc[valid_lc & (flood_fraction > 0)]

        write_array(
            source_dir / "flood_fraction_nlcd.tif",
            fraction_out,
            lc_profile,
            dtype="float32",
            nodata=-9999.0,
        )
        lc_nodata = 0
        write_array(
            source_dir / "impacted_landcover.tif",
            impacted_lc,
            lc_profile,
            dtype=str(lc.dtype),
            nodata=lc_nodata,
        )

    with rasterio.open(impervious_path) as imp_src:
        imp_profile = imp_src.profile.copy()
        imp = imp_src.read(1).astype(np.float64)
        imp_fraction = reproject_fraction(
            flood,
            hazard_profile["transform"],
            hazard_profile["crs"],
            imp_profile,
        )
        imp_aoi = aoi_mask_on_grid(aoi, imp_src.crs, imp_src.height, imp_src.width, imp_src.transform)
        valid_imp = imp_aoi & np.isfinite(imp)
        if imp_src.nodata is not None:
            valid_imp &= imp != imp_src.nodata
        imp = np.clip(imp, 0.0, 100.0)
        imp_pixel_km2 = pixel_area_m2(imp_src.transform) / 1e6
        equiv_km2 = float(
            np.sum((imp[valid_imp] / 100.0) * imp_fraction[valid_imp]) * imp_pixel_km2
        )

        impacted_imp = np.full(imp.shape, 255, dtype=np.uint8)
        flooded_valid = valid_imp & (imp_fraction > 0)
        impacted_imp[flooded_valid] = np.rint(imp[flooded_valid]).astype(np.uint8)
        write_array(
            source_dir / "impacted_impervious_percent.tif",
            impacted_imp,
            imp_profile,
            dtype="uint8",
            nodata=255,
        )

    return class_areas, equiv_km2


def main() -> int:
    a = parse_args()
    hazards = parse_hazards(a.hazard)

    require(a.aoi)
    for _, path in hazards:
        require(path)

    worldpop = a.input_dir / "worldpop_usa_2018_population_100m.tif"
    nlcd_lc = a.input_dir / "nlcd_2016_landcover_30m.tif"
    nlcd_imp = a.input_dir / "nlcd_2016_impervious_30m.tif"
    osm = a.input_dir / "osm_current_exposure.gpkg"
    for p in [worldpop, nlcd_lc, nlcd_imp, osm]:
        require(p)

    if a.out_dir.exists() and any(a.out_dir.iterdir()) and not a.overwrite:
        raise FileExistsError(f"{a.out_dir} is not empty. Use --overwrite.")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.overwrite:
        for child in a.out_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                import shutil
                shutil.rmtree(child)

    aoi = read_aoi(a.aoi)
    aoi_utm = aoi.to_crs("EPSG:32618")
    aoi_area_km2 = float(aoi_utm.geometry.union_all().area / 1e6)

    print("=" * 110)
    print("FLORENCE 2018 — ORIGINAL-AOI MODEL IMPACT ASSESSMENT")
    print("=" * 110)
    print(f"AOI                 : {a.aoi}")
    print(f"AOI area            : {aoi_area_km2:.3f} km²")
    print(f"Exposure input dir  : {a.input_dir}")
    print(f"Hazard sources      : {len(hazards)}")

    summary_rows: list[dict[str, Any]] = []
    lc_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "status": "PASS_ORIGINAL_AOI_IMPACT_COMPLETE",
        "aoi": str(a.aoi),
        "aoi_area_km2": aoi_area_km2,
        "exposure_input_dir": str(a.input_dir),
        "hazards": {},
        "warning": (
            "OSM exposure is current at acquisition time unless the input GeoPackage "
            "was replaced with a historical September 2018 extract."
        ),
    }

    for source_name, hazard_path in hazards:
        slug = safe_slug(source_name)
        source_dir = a.out_dir / slug
        source_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 110)
        print(f"{source_name}")
        print("-" * 110)
        print(f"Hazard raster       : {hazard_path}")

        hazard = model_flood_from_raster(
            hazard_path,
            aoi,
            a.flood_threshold,
            a.exclude_value,
        )
        coverage = hazard["coverage"]
        print(f"AOI footprint cover : {coverage * 100:.2f}%")
        print(f"Valid AOI pixels    : {hazard['valid_pixel_fraction'] * 100:.2f}%")

        if coverage < a.min_hazard_aoi_coverage and not a.allow_partial_hazard_coverage:
            raise RuntimeError(
                f"{source_name} hazard covers only {coverage * 100:.2f}% of the original AOI. "
                f"Required >= {a.min_hazard_aoi_coverage * 100:.2f}%. "
                "Do not treat uncovered AOI as dry. Supply a full-domain hazard raster or "
                "explicitly use --allow-partial-hazard-coverage."
            )

        flood_geom = polygonize_flood(hazard["flood"], hazard["transform"])
        vector_stats, impacted = vector_impacts(flood_geom, hazard["crs"], osm)

        road_vector_path = source_dir / "impacted_roads.geojson"
        road_feature_count, road_vector_length_km = write_impacted_roads_geojson(
            impacted["roads"],
            flood_geom,
            road_vector_path,
            source_name,
        )
        print(f"Impacted road vector : {road_vector_path}")
        print(
            f"Road vector features : {road_feature_count:,} "
            f"({road_vector_length_km:.3f} km clipped geometry)"
        )

        population = population_impacts(
            hazard["flood"],
            hazard["profile"],
            aoi,
            worldpop,
            source_dir,
        )
        class_areas, impervious_km2 = nlcd_impacts(
            hazard["flood"],
            hazard["profile"],
            aoi,
            nlcd_lc,
            nlcd_imp,
            source_dir,
        )

        asset_arrays = rasterize_impacted_assets(
            hazard["flood"],
            hazard["aoi_mask"],
            hazard["profile"],
            impacted,
        )
        asset_profile = hazard["profile"].copy()
        for key, filename in [
            ("flood", "flood_extent.tif"),
            ("buildings", "impacted_buildings.tif"),
            ("roads", "impacted_roads.tif"),
            ("critical", "impacted_critical_facilities.tif"),
            ("combined", "impacted_assets_combined.tif"),
        ]:
            write_array(
                source_dir / filename,
                asset_arrays[key],
                asset_profile,
                dtype="uint8",
                nodata=255,
            )

        summary = {
            "source": source_name,
            "hazard_raster": str(hazard_path),
            "aoi_area_km2": aoi_area_km2,
            "hazard_aoi_coverage_fraction": coverage,
            "hazard_valid_pixel_fraction_inside_aoi": hazard["valid_pixel_fraction"],
            "flood_area_km2": hazard["flood_area_km2"],
            "affected_population_est": population,
            **vector_stats,
            "impacted_roads_vector": str(road_vector_path),
            "impacted_roads_vector_feature_count": road_feature_count,
            "impacted_roads_vector_length_km": road_vector_length_km,
            "impervious_equivalent_area_km2": impervious_km2,
        }
        summary_rows.append(summary)

        for code, area in sorted(class_areas.items()):
            lc_rows.append(
                {
                    "source": source_name,
                    "nlcd_class": code,
                    "nlcd_class_name": NLCD_CLASSES.get(code, f"Class {code}"),
                    "flooded_area_km2": area,
                }
            )

        metadata["hazards"][source_name] = {
            **summary,
            "extra_nodata_values": hazard["extra_nodata"],
            "output_dir": str(source_dir),
        }

        print(f"Flood area          : {hazard['flood_area_km2']:.3f} km²")
        print(f"Population exposed  : {population:.1f}")
        print(f"Buildings exposed   : {vector_stats['flooded_buildings']:,}")
        print(f"Flooded road length : {vector_stats['flooded_road_length_km']:.3f} km")
        print(f"Critical facilities : {vector_stats['affected_critical_facilities']:,}")
        print(f"Impervious area     : {impervious_km2:.3f} km²")
        print(f"Raster output dir   : {source_dir}")

    summary_df = pd.DataFrame(summary_rows)
    lc_df = pd.DataFrame(lc_rows)
    summary_path = a.out_dir / "full_aoi_impact_summary.csv"
    lc_path = a.out_dir / "full_aoi_landcover_impact.csv"
    meta_path = a.out_dir / "full_aoi_impact_metadata.json"
    summary_df.to_csv(summary_path, index=False)
    lc_df.to_csv(lc_path, index=False)
    tmp = meta_path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, meta_path)

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    cols = [
        "source",
        "hazard_aoi_coverage_fraction",
        "flood_area_km2",
        "affected_population_est",
        "flooded_buildings",
        "flooded_road_length_km",
        "affected_critical_facilities",
        "impervious_equivalent_area_km2",
    ]
    print(summary_df[cols].to_string(index=False))
    print(f"\nSummary CSV         : {summary_path}")
    print(f"Land-cover CSV      : {lc_path}")
    print(f"Metadata            : {meta_path}")
    print("STATUS: PASS_ORIGINAL_AOI_IMPACT_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
