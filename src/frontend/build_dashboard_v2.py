#!/usr/bin/env python3
"""Build the Hurricane Florence dashboard V2 static data bundle.

This builder is intentionally read-only with respect to the retained scientific
outputs. It creates lightweight web-preview products under ``output/frontend_v2``
without modifying Physics V3.5, ML V4, USGS, NOAA FIM, or impact results.

Dashboard V2 adds:
- reliable single / side-by-side map comparison;
- map viewport persistence across date/source/layer/compare changes;
- working impact-metric selector with source-specific bar colors;
- impact-map previews for population/buildings/roads/facilities/land cover;
- zero-valued impact population pixels are transparent and discrete asset impacts use icon-pointer previews;
- graph-first impact summary without the duplicated selected-metric chart;
- 3-source modeling-AOI land-cover impact;
- Sep 14-23 daily MRMS rainfall map previews with dry/no-data transparency;
- synchronized discharge / stage / WSE / HAND-equivalent hydraulic cards;
- branch-aware H-Q surrogate visualization;
- upstream USGS 02089000 watershed context on the interactive map;
- a compact Data & Methodology section instead of a generic reference-data block.

The map PNGs are visualization previews only. The retained GeoTIFFs remain the
scientific source products and are never overwritten or resampled in place.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_bounds


SCRIPT_BUILD = "FLORENCE_DASHBOARD_V2_BUNDLE"
FT_TO_M = 0.3048
FLOAT_NODATA = -9999.0
RAIN_DISPLAY_EPSILON_MM = 0.01
SOURCE_ORDER = ["ml", "physics", "observed"]
SOURCE_LABELS = {
    "ml": "ML V4",
    "physics": "Physics V3.5",
    "observed": "USGS observed-driver",
}
SOURCE_SHORT = {
    "ml": "ML",
    "physics": "Physics",
    "observed": "USGS",
}

DEFAULT_SOURCE_DIRS = {
    "observed": Path("output/flood_mapping/florence_2018_usgs_observed_retained"),
    "physics": Path("output/flood_mapping/florence_2018_physics_v3_5_retained"),
    "ml": Path("output/flood_mapping/florence_2018_ml_v4_retained"),
}

IMPACT_METRICS = [
    ("Flood Area", "flood_area_km2", "km²", 2),
    ("Affected Population", "affected_population_est", "people", 0),
    ("Flooded Buildings", "flooded_buildings", "count", 0),
    ("Flooded Road Length", "flooded_road_length_km", "km", 2),
    ("Affected Critical Facilities", "affected_critical_facilities", "count", 0),
]

LANDCOVER_GROUPS = [
    ("Impervious Equivalent Area", "impervious", None),
    ("Cultivated Crops", "nlcd", {82}),
    ("Wetlands", "nlcd", {90, 95}),
    ("Open Water", "nlcd", {11}),
    ("Developed", "nlcd", {21, 22, 23, 24}),
    ("Forest", "nlcd", {41, 42, 43}),
]

# Browser legend for the categorical impacted-land-cover map. Colors are kept
# identical to the preview raster palette so the user can decode every class.
NLCD_MAP_CLASSES = [
    (11, "Open Water", (70, 120, 190)),
    (21, "Developed, Open Space", (222, 197, 197)),
    (22, "Developed, Low Intensity", (217, 146, 130)),
    (23, "Developed, Medium Intensity", (235, 0, 0)),
    (24, "Developed, High Intensity", (171, 0, 0)),
    (31, "Barren Land (Rock/Sand/Clay)", (179, 172, 159)),
    (41, "Deciduous Forest", (104, 171, 95)),
    (42, "Evergreen Forest", (28, 95, 44)),
    (43, "Mixed Forest", (181, 197, 143)),
    (52, "Shrub/Scrub", (204, 184, 121)),
    (71, "Grassland/Herbaceous", (223, 223, 194)),
    (81, "Pasture/Hay", (220, 217, 57)),
    (82, "Cultivated Crops", (171, 108, 40)),
    (90, "Woody Wetlands", (184, 217, 235)),
    (95, "Emergent Herbaceous Wetlands", (108, 159, 184)),
]


@dataclass(frozen=True)
class OverlaySpec:
    path: str
    bounds: list[list[float]]
    label: str
    unit: str
    value_min: float | None = None
    value_max: float | None = None
    source_raster: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path,
            "bounds": self.bounds,
            "label": self.label,
            "unit": self.unit,
        }
        if self.value_min is not None:
            out["value_min"] = float(self.value_min)
        if self.value_max is not None:
            out["value_max"] = float(self.value_max)
        if self.source_raster is not None:
            out["source_raster"] = self.source_raster
        if self.note is not None:
            out["note"] = self.note
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Hurricane Florence dashboard V2 bundle.")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, default=Path("output/frontend_v2"))
    p.add_argument(
        "--rainfall-hourly-dir",
        type=Path,
        default=Path("output/rainfall/florence_standardized/hourly"),
    )
    p.add_argument("--rainfall-start", default="2018-09-14")
    p.add_argument("--rainfall-end", default="2018-09-23")
    p.add_argument("--preview-max-dimension", type=int, default=1000)
    p.add_argument("--hydraulic-preview-resolution-m", type=float, default=50.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def maybe(path: Path) -> Path | None:
    return path if path.exists() else None


def normalize_source(value: object) -> str:
    s = str(value).strip().casefold()
    if s in {"ml", "ml v4", "machine learning", "machine-learning", "ml_v4"}:
        return "ml"
    if s in {"physics", "physics v3.5", "physics v3_5", "physics_v3_5", "physics-based"}:
        return "physics"
    if s in {"usgs", "observed", "observation", "usgs observed", "usgs observed-driver", "observed-driver"}:
        return "observed"
    return s.replace(" ", "_")


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def day_range(start: str, end: str) -> list[str]:
    a = parse_day(start)
    b = parse_day(end)
    if b < a:
        raise ValueError("--rainfall-end must be on/after --rainfall-start")
    return [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def make_clean_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} contains outputs. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_dashboard_assets(script_dir: Path, output_dir: Path) -> None:
    src = script_dir / "dashboard_v2"
    dst = output_dir / "dashboard"
    require(src / "index.html", "Dashboard HTML")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_geo_context(root: Path, output_dir: Path) -> dict[str, str]:
    candidates = {
        "aoi": [
            root / "output/frontend/geo/aoi.geojson",
            root / "output/spatial/goldsboro_florence_modeling_aoi/goldsboro_florence_modeling_aoi.geojson",
        ],
        "mainstem": [
            root / "output/frontend/geo/neuse_mainstem.geojson",
        ],
        "gauge": [
            root / "output/frontend/geo/usgs_02089000.geojson",
        ],
        "upstream_watershed": [
            root / "output/spatial/upstream_watershed/usgs_02089000_upstream_basin.geojson",
            root / "output/spatial/usgs_02089000_domain/usgs_02089000_upstream_basin.geojson",
        ],
    }
    out: dict[str, str] = {}
    geo_dir = output_dir / "geo"
    geo_dir.mkdir(parents=True, exist_ok=True)
    for key, opts in candidates.items():
        src = next((p for p in opts if p.exists()), None)
        if src is None:
            continue
        dst = geo_dir / src.name
        shutil.copy2(src, dst)
        out[key] = f"../geo/{dst.name}"
    return out


def raster_to_wgs84_array(
    path: Path,
    max_dimension: int,
    resampling: Resampling,
) -> tuple[np.ndarray, rasterio.Affine, list[list[float]]]:
    """Reproject one raster band to a lightweight EPSG:4326 preview array."""
    with rasterio.open(path) as src:
        left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        raw_w = max(1, src.width)
        raw_h = max(1, src.height)
        aspect = raw_w / raw_h
        if aspect >= 1:
            width = min(max_dimension, raw_w)
            height = max(1, int(round(width / aspect)))
        else:
            height = min(max_dimension, raw_h)
            width = max(1, int(round(height * aspect)))

        dst_transform = from_bounds(left, bottom, right, top, width, height)
        dst = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=resampling,
        )
    bounds = [[float(bottom), float(left)], [float(top), float(right)]]
    return dst, dst_transform, bounds


def rgba_from_values(
    arr: np.ndarray,
    *,
    cmap_name: str,
    valid: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: int = 190,
    transparent_zeros: bool = False,
) -> tuple[np.ndarray, float | None, float | None]:
    finite = np.isfinite(arr)
    if valid is None:
        valid = finite.copy()
    else:
        valid = valid & finite
    if transparent_zeros:
        valid &= arr > 0
    if not np.any(valid):
        rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
        return rgba, None, None

    vals = arr[valid].astype(float)
    if vmin is None:
        vmin = float(np.nanpercentile(vals, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(vals, 98))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(vals))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(vals))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = colormaps[cmap_name]
    rgba_f = cmap(norm(np.where(valid, arr, vmin)))
    rgba = np.clip(rgba_f * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = 0
    rgba[..., 3][valid] = np.uint8(alpha)
    return rgba, vmin, vmax


def write_png_rgba(rgba: np.ndarray, path: Path) -> None:
    from matplotlib import image as mpimg

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    mpimg.imsave(tmp, rgba)
    os.replace(tmp, path)


def create_raster_preview(
    src: Path,
    dst: Path,
    *,
    max_dimension: int,
    kind: str,
    fixed_range: tuple[float, float] | None = None,
) -> OverlaySpec:
    if kind == "extent":
        arr, _, bounds = raster_to_wgs84_array(src, max_dimension, Resampling.nearest)
        valid = np.isfinite(arr) & (arr >= 0.5) & (arr < 254)
        rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
        # Floodwater blue, transparent elsewhere.
        rgba[..., 0][valid] = 20
        rgba[..., 1][valid] = 105
        rgba[..., 2][valid] = 190
        rgba[..., 3][valid] = 185
        write_png_rgba(rgba, dst)
        return OverlaySpec(
            path="../overlays/" + dst.relative_to(next(p for p in dst.parents if p.name == "overlays")).as_posix(),
            bounds=bounds,
            label="Flood extent",
            unit="binary",
            value_min=0,
            value_max=1,
            source_raster=str(src),
        )

    arr, _, bounds = raster_to_wgs84_array(src, max_dimension, Resampling.bilinear)
    if kind == "depth":
        valid = np.isfinite(arr) & (arr > 0)
        cmap = "Blues"
        label, unit = "Flood depth", "m"
        transparent_zeros = True
    elif kind == "hand":
        valid = np.isfinite(arr)
        cmap = "terrain"
        label, unit = "HAND-equivalent relative terrain height", "m"
        transparent_zeros = False
    elif kind == "wse":
        valid = np.isfinite(arr)
        cmap = "viridis"
        label, unit = "WSE NAVD88", "m"
        transparent_zeros = False
    elif kind == "rainfall":
        valid = np.isfinite(arr) & (arr >= 0)
        cmap = "turbo"
        label, unit = "24-hour MRMS rainfall", "mm"
        transparent_zeros = True
    else:
        raise ValueError(kind)

    vmin, vmax = fixed_range if fixed_range else (None, None)
    rgba, use_min, use_max = rgba_from_values(
        arr,
        cmap_name=cmap,
        valid=valid,
        vmin=vmin,
        vmax=vmax,
        transparent_zeros=transparent_zeros,
    )
    write_png_rgba(rgba, dst)
    return OverlaySpec(
        path="../overlays/" + dst.relative_to(next(p for p in dst.parents if p.name == "overlays")).as_posix(),
        bounds=bounds,
        label=label,
        unit=unit,
        value_min=use_min,
        value_max=use_max,
        source_raster=str(src),
    )


def find_daily_summary(source_dir: Path) -> pd.DataFrame:
    path = require(source_dir / "daily_flood_summary.csv", "Daily flood summary")
    df = pd.read_csv(path)
    if "state_type" in df.columns:
        daily = df[df["state_type"].astype(str).eq("daily_max")].copy()
        if not daily.empty:
            df = daily
    required_cols = [
        "date",
        "predicted_q_m3s",
        "predicted_gage_height_ft",
        "gauge_wse_navd88_m",
        "mainstem_hand_equivalent_threshold_m",
        "flood_area_km2",
        "inundation_raster",
        "depth_raster",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    return df.sort_values("date").reset_index(drop=True)


def load_mapper_metadata(source_dir: Path) -> dict[str, Any]:
    p = require(source_dir / "daily_flood_metadata.json", "Daily flood metadata")
    return json.loads(p.read_text(encoding="utf-8"))


def source_hydraulics(
    root: Path,
    source_key: str,
    source_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    df = find_daily_summary(source_dir)
    meta = load_mapper_metadata(source_dir)
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        wse_m = float(r["gauge_wse_navd88_m"])
        stage_ft = float(r["predicted_gage_height_ft"])
        branch = str(r.get("hydrograph_branch", "")).strip().lower()
        row = {
            "date": str(r["date"]),
            "source": source_key,
            "source_label": SOURCE_LABELS[source_key],
            "source_short": SOURCE_SHORT[source_key],
            "discharge_m3s": float(r["predicted_q_m3s"]),
            "stage_ft": stage_ft,
            "wse_navd88_m": wse_m,
            "wse_navd88_ft": wse_m / FT_TO_M,
            "hand_equivalent_threshold_m": float(r["mainstem_hand_equivalent_threshold_m"]),
            "hq_branch": branch or None,
            "flood_area_km2": float(r["flood_area_km2"]),
            "inundation_raster": rel(root / str(r["inundation_raster"]), root)
            if not Path(str(r["inundation_raster"])).is_absolute()
            else str(r["inundation_raster"]),
            "depth_raster": rel(root / str(r["depth_raster"]), root)
            if not Path(str(r["depth_raster"])).is_absolute()
            else str(r["depth_raster"]),
        }
        rows.append(row)
    return rows, meta, df


def resolve_raster_path(root: Path, value: object) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    return root / p


def build_flood_previews(
    root: Path,
    output_dir: Path,
    source_key: str,
    summary_df: pd.DataFrame,
    max_dimension: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for _, r in summary_df.iterrows():
        d = str(r["date"])
        extent_src = require(resolve_raster_path(root, r["inundation_raster"]), f"{source_key} {d} inundation")
        depth_src = require(resolve_raster_path(root, r["depth_raster"]), f"{source_key} {d} depth")

        extent_dst = output_dir / "overlays" / "flood" / source_key / f"{d}_extent.png"
        depth_dst = output_dir / "overlays" / "flood" / source_key / f"{d}_depth.png"
        extent_spec = create_raster_preview(
            extent_src, extent_dst, max_dimension=max_dimension, kind="extent"
        )
        depth_spec = create_raster_preview(
            depth_src, depth_dst, max_dimension=max_dimension, kind="depth"
        )
        out[d] = {
            "extent": extent_spec.as_dict(),
            "depth": depth_spec.as_dict(),
        }
    return out


def aoi_projected_geometry(aoi_path: Path, crs: Any) -> Any:
    gdf = gpd.read_file(aoi_path)
    if gdf.empty:
        raise RuntimeError(f"AOI contains no features: {aoi_path}")
    gdf = gdf.to_crs(crs)
    return gdf.geometry.union_all()


def build_projected_grid(bounds: rasterio.coords.BoundingBox, resolution: float) -> tuple[rasterio.Affine, int, int]:
    width = max(1, int(math.ceil((bounds.right - bounds.left) / resolution)))
    height = max(1, int(math.ceil((bounds.top - bounds.bottom) / resolution)))
    transform = from_origin(bounds.left, bounds.top, resolution, resolution)
    return transform, width, height


def grid_xy(transform: rasterio.Affine, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    x = transform.c + cols * transform.a
    y = transform.f + rows * transform.e
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def projected_array_to_wgs84(
    arr: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: Any,
    max_dimension: int,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, list[list[float]]]:
    h, w = arr.shape
    left = src_transform.c
    top = src_transform.f
    right = left + src_transform.a * w
    bottom = top + src_transform.e * h
    wgs_left, wgs_bottom, wgs_right, wgs_top = transform_bounds(
        src_crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
    )
    aspect = w / h
    if aspect >= 1:
        dst_w = min(max_dimension, w)
        dst_h = max(1, int(round(dst_w / aspect)))
    else:
        dst_h = min(max_dimension, h)
        dst_w = max(1, int(round(dst_h * aspect)))
    dst_transform = from_bounds(wgs_left, wgs_bottom, wgs_right, wgs_top, dst_w, dst_h)
    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=arr.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst, [[float(wgs_bottom), float(wgs_left)], [float(wgs_top), float(wgs_right)]]


def render_projected_overlay(
    arr: np.ndarray,
    transform: rasterio.Affine,
    crs: Any,
    dst: Path,
    *,
    label: str,
    unit: str,
    cmap_name: str,
    max_dimension: int,
    fixed_range: tuple[float, float] | None = None,
    transparent_zeros: bool = False,
    transparent_below: float | None = None,
    note: str | None = None,
) -> OverlaySpec:
    wgs, bounds = projected_array_to_wgs84(arr, transform, crs, max_dimension)
    if transparent_below is not None:
        # Browser-preview-only cleanup: tiny interpolation artefacts around zero
        # should not be painted as rainfall/impact. Scientific source values are
        # untouched and summary statistics are computed before this display mask.
        wgs = wgs.copy()
        wgs[np.isfinite(wgs) & (wgs <= float(transparent_below))] = 0.0
    vmin, vmax = fixed_range if fixed_range else (None, None)
    rgba, use_min, use_max = rgba_from_values(
        wgs,
        cmap_name=cmap_name,
        valid=np.isfinite(wgs),
        vmin=vmin,
        vmax=vmax,
        transparent_zeros=transparent_zeros,
    )
    write_png_rgba(rgba, dst)
    return OverlaySpec(
        path="../overlays/" + dst.relative_to(next(p for p in dst.parents if p.name == "overlays")).as_posix(),
        bounds=bounds,
        label=label,
        unit=unit,
        value_min=use_min,
        value_max=use_max,
        note=note,
    )


def derive_hand_and_wse_previews(
    root: Path,
    output_dir: Path,
    hydraulics_by_source: dict[str, list[dict[str, Any]]],
    metadata_by_source: dict[str, dict[str, Any]],
    geo_paths: dict[str, str],
    max_dimension: int,
    resolution_m: float,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    dem_candidates = [
        root / "output/terrain/goldsboro_dem_2014/goldsboro_dem_2014_1m_navd88_m.tif",
    ]
    dem_path = next((p for p in dem_candidates if p.exists()), None)
    if dem_path is None:
        return None, {}

    # Need an AOI. The copied dashboard path uses ../geo/...; resolve source path instead.
    aoi_candidates = [
        root / "output/frontend/geo/aoi.geojson",
        root / "output/spatial/goldsboro_florence_modeling_aoi/goldsboro_florence_modeling_aoi.geojson",
    ]
    aoi_path = next((p for p in aoi_candidates if p.exists()), None)
    if aoi_path is None:
        return None, {}

    # Physics and ML use the same retained 1 m terrain/profile construction.
    ref_key = "physics" if "physics" in metadata_by_source else next(iter(metadata_by_source), None)
    if ref_key is None:
        return None, {}
    ref_meta = metadata_by_source[ref_key]
    fit = ref_meta.get("mainstem_profile_fit", {})
    axis = np.asarray(ref_meta.get("longitudinal_axis_xy", []), dtype=float)
    if axis.shape != (2,):
        return None, {}
    slope = float(fit.get("slope_m_per_m"))
    intercept = float(fit.get("intercept_dem_m"))

    with rasterio.open(dem_path) as dem:
        transform, width, height = build_projected_grid(dem.bounds, resolution_m)
        dem_arr = dem.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        # read(out_shape) changes effective transform.
        transform = from_bounds(*dem.bounds, width, height)
        valid = np.isfinite(dem_arr)
        if dem.nodata is not None:
            valid &= dem_arr != dem.nodata
        geom = aoi_projected_geometry(aoi_path, dem.crs)
        inside = geometry_mask(
            [geom], out_shape=(height, width), transform=transform, invert=True, all_touched=False
        )
        valid &= inside
        xx, yy = grid_xy(transform, width, height)

        gauge = ref_meta.get("gauge", {})
        gauge_lon = float(gauge.get("lon", -77.9975))
        gauge_lat = float(gauge.get("lat", 35.3375))
        tx = Transformer.from_crs("EPSG:4326", dem.crs, always_xy=True)
        gauge_x, gauge_y = tx.transform(gauge_lon, gauge_lat)
        s = (xx - gauge_x) * axis[0] + (yy - gauge_y) * axis[1]
        reference_elev = slope * s + intercept
        hand = dem_arr.astype(np.float64) - reference_elev
        hand[~valid] = np.nan
        # Keep small negative profile residuals visible but avoid extreme outliers.
        hand_dst = output_dir / "overlays" / "hydraulics" / "hand_equivalent.png"
        hand_spec = render_projected_overlay(
            hand.astype(np.float32),
            transform,
            dem.crs,
            hand_dst,
            label="HAND-equivalent relative terrain height",
            unit="m",
            cmap_name="terrain",
            max_dimension=max_dimension,
            fixed_range=(0.0, float(np.nanpercentile(np.maximum(hand, 0), 98))),
            note=(
                "Derived for visualization from the retained longitudinal mainstem profile: "
                "DEM minus fitted mainstem reference elevation. This is the HAND-equivalent "
                "surface used by the retained WSE/DEM inundation logic, not a replacement scientific output."
            ),
        ).as_dict()

        # Generate date/source WSE previews with a common color range.
        raw_wse: dict[tuple[str, str], np.ndarray] = {}
        all_values: list[np.ndarray] = []
        for source_key, rows in hydraulics_by_source.items():
            meta = metadata_by_source.get(source_key, ref_meta)
            src_fit = meta.get("mainstem_profile_fit", fit)
            src_axis = np.asarray(meta.get("longitudinal_axis_xy", axis), dtype=float)
            if src_axis.shape != (2,):
                src_axis = axis
            src_slope = float(src_fit.get("slope_m_per_m", slope))
            src_gauge = meta.get("gauge", gauge)
            lon = float(src_gauge.get("lon", gauge_lon))
            lat = float(src_gauge.get("lat", gauge_lat))
            gx, gy = tx.transform(lon, lat)
            src_s = (xx - gx) * src_axis[0] + (yy - gy) * src_axis[1]
            for row in rows:
                arr = float(row["wse_navd88_m"]) + src_slope * src_s
                arr = arr.astype(np.float32)
                arr[~inside] = np.nan
                raw_wse[(source_key, row["date"])] = arr
                vals = arr[np.isfinite(arr)]
                if vals.size:
                    all_values.append(vals)
        if all_values:
            merged = np.concatenate(all_values)
            wse_range = (float(np.nanpercentile(merged, 2)), float(np.nanpercentile(merged, 98)))
        else:
            wse_range = None

        wse_specs: dict[str, dict[str, Any]] = {k: {} for k in hydraulics_by_source}
        for (source_key, d), arr in raw_wse.items():
            dst = output_dir / "overlays" / "hydraulics" / "wse" / source_key / f"{d}.png"
            spec = render_projected_overlay(
                arr,
                transform,
                dem.crs,
                dst,
                label="Spatial WSE NAVD88",
                unit="m",
                cmap_name="viridis",
                max_dimension=max_dimension,
                fixed_range=wse_range,
                note=(
                    "Lightweight visualization of the retained longitudinal WSE surface for the selected "
                    "source/date. Gauge WSE is the mapper value; the surface varies only along the fitted "
                    "mainstem longitudinal axis."
                ),
            )
            wse_specs[source_key][d] = spec.as_dict()

    return hand_spec, wse_specs


def find_hourly_rain_file(hourly_dir: Path, dt: datetime) -> Path | None:
    exact = hourly_dir / f"mrms_qpe_{dt:%Y%m%dT%H%M%SZ}.tif"
    if exact.exists():
        return exact
    # Be tolerant of seconds omitted or alternate prefix while still requiring timestamp.
    token = dt.strftime("%Y%m%dT%H")
    candidates = sorted(hourly_dir.glob(f"*{token}*.tif"))
    return candidates[0] if candidates else None


def build_daily_rainfall(
    root: Path,
    hourly_dir: Path,
    output_dir: Path,
    dates: list[str],
    aoi_path: Path | None,
    max_dimension: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not hourly_dir.exists():
        return [], [f"Rainfall hourly directory not found: {hourly_dir}"]
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    rainfall_dir = output_dir / "overlays" / "rainfall"

    # First pass: aggregate each day in native MRMS grid.
    daily_arrays: dict[str, tuple[np.ndarray, rasterio.Affine, Any, np.ndarray, int]] = {}
    for d in dates:
        start = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        files: list[Path] = []
        for h in range(24):
            p = find_hourly_rain_file(hourly_dir, start + timedelta(hours=h))
            if p is not None:
                files.append(p)
        if not files:
            warnings.append(f"No MRMS standardized rasters found for {d}")
            continue
        if len(files) < 24:
            warnings.append(f"{d}: only {len(files)}/24 hourly MRMS rasters found")

        with rasterio.open(files[0]) as src0:
            transform = src0.transform
            crs = src0.crs
            shape = (src0.height, src0.width)
            total = np.zeros(shape, dtype=np.float32)
            valid_hours = np.zeros(shape, dtype=np.uint16)
            for p in files:
                with rasterio.open(p) as src:
                    if (
                        src.shape != shape
                        or src.crs != crs
                        or not src.transform.almost_equals(transform)
                    ):
                        raise RuntimeError(f"MRMS grid mismatch within {d}: {p}")
                    arr = src.read(1).astype(np.float32)
                    valid = np.isfinite(arr)
                    if src.nodata is not None:
                        valid &= arr != src.nodata
                    valid &= arr >= 0
                    total[valid] += arr[valid]
                    valid_hours[valid] += 1
            # Require at least one valid hour; record completeness separately.
            total[valid_hours == 0] = np.nan
            daily_arrays[d] = (total, transform, crs, valid_hours, len(files))

    if not daily_arrays:
        return [], warnings

    # Common rainfall range makes date-to-date comparisons meaningful.
    all_vals = []
    for total, _, _, hours, _ in daily_arrays.values():
        vals = total[np.isfinite(total) & (hours > 0)]
        if vals.size:
            all_vals.append(vals)
    if all_vals:
        merged = np.concatenate(all_vals)
        rain_range = (0.0, float(np.nanpercentile(merged, 99)))
        if rain_range[1] <= 0:
            rain_range = (0.0, 1.0)
    else:
        rain_range = (0.0, 1.0)

    for d, (total, transform, crs, valid_hours, file_count) in daily_arrays.items():
        local = total.copy()
        if aoi_path is not None and aoi_path.exists():
            geom = aoi_projected_geometry(aoi_path, crs)
            inside = geometry_mask(
                [geom], out_shape=local.shape, transform=transform, invert=True, all_touched=True
            )
            local[~inside] = np.nan
            valid_hours = np.where(inside, valid_hours, 0)

        vals = local[np.isfinite(local)]
        mean_mm = float(np.nanmean(vals)) if vals.size else None
        max_mm = float(np.nanmax(vals)) if vals.size else None
        if aoi_path is not None and aoi_path.exists():
            inside_count = int(np.count_nonzero(inside))
            valid_fraction = (
                float(np.count_nonzero((valid_hours > 0) & inside) / inside_count)
                if inside_count > 0
                else 0.0
            )
        else:
            valid_fraction = float(np.mean(valid_hours > 0)) if valid_hours.size else 0.0

        # Keep raw rainfall values for statistics, but make truly dry/no-data
        # browser pixels transparent. This avoids a misleading purple sheet on
        # days where the retained MRMS field is zero (or numerical near-zero).
        if vals.size == 0:
            display_status = "no_valid_data"
        elif max_mm is not None and max_mm <= RAIN_DISPLAY_EPSILON_MM:
            display_status = "dry"
        else:
            display_status = "rainfall"

        display_local = local.copy()
        display_local[np.isfinite(display_local) & (display_local <= RAIN_DISPLAY_EPSILON_MM)] = 0.0

        dst = rainfall_dir / f"rainfall_{d}.png"
        spec = render_projected_overlay(
            display_local,
            transform,
            crs,
            dst,
            label="24-hour MRMS rainfall",
            unit="mm",
            cmap_name="turbo",
            max_dimension=max_dimension,
            fixed_range=rain_range,
            transparent_zeros=True,
            transparent_below=RAIN_DISPLAY_EPSILON_MM,
            note=(
                "24-hour accumulation from the retained standardized MRMS hourly QPE. "
                "Preview is clipped to the Goldsboro modeling AOI; zero/dry pixels are transparent."
            ),
        )
        rec = {
            "date": d,
            "hours_found": int(file_count),
            "mean_mm": mean_mm,
            "max_mm": max_mm,
            "valid_pixel_fraction": valid_fraction,
            "display_status": display_status,
            "display_zero_threshold_mm": RAIN_DISPLAY_EPSILON_MM,
            "overlay": spec.as_dict(),
        }
        records.append(rec)

    return records, warnings


def read_hq_surrogate(root: Path) -> dict[str, list[dict[str, float]]]:
    p = root / "output/observations/florence_hq/florence_effective_q_to_h_surrogate.csv"
    if not p.exists():
        return {"rising": [], "falling": []}
    df = pd.read_csv(p)
    qcol = "discharge_cms" if "discharge_cms" in df.columns else "discharge_m3s" if "discharge_m3s" in df.columns else None
    if qcol is None:
        return {"rising": [], "falling": []}
    out: dict[str, list[dict[str, float]]] = {"rising": [], "falling": []}
    for branch in ("rising", "falling"):
        hcol = f"{branch}_gage_height_ft"
        if hcol not in df.columns:
            continue
        x = df[[qcol, hcol]].dropna().copy()
        if x.empty:
            continue
        # Keep enough shape detail while avoiding a huge browser JSON payload.
        step = max(1, int(math.ceil(len(x) / 350)))
        x = x.iloc[::step].copy()
        if x.index[-1] != df[[qcol, hcol]].dropna().index[-1]:
            last = df[[qcol, hcol]].dropna().iloc[[-1]]
            x = pd.concat([x, last], ignore_index=True)
        out[branch] = [
            {"q_m3s": float(r[qcol]), "stage_ft": float(r[hcol])}
            for _, r in x.iterrows()
        ]
    return out


def normalize_impact_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "source" not in df.columns:
        return []
    df = df.copy()
    df["source_key"] = df["source"].map(normalize_source)
    rows = []
    for _, r in df.iterrows():
        source_key = str(r["source_key"])
        if source_key not in SOURCE_LABELS:
            continue
        row = {
            "source": source_key,
            "source_label": SOURCE_LABELS[source_key],
            "source_short": SOURCE_SHORT[source_key],
        }
        for _, key, _, _ in IMPACT_METRICS:
            value = pd.to_numeric(pd.Series([r.get(key)]), errors="coerce").iloc[0]
            row[key] = None if pd.isna(value) else float(value)
        imp = pd.to_numeric(pd.Series([r.get("impervious_equivalent_area_km2")]), errors="coerce").iloc[0]
        row["impervious_equivalent_area_km2"] = None if pd.isna(imp) else float(imp)
        rows.append(row)
    rows.sort(key=lambda x: SOURCE_ORDER.index(x["source"]) if x["source"] in SOURCE_ORDER else 99)
    return rows


def normalize_landcover(path: Path, impact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if not {"source", "nlcd_class", "flooded_area_km2"}.issubset(df.columns):
        return []
    df = df.copy()
    df["source_key"] = df["source"].map(normalize_source)
    df["nlcd_class"] = pd.to_numeric(df["nlcd_class"], errors="coerce")
    df["flooded_area_km2"] = pd.to_numeric(df["flooded_area_km2"], errors="coerce")
    impervious = {r["source"]: r.get("impervious_equivalent_area_km2") for r in impact_rows}
    rows: list[dict[str, Any]] = []
    for source_key in SOURCE_ORDER:
        src = df[df["source_key"] == source_key]
        if src.empty and source_key not in impervious:
            continue
        for label, kind, classes in LANDCOVER_GROUPS:
            if kind == "impervious":
                value = impervious.get(source_key)
            else:
                value = float(src[src["nlcd_class"].isin(classes)]["flooded_area_km2"].sum())
            rows.append(
                {
                    "source": source_key,
                    "source_label": SOURCE_LABELS[source_key],
                    "source_short": SOURCE_SHORT[source_key],
                    "category": label,
                    "area_km2": None if value is None or (isinstance(value, float) and not np.isfinite(value)) else float(value),
                }
            )
    return rows


def find_modeling_impact_files(root: Path) -> tuple[Path | None, Path | None, str | None]:
    candidates = [
        root / "output/impact/florence_2018/modeling_aoi_3source",
        root / "output/impact/florence_2018/modeling_aoi",
    ]
    for d in candidates:
        s = d / "full_aoi_impact_summary.csv"
        l = d / "full_aoi_landcover_impact.csv"
        if s.exists() and l.exists():
            return s, l, rel(d, root)
    return None, None, None


IMPACT_RASTER_PRODUCTS = {
    "flood_area": ("flood_extent.tif", "Flood extent", "binary", "impact_extent"),
    "population": (
        "affected_population_estimate.tif",
        "Affected population estimate",
        "people / raster cell",
        "impact_population",
    ),
    "buildings": ("impacted_buildings.tif", "Impacted buildings", "binary", "impact_buildings"),
    "roads": ("impacted_roads.tif", "Impacted roads", "binary", "impact_roads"),
    "critical": (
        "impacted_critical_facilities.tif",
        "Impacted critical facilities",
        "binary",
        "impact_critical",
    ),
    "landcover": ("impacted_landcover.tif", "Impacted land cover", "NLCD class", "impact_landcover"),
    "impervious": (
        "impacted_impervious_percent.tif",
        "Impacted imperviousness",
        "%",
        "impact_impervious",
    ),
}


def _impact_source_dir(impact_root: Path, source_key: str) -> Path | None:
    candidates = {
        "ml": ["ml", "ml_v4", "machine_learning"],
        "physics": ["physics", "physics_v3_5", "physics_v3.5"],
        "observed": ["usgs", "observed", "usgs_observed_driver", "usgs_observed-driver"],
    }.get(source_key, [source_key])
    for name in candidates:
        p = impact_root / name
        if p.is_dir():
            return p
    # Fall back to a case/slug-insensitive scan so old source labels still work.
    wanted = {x.replace("-", "_").replace(".", "_").casefold() for x in candidates}
    for p in impact_root.iterdir() if impact_root.exists() else []:
        if not p.is_dir():
            continue
        token = p.name.replace("-", "_").replace(".", "_").casefold()
        if token in wanted:
            return p
    return None



def _representative_mask_points(mask: np.ndarray, *, cell_px: int, max_points: int) -> list[tuple[int, int]]:
    """Return spatially distributed pixel representatives for icon pointers."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    buckets: dict[tuple[int, int], list[float]] = {}
    for row, col in coords:
        key = (int(row) // cell_px, int(col) // cell_px)
        acc = buckets.setdefault(key, [0.0, 0.0, 0.0])
        acc[0] += float(row)
        acc[1] += float(col)
        acc[2] += 1.0
    pts = [
        (int(round(v[0] / v[2])), int(round(v[1] / v[2])))
        for _, v in sorted(buckets.items())
        if v[2] > 0
    ]
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
    return [pts[int(i)] for i in idx]


def _draw_pin_icon(draw: Any, x: int, y: int, kind: str, scale: int = 15) -> None:
    """Draw a compact pointer icon directly into the web preview PNG."""
    from PIL import ImageDraw

    r = scale
    palette = {
        "impact_buildings": ((202, 53, 49, 255), (255, 247, 245, 255)),
        "impact_roads": ((43, 43, 43, 255), (255, 193, 7, 255)),
        "impact_critical": ((123, 31, 162, 255), (250, 245, 255, 255)),
    }
    edge, fill = palette[kind]
    # pointer body + small stem
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=edge, width=max(2, r // 5))
    draw.polygon([(x - 5, y + r - 1), (x + 5, y + r - 1), (x, y + r + 9)], fill=edge)

    if kind == "impact_buildings":
        # broken building / house symbol
        draw.rectangle((x - 7, y - 4, x + 7, y + 7), outline=edge, width=2)
        draw.polygon([(x - 9, y - 4), (x, y - 11), (x + 9, y - 4)], outline=edge, fill=None)
        draw.line([(x - 1, y - 8), (x + 2, y - 3), (x - 2, y + 1), (x + 2, y + 6)], fill=edge, width=2)
    elif kind == "impact_roads":
        # damaged-road sign: two lane edges with a central fracture
        draw.line([(x - 7, y + 7), (x - 3, y - 8)], fill=edge, width=2)
        draw.line([(x + 7, y + 7), (x + 3, y - 8)], fill=edge, width=2)
        draw.line([(x, y - 8), (x - 3, y - 2), (x + 3, y + 1), (x - 2, y + 7)], fill=edge, width=2)
    elif kind == "impact_critical":
        # critical-facility cross with a fracture
        draw.rectangle((x - 3, y - 9, x + 3, y + 9), fill=edge)
        draw.rectangle((x - 9, y - 3, x + 9, y + 3), fill=edge)
        draw.line([(x - 2, y - 7), (x + 2, y - 2), (x - 2, y + 3), (x + 2, y + 8)], fill=(255, 255, 255, 255), width=2)


def _decorate_impact_with_icons(rgba: np.ndarray, valid: np.ndarray, kind: str) -> np.ndarray:
    """Overlay intuitive asset pointers while keeping a faint exact raster footprint."""
    from PIL import Image, ImageDraw

    if kind not in {"impact_buildings", "impact_roads", "impact_critical"} or not np.any(valid):
        return rgba
    settings = {
        "impact_buildings": (34, 60),
        "impact_roads": (70, 32),
        "impact_critical": (46, 30),
    }
    cell_px, max_points = settings[kind]
    pts = _representative_mask_points(valid, cell_px=cell_px, max_points=max_points)
    image = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(image, mode="RGBA")
    for row, col in pts:
        _draw_pin_icon(draw, int(col), int(row), kind)
    # np.asarray(PIL.Image) can return a read-only NumPy view.
    # The caller applies the final strict flood alpha mask afterward, so this
    # array MUST be writable. Force an owned writable copy here.
    return np.array(image, dtype=np.uint8, copy=True)


def preview_flood_mask(
    flood_src: Path,
    dst_transform: rasterio.Affine,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject retained flood extent onto a web-preview grid without expanding it.

    Exposure products live on coarser WorldPop/NLCD grids. A partially flooded
    coarse cell is scientifically valid for weighted totals, but painting the
    whole cell can visually extend beyond the 1 m flood boundary. This mask is
    therefore display-only: it keeps the impact preview visible only where the
    retained flood raster itself is flooded. Scientific impact totals remain
    unchanged.
    """
    dst = np.zeros(dst_shape, dtype=np.float32)
    with rasterio.open(flood_src) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=0.0,
            resampling=Resampling.nearest,
        )
    return np.isfinite(dst) & (dst > 0.0) & (dst < 254.0)


def create_impact_preview(
    src: Path,
    dst: Path,
    *,
    max_dimension: int,
    label: str,
    unit: str,
    kind: str,
    flood_mask_src: Path | None = None,
) -> OverlaySpec:
    nearest_kinds = {"impact_extent", "impact_buildings", "impact_roads", "impact_critical", "impact_landcover", "impact_population"}
    resampling = Resampling.nearest if kind in nearest_kinds else Resampling.bilinear
    arr, dst_transform, bounds = raster_to_wgs84_array(src, max_dimension, resampling)
    finite = np.isfinite(arr)
    strict_flood = None
    if (
        flood_mask_src is not None
        and flood_mask_src.exists()
        and kind in {"impact_population", "impact_landcover", "impact_impervious"}
    ):
        strict_flood = preview_flood_mask(flood_mask_src, dst_transform, arr.shape)
    # 255-style nodata is used by categorical/binary asset rasters. Do not
    # apply that byte-range filter to population, where valid exposure values
    # can legitimately exceed 254 people per raster cell.
    if kind in {"impact_extent", "impact_buildings", "impact_roads", "impact_critical", "impact_landcover"}:
        finite &= arr < 254

    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    use_min: float | None = None
    use_max: float | None = None

    if kind in {"impact_extent", "impact_buildings", "impact_roads", "impact_critical"}:
        valid = finite & (arr > 0)
        colors = {
            "impact_extent": (20, 105, 190),
            "impact_buildings": (207, 67, 61),
            "impact_roads": (48, 48, 48),
            "impact_critical": (156, 39, 176),
        }
        r, g, b = colors[kind]
        rgba[..., 0][valid] = r
        rgba[..., 1][valid] = g
        rgba[..., 2][valid] = b
        # Asset rasters stay faint so the icon pointers carry the visual meaning.
        rgba[..., 3][valid] = 190 if kind == "impact_extent" else 72
        if kind in {"impact_buildings", "impact_roads", "impact_critical"}:
            rgba = _decorate_impact_with_icons(rgba, valid, kind)
        use_min, use_max = 0.0, 1.0
    elif kind == "impact_landcover":
        valid = finite & (arr > 0)
        if strict_flood is not None:
            valid &= strict_flood
        # Stable categorical palette keyed by NLCD class code. The scientific
        # class values are retained; this is only a browser preview palette.
        palette = {code: color for code, _, color in NLCD_MAP_CLASSES}
        rounded = np.where(finite, np.rint(arr), -9999).astype(np.int16, copy=False)
        for code, color in palette.items():
            mask = valid & (rounded == code)
            if np.any(mask):
                rgba[..., 0][mask] = color[0]
                rgba[..., 1][mask] = color[1]
                rgba[..., 2][mask] = color[2]
                rgba[..., 3][mask] = 215
        # Any valid class not listed gets a neutral tint rather than vanishing.
        remaining = valid & (rgba[..., 3] == 0)
        rgba[..., :3][remaining] = (125, 125, 125)
        rgba[..., 3][remaining] = 190
        vals = arr[valid]
        if vals.size:
            use_min, use_max = float(np.nanmin(vals)), float(np.nanmax(vals))
    elif kind == "impact_population":
        # Population rasters use 0 for background/no exposure. Nearest-neighbour
        # reprojection above prevents bilinear smearing of populated cells into
        # those zeros, and every value <= 0 remains fully transparent.
        valid = finite & (arr > 0.0)
        if strict_flood is not None:
            valid &= strict_flood
        rgba, use_min, use_max = rgba_from_values(
            arr,
            cmap_name="YlGn",
            valid=valid,
            transparent_zeros=True,
            alpha=220,
        )
        rgba[..., 3][~valid] = 0
    elif kind == "impact_impervious":
        valid = finite & (arr > 0)
        if strict_flood is not None:
            valid &= strict_flood
        rgba, use_min, use_max = rgba_from_values(
            arr,
            cmap_name="Oranges",
            valid=valid,
            vmin=0.0,
            vmax=100.0,
            transparent_zeros=True,
            alpha=220,
        )
    else:
        raise ValueError(kind)

    write_png_rgba(rgba, dst)
    preview_note = "Visualization preview generated from the retained modeling-AOI impact GeoTIFF."
    if kind == "impact_population":
        preview_note += " Population values <= 0 are rendered with alpha=0 (fully transparent)."
    if strict_flood is not None:
        preview_note += " Browser preview is strictly masked by the retained flood extent so no impact color is shown outside flooded cells."
    return OverlaySpec(
        path="../overlays/" + dst.relative_to(next(p for p in dst.parents if p.name == "overlays")).as_posix(),
        bounds=bounds,
        label=label,
        unit=unit,
        value_min=use_min,
        value_max=use_max,
        source_raster=str(src),
        note=preview_note,
    )



def reproject_raster_to_preview_grid(
    src_path: Path,
    dst_transform: rasterio.Affine,
    dst_shape: tuple[int, int],
    *,
    resampling: Resampling,
) -> np.ndarray:
    """Reproject one source band onto an already-defined EPSG:4326 preview grid."""
    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def create_date_matched_impact_preview(
    impact_src: Path,
    daily_flood_src: Path,
    dst: Path,
    *,
    max_dimension: int,
    label: str,
    unit: str,
    kind: str,
) -> OverlaySpec:
    """Create an impact preview whose alpha footprint is EXACTLY the selected-day flood preview.

    The previous dashboard used the peak/static impact raster footprint and then
    tried to mask it afterward. That was still confusing because the map date
    could be 14 Sep while the impact raster represented the event peak. Here the
    selected daily inundation raster owns the output grid and alpha mask. Impact
    values are reprojected onto that same grid and can never paint beyond the
    flood shown for that source/date.

    This is a browser-visualization correction only. Peak impact summary tables
    and scientific GeoTIFFs remain unchanged.
    """
    flood_arr, dst_transform, bounds = raster_to_wgs84_array(
        daily_flood_src,
        max_dimension,
        Resampling.nearest,
    )
    flood_valid = np.isfinite(flood_arr) & (flood_arr > 0.0) & (flood_arr < 254.0)

    resampling = (
        Resampling.bilinear
        if kind == "impact_impervious"
        else Resampling.nearest
    )
    arr = reproject_raster_to_preview_grid(
        impact_src,
        dst_transform,
        flood_arr.shape,
        resampling=resampling,
    )
    finite = np.isfinite(arr)
    if kind in {
        "impact_buildings",
        "impact_roads",
        "impact_critical",
        "impact_landcover",
    }:
        finite &= arr < 254.0

    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    use_min: float | None = None
    use_max: float | None = None

    if kind in {"impact_buildings", "impact_roads", "impact_critical"}:
        valid = flood_valid & finite & (arr > 0.0)
        colors = {
            "impact_buildings": (207, 67, 61),
            "impact_roads": (48, 48, 48),
            "impact_critical": (156, 39, 176),
        }
        r, g, b = colors[kind]
        rgba[..., 0][valid] = r
        rgba[..., 1][valid] = g
        rgba[..., 2][valid] = b
        rgba[..., 3][valid] = 72
        rgba = _decorate_impact_with_icons(rgba, valid, kind)
        # Defensive guarantee: icon decoration must never leave a read-only
        # PIL-backed NumPy view before the strict alpha mask is applied.
        if not rgba.flags.writeable:
            rgba = rgba.copy()
        # The asset footprint itself is also clipped to the daily flood mask.
        rgba[..., 3][~flood_valid] = 0
        use_min, use_max = 0.0, 1.0

    elif kind == "impact_landcover":
        valid = flood_valid & finite & (arr > 0.0)
        rounded = np.where(finite, np.rint(arr), -9999).astype(np.int16, copy=False)
        palette = {code: color for code, _, color in NLCD_MAP_CLASSES}
        for code, color in palette.items():
            mask = valid & (rounded == code)
            if np.any(mask):
                rgba[..., 0][mask] = color[0]
                rgba[..., 1][mask] = color[1]
                rgba[..., 2][mask] = color[2]
                rgba[..., 3][mask] = 215
        remaining = valid & (rgba[..., 3] == 0)
        rgba[..., :3][remaining] = (125, 125, 125)
        rgba[..., 3][remaining] = 190
        vals = arr[valid]
        if vals.size:
            use_min, use_max = float(np.nanmin(vals)), float(np.nanmax(vals))

    elif kind == "impact_population":
        valid = flood_valid & finite & (arr > 0.0)
        rgba, use_min, use_max = rgba_from_values(
            arr,
            cmap_name="YlGn",
            valid=valid,
            transparent_zeros=True,
            alpha=220,
        )
        rgba[..., 3][~valid] = 0

    elif kind == "impact_impervious":
        valid = flood_valid & finite & (arr > 0.0) & (arr <= 100.0)
        rgba, use_min, use_max = rgba_from_values(
            arr,
            cmap_name="Oranges",
            valid=valid,
            vmin=0.0,
            vmax=100.0,
            transparent_zeros=True,
            alpha=220,
        )
        rgba[..., 3][~valid] = 0

    else:
        raise ValueError(kind)

    # Hard QA contract: no visible pixel may exist outside the selected-day
    # flood mask. This catches any future resampling or styling regression.
    outside = int(np.count_nonzero((rgba[..., 3] > 0) & ~flood_valid))
    if outside:
        raise RuntimeError(
            f"Date-matched impact preview escaped flood mask: {dst} ({outside} pixels)"
        )

    write_png_rgba(rgba, dst)
    return OverlaySpec(
        path="../overlays/" + dst.relative_to(next(p for p in dst.parents if p.name == "overlays")).as_posix(),
        bounds=bounds,
        label=label,
        unit=unit,
        value_min=use_min,
        value_max=use_max,
        source_raster=str(impact_src),
        note=(
            "Selected-date dashboard impact preview. Its alpha footprint is built "
            "on the exact same web grid as the retained daily inundation raster; "
            "no impact color can appear outside the selected-day flood extent."
        ),
    )


def build_date_matched_impact_previews(
    root: Path,
    impact_summary_path: Path | None,
    output_dir: Path,
    source_summaries: dict[str, pd.DataFrame],
    flood_layers: dict[str, dict[str, Any]],
    max_dimension: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build source/date impact-map previews synchronized to the date selector."""
    impact_roots: list[Path] = []
    if impact_summary_path is not None:
        impact_roots.append(impact_summary_path.parent)
    for candidate in [
        root / "output/impact/florence_2018/modeling_aoi_3source",
        root / "output/impact/florence_2018/modeling_aoi",
    ]:
        if candidate.exists() and candidate not in impact_roots:
            impact_roots.append(candidate)
    if not impact_roots:
        return {}

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for source_key in SOURCE_ORDER:
        summary_df = source_summaries.get(source_key)
        if summary_df is None or summary_df.empty:
            continue
        source_dir = next(
            (
                d
                for impact_root in impact_roots
                if (d := _impact_source_dir(impact_root, source_key)) is not None
            ),
            None,
        )
        if source_dir is None:
            continue

        source_dates: dict[str, dict[str, Any]] = {}
        for _, row in summary_df.iterrows():
            day = str(row["date"])
            daily_flood_src = require(
                resolve_raster_path(root, row["inundation_raster"]),
                f"{source_key} {day} inundation",
            )
            day_specs: dict[str, Any] = {}

            # Flood extent in Impact Assessment MUST be the selected-day flood,
            # never the old peak/static impact flood extent.
            day_flood_spec = flood_layers.get(source_key, {}).get(day, {}).get("extent")
            if day_flood_spec:
                day_specs["flood_area"] = day_flood_spec

            for key, (filename, label, unit, kind) in IMPACT_RASTER_PRODUCTS.items():
                if key == "flood_area":
                    continue
                impact_src = source_dir / filename
                if not impact_src.exists():
                    continue
                dst = output_dir / "overlays" / "impact" / source_key / day / f"{key}.png"
                day_specs[key] = create_date_matched_impact_preview(
                    impact_src,
                    daily_flood_src,
                    dst,
                    max_dimension=max_dimension,
                    label=label,
                    unit=unit,
                    kind=kind,
                ).as_dict()

            source_dates[day] = day_specs
        if source_dates:
            out[source_key] = source_dates
    return out


def build_impact_previews(
    root: Path,
    impact_summary_path: Path | None,
    output_dir: Path,
    max_dimension: int,
) -> dict[str, dict[str, Any]]:
    impact_roots: list[Path] = []
    if impact_summary_path is not None:
        impact_roots.append(impact_summary_path.parent)
    for candidate in [
        root / "output/impact/florence_2018/modeling_aoi_3source",
        root / "output/impact/florence_2018/modeling_aoi",
    ]:
        if candidate.exists() and candidate not in impact_roots:
            impact_roots.append(candidate)
    if not impact_roots:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for source_key in SOURCE_ORDER:
        source_dir = next(
            (
                d
                for impact_root in impact_roots
                if (d := _impact_source_dir(impact_root, source_key)) is not None
            ),
            None,
        )
        if source_dir is None:
            continue
        source_specs: dict[str, Any] = {}
        flood_mask_src = source_dir / "flood_extent.tif"
        for key, (filename, label, unit, kind) in IMPACT_RASTER_PRODUCTS.items():
            src = source_dir / filename
            if not src.exists():
                continue
            dst = output_dir / "overlays" / "impact" / source_key / f"{key}.png"
            source_specs[key] = create_impact_preview(
                src,
                dst,
                max_dimension=max_dimension,
                label=label,
                unit=unit,
                kind=kind,
                flood_mask_src=flood_mask_src,
            ).as_dict()
        if source_specs:
            out[source_key] = source_specs
    return out



def build_impact_vector_refs(
    root: Path,
    impact_summary_path: Path | None,
    impact_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Copy retained impacted-road GeoJSONs into the static frontend bundle."""
    impact_roots: list[Path] = []
    if impact_summary_path is not None:
        impact_roots.append(impact_summary_path.parent)
    for candidate in [
        root / "output/impact/florence_2018/modeling_aoi_3source",
        root / "output/impact/florence_2018/modeling_aoi",
    ]:
        if candidate.exists() and candidate not in impact_roots:
            impact_roots.append(candidate)

    summary_by_source = {r.get("source"): r for r in impact_rows}
    out: dict[str, dict[str, Any]] = {}
    for source_key in SOURCE_ORDER:
        source_dir = next(
            (
                d
                for impact_root in impact_roots
                if (d := _impact_source_dir(impact_root, source_key)) is not None
            ),
            None,
        )
        if source_dir is None:
            continue
        src = source_dir / "impacted_roads.geojson"
        if not src.exists():
            continue

        dst = output_dir / "vectors" / "impact" / source_key / "impacted_roads.geojson"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        feature_count = 0
        vector_length_km: float | None = None
        try:
            roads = gpd.read_file(src)
            feature_count = int(len(roads))
            if "segment_length_km" in roads.columns:
                vals = pd.to_numeric(roads["segment_length_km"], errors="coerce")
                vector_length_km = float(vals.sum()) if vals.notna().any() else None
        except Exception:
            # The browser can still load the copied GeoJSON; this QA metadata is
            # optional and must not break a valid static build.
            pass

        summary_length = summary_by_source.get(source_key, {}).get("flooded_road_length_km")
        out[source_key] = {
            "roads": {
                "path": f"../vectors/impact/{source_key}/impacted_roads.geojson",
                "label": "Flood-clipped impacted roads",
                "feature_count": feature_count,
                "vector_length_km": vector_length_km,
                "summary_length_km": summary_length,
                "source_geojson": rel(src, root),
                "note": (
                    "Exact OSM road line geometry clipped by the retained source-specific flood extent. "
                    "Rendered as vector lines above a faint flood background."
                ),
            }
        }
    return out


def read_noaa_peak_reference(root: Path) -> list[dict[str, Any]]:
    p = root / "output/impact/florence_2018/noaa_fim/peak_impact_summary.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        source_key = normalize_source(r.get("source", ""))
        if source_key not in SOURCE_LABELS:
            continue
        out = {"source": source_key, "source_short": SOURCE_SHORT[source_key]}
        for c in [
            "date",
            "discharge_m3s",
            "stage_ft",
            "wse_navd88_ft",
            "noaa_fim_level_ft",
            "noaa_status",
            "flood_area_km2",
            "affected_population_est",
            "flooded_buildings",
            "flooded_road_length_km",
            "affected_critical_facilities",
        ]:
            v = r.get(c)
            if c in {"date", "noaa_status"}:
                out[c] = None if pd.isna(v) else str(v)
            else:
                num = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
                out[c] = None if pd.isna(num) else float(num)
        rows.append(out)
    return rows


def main() -> None:
    a = parse_args()
    root = a.project_root.resolve()
    output_dir = (root / a.output_dir).resolve() if not a.output_dir.is_absolute() else a.output_dir.resolve()
    hourly_dir = (root / a.rainfall_hourly_dir).resolve() if not a.rainfall_hourly_dir.is_absolute() else a.rainfall_hourly_dir.resolve()
    if a.preview_max_dimension < 256:
        raise ValueError("--preview-max-dimension must be >= 256")
    if a.hydraulic_preview_resolution_m <= 0:
        raise ValueError("--hydraulic-preview-resolution-m must be > 0")

    make_clean_output(output_dir, a.overwrite)
    copy_dashboard_assets(Path(__file__).resolve().parent, output_dir)
    geo_paths = copy_geo_context(root, output_dir)

    print("=" * 110)
    print("NEUSE RIVER / HURRICANE FLORENCE 2018 — DASHBOARD V2 BUNDLE")
    print("=" * 110)
    print(f"Project root                         : {root}")
    print(f"Output dir                           : {output_dir}")

    hydraulics_by_source: dict[str, list[dict[str, Any]]] = {}
    metadata_by_source: dict[str, dict[str, Any]] = {}
    flood_layers: dict[str, dict[str, Any]] = {}
    source_summaries: dict[str, pd.DataFrame] = {}

    for source_key, rel_dir in DEFAULT_SOURCE_DIRS.items():
        source_dir = root / rel_dir
        require(source_dir, f"{SOURCE_LABELS[source_key]} retained flood directory")
        rows, meta, summary_df = source_hydraulics(root, source_key, source_dir)
        hydraulics_by_source[source_key] = rows
        metadata_by_source[source_key] = meta
        source_summaries[source_key] = summary_df.copy()
        flood_layers[source_key] = build_flood_previews(
            root,
            output_dir,
            source_key,
            summary_df,
            a.preview_max_dimension,
        )
        print(f"Hydraulics + flood previews {SOURCE_LABELS[source_key]:<22}: {len(rows)} dates")

    hand_spec, wse_specs = derive_hand_and_wse_previews(
        root,
        output_dir,
        hydraulics_by_source,
        metadata_by_source,
        geo_paths,
        a.preview_max_dimension,
        a.hydraulic_preview_resolution_m,
    )
    print(f"HAND-equivalent preview               : {'READY' if hand_spec else 'UNAVAILABLE'}")
    print(f"Spatial WSE previews                  : {sum(len(v) for v in wse_specs.values())}")

    rain_dates = day_range(a.rainfall_start, a.rainfall_end)
    aoi_source = next(
        (
            p
            for p in [
                root / "output/frontend/geo/aoi.geojson",
                root / "output/spatial/goldsboro_florence_modeling_aoi/goldsboro_florence_modeling_aoi.geojson",
            ]
            if p.exists()
        ),
        None,
    )
    rainfall_records, rainfall_warnings = build_daily_rainfall(
        root,
        hourly_dir,
        output_dir,
        rain_dates,
        aoi_source,
        a.preview_max_dimension,
    )
    print(f"Daily MRMS rainfall previews           : {len(rainfall_records)}/{len(rain_dates)}")

    hq = read_hq_surrogate(root)
    impact_summary_path, landcover_path, impact_domain = find_modeling_impact_files(root)
    impact_rows = normalize_impact_summary(impact_summary_path) if impact_summary_path else []
    landcover_rows = normalize_landcover(landcover_path, impact_rows) if landcover_path else []
    impact_map_layers = build_date_matched_impact_previews(
        root,
        impact_summary_path,
        output_dir,
        source_summaries,
        flood_layers,
        a.preview_max_dimension,
    )
    impact_vector_layers = build_impact_vector_refs(
        root,
        impact_summary_path,
        impact_rows,
        output_dir,
    )
    print(f"Date-matched impact source-days      : {sum(len(v) for v in impact_map_layers.values())}")
    print(f"Date-matched impact previews         : {sum(len(day) for src in impact_map_layers.values() for day in src.values())}")
    print(f"Impacted-road vector sources         : {len(impact_vector_layers)}")
    noaa_reference = read_noaa_peak_reference(root)

    all_hydraulics = []
    for source_key in SOURCE_ORDER:
        all_hydraulics.extend(hydraulics_by_source.get(source_key, []))
    all_hydraulics.sort(key=lambda r: (r["date"], SOURCE_ORDER.index(r["source"])))

    peak_dates: dict[str, str] = {}
    for source_key, rows in hydraulics_by_source.items():
        if rows:
            peak_row = max(rows, key=lambda r: float(r.get("flood_area_km2") or 0.0))
            peak_dates[source_key] = str(peak_row["date"])
    hydraulics_csv = pd.DataFrame(all_hydraulics)
    if not hydraulics_csv.empty:
        atomic_csv(hydraulics_csv, output_dir / "hydraulics_surrogate_daily.csv")

    if rainfall_records:
        atomic_csv(pd.DataFrame([{k: v for k, v in r.items() if k != "overlay"} for r in rainfall_records]), output_dir / "rainfall_daily_summary.csv")

    map_dates = sorted({d for src in flood_layers.values() for d in src.keys()})
    hydraulic_dates = sorted({r["date"] for r in all_hydraulics})
    all_dates = sorted(set(rain_dates) | set(hydraulic_dates) | set(map_dates))

    bundle = {
        "build": SCRIPT_BUILD,
        "build_revision": "2026-09-08-date-matched-impact-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "Neuse River — Hurricane Florence 2018",
        "gauge": {
            "site": "02089000",
            "name": "Neuse River near Goldsboro",
            "longitude": -77.9975,
            "latitude": 35.3375,
            "gage_datum_navd88_ft": 41.926,
        },
        "dates": all_dates,
        "map_dates": map_dates,
        "rainfall_dates": [r["date"] for r in rainfall_records],
        "sources": {
            key: {
                "label": SOURCE_LABELS[key],
                "short": SOURCE_SHORT[key],
                "scientific_role": (
                    "Model prediction"
                    if key in {"physics", "ml"}
                    else "USGS observed-driver reference generated with the same retained terrain mapper"
                ),
            }
            for key in ["observed", "physics", "ml"]
        },
        "map": {
            "flood": flood_layers,
            "wse": wse_specs,
            "hand_equivalent": hand_spec,
            "rainfall": {r["date"]: r["overlay"] for r in rainfall_records},
            "impact": impact_map_layers,
            "impact_vectors": impact_vector_layers,
            "geo": geo_paths,
            "comparison_modes": ["single", "side_by_side"],
            "layer_modes": ["extent", "depth", "rainfall", "hand", "wse", "impact"],
        },
        "hydraulics": all_hydraulics,
        "hq_surrogate": hq,
        "rainfall": rainfall_records,
        "impact": {
            "domain": impact_domain,
            "peak_dates": peak_dates,
            "map_temporal_mode": "selected_date",
            "graph_temporal_mode": "peak_summary",
            "domain_label": "Goldsboro modeling AOI",
            "rows": impact_rows,
            "metrics": [
                {"label": label, "key": key, "unit": unit, "decimals": decimals}
                for label, key, unit, decimals in IMPACT_METRICS
            ],
            "landcover": landcover_rows,
            "landcover_categories": [x[0] for x in LANDCOVER_GROUPS],
            "landcover_map_legend": [
                {
                    "code": code,
                    "name": name,
                    "color": f"rgb({color[0]}, {color[1]}, {color[2]})",
                }
                for code, name, color in NLCD_MAP_CLASSES
            ],
            "map_layers": [
                {"key": "flood_area", "label": "Flood extent"},
                {"key": "population", "label": "Affected population"},
                {"key": "buildings", "label": "Impacted buildings"},
                {"key": "roads", "label": "Impacted roads"},
                {"key": "critical", "label": "Critical facilities"},
                {"key": "landcover", "label": "Impacted land cover"},
                {"key": "impervious", "label": "Impacted imperviousness"},
            ],
        },
        "noaa_fim_reference": {
            "role": "Limited-domain hydraulic planning/reference library only",
            "rows": noaa_reference,
        },
        "methodology": [
            {
                "title": "USGS 02089000",
                "text": "Observed discharge/stage context for the Neuse River near Goldsboro gauge.",
            },
            {
                "title": "Effective Florence H-Q surrogate",
                "text": "Branch-aware rising/falling discharge-to-stage relationship used by the retained flood-mapping workflow.",
            },
            {
                "title": "WSE",
                "text": "Gauge WSE = gage datum (41.926 ft NAVD88) + stage. The retained mapper spatializes WSE only along the robust longitudinal mainstem profile.",
            },
            {
                "title": "HAND-equivalent terrain height",
                "text": "For the retained 1 m mapper, the relevant relative-height field is DEM minus fitted mainstem reference elevation; the dashboard shows this as a HAND-equivalent visualization and the daily threshold used by the selected state.",
            },
            {
                "title": "MRMS rainfall",
                "text": "Daily 24-hour accumulations are built from the retained standardized hourly MRMS GaugeCorr QPE and clipped to the Goldsboro modeling AOI for visualization.",
            },
            {
                "title": "Impact",
                "text": "Primary graphs use the modeling-AOI 3-source impact products (USGS observed-driver, Physics V3.5, ML V4). NOAA FIM remains a separate limited-domain reference and is not silently mixed with the modeling-AOI impact domain.",
            },
        ],
        "warnings": rainfall_warnings,
        "provenance": {
            "preview_note": "PNG map overlays are lightweight web previews only. Retained GeoTIFFs/CSVs remain the scientific source products.",
            "observed_map_note": "USGS reference flood maps are terrain-mapper products driven by observed-event gauge information; they are not satellite-observed flood extents.",
            "physics": "Retained Physics V3.5",
            "ml": "Retained ML V4",
            "no_satellite_validation": True,
        },
    }
    atomic_json(bundle, output_dir / "data_bundle.json")

    qa = {
        "script_build": SCRIPT_BUILD,
        "status": "PASS_FLORENCE_DASHBOARD_V2_BUNDLE_READY",
        "sources": {k: len(v) for k, v in hydraulics_by_source.items()},
        "flood_preview_source_date_count": sum(len(v) for v in flood_layers.values()),
        "wse_preview_source_date_count": sum(len(v) for v in wse_specs.values()),
        "hand_equivalent_ready": hand_spec is not None,
        "rainfall_preview_days": len(rainfall_records),
        "rainfall_transparent_dry_days": [r["date"] for r in rainfall_records if r.get("display_status") in {"dry", "no_valid_data"}],
        "upstream_watershed_ready": "upstream_watershed" in geo_paths,
        "impact_map_preview_count": sum(
            len(day_specs)
            for source_dates in impact_map_layers.values()
            for day_specs in source_dates.values()
        ),
        "impact_preview_strict_flood_mask": True,
        "impact_preview_date_matched": True,
        "impact_road_vector_source_count": len(impact_vector_layers),
        "impact_road_vector_sources": sorted(impact_vector_layers),
        "impact_sources": [r["source"] for r in impact_rows],
        "landcover_rows": len(landcover_rows),
        "hq_rising_points": len(hq.get("rising", [])),
        "hq_falling_points": len(hq.get("falling", [])),
        "warnings": rainfall_warnings,
    }
    atomic_json(qa, output_dir / "dashboard_v2_qa.json")

    print("\n" + "=" * 110)
    print("DASHBOARD V2 BUILD SUMMARY")
    print("=" * 110)
    print(f"Upstream watershed context           : {'READY' if 'upstream_watershed' in geo_paths else 'MISSING'}")
    print(f"Date-matched impact source-days      : {sum(len(v) for v in impact_map_layers.values())}")
    print(f"Date-matched impact previews         : {sum(len(day) for src in impact_map_layers.values() for day in src.values())}")
    print("Impact preview flood clipping        : STRICT + DATE-MATCHED to selected daily flood")
    print(f"Impacted-road vector sources         : {sorted(impact_vector_layers)}")
    print(f"Impact sources                       : {[r['source_short'] for r in impact_rows]}")
    print(f"Land-cover comparison rows           : {len(landcover_rows)}")
    print(f"H-Q rising / falling points          : {len(hq.get('rising', []))} / {len(hq.get('falling', []))}")
    if rainfall_warnings:
        print("Rainfall warnings:")
        for w in rainfall_warnings:
            print(f"  - {w}")
    print(f"Bundle                               : {output_dir / 'data_bundle.json'}")
    print(f"Dashboard                            : {output_dir / 'dashboard/index.html'}")
    print("Status                               : PASS_FLORENCE_DASHBOARD_V2_BUNDLE_READY")


if __name__ == "__main__":
    main()
