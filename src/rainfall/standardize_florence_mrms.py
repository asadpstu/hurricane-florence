"""
STEP 5B - Decode, crop, reproject, and standardize MRMS Florence hourly QPE.

Purpose
-------
Convert the Step 5A GaugeCorr_QPE_01H GRIB2.GZ archive into a fixed 1 km
projected rainfall grid covering the FULL USGS 02089000 upstream watershed.

This is the hydrologic forcing domain (~6,232 km2), NOT the 149.69 km2 local
flood-mapping AOI.

Workflow
--------
1. Validate Step 5A acquisition metadata and 624-hour inventory.
2. Auto-detect the Step 2A upstream watershed polygon by matching polygon area
   against the expected ~6,232.005499 km2 drainage domain.
3. Build a fixed EPSG:32618, 1,000 m grid around that watershed.
4. Decompress one MRMS GRIB2.GZ at a time to temporary storage.
5. Read the GRIB2 using GDAL/Rasterio's GRIB driver.
6. Warp only the watershed-sized target grid using bilinear resampling.
7. Mask output to the watershed raster mask and preserve rainfall in mm/hour.
8. Write one standardized GeoTIFF per hour.
9. Record per-hour spatial coverage, min/mean/max rainfall, source signatures,
   and quality-control results.

Important scientific notes
--------------------------
- GaugeCorr_QPE_01H is a 1-hour precipitation accumulation product at about
  1-km spatial resolution.
- Negative values are physically invalid as precipitation and are treated as
  missing after reprojection; their frequency is explicitly recorded.
- No temporal interpolation or missing-hour filling is performed here.
- Output precipitation is a continuous field; bilinear reprojection is used
  because the target resolution is approximately the native MRMS resolution.
- Basin-average rainfall is NOT computed in this step; that is Step 5C.

Outputs
-------
output/rainfall/mrms_florence_2018_standardized/
  hourly/
    mrms_qpe_20180901T000000Z.tif
    ...
  watershed_mask_1km.tif
  watershed_selection.csv
  source_grid_signatures.csv
  hourly_spatial_qc.csv
  mrms_standardization_qc.csv
  mrms_standardization_metadata.json
  sample_hour_quicklook.png
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT


SCRIPT_BUILD = "STEP_5B_MRMS_STANDARDIZE_V1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inventory",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--acquisition-metadata",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--watershed-dir",
        type=Path,
        required=True,
        help=(
            "Step 2A output directory. Polygon layers are searched and the "
            "one closest to --expected-watershed-area-km2 is selected."
        ),
    )
    p.add_argument(
        "--expected-watershed-area-km2",
        type=float,
        default=6232.005499,
    )
    p.add_argument(
        "--max-watershed-area-difference-percent",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--area-crs",
        default="EPSG:5070",
    )
    p.add_argument(
        "--target-crs",
        default="EPSG:32618",
    )
    p.add_argument(
        "--target-resolution-m",
        type=float,
        default=1000.0,
    )
    p.add_argument(
        "--min-basin-valid-coverage-percent",
        type=float,
        default=99.0,
    )
    p.add_argument(
        "--max-hourly-rainfall-mm",
        type=float,
        default=300.0,
        help=(
            "Quality guardrail only. Values are not clipped to this threshold."
        ),
    )
    p.add_argument(
        "--max-grid-area-difference-percent",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    tmp.unlink(missing_ok=True)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(payload: Any, path: Path) -> None:
    atomic_text(json.dumps(payload, indent=2, default=str), path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_vector_candidates(root: Path) -> list[tuple[Path, str | None, str]]:
    """
    Return (path, layer, geometry_type) tuples.
    """
    import pyogrio

    candidates: list[tuple[Path, str | None, str]] = []
    exts = {".gpkg", ".geojson", ".json", ".shp"}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue

        if path.suffix.lower() == ".gpkg":
            try:
                for row in pyogrio.list_layers(path):
                    layer = str(row[0])
                    geom = str(row[1])
                    if "polygon" in geom.lower():
                        candidates.append((path, layer, geom))
            except Exception:
                continue
        else:
            try:
                gdf = gpd.read_file(path)
                geom_types = set(gdf.geom_type.dropna().astype(str))
                if any("Polygon" in x for x in geom_types):
                    candidates.append(
                        (
                            path,
                            None,
                            ",".join(sorted(geom_types)),
                        )
                    )
            except Exception:
                continue

    return candidates


def select_watershed(
    root: Path,
    expected_area_km2: float,
    area_crs: str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]:
    candidates = list_vector_candidates(root)
    if not candidates:
        raise RuntimeError(
            f"No polygon vector layers found under {root}"
        )

    rows = []
    best_gdf = None
    best_record = None

    for path, layer, geom_type in candidates:
        try:
            if layer is None:
                gdf = gpd.read_file(path)
            else:
                gdf = gpd.read_file(path, layer=layer)

            gdf = gdf.loc[
                gdf.geometry.notna()
                & ~gdf.geometry.is_empty
            ].copy()
            if gdf.empty or gdf.crs is None:
                continue

            poly = gdf[
                gdf.geom_type.isin(
                    ["Polygon", "MultiPolygon"]
                )
            ].copy()
            if poly.empty:
                continue

            dissolved = gpd.GeoDataFrame(
                {"geometry": [poly.geometry.union_all()]},
                crs=poly.crs,
            )
            area_km2 = float(
                dissolved.to_crs(area_crs).geometry.area.iloc[0]
                / 1_000_000.0
            )
            diff_pct = (
                (area_km2 - expected_area_km2)
                / expected_area_km2
                * 100.0
            )

            rec = {
                "path": str(path),
                "layer": layer,
                "geometry_type": geom_type,
                "feature_count": int(len(poly)),
                "area_km2": area_km2,
                "difference_percent": diff_pct,
                "absolute_difference_percent": abs(diff_pct),
            }
            rows.append(rec)

            if (
                best_record is None
                or rec["absolute_difference_percent"]
                < best_record["absolute_difference_percent"]
            ):
                best_record = rec
                best_gdf = dissolved

        except Exception as exc:
            rows.append(
                {
                    "path": str(path),
                    "layer": layer,
                    "geometry_type": geom_type,
                    "feature_count": None,
                    "area_km2": None,
                    "difference_percent": None,
                    "absolute_difference_percent": None,
                    "error": repr(exc),
                }
            )

    table = pd.DataFrame(rows)
    if best_record is None or best_gdf is None:
        raise RuntimeError(
            "No readable polygon candidate could be evaluated."
        )

    return best_gdf, table, best_record


def open_grib_dataset(path: Path):
    """
    Open GRIB raster. If the root dataset has no raster band but exposes
    subdatasets, open the first subdataset.
    """
    ds = rasterio.open(path)
    if ds.count >= 1 and ds.width > 0 and ds.height > 0:
        return ds, None

    subs = list(ds.subdatasets)
    ds.close()

    if not subs:
        raise RuntimeError(
            f"GRIB has no raster bands or subdatasets: {path}"
        )

    sub = rasterio.open(subs[0])
    return sub, subs[0]


def normalize_timestamp(text: str) -> str:
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    args = parse_args()

    if args.target_resolution_m <= 0:
        raise ValueError("--target-resolution-m must be > 0.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hourly_dir = args.output_dir / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)

    if any(args.output_dir.iterdir()) and not args.overwrite:
        # Allow an empty hourly directory just created above.
        existing = [
            p for p in args.output_dir.iterdir()
            if not (p == hourly_dir and not any(hourly_dir.iterdir()))
        ]
        if existing:
            raise FileExistsError(
                f"{args.output_dir} contains outputs. Use --overwrite."
            )

    if args.overwrite:
        for p in hourly_dir.glob("*.tif"):
            p.unlink()

    metadata_path = (
        args.output_dir / "mrms_standardization_metadata.json"
    )
    qc_path = (
        args.output_dir / "mrms_standardization_qc.csv"
    )
    spatial_qc_path = (
        args.output_dir / "hourly_spatial_qc.csv"
    )
    signature_path = (
        args.output_dir / "source_grid_signatures.csv"
    )
    selection_path = (
        args.output_dir / "watershed_selection.csv"
    )
    mask_path = args.output_dir / "watershed_mask_1km.tif"
    quicklook_path = (
        args.output_dir / "sample_hour_quicklook.png"
    )

    acquisition_meta = json.loads(
        args.acquisition_metadata.read_text(encoding="utf-8")
    )
    if not bool(acquisition_meta.get("safe_for_step_5b")):
        raise RuntimeError(
            "Step 5A metadata does not authorize Step 5B."
        )

    inventory = pd.read_csv(args.inventory)
    available = inventory[
        inventory["status"].astype(str) == "AVAILABLE"
    ].copy()

    if available.empty:
        raise RuntimeError(
            "Step 5A inventory contains no available hourly files."
        )

    available = available.sort_values(
        "expected_index"
    ).reset_index(drop=True)

    expected_hours = int(
        acquisition_meta.get("window", {}).get(
            "expected_hours",
            len(available),
        )
    )

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {SCRIPT_BUILD}")
    print("STEP 5B - STANDARDIZE MRMS QPE TO FULL UPSTREAM WATERSHED")
    print("=" * 100)
    print(f"Available Step 5A hours            : {len(available):,}")
    print(f"Expected Step 5A hours             : {expected_hours:,}")
    print(f"Watershed search directory         : {args.watershed_dir}")
    print(f"Expected watershed area            : {args.expected_watershed_area_km2:.6f} km2")
    print(f"Target CRS                         : {args.target_crs}")
    print(f"Target resolution                  : {args.target_resolution_m:.1f} m")
    print()

    # ------------------------------------------------------------------
    # Locate the authoritative Step 2A watershed.
    # ------------------------------------------------------------------
    watershed, selection_table, selected = select_watershed(
        args.watershed_dir,
        args.expected_watershed_area_km2,
        args.area_crs,
    )
    atomic_csv(selection_table, selection_path)

    watershed_area_diff_pct = float(
        selected["difference_percent"]
    )
    watershed_area_km2 = float(selected["area_km2"])

    if (
        abs(watershed_area_diff_pct)
        > args.max_watershed_area_difference_percent
    ):
        raise RuntimeError(
            "Best polygon candidate does not match expected Step 2A "
            f"watershed area closely enough: {watershed_area_km2:.3f} km2 "
            f"({watershed_area_diff_pct:+.3f}%)."
        )

    watershed_target = watershed.to_crs(args.target_crs)

    minx, miny, maxx, maxy = watershed_target.total_bounds
    res = float(args.target_resolution_m)

    left = math.floor(minx / res) * res
    bottom = math.floor(miny / res) * res
    right = math.ceil(maxx / res) * res
    top = math.ceil(maxy / res) * res

    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    transform = from_origin(left, top, res, res)

    watershed_mask = rasterize(
        [
            (geom, 1)
            for geom in watershed_target.geometry
            if geom is not None and not geom.is_empty
        ],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    mask_cells = int(watershed_mask.sum())
    grid_mask_area_km2 = (
        mask_cells * res * res / 1_000_000.0
    )
    grid_area_diff_pct = (
        (grid_mask_area_km2 - watershed_area_km2)
        / watershed_area_km2
        * 100.0
    )

    mask_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": args.target_crs,
        "transform": transform,
        "nodata": 0,
        "compress": "DEFLATE",
        "predictor": 1,
    }
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(watershed_mask.astype("uint8"), 1)
        dst.set_band_description(1, "USGS 02089000 upstream watershed mask")
        dst.update_tags(
            SOURCE_VECTOR=str(selected["path"]),
            SOURCE_LAYER=str(selected.get("layer")),
            VECTOR_AREA_KM2=f"{watershed_area_km2:.9f}",
            RASTERIZED_AREA_KM2=f"{grid_mask_area_km2:.9f}",
        )

    print(f"Selected watershed vector          : {selected['path']}")
    print(f"Selected watershed layer           : {selected.get('layer')}")
    print(f"Selected watershed area            : {watershed_area_km2:.6f} km2")
    print(
        f"Difference vs expected             : "
        f"{watershed_area_diff_pct:+.6f} %"
    )
    print(
        f"Rasterized 1-km mask area          : "
        f"{grid_mask_area_km2:.6f} km2"
    )
    print(
        f"Rasterized/vector area difference  : "
        f"{grid_area_diff_pct:+.6f} %"
    )
    print(f"Target grid                        : {width} x {height}")
    print()

    # Verify GRIB driver exists.
    with rasterio.Env() as env:
        drivers = env.drivers()
        if "GRIB" not in drivers:
            raise RuntimeError(
                "GDAL/Rasterio GRIB driver is not available in this "
                "environment. Install a GDAL/Rasterio build with GRIB support "
                "before Step 5B."
            )

    hourly_rows = []
    signature_rows = []
    sample_array = None
    sample_time = None

    temp_root = args.output_dir / "_tmp_grib"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        for i, row in available.iterrows():
            timestamp = str(row["valid_time_utc"])
            token = normalize_timestamp(timestamp)

            gz_path = Path(str(row["local_path"]))
            if not gz_path.exists():
                raise FileNotFoundError(
                    f"Inventory file missing on disk: {gz_path}"
                )

            temp_grib = temp_root / f"{token}.grib2"
            with gzip.open(gz_path, "rb") as src, temp_grib.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)

            ds = None
            try:
                ds, selected_subdataset = open_grib_dataset(temp_grib)

                if ds.crs is None:
                    raise RuntimeError(
                        f"MRMS GRIB has no CRS: {gz_path}"
                    )

                band_tags = ds.tags(1)
                dataset_tags = ds.tags()

                source_signature = {
                    "valid_time_utc": timestamp,
                    "source_product": row.get("source_product"),
                    "driver": ds.driver,
                    "crs": str(ds.crs),
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "count": int(ds.count),
                    "dtype": str(ds.dtypes[0]),
                    "nodata": ds.nodata,
                    "transform": repr(ds.transform),
                    "subdataset": selected_subdataset,
                    "band_description": (
                        ds.descriptions[0]
                        if ds.descriptions
                        else None
                    ),
                    "band_unit": (
                        ds.units[0]
                        if ds.units
                        else None
                    ),
                    "GRIB_ELEMENT": band_tags.get("GRIB_ELEMENT"),
                    "GRIB_SHORT_NAME": band_tags.get("GRIB_SHORT_NAME"),
                    "GRIB_UNIT": band_tags.get("GRIB_UNIT"),
                    "GRIB_COMMENT": band_tags.get("GRIB_COMMENT"),
                    "GRIB_REF_TIME": band_tags.get("GRIB_REF_TIME"),
                    "GRIB_VALID_TIME": band_tags.get("GRIB_VALID_TIME"),
                }
                signature_rows.append(source_signature)

                vrt_kwargs = {
                    "crs": args.target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "resampling": Resampling.bilinear,
                    "nodata": -9999.0,
                }
                if ds.nodata is not None and np.isfinite(ds.nodata):
                    vrt_kwargs["src_nodata"] = ds.nodata

                with WarpedVRT(ds, **vrt_kwargs) as vrt:
                    arr = vrt.read(1, masked=True)

                data = np.asarray(arr.data, dtype="float64")
                valid = (
                    ~np.ma.getmaskarray(arr)
                    & np.isfinite(data)
                    & watershed_mask
                )

                # Negative precipitation is physically invalid / missing.
                negative_inside = valid & (data < 0.0)
                negative_count = int(negative_inside.sum())
                valid &= data >= 0.0

                basin_valid_count = int(valid.sum())
                basin_coverage_pct = (
                    basin_valid_count / mask_cells * 100.0
                    if mask_cells
                    else 0.0
                )

                values = data[valid]
                if values.size:
                    rain_min = float(values.min())
                    rain_mean = float(values.mean())
                    rain_max = float(values.max())
                    rain_p95 = float(np.percentile(values, 95))
                    rain_p99 = float(np.percentile(values, 99))
                else:
                    rain_min = rain_mean = rain_max = rain_p95 = rain_p99 = None

                out = np.full(
                    (height, width),
                    -9999.0,
                    dtype="float32",
                )
                out[valid] = data[valid].astype("float32")

                out_path = hourly_dir / f"mrms_qpe_{token}.tif"
                profile = {
                    "driver": "GTiff",
                    "height": height,
                    "width": width,
                    "count": 1,
                    "dtype": "float32",
                    "crs": args.target_crs,
                    "transform": transform,
                    "nodata": -9999.0,
                    "compress": "DEFLATE",
                    "predictor": 1,
                    "BIGTIFF": "IF_SAFER",
                }

                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(out, 1)
                    dst.set_band_description(
                        1,
                        "MRMS Gauge-Corrected 1-hour precipitation accumulation",
                    )
                    dst.update_tags(
                        VALID_TIME_UTC=timestamp,
                        SOURCE_PRODUCT=str(row.get("source_product")),
                        SOURCE_ARCHIVE_FILE=str(gz_path),
                        UNIT="mm",
                        TEMPORAL_ACCUMULATION="1 hour",
                        TARGET_DOMAIN="USGS 02089000 full upstream watershed",
                        RESAMPLING="bilinear",
                    )

                hourly_rows.append(
                    {
                        "valid_time_utc": timestamp,
                        "source_product": row.get("source_product"),
                        "source_file": str(gz_path),
                        "output_file": str(out_path),
                        "basin_mask_cells": mask_cells,
                        "basin_valid_cells": basin_valid_count,
                        "basin_valid_coverage_percent": basin_coverage_pct,
                        "negative_cells_before_masking": negative_count,
                        "rainfall_min_mm": rain_min,
                        "rainfall_mean_gridcell_mm": rain_mean,
                        "rainfall_p95_mm": rain_p95,
                        "rainfall_p99_mm": rain_p99,
                        "rainfall_max_mm": rain_max,
                        "output_sha256": sha256_file(out_path),
                    }
                )

                if sample_array is None and values.size:
                    sample_array = np.where(
                        out == -9999.0,
                        np.nan,
                        out,
                    )
                    sample_time = timestamp

            finally:
                if ds is not None:
                    ds.close()
                temp_grib.unlink(missing_ok=True)

            completed = i + 1
            if (
                completed == 1
                or completed % 25 == 0
                or completed == len(available)
            ):
                print(
                    f"Standardized                       : "
                    f"{completed:,}/{len(available):,}"
                )

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    hourly_qc = pd.DataFrame(hourly_rows)
    signatures = pd.DataFrame(signature_rows)

    atomic_csv(hourly_qc, spatial_qc_path)
    atomic_csv(signatures, signature_path)

    if len(hourly_qc) != len(available):
        raise RuntimeError(
            "Hourly output count differs from available Step 5A count."
        )

    min_coverage = float(
        hourly_qc["basin_valid_coverage_percent"].min()
    )
    mean_coverage = float(
        hourly_qc["basin_valid_coverage_percent"].mean()
    )
    low_coverage_hours = int(
        (
            hourly_qc["basin_valid_coverage_percent"]
            < args.min_basin_valid_coverage_percent
        ).sum()
    )

    maximum_hourly_value = float(
        hourly_qc["rainfall_max_mm"].max()
    )
    total_negative_cells = int(
        hourly_qc["negative_cells_before_masking"].sum()
    )

    # Count distinct core grid signatures.
    core_cols = [
        "driver",
        "crs",
        "width",
        "height",
        "count",
        "dtype",
        "nodata",
        "transform",
        "GRIB_ELEMENT",
        "GRIB_SHORT_NAME",
        "GRIB_UNIT",
    ]
    unique_grid_signatures = int(
        signatures[core_cols]
        .astype(str)
        .drop_duplicates()
        .shape[0]
    )

    qc_rows = []

    def qc(severity: str, check: str, passed: bool, detail: str):
        qc_rows.append(
            {
                "severity": severity,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    qc(
        "BLOCKING",
        "STEP5A_HOUR_COUNT_PRESERVED",
        len(hourly_qc) == expected_hours,
        (
            f"standardized_hours={len(hourly_qc)}; "
            f"expected_hours={expected_hours}"
        ),
    )

    qc(
        "BLOCKING",
        "WATERSHED_AREA_MATCH",
        abs(watershed_area_diff_pct)
        <= args.max_watershed_area_difference_percent,
        (
            f"selected_area={watershed_area_km2:.6f} km2; "
            f"expected={args.expected_watershed_area_km2:.6f} km2; "
            f"difference={watershed_area_diff_pct:+.6f}%"
        ),
    )

    qc(
        "QUALITY",
        "RASTERIZED_WATERSHED_AREA_MATCH",
        abs(grid_area_diff_pct)
        <= args.max_grid_area_difference_percent,
        (
            f"rasterized_area={grid_mask_area_km2:.6f} km2; "
            f"vector_area={watershed_area_km2:.6f} km2; "
            f"difference={grid_area_diff_pct:+.6f}%"
        ),
    )

    qc(
        "BLOCKING",
        "BASIN_SPATIAL_COVERAGE",
        low_coverage_hours == 0,
        (
            f"minimum_hourly_coverage={min_coverage:.6f}%; "
            f"mean_coverage={mean_coverage:.6f}%; "
            f"hours_below_{args.min_basin_valid_coverage_percent:.3f}%="
            f"{low_coverage_hours}"
        ),
    )

    qc(
        "QUALITY",
        "HOURLY_RAINFALL_PLAUSIBILITY",
        maximum_hourly_value
        <= args.max_hourly_rainfall_mm,
        (
            f"maximum_gridcell_hourly_rainfall="
            f"{maximum_hourly_value:.4f} mm; "
            f"guardrail={args.max_hourly_rainfall_mm:.4f} mm"
        ),
    )

    qc(
        "QUALITY",
        "SOURCE_GRID_SIGNATURE_CONSISTENCY",
        unique_grid_signatures == 1,
        (
            f"unique_core_source_signatures="
            f"{unique_grid_signatures}"
        ),
    )

    qc_rows.append(
        {
            "severity": "NOTE",
            "check": "NEGATIVE_SOURCE_OR_WARPED_VALUES",
            "status": "RECORDED",
            "detail": (
                f"total_negative_target_cells_removed="
                f"{total_negative_cells}. Negative precipitation is "
                "treated as missing and never clipped to zero."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "RAINFALL_FORCING_DOMAIN",
            "status": "SELECTED",
            "detail": (
                f"Full USGS 02089000 upstream watershed "
                f"({watershed_area_km2:.3f} km2), not the local "
                "149.69 km2 flood-mapping AOI."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "TARGET_RAINFALL_GRID",
            "status": "SELECTED",
            "detail": (
                f"{args.target_crs}, {res:.1f} m cells, "
                f"{width} x {height}; bilinear resampling."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_CONSTRAINT",
            "check": "BASIN_MEAN_NOT_YET_COMPUTED",
            "status": "OPEN",
            "detail": (
                "Hourly standardized rasters are spatial forcing fields. "
                "Basin-average rainfall, cumulative storm rainfall, timing, "
                "and precipitation-discharge alignment are deferred to Step 5C."
            ),
        }
    )

    qc_df = pd.DataFrame(qc_rows)
    atomic_csv(qc_df, qc_path)

    blocking_fail = qc_df[
        (qc_df["severity"] == "BLOCKING")
        & (qc_df["status"] == "FAIL")
    ]
    quality_fail = qc_df[
        (qc_df["severity"] == "QUALITY")
        & (qc_df["status"] == "FAIL")
    ]

    if len(blocking_fail):
        status = "FAIL_MRMS_FLORENCE_STANDARDIZATION"
    elif len(quality_fail):
        status = "FAIL_MRMS_FLORENCE_STANDARDIZATION_QUALITY"
    else:
        status = "PASS_MRMS_FLORENCE_STANDARDIZED"

    if sample_array is not None:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(
            np.ma.masked_invalid(sample_array)
        )
        ax.set_title(
            f"MRMS 1-hour QPE — sample standardized hour\n{sample_time}"
        )
        ax.set_xlabel("1-km grid column")
        ax.set_ylabel("1-km grid row")
        fig.colorbar(im, ax=ax, label="Precipitation (mm / 1 h)")
        fig.tight_layout()
        fig.savefig(quicklook_path, dpi=180)
        plt.close(fig)

    metadata = {
        "status": status,
        "step": "STEP_5B",
        "script_build": SCRIPT_BUILD,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "inventory": str(args.inventory),
            "acquisition_metadata": str(args.acquisition_metadata),
            "watershed_search_directory": str(args.watershed_dir),
        },
        "watershed": {
            "selected_vector_path": selected["path"],
            "selected_layer": selected.get("layer"),
            "vector_area_km2": watershed_area_km2,
            "expected_area_km2": args.expected_watershed_area_km2,
            "area_difference_percent": watershed_area_diff_pct,
            "rasterized_mask_area_km2": grid_mask_area_km2,
            "rasterized_area_difference_percent": grid_area_diff_pct,
            "mask_cell_count": mask_cells,
        },
        "target_grid": {
            "crs": args.target_crs,
            "resolution_m": res,
            "width": width,
            "height": height,
            "transform": repr(transform),
            "bounds": [left, bottom, right, top],
            "resampling": "bilinear",
            "nodata": -9999.0,
            "unit": "mm per 1-hour accumulation",
        },
        "hourly_standardization": {
            "hour_count": int(len(hourly_qc)),
            "minimum_basin_valid_coverage_percent": min_coverage,
            "mean_basin_valid_coverage_percent": mean_coverage,
            "low_coverage_hour_count": low_coverage_hours,
            "maximum_gridcell_hourly_rainfall_mm": maximum_hourly_value,
            "negative_target_cell_count_removed": total_negative_cells,
            "unique_core_source_grid_signatures": unique_grid_signatures,
        },
        "scientific_constraints": {
            "hydrologic_domain_is_full_upstream_watershed": True,
            "local_flood_aoi_used_for_rainfall_forcing": False,
            "temporal_gap_filling_applied": False,
            "negative_precipitation_clipped_to_zero": False,
            "basin_mean_computed_in_this_step": False,
        },
        "blocking_failure_count": int(len(blocking_fail)),
        "quality_failure_count": int(len(quality_fail)),
        "safe_for_step_5c": bool(
            len(blocking_fail) == 0
            and len(quality_fail) == 0
        ),
        "output_paths": {
            "hourly_directory": str(hourly_dir),
            "watershed_mask": str(mask_path),
            "watershed_selection": str(selection_path),
            "source_grid_signatures": str(signature_path),
            "hourly_spatial_qc": str(spatial_qc_path),
            "qc": str(qc_path),
            "metadata": str(metadata_path),
            "quicklook": str(quicklook_path),
        },
    }
    atomic_json(metadata, metadata_path)

    print()
    print(f"Standardized hourly rasters        : {len(hourly_qc):,}")
    print(f"Hourly output directory            : {hourly_dir}")
    print(
        f"Minimum basin valid coverage       : "
        f"{min_coverage:.6f} %"
    )
    print(
        f"Mean basin valid coverage          : "
        f"{mean_coverage:.6f} %"
    )
    print(
        f"Maximum hourly grid-cell rainfall  : "
        f"{maximum_hourly_value:.4f} mm"
    )
    print(
        f"Negative cells removed             : "
        f"{total_negative_cells:,}"
    )
    print(
        f"Unique source grid signatures      : "
        f"{unique_grid_signatures}"
    )
    print()
    print(f"Watershed mask                     : {mask_path}")
    print(f"Watershed selection                : {selection_path}")
    print(f"Source signatures                  : {signature_path}")
    print(f"Hourly spatial QC                  : {spatial_qc_path}")
    print(f"QC                                 : {qc_path}")
    print(f"Metadata                           : {metadata_path}")
    print()
    print(f"Blocking failures                  : {len(blocking_fail)}")
    print(f"Quality failures                   : {len(quality_fail)}")
    print(
        f"Safe for Step 5C                   : "
        f"{'YES' if metadata['safe_for_step_5c'] else 'NO'}"
    )
    print(f"Status                             : {status}")

    if len(blocking_fail) or len(quality_fail):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
