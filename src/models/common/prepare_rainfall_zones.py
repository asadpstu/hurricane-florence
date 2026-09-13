"""
MODEL IMPROVEMENT A2.4
Prepare a 1-km subcatchment rainfall-zone raster aligned exactly to the
existing Florence/MRMS watershed grid.

Purpose
-------
The 37 hydrologic modeling subcatchments from A2.3 will be used by BOTH:
  * Physics: subcatchment rainfall -> runoff -> routing
  * ML: spatial rainfall/state predictors

This step rasterizes the subcatchments to the existing MRMS 1-km grid.

Important residual policy
-------------------------
The NHDPlus catchment polygons cover ~99.72% of the authoritative basin.
For rainfall accounting only, any watershed-mask cells not assigned by
polygon rasterization are assigned to the nearest modeling subcatchment.
This produces 100% MRMS watershed-cell coverage without modifying the
hydrographic vector polygons themselves.

No rainfall data are downloaded in this step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy
from shapely.geometry import Point


BUILD = "MODEL_IMPROVEMENT_A2_4_MRMS_SUBCATCHMENT_ZONE_GRID_V1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subcatchments-gpkg", type=Path, required=True)
    p.add_argument("--subcatchment-layer", default="modeling_subcatchments")
    p.add_argument("--subcatchment-metadata", type=Path, required=True)
    p.add_argument("--watershed-mask", type=Path, required=True)
    p.add_argument("--min-raw-zone-coverage-percent", type=float, default=98.0)
    p.add_argument("--max-residual-fill-percent", type=float, default=2.0)
    p.add_argument("--min-cells-per-subcatchment", type=int, default=50)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/model_improvement/subcatchment_mrms_grid"),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def atomic_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(a.subcatchment_metadata.read_text(encoding="utf-8"))
    if not meta.get("safe_for_a2_4", False):
        raise RuntimeError("A2.3 metadata does not authorize A2.4.")

    sub = gpd.read_file(
        a.subcatchments_gpkg,
        layer=a.subcatchment_layer,
    )

    if "subcatchment_id" not in sub.columns:
        raise RuntimeError(
            "Subcatchment layer lacks subcatchment_id."
        )

    if len(sub) == 0:
        raise RuntimeError("Subcatchment layer is empty.")

    if sub["subcatchment_id"].duplicated().any():
        raise RuntimeError("Duplicate subcatchment_id values found.")

    sub = sub.sort_values("subcatchment_id").reset_index(drop=True)

    expected_count = int(
        meta["spatial_discretization"]["subcatchment_count"]
    )

    if len(sub) != expected_count:
        raise RuntimeError(
            f"Subcatchment count mismatch: layer={len(sub)}, "
            f"metadata={expected_count}"
        )

    with rasterio.open(a.watershed_mask) as src:
        mask_arr = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        nodata = src.nodata

    if crs is None:
        raise RuntimeError("Watershed mask has no CRS.")

    if nodata is not None:
        basin_mask = (
            np.isfinite(mask_arr)
            & (mask_arr != nodata)
            & (mask_arr > 0)
        )
    else:
        basin_mask = (
            np.isfinite(mask_arr)
            & (mask_arr > 0)
        )

    basin_cells = int(basin_mask.sum())

    if basin_cells <= 0:
        raise RuntimeError("Watershed mask contains no positive basin cells.")

    sub = sub.to_crs(crs)

    # Stable integer zone codes.
    lookup = pd.DataFrame({
        "zone_id": np.arange(1, len(sub) + 1, dtype=int),
        "subcatchment_id": sub["subcatchment_id"].astype(str),
    })

    zone_by_sc = dict(
        zip(lookup["subcatchment_id"], lookup["zone_id"])
    )

    shapes = [
        (
            geom,
            int(zone_by_sc[str(sc)]),
        )
        for sc, geom in zip(
            sub["subcatchment_id"],
            sub.geometry,
        )
        if geom is not None and not geom.is_empty
    ]

    zones_raw = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint16",
        all_touched=False,
    )

    # Never assign cells outside the authoritative MRMS watershed mask.
    zones_raw[~basin_mask] = 0

    raw_assigned = basin_mask & (zones_raw > 0)
    raw_unassigned = basin_mask & (zones_raw == 0)

    raw_assigned_cells = int(raw_assigned.sum())
    raw_unassigned_cells = int(raw_unassigned.sum())

    raw_coverage_pct = 100.0 * raw_assigned_cells / basin_cells
    residual_fill_pct = 100.0 * raw_unassigned_cells / basin_cells

    zones = zones_raw.copy()

    # Residual cells are assigned to nearest subcatchment polygon by cell center.
    # This changes only the rainfall-zone raster, not A2.3 vector geometry.
    if raw_unassigned_cells:
        unassigned_rc = np.argwhere(raw_unassigned)

        geoms = list(sub.geometry)
        sc_ids = list(sub["subcatchment_id"].astype(str))

        for row, col in unassigned_rc:
            x, y = xy(
                transform,
                int(row),
                int(col),
                offset="center",
            )
            pt = Point(x, y)

            distances = [
                geom.distance(pt)
                if geom is not None and not geom.is_empty
                else np.inf
                for geom in geoms
            ]

            idx = int(np.argmin(distances))
            zones[row, col] = int(
                zone_by_sc[sc_ids[idx]]
            )

    final_unassigned = int(
        (basin_mask & (zones == 0)).sum()
    )

    # Force zero outside watershed.
    zones[~basin_mask] = 0

    cell_area_km2 = float(
        abs(transform.a * transform.e) / 1e6
    )

    counts = pd.Series(
        zones[basin_mask].astype(int)
    ).value_counts().sort_index()

    # Build lookup with vector and raster areas.
    sub_attr = sub[
        [
            c for c in [
                "subcatchment_id",
                "model_area_km2",
                "downstream_subcatchment_id",
                "outlet_nhdplusid",
                "outlet_stream_order",
                "outlet_totdasqkm",
                "is_basin_outlet",
            ]
            if c in sub.columns
        ]
    ].copy()

    lookup = lookup.merge(
        sub_attr,
        on="subcatchment_id",
        how="left",
        validate="one_to_one",
    )

    lookup["raster_cell_count"] = (
        lookup["zone_id"].map(counts).fillna(0).astype(int)
    )
    lookup["raster_area_km2"] = (
        lookup["raster_cell_count"] * cell_area_km2
    )

    represented = int(
        (lookup["raster_cell_count"] > 0).sum()
    )
    min_cells = int(
        lookup["raster_cell_count"].min()
    )
    max_cells = int(
        lookup["raster_cell_count"].max()
    )

    total_raster_area = float(
        lookup["raster_area_km2"].sum()
    )

    # QC
    rows = []

    def qc(sev, check, passed, detail):
        rows.append({
            "severity": sev,
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    qc(
        "BLOCKING",
        "SUBCATCHMENT_COUNT",
        len(sub) == expected_count,
        f"layer={len(sub)}, expected={expected_count}",
    )

    qc(
        "BLOCKING",
        "ALL_SUBCATCHMENTS_REPRESENTED",
        represented == expected_count,
        f"represented={represented}/{expected_count}",
    )

    qc(
        "BLOCKING",
        "FINAL_WATERSHED_CELL_COVERAGE",
        final_unassigned == 0,
        f"unassigned_after_fill={final_unassigned}",
    )

    qc(
        "QUALITY",
        "RAW_POLYGON_ZONE_COVERAGE",
        raw_coverage_pct >= a.min_raw_zone_coverage_percent,
        (
            f"{raw_coverage_pct:.6f}% >= "
            f"{a.min_raw_zone_coverage_percent:.3f}%"
        ),
    )

    qc(
        "QUALITY",
        "RESIDUAL_FILL_FRACTION",
        residual_fill_pct <= a.max_residual_fill_percent,
        (
            f"{residual_fill_pct:.6f}% <= "
            f"{a.max_residual_fill_percent:.3f}%"
        ),
    )

    qc(
        "QUALITY",
        "MINIMUM_GRID_SUPPORT_PER_SUBCATCHMENT",
        min_cells >= a.min_cells_per_subcatchment,
        (
            f"minimum={min_cells} cells; "
            f"required>={a.min_cells_per_subcatchment}"
        ),
    )

    qcdf = pd.DataFrame(rows)

    bf = qcdf[
        (qcdf["severity"] == "BLOCKING")
        & (qcdf["status"] == "FAIL")
    ]

    qf = qcdf[
        (qcdf["severity"] == "QUALITY")
        & (qcdf["status"] == "FAIL")
    ]

    safe = len(bf) == 0 and len(qf) == 0

    status = (
        "PASS_A2_4_MRMS_SUBCATCHMENT_GRID_READY"
        if safe
        else (
            "FAIL_A2_4_MRMS_SUBCATCHMENT_GRID_BLOCKING"
            if len(bf)
            else "FAIL_A2_4_MRMS_SUBCATCHMENT_GRID_QUALITY"
        )
    )

    zone_tif = (
        a.output_dir / "subcatchment_rainfall_zones_1km.tif"
    )
    lookup_csv = (
        a.output_dir / "subcatchment_rainfall_zone_lookup.csv"
    )
    qc_csv = (
        a.output_dir / "subcatchment_rainfall_zone_qc.csv"
    )
    metadata_json = (
        a.output_dir / "subcatchment_rainfall_zone_metadata.json"
    )

    if zone_tif.exists() and not a.overwrite:
        raise RuntimeError(
            f"{zone_tif} exists; rerun with --overwrite"
        )

    out_profile = profile.copy()
    out_profile.update(
        dtype="uint16",
        count=1,
        nodata=0,
        compress="deflate",
        predictor=2,
    )

    with rasterio.open(zone_tif, "w", **out_profile) as dst:
        dst.write(zones.astype("uint16"), 1)
        dst.update_tags(
            SCRIPT_BUILD=BUILD,
            PURPOSE="MRMS_SUBCATCHMENT_RAINFALL_ZONES",
            ZONE_ZERO="OUTSIDE_WATERSHED",
            RESIDUAL_POLICY=(
                "UNASSIGNED_WATERSHED_CELLS_NEAREST_SUBCATCHMENT"
            ),
        )

    atomic_csv(lookup, lookup_csv)
    atomic_csv(qcdf, qc_csv)

    metadata = {
        "script_build": BUILD,
        "status": status,
        "safe_for_a3": bool(safe),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": {
            "crs": str(crs),
            "width": int(width),
            "height": int(height),
            "resolution_x_m": float(abs(transform.a)),
            "resolution_y_m": float(abs(transform.e)),
            "cell_area_km2": cell_area_km2,
            "watershed_cells": basin_cells,
            "watershed_raster_area_km2": (
                basin_cells * cell_area_km2
            ),
        },
        "zones": {
            "subcatchment_count": expected_count,
            "represented_count": represented,
            "raw_assigned_cells": raw_assigned_cells,
            "raw_unassigned_cells": raw_unassigned_cells,
            "raw_zone_coverage_percent": raw_coverage_pct,
            "residual_fill_percent": residual_fill_pct,
            "final_unassigned_cells": final_unassigned,
            "minimum_cells_per_subcatchment": min_cells,
            "maximum_cells_per_subcatchment": max_cells,
            "total_zone_raster_area_km2": total_raster_area,
        },
        "scientific_policy": {
            "vector_hydrographic_polygons_modified": False,
            "rainfall_grid_residual_policy": (
                "Any watershed-mask cell not covered by the NHDPlus-derived "
                "subcatchment polygons is assigned to the nearest modeling "
                "subcatchment for rainfall accounting only."
            ),
        },
        "blocking_failure_count": int(len(bf)),
        "quality_failure_count": int(len(qf)),
        "outputs": {
            "zone_raster": str(zone_tif),
            "lookup_csv": str(lookup_csv),
            "qc_csv": str(qc_csv),
        },
    }

    atomic_json(metadata, metadata_json)

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {BUILD}")
    print("MODEL IMPROVEMENT A2.4 - MRMS SUBCATCHMENT RAINFALL GRID")
    print("=" * 100)
    print(f"MRMS grid                          : {width} x {height}")
    print(
        f"Resolution                         : "
        f"{abs(transform.a):.6f} x {abs(transform.e):.6f} m"
    )
    print(f"Watershed cells                    : {basin_cells:,}")
    print(f"Subcatchments                      : {expected_count}")
    print()
    print("ZONE COVERAGE")
    print("-" * 100)
    print(
        f"Raw polygon-assigned cells         : "
        f"{raw_assigned_cells:,}"
    )
    print(
        f"Raw unassigned basin cells         : "
        f"{raw_unassigned_cells:,}"
    )
    print(
        f"Raw polygon zone coverage          : "
        f"{raw_coverage_pct:.6f} %"
    )
    print(
        f"Residual nearest-zone fill         : "
        f"{residual_fill_pct:.6f} %"
    )
    print(
        f"Final unassigned basin cells       : "
        f"{final_unassigned}"
    )
    print(
        f"Subcatchments represented          : "
        f"{represented}/{expected_count}"
    )
    print(
        f"Cells per subcatchment min / max   : "
        f"{min_cells} / {max_cells}"
    )
    print()
    print("READINESS")
    print("-" * 100)
    print(f"Blocking failures                  : {len(bf)}")
    print(f"Quality failures                   : {len(qf)}")
    print(f"Safe for spatial MRMS extraction   : {'YES' if safe else 'NO'}")
    print(f"Status                             : {status}")
    print(f"Zone raster                        : {zone_tif}")
    print(f"Lookup                             : {lookup_csv}")
    print(f"Metadata                           : {metadata_json}")

    if not safe:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
