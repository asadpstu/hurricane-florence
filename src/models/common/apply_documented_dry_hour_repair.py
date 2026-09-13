#!/usr/bin/env python3
"""Apply an evidence-checked single-hour dry-rainfall repair.

This is NOT temporal interpolation. It is intended for rare historical MRMS
hours where the source file is unavailable but a surrounding block of observed
MRMS hours demonstrates basin-wide zero rainfall (including the basin maximum).

The script:
1. verifies the target hour exists,
2. verifies all available surrounding evidence hours are effectively dry,
3. backs up the basin and optional spatial rainfall tables,
4. sets the target rainfall to exactly 0 mm with explicit provenance,
5. removes the target from missing-hour reports when present,
6. writes a reproducibility audit JSON/CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


PRODUCT = "DRY_HOUR_INFERRED_ZERO"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--basin-rainfall", type=Path, required=True)
    p.add_argument("--target-time", required=True)
    p.add_argument("--evidence-start", required=True)
    p.add_argument("--evidence-end", required=True)
    p.add_argument("--dry-threshold-mm", type=float, default=0.001)
    p.add_argument("--minimum-evidence-hours", type=int, default=8)
    p.add_argument("--spatial-rainfall", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite-audit", action="store_true")
    return p.parse_args()


def utc(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def backup(path: Path) -> Path:
    out = path.with_suffix(path.suffix + ".pre_documented_dry_repair.bak")
    if not out.exists():
        shutil.copy2(path, out)
    return out


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".partial")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main():
    a = parse_args()
    target = utc(a.target_time)
    ev_start = utc(a.evidence_start)
    ev_end = utc(a.evidence_end)
    if not (ev_start < target < ev_end):
        raise RuntimeError("Target must lie strictly inside evidence window.")

    a.output_dir.mkdir(parents=True, exist_ok=True)
    audit_json = a.output_dir / "documented_dry_hour_repair.json"
    audit_csv = a.output_dir / "documented_dry_hour_repair_evidence.csv"
    if (audit_json.exists() or audit_csv.exists()) and not a.overwrite_audit:
        raise RuntimeError("Audit outputs exist; use --overwrite-audit to replace them.")

    basin = pd.read_csv(a.basin_rainfall)
    required = {
        "interval_end_utc",
        "status",
        "basin_mean_rainfall_mm",
        "basin_max_gridcell_rainfall_mm",
    }
    missing_cols = required - set(basin.columns)
    if missing_cols:
        raise RuntimeError(f"Basin rainfall missing columns: {sorted(missing_cols)}")

    basin["_time"] = pd.to_datetime(
        basin["interval_end_utc"], utc=True, errors="raise"
    )
    target_mask = basin["_time"].eq(target)
    if int(target_mask.sum()) != 1:
        raise RuntimeError(
            f"Expected exactly one target row, found {int(target_mask.sum())}"
        )

    evidence = basin[
        basin["_time"].between(ev_start, ev_end, inclusive="both")
        & ~basin["_time"].eq(target)
        & basin["status"].astype(str).eq("AVAILABLE")
    ].copy()

    if len(evidence) < a.minimum_evidence_hours:
        raise RuntimeError(
            f"Only {len(evidence)} available evidence hours; "
            f"minimum={a.minimum_evidence_hours}"
        )

    ev_mean = pd.to_numeric(
        evidence["basin_mean_rainfall_mm"], errors="coerce"
    )
    ev_max = pd.to_numeric(
        evidence["basin_max_gridcell_rainfall_mm"], errors="coerce"
    )
    if ev_mean.isna().any() or ev_max.isna().any():
        raise RuntimeError("Evidence hours contain missing basin rainfall values.")

    max_observed_mean = float(ev_mean.max())
    max_observed_grid = float(ev_max.max())
    if (
        max_observed_mean > a.dry_threshold_mm
        or max_observed_grid > a.dry_threshold_mm
    ):
        raise RuntimeError(
            "Evidence window is not dry enough: "
            f"max basin mean={max_observed_mean:.9g} mm; "
            f"max gridcell={max_observed_grid:.9g} mm; "
            f"threshold={a.dry_threshold_mm} mm"
        )

    target_before = basin.loc[target_mask].iloc[0].to_dict()
    basin_backup = backup(a.basin_rainfall)

    patch = {
        "status": "AVAILABLE",
        "source_product": PRODUCT,
        "used_fallback": False,
        "basin_valid_coverage_percent": 100.0,
        "basin_mean_rainfall_mm": 0.0,
        "basin_min_gridcell_rainfall_mm": 0.0,
        "basin_p95_gridcell_rainfall_mm": 0.0,
        "basin_max_gridcell_rainfall_mm": 0.0,
        "compressed_bytes": 0,
        "error": np.nan,
        "repair_method": PRODUCT,
        "repair_reason": (
            "Authoritative GaugeCorr and RadarOnly source unavailable for this exact hour; "
            "surrounding observed MRMS hours are basin-wide dry and satisfy the configured "
            "dry threshold."
        ),
    }
    for col, value in patch.items():
        if col not in basin.columns:
            if isinstance(value, str):
                basin[col] = pd.Series([None] * len(basin), dtype="object")
            else:
                basin[col] = np.nan
        elif isinstance(value, str):
            basin[col] = basin[col].astype("object")
        basin.loc[target_mask, col] = value

    basin = basin.drop(columns=["_time"])
    atomic_csv(basin, a.basin_rainfall)

    basin_missing = a.basin_rainfall.parent / "missing_hours.csv"
    if basin_missing.exists():
        miss = pd.read_csv(basin_missing)
        if "interval_end_utc" in miss.columns:
            mt = pd.to_datetime(
                miss["interval_end_utc"], utc=True, errors="coerce"
            )
            miss = miss.loc[~mt.eq(target)].copy()
            atomic_csv(miss, basin_missing)

    spatial_backup = None
    spatial_patched = False
    spatial_rain_cols = []
    if a.spatial_rainfall is not None and a.spatial_rainfall.exists():
        spatial = pd.read_csv(a.spatial_rainfall)
        if "interval_end_utc" not in spatial.columns:
            raise RuntimeError("Spatial rainfall lacks interval_end_utc.")
        st = pd.to_datetime(
            spatial["interval_end_utc"], utc=True, errors="raise"
        )
        smask = st.eq(target)
        if int(smask.sum()) != 1:
            raise RuntimeError(
                f"Expected exactly one spatial target row, found {int(smask.sum())}"
            )

        spatial_rain_cols = [
            c for c in spatial.columns
            if c.startswith("rain_") and c.endswith("_mm")
        ]
        if not spatial_rain_cols:
            raise RuntimeError(
                "No spatial subcatchment rainfall columns detected."
            )

        spatial_backup = backup(a.spatial_rainfall)
        spatial_patch = {
            "reference_basin_mean_rainfall_mm": 0.0,
            "reference_source_product": PRODUCT,
            "status": "OK",
            "source_product": PRODUCT,
            "source_url": "manual://documented-dry-hour",
            "min_zone_valid_coverage_percent": 100.0,
            "reconstructed_basin_mean_rainfall_mm": 0.0,
            "basin_mean_difference_mm": 0.0,
            "error": np.nan,
        }
        for col, value in spatial_patch.items():
            if col not in spatial.columns:
                if isinstance(value, str):
                    spatial[col] = pd.Series(
                        [None] * len(spatial), dtype="object"
                    )
                else:
                    spatial[col] = np.nan
            elif isinstance(value, str):
                spatial[col] = spatial[col].astype("object")
            spatial.loc[smask, col] = value

        spatial.loc[smask, spatial_rain_cols] = 0.0
        atomic_csv(spatial, a.spatial_rainfall)
        spatial_patched = True

        spatial_missing = (
            a.spatial_rainfall.parent
            / "spatial_subcatchment_missing_hours.csv"
        )
        if spatial_missing.exists():
            miss = pd.read_csv(spatial_missing)
            if "interval_end_utc" in miss.columns:
                mt = pd.to_datetime(
                    miss["interval_end_utc"], utc=True, errors="coerce"
                )
                miss = miss.loc[~mt.eq(target)].copy()
                atomic_csv(miss, spatial_missing)

    evidence_out = evidence.drop(columns=["_time"]).copy()
    evidence_out.insert(0, "repair_target_utc", target.isoformat())
    atomic_csv(evidence_out, audit_csv)

    audit = {
        "status": "PASS_DOCUMENTED_DRY_HOUR_REPAIR_APPLIED",
        "target_time_utc": target.isoformat(),
        "repair_product": PRODUCT,
        "repair_value_mm": 0.0,
        "method": "evidence_checked_dry_hour_zero; not temporal interpolation",
        "evidence_window": {
            "start_utc": ev_start.isoformat(),
            "end_utc": ev_end.isoformat(),
            "available_evidence_hours": int(len(evidence)),
            "dry_threshold_mm": float(a.dry_threshold_mm),
            "maximum_observed_basin_mean_mm": max_observed_mean,
            "maximum_observed_gridcell_mm": max_observed_grid,
        },
        "target_before": {
            k: (None if pd.isna(v) else v)
            for k, v in target_before.items()
            if k != "_time"
        },
        "basin_rainfall": str(a.basin_rainfall),
        "basin_backup": str(basin_backup),
        "spatial_rainfall": (
            str(a.spatial_rainfall) if a.spatial_rainfall else None
        ),
        "spatial_backup": (
            str(spatial_backup) if spatial_backup else None
        ),
        "spatial_patched": spatial_patched,
        "spatial_subcatchment_columns_zeroed": len(spatial_rain_cols),
        "scientific_note": (
            "The missing hour is assigned zero only because the entire surrounding "
            "observed MRMS block is dry at both basin-mean and basin-maximum levels. "
            "This repair remains explicit in provenance."
        ),
    }
    audit_json.write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )

    print("=" * 100)
    print("DOCUMENTED DRY-HOUR MRMS REPAIR")
    print("=" * 100)
    print(f"Target hour                        : {target}")
    print(f"Evidence available hours           : {len(evidence)}")
    print(
        f"Max evidence basin mean            : "
        f"{max_observed_mean:.12g} mm"
    )
    print(
        f"Max evidence gridcell              : "
        f"{max_observed_grid:.12g} mm"
    )
    print(
        f"Dry threshold                      : "
        f"{a.dry_threshold_mm} mm"
    )
    print("Basin archive patched              : YES")
    print(
        f"Spatial archive patched            : "
        f"{'YES' if spatial_patched else 'NO'}"
    )
    print(f"Audit JSON                         : {audit_json}")
    print("Status                             : PASS_DOCUMENTED_DRY_HOUR_REPAIR_APPLIED")


if __name__ == "__main__":
    main()
