#!/usr/bin/env python3
"""Conservatively repair a specifically documented dry MRMS hour with zero rainfall."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", type=Path, required=True)
    p.add_argument("--timestamp", required=True)
    p.add_argument("--context-hours", type=int, default=6)
    p.add_argument("--dry-threshold-mm", type=float, default=0.05)
    return p.parse_args()


def utc(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def main():
    a = args()
    path = a.archive_dir / "multiyear_basin_hourly_rainfall.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    tc = "interval_end_utc" if "interval_end_utc" in df else "timestamp"
    df[tc] = pd.to_datetime(df[tc], utc=True, errors="raise")
    target = utc(a.timestamp)

    hit = np.flatnonzero(df[tc].eq(target).to_numpy())
    if len(hit) != 1:
        raise RuntimeError(
            f"Expected exactly one archive row at {target}; found {len(hit)}"
        )

    i = int(hit[0])
    rain = "basin_mean_rainfall_mm"
    before = df[
        (df[tc] >= target - pd.Timedelta(hours=a.context_hours))
        & (df[tc] < target)
    ]
    after = df[
        (df[tc] > target)
        & (df[tc] <= target + pd.Timedelta(hours=a.context_hours))
    ]

    for label, x in [("before", before), ("after", after)]:
        vals = pd.to_numeric(x[rain], errors="coerce").dropna()
        if len(vals) < a.context_hours or float(vals.max()) > a.dry_threshold_mm:
            raise RuntimeError(
                f"Refusing repair: {label} context is not demonstrably dry/completed."
            )

    old = df.loc[i].copy()
    if str(old.get("status", "")).upper() == "AVAILABLE" and pd.notna(
        old.get(rain)
    ):
        print("Target hour is already available; no repair required.")
        return

    backup = (
        a.archive_dir
        / "multiyear_basin_hourly_rainfall_before_documented_dry_repair.csv"
    )
    if not backup.exists():
        pd.read_csv(path).to_csv(backup, index=False)

    df.loc[i, rain] = 0.0
    for c in [
        "basin_min_gridcell_rainfall_mm",
        "basin_p95_gridcell_rainfall_mm",
        "basin_max_gridcell_rainfall_mm",
    ]:
        if c in df:
            df.loc[i, c] = 0.0

    if "basin_valid_coverage_percent" in df:
        df.loc[i, "basin_valid_coverage_percent"] = 100.0
    if "status" in df:
        df.loc[i, "status"] = "AVAILABLE"
    if "source_product" in df:
        df.loc[i, "source_product"] = "DRY_HOUR_INFERRED_ZERO"
    if "used_fallback" in df:
        df.loc[i, "used_fallback"] = False
    if "error" in df:
        df.loc[i, "error"] = (
            "documented dry-hour zero repair after dry-context guard"
        )

    tmp = path.with_suffix(".csv.partial")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

    log = a.archive_dir / "documented_dry_rainfall_gap_fills.csv"
    row = pd.DataFrame(
        [
            {
                "interval_end_utc": target.isoformat(),
                "repair_mm": 0.0,
                "context_hours": a.context_hours,
                "dry_threshold_mm": a.dry_threshold_mm,
                "method": "guarded_zero_fill",
            }
        ]
    )
    if log.exists():
        row = (
            pd.concat([pd.read_csv(log), row], ignore_index=True)
            .drop_duplicates("interval_end_utc", keep="last")
        )
    row.to_csv(log, index=False)

    miss = a.archive_dir / "missing_hours.csv"
    if miss.exists():
        m = pd.read_csv(miss)
        cols = [c for c in m.columns if "time" in c.lower()]
        if cols:
            mt = pd.to_datetime(m[cols[0]], utc=True, errors="coerce")
            m = m.loc[~mt.eq(target)]
            m.to_csv(miss, index=False)

    print(f"PASS_DOCUMENTED_DRY_HOUR_REPAIR {target.isoformat()}")


if __name__ == "__main__":
    main()
