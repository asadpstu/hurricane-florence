"""
STEP 5C - Basin rainfall time series, storm statistics, and final rainfall QC.

Purpose
-------
Close the Hurricane Florence rainfall-forcing module by converting the 624
standardized MRMS hourly fields from Step 5B into hydrologically meaningful
basin rainfall diagnostics over the FULL USGS 02089000 upstream watershed.

Processing
----------
1. Validate Step 5B metadata and hourly spatial QC.
2. Read each standardized 1-km MRMS rainfall raster.
3. Compute equal-area basin mean rainfall from the projected watershed mask.
   The 1-km rasterized basin area differs from the Step 2A vector area by only
   ~0.13%, so no ad-hoc boundary correction is applied.
4. Build hourly and daily basin rainfall time series.
5. Calculate:
   - full forcing-window rainfall,
   - antecedent rainfall,
   - formal Florence evaluation-period rainfall,
   - post-evaluation rainfall,
   - 3/6/12/24-hour rolling maxima,
   - rainfall temporal centroid and 50%-mass time.
6. Accumulate spatial rainfall rasters for the full and evaluation windows.
7. Auto-discover the validated Step 1 USGS event-observation CSV and extract
   discharge for timing comparison.
8. Report rainfall-to-discharge timing markers without treating them as model
   calibration.
9. Perform final Step 5 QC.

Default temporal interpretation in the command
----------------------------------------------
Forcing:
  2018-09-01 00:00 UTC through 2018-09-27 00:00 UTC (end-exclusive)

Formal Florence evaluation:
  2018-09-10 00:00 UTC through 2018-09-26 00:00 UTC (end-exclusive)
This includes September 10-25 inclusive.

Outputs
-------
output/rainfall/mrms_florence_2018_basin/
  basin_hourly_rainfall.csv
  basin_daily_rainfall.csv
  rainfall_event_summary.csv
  rainfall_discharge_timing.csv
  rainfall_spatial_summary.csv
  florence_evaluation_cumulative_rainfall_1km.tif
  full_forcing_cumulative_rainfall_1km.tif
  rainfall_final_qc.csv
  rainfall_final_metadata.json
  basin_rainfall_hydrograph.png
  evaluation_cumulative_rainfall_quicklook.png
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


SCRIPT_BUILD = "STEP_5C_MRMS_BASIN_RAINFALL_FINAL_QC_V2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hourly-spatial-qc",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--standardization-metadata",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--watershed-mask",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--usgs-observation-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--evaluation-start",
        required=True,
    )
    p.add_argument(
        "--evaluation-end",
        required=True,
        help="UTC end-exclusive.",
    )
    p.add_argument(
        "--min-observed-q-hourly-coverage-percent",
        type=float,
        default=95.0,
    )
    p.add_argument(
        "--min-evaluation-rainfall-mm",
        type=float,
        default=25.0,
    )
    p.add_argument(
        "--max-evaluation-rainfall-mm",
        type=float,
        default=1500.0,
    )
    p.add_argument(
        "--max-basin-hourly-rainfall-mm",
        type=float,
        default=150.0,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_utc(text: str) -> pd.Timestamp:
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def iso_z(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


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


def prepare_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{path} contains outputs. Use --overwrite."
        )
    if overwrite:
        for child in path.iterdir():
            if child.is_file():
                child.unlink()


def detect_time_column(columns: list[str]) -> str | None:
    priority = [
        "datetime_utc",
        "timestamp_utc",
        "time_utc",
        "date_time_utc",
        "datetime",
        "timestamp",
        "dateTime",
        "time",
    ]
    lookup = {c.lower(): c for c in columns}

    for candidate in priority:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    # Conservative fallback.
    for c in columns:
        low = c.lower()
        if (
            "datetime" in low
            or "timestamp" in low
            or ("time" in low and "timezone" not in low)
        ):
            return c

    return None


def detect_discharge_column(
    columns: list[str],
) -> tuple[str | None, str | None]:
    """
    Return (column, unit), where unit is m3/s or cfs.
    """
    # Strong metric names first.
    metric_patterns = [
        r"^discharge_m3s$",
        r"^discharge_m3_s$",
        r"^discharge_cms$",
        r"^q_m3s$",
        r"^q_cms$",
    ]
    for pattern in metric_patterns:
        for c in columns:
            if re.search(pattern, c.lower()):
                return c, "m3/s"

    # Flexible metric search.
    for c in columns:
        low = c.lower()
        if (
            ("discharge" in low or low.startswith("q_"))
            and (
                "m3" in low
                or "cms" in low
                or "cumec" in low
            )
        ):
            return c, "m3/s"

    # CFS names.
    cfs_patterns = [
        r"^discharge_cfs$",
        r"^q_cfs$",
    ]
    for pattern in cfs_patterns:
        for c in columns:
            if re.search(pattern, c.lower()):
                return c, "cfs"

    for c in columns:
        low = c.lower()
        if (
            ("discharge" in low or low.startswith("q_"))
            and "cfs" in low
        ):
            return c, "cfs"

    return None, None


def find_usgs_observations(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = []

    for csv_path in sorted(root.rglob("*.csv")):
        try:
            sample = pd.read_csv(csv_path, nrows=50)
        except Exception:
            continue

        tcol = detect_time_column(list(sample.columns))
        qcol, qunit = detect_discharge_column(
            list(sample.columns)
        )

        if tcol is None or qcol is None:
            continue

        score = 100
        low_name = csv_path.name.lower()
        if "observation" in low_name:
            score += 30
        if "event" in low_name:
            score += 20
        if "florence" in low_name:
            score += 20
        if "02089000" in low_name:
            score += 20
        if qunit == "m3/s":
            score += 10

        candidates.append(
            {
                "path": csv_path,
                "time_column": tcol,
                "discharge_column": qcol,
                "discharge_unit": qunit,
                "score": score,
            }
        )

    if not candidates:
        raise RuntimeError(
            f"No CSV under {root} contains both a recognizable "
            "timestamp and discharge column."
        )

    candidates.sort(
        key=lambda x: (-x["score"], str(x["path"]))
    )
    selected = candidates[0]

    df = pd.read_csv(selected["path"])
    tcol = selected["time_column"]
    qcol = selected["discharge_column"]

    time = pd.to_datetime(
        df[tcol],
        utc=True,
        errors="coerce",
    )
    q = pd.to_numeric(
        df[qcol],
        errors="coerce",
    )

    if selected["discharge_unit"] == "cfs":
        q = q * 0.028316846592

    out = pd.DataFrame(
        {
            "time_utc": time,
            "discharge_m3s": q,
        }
    ).dropna()

    out = (
        out.sort_values("time_utc")
        .drop_duplicates("time_utc")
        .reset_index(drop=True)
    )

    if out.empty:
        raise RuntimeError(
            f"Selected USGS CSV contains no valid Q/time rows: "
            f"{selected['path']}"
        )

    info = {
        "selected_path": str(selected["path"]),
        "time_column": tcol,
        "discharge_column": qcol,
        "original_discharge_unit": selected["discharge_unit"],
        "valid_rows": int(len(out)),
        "candidate_count": len(candidates),
    }
    return out, info


def write_cumulative_raster(
    path: Path,
    array: np.ndarray,
    valid: np.ndarray,
    profile: dict[str, Any],
    description: str,
    tags: dict[str, str],
) -> None:
    out = np.full(
        array.shape,
        -9999.0,
        dtype="float32",
    )
    out[valid] = array[valid].astype("float32")

    p = profile.copy()
    p.pop("compress", None)
    p.pop("predictor", None)
    p.update(
        dtype="float32",
        nodata=-9999.0,
        compress="DEFLATE",
        predictor=1,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, description)
        dst.update_tags(**tags)


def weighted_time_centroid(
    times: pd.Series,
    weights: pd.Series,
) -> pd.Timestamp | None:
    """
    Return a precipitation-mass-weighted UTC timestamp.

    IMPORTANT:
    Do not use ``times.astype("int64")`` here. In newer pandas builds,
    timezone-aware datetime arrays may carry a microsecond internal unit, so
    the resulting integers are not guaranteed to be nanoseconds. Passing such
    values directly to pd.Timestamp(int) makes a valid 2018 date appear near
    1970.

    Timestamp.value is explicitly nanoseconds since Unix epoch, making the
    calculation independent of pandas' internal datetime resolution.
    """
    w = weights.to_numpy(dtype="float64")
    if len(w) == 0 or np.nansum(w) <= 0:
        return None

    ns = np.array(
        [
            pd.Timestamp(t).value
            for t in times
        ],
        dtype="float64",
    )

    valid = np.isfinite(w) & np.isfinite(ns)
    if not valid.any() or np.sum(w[valid]) <= 0:
        return None

    centroid_ns = (
        np.sum(ns[valid] * w[valid])
        / np.sum(w[valid])
    )

    return pd.Timestamp(
        int(round(centroid_ns)),
        unit="ns",
        tz="UTC",
    )


def main() -> None:
    args = parse_args()

    evaluation_start = parse_utc(args.evaluation_start)
    evaluation_end = parse_utc(args.evaluation_end)
    if evaluation_end <= evaluation_start:
        raise ValueError(
            "--evaluation-end must be later than --evaluation-start."
        )

    prepare_dir(args.output_dir, args.overwrite)

    standard_meta = json.loads(
        args.standardization_metadata.read_text(
            encoding="utf-8"
        )
    )
    if not bool(standard_meta.get("safe_for_step_5c")):
        raise RuntimeError(
            "Step 5B metadata does not authorize Step 5C."
        )

    hourly_qc = pd.read_csv(args.hourly_spatial_qc)
    if hourly_qc.empty:
        raise RuntimeError(
            "Step 5B hourly spatial QC table is empty."
        )

    hourly_qc["time_utc"] = pd.to_datetime(
        hourly_qc["valid_time_utc"],
        utc=True,
        errors="coerce",
    )
    if hourly_qc["time_utc"].isna().any():
        raise RuntimeError(
            "Step 5B hourly spatial QC contains invalid timestamps."
        )

    hourly_qc = (
        hourly_qc.sort_values("time_utc")
        .reset_index(drop=True)
    )

    expected_hours = int(
        standard_meta.get(
            "hourly_standardization",
            {},
        ).get("hour_count", len(hourly_qc))
    )

    with rasterio.open(args.watershed_mask) as mask_ds:
        basin_mask = mask_ds.read(1) > 0
        mask_profile = mask_ds.profile.copy()
        mask_transform = mask_ds.transform
        mask_crs = mask_ds.crs

    mask_cells = int(basin_mask.sum())
    if mask_cells <= 0:
        raise RuntimeError(
            "Watershed mask has zero valid cells."
        )

    pixel_area_m2 = (
        abs(float(mask_transform.a))
        * abs(float(mask_transform.e))
    )
    rasterized_basin_area_km2 = (
        mask_cells * pixel_area_m2 / 1_000_000.0
    )

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {SCRIPT_BUILD}")
    print("STEP 5C - BASIN RAINFALL SERIES + FINAL RAINFALL QC")
    print("=" * 100)
    print(f"Standardized hourly rasters        : {len(hourly_qc):,}")
    print(f"Rasterized watershed area          : {rasterized_basin_area_km2:.3f} km2")
    print(f"Evaluation start UTC               : {iso_z(evaluation_start)}")
    print(f"Evaluation end UTC                 : {iso_z(evaluation_end)}")
    print()

    series_rows = []
    cumulative_full = np.zeros(
        basin_mask.shape,
        dtype="float64",
    )
    cumulative_eval = np.zeros(
        basin_mask.shape,
        dtype="float64",
    )
    valid_count_full = np.zeros(
        basin_mask.shape,
        dtype="uint16",
    )
    valid_count_eval = np.zeros(
        basin_mask.shape,
        dtype="uint16",
    )

    source_profile = None
    previous_grid = None

    for i, row in hourly_qc.iterrows():
        raster_path = Path(str(row["output_file"]))
        if not raster_path.exists():
            raise FileNotFoundError(
                f"Step 5B hourly raster missing: {raster_path}"
            )

        with rasterio.open(raster_path) as ds:
            if source_profile is None:
                source_profile = ds.profile.copy()
                previous_grid = (
                    ds.crs,
                    ds.transform,
                    ds.width,
                    ds.height,
                )
            else:
                grid = (
                    ds.crs,
                    ds.transform,
                    ds.width,
                    ds.height,
                )
                if grid != previous_grid:
                    raise RuntimeError(
                        f"Hourly raster grid changed: {raster_path}"
                    )

            if (
                ds.crs != mask_crs
                or ds.transform != mask_transform
                or ds.width != basin_mask.shape[1]
                or ds.height != basin_mask.shape[0]
            ):
                raise RuntimeError(
                    f"Hourly raster does not align with watershed mask: "
                    f"{raster_path}"
                )

            arr = ds.read(1, masked=True)
            data = np.asarray(
                arr.data,
                dtype="float64",
            )
            valid = (
                basin_mask
                & ~np.ma.getmaskarray(arr)
                & np.isfinite(data)
                & (data >= 0.0)
            )

        coverage_pct = (
            valid.sum() / mask_cells * 100.0
        )
        values = data[valid]

        if values.size == 0:
            basin_mean = np.nan
            basin_min = np.nan
            basin_max = np.nan
            basin_p95 = np.nan
        else:
            # Equal-area 1-km UTM cells: arithmetic grid-cell mean is
            # the area-weighted basin mean over the rasterized basin.
            basin_mean = float(np.mean(values))
            basin_min = float(np.min(values))
            basin_max = float(np.max(values))
            basin_p95 = float(
                np.percentile(values, 95)
            )

        time_utc = row["time_utc"]

        cumulative_full[valid] += data[valid]
        valid_count_full[valid] += 1

        in_eval = (
            time_utc >= evaluation_start
            and time_utc < evaluation_end
        )
        if in_eval:
            cumulative_eval[valid] += data[valid]
            valid_count_eval[valid] += 1

        series_rows.append(
            {
                "time_utc": time_utc,
                "basin_mean_rainfall_mm": basin_mean,
                "basin_min_gridcell_rainfall_mm": basin_min,
                "basin_p95_gridcell_rainfall_mm": basin_p95,
                "basin_max_gridcell_rainfall_mm": basin_max,
                "basin_valid_coverage_percent": coverage_pct,
                "source_product": row.get("source_product"),
                "source_raster": str(raster_path),
                "is_evaluation_period": in_eval,
            }
        )

    hourly = pd.DataFrame(series_rows)
    hourly = hourly.sort_values("time_utc").reset_index(drop=True)

    # Temporal continuity.
    dt_hours = (
        hourly["time_utc"]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 3600.0
    )
    nonhourly_gaps = int(
        (~np.isclose(dt_hours, 1.0)).sum()
    )

    hourly["cumulative_forcing_rainfall_mm"] = (
        hourly["basin_mean_rainfall_mm"].cumsum()
    )

    eval_mask = (
        (hourly["time_utc"] >= evaluation_start)
        & (hourly["time_utc"] < evaluation_end)
    )
    eval_hourly = hourly.loc[eval_mask].copy()

    expected_eval_hours = int(
        (evaluation_end - evaluation_start)
        / pd.Timedelta(hours=1)
    )

    # Rolling rainfall ending at each hour.
    for window in [3, 6, 12, 24]:
        hourly[f"rolling_{window}h_rainfall_mm"] = (
            hourly["basin_mean_rainfall_mm"]
            .rolling(window, min_periods=window)
            .sum()
        )

    # Period totals.
    forcing_start = hourly["time_utc"].min()
    forcing_end_exclusive = (
        hourly["time_utc"].max()
        + pd.Timedelta(hours=1)
    )

    antecedent = hourly[
        hourly["time_utc"] < evaluation_start
    ]
    post_eval = hourly[
        hourly["time_utc"] >= evaluation_end
    ]

    full_total = float(
        hourly["basin_mean_rainfall_mm"].sum()
    )
    antecedent_total = float(
        antecedent["basin_mean_rainfall_mm"].sum()
    )
    evaluation_total = float(
        eval_hourly["basin_mean_rainfall_mm"].sum()
    )
    post_total = float(
        post_eval["basin_mean_rainfall_mm"].sum()
    )

    peak_hour_idx = hourly[
        "basin_mean_rainfall_mm"
    ].idxmax()
    peak_hour_row = hourly.loc[peak_hour_idx]

    roll_records = {}
    for window in [3, 6, 12, 24]:
        col = f"rolling_{window}h_rainfall_mm"
        subset = hourly.loc[
            eval_mask & hourly[col].notna(),
            ["time_utc", col],
        ]
        if subset.empty:
            roll_records[window] = {
                "amount_mm": None,
                "window_end_utc": None,
                "window_start_utc": None,
            }
        else:
            idx = subset[col].idxmax()
            end_time = hourly.loc[idx, "time_utc"]
            amount = float(hourly.loc[idx, col])
            roll_records[window] = {
                "amount_mm": amount,
                "window_end_utc": iso_z(end_time),
                "window_start_utc": iso_z(
                    end_time
                    - pd.Timedelta(hours=window - 1)
                ),
            }

    # Evaluation-period rainfall temporal mass metrics.
    eval_positive = eval_hourly[
        "basin_mean_rainfall_mm"
    ].clip(lower=0.0)

    rainfall_centroid = weighted_time_centroid(
        eval_hourly["time_utc"],
        eval_positive,
    )

    eval_cumulative_series = eval_positive.cumsum()
    rainfall_50pct_time = None
    if evaluation_total > 0:
        half = evaluation_total * 0.5
        hit = np.where(
            eval_cumulative_series.to_numpy() >= half
        )[0]
        if len(hit):
            rainfall_50pct_time = (
                eval_hourly.iloc[int(hit[0])][
                    "time_utc"
                ]
            )

    # Daily UTC rainfall.
    daily = (
        hourly.set_index("time_utc")[
            "basin_mean_rainfall_mm"
        ]
        .resample("1D")
        .sum()
        .rename("basin_rainfall_mm")
        .to_frame()
        .reset_index()
    )
    daily["is_evaluation_period_day"] = (
        (daily["time_utc"] >= evaluation_start.floor("D"))
        & (daily["time_utc"] < evaluation_end)
    )

    # ------------------------------------------------------------------
    # USGS discharge timing diagnostics.
    # ------------------------------------------------------------------
    usgs, usgs_info = find_usgs_observations(
        args.usgs_observation_dir
    )

    q_eval = usgs[
        (usgs["time_utc"] >= evaluation_start)
        & (usgs["time_utc"] < evaluation_end)
    ].copy()

    if q_eval.empty:
        raise RuntimeError(
            "USGS observations do not overlap the evaluation period."
        )

    peak_q_idx = q_eval["discharge_m3s"].idxmax()
    peak_q_time = q_eval.loc[peak_q_idx, "time_utc"]
    peak_q = float(
        q_eval.loc[peak_q_idx, "discharge_m3s"]
    )

    q_hourly = (
        q_eval.set_index("time_utc")[
            "discharge_m3s"
        ]
        .resample("1h")
        .mean()
    )

    expected_q_hours = pd.date_range(
        evaluation_start,
        evaluation_end - pd.Timedelta(hours=1),
        freq="1h",
        tz="UTC",
    )
    q_hourly = q_hourly.reindex(expected_q_hours)
    q_hourly_coverage_pct = float(
        q_hourly.notna().mean() * 100.0
    )

    def lag_hours(
        earlier: pd.Timestamp | None,
        later: pd.Timestamp,
    ) -> float | None:
        if earlier is None:
            return None
        return float(
            (later - earlier).total_seconds()
            / 3600.0
        )

    timing_rows = [
        {
            "marker": "peak_basin_hourly_rainfall",
            "rainfall_time_utc": iso_z(
                peak_hour_row["time_utc"]
            ),
            "rainfall_metric_mm": float(
                peak_hour_row[
                    "basin_mean_rainfall_mm"
                ]
            ),
            "peak_discharge_time_utc": iso_z(
                peak_q_time
            ),
            "peak_discharge_m3s": peak_q,
            "lag_to_peak_discharge_hours": lag_hours(
                peak_hour_row["time_utc"],
                peak_q_time,
            ),
            "interpretation": "timing diagnostic only",
        },
        {
            "marker": "evaluation_rainfall_temporal_centroid",
            "rainfall_time_utc": (
                iso_z(rainfall_centroid)
                if rainfall_centroid is not None
                else None
            ),
            "rainfall_metric_mm": evaluation_total,
            "peak_discharge_time_utc": iso_z(
                peak_q_time
            ),
            "peak_discharge_m3s": peak_q,
            "lag_to_peak_discharge_hours": lag_hours(
                rainfall_centroid,
                peak_q_time,
            ),
            "interpretation": "timing diagnostic only",
        },
        {
            "marker": "evaluation_rainfall_50pct_mass_time",
            "rainfall_time_utc": (
                iso_z(rainfall_50pct_time)
                if rainfall_50pct_time is not None
                else None
            ),
            "rainfall_metric_mm": (
                evaluation_total * 0.5
            ),
            "peak_discharge_time_utc": iso_z(
                peak_q_time
            ),
            "peak_discharge_m3s": peak_q,
            "lag_to_peak_discharge_hours": lag_hours(
                rainfall_50pct_time,
                peak_q_time,
            ),
            "interpretation": "timing diagnostic only",
        },
    ]

    for window in [6, 12, 24]:
        rec = roll_records[window]
        end_time = (
            parse_utc(rec["window_end_utc"])
            if rec["window_end_utc"] is not None
            else None
        )
        timing_rows.append(
            {
                "marker": f"peak_{window}h_basin_rainfall_window_end",
                "rainfall_time_utc": rec["window_end_utc"],
                "rainfall_metric_mm": rec["amount_mm"],
                "peak_discharge_time_utc": iso_z(
                    peak_q_time
                ),
                "peak_discharge_m3s": peak_q,
                "lag_to_peak_discharge_hours": lag_hours(
                    end_time,
                    peak_q_time,
                ),
                "interpretation": "timing diagnostic only",
            }
        )

    timing = pd.DataFrame(timing_rows)

    # ------------------------------------------------------------------
    # Spatial cumulative rainfall statistics.
    # ------------------------------------------------------------------
    full_valid = (
        basin_mask
        & (valid_count_full == len(hourly))
    )
    eval_valid = (
        basin_mask
        & (valid_count_eval == expected_eval_hours)
    )

    full_values = cumulative_full[full_valid]
    eval_values = cumulative_eval[eval_valid]

    def spatial_stats(
        label: str,
        values: np.ndarray,
    ) -> dict[str, Any]:
        if values.size == 0:
            return {
                "period": label,
                "valid_cells": 0,
            }

        mean = float(np.mean(values))
        std = float(np.std(values))
        return {
            "period": label,
            "valid_cells": int(values.size),
            "min_mm": float(np.min(values)),
            "p05_mm": float(np.percentile(values, 5)),
            "p25_mm": float(np.percentile(values, 25)),
            "median_mm": float(np.percentile(values, 50)),
            "mean_mm": mean,
            "p75_mm": float(np.percentile(values, 75)),
            "p90_mm": float(np.percentile(values, 90)),
            "p95_mm": float(np.percentile(values, 95)),
            "p99_mm": float(np.percentile(values, 99)),
            "max_mm": float(np.max(values)),
            "std_mm": std,
            "coefficient_of_variation": (
                std / mean if mean > 0 else None
            ),
        }

    spatial_summary = pd.DataFrame(
        [
            spatial_stats(
                "full_forcing_window",
                full_values,
            ),
            spatial_stats(
                "florence_evaluation_window",
                eval_values,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Output tables and rasters.
    # ------------------------------------------------------------------
    hourly_path = (
        args.output_dir / "basin_hourly_rainfall.csv"
    )
    daily_path = (
        args.output_dir / "basin_daily_rainfall.csv"
    )
    event_summary_path = (
        args.output_dir / "rainfall_event_summary.csv"
    )
    timing_path = (
        args.output_dir / "rainfall_discharge_timing.csv"
    )
    spatial_summary_path = (
        args.output_dir / "rainfall_spatial_summary.csv"
    )
    eval_raster_path = (
        args.output_dir
        / "florence_evaluation_cumulative_rainfall_1km.tif"
    )
    full_raster_path = (
        args.output_dir
        / "full_forcing_cumulative_rainfall_1km.tif"
    )
    qc_path = (
        args.output_dir / "rainfall_final_qc.csv"
    )
    metadata_path = (
        args.output_dir / "rainfall_final_metadata.json"
    )
    hydrograph_path = (
        args.output_dir / "basin_rainfall_hydrograph.png"
    )
    quicklook_path = (
        args.output_dir
        / "evaluation_cumulative_rainfall_quicklook.png"
    )

    # Serialize timestamps as ISO strings.
    hourly_out = hourly.copy()
    hourly_out["time_utc"] = hourly_out[
        "time_utc"
    ].map(iso_z)
    atomic_csv(hourly_out, hourly_path)

    daily_out = daily.copy()
    daily_out["time_utc"] = daily_out[
        "time_utc"
    ].map(iso_z)
    atomic_csv(daily_out, daily_path)

    event_summary = pd.DataFrame(
        [
            {
                "metric": "forcing_window_start_utc",
                "value": iso_z(forcing_start),
                "unit": "UTC",
            },
            {
                "metric": "forcing_window_end_exclusive_utc",
                "value": iso_z(forcing_end_exclusive),
                "unit": "UTC",
            },
            {
                "metric": "forcing_window_basin_rainfall",
                "value": full_total,
                "unit": "mm",
            },
            {
                "metric": "antecedent_rainfall_before_evaluation",
                "value": antecedent_total,
                "unit": "mm",
            },
            {
                "metric": "evaluation_window_basin_rainfall",
                "value": evaluation_total,
                "unit": "mm",
            },
            {
                "metric": "post_evaluation_rainfall",
                "value": post_total,
                "unit": "mm",
            },
            {
                "metric": "peak_hourly_basin_rainfall",
                "value": float(
                    peak_hour_row[
                        "basin_mean_rainfall_mm"
                    ]
                ),
                "unit": "mm/h",
            },
            {
                "metric": "peak_hourly_basin_rainfall_time_utc",
                "value": iso_z(
                    peak_hour_row["time_utc"]
                ),
                "unit": "UTC",
            },
            {
                "metric": "evaluation_rainfall_temporal_centroid_utc",
                "value": (
                    iso_z(rainfall_centroid)
                    if rainfall_centroid is not None
                    else None
                ),
                "unit": "UTC",
            },
            {
                "metric": "evaluation_rainfall_50pct_mass_time_utc",
                "value": (
                    iso_z(rainfall_50pct_time)
                    if rainfall_50pct_time is not None
                    else None
                ),
                "unit": "UTC",
            },
            {
                "metric": "observed_peak_discharge",
                "value": peak_q,
                "unit": "m3/s",
            },
            {
                "metric": "observed_peak_discharge_time_utc",
                "value": iso_z(peak_q_time),
                "unit": "UTC",
            },
        ]
    )

    for window in [3, 6, 12, 24]:
        rec = roll_records[window]
        event_summary = pd.concat(
            [
                event_summary,
                pd.DataFrame(
                    [
                        {
                            "metric": f"peak_{window}h_basin_rainfall",
                            "value": rec["amount_mm"],
                            "unit": "mm",
                        },
                        {
                            "metric": f"peak_{window}h_window_end_utc",
                            "value": rec["window_end_utc"],
                            "unit": "UTC",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

    atomic_csv(event_summary, event_summary_path)
    atomic_csv(timing, timing_path)
    atomic_csv(spatial_summary, spatial_summary_path)

    if source_profile is None:
        raise RuntimeError(
            "No source rainfall profile was captured."
        )

    write_cumulative_raster(
        full_raster_path,
        cumulative_full,
        full_valid,
        source_profile,
        (
            "MRMS cumulative rainfall over full Step 5 forcing window"
        ),
        {
            "UNIT": "mm",
            "START_UTC": iso_z(forcing_start),
            "END_UTC_EXCLUSIVE": iso_z(forcing_end_exclusive),
            "DOMAIN": "USGS 02089000 full upstream watershed",
        },
    )

    write_cumulative_raster(
        eval_raster_path,
        cumulative_eval,
        eval_valid,
        source_profile,
        (
            "MRMS cumulative rainfall over Florence evaluation window"
        ),
        {
            "UNIT": "mm",
            "START_UTC": iso_z(evaluation_start),
            "END_UTC_EXCLUSIVE": iso_z(evaluation_end),
            "DOMAIN": "USGS 02089000 full upstream watershed",
        },
    )

    # ------------------------------------------------------------------
    # Final Step 5 QC.
    # ------------------------------------------------------------------
    minimum_hour_coverage = float(
        hourly["basin_valid_coverage_percent"].min()
    )
    missing_basin_mean_hours = int(
        hourly["basin_mean_rainfall_mm"].isna().sum()
    )
    max_basin_hourly = float(
        hourly["basin_mean_rainfall_mm"].max()
    )

    eval_raster_coverage_pct = float(
        eval_valid.sum() / mask_cells * 100.0
    )
    full_raster_coverage_pct = float(
        full_valid.sum() / mask_cells * 100.0
    )

    qc_rows = []

    def qc(
        severity: str,
        check: str,
        passed: bool,
        detail: str,
    ):
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
        "HOURLY_TIME_SERIES_COMPLETE",
        (
            len(hourly) == expected_hours
            and nonhourly_gaps == 0
            and missing_basin_mean_hours == 0
        ),
        (
            f"hours={len(hourly)}; expected={expected_hours}; "
            f"nonhourly_gaps={nonhourly_gaps}; "
            f"missing_basin_mean_hours={missing_basin_mean_hours}"
        ),
    )

    qc(
        "BLOCKING",
        "EVALUATION_WINDOW_COMPLETE",
        len(eval_hourly) == expected_eval_hours,
        (
            f"evaluation_hours={len(eval_hourly)}; "
            f"expected={expected_eval_hours}"
        ),
    )

    qc(
        "BLOCKING",
        "RAINFALL_TEMPORAL_CENTROID_WITHIN_EVALUATION_WINDOW",
        (
            rainfall_centroid is not None
            and rainfall_centroid >= evaluation_start
            and rainfall_centroid < evaluation_end
        ),
        (
            f"centroid="
            f"{iso_z(rainfall_centroid) if rainfall_centroid is not None else None}; "
            f"evaluation_start={iso_z(evaluation_start)}; "
            f"evaluation_end_exclusive={iso_z(evaluation_end)}"
        ),
    )

    qc(
        "BLOCKING",
        "RAINFALL_50PCT_TIME_WITHIN_EVALUATION_WINDOW",
        (
            rainfall_50pct_time is not None
            and rainfall_50pct_time >= evaluation_start
            and rainfall_50pct_time < evaluation_end
        ),
        (
            f"rainfall_50pct_time="
            f"{iso_z(rainfall_50pct_time) if rainfall_50pct_time is not None else None}; "
            f"evaluation_start={iso_z(evaluation_start)}; "
            f"evaluation_end_exclusive={iso_z(evaluation_end)}"
        ),
    )

    qc(
        "BLOCKING",
        "CUMULATIVE_RASTER_COVERAGE",
        (
            eval_raster_coverage_pct >= 99.0
            and full_raster_coverage_pct >= 99.0
        ),
        (
            f"evaluation_coverage={eval_raster_coverage_pct:.6f}%; "
            f"full_coverage={full_raster_coverage_pct:.6f}%"
        ),
    )

    qc(
        "QUALITY",
        "EVALUATION_RAINFALL_PLAUSIBILITY",
        (
            evaluation_total
            >= args.min_evaluation_rainfall_mm
            and evaluation_total
            <= args.max_evaluation_rainfall_mm
        ),
        (
            f"basin_evaluation_total={evaluation_total:.4f} mm; "
            f"guardrail=[{args.min_evaluation_rainfall_mm:.1f}, "
            f"{args.max_evaluation_rainfall_mm:.1f}] mm"
        ),
    )

    qc(
        "QUALITY",
        "BASIN_HOURLY_RAINFALL_PLAUSIBILITY",
        max_basin_hourly
        <= args.max_basin_hourly_rainfall_mm,
        (
            f"maximum_basin_hourly={max_basin_hourly:.4f} mm; "
            f"guardrail={args.max_basin_hourly_rainfall_mm:.1f} mm"
        ),
    )

    qc(
        "QUALITY",
        "USGS_HOURLY_DISCHARGE_COVERAGE",
        q_hourly_coverage_pct
        >= args.min_observed_q_hourly_coverage_percent,
        (
            f"hourly_Q_coverage={q_hourly_coverage_pct:.6f}%; "
            f"minimum="
            f"{args.min_observed_q_hourly_coverage_percent:.3f}%"
        ),
    )

    qc_rows.append(
        {
            "severity": "NOTE",
            "check": "TIMESTAMP_RESOLUTION_HANDLING",
            "status": "RECORDED",
            "detail": (
                "Rainfall temporal centroid uses pandas Timestamp.value "
                "(explicit nanoseconds since Unix epoch) to remain correct "
                "when pandas stores timezone-aware datetimes internally at "
                "microsecond rather than nanosecond resolution."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "NOTE",
            "check": "RAINFALL_TO_DISCHARGE_TIMING",
            "status": "RECORDED",
            "detail": (
                f"peak_Q={peak_q:.3f} m3/s at {iso_z(peak_q_time)}; "
                f"rainfall_centroid="
                f"{iso_z(rainfall_centroid) if rainfall_centroid is not None else None}; "
                f"rainfall_50pct_time="
                f"{iso_z(rainfall_50pct_time) if rainfall_50pct_time is not None else None}. "
                "These are timing diagnostics, not a calibrated rainfall-runoff model."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "BASIN_RAINFALL_AGGREGATION",
            "status": "SELECTED",
            "detail": (
                "Arithmetic mean across equal-area 1-km projected "
                "watershed cells. Step 5B rasterized basin area differs "
                "from authoritative vector area by only ~0.13%."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_CONSTRAINT",
            "check": "MRMS_QPE_NOT_GROUND_TRUTH",
            "status": "OPEN",
            "detail": (
                "GaugeCorr_QPE_01H is gauge-corrected radar QPE and "
                "retains precipitation-estimation uncertainty. "
                "Rainfall uncertainty will be propagated later rather "
                "than treating MRMS as error-free truth."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_CONSTRAINT",
            "check": "DISCHARGE_NOT_INDEPENDENT_RAINFALL_VALIDATION",
            "status": "OPEN",
            "detail": (
                "USGS continuous discharge is used here only for event "
                "timing context. Rainfall-to-discharge model validation "
                "occurs in later hydrology steps."
            ),
        }
    )

    qc_df = pd.DataFrame(qc_rows)

    blocking_fail = qc_df[
        (qc_df["severity"] == "BLOCKING")
        & (qc_df["status"] == "FAIL")
    ]
    quality_fail = qc_df[
        (qc_df["severity"] == "QUALITY")
        & (qc_df["status"] == "FAIL")
    ]

    if len(blocking_fail):
        status = "FAIL_MRMS_FLORENCE_RAINFALL_MODULE"
    elif len(quality_fail):
        status = (
            "FAIL_MRMS_FLORENCE_RAINFALL_MODULE_QUALITY"
        )
    else:
        status = "PASS_MRMS_FLORENCE_RAINFALL_MODULE_COMPLETE"

    atomic_csv(qc_df, qc_path)

    # ------------------------------------------------------------------
    # Plots.
    # ------------------------------------------------------------------
    plot_hourly = hourly.set_index("time_utc")
    plot_q = q_hourly

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        plot_hourly.index,
        plot_hourly["basin_mean_rainfall_mm"],
        width=0.035,
        label="MRMS basin rainfall",
    )
    ax.set_ylabel("Basin rainfall (mm/h)")
    ax.set_xlabel("UTC")
    ax.set_title(
        "Hurricane Florence — Basin Rainfall and Observed Discharge"
    )
    ax.invert_yaxis()

    ax2 = ax.twinx()
    ax2.plot(
        plot_q.index,
        plot_q.values,
        label="USGS discharge",
    )
    ax2.set_ylabel("Observed discharge (m³/s)")
    fig.tight_layout()
    fig.savefig(hydrograph_path, dpi=180)
    plt.close(fig)

    display = np.where(
        eval_valid,
        cumulative_eval,
        np.nan,
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        np.ma.masked_invalid(display)
    )
    ax.set_title(
        "Florence Evaluation-Period Cumulative MRMS Rainfall"
    )
    ax.set_xlabel("1-km grid column")
    ax.set_ylabel("1-km grid row")
    fig.colorbar(
        im,
        ax=ax,
        label="Cumulative rainfall (mm)",
    )
    fig.tight_layout()
    fig.savefig(quicklook_path, dpi=180)
    plt.close(fig)

    metadata = {
        "status": status,
        "step": "STEP_5C",
        "script_build": SCRIPT_BUILD,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "hourly_spatial_qc": str(args.hourly_spatial_qc),
            "standardization_metadata": str(
                args.standardization_metadata
            ),
            "watershed_mask": str(args.watershed_mask),
            "usgs_observation_directory": str(
                args.usgs_observation_dir
            ),
        },
        "rainfall_domain": {
            "rasterized_basin_area_km2": rasterized_basin_area_km2,
            "mask_cell_count": mask_cells,
            "crs": str(mask_crs),
            "pixel_area_m2": pixel_area_m2,
        },
        "time_windows": {
            "forcing_start_utc": iso_z(forcing_start),
            "forcing_end_utc_exclusive": iso_z(
                forcing_end_exclusive
            ),
            "evaluation_start_utc": iso_z(
                evaluation_start
            ),
            "evaluation_end_utc_exclusive": iso_z(
                evaluation_end
            ),
            "forcing_hours": int(len(hourly)),
            "evaluation_hours": int(len(eval_hourly)),
        },
        "basin_rainfall": {
            "full_forcing_total_mm": full_total,
            "antecedent_before_evaluation_mm": antecedent_total,
            "evaluation_total_mm": evaluation_total,
            "post_evaluation_total_mm": post_total,
            "maximum_hourly_basin_mean_mm": max_basin_hourly,
            "maximum_hourly_basin_mean_time_utc": iso_z(
                peak_hour_row["time_utc"]
            ),
            "rolling_maxima": roll_records,
            "evaluation_temporal_centroid_utc": (
                iso_z(rainfall_centroid)
                if rainfall_centroid is not None
                else None
            ),
            "evaluation_50pct_mass_time_utc": (
                iso_z(rainfall_50pct_time)
                if rainfall_50pct_time is not None
                else None
            ),
        },
        "observed_discharge_context": {
            **usgs_info,
            "evaluation_hourly_coverage_percent": (
                q_hourly_coverage_pct
            ),
            "peak_discharge_m3s": peak_q,
            "peak_discharge_time_utc": iso_z(
                peak_q_time
            ),
        },
        "spatial_cumulative_rainfall": {
            "evaluation_valid_coverage_percent": (
                eval_raster_coverage_pct
            ),
            "full_forcing_valid_coverage_percent": (
                full_raster_coverage_pct
            ),
            "summary_records": (
                spatial_summary.to_dict(
                    orient="records"
                )
            ),
        },
        "scientific_constraints": {
            "mrms_is_gauge_corrected_radar_qpe_not_truth": True,
            "rainfall_gap_filling_applied": False,
            "basin_mean_uses_equal_area_projected_cells": True,
            "rainfall_discharge_timing_is_calibration": False,
        },
        "blocking_failure_count": int(
            len(blocking_fail)
        ),
        "quality_failure_count": int(
            len(quality_fail)
        ),
        "step_5_complete": bool(
            len(blocking_fail) == 0
            and len(quality_fail) == 0
        ),
        "safe_for_step_6": bool(
            len(blocking_fail) == 0
            and len(quality_fail) == 0
        ),
        "output_paths": {
            "hourly_rainfall": str(hourly_path),
            "daily_rainfall": str(daily_path),
            "event_summary": str(event_summary_path),
            "rainfall_discharge_timing": str(
                timing_path
            ),
            "spatial_summary": str(
                spatial_summary_path
            ),
            "evaluation_cumulative_raster": str(
                eval_raster_path
            ),
            "full_cumulative_raster": str(
                full_raster_path
            ),
            "qc": str(qc_path),
            "metadata": str(metadata_path),
            "hydrograph_plot": str(
                hydrograph_path
            ),
            "cumulative_quicklook": str(
                quicklook_path
            ),
        },
    }
    atomic_json(metadata, metadata_path)

    print(f"Full forcing basin rainfall        : {full_total:.3f} mm")
    print(
        f"Antecedent rainfall (< eval start) : "
        f"{antecedent_total:.3f} mm"
    )
    print(
        f"Evaluation-period basin rainfall   : "
        f"{evaluation_total:.3f} mm"
    )
    print(
        f"Post-evaluation rainfall           : "
        f"{post_total:.3f} mm"
    )
    print()
    print(
        f"Peak hourly basin rainfall         : "
        f"{max_basin_hourly:.3f} mm/h"
    )
    print(
        f"Peak hourly rainfall time          : "
        f"{iso_z(peak_hour_row['time_utc'])}"
    )
    print(
        f"Peak 6-h basin rainfall            : "
        f"{roll_records[6]['amount_mm']:.3f} mm"
    )
    print(
        f"Peak 24-h basin rainfall           : "
        f"{roll_records[24]['amount_mm']:.3f} mm"
    )
    print(
        f"Rainfall temporal centroid         : "
        f"{iso_z(rainfall_centroid) if rainfall_centroid is not None else None}"
    )
    print(
        f"Rainfall 50% mass time             : "
        f"{iso_z(rainfall_50pct_time) if rainfall_50pct_time is not None else None}"
    )
    print()
    print(
        f"Observed peak discharge            : "
        f"{peak_q:.3f} m3/s"
    )
    print(
        f"Observed peak discharge time       : "
        f"{iso_z(peak_q_time)}"
    )
    print(
        f"Observed Q hourly coverage         : "
        f"{q_hourly_coverage_pct:.6f} %"
    )
    print()
    print(
        f"Evaluation cumulative coverage     : "
        f"{eval_raster_coverage_pct:.6f} %"
    )
    print(
        f"Full cumulative coverage           : "
        f"{full_raster_coverage_pct:.6f} %"
    )
    print()
    print(f"Hourly rainfall                    : {hourly_path}")
    print(f"Daily rainfall                     : {daily_path}")
    print(f"Event summary                      : {event_summary_path}")
    print(f"Rainfall/Q timing                  : {timing_path}")
    print(f"Spatial summary                    : {spatial_summary_path}")
    print(f"QC                                 : {qc_path}")
    print(f"Metadata                           : {metadata_path}")
    print()
    print(
        f"Blocking failures                  : "
        f"{len(blocking_fail)}"
    )
    print(
        f"Quality failures                   : "
        f"{len(quality_fail)}"
    )
    print(
        f"Step 5 complete                    : "
        f"{'YES' if metadata['step_5_complete'] else 'NO'}"
    )
    print(
        f"Safe for Step 6                    : "
        f"{'YES' if metadata['safe_for_step_6'] else 'NO'}"
    )
    print(f"Status                             : {status}")

    if len(blocking_fail) or len(quality_fail):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
