"""
Parallel, resume-safe multiyear MRMS basin rainfall archive.

Scientific behavior is unchanged from Step 8A v2:
- Primary source: GaugeCorr_QPE_01H
- Recommended period: 2015-05-10 through 2017-12-31
- Same Step 5B 1-km USGS 02089000 watershed mask
- Same MRMS time convention: timestamp T is interval end for (T-1h, T]
- No temporal interpolation
- RadarOnly fallback disabled unless explicitly requested

Execution improvements
----------------------
- Parallel hour processing with ProcessPoolExecutor
- Each worker loads the watershed mask once via initializer
- Main process alone writes checkpoint files
- Existing v2/v3 checkpoints are reused automatically
- No --overwrite is needed when resuming
- Temporary GRIB files are deleted after processing by default

Recommended workers
-------------------
Use 6 initially on a MacBook Pro. Increasing to 8 may help if network and
archive throttling remain stable, but 6 is the conservative default.

Outputs
-------
output/rainfall/mrms_ml_2015_2017/
  multiyear_basin_hourly_rainfall.csv
  missing_hours.csv
  fallback_hours.csv
  source_grid_signatures.csv
  multiyear_rainfall_qc.csv
  multiyear_rainfall_metadata.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import requests


SCRIPT_BUILD = "HISTORICAL_MRMS_BASIN_RAINFALL"

IEM_BASE = "https://mtarchive.geol.iastate.edu"
PRIMARY_PRODUCT = "GaugeCorr_QPE_01H"
FALLBACK_PRODUCT = "RadarOnly_QPE_01H"

# Per-process globals loaded once by initializer.
_G_MASK = None
_G_CRS = None
_G_TRANSFORM = None
_G_WIDTH = None
_G_HEIGHT = None
_G_BASIN_CELLS = None
_G_MIN_COVERAGE = None
_G_TIMEOUT = None
_G_RETRIES = None
_G_ALLOW_FALLBACK = None
_G_KEEP_GRIB = None
_G_CACHE_DIR = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="UTC end-exclusive.")
    p.add_argument(
        "--watershed-mask",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=6,
    )
    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
    )
    p.add_argument(
        "--retries",
        type=int,
        default=3,
    )
    p.add_argument(
        "--allow-radaronly-fallback",
        action="store_true",
    )
    p.add_argument(
        "--min-completeness-percent",
        type=float,
        default=99.0,
    )
    p.add_argument(
        "--max-fallback-percent",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--min-basin-valid-coverage-percent",
        type=float,
        default=99.0,
    )
    p.add_argument(
        "--checkpoint-hours",
        type=int,
        default=24,
    )
    p.add_argument(
        "--keep-compressed-grib",
        action="store_true",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--max-hours",
        type=int,
        default=None,
    )
    p.add_argument(
        "--retry-nonavailable",
        action="store_true",
        help=(
            "When resuming from an existing archive, retry selected rows whose "
            "status is not AVAILABLE instead of treating them as completed."
        ),
    )
    p.add_argument(
        "--retry-times-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV/text file limiting --retry-nonavailable to exact UTC "
            "hours. CSV columns interval_end_utc or missing_time_utc are accepted. "
            "Without this file, all non-AVAILABLE rows in the requested window "
            "are retried."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
    )
    return p.parse_args()


def parse_utc(text: str) -> datetime:
    t = text.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    if dt.minute or dt.second or dt.microsecond:
        raise ValueError("Start/end must be aligned to whole UTC hours.")
    return dt


def iso_z(dt: datetime | pd.Timestamp) -> str:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def hourly_range(start: datetime, end: datetime) -> list[datetime]:
    if end <= start:
        raise ValueError("--end must be later than --start.")
    out = []
    t = start
    while t < end:
        out.append(t)
        t += timedelta(hours=1)
    return out


def product_filename(product: str, dt: datetime) -> str:
    return (
        f"{product}_00.00_"
        f"{dt:%Y%m%d}-{dt:%H}0000.grib2.gz"
    )


def product_url(product: str, dt: datetime) -> str:
    return (
        f"{IEM_BASE}/{dt:%Y}/{dt:%m}/{dt:%d}/"
        f"mrms/ncep/{product}/"
        f"{product_filename(product, dt)}"
    )


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    tmp.unlink(missing_ok=True)
    tmp.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def init_worker(
    watershed_mask_path: str,
    min_coverage: float,
    timeout: float,
    retries: int,
    allow_fallback: bool,
    keep_grib: bool,
    cache_dir: str | None,
) -> None:
    global _G_MASK, _G_CRS, _G_TRANSFORM, _G_WIDTH, _G_HEIGHT
    global _G_BASIN_CELLS, _G_MIN_COVERAGE, _G_TIMEOUT, _G_RETRIES
    global _G_ALLOW_FALLBACK, _G_KEEP_GRIB, _G_CACHE_DIR

    with rasterio.open(watershed_mask_path) as ds:
        _G_MASK = ds.read(1) > 0
        _G_CRS = ds.crs
        _G_TRANSFORM = ds.transform
        _G_WIDTH = ds.width
        _G_HEIGHT = ds.height

    if _G_CRS is None:
        raise RuntimeError("Watershed mask has no CRS.")

    _G_BASIN_CELLS = int(_G_MASK.sum())
    if _G_BASIN_CELLS <= 0:
        raise RuntimeError("Watershed mask contains no valid cells.")

    _G_MIN_COVERAGE = float(min_coverage)
    _G_TIMEOUT = float(timeout)
    _G_RETRIES = int(retries)
    _G_ALLOW_FALLBACK = bool(allow_fallback)
    _G_KEEP_GRIB = bool(keep_grib)
    _G_CACHE_DIR = cache_dir


def open_grib_dataset(path: Path):
    ds = rasterio.open(path)
    if ds.count >= 1 and ds.width > 0 and ds.height > 0:
        return ds, None

    subs = list(ds.subdatasets)
    ds.close()

    if not subs:
        raise RuntimeError(f"No readable raster bands in {path}")

    sub = rasterio.open(subs[0])
    return sub, subs[0]


def download_file(
    url: str,
    output: Path,
) -> tuple[bool, int | None, str | None]:
    last_status = None
    last_error = None
    output.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, _G_RETRIES + 1):
        partial = output.with_suffix(output.suffix + ".partial")
        partial.unlink(missing_ok=True)

        try:
            with requests.get(
                url,
                stream=True,
                timeout=_G_TIMEOUT,
                headers={
                    "User-Agent": (
                        "neuse-flood-research/1.0 "
                        "(parallel multiyear MRMS basin archive)"
                    )
                },
            ) as r:
                last_status = r.status_code

                if r.status_code == 404:
                    return False, last_status, "HTTP 404"

                r.raise_for_status()

                with partial.open("wb") as f:
                    for chunk in r.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            f.write(chunk)

            if not partial.exists() or partial.stat().st_size <= 0:
                raise RuntimeError("Downloaded file is empty.")

            os.replace(partial, output)
            return True, last_status, None

        except Exception as exc:
            partial.unlink(missing_ok=True)
            last_error = repr(exc)
            if attempt < _G_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))

    return False, last_status, last_error


def core_signature(ds, subdataset: str | None) -> dict[str, Any]:
    tags = ds.tags(1)
    return {
        "driver": ds.driver,
        "crs": str(ds.crs),
        "width": int(ds.width),
        "height": int(ds.height),
        "count": int(ds.count),
        "dtype": str(ds.dtypes[0]),
        "nodata": ds.nodata,
        "transform": repr(ds.transform),
        "subdataset": subdataset,
        "GRIB_ELEMENT": tags.get("GRIB_ELEMENT"),
        "GRIB_SHORT_NAME": tags.get("GRIB_SHORT_NAME"),
        "GRIB_UNIT": tags.get("GRIB_UNIT"),
    }


def missing_row(
    dt: datetime,
    error: str,
) -> dict[str, Any]:
    return {
        "interval_start_utc": iso_z(dt - timedelta(hours=1)),
        "interval_end_utc": iso_z(dt),
        "status": "MISSING",
        "source_product": None,
        "used_fallback": False,
        "basin_valid_coverage_percent": 0.0,
        "basin_mean_rainfall_mm": np.nan,
        "basin_min_gridcell_rainfall_mm": np.nan,
        "basin_p95_gridcell_rainfall_mm": np.nan,
        "basin_max_gridcell_rainfall_mm": np.nan,
        "compressed_bytes": np.nan,
        "error": error,
    }


def process_hour(dt_iso: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Worker function. Processes exactly one hour.
    """
    dt = parse_utc(dt_iso)

    products = [PRIMARY_PRODUCT]
    if _G_ALLOW_FALLBACK:
        products.append(FALLBACK_PRODUCT)

    last_error = None

    for product in products:
        filename = product_filename(product, dt)

        if _G_CACHE_DIR:
            gz_path = (
                Path(_G_CACHE_DIR)
                / f"{dt:%Y}"
                / f"{dt:%m}"
                / filename
            )
            gz_path.parent.mkdir(parents=True, exist_ok=True)
            persistent = True
        else:
            worker_tmp = Path(tempfile.mkdtemp(prefix="neuse_mrms_hour_"))
            gz_path = worker_tmp / filename
            persistent = False

        temp_grib = None

        try:
            if not gz_path.exists():
                ok, http_status, error = download_file(
                    product_url(product, dt),
                    gz_path,
                )
                if not ok:
                    last_error = (
                        f"{product}: {error}; HTTP={http_status}"
                    )
                    if not persistent:
                        shutil.rmtree(
                            gz_path.parent,
                            ignore_errors=True,
                        )
                    continue

            temp_grib = gz_path.with_suffix("")
            with gzip.open(gz_path, "rb") as src, temp_grib.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)

            ds = None
            try:
                ds, subdataset = open_grib_dataset(temp_grib)

                if ds.crs is None:
                    raise RuntimeError("MRMS GRIB has no CRS.")

                signature = core_signature(ds, subdataset)
                signature["interval_end_utc"] = iso_z(dt)
                signature["source_product"] = product

                vrt_kwargs = {
                    "crs": _G_CRS,
                    "transform": _G_TRANSFORM,
                    "width": _G_WIDTH,
                    "height": _G_HEIGHT,
                    "resampling": Resampling.bilinear,
                    "nodata": -9999.0,
                }
                if ds.nodata is not None and np.isfinite(ds.nodata):
                    vrt_kwargs["src_nodata"] = ds.nodata

                with WarpedVRT(ds, **vrt_kwargs) as vrt:
                    arr = vrt.read(1, masked=True)

                data = np.asarray(arr.data, dtype="float64")
                valid = (
                    _G_MASK
                    & ~np.ma.getmaskarray(arr)
                    & np.isfinite(data)
                    & (data >= 0.0)
                )

                coverage = (
                    valid.sum() / _G_BASIN_CELLS * 100.0
                )
                vals = data[valid]

                if vals.size:
                    mean_mm = float(np.mean(vals))
                    min_mm = float(np.min(vals))
                    p95_mm = float(np.percentile(vals, 95))
                    max_mm = float(np.max(vals))
                else:
                    mean_mm = min_mm = p95_mm = max_mm = np.nan

                status = (
                    "AVAILABLE"
                    if coverage >= _G_MIN_COVERAGE
                    else "LOW_COVERAGE"
                )

                row = {
                    "interval_start_utc": iso_z(
                        dt - timedelta(hours=1)
                    ),
                    "interval_end_utc": iso_z(dt),
                    "status": status,
                    "source_product": product,
                    "used_fallback": product == FALLBACK_PRODUCT,
                    "basin_valid_coverage_percent": coverage,
                    "basin_mean_rainfall_mm": mean_mm,
                    "basin_min_gridcell_rainfall_mm": min_mm,
                    "basin_p95_gridcell_rainfall_mm": p95_mm,
                    "basin_max_gridcell_rainfall_mm": max_mm,
                    "compressed_bytes": gz_path.stat().st_size,
                    "error": (
                        None
                        if status == "AVAILABLE"
                        else (
                            f"Basin coverage {coverage:.6f}% below "
                            f"required {_G_MIN_COVERAGE:.3f}%"
                        )
                    ),
                }

                return row, signature

            finally:
                if ds is not None:
                    ds.close()

        except Exception as exc:
            last_error = f"{product}: {repr(exc)}"

        finally:
            if temp_grib is not None:
                temp_grib.unlink(missing_ok=True)

            if persistent:
                if not _G_KEEP_GRIB:
                    # A persistent cache is preserved regardless of keep flag.
                    # User supplied --cache-dir explicitly.
                    pass
            else:
                if not _G_KEEP_GRIB:
                    gz_path.unlink(missing_ok=True)
                shutil.rmtree(
                    gz_path.parent,
                    ignore_errors=True,
                )

    return missing_row(
        dt,
        last_error or "No source product available",
    ), None


def checkpoint(
    rows: list[dict[str, Any]],
    signatures: list[dict[str, Any]],
    output_csv: Path,
    signatures_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    df = (
        df.sort_values("interval_end_utc")
        .drop_duplicates("interval_end_utc", keep="last")
        .reset_index(drop=True)
    )
    atomic_csv(df, output_csv)

    if signatures:
        sig = pd.DataFrame(signatures)
        sig = (
            sig.sort_values("interval_end_utc")
            .drop_duplicates("interval_end_utc", keep="last")
            .reset_index(drop=True)
        )
        atomic_csv(sig, signatures_csv)
    else:
        sig = pd.DataFrame()

    return df, sig


def load_retry_times(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = None
        for c in ["interval_end_utc", "missing_time_utc", "time_utc", "timestamp_utc"]:
            if c in df.columns:
                col = c
                break
        if col is None:
            raise RuntimeError(
                f"Retry CSV needs interval_end_utc or missing_time_utc. Columns={list(df.columns)}"
            )
        vals = df[col]
    else:
        vals = pd.Series([
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ])

    out = set()
    for value in vals:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if ts.minute or ts.second or ts.microsecond:
            raise ValueError(f"Retry timestamp is not aligned to an hour: {value}")
        out.add(iso_z(ts))
    return out


def main() -> None:
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")
    if args.checkpoint_hours < 1:
        raise ValueError("--checkpoint-hours must be >= 1.")

    start = parse_utc(args.start)
    end = parse_utc(args.end)
    hours = hourly_range(start, end)

    if args.max_hours is not None:
        hours = hours[: args.max_hours]

    requested_times = [iso_z(dt) for dt in hours]
    requested_set_for_retry = set(requested_times)
    retry_times_from_file = load_retry_times(args.retry_times_file)
    if retry_times_from_file is not None:
        outside = sorted(retry_times_from_file - requested_set_for_retry)
        if outside:
            raise RuntimeError(
                "Retry timestamps fall outside --start/--end requested window: "
                + ", ".join(outside[:10])
            )
        retry_target_times = retry_times_from_file
    else:
        retry_target_times = requested_set_for_retry

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = args.output_dir / "multiyear_basin_hourly_rainfall.csv"
    missing_csv = args.output_dir / "missing_hours.csv"
    fallback_csv = args.output_dir / "fallback_hours.csv"
    signatures_csv = args.output_dir / "source_grid_signatures.csv"
    qc_csv = args.output_dir / "multiyear_rainfall_qc.csv"
    metadata_json = args.output_dir / "multiyear_rainfall_metadata.json"

    if args.overwrite:
        for p in [
            output_csv,
            missing_csv,
            fallback_csv,
            signatures_csv,
            qc_csv,
            metadata_json,
        ]:
            p.unlink(missing_ok=True)

    retried_nonavailable_at_start = 0
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        if "interval_end_utc" not in existing.columns:
            raise RuntimeError(
                f"Incompatible checkpoint: {output_csv}"
            )

        if args.retry_nonavailable:
            if "status" not in existing.columns:
                raise RuntimeError(
                    "--retry-nonavailable requires an existing status column."
                )
            retry_mask = (
                existing["status"].astype(str).ne("AVAILABLE")
                & existing["interval_end_utc"].astype(str).isin(retry_target_times)
            )
            retried_nonavailable_at_start = int(retry_mask.sum())
            retained = existing.loc[~retry_mask].copy()
            completed_times = set(
                retained["interval_end_utc"].astype(str)
            )
            # Drop old failed rows before appending repaired results. This
            # avoids duplicate timestamps and makes replacement deterministic.
            rows = retained.to_dict(orient="records")
        else:
            completed_times = set(
                existing["interval_end_utc"].astype(str)
            )
            rows = existing.to_dict(orient="records")
    else:
        completed_times = set()
        rows = []

    if signatures_csv.exists() and not args.overwrite:
        old_sig = pd.read_csv(signatures_csv)
        signatures = old_sig.to_dict(orient="records")
    else:
        signatures = []

    pending = [
        t for t in requested_times
        if t not in completed_times
    ]

    with rasterio.open(args.watershed_mask) as ds:
        mask_cells = int((ds.read(1) > 0).sum())
        target_width = ds.width
        target_height = ds.height
        target_crs = ds.crs

    with rasterio.Env() as env:
        if "GRIB" not in env.drivers():
            raise RuntimeError(
                "GDAL/Rasterio GRIB driver is unavailable."
            )

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {SCRIPT_BUILD}")
    print("MULTIYEAR PRE-FLORENCE MRMS BASIN RAINFALL")
    print("=" * 100)
    print(f"Start UTC (inclusive)              : {iso_z(start)}")
    print(f"End UTC (exclusive)                : {iso_z(end)}")
    print(f"Requested hours                    : {len(hours):,}")
    print(f"Already checkpointed               : {len(completed_times):,}")
    print(f"Retry non-AVAILABLE rows            : {'YES' if args.retry_nonavailable else 'NO'}")
    if args.retry_nonavailable:
        print(f"Non-AVAILABLE rows queued to retry  : {retried_nonavailable_at_start:,}")
        print(f"Retry-times file                    : {args.retry_times_file}")
    print(f"Pending hours                      : {len(pending):,}")
    print(f"Parallel workers                   : {args.workers}")
    print(f"Watershed mask                     : {args.watershed_mask}")
    print(f"Target grid                        : {target_width} x {target_height}")
    print(f"Target CRS                         : {target_crs}")
    print(f"Basin mask cells                   : {mask_cells:,}")
    print()

    processed = 0

    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(
                str(args.watershed_mask),
                args.min_basin_valid_coverage_percent,
                args.timeout_seconds,
                args.retries,
                args.allow_radaronly_fallback,
                args.keep_compressed_grib,
                (
                    str(args.cache_dir)
                    if args.cache_dir is not None
                    else None
                ),
            ),
        ) as pool:
            futures = {
                pool.submit(process_hour, t): t
                for t in pending
            }

            for future in as_completed(futures):
                t = futures[future]
                try:
                    row, signature = future.result()
                except Exception as exc:
                    row = missing_row(
                        parse_utc(t),
                        f"Worker exception: {repr(exc)}",
                    )
                    signature = None

                rows.append(row)
                if signature is not None:
                    signatures.append(signature)

                completed_times.add(t)
                processed += 1

                if (
                    processed == 1
                    or processed % args.checkpoint_hours == 0
                    or processed == len(pending)
                ):
                    current, _ = checkpoint(
                        rows,
                        signatures,
                        output_csv,
                        signatures_csv,
                    )
                    print(
                        f"Checkpoint                         : "
                        f"{len(current):,} total hours "
                        f"({processed:,}/{len(pending):,} pending processed)"
                    )

    final, sig_df = checkpoint(
        rows,
        signatures,
        output_csv,
        signatures_csv,
    )

    requested_set = set(requested_times)
    requested = final[
        final["interval_end_utc"].astype(str).isin(requested_set)
    ].copy()

    available = requested[
        requested["status"] == "AVAILABLE"
    ].copy()
    missing = requested[
        requested["status"] != "AVAILABLE"
    ].copy()
    fallback = available[
        available["used_fallback"] == True  # noqa: E712
    ].copy()

    expected = len(hours)
    available_n = len(available)
    missing_n = len(missing)
    fallback_n = len(fallback)

    completeness_pct = (
        available_n / expected * 100.0
        if expected
        else 0.0
    )
    fallback_pct = (
        fallback_n / expected * 100.0
        if expected
        else 0.0
    )
    min_coverage = (
        float(
            available["basin_valid_coverage_percent"].min()
        )
        if available_n
        else 0.0
    )

    unique_signatures = None
    if not sig_df.empty:
        req_sig = sig_df[
            sig_df["interval_end_utc"].astype(str).isin(requested_set)
        ].copy()
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
        if not req_sig.empty:
            unique_signatures = int(
                req_sig[core_cols]
                .astype(str)
                .drop_duplicates()
                .shape[0]
            )

    atomic_csv(missing, missing_csv)
    atomic_csv(fallback, fallback_csv)

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
        "TEMPORAL_COMPLETENESS",
        completeness_pct >= args.min_completeness_percent,
        (
            f"available={available_n}; expected={expected}; "
            f"completeness={completeness_pct:.6f}%; "
            f"minimum={args.min_completeness_percent:.3f}%"
        ),
    )

    qc(
        "BLOCKING",
        "BASIN_SPATIAL_COVERAGE",
        min_coverage >= args.min_basin_valid_coverage_percent,
        (
            f"minimum_available_hour_coverage={min_coverage:.6f}%; "
            f"minimum={args.min_basin_valid_coverage_percent:.3f}%"
        ),
    )

    qc(
        "QUALITY",
        "RADARONLY_FALLBACK_FRACTION",
        fallback_pct <= args.max_fallback_percent,
        (
            f"fallback_hours={fallback_n}; "
            f"fallback_percent={fallback_pct:.6f}%; "
            f"maximum={args.max_fallback_percent:.3f}%"
        ),
    )

    qc_rows.append(
        {
            "severity": "WARNING",
            "check": "NATIVE_SOURCE_GRID_SIGNATURE_VARIATION",
            "status": (
                "WARNING"
                if unique_signatures not in (None, 1)
                else "PASS"
            ),
            "detail": (
                f"unique_native_source_signatures={unique_signatures}. "
                "Native MRMS geometry/product metadata may vary across a "
                "multiyear archive and when an explicitly allowed RadarOnly "
                "fallback is used. Every available hour is warped onto the "
                "same watershed-mask target grid before basin aggregation, "
                "so native signature variation is retained as provenance "
                "rather than treated as a model-quality failure."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "ML_PRIMARY_RAINFALL_SOURCE",
            "status": "SELECTED",
            "detail": (
                "GaugeCorr_QPE_01H is used consistently. RadarOnly "
                "fallback remains disabled unless explicitly requested."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "PARALLEL_EXECUTION",
            "status": "SELECTED",
            "detail": (
                f"ProcessPoolExecutor with {args.workers} workers. "
                "Parallelization changes execution only, not rainfall "
                "source, reprojection, watershed mask, or aggregation."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_CONSTRAINT",
            "check": "NO_TEMPORAL_GAP_FILLING",
            "status": "OPEN",
            "detail": (
                "Missing or low-coverage hours are retained as explicit "
                "failures; no temporal interpolation is performed."
            ),
        }
    )

    qc_df = pd.DataFrame(qc_rows)
    atomic_csv(qc_df, qc_csv)

    blocking_fail = qc_df[
        (qc_df["severity"] == "BLOCKING")
        & (qc_df["status"] == "FAIL")
    ]
    quality_fail = qc_df[
        (qc_df["severity"] == "QUALITY")
        & (qc_df["status"] == "FAIL")
    ]

    if len(blocking_fail):
        status = "FAIL_MULTIYEAR_MRMS_ML_RAINFALL"
    elif len(quality_fail):
        status = "FAIL_MULTIYEAR_MRMS_ML_RAINFALL_QUALITY"
    else:
        status = "PASS_MULTIYEAR_MRMS_ML_RAINFALL_READY"

    metadata = {
        "status": status,
        "workflow": "historical_mrms_basin_rainfall",
        "script_build": SCRIPT_BUILD,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "archive": IEM_BASE,
            "primary_product": PRIMARY_PRODUCT,
            "fallback_product": (
                FALLBACK_PRODUCT
                if args.allow_radaronly_fallback
                else None
            ),
        },
        "period": {
            "start_utc_inclusive": iso_z(start),
            "end_utc_exclusive": iso_z(end),
            "requested_hours": expected,
            "max_hours_smoke_test": args.max_hours,
        },
        "execution": {
            "workers": args.workers,
            "checkpoint_hours": args.checkpoint_hours,
            "already_checkpointed_at_start": (
                len(completed_times) - processed
            ),
            "processed_this_run": processed,
            "resume_supported": True,
            "retry_nonavailable": bool(args.retry_nonavailable),
            "retry_times_file": str(args.retry_times_file) if args.retry_times_file else None,
            "retry_target_count": int(len(retry_target_times)) if args.retry_nonavailable else 0,
            "nonavailable_rows_retried_at_start": int(retried_nonavailable_at_start),
        },
        "inventory": {
            "available_hours": available_n,
            "missing_or_low_coverage_hours": missing_n,
            "fallback_hours": fallback_n,
            "completeness_percent": completeness_pct,
            "fallback_percent": fallback_pct,
            "minimum_available_basin_coverage_percent": min_coverage,
            "unique_core_source_signatures": unique_signatures,
        },
        "recommended_ml_split": {
            "training": "2015-05-10 through 2016-12-31",
            "validation": "2017-01-01 through 2017-12-31",
            "final_test": "Hurricane Florence 2018",
        },
        "blocking_failure_count": int(len(blocking_fail)),
        "quality_failure_count": int(len(quality_fail)),
        "safe_for_historical_discharge": bool(
            len(blocking_fail) == 0
            and len(quality_fail) == 0
            and args.max_hours is None
        ),
        "outputs": {
            "hourly_basin_rainfall": str(output_csv),
            "missing_hours": str(missing_csv),
            "fallback_hours": str(fallback_csv),
            "source_grid_signatures": str(signatures_csv),
            "qc": str(qc_csv),
            "metadata": str(metadata_json),
        },
    }
    atomic_json(metadata, metadata_json)

    print()
    print(f"Requested hours                    : {expected:,}")
    print(f"Available hours                    : {available_n:,}")
    print(f"Missing/low-coverage hours         : {missing_n:,}")
    print(f"Fallback hours                     : {fallback_n:,}")
    print(f"Completeness                       : {completeness_pct:.6f} %")
    print(f"Minimum basin coverage             : {min_coverage:.6f} %")
    print(f"Unique source grid signatures      : {unique_signatures}")
    print()
    print(f"Hourly rainfall                    : {output_csv}")
    print(f"Missing-hour report                : {missing_csv}")
    print(f"Fallback-hour report               : {fallback_csv}")
    print(f"QC                                 : {qc_csv}")
    print(f"Metadata                           : {metadata_json}")
    print()
    print(f"Blocking failures                  : {len(blocking_fail)}")
    print(f"Quality failures                   : {len(quality_fail)}")
    print(
        f"Safe for Step 8B                   : "
        f"{'YES' if metadata['safe_for_historical_discharge'] else 'NO'}"
    )
    print(f"Status                             : {status}")

    if len(blocking_fail) or len(quality_fail):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
