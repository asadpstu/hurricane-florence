"""
STEP 5A - Acquire MRMS hourly precipitation for Hurricane Florence (2018).

Primary source
--------------
NOAA/NCEP MRMS GaugeCorr_QPE_01H archived by Iowa State University IEM:
  https://mtarchive.geol.iastate.edu/YYYY/MM/DD/mrms/ncep/
      GaugeCorr_QPE_01H/
      GaugeCorr_QPE_01H_00.00_YYYYMMDD-HH0000.grib2.gz

Fallback
--------
MRMS RadarOnly_QPE_01H at the same archive location.

Scientific interpretation
-------------------------
- GaugeCorr_QPE_01H is hourly accumulated precipitation (mm), radar-based and
  gauge-bias corrected.
- RadarOnly_QPE_01H is allowed only as an explicitly flagged continuity
  fallback for missing GaugeCorr hours.
- This script does NOT blend, interpolate, or silently fill missing hours.
- The acquisition forcing window can extend beyond the formal event-evaluation
  period to provide antecedent/spin-up and recession rainfall.

Default study window used by the command:
  2018-09-01T00:00:00Z <= valid time < 2018-09-27T00:00:00Z
  = 624 expected hourly files.

Outputs
-------
input/rainfall/mrms/florence_2018/raw/
  *.grib2.gz

output/rainfall/mrms_florence_2018_acquisition/
  mrms_hourly_inventory.csv
  mrms_acquisition_qc.csv
  mrms_acquisition_metadata.json
  missing_hours.csv
  fallback_hours.csv
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests


SCRIPT_BUILD = "STEP_5A_MRMS_FLORENCE_ACQUISITION_V1"

IEM_BASE = "https://mtarchive.geol.iastate.edu"

PRIMARY_PRODUCT = "GaugeCorr_QPE_01H"
FALLBACK_PRODUCT = "RadarOnly_QPE_01H"


@dataclass
class HourResult:
    valid_time_utc: str
    expected_index: int
    status: str
    source_product: str | None
    used_fallback: bool
    url: str | None
    local_path: str | None
    http_status: int | None
    compressed_bytes: int | None
    sha256: str | None
    gzip_valid: bool | None
    attempts: int
    error: str | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--start",
        required=True,
        help="UTC inclusive, e.g. 2018-09-01T00:00:00Z",
    )
    p.add_argument(
        "--end",
        required=True,
        help="UTC exclusive, e.g. 2018-09-27T00:00:00Z",
    )
    p.add_argument(
        "--download-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--threads",
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
        default=5.0,
    )
    p.add_argument(
        "--skip-gzip-validation",
        action="store_true",
    )
    p.add_argument("--overwrite", action="store_true")
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
    if dt.minute != 0 or dt.second != 0 or dt.microsecond != 0:
        raise ValueError(
            "Start/end must be aligned to whole UTC hours."
        )
    return dt


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def validate_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            for chunk in iter(
                lambda: f.read(1024 * 1024),
                b"",
            ):
                pass
        return True
    except Exception:
        return False


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.stem}.partial{path.suffix}"
    )
    tmp.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.stem}.partial{path.suffix}"
    )
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def download_one(
    dt: datetime,
    index: int,
    download_dir: Path,
    timeout: float,
    retries: int,
    allow_fallback: bool,
    validate_gz: bool,
    overwrite: bool,
) -> HourResult:
    products = [PRIMARY_PRODUCT]
    if allow_fallback:
        products.append(FALLBACK_PRODUCT)

    last_error = None
    last_http = None
    total_attempts = 0

    for product in products:
        url = product_url(product, dt)
        filename = product_filename(product, dt)
        local = download_dir / filename

        # Reuse an existing validated file when overwrite is false.
        if local.exists() and not overwrite:
            gz_ok = (
                validate_gzip(local)
                if validate_gz
                else True
            )
            if gz_ok:
                return HourResult(
                    valid_time_utc=iso_z(dt),
                    expected_index=index,
                    status="AVAILABLE",
                    source_product=product,
                    used_fallback=(
                        product == FALLBACK_PRODUCT
                    ),
                    url=url,
                    local_path=str(local),
                    http_status=None,
                    compressed_bytes=local.stat().st_size,
                    sha256=sha256_file(local),
                    gzip_valid=gz_ok,
                    attempts=0,
                    error=None,
                )
            local.unlink(missing_ok=True)

        for attempt in range(1, retries + 1):
            total_attempts += 1
            partial = local.with_suffix(
                local.suffix + ".partial"
            )
            partial.unlink(missing_ok=True)

            try:
                with requests.get(
                    url,
                    stream=True,
                    timeout=timeout,
                    headers={
                        "User-Agent": (
                            "neuse-flood-research/1.0 "
                            "(MRMS historical QPE acquisition)"
                        )
                    },
                ) as r:
                    last_http = r.status_code

                    if r.status_code == 404:
                        last_error = "HTTP 404"
                        break

                    r.raise_for_status()

                    with partial.open("wb") as f:
                        for chunk in r.iter_content(
                            chunk_size=1024 * 1024
                        ):
                            if chunk:
                                f.write(chunk)

                if partial.stat().st_size <= 0:
                    raise RuntimeError(
                        "Downloaded file is empty."
                    )

                os.replace(partial, local)

                gz_ok = (
                    validate_gzip(local)
                    if validate_gz
                    else True
                )
                if not gz_ok:
                    local.unlink(missing_ok=True)
                    raise RuntimeError(
                        "GZIP integrity validation failed."
                    )

                return HourResult(
                    valid_time_utc=iso_z(dt),
                    expected_index=index,
                    status="AVAILABLE",
                    source_product=product,
                    used_fallback=(
                        product == FALLBACK_PRODUCT
                    ),
                    url=url,
                    local_path=str(local),
                    http_status=last_http,
                    compressed_bytes=local.stat().st_size,
                    sha256=sha256_file(local),
                    gzip_valid=gz_ok,
                    attempts=total_attempts,
                    error=None,
                )

            except requests.HTTPError as exc:
                partial.unlink(missing_ok=True)
                last_error = str(exc)

                # Missing product: try fallback immediately.
                if (
                    exc.response is not None
                    and exc.response.status_code == 404
                ):
                    break

                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

            except Exception as exc:
                partial.unlink(missing_ok=True)
                last_error = repr(exc)
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

    return HourResult(
        valid_time_utc=iso_z(dt),
        expected_index=index,
        status="MISSING",
        source_product=None,
        used_fallback=False,
        url=None,
        local_path=None,
        http_status=last_http,
        compressed_bytes=None,
        sha256=None,
        gzip_valid=None,
        attempts=total_attempts,
        error=last_error,
    )


def main() -> None:
    args = parse_args()

    if args.threads < 1:
        raise ValueError("--threads must be >= 1.")
    if args.retries < 1:
        raise ValueError("--retries must be >= 1.")

    start = parse_utc(args.start)
    end = parse_utc(args.end)
    hours = hourly_range(start, end)

    args.download_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory_path = (
        args.output_dir / "mrms_hourly_inventory.csv"
    )
    qc_path = (
        args.output_dir / "mrms_acquisition_qc.csv"
    )
    metadata_path = (
        args.output_dir
        / "mrms_acquisition_metadata.json"
    )
    missing_path = (
        args.output_dir / "missing_hours.csv"
    )
    fallback_path = (
        args.output_dir / "fallback_hours.csv"
    )

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {SCRIPT_BUILD}")
    print("STEP 5A - MRMS FLORENCE HOURLY QPE ACQUISITION")
    print("=" * 100)
    print(f"Primary product                    : {PRIMARY_PRODUCT}")
    print(
        f"Radar-only fallback                : "
        f"{'ENABLED' if args.allow_radaronly_fallback else 'DISABLED'}"
    )
    print(f"Start UTC (inclusive)              : {iso_z(start)}")
    print(f"End UTC (exclusive)                : {iso_z(end)}")
    print(f"Expected hourly files              : {len(hours):,}")
    print(f"Download directory                 : {args.download_dir}")
    print(f"Threads                            : {args.threads}")
    print()

    results: list[HourResult] = []

    with ThreadPoolExecutor(
        max_workers=args.threads
    ) as pool:
        futures = {
            pool.submit(
                download_one,
                dt,
                i,
                args.download_dir,
                args.timeout_seconds,
                args.retries,
                args.allow_radaronly_fallback,
                not args.skip_gzip_validation,
                args.overwrite,
            ): (i, dt)
            for i, dt in enumerate(hours)
        }

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            if (
                completed == 1
                or completed % 25 == 0
                or completed == len(hours)
            ):
                print(
                    f"Completed                          : "
                    f"{completed:,}/{len(hours):,}"
                )

    results.sort(key=lambda x: x.expected_index)
    inventory = pd.DataFrame(
        [asdict(r) for r in results]
    )

    available = inventory[
        inventory["status"] == "AVAILABLE"
    ].copy()
    missing = inventory[
        inventory["status"] == "MISSING"
    ].copy()
    fallback = inventory[
        inventory["used_fallback"] == True  # noqa: E712
    ].copy()

    expected_count = len(inventory)
    available_count = len(available)
    missing_count = len(missing)
    fallback_count = len(fallback)

    completeness_pct = (
        available_count / expected_count * 100.0
        if expected_count
        else 0.0
    )
    fallback_pct = (
        fallback_count / expected_count * 100.0
        if expected_count
        else 0.0
    )

    primary_count = int(
        (
            available["source_product"]
            == PRIMARY_PRODUCT
        ).sum()
    )

    total_bytes = int(
        available["compressed_bytes"]
        .fillna(0)
        .sum()
    )

    # Temporal continuity check.
    available_times = set(
        pd.to_datetime(
            available["valid_time_utc"],
            utc=True,
        )
    )
    expected_times = pd.date_range(
        start=start,
        end=end - timedelta(hours=1),
        freq="h",
        tz="UTC",
    )
    temporal_missing = [
        t for t in expected_times
        if t not in available_times
    ]

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
                "status": (
                    "PASS" if passed else "FAIL"
                ),
                "detail": detail,
            }
        )

    qc(
        "BLOCKING",
        "HOURLY_COMPLETENESS",
        completeness_pct
        >= args.min_completeness_percent,
        (
            f"available={available_count}; "
            f"expected={expected_count}; "
            f"completeness={completeness_pct:.6f}%; "
            f"minimum={args.min_completeness_percent:.3f}%"
        ),
    )

    qc(
        "BLOCKING",
        "TEMPORAL_INVENTORY_CONSISTENCY",
        len(temporal_missing) == missing_count,
        (
            f"inventory_missing={missing_count}; "
            f"timeline_missing={len(temporal_missing)}"
        ),
    )

    qc(
        "QUALITY",
        "RADARONLY_FALLBACK_FRACTION",
        fallback_pct <= args.max_fallback_percent,
        (
            f"fallback_hours={fallback_count}; "
            f"fallback_percent={fallback_pct:.6f}%; "
            f"maximum={args.max_fallback_percent:.3f}%"
        ),
    )

    qc(
        "BLOCKING",
        "GZIP_INTEGRITY",
        bool(
            available["gzip_valid"]
            .fillna(False)
            .all()
        ),
        (
            f"validated_files={available_count}; "
            f"gzip_failures="
            f"{int((available['gzip_valid'] == False).sum())}"
        ),
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "PRIMARY_PRECIPITATION_PRODUCT",
            "status": "SELECTED",
            "detail": (
                "NOAA/NCEP MRMS GaugeCorr_QPE_01H via "
                "Iowa State IEM historical archive; hourly "
                "accumulated precipitation in mm."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "FORCING_WINDOW",
            "status": "SELECTED",
            "detail": (
                f"{iso_z(start)} to {iso_z(end)} end-exclusive. "
                "This extends beyond the formal Florence evaluation "
                "window to provide antecedent/spin-up and recession "
                "precipitation."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_CONSTRAINT",
            "check": "FALLBACK_NOT_BLEND",
            "status": "OPEN",
            "detail": (
                "Any RadarOnly_QPE_01H hours are explicitly flagged "
                "and must be carried as source-quality information. "
                "No temporal interpolation or silent blending occurs."
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
        status = "FAIL_MRMS_FLORENCE_ACQUISITION"
    elif len(quality_fail):
        status = (
            "FAIL_MRMS_FLORENCE_ACQUISITION_QUALITY"
        )
    elif fallback_count:
        status = (
            "PASS_MRMS_FLORENCE_ACQUIRED_WITH_FALLBACK"
        )
    else:
        status = "PASS_MRMS_FLORENCE_ACQUIRED"

    atomic_csv(inventory, inventory_path)
    atomic_csv(qc_df, qc_path)
    atomic_csv(missing, missing_path)
    atomic_csv(fallback, fallback_path)

    metadata = {
        "status": status,
        "step": "STEP_5A",
        "script_build": SCRIPT_BUILD,
        "created_utc": iso_z(
            datetime.now(timezone.utc)
        ),
        "source": {
            "provider": (
                "NOAA/NCEP MRMS via Iowa State University IEM archive"
            ),
            "archive_base": IEM_BASE,
            "primary_product": PRIMARY_PRODUCT,
            "fallback_product": (
                FALLBACK_PRODUCT
                if args.allow_radaronly_fallback
                else None
            ),
            "primary_interpretation": (
                "Local gauge bias-corrected radar precipitation "
                "accumulation, 1-hour, unit mm."
            ),
            "archive_note": (
                "IEM historical MRMS archive contains selected "
                "products back to October 2014."
            ),
        },
        "window": {
            "start_utc_inclusive": iso_z(start),
            "end_utc_exclusive": iso_z(end),
            "expected_hours": expected_count,
        },
        "inventory": {
            "available_hours": available_count,
            "primary_hours": primary_count,
            "fallback_hours": fallback_count,
            "missing_hours": missing_count,
            "completeness_percent": completeness_pct,
            "fallback_percent": fallback_pct,
            "compressed_total_bytes": total_bytes,
            "compressed_total_gib": (
                total_bytes / 1024**3
            ),
        },
        "qc_thresholds": {
            "minimum_completeness_percent": (
                args.min_completeness_percent
            ),
            "maximum_fallback_percent": (
                args.max_fallback_percent
            ),
        },
        "blocking_failure_count": int(
            len(blocking_fail)
        ),
        "quality_failure_count": int(
            len(quality_fail)
        ),
        "safe_for_step_5b": bool(
            len(blocking_fail) == 0
            and len(quality_fail) == 0
        ),
        "output_paths": {
            "raw_download_dir": str(
                args.download_dir
            ),
            "hourly_inventory": str(
                inventory_path
            ),
            "qc": str(qc_path),
            "metadata": str(metadata_path),
            "missing_hours": str(missing_path),
            "fallback_hours": str(
                fallback_path
            ),
        },
    }
    atomic_json(metadata, metadata_path)

    print()
    print(
        f"Available hourly files             : "
        f"{available_count:,}/{expected_count:,}"
    )
    print(
        f"Completeness                       : "
        f"{completeness_pct:.6f} %"
    )
    print(
        f"Gauge-corrected hours              : "
        f"{primary_count:,}"
    )
    print(
        f"Radar-only fallback hours          : "
        f"{fallback_count:,} "
        f"({fallback_pct:.6f} %)"
    )
    print(
        f"Missing hours                      : "
        f"{missing_count:,}"
    )
    print(
        f"Compressed archive size            : "
        f"{total_bytes / 1024**3:.3f} GiB"
    )
    print()
    print(f"Inventory                          : {inventory_path}")
    print(f"Missing hours                      : {missing_path}")
    print(f"Fallback hours                     : {fallback_path}")
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
        f"Safe for Step 5B                   : "
        f"{'YES' if metadata['safe_for_step_5b'] else 'NO'}"
    )
    print(f"Status                             : {status}")

    if len(blocking_fail) or len(quality_fail):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
