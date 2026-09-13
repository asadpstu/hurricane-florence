"""
Final QC for the multiyear MRMS model-development rainfall archive.

This finalizer distinguishes:
1) FULL source signature:
   includes source dataset band count; useful for provenance.
2) ANALYSIS signature:
   includes the geometry and band-1 metadata actually used to extract rainfall.
   An extra unused GRIB band does not make the extracted predictor inconsistent
   if band 1 has the same geometry/product/unit metadata.

No missing-hour interpolation is performed. Step 8B must discard any ML
feature/target row whose required rainfall-history window crosses a missing
or low-coverage hour.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd


SCRIPT_BUILD = "HISTORICAL_MRMS_ARCHIVE_QC"

FULL_SIGNATURE_COLS = [
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

# `count` intentionally excluded. Rainfall extraction reads band 1 only.
ANALYSIS_SIGNATURE_COLS = [
    "driver",
    "crs",
    "width",
    "height",
    "dtype",
    "nodata",
    "transform",
    "GRIB_ELEMENT",
    "GRIB_SHORT_NAME",
    "GRIB_UNIT",
]

ALLOWED_SOURCE_PRODUCTS = {
    "GaugeCorr_QPE_01H",
    "RadarOnly_QPE_01H",
    "DRY_HOUR_INFERRED_ZERO",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hourly-rainfall", type=Path, required=True)
    p.add_argument("--source-signatures", type=Path, required=True)
    p.add_argument("--missing-hours", type=Path, required=True)
    p.add_argument("--fallback-hours", type=Path, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--min-completeness-percent", type=float, default=99.0)
    p.add_argument("--min-basin-coverage-percent", type=float, default=99.0)
    p.add_argument("--max-fallback-hours", type=int, default=0)
    p.add_argument("--max-documented-dry-hours", type=int, default=0)
    p.add_argument("--max-consecutive-missing-hours", type=int, default=24)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def parse_utc(text: str) -> pd.Timestamp:
    t = pd.Timestamp(text)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    if t.minute or t.second or t.microsecond or t.nanosecond:
        raise ValueError("Start/end must be aligned to complete UTC hours.")
    return t


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def signature_count(df: pd.DataFrame, cols: list[str]) -> int:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError("Missing signature columns: " + ", ".join(missing))
    x = df[cols].copy().fillna("<NA>")
    for c in cols:
        x[c] = x[c].map(str)
    return int(x.drop_duplicates().shape[0])


def longest_consecutive_gap(times: pd.Series):
    if len(times) == 0:
        return 0, None, None

    ts = (
        pd.to_datetime(times, utc=True)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    best_len = cur_len = 1
    best_start = cur_start = ts.iloc[0]
    best_end = prev = ts.iloc[0]

    for current in ts.iloc[1:]:
        if current - prev == pd.Timedelta(hours=1):
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = prev
            cur_len = 1
            cur_start = current
        prev = current

    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start
        best_end = prev

    return int(best_len), best_start, best_end


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = parse_utc(args.start)
    end = parse_utc(args.end)
    expected_hours = int((end - start) / pd.Timedelta(hours=1))

    rain = pd.read_csv(args.hourly_rainfall)
    sig = pd.read_csv(args.source_signatures)
    missing = pd.read_csv(args.missing_hours)
    fallback = pd.read_csv(args.fallback_hours)

    if "interval_end_utc" not in rain.columns:
        raise RuntimeError("Rainfall table lacks interval_end_utc.")

    rain["interval_end_utc"] = pd.to_datetime(
        rain["interval_end_utc"], utc=True, errors="raise"
    )
    selected = rain[
        (rain["interval_end_utc"] >= start)
        & (rain["interval_end_utc"] < end)
    ].copy()

    if "status" not in selected.columns:
        raise RuntimeError("Rainfall table lacks status.")

    available = selected[selected["status"] == "AVAILABLE"].copy()
    unavailable = selected[selected["status"] != "AVAILABLE"].copy()

    available_hours = len(available)
    completeness = (
        available_hours / expected_hours * 100.0
        if expected_hours > 0
        else 0.0
    )

    min_coverage = (
        float(available["basin_valid_coverage_percent"].min())
        if available_hours
        else 0.0
    )

    fallback_hours = len(fallback)

    full_signature_count = signature_count(sig, FULL_SIGNATURE_COLS)
    analysis_signature_count = signature_count(sig, ANALYSIS_SIGNATURE_COLS)

    sig_requested = sig.copy()
    if "interval_end_utc" in sig_requested.columns:
        sig_requested["interval_end_utc"] = pd.to_datetime(
            sig_requested["interval_end_utc"], utc=True, errors="coerce"
        )
        sig_requested = sig_requested[
            (sig_requested["interval_end_utc"] >= start)
            & (sig_requested["interval_end_utc"] < end)
        ].copy()

    signature_products = set(
        sig_requested.get("source_product", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    rainfall_products = set(
        available.get("source_product", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    products = signature_products | rainfall_products
    unexpected_products = sorted(products - ALLOWED_SOURCE_PRODUCTS)
    documented_dry_hours = int(
        available.get("source_product", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .eq("DRY_HOUR_INFERRED_ZERO")
        .sum()
    )

    units = sorted(
        sig_requested.get("GRIB_UNIT", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    unit_count = len(units)

    band_counts = (
        pd.to_numeric(sig["count"], errors="coerce")
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    if not missing.empty:
        if "interval_end_utc" not in missing.columns:
            raise RuntimeError("Missing-hour table lacks interval_end_utc.")
        missing["interval_end_utc"] = pd.to_datetime(
            missing["interval_end_utc"], utc=True, errors="raise"
        )

    gap_len, gap_start, gap_end = longest_consecutive_gap(
        missing["interval_end_utc"] if not missing.empty
        else pd.Series([], dtype="datetime64[ns, UTC]")
    )

    qc_rows = []

    def qc(severity, check, passed, detail):
        qc_rows.append({
            "severity": severity,
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    qc(
        "BLOCKING",
        "EXPECTED_TIME_AXIS",
        len(selected) == expected_hours,
        (
            f"rows_in_requested_period={len(selected)}; "
            f"expected_hours={expected_hours}"
        ),
    )

    qc(
        "BLOCKING",
        "TEMPORAL_COMPLETENESS",
        completeness >= args.min_completeness_percent,
        (
            f"available={available_hours}; expected={expected_hours}; "
            f"completeness={completeness:.6f}%; "
            f"minimum={args.min_completeness_percent:.3f}%"
        ),
    )

    qc(
        "BLOCKING",
        "AVAILABLE_HOUR_BASIN_COVERAGE",
        min_coverage >= args.min_basin_coverage_percent,
        (
            f"minimum_available_coverage={min_coverage:.6f}%; "
            f"minimum={args.min_basin_coverage_percent:.3f}%"
        ),
    )

    qc(
        "QUALITY",
        "NO_RADARONLY_FALLBACK",
        fallback_hours <= args.max_fallback_hours,
        (
            f"fallback_hours={fallback_hours}; "
            f"maximum={args.max_fallback_hours}"
        ),
    )

    qc(
        "QUALITY",
        "ALLOWED_MRMS_SOURCE_PRODUCTS",
        len(unexpected_products) == 0,
        (
            f"source_products={sorted(products)}; "
            f"unexpected_products={unexpected_products}; "
            "GaugeCorr_QPE_01H is primary and RadarOnly_QPE_01H is an "
            "explicitly permitted fallback subject to the fallback-hour cap."
        ),
    )


    qc(
        "QUALITY",
        "DOCUMENTED_DRY_HOUR_REPAIR_LIMIT",
        documented_dry_hours <= args.max_documented_dry_hours,
        (
            f"documented_dry_hours={documented_dry_hours}; "
            f"maximum={args.max_documented_dry_hours}. "
            "DRY_HOUR_INFERRED_ZERO is permitted only as an explicitly "
            "documented, evidence-checked repair; it is not interpolation."
        ),
    )

    qc(
        "QUALITY",
        "RAINFALL_UNIT_CONSISTENCY",
        unit_count <= 1,
        (
            f"band1_units={units}; unique_unit_count={unit_count}. "
            "Native grids/product identifiers may vary, but extracted rainfall "
            "must retain one consistent band-1 unit convention."
        ),
    )

    qc(
        "QUALITY",
        "MISSING_GAP_LENGTH",
        gap_len <= args.max_consecutive_missing_hours,
        (
            f"longest_missing_gap_hours={gap_len}; "
            f"maximum={args.max_consecutive_missing_hours}; "
            f"gap_start={gap_start}; gap_end={gap_end}"
        ),
    )

    qc_rows.append({
        "severity": "WARNING",
        "check": "NATIVE_SOURCE_SIGNATURE_VARIATION",
        "status": "WARNING" if analysis_signature_count > 1 else "PASS",
        "detail": (
            f"full_source_signatures={full_signature_count}; "
            f"analysis_source_signatures={analysis_signature_count}; "
            f"band_count_distribution={band_counts}; "
            f"source_products={sorted(products)}. Native MRMS source geometry "
            "and product metadata can vary over a multiyear archive and when "
            "RadarOnly fallback is explicitly used. The extraction workflow "
            "warps every hour to the fixed watershed-mask target grid before "
            "aggregation; native signature variation is therefore retained as "
            "provenance warning, while unit consistency and allowed-product "
            "checks remain quality gates."
        ),
    })

    qc_rows.append({
        "severity": "WARNING",
        "check": "MISSING_HOURS_RETAINED",
        "status": "WARNING" if len(missing) else "PASS",
        "detail": (
            f"missing_or_low_coverage_hours={len(missing)}; "
            f"longest_gap_hours={gap_len}. No rainfall interpolation is "
            "permitted. Step 8B must invalidate every feature row whose "
            "required rainfall-history window intersects one of these hours."
        ),
    })

    qc_rows.append({
        "severity": "SCIENTIFIC_PARAMETER",
        "check": "MRMS_PRODUCT",
        "status": "SELECTED",
        "detail": (
            "GaugeCorr_QPE_01H; timestamp T represents accumulation "
            "interval (T-1h, T]."
        ),
    })

    qc_rows.append({
        "severity": "SCIENTIFIC_PARAMETER",
        "check": "ML_SPLIT_POLICY",
        "status": "SELECTED",
        "detail": (
            "2015-05-10 through 2016-12-31 training; calendar year 2017 "
            "temporal validation; Hurricane Florence 2018 untouched final test."
        ),
    })

    qc_df = pd.DataFrame(qc_rows)

    blocking_fail = qc_df[
        (qc_df["severity"] == "BLOCKING")
        & (qc_df["status"] == "FAIL")
    ]
    quality_fail = qc_df[
        (qc_df["severity"] == "QUALITY")
        & (qc_df["status"] == "FAIL")
    ]
    warnings = qc_df[
        (qc_df["severity"] == "WARNING")
        & (qc_df["status"] == "WARNING")
    ]

    safe = len(blocking_fail) == 0 and len(quality_fail) == 0

    if safe and len(warnings):
        status = "PASS_HISTORICAL_MRMS_READY_WITH_WARNINGS"
    elif safe:
        status = "PASS_HISTORICAL_MRMS_READY"
    elif len(blocking_fail):
        status = "FAIL_HISTORICAL_MRMS_BLOCKING"
    else:
        status = "FAIL_HISTORICAL_MRMS_QUALITY"

    qc_path = args.output_dir / "multiyear_rainfall_final_qc.csv"
    meta_path = args.output_dir / "multiyear_rainfall_final_metadata.json"

    atomic_csv(qc_df, qc_path)

    metadata = {
        "status": status,
        "workflow": "historical_mrms_archive_qc",
        "script_build": SCRIPT_BUILD,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": {
            "start_utc_inclusive": start.isoformat(),
            "end_utc_exclusive": end.isoformat(),
            "expected_hours": expected_hours,
        },
        "archive": {
            "available_hours": available_hours,
            "missing_or_low_coverage_hours": int(len(missing)),
            "completeness_percent": completeness,
            "minimum_available_basin_coverage_percent": min_coverage,
            "fallback_hours": fallback_hours,
            "longest_consecutive_missing_hours": gap_len,
            "longest_gap_start": (
                gap_start.isoformat() if gap_start is not None else None
            ),
            "longest_gap_end": (
                gap_end.isoformat() if gap_end is not None else None
            ),
        },
        "source_consistency": {
            "full_source_signature_count": full_signature_count,
            "analysis_signature_count": analysis_signature_count,
            "band_count_distribution": {
                str(k): int(v) for k, v in band_counts.items()
            },
            "analysis_signature_excludes_total_band_count": True,
            "rainfall_band_used": 1,
            "source_products": sorted(products),
            "unexpected_source_products": unexpected_products,
            "band1_units": units,
            "native_signature_variation_is_provenance_warning": True,
        },
        "ml_constraints": {
            "no_temporal_rainfall_interpolation": True,
            "invalidate_features_crossing_missing_rainfall": True,
            "training_period": "2015-05-10 through 2016-12-31",
            "validation_period": "2017-01-01 through 2017-12-31",
            "final_test": "Hurricane Florence 2018",
        },
        "blocking_failure_count": int(len(blocking_fail)),
        "quality_failure_count": int(len(quality_fail)),
        "warning_count": int(len(warnings)),
        "safe_for_historical_discharge": bool(safe),
        "outputs": {
            "final_qc": str(qc_path),
            "final_metadata": str(meta_path),
        },
    }

    atomic_json(metadata, meta_path)

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {SCRIPT_BUILD}")
    print("FINAL MULTIYEAR MRMS ARCHIVE QC")
    print("=" * 100)
    print(f"Expected hours                     : {expected_hours:,}")
    print(f"Available hours                    : {available_hours:,}")
    print(f"Missing/low-coverage hours         : {len(missing):,}")
    print(f"Completeness                       : {completeness:.6f} %")
    print(f"Minimum available basin coverage   : {min_coverage:.6f} %")
    print(f"Fallback hours                     : {fallback_hours:,}")
    print(f"Longest missing gap                : {gap_len} h")
    print(f"Full source signatures             : {full_signature_count}")
    print(f"Analysis source signatures         : {analysis_signature_count}")
    print(f"Source products                    : {sorted(products)}")
    print(f"Band-1 rainfall units              : {units}")
    print(f"Source band counts                 : {band_counts}")
    print()
    print(f"Blocking failures                  : {len(blocking_fail)}")
    print(f"Quality failures                   : {len(quality_fail)}")
    print(f"Warnings                           : {len(warnings)}")
    print(f"Safe for Step 8B                   : {'YES' if safe else 'NO'}")
    print(f"Final QC                           : {qc_path}")
    print(f"Final metadata                     : {meta_path}")
    print(f"Status                             : {status}")

    if not safe:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
