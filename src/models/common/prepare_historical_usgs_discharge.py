"""
Build multiyear USGS discharge and rainfall-discharge development dataset.

Study
-----
USGS 02089000 - Neuse River near Goldsboro, NC
MRMS training archive - Step 8A
Target variable - hourly mean USGS discharge, m3/s

Scientific design
-----------------
* Training:   2015-05-10 through 2016-12-31
* Validation: 2017-01-01 through 2017-12-31
* Florence 2018 is NOT included and remains the final test event.
* Predictors are rainfall-derived + seasonal timing only.
* Observed discharge is NEVER used as an input feature.
* No future rainfall is used.
* No temporal interpolation of missing MRMS rainfall is allowed.
* A row is model-ready only if its complete 168-h rainfall feature history
  is available and the target-hour USGS discharge passes coverage QC.
* USGS instantaneous Q is aggregated to the same interval convention used
  by MRMS: hourly interval ending T represents (T-1h, T].

USGS source
-----------
USGS Water Data API continuous values:
https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items
parameter 00060 = discharge, ft3/s.

Outputs
-------
USGS:
  input/observations/usgs/02089000/ml_2015_2017/
    usgs_02089000_q_00060_raw.csv
    usgs_02089000_q_hourly.csv
    usgs_download_metadata.json

ML dataset:
  output/ml/rainfall_discharge_2015_2017/
    rainfall_discharge_hourly_full.csv
    rainfall_discharge_train.csv
    rainfall_discharge_validation.csv
    feature_manifest.csv
    dataset_qc.csv
    dataset_metadata.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


SCRIPT_BUILD = "HISTORICAL_USGS_DISCHARGE_DATASET"
USGS_IV_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"
CFS_TO_M3S = 0.028316846592


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="02089000")
    p.add_argument("--rainfall", type=Path, required=True)
    p.add_argument("--rainfall-final-metadata", type=Path, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--training-end", required=True)
    p.add_argument("--end", required=True, help="End-exclusive UTC.")
    p.add_argument("--usgs-output-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--usgs-chunk-days", type=int, default=30)
    p.add_argument("--timeout-seconds", type=float, default=90.0)
    p.add_argument("--retries", type=int, default=6)
    p.add_argument(
        "--retry-base-seconds",
        type=float,
        default=5.0,
        help="Base delay for exponential retry backoff on transient USGS errors.",
    )
    p.add_argument(
        "--retry-max-seconds",
        type=float,
        default=120.0,
        help="Maximum retry sleep for transient USGS errors.",
    )
    p.add_argument(
        "--min-q-hourly-coverage-percent",
        type=float,
        default=75.0,
        help="Minimum within-hour expected USGS sample coverage.",
    )
    p.add_argument(
        "--min-period-q-availability-percent",
        type=float,
        default=95.0,
        help="Minimum valid hourly-Q availability separately in train/validation.",
    )
    p.add_argument("--min-train-rows", type=int, default=5000)
    p.add_argument("--min-validation-rows", type=int, default=5000)
    p.add_argument(
        "--reuse-usgs-if-present",
        action="store_true",
        help="Reuse existing raw USGS CSV instead of downloading again.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_utc(text: str) -> pd.Timestamp:
    t = pd.Timestamp(text)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    if t.minute or t.second or t.microsecond or t.nanosecond:
        raise ValueError("All split timestamps must align to UTC hours.")
    return t


def iso_z(value: Any) -> str:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    tmp.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def request_json(
    session: requests.Session,
    params: dict[str, str],
    timeout: float,
    retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> dict[str, Any]:
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                USGS_IV_URL,
                params=params,
                timeout=timeout,
                headers={"Accept-Encoding": "identity"},
            )

            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = retry_base_seconds * (2 ** (attempt - 1))
                else:
                    delay = retry_base_seconds * (2 ** (attempt - 1))

                delay = min(delay, retry_max_seconds)

                if attempt < retries:
                    print(
                        f"USGS transient HTTP {response.status_code}; "
                        f"retry {attempt}/{retries} after {delay:.1f} s"
                    )
                    time.sleep(delay)
                    continue

            response.raise_for_status()
            return response.json()

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                delay = min(
                    retry_base_seconds * (2 ** (attempt - 1)),
                    retry_max_seconds,
                )
                print(
                    f"USGS request error                 : "
                    f"{type(exc).__name__}: {exc}; "
                    f"retry {attempt}/{retries} after {delay:.1f} s"
                )
                time.sleep(delay)

    raise RuntimeError(
        f"USGS request failed after {retries} attempts: {last_error!r}"
    )


def extract_q_series(payload: dict[str, Any], site: str) -> pd.DataFrame:
    features = payload.get("features", [])
    rows = []

    for feature in features:
        props = feature.get("properties", {})

        if props.get("monitoring_location_id") != f"USGS-{site}":
            continue
        if str(props.get("parameter_code")) != "00060":
            continue

        try:
            value = float(props.get("value"))
        except (TypeError, ValueError):
            continue

        dt = pd.to_datetime(
            props.get("time"),
            utc=True,
            errors="coerce",
        )
        if pd.isna(dt):
            continue

        rows.append(
            {
                "timestamp_utc": dt,
                "q_cfs": value,
                "qualifiers": (
                    ""
                    if props.get("qualifier") is None
                    else str(props.get("qualifier"))
                ),
                "unit": props.get("unit_of_measure"),
                "approval_status": props.get("approval_status"),
                "time_series_id": props.get("time_series_id"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "q_cfs",
                "qualifiers",
                "unit",
                "approval_status",
                "time_series_id",
            ]
        )

    out = pd.DataFrame(rows)
    out = (
        out.dropna(subset=["timestamp_utc", "q_cfs"])
        .sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc", keep="last")
        .reset_index(drop=True)
    )
    return out


def build_year_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Build requests no longer than one calendar year.
    Includes one preceding hour for first model interval.
    """
    request_start = start - pd.Timedelta(hours=1)
    chunks = []
    current = request_start

    while current < end:
        next_year = pd.Timestamp(
            year=current.year + 1,
            month=1,
            day=1,
            tz="UTC",
        )
        chunk_end = min(next_year, end)
        chunks.append((current, chunk_end))
        current = chunk_end

    return chunks


def download_usgs_q(
    site: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
    timeout: float,
    retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    checkpoint_raw_path: Path,
    checkpoint_manifest_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    # chunk_days retained in function signature for backward CLI compatibility.
    del chunk_days

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "neuse-flood-research/1.0 "
                "(USGS modern continuous API)"
            ),
            "Accept-Encoding": "identity",
        }
    )

    # V3 intentionally uses a clean modern-API checkpoint.
    modern_raw_path = checkpoint_raw_path.with_name(
        checkpoint_raw_path.stem + "_modern_api.csv"
    )
    modern_manifest_path = checkpoint_manifest_path.with_name(
        checkpoint_manifest_path.stem + "_modern_api.csv"
    )

    if modern_raw_path.exists():
        existing_raw = pd.read_csv(modern_raw_path)
        if not existing_raw.empty:
            existing_raw["timestamp_utc"] = pd.to_datetime(
                existing_raw["timestamp_utc"],
                utc=True,
                errors="raise",
            )
        frames = [existing_raw]
    else:
        frames = []

    if modern_manifest_path.exists():
        manifest = pd.read_csv(modern_manifest_path)
        completed_chunks = set(
            manifest.loc[
                manifest["status"].eq("COMPLETE"),
                "chunk_key",
            ].astype(str)
        )
        chunk_records = manifest.to_dict(orient="records")
    else:
        completed_chunks = set()
        chunk_records = []

    chunks = build_year_chunks(start, end)

    for chunk_start, chunk_end in chunks:
        chunk_key = f"{iso_z(chunk_start)}__{iso_z(chunk_end)}"

        if chunk_key in completed_chunks:
            print(
                f"USGS annual chunk (resume skip)   : "
                f"{iso_z(chunk_start)} -> {iso_z(chunk_end)}"
            )
            continue

        params = {
            "f": "json",
            "monitoring_location_id": f"USGS-{site}",
            "parameter_code": "00060",
            "time": f"{iso_z(chunk_start)}/{iso_z(chunk_end)}",
            "skipGeometry": "true",
            "limit": "50000",
        }

        payload = request_json(
            session,
            params,
            timeout,
            retries,
            retry_base_seconds,
            retry_max_seconds,
        )

        frame = extract_q_series(payload, site)

        number_matched = payload.get("numberMatched")
        number_returned = payload.get("numberReturned")

        if number_matched is not None:
            try:
                matched = int(number_matched)
            except (TypeError, ValueError):
                matched = None
        else:
            matched = None

        if number_returned is not None:
            try:
                returned = int(number_returned)
            except (TypeError, ValueError):
                returned = len(frame)
        else:
            returned = len(frame)

        # One year of 15-minute data should be < 50,000 rows. Refuse
        # silently truncated results.
        if matched is not None and matched > returned:
            raise RuntimeError(
                f"USGS modern API pagination required but not implemented: "
                f"numberMatched={matched}, numberReturned={returned}, "
                f"chunk={chunk_key}"
            )

        print(
            f"USGS annual chunk                 : "
            f"{iso_z(chunk_start)} -> {iso_z(chunk_end)} | "
            f"{len(frame):,} observations"
        )

        frames.append(frame)

        current = pd.concat(frames, ignore_index=True)
        if not current.empty:
            current = (
                current.dropna(subset=["timestamp_utc", "q_cfs"])
                .sort_values("timestamp_utc")
                .drop_duplicates("timestamp_utc", keep="last")
                .reset_index(drop=True)
            )

        atomic_csv(current, modern_raw_path)

        chunk_records = [
            r for r in chunk_records
            if str(r.get("chunk_key")) != chunk_key
        ]
        chunk_records.append(
            {
                "chunk_key": chunk_key,
                "request_start_utc": iso_z(chunk_start),
                "request_end_utc": iso_z(chunk_end),
                "records": int(len(frame)),
                "status": "COMPLETE",
                "number_matched": matched,
                "number_returned": returned,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "api": "USGS Water Data /continuous",
            }
        )

        manifest = pd.DataFrame(chunk_records).sort_values(
            ["request_start_utc", "request_end_utc"]
        )
        atomic_csv(manifest, modern_manifest_path)

        completed_chunks.add(chunk_key)
        frames = [current]

    if not frames:
        return pd.DataFrame(), chunk_records

    raw = pd.concat(frames, ignore_index=True)
    raw = (
        raw.dropna(subset=["timestamp_utc", "q_cfs"])
        .sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc", keep="last")
        .reset_index(drop=True)
    )

    # Publish modern API output to the canonical Step 8B raw path only
    # after all annual chunks are successfully assembled.
    atomic_csv(raw, checkpoint_raw_path)
    if modern_manifest_path.exists():
        shutil_manifest = pd.read_csv(modern_manifest_path)
        atomic_csv(shutil_manifest, checkpoint_manifest_path)

    return raw, chunk_records

def estimate_sampling_interval_minutes(raw: pd.DataFrame) -> float:
    if len(raw) < 2:
        return float("nan")
    diff = (
        raw["timestamp_utc"]
        .sort_values()
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )
    diff = diff[(diff > 0) & (diff <= 120)]
    if diff.empty:
        return float("nan")
    return float(diff.median())


def aggregate_q_hourly(
    raw: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_coverage_percent: float,
) -> tuple[pd.DataFrame, float, int]:
    if raw.empty:
        raise RuntimeError("No USGS discharge observations were downloaded.")

    negative_count = int((raw["q_cfs"] < 0).sum())
    usable = raw[raw["q_cfs"] >= 0].copy()

    median_interval_min = estimate_sampling_interval_minutes(usable)
    if not np.isfinite(median_interval_min) or median_interval_min <= 0:
        raise RuntimeError(
            "Could not estimate the USGS instantaneous sampling interval."
        )

    expected_samples = max(
        1,
        int(round(60.0 / median_interval_min)),
    )

    x = usable.set_index("timestamp_utc")["q_cfs"]

    agg = x.resample(
        "1h",
        label="right",
        closed="right",
        origin="epoch",
    ).agg(["mean", "count", "min", "max"])

    agg = agg.rename(
        columns={
            "mean": "q_mean_cfs",
            "count": "q_obs_count",
            "min": "q_min_cfs",
            "max": "q_max_cfs",
        }
    )

    full_index = pd.date_range(
        start=start,
        end=end - pd.Timedelta(hours=1),
        freq="1h",
        tz="UTC",
    )
    agg = agg.reindex(full_index)
    agg.index.name = "interval_end_utc"

    agg["q_expected_obs_count"] = expected_samples
    agg["q_hourly_coverage_percent"] = (
        agg["q_obs_count"].fillna(0)
        / expected_samples
        * 100.0
    ).clip(upper=100.0)

    agg["q_valid"] = (
        agg["q_mean_cfs"].notna()
        & (
            agg["q_hourly_coverage_percent"]
            >= min_coverage_percent
        )
    )

    agg["q_mean_m3s"] = (
        agg["q_mean_cfs"] * CFS_TO_M3S
    )
    agg["q_min_m3s"] = (
        agg["q_min_cfs"] * CFS_TO_M3S
    )
    agg["q_max_m3s"] = (
        agg["q_max_cfs"] * CFS_TO_M3S
    )

    return agg.reset_index(), median_interval_min, negative_count


def build_features(
    rainfall: pd.DataFrame,
    hourly_q: pd.DataFrame,
    start: pd.Timestamp,
    training_end: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    rain = rainfall.copy()

    required_rain = [
        "interval_end_utc",
        "status",
        "basin_mean_rainfall_mm",
    ]
    missing = [c for c in required_rain if c not in rain.columns]
    if missing:
        raise RuntimeError(
            "Rainfall table missing columns: " + ", ".join(missing)
        )

    rain["interval_end_utc"] = pd.to_datetime(
        rain["interval_end_utc"],
        utc=True,
        errors="raise",
    )

    rain = (
        rain.sort_values("interval_end_utc")
        .drop_duplicates("interval_end_utc", keep="last")
    )

    full_index = pd.date_range(
        start=start,
        end=end - pd.Timedelta(hours=1),
        freq="1h",
        tz="UTC",
    )
    base = pd.DataFrame({"interval_end_utc": full_index})
    base = base.merge(
        rain[
            [
                "interval_end_utc",
                "status",
                "basin_mean_rainfall_mm",
                "basin_valid_coverage_percent",
            ]
        ],
        on="interval_end_utc",
        how="left",
    )

    base["rain_available"] = (
        base["status"].eq("AVAILABLE")
        & base["basin_mean_rainfall_mm"].notna()
        & (base["basin_mean_rainfall_mm"] >= 0)
    )
    base["rain_1h_mm"] = base[
        "basin_mean_rainfall_mm"
    ].where(base["rain_available"])

    rain_series = base["rain_1h_mm"]

    sum_windows = [3, 6, 12, 24, 48, 72, 96, 168]
    for hours in sum_windows:
        base[f"rain_sum_{hours}h_mm"] = rain_series.rolling(
            hours,
            min_periods=hours,
        ).sum()

    for lag in [1, 3, 6, 12, 24, 48, 72, 96]:
        base[f"rain_lag_{lag}h_mm"] = rain_series.shift(lag)

    base["rain_max_24h_mm"] = rain_series.rolling(
        24,
        min_periods=24,
    ).max()
    base["rain_max_72h_mm"] = rain_series.rolling(
        72,
        min_periods=72,
    ).max()

    # Seasonal timing only; no observed-Q state is provided to the model.
    doy = base["interval_end_utc"].dt.dayofyear.astype(float)
    base["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    base["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    feature_cols = [
        "rain_1h_mm",
        *[f"rain_sum_{h}h_mm" for h in sum_windows],
        *[f"rain_lag_{h}h_mm" for h in [1, 3, 6, 12, 24, 48, 72, 96]],
        "rain_max_24h_mm",
        "rain_max_72h_mm",
        "doy_sin",
        "doy_cos",
    ]

    base["rain_feature_complete"] = (
        base[feature_cols].notna().all(axis=1)
    )

    q = hourly_q.copy()
    q["interval_end_utc"] = pd.to_datetime(
        q["interval_end_utc"], utc=True, errors="raise"
    )
    base = base.merge(
        q,
        on="interval_end_utc",
        how="left",
    )

    base["split"] = np.where(
        base["interval_end_utc"] < training_end,
        "TRAIN",
        "VALIDATION",
    )

    base["model_ready"] = (
        base["rain_feature_complete"]
        & base["q_valid"].fillna(False)
        & base["q_mean_m3s"].notna()
    )

    base["invalid_reason"] = ""
    base.loc[
        ~base["rain_feature_complete"],
        "invalid_reason",
    ] = "INCOMPLETE_168H_RAIN_HISTORY"

    mask_q = (
        base["rain_feature_complete"]
        & ~base["q_valid"].fillna(False)
    )
    base.loc[mask_q, "invalid_reason"] = "INVALID_TARGET_Q"

    return base, feature_cols


def main() -> None:
    args = parse_args()

    start = parse_utc(args.start)
    training_end = parse_utc(args.training_end)
    end = parse_utc(args.end)

    if not (start < training_end < end):
        raise ValueError(
            "Require start < training-end < end."
        )

    rainfall_meta = load_json(args.rainfall_final_metadata)
    if not rainfall_meta.get("safe_for_historical_discharge", False):
        raise RuntimeError(
            "Step 8A final metadata does not authorize Step 8B."
        )

    args.usgs_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = (
        args.usgs_output_dir
        / f"usgs_{args.site}_q_00060_raw.csv"
    )
    hourly_q_path = (
        args.usgs_output_dir
        / f"usgs_{args.site}_q_hourly.csv"
    )
    usgs_meta_path = (
        args.usgs_output_dir
        / "usgs_download_metadata.json"
    )
    usgs_chunk_manifest_path = (
        args.usgs_output_dir
        / "usgs_download_chunks.csv"
    )

    full_path = args.output_dir / "rainfall_discharge_hourly_full.csv"
    train_path = args.output_dir / "rainfall_discharge_train.csv"
    validation_path = (
        args.output_dir / "rainfall_discharge_validation.csv"
    )
    feature_path = args.output_dir / "feature_manifest.csv"
    qc_path = args.output_dir / "dataset_qc.csv"
    metadata_path = args.output_dir / "dataset_metadata.json"

    output_files = [
        raw_path,
        hourly_q_path,
        usgs_meta_path,
        usgs_chunk_manifest_path,
        full_path,
        train_path,
        validation_path,
        feature_path,
        qc_path,
        metadata_path,
    ]
    if args.overwrite:
        for path in output_files:
            path.unlink(missing_ok=True)

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {SCRIPT_BUILD}")
    print("MULTIYEAR USGS DISCHARGE + MODEL DEVELOPMENT DATASET")
    print("=" * 100)
    print(f"USGS site                          : {args.site}")
    print(f"Period                             : {iso_z(start)} -> {iso_z(end)}")
    print(f"Training                           : {iso_z(start)} -> {iso_z(training_end)}")
    print(f"Validation                         : {iso_z(training_end)} -> {iso_z(end)}")
    print("Predictor policy                   : rainfall + seasonality only; NO Q lags")
    print("Maximum rainfall history           : 168 h")
    print()

    chunks = []

    if args.reuse_usgs_if_present and raw_path.exists():
        print(f"Reusing USGS raw observations      : {raw_path}")
        raw = pd.read_csv(raw_path)
        raw["timestamp_utc"] = pd.to_datetime(
            raw["timestamp_utc"],
            utc=True,
            errors="raise",
        )
    else:
        raw, chunks = download_usgs_q(
            args.site,
            start,
            end,
            args.usgs_chunk_days,
            args.timeout_seconds,
            args.retries,
            args.retry_base_seconds,
            args.retry_max_seconds,
            raw_path,
            usgs_chunk_manifest_path,
        )
        if raw.empty:
            raise RuntimeError(
                "USGS returned no discharge observations."
            )
        atomic_csv(raw, raw_path)

    hourly_q, median_interval, negative_q_count = aggregate_q_hourly(
        raw,
        start,
        end,
        args.min_q_hourly_coverage_percent,
    )
    atomic_csv(hourly_q, hourly_q_path)

    rainfall = pd.read_csv(args.rainfall)
    dataset, feature_cols = build_features(
        rainfall,
        hourly_q,
        start,
        training_end,
        end,
    )

    train_all = dataset[dataset["split"] == "TRAIN"].copy()
    val_all = dataset[
        dataset["split"] == "VALIDATION"
    ].copy()

    train = train_all[train_all["model_ready"]].copy()
    val = val_all[val_all["model_ready"]].copy()

    # Q availability independent of rainfall-window validity.
    train_q_availability = (
        train_all["q_valid"].fillna(False).mean() * 100.0
    )
    val_q_availability = (
        val_all["q_valid"].fillna(False).mean() * 100.0
    )

    train_feature_ready = (
        train_all["rain_feature_complete"].mean() * 100.0
    )
    val_feature_ready = (
        val_all["rain_feature_complete"].mean() * 100.0
    )

    atomic_csv(dataset, full_path)
    atomic_csv(train, train_path)
    atomic_csv(val, validation_path)

    manifest_rows = []
    for feature in feature_cols:
        if feature.startswith("rain_"):
            source = "MRMS GaugeCorr_QPE_01H"
        else:
            source = "UTC timestamp"

        manifest_rows.append(
            {
                "feature": feature,
                "source": source,
                "uses_observed_discharge": False,
                "uses_future_information": False,
            }
        )

    feature_manifest = pd.DataFrame(manifest_rows)
    atomic_csv(feature_manifest, feature_path)

    expected_hours = int(
        (end - start) / pd.Timedelta(hours=1)
    )

    qc_rows = []

    def qc(severity, check, passed, detail):
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
        "HISTORICAL_RAINFALL_AUTHORIZATION",
        bool(rainfall_meta.get("safe_for_historical_discharge")),
        (
            f"Historical rainfall status={rainfall_meta.get('status')}; "
            f"safe_for_historical_discharge={rainfall_meta.get('safe_for_historical_discharge')}"
        ),
    )

    qc(
        "BLOCKING",
        "EXACT_TIME_AXIS",
        len(dataset) == expected_hours,
        f"rows={len(dataset)}; expected={expected_hours}",
    )

    qc(
        "BLOCKING",
        "NO_FLORENCE_LEAKAGE",
        bool(
            (
                dataset["interval_end_utc"]
                < pd.Timestamp("2018-01-01T00:00:00Z")
            ).all()
        ),
        "Historical ML dataset ends before 2018-01-01 UTC.",
    )

    qc(
        "BLOCKING",
        "NO_Q_INPUT_FEATURES",
        not any(
            feature.lower().startswith("q_")
            for feature in feature_cols
        ),
        f"features={feature_cols}",
    )

    qc(
        "BLOCKING",
        "NO_NEGATIVE_USGS_Q",
        negative_q_count == 0,
        f"negative_raw_q_count={negative_q_count}",
    )

    qc(
        "QUALITY",
        "TRAIN_Q_AVAILABILITY",
        train_q_availability
        >= args.min_period_q_availability_percent,
        (
            f"availability={train_q_availability:.6f}%; "
            f"minimum={args.min_period_q_availability_percent:.3f}%"
        ),
    )

    qc(
        "QUALITY",
        "VALIDATION_Q_AVAILABILITY",
        val_q_availability
        >= args.min_period_q_availability_percent,
        (
            f"availability={val_q_availability:.6f}%; "
            f"minimum={args.min_period_q_availability_percent:.3f}%"
        ),
    )

    qc(
        "QUALITY",
        "TRAIN_MODEL_READY_ROWS",
        len(train) >= args.min_train_rows,
        (
            f"model_ready_rows={len(train)}; "
            f"minimum={args.min_train_rows}"
        ),
    )

    qc(
        "QUALITY",
        "VALIDATION_MODEL_READY_ROWS",
        len(val) >= args.min_validation_rows,
        (
            f"model_ready_rows={len(val)}; "
            f"minimum={args.min_validation_rows}"
        ),
    )

    qc(
        "QUALITY",
        "USGS_SAMPLING_INTERVAL",
        np.isfinite(median_interval)
        and median_interval <= 30.0,
        (
            f"median_raw_sampling_interval_minutes="
            f"{median_interval:.6f}"
        ),
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "TEMPORAL_SPLIT",
            "status": "SELECTED",
            "detail": (
                f"TRAIN < {iso_z(training_end)}; "
                f"VALIDATION >= {iso_z(training_end)}; "
                "no random temporal split."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "RAINFALL_HISTORY_POLICY",
            "status": "SELECTED",
            "detail": (
                "Maximum predictor history is 168 h. Any sample whose "
                "required rainfall history contains a missing Step 8A hour "
                "is excluded. No rainfall interpolation."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_CONSTRAINT",
            "check": "FINAL_TEST_UNTOUCHED",
            "status": "PASS",
            "detail": (
                "Hurricane Florence 2018 is absent from Step 8B training "
                "and validation data and remains reserved for final testing."
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

    safe = (
        len(blocking_fail) == 0
        and len(quality_fail) == 0
    )
    status = (
        "PASS_HISTORICAL_USGS_DATASET_READY"
        if safe
        else (
            "FAIL_HISTORICAL_USGS_DATASET_BLOCKING"
            if len(blocking_fail)
            else "FAIL_HISTORICAL_USGS_DATASET_QUALITY"
        )
    )

    usgs_metadata = {
        "source": USGS_IV_URL,
        "site": args.site,
        "parameter_code": "00060",
        "source_unit": "ft3/s",
        "output_unit": "m3/s",
        "conversion_factor": CFS_TO_M3S,
        "requested_start_utc": iso_z(start),
        "requested_end_utc_exclusive": iso_z(end),
        "raw_record_count": int(len(raw)),
        "median_sampling_interval_minutes": median_interval,
        "hourly_min_coverage_percent": args.min_q_hourly_coverage_percent,
        "chunks": chunks,
        "raw_file": str(raw_path),
        "hourly_file": str(hourly_q_path),
        "chunk_manifest_file": str(usgs_chunk_manifest_path),
        "chunk_days": args.usgs_chunk_days,
        "retry_base_seconds": args.retry_base_seconds,
        "retry_max_seconds": args.retry_max_seconds,
    }
    atomic_json(usgs_metadata, usgs_meta_path)

    metadata = {
        "status": status,
        "workflow": "historical_usgs_discharge_dataset",
        "script_build": SCRIPT_BUILD,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": {
            "start": iso_z(start),
            "training_end": iso_z(training_end),
            "end_exclusive": iso_z(end),
        },
        "usgs": usgs_metadata,
        "dataset": {
            "full_rows": int(len(dataset)),
            "train_total_hours": int(len(train_all)),
            "validation_total_hours": int(len(val_all)),
            "train_q_availability_percent": train_q_availability,
            "validation_q_availability_percent": val_q_availability,
            "train_rain_feature_ready_percent": train_feature_ready,
            "validation_rain_feature_ready_percent": val_feature_ready,
            "train_model_ready_rows": int(len(train)),
            "validation_model_ready_rows": int(len(val)),
            "feature_count": int(len(feature_cols)),
            "features": feature_cols,
            "target": "q_mean_m3s",
        },
        "leakage_controls": {
            "observed_q_used_as_predictor": False,
            "future_rainfall_used": False,
            "random_temporal_split": False,
            "florence_2018_used": False,
            "missing_rainfall_interpolated": False,
        },
        "blocking_failure_count": int(len(blocking_fail)),
        "quality_failure_count": int(len(quality_fail)),
        "safe_for_model_training": bool(safe),
        "outputs": {
            "full_dataset": str(full_path),
            "train": str(train_path),
            "validation": str(validation_path),
            "feature_manifest": str(feature_path),
            "qc": str(qc_path),
            "metadata": str(metadata_path),
        },
    }
    atomic_json(metadata, metadata_path)

    print()
    print("USGS discharge")
    print(f"Raw observations                   : {len(raw):,}")
    print(f"Median sampling interval           : {median_interval:.3f} min")
    print(f"Negative raw Q                     : {negative_q_count}")
    print()
    print("ML dataset")
    print(f"Full hourly rows                   : {len(dataset):,}")
    print(f"Feature count                      : {len(feature_cols)}")
    print(f"Training Q availability            : {train_q_availability:.6f} %")
    print(f"Validation Q availability          : {val_q_availability:.6f} %")
    print(f"Training rain-feature ready        : {train_feature_ready:.6f} %")
    print(f"Validation rain-feature ready      : {val_feature_ready:.6f} %")
    print(f"Training model-ready rows          : {len(train):,}")
    print(f"Validation model-ready rows        : {len(val):,}")
    print()
    print(f"Training dataset                   : {train_path}")
    print(f"Validation dataset                 : {validation_path}")
    print(f"QC                                 : {qc_path}")
    print(f"Metadata                           : {metadata_path}")
    print()
    print(f"Blocking failures                  : {len(blocking_fail)}")
    print(f"Quality failures                   : {len(quality_fail)}")
    print(f"Safe for Step 8C                   : {'YES' if safe else 'NO'}")
    print(f"Status                             : {status}")

    if not safe:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
