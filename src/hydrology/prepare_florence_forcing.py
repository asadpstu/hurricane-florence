"""
STEP 6 - Prepare aligned hourly hydrologic forcing for USGS 02089000.

Purpose
-------
Create the model-ready hourly forcing/observation package for the physical
rainfall-runoff model in Step 7.

Inputs
------
- Step 5C basin_hourly_rainfall.csv
- Step 5C rainfall_final_metadata.json
- Validated Step 1 USGS event observations directory

Time convention
---------------
MRMS GaugeCorr_QPE_01H timestamps are the END of the one-hour accumulation.
For a rainfall value stamped at time T:

    interval_start = T - 1 hour
    interval_end   = T

Observed USGS discharge is aggregated to the same interval using the mean of
all valid instantaneous observations with:

    interval_start < observation_time <= interval_end

The original rainfall timestamp is preserved as interval_end_utc.

No discharge interpolation is performed.

Hydrologic diagnostics
----------------------
- Convert mean discharge (m3/s) to an equivalent runoff depth (mm/h) over the
  authoritative 6,232.005499 km2 watershed.
- Compute cumulative rainfall and observed runoff depth over intervals with
  observed Q.
- Compute an event water-balance ratio as a diagnostic only. It is NOT a
  direct-response runoff coefficient because streamflow contains antecedent
  storage/baseflow and the event may not fully return to pre-event storage.

Default windows
---------------
Warm-up / antecedent:
  2018-09-01T00:00Z to 2018-09-10T00:00Z

Evaluation:
  2018-09-10T00:00Z to 2018-09-26T00:00Z

Outputs
-------
output/hydrology/florence_2018_forcing/
  hydrologic_forcing_hourly.csv
  hydrologic_forcing_daily.csv
  hydrologic_forcing_summary.csv
  hydrologic_forcing_qc.csv
  hydrologic_forcing_metadata.json
  forcing_rainfall_discharge.png
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_BUILD = "STEP_6_HYDROLOGIC_FORCING_V2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--basin-hourly-rainfall",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--rainfall-metadata",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--usgs-observation-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--watershed-area-km2",
        type=float,
        default=6232.005499,
    )
    p.add_argument(
        "--warmup-start",
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
        "--min-evaluation-q-coverage-percent",
        type=float,
        default=95.0,
    )
    p.add_argument(
        "--min-warmup-q-coverage-percent",
        type=float,
        default=95.0,
        help=(
            "Minimum hourly observed-Q coverage during the antecedent "
            "warm-up window. This protects physical-model initialization."
        ),
    )
    p.add_argument(
        "--max-negative-q-count",
        type=int,
        default=0,
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
    for c in priority:
        if c.lower() in lookup:
            return lookup[c.lower()]
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
            sample = pd.read_csv(csv_path, nrows=100)
        except Exception:
            continue

        tcol = detect_time_column(list(sample.columns))
        qcol, unit = detect_discharge_column(list(sample.columns))

        if tcol is None or qcol is None:
            continue

        score = 100
        low = csv_path.name.lower()
        if "observation" in low:
            score += 30
        if "event" in low:
            score += 20
        if "florence" in low:
            score += 20
        if "02089000" in low:
            score += 20
        if unit == "m3/s":
            score += 10

        candidates.append(
            {
                "path": csv_path,
                "time_column": tcol,
                "q_column": qcol,
                "unit": unit,
                "score": score,
            }
        )

    if not candidates:
        raise RuntimeError(
            f"No suitable USGS observation CSV found under {root}"
        )

    candidates.sort(
        key=lambda x: (-x["score"], str(x["path"]))
    )
    selected = candidates[0]

    raw = pd.read_csv(selected["path"])
    time = pd.to_datetime(
        raw[selected["time_column"]],
        utc=True,
        errors="coerce",
    )
    q = pd.to_numeric(
        raw[selected["q_column"]],
        errors="coerce",
    )

    if selected["unit"] == "cfs":
        q = q * 0.028316846592

    obs = pd.DataFrame(
        {
            "time_utc": time,
            "discharge_m3s": q,
        }
    )
    obs = obs.dropna().sort_values("time_utc")
    obs = obs.drop_duplicates("time_utc").reset_index(drop=True)

    if obs.empty:
        raise RuntimeError(
            "Selected USGS observation file contains no valid Q/time rows."
        )

    return obs, {
        "selected_path": str(selected["path"]),
        "time_column": selected["time_column"],
        "discharge_column": selected["q_column"],
        "original_unit": selected["unit"],
        "valid_rows": int(len(obs)),
        "candidate_count": int(len(candidates)),
    }


def aggregate_q_to_rainfall_intervals(
    obs: pd.DataFrame,
    interval_ends: pd.Series,
) -> pd.DataFrame:
    """
    Aggregate observations to (T-1h, T] using arithmetic mean.
    """
    obs = obs.copy().set_index("time_utc")
    records = []

    for end in interval_ends:
        start = end - pd.Timedelta(hours=1)
        values = obs.loc[
            (obs.index > start) & (obs.index <= end),
            "discharge_m3s",
        ]

        records.append(
            {
                "interval_end_utc": end,
                "q_obs_count": int(values.notna().sum()),
                "q_obs_mean_m3s": (
                    float(values.mean())
                    if values.notna().any()
                    else np.nan
                ),
                "q_obs_min_m3s": (
                    float(values.min())
                    if values.notna().any()
                    else np.nan
                ),
                "q_obs_max_m3s": (
                    float(values.max())
                    if values.notna().any()
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()

    prepare_dir(args.output_dir, args.overwrite)

    warmup_start = parse_utc(args.warmup_start)
    eval_start = parse_utc(args.evaluation_start)
    eval_end = parse_utc(args.evaluation_end)

    if not (warmup_start < eval_start < eval_end):
        raise ValueError(
            "Require warmup-start < evaluation-start < evaluation-end."
        )

    rainfall_meta = json.loads(
        args.rainfall_metadata.read_text(encoding="utf-8")
    )
    if not bool(rainfall_meta.get("safe_for_step_6")):
        raise RuntimeError(
            "Step 5 metadata does not authorize Step 6."
        )

    rain = pd.read_csv(args.basin_hourly_rainfall)
    if "time_utc" not in rain.columns:
        raise RuntimeError(
            "Rainfall CSV must contain time_utc."
        )
    if "basin_mean_rainfall_mm" not in rain.columns:
        raise RuntimeError(
            "Rainfall CSV must contain basin_mean_rainfall_mm."
        )

    rain["interval_end_utc"] = pd.to_datetime(
        rain["time_utc"],
        utc=True,
        errors="coerce",
    )
    if rain["interval_end_utc"].isna().any():
        raise RuntimeError(
            "Rainfall CSV contains invalid timestamps."
        )

    rain = rain.sort_values("interval_end_utc").reset_index(drop=True)
    rain["interval_start_utc"] = (
        rain["interval_end_utc"]
        - pd.Timedelta(hours=1)
    )

    # Enforce strict hourly continuity.
    dt = (
        rain["interval_end_utc"]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 3600.0
    )
    nonhourly_gaps = int((~np.isclose(dt, 1.0)).sum())

    obs, obs_info = find_usgs_observations(
        args.usgs_observation_dir
    )

    q_hourly = aggregate_q_to_rainfall_intervals(
        obs,
        rain["interval_end_utc"],
    )

    forcing = rain.merge(
        q_hourly,
        on="interval_end_utc",
        how="left",
        validate="one_to_one",
    )

    forcing["rainfall_mm"] = pd.to_numeric(
        forcing["basin_mean_rainfall_mm"],
        errors="coerce",
    )

    forcing["has_observed_q"] = (
        forcing["q_obs_mean_m3s"].notna()
    )

    area_m2 = args.watershed_area_km2 * 1_000_000.0

    # Equivalent observed streamflow volume depth during each 1-h interval.
    forcing["observed_runoff_depth_mm"] = (
        forcing["q_obs_mean_m3s"]
        * 3600.0
        / area_m2
        * 1000.0
    )

    forcing["period"] = "outside_selected_windows"
    forcing.loc[
        (
            forcing["interval_end_utc"] > warmup_start
        )
        & (
            forcing["interval_end_utc"] <= eval_start
        ),
        "period",
    ] = "warmup"

    forcing.loc[
        (
            forcing["interval_end_utc"] > eval_start
        )
        & (
            forcing["interval_end_utc"] <= eval_end
        ),
        "period",
    ] = "evaluation"

    # This reflects the accumulation interval convention. A rainfall file
    # valid at evaluation_start covers the preceding hour and belongs to
    # warmup; a file valid at evaluation_end covers the final evaluation hour.
    evaluation = forcing[
        forcing["period"] == "evaluation"
    ].copy()
    warmup = forcing[
        forcing["period"] == "warmup"
    ].copy()

    expected_eval_hours = int(
        (eval_end - eval_start)
        / pd.Timedelta(hours=1)
    )
    expected_warmup_hours = int(
        (eval_start - warmup_start)
        / pd.Timedelta(hours=1)
    )

    eval_q_coverage_pct = float(
        evaluation["has_observed_q"].mean()
        * 100.0
    )
    warmup_q_coverage_pct = float(
        warmup["has_observed_q"].mean()
        * 100.0
    )

    negative_q_count = int(
        (
            forcing["q_obs_mean_m3s"].notna()
            & (forcing["q_obs_mean_m3s"] < 0.0)
        ).sum()
    )

    # Cumulative forcing diagnostics.
    forcing["cumulative_rainfall_mm"] = (
        forcing["rainfall_mm"].fillna(0.0).cumsum()
    )
    forcing["cumulative_observed_runoff_depth_mm"] = (
        forcing["observed_runoff_depth_mm"]
        .fillna(0.0)
        .cumsum()
    )

    # Evaluation water balance on common Q-valid hours only.
    eval_common = evaluation[
        evaluation["has_observed_q"]
        & evaluation["rainfall_mm"].notna()
    ].copy()

    eval_rain_common_mm = float(
        eval_common["rainfall_mm"].sum()
    )
    eval_runoff_common_mm = float(
        eval_common["observed_runoff_depth_mm"].sum()
    )
    apparent_water_balance_ratio = (
        eval_runoff_common_mm / eval_rain_common_mm
        if eval_rain_common_mm > 0
        else None
    )

    # Initial / pre-event discharge context.
    warmup_q = warmup.loc[
        warmup["has_observed_q"],
        "q_obs_mean_m3s",
    ]
    initial_q = (
        float(warmup_q.iloc[-1])
        if len(warmup_q)
        else None
    )
    warmup_q_median = (
        float(warmup_q.median())
        if len(warmup_q)
        else None
    )

    evaluation_q = evaluation.loc[
        evaluation["has_observed_q"],
        ["interval_end_utc", "q_obs_mean_m3s"],
    ]
    first_evaluation_q = (
        float(evaluation_q.iloc[0]["q_obs_mean_m3s"])
        if len(evaluation_q)
        else None
    )
    first_evaluation_q_time = (
        evaluation_q.iloc[0]["interval_end_utc"]
        if len(evaluation_q)
        else None
    )

    # Daily aligned forcing.
    daily = (
        forcing.set_index("interval_end_utc")
        .resample("1D")
        .agg(
            rainfall_mm=("rainfall_mm", "sum"),
            observed_q_mean_m3s=(
                "q_obs_mean_m3s",
                "mean",
            ),
            observed_q_max_m3s=(
                "q_obs_mean_m3s",
                "max",
            ),
            observed_q_valid_hours=(
                "has_observed_q",
                "sum",
            ),
            observed_runoff_depth_mm=(
                "observed_runoff_depth_mm",
                "sum",
            ),
        )
        .reset_index()
    )

    # QC.
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
        "RAIN_FORCING_HOURLY_CONTINUITY",
        nonhourly_gaps == 0,
        f"nonhourly_gaps={nonhourly_gaps}",
    )

    qc(
        "BLOCKING",
        "WARMUP_HOUR_COUNT",
        len(warmup) == expected_warmup_hours,
        (
            f"warmup_hours={len(warmup)}; "
            f"expected={expected_warmup_hours}"
        ),
    )

    qc(
        "BLOCKING",
        "EVALUATION_HOUR_COUNT",
        len(evaluation) == expected_eval_hours,
        (
            f"evaluation_hours={len(evaluation)}; "
            f"expected={expected_eval_hours}"
        ),
    )

    qc(
        "BLOCKING",
        "WARMUP_Q_COVERAGE_FOR_MODEL_INITIALIZATION",
        warmup_q_coverage_pct
        >= args.min_warmup_q_coverage_percent,
        (
            f"coverage={warmup_q_coverage_pct:.6f}%; "
            f"minimum={args.min_warmup_q_coverage_percent:.3f}%; "
            f"initial_q_from_last_warmup_hour={initial_q}"
        ),
    )

    qc(
        "BLOCKING",
        "INITIAL_Q_AVAILABLE_AT_END_OF_WARMUP",
        initial_q is not None,
        (
            f"initial_q_m3s={initial_q}; "
            f"warmup_end={iso_z(eval_start)}"
        ),
    )

    qc(
        "QUALITY",
        "EVALUATION_Q_COVERAGE",
        eval_q_coverage_pct
        >= args.min_evaluation_q_coverage_percent,
        (
            f"coverage={eval_q_coverage_pct:.6f}%; "
            f"minimum={args.min_evaluation_q_coverage_percent:.3f}%"
        ),
    )

    qc(
        "BLOCKING",
        "NONNEGATIVE_OBSERVED_Q",
        negative_q_count <= args.max_negative_q_count,
        (
            f"negative_hourly_Q_count={negative_q_count}; "
            f"maximum={args.max_negative_q_count}"
        ),
    )

    qc(
        "BLOCKING",
        "RAINFALL_VALUES_COMPLETE",
        forcing["rainfall_mm"].notna().all(),
        (
            f"missing_rainfall_hours="
            f"{int(forcing['rainfall_mm'].isna().sum())}"
        ),
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "MRMS_INTERVAL_CONVENTION",
            "status": "SELECTED",
            "detail": (
                "MRMS 1-hour QPE timestamp is treated as interval end: "
                "(T-1h, T]. USGS instantaneous observations are averaged "
                "within the same interval."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "WARMUP_WINDOW",
            "status": "SELECTED",
            "detail": (
                f"{iso_z(warmup_start)} to {iso_z(eval_start)}; "
                f"{len(warmup)} hourly forcing intervals."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_PARAMETER",
            "check": "EVALUATION_WINDOW",
            "status": "SELECTED",
            "detail": (
                f"{iso_z(eval_start)} to {iso_z(eval_end)}; "
                f"{len(evaluation)} hourly forcing intervals."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "NOTE",
            "check": "APPARENT_EVENT_WATER_BALANCE",
            "status": "RECORDED",
            "detail": (
                f"On Q-valid evaluation hours: rainfall="
                f"{eval_rain_common_mm:.3f} mm; observed streamflow "
                f"equivalent depth={eval_runoff_common_mm:.3f} mm; "
                f"ratio={apparent_water_balance_ratio}. This is not a "
                "direct-runoff coefficient because Q includes antecedent "
                "storage/baseflow and the finite event window does not "
                "close the catchment storage balance."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_CONSTRAINT",
            "check": "NO_DISCHARGE_GAP_FILLING",
            "status": "OPEN",
            "detail": (
                "Missing USGS hourly discharge intervals remain missing. "
                "Step 7 objective functions must evaluate only intervals "
                "with valid observed Q."
            ),
        }
    )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_CONSTRAINT",
            "check": "STEP6_DOES_NOT_CALIBRATE_MODEL",
            "status": "OPEN",
            "detail": (
                "This step only aligns forcing and observations. Rainfall "
                "losses, routing, baseflow parameters, and model calibration "
                "are deferred to Step 7."
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
        status = "FAIL_FLORENCE_HYDROLOGIC_FORCING"
    elif len(quality_fail):
        status = "FAIL_FLORENCE_HYDROLOGIC_FORCING_QUALITY"
    else:
        status = "PASS_FLORENCE_HYDROLOGIC_FORCING_READY"

    # Outputs.
    hourly_path = (
        args.output_dir / "hydrologic_forcing_hourly.csv"
    )
    daily_path = (
        args.output_dir / "hydrologic_forcing_daily.csv"
    )
    summary_path = (
        args.output_dir / "hydrologic_forcing_summary.csv"
    )
    qc_path = (
        args.output_dir / "hydrologic_forcing_qc.csv"
    )
    metadata_path = (
        args.output_dir / "hydrologic_forcing_metadata.json"
    )
    plot_path = (
        args.output_dir / "forcing_rainfall_discharge.png"
    )

    hourly_out = forcing.copy()
    for col in [
        "interval_start_utc",
        "interval_end_utc",
    ]:
        hourly_out[col] = hourly_out[col].map(iso_z)
    # Original Step 5 timestamp duplicated by interval_end; remove to avoid
    # ambiguous time semantics in downstream code.
    if "time_utc" in hourly_out.columns:
        hourly_out = hourly_out.drop(columns=["time_utc"])
    atomic_csv(hourly_out, hourly_path)

    daily_out = daily.copy()
    daily_out["interval_end_utc"] = daily_out[
        "interval_end_utc"
    ].map(iso_z)
    atomic_csv(daily_out, daily_path)

    summary = pd.DataFrame(
        [
            {
                "metric": "watershed_area_km2",
                "value": args.watershed_area_km2,
                "unit": "km2",
            },
            {
                "metric": "warmup_hours",
                "value": len(warmup),
                "unit": "hours",
            },
            {
                "metric": "evaluation_hours",
                "value": len(evaluation),
                "unit": "hours",
            },
            {
                "metric": "warmup_q_coverage",
                "value": warmup_q_coverage_pct,
                "unit": "percent",
            },
            {
                "metric": "evaluation_q_coverage",
                "value": eval_q_coverage_pct,
                "unit": "percent",
            },
            {
                "metric": "evaluation_rainfall_common_q_hours",
                "value": eval_rain_common_mm,
                "unit": "mm",
            },
            {
                "metric": "evaluation_observed_runoff_depth_common_q_hours",
                "value": eval_runoff_common_mm,
                "unit": "mm",
            },
            {
                "metric": "apparent_water_balance_ratio",
                "value": apparent_water_balance_ratio,
                "unit": "dimensionless",
            },
            {
                "metric": "initial_observed_q_near_evaluation_start",
                "value": initial_q,
                "unit": "m3/s",
            },
            {
                "metric": "warmup_observed_q_median",
                "value": warmup_q_median,
                "unit": "m3/s",
            },
        ]
    )
    atomic_csv(summary, summary_path)
    atomic_csv(qc_df, qc_path)

    # Plot.
    p = forcing.set_index("interval_end_utc")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        p.index,
        p["rainfall_mm"],
        width=0.035,
        label="MRMS basin rainfall",
    )
    ax.set_ylabel("Rainfall accumulation (mm / 1 h)")
    ax.set_xlabel("Interval end (UTC)")
    ax.invert_yaxis()

    ax2 = ax.twinx()
    ax2.plot(
        p.index,
        p["q_obs_mean_m3s"],
        label="Observed hourly mean Q",
    )
    ax2.set_ylabel("Observed discharge (m³/s)")
    ax.set_title(
        "Florence 2018 — Model-Ready Hourly Hydrologic Forcing"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    metadata = {
        "status": status,
        "step": "STEP_6",
        "script_build": SCRIPT_BUILD,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "basin_hourly_rainfall": str(
                args.basin_hourly_rainfall
            ),
            "basin_hourly_rainfall_sha256": sha256_file(
                args.basin_hourly_rainfall
            ),
            "rainfall_metadata": str(args.rainfall_metadata),
            "usgs_observation_directory": str(
                args.usgs_observation_dir
            ),
            "selected_usgs_observations": obs_info,
        },
        "time_convention": {
            "mrms_timestamp_role": "one_hour_accumulation_interval_end",
            "rainfall_interval": "(T-1h, T]",
            "usgs_q_aggregation": (
                "arithmetic mean of valid instantaneous Q observations "
                "with T-1h < observation_time <= T"
            ),
            "discharge_interpolation_applied": False,
        },
        "windows": {
            "warmup_start_utc": iso_z(warmup_start),
            "evaluation_start_utc": iso_z(eval_start),
            "evaluation_end_utc_exclusive": iso_z(eval_end),
            "warmup_hours": int(len(warmup)),
            "evaluation_hours": int(len(evaluation)),
        },
        "watershed": {
            "area_km2": args.watershed_area_km2,
            "area_m2": area_m2,
        },
        "observed_q": {
            "warmup_coverage_percent": warmup_q_coverage_pct,
            "evaluation_coverage_percent": eval_q_coverage_pct,
            "negative_hour_count": negative_q_count,
            "initial_q_at_end_of_warmup_m3s": initial_q,
            "warmup_q_median_m3s": warmup_q_median,
            "first_evaluation_q_m3s": first_evaluation_q,
            "first_evaluation_q_time_utc": (
                iso_z(first_evaluation_q_time)
                if first_evaluation_q_time is not None
                else None
            ),
        },
        "water_balance_diagnostic": {
            "common_q_valid_evaluation_rainfall_mm": eval_rain_common_mm,
            "common_q_valid_observed_runoff_depth_mm": eval_runoff_common_mm,
            "apparent_ratio": apparent_water_balance_ratio,
            "is_direct_runoff_coefficient": False,
        },
        "scientific_constraints": {
            "step6_calibrates_hydrologic_model": False,
            "missing_q_gap_filling_applied": False,
            "runoff_depth_includes_baseflow_and_storage_release": True,
        },
        "blocking_failure_count": int(len(blocking_fail)),
        "quality_failure_count": int(len(quality_fail)),
        "step_6_complete": bool(
            len(blocking_fail) == 0
            and len(quality_fail) == 0
        ),
        "safe_for_step_7": bool(
            len(blocking_fail) == 0
            and len(quality_fail) == 0
        ),
        "output_paths": {
            "hourly_forcing": str(hourly_path),
            "daily_forcing": str(daily_path),
            "summary": str(summary_path),
            "qc": str(qc_path),
            "metadata": str(metadata_path),
            "plot": str(plot_path),
        },
    }
    atomic_json(metadata, metadata_path)

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {SCRIPT_BUILD}")
    print("STEP 6 - HYDROLOGIC FORCING PREPARATION")
    print("=" * 100)
    print(
        f"Rainfall forcing hours             : "
        f"{len(forcing):,}"
    )
    print(
        f"Warm-up hours                      : "
        f"{len(warmup):,}/{expected_warmup_hours:,}"
    )
    print(
        f"Evaluation hours                   : "
        f"{len(evaluation):,}/{expected_eval_hours:,}"
    )
    print()
    print(
        f"Warm-up observed-Q coverage        : "
        f"{warmup_q_coverage_pct:.6f} %"
    )
    print(
        f"Evaluation observed-Q coverage     : "
        f"{eval_q_coverage_pct:.6f} %"
    )
    print(
        f"Initial Q at end of warm-up        : "
        f"{initial_q} m3/s"
    )
    print(
        f"Warm-up median Q                   : "
        f"{warmup_q_median} m3/s"
    )
    print()
    print(
        f"Rainfall on common Q-valid hours   : "
        f"{eval_rain_common_mm:.3f} mm"
    )
    print(
        f"Observed runoff equivalent depth   : "
        f"{eval_runoff_common_mm:.3f} mm"
    )
    print(
        f"Apparent water-balance ratio       : "
        f"{apparent_water_balance_ratio}"
    )
    print()
    print(f"Hourly forcing                     : {hourly_path}")
    print(f"Daily forcing                      : {daily_path}")
    print(f"Summary                            : {summary_path}")
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
        f"Step 6 complete                    : "
        f"{'YES' if metadata['step_6_complete'] else 'NO'}"
    )
    print(
        f"Safe for Step 7                    : "
        f"{'YES' if metadata['safe_for_step_7'] else 'NO'}"
    )
    print(f"Status                             : {status}")

    if len(blocking_fail) or len(quality_fail):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
