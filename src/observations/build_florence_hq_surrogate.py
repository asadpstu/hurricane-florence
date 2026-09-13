"""
Step 1E - Reconstruct the effective Hurricane Florence 2018 H-Q surrogate.

This is NOT an official historical USGS rating curve.

The continuous USGS discharge series is a derived product. This step reconstructs
the effective stage-discharge behavior present in that published event record,
while preserving separate rising- and falling-limb relations.

Why separate limbs?
-------------------
For an unsteady flood, the same gage height can correspond to different
discharges on the rising and falling limbs (hysteresis and/or time-varying
rating shifts). A single fitted curve would hide this behavior.

Method
------
1. Read paired continuous gage-height/discharge observations.
2. Keep valid, approved observations.
3. Identify peak gage-height time.
4. Split observations into rising and falling limbs at the peak.
5. Aggregate duplicate stage values robustly using the median discharge.
6. Fit a monotonic isotonic regression separately to each limb.
7. Export:
   - empirical H-Q points
   - dense H->Q branch curves
   - dense Q->H inverse branch curves
   - hysteresis comparison over the common stage range
   - reconstruction-fidelity metrics
   - QC + metadata

Scientific boundary
-------------------
The output must be called an "effective Florence 2018 H-Q surrogate", not an
"official 2018 USGS rating", unless the actual historical USGS rating/shift
files are later obtained.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


CFS_TO_CMS = 0.028316846592
FT_TO_M = 0.3048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a limb-specific Florence 2018 effective H-Q surrogate."
    )
    parser.add_argument(
        "--event-observations",
        type=Path,
        required=True,
        help="Step 1 usgs_event_observations.csv",
    )
    parser.add_argument(
        "--event-metadata",
        type=Path,
        default=None,
        help="Optional Step 1 metadata JSON.",
    )
    parser.add_argument(
        "--stage-step-ft",
        type=float,
        default=0.01,
        help="Dense H->Q output spacing in feet.",
    )
    parser.add_argument(
        "--discharge-step-cfs",
        type=float,
        default=25.0,
        help="Dense Q->H inverse output spacing in ft3/s.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/observations/florence_2018_effective_hq"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.partial{path.suffix}")
    temp.unlink(missing_ok=True)
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_write_json(payload: Any, path: Path) -> None:
    atomic_write_text(json.dumps(payload, indent=2), path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.partial{path.suffix}")
    temp.unlink(missing_ok=True)
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [p for p in path.iterdir() if p.is_file()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} is not empty. Use --overwrite.")
    if overwrite:
        for p in existing:
            p.unlink()


def qualifier_has_approved(value: Any) -> bool:
    if pd.isna(value):
        return False
    tokens = {token.strip() for token in str(value).split("|")}
    return "A" in tokens


def qualifier_has_estimated(value: Any) -> bool:
    if pd.isna(value):
        return False
    tokens = {token.strip() for token in str(value).split("|")}
    return "e" in tokens


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def mae(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - obs)))


def mape(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & (obs != 0) & np.isfinite(pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - obs[mask]) / obs[mask])) * 100.0)


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def aggregate_limb(frame: pd.DataFrame, limb: str) -> pd.DataFrame:
    subset = frame.loc[frame["limb"].eq(limb)].copy()
    if subset.empty:
        raise RuntimeError(f"No observations available for limb={limb}.")

    # Original gage height is reported to 0.01 ft, so group at that precision.
    subset["stage_key_ft"] = subset["gage_height_ft"].round(2)

    grouped = (
        subset.groupby("stage_key_ft", as_index=False)
        .agg(
            observation_count=("discharge_cfs", "size"),
            discharge_median_cfs=("discharge_cfs", "median"),
            discharge_min_cfs=("discharge_cfs", "min"),
            discharge_max_cfs=("discharge_cfs", "max"),
            discharge_mean_cfs=("discharge_cfs", "mean"),
            discharge_std_cfs=("discharge_cfs", "std"),
            first_time_utc=("datetime_utc", "min"),
            last_time_utc=("datetime_utc", "max"),
            estimated_discharge_count=("discharge_is_estimated", "sum"),
        )
        .rename(columns={"stage_key_ft": "gage_height_ft"})
        .sort_values("gage_height_ft")
        .reset_index(drop=True)
    )

    grouped["gage_height_m"] = grouped["gage_height_ft"] * FT_TO_M
    grouped["discharge_median_cms"] = (
        grouped["discharge_median_cfs"] * CFS_TO_CMS
    )
    grouped["discharge_range_cfs"] = (
        grouped["discharge_max_cfs"] - grouped["discharge_min_cfs"]
    )
    grouped["limb"] = limb
    return grouped


def fit_isotonic(grouped: pd.DataFrame) -> IsotonicRegression:
    x = grouped["gage_height_ft"].to_numpy(dtype=float)
    y = grouped["discharge_median_cfs"].to_numpy(dtype=float)
    w = grouped["observation_count"].to_numpy(dtype=float)

    if len(np.unique(x)) < 3:
        raise RuntimeError("At least 3 unique stages are required for isotonic fitting.")

    model = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=0.0,
    )
    model.fit(x, y, sample_weight=w)
    return model


def predict_only_inside(
    model: IsotonicRegression,
    stages: np.ndarray,
    min_stage: float,
    max_stage: float,
) -> np.ndarray:
    pred = np.full(stages.shape, np.nan, dtype=float)
    mask = (stages >= min_stage) & (stages <= max_stage)
    if mask.any():
        pred[mask] = model.predict(stages[mask])
    return pred


def dense_h_to_q(
    rising_grouped: pd.DataFrame,
    falling_grouped: pd.DataFrame,
    rising_model: IsotonicRegression,
    falling_model: IsotonicRegression,
    step_ft: float,
) -> pd.DataFrame:
    global_min = min(
        rising_grouped["gage_height_ft"].min(),
        falling_grouped["gage_height_ft"].min(),
    )
    global_max = max(
        rising_grouped["gage_height_ft"].max(),
        falling_grouped["gage_height_ft"].max(),
    )

    stages = np.arange(
        math.floor(global_min / step_ft) * step_ft,
        math.ceil(global_max / step_ft) * step_ft + step_ft * 0.5,
        step_ft,
    )
    stages = np.round(stages, 6)

    rmin, rmax = (
        float(rising_grouped["gage_height_ft"].min()),
        float(rising_grouped["gage_height_ft"].max()),
    )
    fmin, fmax = (
        float(falling_grouped["gage_height_ft"].min()),
        float(falling_grouped["gage_height_ft"].max()),
    )

    rq = predict_only_inside(rising_model, stages, rmin, rmax)
    fq = predict_only_inside(falling_model, stages, fmin, fmax)

    out = pd.DataFrame(
        {
            "gage_height_ft": stages,
            "gage_height_m": stages * FT_TO_M,
            "rising_discharge_cfs": rq,
            "falling_discharge_cfs": fq,
        }
    )
    out["rising_discharge_cms"] = out["rising_discharge_cfs"] * CFS_TO_CMS
    out["falling_discharge_cms"] = out["falling_discharge_cfs"] * CFS_TO_CMS
    out["branch_min_discharge_cfs"] = out[
        ["rising_discharge_cfs", "falling_discharge_cfs"]
    ].min(axis=1, skipna=True)
    out["branch_max_discharge_cfs"] = out[
        ["rising_discharge_cfs", "falling_discharge_cfs"]
    ].max(axis=1, skipna=True)
    return out


def monotonic_unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = (
        pd.DataFrame({"x": x, "y": y})
        .dropna()
        .groupby("x", as_index=False)["y"]
        .median()
        .sort_values("x")
    )
    return frame["x"].to_numpy(float), frame["y"].to_numpy(float)


def dense_q_to_h(
    hq: pd.DataFrame,
    discharge_step_cfs: float,
) -> pd.DataFrame:
    series = {}

    q_min_candidates = []
    q_max_candidates = []

    for limb in ("rising", "falling"):
        qcol = f"{limb}_discharge_cfs"
        valid = hq[["gage_height_ft", qcol]].dropna()
        q, h = monotonic_unique_xy(
            valid[qcol].to_numpy(float),
            valid["gage_height_ft"].to_numpy(float),
        )
        if len(q) < 2:
            continue
        series[limb] = (q, h)
        q_min_candidates.append(float(q.min()))
        q_max_candidates.append(float(q.max()))

    if not series:
        raise RuntimeError("Cannot create inverse Q->H surrogate.")

    qmin = min(q_min_candidates)
    qmax = max(q_max_candidates)

    qgrid = np.arange(
        math.floor(qmin / discharge_step_cfs) * discharge_step_cfs,
        math.ceil(qmax / discharge_step_cfs) * discharge_step_cfs
        + discharge_step_cfs * 0.5,
        discharge_step_cfs,
    )

    out = pd.DataFrame({"discharge_cfs": qgrid})
    out["discharge_cms"] = out["discharge_cfs"] * CFS_TO_CMS

    for limb, (q, h) in series.items():
        pred = np.full(qgrid.shape, np.nan, dtype=float)
        mask = (qgrid >= q.min()) & (qgrid <= q.max())
        pred[mask] = np.interp(qgrid[mask], q, h)
        out[f"{limb}_gage_height_ft"] = pred
        out[f"{limb}_gage_height_m"] = pred * FT_TO_M

    if {"rising_gage_height_ft", "falling_gage_height_ft"}.issubset(out.columns):
        out["branch_stage_difference_ft"] = (
            out["falling_gage_height_ft"] - out["rising_gage_height_ft"]
        )
        out["branch_stage_difference_m"] = (
            out["branch_stage_difference_ft"] * FT_TO_M
        )

    return out


def fidelity_metrics(
    frame: pd.DataFrame,
    models: dict[str, IsotonicRegression],
    ranges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    rows = []

    for limb in ("rising", "falling"):
        sub = frame.loc[frame["limb"].eq(limb)].copy()
        lo, hi = ranges[limb]
        mask = sub["gage_height_ft"].between(lo, hi)
        sub = sub.loc[mask]
        obs = sub["discharge_cfs"].to_numpy(float)
        pred = models[limb].predict(sub["gage_height_ft"].to_numpy(float))
        residual = pred - obs

        rows.append(
            {
                "limb": limb,
                "observation_count": int(len(sub)),
                "stage_min_ft": float(sub["gage_height_ft"].min()),
                "stage_max_ft": float(sub["gage_height_ft"].max()),
                "discharge_min_cfs": float(sub["discharge_cfs"].min()),
                "discharge_max_cfs": float(sub["discharge_cfs"].max()),
                "mae_cfs": mae(obs, pred),
                "rmse_cfs": rmse(obs, pred),
                "mape_percent": mape(obs, pred),
                "bias_cfs": float(np.mean(residual)),
                "r2": r2(obs, pred),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if args.stage_step_ft <= 0:
        raise ValueError("--stage-step-ft must be > 0.")
    if args.discharge_step_cfs <= 0:
        raise ValueError("--discharge-step-cfs must be > 0.")

    prepare_output_dir(args.output_dir, args.overwrite)

    frame = pd.read_csv(args.event_observations)
    required = {
        "datetime_utc",
        "discharge_cfs",
        "gage_height_ft",
        "00060_qualifiers",
        "00065_qualifiers",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], utc=True, errors="coerce")
    frame["discharge_cfs"] = pd.to_numeric(frame["discharge_cfs"], errors="coerce")
    frame["gage_height_ft"] = pd.to_numeric(frame["gage_height_ft"], errors="coerce")

    valid = (
        frame["datetime_utc"].notna()
        & frame["discharge_cfs"].notna()
        & frame["gage_height_ft"].notna()
        & (frame["discharge_cfs"] >= 0)
        & frame["00060_qualifiers"].map(qualifier_has_approved)
        & frame["00065_qualifiers"].map(qualifier_has_approved)
    )
    work = frame.loc[valid].copy().sort_values("datetime_utc").reset_index(drop=True)
    if len(work) < 50:
        raise RuntimeError("Too few valid paired event observations for reconstruction.")

    work["discharge_is_estimated"] = work["00060_qualifiers"].map(
        qualifier_has_estimated
    )

    peak_idx = work["gage_height_ft"].idxmax()
    peak_time = work.loc[peak_idx, "datetime_utc"]
    peak_stage_ft = float(work.loc[peak_idx, "gage_height_ft"])
    peak_discharge_cfs = float(work.loc[peak_idx, "discharge_cfs"])

    work["limb"] = np.where(
        work["datetime_utc"] <= peak_time,
        "rising",
        "falling",
    )

    rising_grouped = aggregate_limb(work, "rising")
    falling_grouped = aggregate_limb(work, "falling")

    rising_model = fit_isotonic(rising_grouped)
    falling_model = fit_isotonic(falling_grouped)

    models = {"rising": rising_model, "falling": falling_model}
    ranges = {
        "rising": (
            float(rising_grouped["gage_height_ft"].min()),
            float(rising_grouped["gage_height_ft"].max()),
        ),
        "falling": (
            float(falling_grouped["gage_height_ft"].min()),
            float(falling_grouped["gage_height_ft"].max()),
        ),
    }

    empirical = pd.concat([rising_grouped, falling_grouped], ignore_index=True)
    hq = dense_h_to_q(
        rising_grouped,
        falling_grouped,
        rising_model,
        falling_model,
        args.stage_step_ft,
    )
    qh = dense_q_to_h(hq, args.discharge_step_cfs)
    metrics = fidelity_metrics(work, models, ranges)

    common = hq.dropna(
        subset=["rising_discharge_cfs", "falling_discharge_cfs"]
    ).copy()
    common["falling_minus_rising_cfs"] = (
        common["falling_discharge_cfs"] - common["rising_discharge_cfs"]
    )
    common["absolute_branch_difference_cfs"] = common[
        "falling_minus_rising_cfs"
    ].abs()
    denom = (
        common["falling_discharge_cfs"] + common["rising_discharge_cfs"]
    ) / 2.0
    common["absolute_branch_difference_percent_of_mean"] = np.where(
        denom > 0,
        common["absolute_branch_difference_cfs"] / denom * 100.0,
        np.nan,
    )

    qc_rows = []
    for limb in ("rising", "falling"):
        n = int((work["limb"] == limb).sum())
        unique_stage = int(
            work.loc[work["limb"].eq(limb), "gage_height_ft"].round(2).nunique()
        )
        qc_rows.append(
            {
                "severity": "NOTE" if n >= 100 else "WARNING",
                "issue": f"{limb.upper()}_OBSERVATION_COUNT",
                "detail": f"{n} observations across {unique_stage} unique 0.01-ft stages.",
            }
        )

    estimated_count = int(work["discharge_is_estimated"].sum())
    if estimated_count:
        qc_rows.append(
            {
                "severity": "SCIENTIFIC_NOTE",
                "issue": "ESTIMATED_CONTINUOUS_DISCHARGE_PRESENT",
                "detail": (
                    f"{estimated_count} approved continuous discharge records carry "
                    "the USGS estimated-value qualifier 'e'. They are retained because "
                    "they are part of the published event record."
                ),
            }
        )

    if not common.empty:
        median_branch_pct = float(
            common["absolute_branch_difference_percent_of_mean"].median()
        )
        max_branch_pct = float(
            common["absolute_branch_difference_percent_of_mean"].max()
        )
        qc_rows.append(
            {
                "severity": "SCIENTIFIC_NOTE",
                "issue": "RISING_FALLING_BRANCH_SEPARATION",
                "detail": (
                    f"Across the common stage range, median absolute branch "
                    f"difference is {median_branch_pct:.2f}% of branch mean; "
                    f"maximum is {max_branch_pct:.2f}%. Separate limbs are retained."
                ),
            }
        )
    else:
        median_branch_pct = None
        max_branch_pct = None
        qc_rows.append(
            {
                "severity": "BLOCKING",
                "issue": "NO_COMMON_STAGE_RANGE",
                "detail": "Rising and falling surrogate curves have no overlapping stage range.",
            }
        )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_NOTE",
            "issue": "SURROGATE_NOT_OFFICIAL_USGS_RATING",
            "detail": (
                "This product reconstructs effective H-Q behavior from the published "
                "Florence event record. It is not the unavailable official historical "
                "USGS base rating plus event-period shift."
            ),
        }
    )

    qc = pd.DataFrame(qc_rows)
    blocking = int((qc["severity"] == "BLOCKING").sum())

    empirical_path = args.output_dir / "florence_effective_hq_empirical_points.csv"
    hq_path = args.output_dir / "florence_effective_h_to_q_surrogate.csv"
    qh_path = args.output_dir / "florence_effective_q_to_h_surrogate.csv"
    hysteresis_path = args.output_dir / "florence_hq_hysteresis_comparison.csv"
    metrics_path = args.output_dir / "florence_hq_reconstruction_fidelity.csv"
    qc_path = args.output_dir / "florence_effective_hq_qc.csv"
    metadata_path = args.output_dir / "florence_effective_hq_metadata.json"

    atomic_write_csv(empirical, empirical_path)
    atomic_write_csv(hq, hq_path)
    atomic_write_csv(qh, qh_path)
    atomic_write_csv(common, hysteresis_path)
    atomic_write_csv(metrics, metrics_path)
    atomic_write_csv(qc, qc_path)

    metadata = {
        "status": (
            "PASS_FLORENCE_EFFECTIVE_HQ_SURROGATE_RECONSTRUCTED"
            if blocking == 0
            else "FAIL_FLORENCE_EFFECTIVE_HQ_SURROGATE"
        ),
        "step": "STEP_1E",
        "created_utc": utc_now(),
        "source_event_observations": str(args.event_observations),
        "source_event_metadata": (
            str(args.event_metadata) if args.event_metadata else None
        ),
        "scientific_name": "Florence 2018 effective H-Q surrogate",
        "official_usgs_historical_rating": False,
        "reason": (
            "Official 2018 USGS historical rating/shift was not publicly recovered. "
            "The surrogate is reconstructed from the approved continuous event record."
        ),
        "method": {
            "branching": "Split at maximum observed gage height time.",
            "rising_limb_definition": "datetime <= peak gage-height datetime",
            "falling_limb_definition": "datetime > peak gage-height datetime",
            "stage_aggregation": "0.01-ft stage groups; median discharge",
            "regression": "monotonic isotonic regression, weighted by stage-group count",
            "estimated_approved_discharge_retained": True,
        },
        "valid_paired_event_observation_count": int(len(work)),
        "estimated_discharge_observation_count": estimated_count,
        "event_start_utc": work["datetime_utc"].min().isoformat(),
        "event_end_utc": work["datetime_utc"].max().isoformat(),
        "peak": {
            "datetime_utc": peak_time.isoformat(),
            "gage_height_ft": peak_stage_ft,
            "gage_height_m": peak_stage_ft * FT_TO_M,
            "discharge_cfs_at_peak_stage_time": peak_discharge_cfs,
            "discharge_cms_at_peak_stage_time": peak_discharge_cfs * CFS_TO_CMS,
        },
        "limb_ranges": {
            limb: {
                "stage_min_ft": ranges[limb][0],
                "stage_max_ft": ranges[limb][1],
                "observation_count": int((work["limb"] == limb).sum()),
            }
            for limb in ("rising", "falling")
        },
        "hysteresis_summary": {
            "common_stage_row_count": int(len(common)),
            "median_absolute_branch_difference_percent_of_mean": median_branch_pct,
            "maximum_absolute_branch_difference_percent_of_mean": max_branch_pct,
        },
        "blocking_qc_issue_count": blocking,
        "output_paths": {
            "empirical_points": str(empirical_path),
            "h_to_q_surrogate": str(hq_path),
            "q_to_h_surrogate": str(qh_path),
            "hysteresis_comparison": str(hysteresis_path),
            "reconstruction_fidelity": str(metrics_path),
            "qc": str(qc_path),
            "metadata": str(metadata_path),
        },
    }
    atomic_write_json(metadata, metadata_path)

    print("=" * 100)
    print("STEP 1E - FLORENCE 2018 EFFECTIVE H-Q SURROGATE")
    print("=" * 100)
    print(f"Valid paired event observations    : {len(work):,}")
    print(f"Estimated-Q observations retained  : {estimated_count:,}")
    print(f"Peak stage time                    : {peak_time.isoformat()}")
    print(f"Peak gage height                   : {peak_stage_ft:.3f} ft")
    print(f"Q at peak-stage time               : {peak_discharge_cfs:,.1f} ft3/s")
    print(f"Rising observations                : {(work['limb'] == 'rising').sum():,}")
    print(f"Falling observations               : {(work['limb'] == 'falling').sum():,}")
    if median_branch_pct is not None:
        print(f"Median branch separation           : {median_branch_pct:.2f} %")
        print(f"Maximum branch separation          : {max_branch_pct:.2f} %")
    print()
    print(f"H->Q surrogate                     : {hq_path}")
    print(f"Q->H surrogate                     : {qh_path}")
    print(f"Hysteresis comparison              : {hysteresis_path}")
    print(f"Fidelity metrics                   : {metrics_path}")
    print(f"QC                                 : {qc_path}")
    print(f"Metadata                           : {metadata_path}")
    print()
    print(f"Status                             : {metadata['status']}")

    if blocking:
        raise RuntimeError(f"Step 1E produced {blocking} blocking QC issue(s).")


if __name__ == "__main__":
    main()
