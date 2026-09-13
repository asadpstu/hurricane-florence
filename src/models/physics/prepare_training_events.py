#!/usr/bin/env python3
"""Build the retained September-4 Physics/ML training-event library.

The final retained event inventory is fixed to the event anchors preserved in
project QA/audit output.  This avoids silently changing the scientific
calibration/holdout sample when the surrounding historical archive is rebuilt.

Two event-critical MRMS hours may be repaired as zero rainfall, but only after
an explicit surrounding-dryness check.  This is *not* temporal interpolation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BUILD = "RETAINED_TRAINING_EVENT_LIBRARY_2026_09_04"

# Exact retained event peaks recovered from the final September-4 audit.
RETAINED_EVENTS = [
    ("CAL_01", "CALIBRATION_2015_2016", "2015-11-14T19:00:00Z"),
    ("CAL_02", "CALIBRATION_2015_2016", "2015-12-06T20:00:00Z"),
    ("CAL_03", "CALIBRATION_2015_2016", "2015-12-29T00:00:00Z"),
    ("CAL_04", "CALIBRATION_2015_2016", "2016-02-10T02:00:00Z"),
    ("CAL_05", "CALIBRATION_2015_2016", "2016-02-28T09:00:00Z"),
    ("CAL_06", "CALIBRATION_2015_2016", "2016-05-09T05:00:00Z"),
    ("CAL_07", "CALIBRATION_2015_2016", "2016-10-12T08:00:00Z"),
    ("CAL_08", "CALIBRATION_2015_2016", "2016-11-01T16:00:00Z"),
    ("VAL_01", "TEMPORAL_VALIDATION_2017", "2017-01-06T06:00:00Z"),
    ("VAL_02", "TEMPORAL_VALIDATION_2017", "2017-04-02T21:00:00Z"),
    ("VAL_03", "TEMPORAL_VALIDATION_2017", "2017-04-30T13:00:00Z"),
    ("VAL_04", "TEMPORAL_VALIDATION_2017", "2017-06-30T07:00:00Z"),
    ("VAL_05", "TEMPORAL_VALIDATION_2017", "2017-09-04T12:00:00Z"),
    ("VAL_06", "TEMPORAL_VALIDATION_2017", "2017-12-11T17:00:00Z"),
]

# Explicitly documented event-critical dry-hour repairs from project history.
# A repair is applied only when the target rainfall is missing and the observed
# surrounding rainfall satisfies the stated dry threshold.
DOCUMENTED_DRY_REPAIRS = [
    {
        "timestamp": "2015-11-14T16:00:00Z",
        "evidence_start": "2015-11-14T12:00:00Z",
        "evidence_end": "2015-11-14T20:00:00Z",
        "threshold_mm": 0.001,
        "minimum_evidence_hours": 8,
        "provenance": "verified project-history dry-hour repair",
    },
    {
        "timestamp": "2016-10-11T16:00:00Z",
        "evidence_start": "2016-10-11T10:00:00Z",
        "evidence_end": "2016-10-11T22:00:00Z",
        "threshold_mm": 0.05,
        "minimum_evidence_hours": 12,
        "provenance": "clean-runbook documented dry hour before 2016-10-12 event",
    },
    {
        "timestamp": "2017-04-27T12:00:00Z",
        "evidence_start": "2017-04-27T04:00:00Z",
        "evidence_end": "2017-04-27T20:00:00Z",
        "threshold_mm": 0.001,
        "minimum_evidence_hours": 14,
        "provenance": "reference-chat documented consecutive dry validation gap",
    },
    {
        "timestamp": "2017-04-27T13:00:00Z",
        "evidence_start": "2017-04-27T04:00:00Z",
        "evidence_end": "2017-04-27T20:00:00Z",
        "threshold_mm": 0.001,
        "minimum_evidence_hours": 14,
        "provenance": "reference-chat documented consecutive dry validation gap",
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spatial-rainfall", type=Path, required=True)
    p.add_argument("--usgs-q-hourly", type=Path, required=True)
    p.add_argument("--routing-features", type=Path, required=True)
    p.add_argument("--expected-subcatchments", type=int, default=37)
    p.add_argument("--event-pre-hours", type=int, default=168)
    p.add_argument("--event-post-hours", type=int, default=240)
    p.add_argument("--objective-warmup-hours", type=int, default=72)
    p.add_argument("--event-separation-hours", type=int, default=168)
    p.add_argument("--minimum-pre-hours", type=int, default=72)
    p.add_argument("--minimum-post-hours", type=int, default=120)
    p.add_argument("--minimum-scored-pre-peak-hours", type=int, default=24)
    p.add_argument("--minimum-event-peak-quantile", type=float, default=0.75)
    p.add_argument("--top-calibration-events", type=int, default=12)
    p.add_argument("--top-validation-events", type=int, default=6)
    p.add_argument("--minimum-calibration-events", type=int, default=5)
    p.add_argument("--minimum-validation-events", type=int, default=3)
    p.add_argument("--min-event-rainfall-completeness-percent", type=float, default=100)
    p.add_argument("--min-q-availability-percent", type=float, default=70)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _time_col(df: pd.DataFrame) -> str:
    for c in ["interval_end_utc", "timestamp", "time_utc", "datetime_utc"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"No timestamp column in {list(df.columns)}")


def _q_col(df: pd.DataFrame) -> str:
    for c in ["q_obs_m3s", "q_obs_mean_m3s", "q_mean_m3s", "discharge_m3s", "q_m3s"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"No observed-Q column in {list(df.columns)}")


def _rain_cols(df: pd.DataFrame) -> list[str]:
    cols = [
        c for c in df.columns
        if str(c).startswith("rain_SC") and str(c).endswith("_mm")
    ]
    if cols:
        return cols
    return [
        c for c in df.columns
        if "rain" in str(c).lower()
        and "sc" in str(c).lower()
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _apply_documented_repairs(
    forcing: pd.DataFrame,
    rain_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = forcing.copy()
    audit_rows: list[dict] = []

    for rule in DOCUMENTED_DRY_REPAIRS:
        target = pd.Timestamp(rule["timestamp"])
        ev_start = pd.Timestamp(rule["evidence_start"])
        ev_end = pd.Timestamp(rule["evidence_end"])
        threshold = float(rule["threshold_mm"])

        hit = x.index[x["interval_end_utc"] == target]
        if len(hit) != 1:
            raise RuntimeError(
                f"Documented repair target {target} must occur exactly once; found {len(hit)}"
            )
        idx = hit[0]
        before = x.loc[idx, rain_cols]

        # If a rebuilt upstream archive already contains this hour, preserve it.
        if before.notna().all():
            audit_rows.append({
                "interval_end_utc": target.isoformat(),
                "action": "ALREADY_PRESENT_NO_CHANGE",
                "repair_value_mm": np.nan,
                "dry_threshold_mm": threshold,
                "evidence_hours": 0,
                "max_evidence_rainfall_mm": np.nan,
                "provenance": rule["provenance"],
            })
            continue

        # Never partially overwrite a target hour: the retained repair is a
        # complete 37-subcatchment zero hour.
        if before.notna().any():
            raise RuntimeError(
                f"{target}: documented dry repair refused because rainfall is only partially missing"
            )

        evidence = x[
            (x["interval_end_utc"] >= ev_start)
            & (x["interval_end_utc"] <= ev_end)
            & (x["interval_end_utc"] != target)
        ][rain_cols]
        usable = evidence.dropna(how="all")
        if len(usable) < int(rule["minimum_evidence_hours"]):
            raise RuntimeError(
                f"{target}: only {len(usable)} surrounding evidence hours; "
                f"need {rule['minimum_evidence_hours']}"
            )
        max_ev = float(np.nanmax(usable.to_numpy(dtype=float)))
        if not np.isfinite(max_ev) or max_ev > threshold:
            raise RuntimeError(
                f"{target}: documented dry repair refused; surrounding rainfall "
                f"max={max_ev:.9f} mm exceeds {threshold} mm"
            )

        x.loc[idx, rain_cols] = 0.0
        audit_rows.append({
            "interval_end_utc": target.isoformat(),
            "action": "DRY_HOUR_INFERRED_ZERO",
            "repair_value_mm": 0.0,
            "dry_threshold_mm": threshold,
            "evidence_hours": int(len(usable)),
            "max_evidence_rainfall_mm": max_ev,
            "provenance": rule["provenance"],
        })

    return x, pd.DataFrame(audit_rows)


def main():
    a = parse_args()
    for p in [a.spatial_rainfall, a.usgs_q_hourly, a.routing_features]:
        if not p.exists():
            raise FileNotFoundError(p)

    rain = pd.read_csv(a.spatial_rainfall)
    q = pd.read_csv(a.usgs_q_hourly)
    routing = pd.read_csv(a.routing_features)

    rt = _time_col(rain)
    qt = _time_col(q)
    qc = _q_col(q)
    rain[rt] = pd.to_datetime(rain[rt], utc=True)
    q[qt] = pd.to_datetime(q[qt], utc=True)

    rain_cols = _rain_cols(rain)
    if len(rain_cols) != a.expected_subcatchments:
        raise RuntimeError(
            f"Expected {a.expected_subcatchments} subcatchment rainfall columns; "
            f"found {len(rain_cols)}"
        )

    # Fail fast if the routing feature table is for a different discretization.
    sc_candidates = [c for c in routing.columns if c.lower() in {"subcatchment_id", "sc_id", "id"}]
    if sc_candidates:
        n_sc = routing[sc_candidates[0]].astype(str).nunique()
        if n_sc != a.expected_subcatchments:
            raise RuntimeError(
                f"Routing features contain {n_sc} subcatchments; expected {a.expected_subcatchments}"
            )

    forcing = (
        rain.rename(columns={rt: "interval_end_utc"})
        .merge(
            q[[qt, qc]].rename(columns={qt: "interval_end_utc", qc: "q_obs_m3s"}),
            on="interval_end_utc",
            how="left",
        )
        .sort_values("interval_end_utc")
        .reset_index(drop=True)
    )
    forcing["interval_end_utc"] = pd.to_datetime(forcing["interval_end_utc"], utc=True)
    development_start = pd.Timestamp("2015-05-10T00:00:00Z")
    calibration_end = pd.Timestamp("2017-01-01T00:00:00Z")
    development_end = pd.Timestamp("2018-01-01T00:00:00Z")
    forcing = forcing[
        (forcing["interval_end_utc"] >= development_start)
        & (forcing["interval_end_utc"] < development_end)
    ].copy()

    before_repair = forcing.copy()
    forcing, repair_audit = _apply_documented_repairs(forcing, rain_cols)

    blocks: list[dict] = []
    masks: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for eid, phase, peak_text in RETAINED_EVENTS:
        peak = pd.Timestamp(peak_text)
        phase_start = development_start if phase.startswith("CALIBRATION") else calibration_end
        phase_end_exclusive = calibration_end if phase.startswith("CALIBRATION") else development_end
        phase_end = phase_end_exclusive - pd.Timedelta(hours=1)

        start = max(peak - pd.Timedelta(hours=a.event_pre_hours), phase_start)
        stop = min(peak + pd.Timedelta(hours=a.event_post_hours), phase_end)
        pre_h = float((peak - start) / pd.Timedelta(hours=1))
        post_h = float((stop - peak) / pd.Timedelta(hours=1))
        if pre_h < a.minimum_pre_hours or post_h < a.minimum_post_hours:
            raise RuntimeError(
                f"{eid}: retained event no longer satisfies pre/post-hour requirements"
            )

        win = forcing[
            (forcing["interval_end_utc"] >= start)
            & (forcing["interval_end_utc"] <= stop)
        ].copy()
        expected_hours = int((stop - start) / pd.Timedelta(hours=1)) + 1
        if len(win) != expected_hours:
            raise RuntimeError(
                f"{eid}: event window has {len(win)} rows; expected {expected_hours} hourly rows"
            )

        peak_row = win[win["interval_end_utc"] == peak]
        if len(peak_row) != 1 or pd.isna(peak_row.iloc[0]["q_obs_m3s"]):
            raise RuntimeError(f"{eid}: observed Q missing at retained peak {peak}")
        q_peak = float(peak_row.iloc[0]["q_obs_m3s"])

        rain_comp = float(win[rain_cols].notna().mean().mean() * 100.0)
        q_av = float(win["q_obs_m3s"].notna().mean() * 100.0)
        if rain_comp + 1e-10 < a.min_event_rainfall_completeness_percent:
            raise RuntimeError(
                f"{eid}: rainfall completeness {rain_comp:.6f}% below "
                f"{a.min_event_rainfall_completeness_percent:.6f}%"
            )
        if q_av + 1e-10 < a.min_q_availability_percent:
            raise RuntimeError(
                f"{eid}: Q availability {q_av:.6f}% below {a.min_q_availability_percent:.6f}%"
            )

        score_start = start + pd.Timedelta(hours=a.objective_warmup_hours)
        target = (win["interval_end_utc"] >= score_start) & win["q_obs_m3s"].notna()
        scored_prepeak = int((target & (win["interval_end_utc"] <= peak)).sum())
        if scored_prepeak < a.minimum_scored_pre_peak_hours:
            raise RuntimeError(
                f"{eid}: only {scored_prepeak} scored pre-peak hours; "
                f"minimum={a.minimum_scored_pre_peak_hours}"
            )

        blocks.append({
            "event_block_id": eid,
            "phase": phase,
            "block_start_utc": start.isoformat(),
            "block_end_utc": stop.isoformat(),
            "peak_time_utc": peak.isoformat(),
            "peak_q_obs_m3s": q_peak,
            "rainfall_completeness_percent": rain_comp,
            "q_availability_percent": q_av,
        })
        mask = win[["interval_end_utc"]].copy()
        mask.insert(0, "event_block_id", eid)
        mask["use_for_objective"] = target.to_numpy(dtype=bool)
        masks.append(mask)
        summaries.append({
            "event_block_id": eid,
            "phase": phase,
            "peak_time_utc": peak.isoformat(),
            "peak_q_obs_m3s": q_peak,
            "block_hours": expected_hours,
            "scored_hours": int(target.sum()),
            "rainfall_completeness_percent": rain_comp,
            "q_availability_percent": q_av,
        })

    block_df = pd.DataFrame(blocks)
    n_cal = int(block_df["phase"].str.startswith("CALIBRATION").sum())
    n_val = int(block_df["phase"].str.startswith("TEMPORAL_VALIDATION").sum())
    if n_cal != 8 or n_val != 6:
        raise RuntimeError(f"Retained inventory mismatch: calibration={n_cal}, validation={n_val}")
    if n_cal < a.minimum_calibration_events or n_val < a.minimum_validation_events:
        raise RuntimeError("Retained inventory does not satisfy requested minimum event counts")

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "forcing_before": out / "development_forcing_hourly_before_documented_dry_fill.csv",
        "forcing": out / "development_forcing_hourly.csv",
        "repairs": out / "documented_dry_rainfall_gap_fills.csv",
        "blocks": out / "event_blocks.csv",
        "mask": out / "q_target_mask.csv",
        "summary": out / "event_summary.csv",
        "metadata": out / "training_event_metadata.json",
    }
    if not a.overwrite:
        existing = [p for p in paths.values() if p.exists()]
        if existing:
            raise FileExistsError(existing[0])

    _atomic_csv(before_repair, paths["forcing_before"])
    _atomic_csv(forcing, paths["forcing"])
    _atomic_csv(repair_audit, paths["repairs"])
    _atomic_csv(block_df, paths["blocks"])
    _atomic_csv(pd.concat(masks, ignore_index=True), paths["mask"])
    _atomic_csv(pd.DataFrame(summaries), paths["summary"])

    metadata = {
        "status": "PASS_TRAINING_EVENT_LIBRARY_READY",
        "script_build": BUILD,
        "selection_method": "verified_final_2026_09_04_event_anchors",
        "calibration": "2015-2016",
        "temporal_validation": "2017",
        "florence_used": False,
        "calibration_events": n_cal,
        "validation_events": n_val,
        "event_pre_hours": a.event_pre_hours,
        "event_post_hours": a.event_post_hours,
        "objective_warmup_hours": a.objective_warmup_hours,
        "documented_dry_repairs": repair_audit.to_dict("records"),
        "scientific_note": (
            "Event anchors are the final retained September-4 inventory recovered from "
            "the project audit. Dry-hour zeros are applied only after explicit dry-context checks."
        ),
    }
    tmp = paths["metadata"].with_name(paths["metadata"].name + ".partial")
    tmp.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, paths["metadata"])

    print("=" * 100)
    print("RETAINED PHYSICS TRAINING EVENT LIBRARY")
    print("=" * 100)
    print(f"Calibration blocks                 : {n_cal}")
    print(f"Temporal-validation blocks         : {n_val}")
    print("Florence 2018 used                 : NO")
    print(f"Documented repair records          : {len(repair_audit)}")
    print()
    for row in summaries:
        print(
            f"{row['event_block_id']} | {row['phase']} | "
            f"peak={row['peak_time_utc']} | Q={row['peak_q_obs_m3s']:.3f} m3/s | "
            f"hours={row['block_hours']} | targets={row['scored_hours']}"
        )
    print()
    print("Status                             : PASS_TRAINING_EVENT_LIBRARY_READY")


if __name__ == "__main__":
    main()
