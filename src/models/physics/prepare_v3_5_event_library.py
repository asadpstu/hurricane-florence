#!/usr/bin/env python3
"""Build the retained Physics V3.5 5-calibration/3-validation event library.

The block inventory is recovered from the historical V3.5 training audit.
No event discovery or Florence-2018 data is used here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BUILD = "PHYSICS_V3_5_RETAINED_EVENT_LIBRARY_V1"

BLOCKS = [
    ("CAL_BLOCK_01", "CALIBRATION_2015_2016", "2015-11-16T17:00:00Z", "2015-12-03T17:00:00Z", 409, 314),
    ("CAL_BLOCK_02", "CALIBRATION_2015_2016", "2015-12-29T22:00:00Z", "2016-02-02T10:00:00Z", 829, 757),
    ("CAL_BLOCK_03", "CALIBRATION_2015_2016", "2016-02-03T07:00:00Z", "2016-03-09T10:00:00Z", 844, 654),
    ("CAL_BLOCK_04", "CALIBRATION_2015_2016", "2016-05-05T20:00:00Z", "2016-05-17T03:00:00Z", 272, 177),
    ("CAL_BLOCK_05", "CALIBRATION_2015_2016", "2016-10-05T08:00:00Z", "2016-10-22T08:00:00Z", 409, 337),
    ("VAL_BLOCK_01", "TEMPORAL_VALIDATION_2017", "2017-01-01T00:00:00Z", "2017-01-16T06:00:00Z", 367, 295),
    ("VAL_BLOCK_02", "TEMPORAL_VALIDATION_2017", "2017-05-20T20:00:00Z", "2017-06-06T20:00:00Z", 409, 337),
    ("VAL_BLOCK_03", "TEMPORAL_VALIDATION_2017", "2017-06-23T07:00:00Z", "2017-07-10T07:00:00Z", 409, 337),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spatial-rainfall", type=Path, required=True)
    p.add_argument("--usgs-q-hourly", type=Path, required=True)
    p.add_argument("--routing-features", type=Path, required=True)
    p.add_argument("--expected-subcatchments", type=int, default=37)
    p.add_argument("--objective-warmup-hours", type=int, default=72)
    p.add_argument("--min-rainfall-completeness-percent", type=float, default=100.0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _time_col(df):
    for c in ["interval_end_utc", "timestamp", "time_utc", "datetime_utc"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"No timestamp column in {list(df.columns)}")


def _q_col(df):
    for c in ["q_obs_m3s", "q_obs_mean_m3s", "q_mean_m3s", "discharge_cms", "discharge_m3s", "q_m3s"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"No observed-Q column in {list(df.columns)}")


def _rain_cols(df):
    return [c for c in df.columns if str(c).startswith("rain_SC") and str(c).endswith("_mm")]


def _atomic_csv(df, path):
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    a = parse_args()
    for p in [a.spatial_rainfall, a.usgs_q_hourly, a.routing_features]:
        if not p.exists():
            raise FileNotFoundError(p)

    rain = pd.read_csv(a.spatial_rainfall)
    q = pd.read_csv(a.usgs_q_hourly)
    routing = pd.read_csv(a.routing_features)
    rt, qt, qc = _time_col(rain), _time_col(q), _q_col(q)
    rain[rt] = pd.to_datetime(rain[rt], utc=True, errors="raise")
    q[qt] = pd.to_datetime(q[qt], utc=True, errors="raise")
    rcols = _rain_cols(rain)
    if len(rcols) != a.expected_subcatchments:
        raise RuntimeError(f"Expected {a.expected_subcatchments} rain_SC###_mm columns; found {len(rcols)}")

    sid = next((c for c in routing.columns if str(c).lower() in {"subcatchment_id", "sc_id", "id"}), None)
    if sid is not None and routing[sid].astype(str).nunique() != a.expected_subcatchments:
        raise RuntimeError("Routing-feature subcatchment count mismatch")

    forcing = rain.rename(columns={rt: "interval_end_utc"}).merge(
        q[[qt, qc]].rename(columns={qt: "interval_end_utc", qc: "q_obs_m3s"}),
        on="interval_end_utc", how="left",
    ).sort_values("interval_end_utc").reset_index(drop=True)
    forcing["interval_end_utc"] = pd.to_datetime(forcing["interval_end_utc"], utc=True)
    forcing["q_obs_m3s"] = pd.to_numeric(forcing["q_obs_m3s"], errors="coerce")
    forcing = forcing[(forcing["interval_end_utc"] >= pd.Timestamp("2015-05-10T00:00:00Z")) &
                      (forcing["interval_end_utc"] < pd.Timestamp("2018-01-01T00:00:00Z"))].copy()

    block_rows, target_rows, summary_rows = [], [], []
    for block_id, phase, s, e, expected_hours, expected_targets in BLOCKS:
        start, end = pd.Timestamp(s), pd.Timestamp(e)
        times = pd.date_range(start, end, freq="h", tz="UTC")
        if len(times) != expected_hours:
            raise RuntimeError(f"Internal block definition error {block_id}: {len(times)} != {expected_hours}")
        win = forcing[forcing["interval_end_utc"].isin(times)].copy()
        if len(win) != expected_hours:
            miss = times.difference(pd.DatetimeIndex(win["interval_end_utc"]))
            raise RuntimeError(f"{block_id}: forcing {len(win)}/{expected_hours}; first missing={list(miss[:5])}")
        rain_complete = float(win[rcols].notna().mean().mean() * 100.0)
        if rain_complete + 1e-10 < a.min_rainfall_completeness_percent:
            raise RuntimeError(f"{block_id}: rainfall completeness {rain_complete:.6f}% below {a.min_rainfall_completeness_percent:.6f}%")
        score_start = start + pd.Timedelta(hours=a.objective_warmup_hours)
        target = win[(win["interval_end_utc"] >= score_start) & np.isfinite(win["q_obs_m3s"])][["interval_end_utc", "q_obs_m3s"]].copy()
        if len(target) != expected_targets:
            raise RuntimeError(f"{block_id}: target count {len(target)} != retained historical {expected_targets}")
        target.insert(0, "event_block_id", block_id)
        target.insert(1, "phase", phase)
        target_rows.append(target)
        block_rows.append({
            "event_block_id": block_id, "phase": phase,
            "block_start_utc": start.isoformat(), "block_end_utc": end.isoformat(),
            "block_hours": expected_hours, "historical_target_count": expected_targets,
        })
        summary_rows.append({
            "event_block_id": block_id, "phase": phase,
            "block_start_utc": start.isoformat(), "block_end_utc": end.isoformat(),
            "block_hours": expected_hours, "target_hours": len(target),
            "rainfall_completeness_percent": rain_complete,
            "q_availability_percent": float(win["q_obs_m3s"].notna().mean() * 100.0),
            "max_q_obs_m3s": float(np.nanmax(win["q_obs_m3s"].to_numpy(float))),
        })

    blocks = pd.DataFrame(block_rows)
    targets = pd.concat(target_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    ncal = int(blocks.phase.str.startswith("CALIBRATION").sum())
    nval = int(blocks.phase.str.startswith("TEMPORAL_VALIDATION").sum())
    if (ncal, nval) != (5, 3):
        raise RuntimeError(f"V3.5 retained inventory mismatch: {ncal}/{nval}")

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "forcing": out / "development_forcing_hourly.csv",
        "blocks": out / "event_blocks.csv",
        "mask": out / "q_target_mask.csv",
        "summary": out / "event_summary.csv",
        "metadata": out / "v3_5_event_library_metadata.json",
    }
    if not a.overwrite:
        existing = next((p for p in paths.values() if p.exists()), None)
        if existing is not None:
            raise FileExistsError(existing)
    _atomic_csv(forcing, paths["forcing"])
    _atomic_csv(blocks, paths["blocks"])
    _atomic_csv(targets, paths["mask"])
    _atomic_csv(summary, paths["summary"])
    paths["metadata"].write_text(json.dumps({
        "build": BUILD, "calibration_blocks": ncal, "validation_blocks": nval,
        "florence_2018_used": False, "objective_warmup_hours": a.objective_warmup_hours,
        "blocks": block_rows,
    }, indent=2), encoding="utf-8")

    print("=" * 100)
    print("RETAINED PHYSICS V3.5 EVENT LIBRARY")
    print("=" * 100)
    print(f"Calibration blocks                 : {ncal}")
    print(f"Temporal-validation blocks         : {nval}")
    print("Florence 2018 used                 : NO")
    for r in summary_rows:
        tag = "CAL" if r["phase"].startswith("CALIBRATION") else "VAL"
        print(f"{tag} {r['event_block_id']} | hours={r['block_hours']} | targets={r['target_hours']} | rain={r['rainfall_completeness_percent']:.3f}% | Qmax={r['max_q_obs_m3s']:.3f} m3/s")
    print("Status                             : PASS_PHYSICS_V3_5_EVENT_LIBRARY_READY")


if __name__ == "__main__":
    main()
