#!/usr/bin/env python3
"""Build the recovered historical P3.6B3 4-calibration/3-validation event library.

This branch is intentionally isolated from the later 8/6 timing-aware library.
The block windows are recovered from the project's historical P3.6B3 audit.
No event discovery or retuning is performed here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BUILD = "P3_6B3_RECOVERED_EVENT_LIBRARY_V1"

# Exact historical blocks used by the passing P3.6B3 run.
BLOCKS = [
    ("CAL_BLOCK_01", "CALIBRATION_2015_2016", "2015-11-16T17:00:00Z", "2015-12-03T17:00:00Z", 409, 314),
    ("CAL_BLOCK_02", "CALIBRATION_2015_2016", "2015-12-29T22:00:00Z", "2016-02-02T10:00:00Z", 829, 757),
    ("CAL_BLOCK_03", "CALIBRATION_2015_2016", "2016-02-03T07:00:00Z", "2016-03-09T10:00:00Z", 844, 654),
    ("CAL_BLOCK_04", "CALIBRATION_2015_2016", "2016-05-05T20:00:00Z", "2016-05-17T03:00:00Z", 272, 177),
    ("VAL_BLOCK_01", "TEMPORAL_VALIDATION_2017", "2017-01-01T00:00:00Z", "2017-01-16T06:00:00Z", 367, 295),
    ("VAL_BLOCK_02", "TEMPORAL_VALIDATION_2017", "2017-05-20T20:00:00Z", "2017-06-06T20:00:00Z", 409, 337),
    ("VAL_BLOCK_03", "TEMPORAL_VALIDATION_2017", "2017-06-23T07:00:00Z", "2017-07-10T07:00:00Z", 409, 337),
]


def parse_args() -> argparse.Namespace:
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


def time_col(df: pd.DataFrame) -> str:
    for c in ["interval_end_utc", "timestamp", "time_utc", "datetime_utc"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"No timestamp column in {list(df.columns)}")


def q_col(df: pd.DataFrame) -> str:
    for c in ["q_obs_m3s", "q_obs_mean_m3s", "q_mean_m3s", "discharge_m3s", "q_m3s"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"No observed-Q column in {list(df.columns)}")


def rain_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if str(c).startswith("rain_SC") and str(c).endswith("_mm")
    ]


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    a = parse_args()
    for p in [a.spatial_rainfall, a.usgs_q_hourly, a.routing_features]:
        if not p.exists():
            raise FileNotFoundError(p)

    rain = pd.read_csv(a.spatial_rainfall)
    q = pd.read_csv(a.usgs_q_hourly)
    routing = pd.read_csv(a.routing_features)

    rt = time_col(rain)
    qt = time_col(q)
    qc = q_col(q)
    rain[rt] = pd.to_datetime(rain[rt], utc=True, errors="raise")
    q[qt] = pd.to_datetime(q[qt], utc=True, errors="raise")

    rcols = rain_cols(rain)
    if len(rcols) != a.expected_subcatchments:
        raise RuntimeError(
            f"Expected {a.expected_subcatchments} rain_SC###_mm columns; found {len(rcols)}"
        )

    sid_candidates = [
        c for c in routing.columns
        if str(c).lower() in {"subcatchment_id", "sc_id", "id"}
    ]
    if sid_candidates:
        n_sc = routing[sid_candidates[0]].astype(str).nunique()
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
    forcing["interval_end_utc"] = pd.to_datetime(
        forcing["interval_end_utc"], utc=True, errors="raise"
    )
    forcing["q_obs_m3s"] = pd.to_numeric(forcing["q_obs_m3s"], errors="coerce")
    forcing = forcing[
        (forcing["interval_end_utc"] >= pd.Timestamp("2015-05-10T00:00:00Z"))
        & (forcing["interval_end_utc"] < pd.Timestamp("2018-01-01T00:00:00Z"))
    ].copy()

    block_rows: list[dict] = []
    target_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for block_id, phase, start_text, end_text, expected_hours, expected_targets in BLOCKS:
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        times = pd.date_range(start, end, freq="h", tz="UTC")
        if len(times) != expected_hours:
            raise RuntimeError(
                f"Internal block definition error for {block_id}: {len(times)} != {expected_hours}"
            )

        win = forcing[forcing["interval_end_utc"].isin(times)].copy()
        if len(win) != expected_hours:
            missing = times.difference(pd.DatetimeIndex(win["interval_end_utc"]))
            raise RuntimeError(
                f"{block_id}: forcing has {len(win)}/{expected_hours} hours; "
                f"missing first={list(missing[:5])}"
            )

        rain_complete = float(win[rcols].notna().mean().mean() * 100.0)
        if rain_complete + 1e-10 < a.min_rainfall_completeness_percent:
            raise RuntimeError(
                f"{block_id}: rainfall completeness {rain_complete:.6f}% below "
                f"{a.min_rainfall_completeness_percent:.6f}%"
            )

        score_start = start + pd.Timedelta(hours=a.objective_warmup_hours)
        target = win[
            (win["interval_end_utc"] >= score_start)
            & np.isfinite(win["q_obs_m3s"])
        ][["interval_end_utc", "q_obs_m3s"]].copy()

        if len(target) != expected_targets:
            raise RuntimeError(
                f"{block_id}: recovered target count mismatch: "
                f"{len(target)} != historical {expected_targets}. "
                "Do not train P3.6B3 until the Q archive/event mask is reconciled."
            )

        target.insert(0, "event_block_id", block_id)
        target.insert(1, "phase", phase)
        target_rows.append(target)

        q_availability = float(win["q_obs_m3s"].notna().mean() * 100.0)
        qmax = float(np.nanmax(win["q_obs_m3s"].to_numpy(float)))
        block_rows.append(
            {
                "event_block_id": block_id,
                "phase": phase,
                "block_start_utc": start.isoformat(),
                "block_end_utc": end.isoformat(),
                "block_hours": expected_hours,
                "historical_target_count": expected_targets,
            }
        )
        summary_rows.append(
            {
                "event_block_id": block_id,
                "phase": phase,
                "block_start_utc": start.isoformat(),
                "block_end_utc": end.isoformat(),
                "block_hours": expected_hours,
                "target_hours": len(target),
                "rainfall_completeness_percent": rain_complete,
                "q_availability_percent": q_availability,
                "max_q_obs_m3s": qmax,
            }
        )

    blocks = pd.DataFrame(block_rows)
    targets = pd.concat(target_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    n_cal = int(blocks["phase"].str.startswith("CALIBRATION").sum())
    n_val = int(blocks["phase"].str.startswith("TEMPORAL_VALIDATION").sum())
    if (n_cal, n_val) != (4, 3):
        raise RuntimeError(f"Historical P3.6B3 inventory mismatch: {n_cal}/{n_val}")

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "forcing": out / "development_forcing_hourly.csv",
        "blocks": out / "event_blocks.csv",
        "mask": out / "q_target_mask.csv",
        "summary": out / "event_summary.csv",
        "metadata": out / "p3_6b3_event_library_metadata.json",
    }
    if not a.overwrite:
        for p in paths.values():
            if p.exists():
                raise FileExistsError(p)

    atomic_csv(forcing, paths["forcing"])
    atomic_csv(blocks, paths["blocks"])
    atomic_csv(targets, paths["mask"])
    atomic_csv(summary, paths["summary"])

    metadata = {
        "status": "PASS_P3_6B3_EVENT_LIBRARY_READY",
        "script_build": BUILD,
        "selection_method": "exact_historical_block_windows_recovered_from_project_audit",
        "calibration_blocks": n_cal,
        "validation_blocks": n_val,
        "objective_warmup_hours": a.objective_warmup_hours,
        "florence_used": False,
        "historical_target_counts_enforced": True,
        "scientific_note": (
            "This branch reproduces the P3.6B3 event-window contract. It is kept "
            "separate from the later 8/6 timing-aware event library."
        ),
    }
    tmp = paths["metadata"].with_name(paths["metadata"].name + ".partial")
    tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(tmp, paths["metadata"])

    print("=" * 100)
    print("RECOVERED P3.6B3 EVENT LIBRARY")
    print("=" * 100)
    print(f"Calibration blocks                 : {n_cal}")
    print(f"Temporal-validation blocks         : {n_val}")
    print("Florence 2018 used                 : NO")
    print()
    for row in summary_rows:
        label = "CAL" if row["phase"].startswith("CALIBRATION") else "VAL"
        print(
            f"{label} {row['event_block_id']} | hours={row['block_hours']} | "
            f"targets={row['target_hours']} | rain={row['rainfall_completeness_percent']:.3f}% | "
            f"Qmax={row['max_q_obs_m3s']:.3f} m3/s"
        )
    print()
    print("Status                             : PASS_P3_6B3_EVENT_LIBRARY_READY")
    print(f"Forcing                            : {paths['forcing']}")
    print(f"Event blocks                       : {paths['blocks']}")
    print(f"Q target mask                      : {paths['mask']}")


if __name__ == "__main__":
    main()
