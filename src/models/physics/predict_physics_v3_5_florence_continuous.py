#!/usr/bin/env python3
"""
Physics V3.5 Florence predictor — continuous antecedent-state fix.

Why this script exists
----------------------
The old Florence V2 predictor used rainfall from a file named "...warmup", but
actually simulated only the evaluation interval (2018-09-10 -> 2018-09-26).
It therefore reset soil/base/channel stores exactly at 2018-09-10 instead of
letting those stores evolve through the available antecedent rainfall period.

That reset is especially problematic for Florence because the frozen V2/V3
models derive non-trivial initial groundwater/channel storage from ERA5 deep
wetness.  The old run started around 100 m3/s while observed flow was only
~14 m3/s on Sep 10.

This script fixes the runner, not the calibration:
* Physics V3.5 frozen parameters are used.
* No optimizer.
* No parameter fitting.
* Continuous simulation from --warmup-start to --evaluation-end.
* Only the evaluation interval is written/scored.
* Observed Q is never used to initialize or modify states.
* No performance pass/fail gates.  Metrics are descriptive only.

IMPORTANT
---------
This script cannot guarantee 80-90% accuracy.  It removes an event-runner
state-reset problem and gives the retained V3.5 model a fair Florence test.
If peak Q is still too low after this corrected run, the remaining problem is
the hydrologic structure/parameters, not the Florence prediction wrapper.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BUILD = "PHYSICS_V3_5_FLORENCE_CONTINUOUS_ANTECEDENT_V2_GAP_TOLERANT"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--rainfall-wide", type=Path, required=True)
    p.add_argument("--hydrologic-forcing", type=Path, required=True)
    p.add_argument("--dynamic-state-et0", type=Path, required=True)
    p.add_argument("--deep-state", type=Path, required=True)
    p.add_argument("--routing-features", type=Path, required=True)
    p.add_argument("--routing", type=Path, required=True)
    p.add_argument("--frozen-parameters", type=Path, required=True)

    p.add_argument(
        "--warmup-start",
        default="2018-06-01T00:00:00Z",
    )
    p.add_argument(
        "--evaluation-start",
        default="2018-09-10T00:00:00Z",
    )
    p.add_argument(
        "--evaluation-end",
        default="2018-09-26T00:00:00Z",
    )

    p.add_argument("--expected-subcatchments", type=int, default=37)
    p.add_argument(
        "--max-rain-gap-hours",
        type=int,
        default=6,
        help=(
            "Maximum consecutive hourly MRMS gap repaired by local time "
            "interpolation. Default=6. No observed Q is used."
        ),
    )
    p.add_argument(
        "--max-et0-gap-hours",
        type=int,
        default=6,
        help=(
            "Maximum consecutive hourly ET0 gap repaired by local time "
            "interpolation. Default=6."
        ),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def atomic_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    tmp.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def parse_utc(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def normalize_sc_id(value):
    s = str(value).strip()
    if s.upper().startswith("SC"):
        try:
            return f"SC{int(s[2:]):03d}"
        except Exception:
            return s
    try:
        return f"SC{int(float(s)):03d}"
    except Exception:
        return s


def load_local_module(filename, module_name):
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required project module not found: {path}"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_col(df, candidates, label):
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise RuntimeError(
        f"Could not resolve {label}; columns={list(df.columns)}"
    )


def load_time_frame(path, time_candidates):
    df = pd.read_csv(path)
    tc = find_col(df, time_candidates, "timestamp")
    df["__time"] = pd.to_datetime(
        df[tc],
        utc=True,
        errors="raise",
    )
    if df["__time"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate timestamps.")
    return df.sort_values("__time").reset_index(drop=True)


def load_sc_hourly(path, expected, required_cols):
    df = pd.read_csv(path)

    tc = find_col(
        df,
        ["interval_end_utc", "time_utc", "timestamp"],
        "state timestamp",
    )
    sc = find_col(
        df,
        ["subcatchment_id", "modeling_subcatchment_id", "sc_id"],
        "subcatchment id",
    )

    df["__time"] = pd.to_datetime(
        df[tc],
        utc=True,
        errors="raise",
    )
    df["__sc"] = df[sc].map(normalize_sc_id)

    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"{path}: missing {col}.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["__sc"].nunique() != expected:
        raise RuntimeError(
            f"{path}: expected {expected} subcatchments; "
            f"found {df['__sc'].nunique()}."
        )

    if df.duplicated(["__time", "__sc"]).any():
        raise RuntimeError(
            f"{path}: duplicate time/subcatchment rows."
        )

    return df


def matrix_for_times(df, times, ids, value_col):
    pivot = (
        df[df["__time"].isin(times)]
        .pivot(
            index="__time",
            columns="__sc",
            values=value_col,
        )
        .reindex(index=times, columns=ids)
    )

    arr = pivot.to_numpy(dtype=float)

    if not np.isfinite(arr).all():
        missing = int((~np.isfinite(arr)).sum())
        raise RuntimeError(
            f"{value_col}: {missing} missing/non-finite required values."
        )

    return arr



def repair_short_matrix_gaps(matrix, times, label, max_gap_hours, nonnegative=False):
    """Repair short hourly gaps independently in each subcatchment series."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise RuntimeError(f"{label}: expected a 2-D matrix, got {arr.shape}.")
    if len(times) != arr.shape[0]:
        raise RuntimeError(
            f"{label}: time length {len(times)} != matrix rows {arr.shape[0]}."
        )

    missing_before = ~np.isfinite(arr)
    n_before = int(missing_before.sum())
    if n_before == 0:
        return arr, {
            "missing_cells_before": 0,
            "repaired_cells": 0,
            "remaining_cells": 0,
            "affected_hours": 0,
        }

    if max_gap_hours < 1:
        raise RuntimeError(
            f"{label}: contains {n_before} missing cells and gap repair is disabled."
        )

    frame = pd.DataFrame(arr, index=pd.DatetimeIndex(times))
    repaired = frame.interpolate(
        method="time",
        axis=0,
        limit=int(max_gap_hours),
        limit_direction="both",
    )

    out = repaired.to_numpy(dtype=float).copy()
    if nonnegative:
        finite = np.isfinite(out)
        out[finite] = np.maximum(out[finite], 0.0)

    missing_after = ~np.isfinite(out)
    n_after = int(missing_after.sum())
    repaired_count = n_before - n_after
    affected_hours = int(missing_before.any(axis=1).sum())

    if n_after:
        bad_rows = np.flatnonzero(missing_after.any(axis=1))
        first_bad = [str(pd.Timestamp(times[i])) for i in bad_rows[:10]]
        raise RuntimeError(
            f"{label}: repaired {repaired_count}/{n_before} missing cells, but "
            f"{n_after} remain after max_gap_hours={max_gap_hours}. "
            f"First remaining hours={first_bad}."
        )

    return out, {
        "missing_cells_before": n_before,
        "repaired_cells": repaired_count,
        "remaining_cells": n_after,
        "affected_hours": affected_hours,
    }


def matrix_for_times_allow_gaps(df, times, ids, value_col):
    pivot = (
        df[df["__time"].isin(times)]
        .pivot(index="__time", columns="__sc", values=value_col)
        .reindex(index=times, columns=ids)
    )
    return pivot.to_numpy(dtype=float)

def metric_bundle(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[valid]
    sim = sim[valid]

    if len(obs) < 2:
        return {}

    err = sim - obs
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    nse = (
        1.0 - float(np.sum(err ** 2)) / denom
        if denom > 0
        else np.nan
    )

    mo = float(np.mean(obs))
    ms = float(np.mean(sim))
    so = float(np.std(obs, ddof=1))
    ss = float(np.std(sim, ddof=1))

    r = (
        float(np.corrcoef(obs, sim)[0, 1])
        if so > 0 and ss > 0
        else np.nan
    )

    alpha = ss / so if so > 0 else np.nan
    beta = ms / mo if mo != 0 else np.nan

    kge = (
        1.0 - np.sqrt(
            (r - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
        if np.isfinite([r, alpha, beta]).all()
        else np.nan
    )

    pbias = (
        100.0 * float(np.sum(sim - obs)) / float(np.sum(obs))
        if float(np.sum(obs)) != 0
        else np.nan
    )

    return {
        "nse": float(nse),
        "kge": float(kge),
        "r": float(r),
        "alpha": float(alpha),
        "beta": float(beta),
        "pbias_percent": float(pbias),
        "rmse_m3s": float(np.sqrt(np.mean(err ** 2))),
        "mae_m3s": float(np.mean(np.abs(err))),
    }


def main():
    a = parse_args()

    warmup_start = parse_utc(a.warmup_start)
    eval_start = parse_utc(a.evaluation_start)
    eval_end = parse_utc(a.evaluation_end)

    if not (warmup_start < eval_start < eval_end):
        raise ValueError(
            "Require warmup_start < evaluation_start < evaluation_end."
        )

    # Interval-end convention:
    # simulation represents (T-1h, T], beginning one hour after warmup_start.
    sim_times = pd.date_range(
        start=warmup_start + pd.Timedelta(hours=1),
        end=eval_end,
        freq="h",
        tz="UTC",
    )

    eval_times = pd.date_range(
        start=eval_start + pd.Timedelta(hours=1),
        end=eval_end,
        freq="h",
        tz="UTC",
    )

    # ------------------------------------------------------------------
    # Build frozen Physics V3.5 runtime without using the development-only
    # deep-state loader (which intentionally rejects 2018+ data).
    # ------------------------------------------------------------------
    runtime = load_local_module(
        "physics_v3_5_moisture_interflow_runtime.py",
        "physics_v3_5_runtime_for_florence",
    )

    ns, patch_counts = runtime.load_v3_5_namespace()

    loaded = ns["_load_parent"]()
    if isinstance(loaded, tuple):
        parent = loaded[0]
        parent_build = loaded[2] if len(loaded) >= 3 else str(
            getattr(parent, "BUILD", "")
        )
    else:
        parent = loaded
        parent_build = str(getattr(parent, "BUILD", ""))

    original_names = list(parent.PARAM_NAMES)
    original_bounds = list(parent.PARAM_BOUNDS)

    extended_names, extended_unpack = ns["_make_unpack_params"](
        original_names
    )

    parent.PARAM_NAMES = extended_names
    parent.PARAM_BOUNDS = (
        original_bounds + list(ns["NEW_PARAM_BOUNDS"])
    )
    parent.unpack_params = extended_unpack
    simulate_event = ns["_make_simulate_event"](parent)

    # Routing.
    features = pd.read_csv(a.routing_features)
    routing = pd.read_csv(a.routing)
    routing_model = parent.build_routing(
        features,
        routing,
        a.expected_subcatchments,
    )
    ids = [
        normalize_sc_id(x)
        for x in routing_model["ids"]
    ]

    if len(ids) != a.expected_subcatchments:
        raise RuntimeError(
            f"Routing model contains {len(ids)} subcatchments."
        )

    # Frozen V3.5 parameters.
    payload = json.loads(
        a.frozen_parameters.read_text(encoding="utf-8")
    )
    params = payload.get("parameters", payload)

    missing_params = [
        name for name in extended_names
        if name not in params
    ]
    if missing_params:
        raise RuntimeError(
            f"Frozen V3.5 parameter file missing: {missing_params}"
        )

    x = np.asarray(
        [float(params[name]) for name in extended_names],
        dtype=float,
    )

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Frozen V3.5 parameters contain non-finite values."
        )

    # ------------------------------------------------------------------
    # Spatial MRMS rainfall for the FULL warmup + evaluation simulation.
    # ------------------------------------------------------------------
    rain = load_time_frame(
        a.rainfall_wide,
        ["interval_end_utc", "time_utc", "timestamp"],
    )

    rain_map = parent.rainfall_columns(rain)

    missing_rain_ids = [
        sid for sid in ids
        if sid not in rain_map
    ]
    if missing_rain_ids:
        raise RuntimeError(
            f"Spatial rainfall missing subcatchments: {missing_rain_ids}"
        )

    rain_idx = rain.set_index("__time").reindex(sim_times)

    rain_matrix_raw = np.column_stack(
        [
            pd.to_numeric(
                rain_idx[rain_map[sid]],
                errors="coerce",
            ).to_numpy(float)
            for sid in ids
        ]
    )

    rain_matrix, rain_gap_info = repair_short_matrix_gaps(
        rain_matrix_raw,
        sim_times,
        "MRMS rainfall",
        a.max_rain_gap_hours,
        nonnegative=True,
    )

    # ------------------------------------------------------------------
    # ERA5 ET0 over full simulation. Root/deep state ONLY initializes at
    # warmup_start. The model then evolves continuously; no Sep-10 reset.
    # ------------------------------------------------------------------
    dyn = load_sc_hourly(
        a.dynamic_state_et0,
        a.expected_subcatchments,
        ["root_zone_relative_wetness", "et0_mm_h"],
    )

    deep = load_sc_hourly(
        a.deep_state,
        a.expected_subcatchments,
        ["deep_relative_wetness"],
    )

    et0_raw = matrix_for_times_allow_gaps(
        dyn,
        sim_times,
        ids,
        "et0_mm_h",
    )

    et0_matrix, et0_gap_info = repair_short_matrix_gaps(
        et0_raw,
        sim_times,
        "ERA5 ET0",
        a.max_et0_gap_hours,
        nonnegative=True,
    )

    # Only row zero is used by the conceptual model for root initialization,
    # but preserve matrix dimensions.
    endpoint_root = matrix_for_times(
        dyn,
        sim_times,
        ids,
        "root_zone_relative_wetness",
    )

    start_index = pd.DatetimeIndex([warmup_start])

    root_start = matrix_for_times(
        dyn,
        start_index,
        ids,
        "root_zone_relative_wetness",
    )[0]

    deep_start = matrix_for_times(
        deep,
        start_index,
        ids,
        "deep_relative_wetness",
    )[0]

    model_wetness = endpoint_root.copy()
    model_wetness[0, :] = root_start

    event = {
        "event_block_id": "FLORENCE_2018_V3_5_CONTINUOUS",
        "phase": "FLORENCE_2018_DIAGNOSTIC",
        "start": warmup_start,
        "end": eval_end,
        "times": sim_times,
        "rain": rain_matrix,
        "qobs": np.full(len(sim_times), np.nan),
        "target": np.zeros(len(sim_times), dtype=bool),
        "era5_wetness": model_wetness,
        "et0": et0_matrix,
        "deep_wetness_initial": deep_start,
    }

    print("=" * 108)
    print(
        "PHYSICS V3.5 FLORENCE — CONTINUOUS ANTECEDENT RUN"
    )
    print("=" * 108)
    print(f"Build                              : {BUILD}")
    print(f"Parent                             : {parent_build}")
    print(f"V3.5 patch counts                  : {patch_counts}")
    print(f"Warmup start                       : {warmup_start}")
    print(f"Evaluation start                   : {eval_start}")
    print(f"Evaluation end                     : {eval_end}")
    print(f"Continuous simulation hours        : {len(sim_times)}")
    print(f"Evaluation hours                   : {len(eval_times)}")
    print("State reset at evaluation start    : NO")
    print("Optimizer                          : NO")
    print("Parameter fitting                  : NO")
    print("Observed Q used by simulation      : NO")
    print(
        f"MRMS gap repair                    : "
        f"{rain_gap_info['repaired_cells']} cells across "
        f"{rain_gap_info['affected_hours']} hours "
        f"(max gap={a.max_rain_gap_hours} h)"
    )
    print(
        f"ET0 gap repair                     : "
        f"{et0_gap_info['repaired_cells']} cells across "
        f"{et0_gap_info['affected_hours']} hours "
        f"(max gap={a.max_et0_gap_hours} h)"
    )
    print()

    result = simulate_event(
        event,
        x,
        routing_model,
        return_states=True,
    )

    qsim_all = np.asarray(
        result["q_sim_m3s"],
        dtype=float,
    )

    if len(qsim_all) != len(sim_times):
        raise RuntimeError(
            f"Simulation length={len(qsim_all)}; "
            f"expected={len(sim_times)}."
        )

    if not np.isfinite(qsim_all).all():
        raise RuntimeError(
            "Simulation contains non-finite discharge."
        )

    qsim_series = pd.Series(
        np.maximum(qsim_all, 0.0),
        index=sim_times,
    )

    qsim = qsim_series.reindex(eval_times).to_numpy(float)

    # ------------------------------------------------------------------
    # Observed Q is loaded only after the model has run.
    # ------------------------------------------------------------------
    hydro = load_time_frame(
        a.hydrologic_forcing,
        ["interval_end_utc", "time_utc", "timestamp"],
    )

    q_col = find_col(
        hydro,
        [
            "q_obs_mean_m3s",
            "q_obs_m3s",
            "observed_q_mean_m3s",
        ],
        "observed discharge",
    )

    hydro[q_col] = pd.to_numeric(
        hydro[q_col],
        errors="coerce",
    )

    hydro_idx = hydro.set_index("__time")

    qobs = hydro_idx.reindex(eval_times)[q_col].to_numpy(float)

    metrics = metric_bundle(qobs, qsim)

    common = np.isfinite(qobs) & np.isfinite(qsim)

    if common.any():
        obs_valid = qobs[common]
        sim_valid = qsim[common]
        common_times = eval_times[common]

        obs_peak_i = int(np.nanargmax(obs_valid))
        sim_peak_i = int(np.nanargmax(sim_valid))

        observed_peak_m3s = float(obs_valid[obs_peak_i])
        predicted_peak_m3s = float(sim_valid[sim_peak_i])
        observed_peak_time = common_times[obs_peak_i]
        predicted_peak_time = common_times[sim_peak_i]

        peak_error_percent = (
            100.0
            * (predicted_peak_m3s - observed_peak_m3s)
            / observed_peak_m3s
        )
        peak_accuracy_percent = (
            100.0
            * min(predicted_peak_m3s, observed_peak_m3s)
            / max(predicted_peak_m3s, observed_peak_m3s)
        )
    else:
        observed_peak_m3s = np.nan
        predicted_peak_m3s = np.nan
        observed_peak_time = pd.NaT
        predicted_peak_time = pd.NaT
        peak_error_percent = np.nan
        peak_accuracy_percent = np.nan

    predictions = pd.DataFrame(
        {
            "interval_end_utc": eval_times,
            "q_obs_m3s": qobs,
            "q_sim_m3s": qsim,
            "error_m3s": qsim - qobs,
        }
    )

    predictions["error_percent"] = np.where(
        np.isfinite(qobs) & (np.abs(qobs) > 1.0e-9),
        100.0 * (qsim - qobs) / qobs,
        np.nan,
    )

    predictions["absolute_percent_error"] = np.abs(
        predictions["error_percent"]
    )

    predictions["date"] = predictions[
        "interval_end_utc"
    ].dt.date

    daily = (
        predictions.groupby("date", as_index=False)
        .agg(
            observed_daily_max_m3s=("q_obs_m3s", "max"),
            predicted_daily_max_m3s=("q_sim_m3s", "max"),
            observed_daily_mean_m3s=("q_obs_m3s", "mean"),
            predicted_daily_mean_m3s=("q_sim_m3s", "mean"),
        )
    )

    daily["daily_max_error_m3s"] = (
        daily["predicted_daily_max_m3s"]
        - daily["observed_daily_max_m3s"]
    )

    daily["daily_max_error_percent"] = np.where(
        daily["observed_daily_max_m3s"].abs() > 1.0e-9,
        100.0
        * daily["daily_max_error_m3s"]
        / daily["observed_daily_max_m3s"],
        np.nan,
    )

    daily["daily_max_accuracy_percent"] = np.where(
        (
            daily["observed_daily_max_m3s"] > 0
        )
        & (
            daily["predicted_daily_max_m3s"] > 0
        ),
        100.0
        * np.minimum(
            daily["observed_daily_max_m3s"],
            daily["predicted_daily_max_m3s"],
        )
        / np.maximum(
            daily["observed_daily_max_m3s"],
            daily["predicted_daily_max_m3s"],
        ),
        np.nan,
    )

    a.output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = (
        a.output_dir
        / "physics_v3_5_florence_2018_predictions.csv"
    )
    daily_path = (
        a.output_dir
        / "physics_v3_5_florence_2018_daily.csv"
    )
    metric_path = (
        a.output_dir
        / "physics_v3_5_florence_2018_metrics.json"
    )

    for path in [pred_path, daily_path, metric_path]:
        if path.exists() and not a.overwrite:
            raise FileExistsError(
                f"{path} exists. Use --overwrite."
            )

    atomic_csv(predictions, pred_path)
    atomic_csv(daily, daily_path)

    metadata = {
        "build": BUILD,
        "model": "Physics V3.5 moisture-dependent interflow",
        "frozen_parameter_source": str(a.frozen_parameters),
        "warmup_start_utc": warmup_start,
        "evaluation_start_utc": eval_start,
        "evaluation_end_utc": eval_end,
        "continuous_state_evolution": True,
        "state_reset_at_evaluation_start": False,
        "optimizer_called": False,
        "parameter_fitting_performed": False,
        "observed_q_used_by_simulation": False,
        "rainfall_gap_repair": rain_gap_info,
        "et0_gap_repair": et0_gap_info,
        "max_rain_gap_hours": int(a.max_rain_gap_hours),
        "max_et0_gap_hours": int(a.max_et0_gap_hours),
        "gap_repair_uses_observed_q": False,
        "metrics": metrics,
        "observed_peak_m3s": observed_peak_m3s,
        "predicted_peak_m3s": predicted_peak_m3s,
        "observed_peak_time_utc": observed_peak_time,
        "predicted_peak_time_utc": predicted_peak_time,
        "peak_error_percent": peak_error_percent,
        "peak_accuracy_percent": peak_accuracy_percent,
        "outputs": {
            "predictions": str(pred_path),
            "daily": str(daily_path),
        },
    }

    atomic_json(metadata, metric_path)

    print("FLORENCE V3.5 RESULTS")
    print("-" * 108)

    if metrics:
        print(f"NSE                                : {metrics['nse']:.6f}")
        print(f"KGE                                : {metrics['kge']:.6f}")
        print(f"Correlation                        : {metrics['r']:.6f}")
        print(f"Variability alpha                  : {metrics['alpha']:.6f}")
        print(f"Mean ratio beta                    : {metrics['beta']:.6f}")
        print(f"PBIAS                              : {metrics['pbias_percent']:+.3f}%")
        print(f"RMSE                               : {metrics['rmse_m3s']:.3f} m3/s")
        print(f"MAE                                : {metrics['mae_m3s']:.3f} m3/s")

    print(
        f"Observed peak                      : "
        f"{observed_peak_m3s:.3f} m3/s @ {observed_peak_time}"
    )
    print(
        f"Predicted peak                     : "
        f"{predicted_peak_m3s:.3f} m3/s @ {predicted_peak_time}"
    )
    print(
        f"Peak error                         : "
        f"{peak_error_percent:+.3f}%"
    )
    print(
        f"Peak magnitude accuracy            : "
        f"{peak_accuracy_percent:.3f}%"
    )
    print()
    print("DAY-WISE DISCHARGE")
    print("-" * 108)
    print(daily.round(2).to_string(index=False))
    print()
    print(f"Predictions                        : {pred_path}")
    print(f"Daily summary                      : {daily_path}")
    print(f"Metrics                            : {metric_path}")


if __name__ == "__main__":
    main()
