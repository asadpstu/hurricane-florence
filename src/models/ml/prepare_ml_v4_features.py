#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

BUILD = "ML_V4_EXOGENOUS_HIGHFLOW_FEATURE_DATASET"

SHORT_WINDOWS = (3, 6, 12, 24, 72)
LONG_WINDOWS = (168, 336, 720, 1440, 2160)  # 7,14,30,60,90 days
ROUTING_VELOCITIES = (0.25, 0.50, 1.00)
INTENSITY_THRESHOLDS = (1.0, 2.0, 4.0, 6.0)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--spatial-rainfall", type=Path, required=True)
    p.add_argument("--forcing", type=Path, required=True)
    p.add_argument("--routing-features", type=Path, required=True)
    p.add_argument("--train-start", default="2015-05-10T00:00:00Z")
    p.add_argument("--train-end", default="2017-01-01T00:00:00Z")
    p.add_argument("--validation-end", default="2018-01-01T00:00:00Z")
    p.add_argument("--expected-subcatchments", type=int, default=37)
    p.add_argument("--min-routed-area-coverage-percent", type=float, default=99.0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_csv(df, path, compression=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def sc_id(value):
    m = re.search(r"SC\s*0*(\d+)", str(value), flags=re.I)
    if m:
        return f"SC{int(m.group(1)):03d}"
    try:
        return f"SC{int(float(value)):03d}"
    except Exception:
        return str(value)


def resolve(df, names, label):
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    raise RuntimeError(f"Could not resolve {label}. Available={list(df.columns)}")


def rainfall_columns(df):
    result = {}
    for col in df.columns:
        low = str(col).lower()
        if any(x in low for x in ("coverage", "valid", "missing", "flag", "source", "count", "status")):
            continue
        m = re.search(r"SC\s*0*(\d+)", str(col), flags=re.I)
        if not m:
            continue
        sid = f"SC{int(m.group(1)):03d}"
        score = (2 if "rain" in low else 0) + (1 if "mm" in low else 0)
        if sid not in result or score > result[sid][0]:
            result[sid] = (score, col)
    return {k: v[1] for k, v in result.items()}


def wmean(matrix, weights):
    finite = np.isfinite(matrix)
    denom = (finite * weights[None, :]).sum(axis=1)
    num = np.nansum(matrix * weights[None, :], axis=1)
    return np.divide(num, denom, out=np.full(len(matrix), np.nan), where=denom > 0)


def wstd(matrix, weights):
    mean = wmean(matrix, weights)
    finite = np.isfinite(matrix)
    denom = (finite * weights[None, :]).sum(axis=1)
    num = np.nansum((matrix - mean[:, None]) ** 2 * weights[None, :], axis=1)
    var = np.divide(num, denom, out=np.full(len(matrix), np.nan), where=denom > 0)
    return np.sqrt(var)


def wfraction(matrix, weights, threshold):
    finite = np.isfinite(matrix)
    denom = (finite * weights[None, :]).sum(axis=1)
    num = ((finite & (matrix > threshold)) * weights[None, :]).sum(axis=1)
    return np.divide(num, denom, out=np.full(len(matrix), np.nan), where=denom > 0)


def wp90_row(row, weights):
    good = np.isfinite(row)
    if not good.any():
        return np.nan
    v, w = row[good], weights[good]
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    i = int(np.searchsorted(c, 0.90 * c[-1], side="left"))
    return float(v[min(i, len(v) - 1)])


def rolling_sum(series, hours, fraction=1.0):
    return series.rolling(
        hours,
        min_periods=int(math.ceil(hours * fraction)),
    ).sum()


def routed_index(matrix, weights, lags, minimum_coverage):
    n, m = matrix.shape
    shifted = np.full((n, m), np.nan)
    for j, lag in enumerate(lags):
        lag = int(lag)
        if lag == 0:
            shifted[:, j] = matrix[:, j]
        elif 0 < lag < n:
            shifted[lag:, j] = matrix[:-lag, j]
    finite = np.isfinite(shifted)
    area = (finite * weights[None, :]).sum(axis=1)
    coverage = area / weights.sum()
    value = np.divide(
        np.nansum(shifted * weights[None, :], axis=1),
        area,
        out=np.full(n, np.nan),
        where=area > 0,
    )
    value[coverage < minimum_coverage] = np.nan
    return value, coverage


def main():
    a = args()
    train_start = pd.Timestamp(a.train_start)
    train_end = pd.Timestamp(a.train_end)
    validation_end = pd.Timestamp(a.validation_end)

    rain = pd.read_csv(a.spatial_rainfall)
    forcing = pd.read_csv(a.forcing)
    routing = pd.read_csv(a.routing_features)

    rt = resolve(rain, ["interval_end_utc", "timestamp", "time", "valid_time"], "rainfall time")
    ft = resolve(forcing, ["interval_end_utc", "timestamp", "time"], "forcing time")
    qcol = resolve(
        forcing,
        ["q_obs_m3s", "q_obs_mean_m3s", "q_mean_m3s", "discharge_m3s"],
        "observed Q",
    )
    sidcol = resolve(routing, ["subcatchment_id", "sc_id", "subcatchment"], "subcatchment id")
    areacol = resolve(routing, ["local_area_km2", "area_km2", "subcatchment_area_km2"], "area")
    distcol = resolve(
        routing,
        ["network_distance_proxy_to_gauge_km", "cumulative_distance_to_gauge_km", "network_distance_to_gauge_km"],
        "network distance",
    )

    rain["interval_end_utc"] = pd.to_datetime(rain[rt], utc=True, errors="raise")
    forcing["interval_end_utc"] = pd.to_datetime(forcing[ft], utc=True, errors="raise")
    routing["subcatchment_id"] = routing[sidcol].map(sc_id)
    routing["area_km2"] = pd.to_numeric(routing[areacol], errors="raise")
    routing["distance_km"] = pd.to_numeric(routing[distcol], errors="raise")

    if routing["subcatchment_id"].duplicated().any():
        raise RuntimeError("Duplicate subcatchment IDs in routing features.")

    rmap = rainfall_columns(rain)
    ordered_ids = [x for x in routing["subcatchment_id"] if x in rmap]
    if len(ordered_ids) != a.expected_subcatchments:
        raise RuntimeError(
            f"Expected {a.expected_subcatchments} common subcatchments; found {len(ordered_ids)}."
        )

    routing = routing.set_index("subcatchment_id").loc[ordered_ids]
    cols = [rmap[x] for x in ordered_ids]
    rain = rain[["interval_end_utc"] + cols].sort_values("interval_end_utc")
    if rain["interval_end_utc"].duplicated().any():
        raise RuntimeError("Duplicate spatial-rainfall timestamps.")

    index = pd.date_range(train_start, validation_end, freq="h", inclusive="left")
    rain = rain.set_index("interval_end_utc").reindex(index)
    matrix = rain[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    weights = routing["area_km2"].to_numpy(float)
    distance = routing["distance_km"].to_numpy(float)

    if (weights <= 0).any() or not np.isfinite(weights).all():
        raise RuntimeError("Invalid subcatchment areas.")

    f = pd.DataFrame(index=index)
    f.index.name = "interval_end_utc"
    f["rain_basin_1h_mm"] = wmean(matrix, weights)
    f["rain_spatial_std_1h_mm"] = wstd(matrix, weights)

    with np.errstate(all="ignore"):
        f["rain_spatial_max_1h_mm"] = np.array([
            np.nanmax(row) if np.isfinite(row).any() else np.nan
            for row in matrix
        ])

    f["rain_spatial_p90_1h_mm"] = [wp90_row(row, weights) for row in matrix]

    for threshold in INTENSITY_THRESHOLDS:
        tag = str(threshold).replace(".", "p")
        f[f"rain_area_fraction_gt_{tag}mmh"] = wfraction(matrix, weights, threshold)

    for h in SHORT_WINDOWS:
        f[f"rain_basin_sum_{h}h_mm"] = rolling_sum(f["rain_basin_1h_mm"], h)

    for h in LONG_WINDOWS:
        f[f"rain_basin_sum_{h}h_mm"] = rolling_sum(f["rain_basin_1h_mm"], h, 0.99)

    # Fixed routing zones using the A4 0.5 m/s diagnostic travel-time prior.
    travel05 = distance * 1000.0 / 0.50 / 3600.0
    zones = {
        "near": travel05 <= 48.0,
        "middle": (travel05 > 48.0) & (travel05 <= 96.0),
        "far": travel05 > 96.0,
    }
    zone_meta = {}

    for name, mask in zones.items():
        if not mask.any():
            raise RuntimeError(f"Routing zone {name} has zero subcatchments.")
        f[f"rain_{name}_1h_mm"] = wmean(matrix[:, mask], weights[mask])
        f[f"rain_{name}_sum_6h_mm"] = rolling_sum(f[f"rain_{name}_1h_mm"], 6)
        f[f"rain_{name}_sum_24h_mm"] = rolling_sum(f[f"rain_{name}_1h_mm"], 24)
        zone_meta[name] = {
            "subcatchments": int(mask.sum()),
            "area_km2": float(weights[mask].sum()),
            "travel_time_0p5mps_min_h": float(travel05[mask].min()),
            "travel_time_0p5mps_max_h": float(travel05[mask].max()),
        }

    coverage_cols = []
    for velocity in ROUTING_VELOCITIES:
        tag = str(velocity).replace(".", "p")
        lags = np.rint(distance * 1000.0 / velocity / 3600.0).astype(int)
        routed, coverage = routed_index(
            matrix,
            weights,
            lags,
            a.min_routed_area_coverage_percent / 100.0,
        )
        col = f"rain_routed_v{tag}_1h_mm"
        cov = f"rain_routed_v{tag}_area_coverage"
        f[col] = routed
        f[cov] = coverage
        coverage_cols.append(cov)
        f[f"rain_routed_v{tag}_sum_6h_mm"] = rolling_sum(f[col], 6)
        f[f"rain_routed_v{tag}_sum_24h_mm"] = rolling_sum(f[col], 24)

    doy = f.index.dayofyear.to_numpy(float)
    hour = f.index.hour.to_numpy(float)
    f["season_sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    f["season_cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    f["season_sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    f["season_cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

    # Q target: hourly aggregation only; no filling/interpolation.
    q = forcing[["interval_end_utc", qcol]].copy()
    q["interval_end_utc"] = q["interval_end_utc"].dt.floor("h")
    q[qcol] = pd.to_numeric(q[qcol], errors="coerce")
    q = q.groupby("interval_end_utc")[qcol].mean().rename("q_obs_m3s")
    f = f.join(q, how="left")

    f["split"] = np.where(
        f.index < train_end,
        "TRAIN_2015_2016",
        "VALIDATION_2017",
    )

    features = [
        c for c in f.columns
        if c not in {"q_obs_m3s", "split"} and c not in coverage_cols
    ]
    static_features = list(features)

    f["static_feature_ready"] = np.isfinite(
        f[static_features].to_numpy(float)
    ).all(axis=1)

    f["model_ready"] = (
        np.isfinite(f["q_obs_m3s"].to_numpy(float))
        & f["static_feature_ready"].to_numpy(bool)
    )

    out = f.reset_index()
    train = out["split"].eq("TRAIN_2015_2016")
    val = out["split"].eq("VALIDATION_2017")
    train_ready = int((train & out["model_ready"]).sum())
    val_ready = int((val & out["model_ready"]).sum())

    qc = pd.DataFrame([
        {"check": "subcatchments", "value": len(ordered_ids), "pass": len(ordered_ids) == a.expected_subcatchments},
        {"check": "train_model_ready_rows", "value": train_ready, "pass": train_ready > 0},
        {"check": "validation_model_ready_rows", "value": val_ready, "pass": val_ready > 0},
        {"check": "contains_2018_plus", "value": int((out["interval_end_utc"] >= pd.Timestamp("2018-01-01T00:00:00Z")).sum()), "pass": False if (out["interval_end_utc"] >= pd.Timestamp("2018-01-01T00:00:00Z")).any() else True},
    ])

    failures = int((~qc["pass"]).sum())
    status = "PASS_ML_V3_FEATURES_READY" if failures == 0 else "FAIL_ML_V3_FEATURE_QC"

    dataset = a.output_dir / "ml_v4_features_2015_2017.csv.gz"
    metadata = a.output_dir / "ml_v4_feature_metadata.json"
    qcfile = a.output_dir / "ml_v4_feature_qc.csv"

    if dataset.exists() and not a.overwrite:
        raise FileExistsError(f"{dataset} exists. Use --overwrite.")

    write_csv(out, dataset, compression="gzip")
    write_csv(qc, qcfile)

    payload = {
        "script_build": BUILD,
        "status": status,
        "blocking_failures": failures,
        "florence_2018_used": False,
        "observed_q_lags": False,
        "q_lag_policy": "none",
        "q_state_feature_columns": [],
        "static_feature_columns": static_features,
        "future_rainfall": False,
        "temporal_interpolation": False,
        "feature_columns": features,
        "feature_count": len(features),
        "subcatchments": len(ordered_ids),
        "modeled_basin_area_km2": float(weights.sum()),
        "routing_distance_column": str(distcol),
        "routing_zones": zone_meta,
        "routing_velocity_priors_m_s": list(ROUTING_VELOCITIES),
        "rows": {
            "train_hours": int(train.sum()),
            "train_q_coverage_percent": float(np.isfinite(out.loc[train, "q_obs_m3s"]).mean() * 100),
            "train_model_ready": train_ready,
            "validation_hours": int(val.sum()),
            "validation_q_coverage_percent": float(np.isfinite(out.loc[val, "q_obs_m3s"]).mean() * 100),
            "validation_model_ready": val_ready,
        },
        "inputs": {
            "spatial_rainfall": str(a.spatial_rainfall),
            "forcing": str(a.forcing),
            "routing_features": str(a.routing_features),
        },
        "outputs": {"dataset": str(dataset), "qc": str(qcfile)},
    }
    write_json(payload, metadata)

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {BUILD}")
    print("ML V3 - CAUSAL STATE-AWARE DEVELOPMENT FEATURE DATASET")
    print("=" * 100)
    print(f"Subcatchments                      : {len(ordered_ids)}")
    print(f"Features                           : {len(features)}")
    print("Observed Q lags                    : NO")
    print("Validation/forecast Q-lag policy   : NONE")
    print("Future rainfall                    : NO")
    print("Temporal interpolation             : NONE")
    print("Florence 2018 used                 : NO")
    print()
    print(f"TRAIN model-ready                  : {train_ready}")
    print(f"VALIDATION model-ready             : {val_ready}")
    print(f"Blocking failures                  : {failures}")
    print(f"Status                             : {status}")
    print(f"Dataset                            : {dataset}")
    print(f"Metadata                           : {metadata}")

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
