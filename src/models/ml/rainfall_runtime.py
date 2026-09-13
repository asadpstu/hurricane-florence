#!/usr/bin/env python3
"""Shared rainfall/routing feature and forcing utilities for ML V3."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BUILD = "ML_V3_RAINFALL_RUNTIME"

SHORT_WINDOWS = (3, 6, 12, 24, 72)
LONG_WINDOWS = (168, 336, 720, 1440, 2160)  # 7, 14, 30, 60, 90 days
LONG_MIN_VALID_FRACTION = 0.99

ROUTING_VELOCITIES = (0.25, 0.50, 1.00)
ZONE_BREAKS_H_AT_05MPS = (48.0, 96.0)
INTENSITY_THRESHOLDS = (1.0, 2.0, 4.0, 6.0)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--spatial-rainfall",
        type=Path,
        required=True,
        help="2018 37-subcatchment spatial MRMS archive.",
    )

    p.add_argument(
        "--routing-features",
        type=Path,
        required=True,
        help="Frozen common A4 routing features.",
    )

    p.add_argument(
        "--feature-metadata",
        type=Path,
        required=True,
        help="feature builder development feature metadata containing locked feature order.",
    )

    p.add_argument(
        "--frozen-model",
        type=Path,
        required=True,
        help="Frozen ML V3 joblib model bundle.",
    )

    forcing = p.add_mutually_exclusive_group(required=True)

    forcing.add_argument(
        "--forcing",
        type=Path,
        help="Exact Florence hourly forcing CSV/CSV.GZ.",
    )

    forcing.add_argument(
        "--forcing-dir",
        type=Path,
        help=(
            "Directory containing Florence forcing outputs. "
            "The script will only accept a uniquely identifiable hourly file."
        ),
    )

    p.add_argument(
        "--feature-start",
        default="2018-06-01T00:00:00Z",
    )

    p.add_argument(
        "--feature-end",
        default="2018-09-27T00:00:00Z",
    )

    p.add_argument(
        "--evaluation-start",
        default="2018-09-10T00:00:00Z",
    )

    p.add_argument(
        "--evaluation-end",
        default="2018-09-26T00:00:00Z",
    )

    p.add_argument(
        "--expected-subcatchments",
        type=int,
        default=37,
    )

    p.add_argument(
        "--min-routed-area-coverage-percent",
        type=float,
        default=99.0,
    )

    p.add_argument(
        "--min-evaluation-q-coverage-percent",
        type=float,
        default=90.0,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


def atomic_csv(df, path, compression=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    tmp.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def utc_timestamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def normalize_sc_id(value):
    text = str(value).strip()
    m = re.search(r"SC\s*0*(\d+)", text, flags=re.I)

    if m:
        return f"SC{int(m.group(1)):03d}"

    try:
        return f"SC{int(float(text)):03d}"
    except Exception:
        return text


def resolve_column(df, candidates, label):
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise RuntimeError(
        f"Could not resolve {label}. "
        f"Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def resolve_optional_column(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def resolve_rainfall_columns(df):
    """
    Match one rainfall column per SCxxx using the same conservative logic as feature builder.
    """
    bad_tokens = (
        "coverage",
        "valid",
        "missing",
        "flag",
        "source",
        "count",
        "area",
        "weight",
        "qc",
        "status",
    )

    mapping = {}

    for column in df.columns:
        text = str(column)
        lower = text.lower()

        if any(token in lower for token in bad_tokens):
            continue

        m = re.search(r"(SC\s*0*\d+)", text, flags=re.I)

        if not m:
            continue

        sid = normalize_sc_id(m.group(1))

        priority = 0

        if "rain" in lower:
            priority += 2

        if "mm" in lower:
            priority += 1

        previous = mapping.get(sid)

        if previous is None or priority > previous[0]:
            mapping[sid] = (priority, column)

    return {
        sid: item[1]
        for sid, item in mapping.items()
    }


def weighted_mean(matrix, weights):
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)

    finite = np.isfinite(matrix)
    denominator = (finite * weights[None, :]).sum(axis=1)
    numerator = np.nansum(matrix * weights[None, :], axis=1)

    return np.divide(
        numerator,
        denominator,
        out=np.full(len(matrix), np.nan),
        where=denominator > 0,
    )


def weighted_std(matrix, weights):
    mean = weighted_mean(matrix, weights)
    finite = np.isfinite(matrix)
    denominator = (finite * weights[None, :]).sum(axis=1)

    numerator = np.nansum(
        np.square(matrix - mean[:, None])
        * weights[None, :],
        axis=1,
    )

    variance = np.divide(
        numerator,
        denominator,
        out=np.full(len(matrix), np.nan),
        where=denominator > 0,
    )

    return np.sqrt(variance)


def weighted_fraction_above(matrix, weights, threshold):
    finite = np.isfinite(matrix)
    denominator = (finite * weights[None, :]).sum(axis=1)

    numerator = (
        (finite & (matrix > threshold))
        * weights[None, :]
    ).sum(axis=1)

    return np.divide(
        numerator,
        denominator,
        out=np.full(len(matrix), np.nan),
        where=denominator > 0,
    )


def weighted_p90_row(values, weights):
    finite = np.isfinite(values)

    if not finite.any():
        return np.nan

    v = values[finite]
    w = weights[finite]

    order = np.argsort(v)
    v = v[order]
    w = w[order]

    cumulative = np.cumsum(w)
    total = cumulative[-1]

    if total <= 0:
        return np.nan

    idx = int(
        np.searchsorted(
            cumulative,
            0.90 * total,
            side="left",
        )
    )

    return float(v[min(idx, len(v) - 1)])


def rolling_sum(series, hours, minimum_fraction=1.0):
    minimum = int(
        math.ceil(hours * minimum_fraction)
    )

    return series.rolling(
        window=hours,
        min_periods=minimum,
    ).sum()


def routed_index(matrix, weights, lag_hours, minimum_coverage_fraction):
    """
    Exact feature builder causal routing-index definition:
    value at time t uses each subcatchment's rainfall at t-lag_sc.
    Missing hours are never filled.
    """
    n_time, n_sc = matrix.shape

    shifted = np.full(
        (n_time, n_sc),
        np.nan,
        dtype=float,
    )

    for j in range(n_sc):
        lag = int(lag_hours[j])

        if lag < 0:
            raise RuntimeError(
                "Negative routing lag encountered."
            )

        if lag == 0:
            shifted[:, j] = matrix[:, j]

        elif lag < n_time:
            shifted[lag:, j] = matrix[:-lag, j]

    finite = np.isfinite(shifted)

    covered_area = (
        finite * weights[None, :]
    ).sum(axis=1)

    total_area = float(weights.sum())

    coverage_fraction = (
        covered_area / total_area
    )

    numerator = np.nansum(
        shifted * weights[None, :],
        axis=1,
    )

    result = np.divide(
        numerator,
        covered_area,
        out=np.full(n_time, np.nan),
        where=covered_area > 0,
    )

    result[
        coverage_fraction
        < minimum_coverage_fraction
    ] = np.nan

    return result, coverage_fraction


def hydrologic_metrics(obs, sim, times):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    times = pd.DatetimeIndex(times)

    valid = np.isfinite(obs) & np.isfinite(sim)

    obs = obs[valid]
    sim = sim[valid]
    times = times[valid]

    if len(obs) < 2:
        raise RuntimeError(
            "Too few common observed/predicted rows for final evaluation."
        )

    mean_obs = float(np.mean(obs))
    denominator = float(
        np.sum(
            np.square(obs - mean_obs)
        )
    )

    nse = (
        1.0
        - float(
            np.sum(
                np.square(sim - obs)
            )
        )
        / denominator
        if denominator > 0
        else np.nan
    )

    std_obs = float(
        np.std(obs, ddof=0)
    )

    std_sim = float(
        np.std(sim, ddof=0)
    )

    if std_obs > 0 and std_sim > 0:
        correlation = float(
            np.corrcoef(obs, sim)[0, 1]
        )
    else:
        correlation = np.nan

    alpha = (
        std_sim / std_obs
        if std_obs > 0
        else np.nan
    )

    beta = (
        float(
            np.mean(sim)
            / np.mean(obs)
        )
        if abs(np.mean(obs)) > 1e-12
        else np.nan
    )

    kge = (
        1.0
        - math.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
        if np.isfinite(
            [correlation, alpha, beta]
        ).all()
        else np.nan
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(sim - obs)
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(sim - obs)
        )
    )

    pbias = (
        float(
            100.0
            * np.sum(sim - obs)
            / np.sum(obs)
        )
        if abs(np.sum(obs)) > 1e-12
        else np.nan
    )

    obs_peak_index = int(
        np.argmax(obs)
    )

    sim_peak_index = int(
        np.argmax(sim)
    )

    obs_peak = float(
        obs[obs_peak_index]
    )

    sim_peak = float(
        sim[sim_peak_index]
    )

    peak_error = (
        float(
            100.0
            * (sim_peak - obs_peak)
            / obs_peak
        )
        if abs(obs_peak) > 1e-12
        else np.nan
    )

    peak_timing_error = float(
        (
            times[sim_peak_index]
            - times[obs_peak_index]
        )
        / pd.Timedelta(hours=1)
    )

    observed_volume_m3 = float(
        np.sum(obs) * 3600.0
    )

    predicted_volume_m3 = float(
        np.sum(sim) * 3600.0
    )

    volume_bias_percent = (
        float(
            100.0
            * (
                predicted_volume_m3
                - observed_volume_m3
            )
            / observed_volume_m3
        )
        if abs(observed_volume_m3) > 1e-12
        else np.nan
    )

    return {
        "common_rows": int(len(obs)),
        "nse": float(nse),
        "kge": float(kge),
        "correlation": float(correlation),
        "variability_ratio_alpha": float(alpha),
        "mean_ratio_beta": float(beta),
        "rmse_m3s": rmse,
        "mae_m3s": mae,
        "pbias_percent": pbias,
        "observed_peak_m3s": obs_peak,
        "predicted_peak_m3s": sim_peak,
        "peak_error_percent": peak_error,
        "observed_peak_time_utc": str(
            times[obs_peak_index]
        ),
        "predicted_peak_time_utc": str(
            times[sim_peak_index]
        ),
        "peak_timing_error_h": peak_timing_error,
        "observed_volume_m3": observed_volume_m3,
        "predicted_volume_m3": predicted_volume_m3,
        "volume_bias_percent": volume_bias_percent,
    }


def inspect_forcing_candidate(path, evaluation_start, evaluation_end):
    """
    Return a candidate description or None.

    We require:
    * recognized timestamp column
    * recognized observed-Q column
    * substantial hourly-like coverage of the Florence evaluation window
    * no duplicate timestamps after hourly flooring

    The function never interpolates.
    """
    try:
        header = pd.read_csv(
            path,
            nrows=5,
        )
    except Exception:
        return None

    time_col = resolve_optional_column(
        header,
        [
            "interval_end_utc",
            "timestamp",
            "time",
        ],
    )

    q_col = resolve_optional_column(
        header,
        [
            "q_obs_mean_m3s",
            "q_obs_m3s",
            "q_mean_m3s",
            "discharge_m3s",
        ],
    )

    if time_col is None or q_col is None:
        return None

    try:
        data = pd.read_csv(
            path,
            usecols=[time_col, q_col],
        )
    except Exception:
        return None

    try:
        data["time"] = pd.to_datetime(
            data[time_col],
            utc=True,
            errors="raise",
        )
    except Exception:
        return None

    data["q"] = pd.to_numeric(
        data[q_col],
        errors="coerce",
    )

    data["hour"] = (
        data["time"]
        .dt.floor("h")
    )

    duplicate_hours = int(
        data["hour"]
        .duplicated()
        .sum()
    )

    in_eval = data[
        (data["hour"] >= evaluation_start)
        & (data["hour"] < evaluation_end)
    ].copy()

    if in_eval.empty:
        return None

    unique_eval_hours = int(
        in_eval["hour"]
        .nunique()
    )

    finite_q_hours = int(
        in_eval.loc[
            np.isfinite(
                in_eval["q"]
                .to_numpy(dtype=float)
            ),
            "hour",
        ]
        .nunique()
    )

    total_eval_hours = int(
        (
            evaluation_end
            - evaluation_start
        )
        / pd.Timedelta(hours=1)
    )

    q_coverage_percent = (
        100.0
        * finite_q_hours
        / total_eval_hours
    )

    # Conservative hourly-file gate.
    if unique_eval_hours < int(
        0.75 * total_eval_hours
    ):
        return None

    score = 0

    filename = path.name.lower()

    if "hour" in filename:
        score += 5

    if "forcing" in filename:
        score += 3

    if q_col == "q_obs_mean_m3s":
        score += 5

    score += min(
        unique_eval_hours,
        total_eval_hours,
    ) / total_eval_hours

    score += (
        q_coverage_percent / 100.0
    )

    return {
        "path": path,
        "time_col": time_col,
        "q_col": q_col,
        "unique_eval_hours": unique_eval_hours,
        "finite_q_hours": finite_q_hours,
        "q_coverage_percent": q_coverage_percent,
        "duplicate_hours_global": duplicate_hours,
        "score": float(score),
    }


def select_forcing(
    forcing_path,
    forcing_dir,
    evaluation_start,
    evaluation_end,
):
    if forcing_path is not None:
        candidate = inspect_forcing_candidate(
            forcing_path,
            evaluation_start,
            evaluation_end,
        )

        if candidate is None:
            raise RuntimeError(
                f"Provided forcing file does not pass hourly Florence forcing checks: "
                f"{forcing_path}"
            )

        return candidate

    if forcing_dir is None or not forcing_dir.exists():
        raise RuntimeError(
            f"Forcing directory does not exist: {forcing_dir}"
        )

    paths = sorted(
        list(
            forcing_dir.rglob("*.csv")
        )
        + list(
            forcing_dir.rglob("*.csv.gz")
        )
    )

    candidates = []

    for path in paths:
        candidate = inspect_forcing_candidate(
            path,
            evaluation_start,
            evaluation_end,
        )

        if candidate is not None:
            candidates.append(
                candidate
            )

    if not candidates:
        raise RuntimeError(
            "No hourly Florence forcing candidate found. "
            f"Directory searched: {forcing_dir}"
        )

    candidates.sort(
        key=lambda item: (
            item["score"],
            item["finite_q_hours"],
            item["unique_eval_hours"],
        ),
        reverse=True,
    )

    best = candidates[0]

    if len(candidates) > 1:
        second = candidates[1]

        if abs(
            best["score"]
            - second["score"]
        ) < 1e-9:
            lines = [
                (
                    f"{item['path']} | "
                    f"Q={item['q_col']} | "
                    f"hours={item['unique_eval_hours']} | "
                    f"Qcov={item['q_coverage_percent']:.3f}% | "
                    f"score={item['score']:.3f}"
                )
                for item in candidates[:10]
            ]

            raise RuntimeError(
                "Multiple equally ranked hourly forcing candidates found. "
                "Refusing to guess.\n"
                + "\n".join(lines)
            )

    return best


def build_features(
    spatial_rainfall_path,
    routing_features_path,
    feature_start,
    feature_end,
    expected_subcatchments,
    minimum_routed_coverage_percent,
):
    rain = pd.read_csv(
        spatial_rainfall_path
    )

    routing = pd.read_csv(
        routing_features_path
    )

    rain_time_col = resolve_column(
        rain,
        [
            "interval_end_utc",
            "timestamp",
            "time",
            "valid_time",
        ],
        "spatial rainfall time",
    )

    routing_sc_col = resolve_column(
        routing,
        [
            "subcatchment_id",
            "sc_id",
            "subcatchment",
        ],
        "routing subcatchment ID",
    )

    routing_area_col = resolve_column(
        routing,
        [
            "local_area_km2",
            "area_km2",
            "subcatchment_area_km2",
        ],
        "subcatchment area",
    )

    routing_distance_col = resolve_column(
        routing,
        [
            "network_distance_proxy_to_gauge_km",
            "cumulative_distance_to_gauge_km",
            "network_distance_to_gauge_km",
        ],
        "cumulative routing distance",
    )

    rain["interval_end_utc"] = pd.to_datetime(
        rain[rain_time_col],
        utc=True,
        errors="raise",
    )

    routing["subcatchment_id"] = (
        routing[routing_sc_col]
        .map(normalize_sc_id)
    )

    routing["area_km2"] = pd.to_numeric(
        routing[routing_area_col],
        errors="raise",
    )

    routing["distance_km"] = pd.to_numeric(
        routing[routing_distance_col],
        errors="raise",
    )

    if routing["subcatchment_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate routing subcatchment IDs."
        )

    rain_map = resolve_rainfall_columns(
        rain
    )

    ordered_ids = [
        sid
        for sid in routing["subcatchment_id"]
        if sid in rain_map
    ]

    if len(ordered_ids) != expected_subcatchments:
        raise RuntimeError(
            f"Expected {expected_subcatchments} common subcatchments; "
            f"found {len(ordered_ids)}."
        )

    routing = (
        routing
        .set_index("subcatchment_id")
        .loc[ordered_ids]
    )

    rainfall_columns = [
        rain_map[sid]
        for sid in ordered_ids
    ]

    rain = (
        rain[
            ["interval_end_utc"]
            + rainfall_columns
        ]
        .sort_values(
            "interval_end_utc"
        )
    )

    if rain["interval_end_utc"].duplicated().any():
        raise RuntimeError(
            "Duplicate timestamps in 2018 spatial rainfall archive."
        )

    index = pd.date_range(
        start=feature_start,
        end=feature_end,
        freq="h",
        inclusive="left",
    )

    rain = (
        rain
        .set_index(
            "interval_end_utc"
        )
        .reindex(index)
    )

    matrix = (
        rain[rainfall_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    weights = (
        routing["area_km2"]
        .to_numpy(
            dtype=float
        )
    )

    distance_km = (
        routing["distance_km"]
        .to_numpy(
            dtype=float
        )
    )

    if (
        (weights <= 0).any()
        or not np.isfinite(weights).all()
    ):
        raise RuntimeError(
            "Invalid subcatchment areas."
        )

    feature = pd.DataFrame(
        index=index
    )

    feature.index.name = (
        "interval_end_utc"
    )

    # ================================================================
    # Exact feature builder feature order begins here.
    # ================================================================
    feature["rain_basin_1h_mm"] = weighted_mean(
        matrix,
        weights,
    )

    feature["rain_spatial_std_1h_mm"] = weighted_std(
        matrix,
        weights,
    )

    feature["rain_spatial_max_1h_mm"] = np.array(
        [
            (
                np.nanmax(row)
                if np.isfinite(row).any()
                else np.nan
            )
            for row in matrix
        ]
    )

    feature["rain_spatial_p90_1h_mm"] = [
        weighted_p90_row(
            row,
            weights,
        )
        for row in matrix
    ]

    for threshold in INTENSITY_THRESHOLDS:
        tag = str(
            threshold
        ).replace(
            ".",
            "p",
        )

        feature[
            f"rain_area_fraction_gt_{tag}mmh"
        ] = weighted_fraction_above(
            matrix,
            weights,
            threshold,
        )

    for hours in SHORT_WINDOWS:
        feature[
            f"rain_basin_sum_{hours}h_mm"
        ] = rolling_sum(
            feature["rain_basin_1h_mm"],
            hours,
            1.0,
        )

    for hours in LONG_WINDOWS:
        feature[
            f"rain_basin_sum_{hours}h_mm"
        ] = rolling_sum(
            feature["rain_basin_1h_mm"],
            hours,
            LONG_MIN_VALID_FRACTION,
        )

    travel_h_05 = (
        distance_km
        * 1000.0
        / 0.50
        / 3600.0
    )

    zone_masks = {
        "near": (
            travel_h_05
            <= ZONE_BREAKS_H_AT_05MPS[0]
        ),
        "middle": (
            (
                travel_h_05
                > ZONE_BREAKS_H_AT_05MPS[0]
            )
            & (
                travel_h_05
                <= ZONE_BREAKS_H_AT_05MPS[1]
            )
        ),
        "far": (
            travel_h_05
            > ZONE_BREAKS_H_AT_05MPS[1]
        ),
    }

    zone_metadata = {}

    for zone_name, mask in zone_masks.items():
        if not mask.any():
            raise RuntimeError(
                f"Routing zone {zone_name!r} has zero subcatchments."
            )

        zone_rain = (
            matrix[:, mask]
        )

        zone_weights = (
            weights[mask]
        )

        feature[
            f"rain_{zone_name}_1h_mm"
        ] = weighted_mean(
            zone_rain,
            zone_weights,
        )

        feature[
            f"rain_{zone_name}_sum_6h_mm"
        ] = rolling_sum(
            feature[
                f"rain_{zone_name}_1h_mm"
            ],
            6,
            1.0,
        )

        feature[
            f"rain_{zone_name}_sum_24h_mm"
        ] = rolling_sum(
            feature[
                f"rain_{zone_name}_1h_mm"
            ],
            24,
            1.0,
        )

        zone_metadata[
            zone_name
        ] = {
            "subcatchments": int(
                mask.sum()
            ),
            "area_km2": float(
                zone_weights.sum()
            ),
            "travel_time_0p5mps_min_h": float(
                travel_h_05[mask].min()
            ),
            "travel_time_0p5mps_max_h": float(
                travel_h_05[mask].max()
            ),
        }

    coverage_columns = []

    for velocity in ROUTING_VELOCITIES:
        tag = str(
            velocity
        ).replace(
            ".",
            "p",
        )

        lag_hours = np.rint(
            distance_km
            * 1000.0
            / velocity
            / 3600.0
        ).astype(
            int
        )

        routed, coverage = routed_index(
            matrix,
            weights,
            lag_hours,
            minimum_routed_coverage_percent
            / 100.0,
        )

        rainfall_col = (
            f"rain_routed_v{tag}_1h_mm"
        )

        coverage_col = (
            f"rain_routed_v{tag}_area_coverage"
        )

        feature[
            rainfall_col
        ] = routed

        feature[
            coverage_col
        ] = coverage

        coverage_columns.append(
            coverage_col
        )

        feature[
            f"rain_routed_v{tag}_sum_6h_mm"
        ] = rolling_sum(
            feature[
                rainfall_col
            ],
            6,
            1.0,
        )

        feature[
            f"rain_routed_v{tag}_sum_24h_mm"
        ] = rolling_sum(
            feature[
                rainfall_col
            ],
            24,
            1.0,
        )

    day_of_year = (
        feature.index
        .dayofyear
        .to_numpy(
            dtype=float
        )
    )

    hour = (
        feature.index
        .hour
        .to_numpy(
            dtype=float
        )
    )

    feature[
        "season_sin_doy"
    ] = np.sin(
        2.0
        * np.pi
        * day_of_year
        / 365.25
    )

    feature[
        "season_cos_doy"
    ] = np.cos(
        2.0
        * np.pi
        * day_of_year
        / 365.25
    )

    feature[
        "season_sin_hour"
    ] = np.sin(
        2.0
        * np.pi
        * hour
        / 24.0
    )

    feature[
        "season_cos_hour"
    ] = np.cos(
        2.0
        * np.pi
        * hour
        / 24.0
    )

    predictive_columns = [
        column
        for column in feature.columns
        if column not in coverage_columns
    ]
    # ================================================================
    # Exact feature builder feature definitions end here.
    # ================================================================

    return (
        feature,
        predictive_columns,
        coverage_columns,
        ordered_ids,
        zone_metadata,
        float(weights.sum()),
        str(routing_distance_col),
    )

