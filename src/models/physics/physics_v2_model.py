"""
PHYSICS V2 SEMI-DISTRIBUTED MODEL
Semi-distributed rainfall-runoff calibration using:
  * 37 MRMS subcatchment rainfall forcings
  * ERA5-Land calibration-scaled dynamic antecedent wetness
  * hourly FAO-56 reference ET0
  * NHDPlus-derived subcatchment routing

Development protocol
--------------------
Calibration : 2015-2016 event blocks
Validation  : 2017 event blocks
Florence    : NOT USED

Key revision relative to earlier prototypes
--------------------------------
* No arbitrary fixed initial soil fraction.
  Each event/subcatchment starts from ERA5-Land relative root-zone wetness:
      S0 = wetness_ERA5 * calibrated soil capacity

* No constant soil_loss_rate_mm_h.
  Soil-water loss is driven by hourly reference ET0:
      AET = ET0 * et_multiplier * moisture_limitation

* ERA5-Land is used only to initialize the conceptual event state.
  After initialization, the conceptual soil store evolves independently from
  rainfall, ET, runoff and drainage. ERA5 soil moisture is NOT imposed every
  model hour.

* Initial base reservoir is placed in equilibrium with the initial soil
  drainage rate. This avoids an arbitrary fixed base-store reset while using
  no observed discharge for initialization.

* Observed Q is used only for calibration/validation scoring.

Conceptual model
----------------
For each subcatchment and hour:

  1. Dynamic actual ET:
         AET = min(S, ET0 * et_multiplier * (S/C)^et_moisture_exponent)

  2. Saturation-dependent rainfall partition:
         quick_fraction = (S/C)^runoff_beta
         quick_input = P * quick_fraction
         infiltration = P - quick_input

  3. Add infiltration to soil; capacity overflow becomes quick runoff.

  4. Drain soil water above field capacity to baseflow:
         recharge = max(S - FC*C, 0) / K_drain

  5. Route quick input through local quick reservoir.
  6. Route recharge through local base reservoir.
  7. Route local + upstream discharge through a channel linear reservoir.

All fluxes are represented as equivalent depth over each local subcatchment,
then converted to m3/s before network routing.

Parameter vector
----------------
soil_capacity_mm
field_capacity_fraction
runoff_beta
soil_drainage_time_h
quick_reservoir_time_h
base_reservoir_time_h
et_multiplier
et_moisture_exponent
channel_velocity_m_s

Calibration objective
---------------------
Event-balanced objective:
  95% mean accepted event-block objective
   5% pooled regularization objective

Default block metric weights:
  NSE      30%
  KGE      30%
  log-NSE  15%
  |PBIAS|  10%
  peak err 15%

All objective weights are exposed as CLI arguments and recorded in metadata.

The first objective_warmup_hours after each ORIGINAL event block start are
excluded from scoring. No extra rainfall spin-up is used because the event
initial soil state now comes from ERA5-Land.

No rainfall/Q/ERA5 interpolation is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


BUILD = "PHYSICS_V2_SEMIDISTRIBUTED_ERA5_ROUTING_EVENT_BALANCED_V2"


# ---------------------------------------------------------------------------
# CLI / I/O
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--forcing", type=Path, required=True)
    p.add_argument("--event-blocks", type=Path, required=True)
    p.add_argument("--q-target-mask", type=Path, required=True)

    p.add_argument("--dynamic-state-et0", type=Path, required=True)

    p.add_argument("--routing-features", type=Path, required=True)
    p.add_argument("--routing", type=Path, required=True)

    p.add_argument("--expected-subcatchments", type=int, default=37)
    p.add_argument("--objective-warmup-hours", type=int, default=72)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer-maxiter", type=int, default=40)
    p.add_argument("--optimizer-popsize", type=int, default=6)
    p.add_argument("--optimizer-tol", type=float, default=0.01)
    p.add_argument("--optimizer-polish", action="store_true")

    # Calibration-objective controls.  The default deliberately gives nearly
    # equal importance to each accepted calibration block and only a small
    # pooled term.  This prevents a single extreme event from dominating the
    # optimization simply because its discharge magnitudes are much larger.
    p.add_argument("--objective-block-weight", type=float, default=0.95)
    p.add_argument("--objective-pooled-weight", type=float, default=0.05)

    p.add_argument("--metric-weight-nse", type=float, default=0.30)
    p.add_argument("--metric-weight-kge", type=float, default=0.30)
    p.add_argument("--metric-weight-log-nse", type=float, default=0.15)
    p.add_argument("--metric-weight-pbias", type=float, default=0.10)
    p.add_argument("--metric-weight-peak", type=float, default=0.15)

    p.add_argument("--min-validation-nse", type=float, default=0.30)
    p.add_argument("--min-validation-kge", type=float, default=0.30)
    p.add_argument(
        "--max-validation-abs-pbias-percent",
        type=float,
        default=30.0,
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


# ---------------------------------------------------------------------------
# Flexible schema helpers
# ---------------------------------------------------------------------------

def find_column(df, candidates, *, required=True, description="column"):
    lower = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if required:
        raise RuntimeError(
            f"Could not resolve {description}. "
            f"Tried {candidates}; available={list(df.columns)}"
        )

    return None


def normalize_sc_id(x):
    s = str(x).strip()

    if s.upper().startswith("SC"):
        try:
            return f"SC{int(s[2:]):03d}"
        except Exception:
            return s

    try:
        return f"SC{int(float(s)):03d}"
    except Exception:
        return s


def rainfall_columns(df):
    cols = [
        c for c in df.columns
        if str(c).startswith("rain_SC")
        and str(c).endswith("_mm")
    ]

    if not cols:
        # Broader fallback.
        cols = [
            c for c in df.columns
            if "rain" in str(c).lower()
            and "sc" in str(c).lower()
        ]

    mapping = {}

    for c in cols:
        text = str(c)

        if text.startswith("rain_") and text.endswith("_mm"):
            sid = text[len("rain_"):-len("_mm")]
        else:
            # Find SC### token.
            upper = text.upper()
            pos = upper.find("SC")

            if pos < 0:
                continue

            digits = ""
            for ch in upper[pos + 2:]:
                if ch.isdigit():
                    digits += ch
                elif digits:
                    break

            if not digits:
                continue

            sid = f"SC{int(digits):03d}"

        mapping[normalize_sc_id(sid)] = c

    return mapping


def infer_bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return (
            pd.to_numeric(series, errors="coerce")
            .fillna(0)
            .astype(float)
            > 0
        )

    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return values.isin(
        {
            "1",
            "true",
            "yes",
            "y",
            "valid",
            "use",
            "target",
            "keep",
        }
    )


# ---------------------------------------------------------------------------
# Hydrologic metrics
# ---------------------------------------------------------------------------

def nse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    valid = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[valid]
    sim = sim[valid]

    if len(obs) < 2:
        return np.nan

    den = np.sum((obs - np.mean(obs)) ** 2)

    if den <= 0:
        return np.nan

    return float(
        1.0
        - np.sum((sim - obs) ** 2) / den
    )


def kge(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    valid = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[valid]
    sim = sim[valid]

    if len(obs) < 2:
        return np.nan

    mean_o = np.mean(obs)
    mean_s = np.mean(sim)
    std_o = np.std(obs, ddof=1)
    std_s = np.std(sim, ddof=1)

    if (
        not np.isfinite(mean_o)
        or not np.isfinite(mean_s)
        or abs(mean_o) <= 1e-12
        or std_o <= 1e-12
        or std_s <= 1e-12
    ):
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = std_s / std_o
    beta = mean_s / mean_o

    return float(
        1.0
        - np.sqrt(
            (r - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def log_nse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    valid = (
        np.isfinite(obs)
        & np.isfinite(sim)
        & (obs >= 0)
        & (sim >= 0)
    )

    if valid.sum() < 2:
        return np.nan

    return nse(
        np.log1p(obs[valid]),
        np.log1p(sim[valid]),
    )


def rmse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    valid = np.isfinite(obs) & np.isfinite(sim)

    if not valid.any():
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                (sim[valid] - obs[valid]) ** 2
            )
        )
    )


def mae(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    valid = np.isfinite(obs) & np.isfinite(sim)

    if not valid.any():
        return np.nan

    return float(
        np.mean(
            np.abs(sim[valid] - obs[valid])
        )
    )


def pbias(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    valid = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[valid]
    sim = sim[valid]

    if len(obs) == 0:
        return np.nan

    den = np.sum(obs)

    if abs(den) <= 1e-12:
        return np.nan

    return float(
        100.0
        * np.sum(sim - obs)
        / den
    )


def peak_error_percent(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    valid = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[valid]
    sim = sim[valid]

    if len(obs) == 0:
        return np.nan

    op = float(np.max(obs))
    sp = float(np.max(sim))

    if op <= 1e-12:
        return np.nan

    return float(
        100.0 * (sp - op) / op
    )


def metric_bundle(obs, sim):
    return {
        "nse": nse(obs, sim),
        "kge": kge(obs, sim),
        "log_nse": log_nse(obs, sim),
        "rmse_m3s": rmse(obs, sim),
        "mae_m3s": mae(obs, sim),
        "pbias_percent": pbias(obs, sim),
        "peak_error_percent": peak_error_percent(obs, sim),
    }


def metric_penalty(metrics, weights=None):
    """
    Lower is better. Finite and deliberately bounded enough to keep
    differential evolution stable when one metric is bad.

    ``weights`` is explicit so the calibration policy is versioned in the
    command line and metadata instead of being hidden in code.
    """
    if weights is None:
        weights = {
            "nse": 0.30,
            "kge": 0.30,
            "log_nse": 0.15,
            "pbias": 0.10,
            "peak": 0.15,
        }

    total = float(sum(weights.values()))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Metric weights must sum to a positive finite value.")

    weights = {k: float(v) / total for k, v in weights.items()}
    def bad_efficiency(v):
        if not np.isfinite(v):
            return 3.0

        # 1 - NSE/KGE/logNSE, clipped to prevent one catastrophic event from
        # overwhelming every other block.
        return min(max(1.0 - v, 0.0), 4.0)

    nse_p = bad_efficiency(metrics["nse"])
    kge_p = bad_efficiency(metrics["kge"])
    log_p = bad_efficiency(metrics["log_nse"])

    p = metrics["pbias_percent"]
    peak = metrics["peak_error_percent"]

    pbias_p = (
        min(abs(p) / 100.0, 3.0)
        if np.isfinite(p)
        else 3.0
    )

    peak_p = (
        min(abs(peak) / 100.0, 3.0)
        if np.isfinite(peak)
        else 3.0
    )

    return (
        weights["nse"] * nse_p
        + weights["kge"] * kge_p
        + weights["log_nse"] * log_p
        + weights["pbias"] * pbias_p
        + weights["peak"] * peak_p
    )


# ---------------------------------------------------------------------------
# Routing framework
# ---------------------------------------------------------------------------

def resolve_routing_schema(routing):
    up_col = find_column(
        routing,
        [
            "upstream_subcatchment_id",
            "upstream_id",
            "from_subcatchment_id",
            "from_id",
            "source_subcatchment_id",
            "source",
        ],
        description="routing upstream ID",
    )

    down_col = find_column(
        routing,
        [
            "downstream_subcatchment_id",
            "downstream_id",
            "to_subcatchment_id",
            "to_id",
            "target_subcatchment_id",
            "target",
        ],
        description="routing downstream ID",
    )

    return up_col, down_col


def resolve_feature_schema(features):
    sid_col = find_column(
        features,
        [
            "subcatchment_id",
            "modeling_subcatchment_id",
            "sc_id",
        ],
        description="routing-feature subcatchment ID",
    )

    area_col = find_column(
        features,
        [
            "local_area_km2",
            "area_km2",
            "subcatchment_area_km2",
            "modeled_area_km2",
        ],
        description="local subcatchment area",
    )

    distance_col = find_column(
        features,
        [
            "network_distance_proxy_to_gauge_km",
            "network_distance_to_gauge_km",
            "network_distance_to_outlet_km",
            "distance_to_outlet_km",
            "network_distance_km",
            "routing_distance_to_outlet_km",
            "primary_routing_distance_to_outlet_km",
        ],
        required=False,
        description="network distance to outlet",
    )

    topo_col = find_column(
        features,
        [
            "topological_index",
            "topo_index",
            "routing_order",
            "topological_order",
        ],
        required=False,
        description="topological index",
    )

    return sid_col, area_col, distance_col, topo_col


def topological_order(ids, edges):
    downstream = {sid: [] for sid in ids}
    indegree = {sid: 0 for sid in ids}

    for u, v in edges:
        downstream[u].append(v)
        indegree[v] += 1

    queue = sorted(
        [
            sid
            for sid in ids
            if indegree[sid] == 0
        ]
    )

    order = []

    while queue:
        u = queue.pop(0)
        order.append(u)

        for v in downstream[u]:
            indegree[v] -= 1

            if indegree[v] == 0:
                queue.append(v)
                queue.sort()

    if len(order) != len(ids):
        raise RuntimeError(
            "Routing graph is cyclic or disconnected from "
            "the supplied node inventory."
        )

    return order


def build_routing(features, routing, expected):
    sid_col, area_col, dist_col, topo_col = resolve_feature_schema(features)
    up_col, down_col = resolve_routing_schema(routing)

    f = features.copy()
    f[sid_col] = f[sid_col].map(normalize_sc_id)
    f[area_col] = pd.to_numeric(f[area_col], errors="raise")

    if f[sid_col].nunique() != expected:
        raise RuntimeError(
            f"Expected {expected} routing-feature subcatchments; "
            f"found {f[sid_col].nunique()}."
        )

    ids = sorted(f[sid_col].unique())
    id_set = set(ids)

    r = routing.copy()
    r[up_col] = r[up_col].map(normalize_sc_id)
    r[down_col] = r[down_col].map(normalize_sc_id)

    r = r[
        r[up_col].isin(id_set)
        & r[down_col].isin(id_set)
    ].copy()

    edges = [
        (str(u), str(v))
        for u, v in zip(r[up_col], r[down_col])
    ]

    if len(edges) != expected - 1:
        raise RuntimeError(
            f"Expected {expected - 1} routing edges; "
            f"found {len(edges)}."
        )

    order = topological_order(ids, edges)
    position = {sid: i for i, sid in enumerate(order)}

    for u, v in edges:
        if position[u] >= position[v]:
            raise RuntimeError(
                f"Topological ordering failed for edge {u}->{v}."
            )

    downstream_of = {sid: None for sid in ids}
    upstream_of = {sid: [] for sid in ids}

    for u, v in edges:
        if downstream_of[u] is not None:
            raise RuntimeError(
                f"Subcatchment {u} has multiple downstream model nodes."
            )

        downstream_of[u] = v
        upstream_of[v].append(u)

    outlets = [
        sid
        for sid in ids
        if downstream_of[sid] is None
    ]

    if len(outlets) != 1:
        raise RuntimeError(
            f"Expected one basin outlet; found {outlets}."
        )

    outlet = outlets[0]

    area = {
        str(row[sid_col]): float(row[area_col])
        for _, row in f.iterrows()
    }

    # Segment-length proxy.
    #
    # Prefer difference in network distance-to-outlet between an upstream
    # node and its downstream node. If no usable distance column exists,
    # use a conservative 5 km local segment proxy and record the fallback.
    segment_km = {}
    segment_method = "fixed_5km_fallback"

    if dist_col is not None:
        d = {
            str(row[sid_col]): float(row[dist_col])
            for _, row in f.iterrows()
            if np.isfinite(
                pd.to_numeric(row[dist_col], errors="coerce")
            )
        }

        usable = True

        for sid in ids:
            dn = downstream_of[sid]

            if dn is None:
                # Outlet gets a small local travel segment to the gauge.
                segment_km[sid] = 1.0
                continue

            if sid not in d or dn not in d:
                usable = False
                break

            seg = d[sid] - d[dn]

            if not np.isfinite(seg):
                usable = False
                break

            segment_km[sid] = max(float(seg), 0.25)

        if usable:
            segment_method = (
                f"difference_of_{dist_col}"
            )
        else:
            segment_km = {}

    if not segment_km:
        segment_km = {
            sid: 1.0 if sid == outlet else 5.0
            for sid in ids
        }

    segment_values = np.asarray(
        [segment_km[sid] for sid in ids],
        dtype=float,
    )

    cumulative_distance_stats = None

    if dist_col is not None and dist_col in f.columns:
        cumulative = pd.to_numeric(
            f[dist_col],
            errors="coerce",
        )

        finite_cumulative = cumulative[
            np.isfinite(cumulative)
        ]

        if len(finite_cumulative):
            cumulative_distance_stats = {
                "column": dist_col,
                "min_km": float(finite_cumulative.min()),
                "median_km": float(finite_cumulative.median()),
                "max_km": float(finite_cumulative.max()),
            }

    return {
        "ids": ids,
        "order": order,
        "position": position,
        "edges": edges,
        "downstream_of": downstream_of,
        "upstream_of": upstream_of,
        "outlet": outlet,
        "area_km2": area,
        "segment_km": segment_km,
        "segment_method": segment_method,
        "segment_distance_stats": {
            "min_km": float(np.min(segment_values)),
            "median_km": float(np.median(segment_values)),
            "max_km": float(np.max(segment_values)),
        },
        "cumulative_distance_stats": cumulative_distance_stats,
    }


# ---------------------------------------------------------------------------
# Event forcing assembly
# ---------------------------------------------------------------------------

def load_q_target_mask(path):
    mask = pd.read_csv(path)

    time_col = find_column(
        mask,
        [
            "interval_end_utc",
            "time_utc",
            "timestamp_utc",
            "datetime_utc",
        ],
        required=False,
        description="Q target mask timestamp",
    )

    block_col = find_column(
        mask,
        [
            "event_block_id",
            "block_id",
        ],
        required=False,
        description="Q target mask block ID",
    )

    bool_candidates = [
        "q_target_valid",
        "use_q_target",
        "q_target_mask",
        "is_q_target",
        "target_valid",
        "use_for_objective",
        "score_q",
    ]

    bool_col = find_column(
        mask,
        bool_candidates,
        required=False,
        description="Q target mask boolean",
    )

    if bool_col is None:
        excluded = {time_col, block_col}

        possible = [
            c for c in mask.columns
            if c not in excluded
        ]

        for c in possible:
            values = (
                mask[c]
                .dropna()
                .astype(str)
                .str.lower()
                .unique()
            )

            if set(values).issubset(
                {
                    "0",
                    "1",
                    "true",
                    "false",
                    "yes",
                    "no",
                }
            ):
                bool_col = c
                break

    if time_col is None:
        raise RuntimeError(
            "Could not resolve q-target-mask timestamp column. "
            f"Columns={list(mask.columns)}"
        )

    mask["__time"] = pd.to_datetime(
        mask[time_col],
        utc=True,
        errors="raise",
    )

    if bool_col is not None:
        # Explicit boolean-mask schema.
        mask["__use"] = infer_bool_series(
            mask[bool_col]
        )
        mask_schema = f"EXPLICIT_BOOLEAN:{bool_col}"
    else:
        # final target-mask schema is a row-presence table:
        #
        #   interval_end_utc, phase, event_block_id, q_obs_m3s
        #
        # Each retained row is itself an eligible target hour. If an
        # observed-Q column is present, require it to be finite; otherwise
        # row presence alone means use=True.
        q_obs_col = find_column(
            mask,
            [
                "q_obs_m3s",
                "q_obs_mean_m3s",
                "q_mean_m3s",
            ],
            required=False,
            description="q-target observed discharge",
        )

        if q_obs_col is not None:
            mask["__use"] = np.isfinite(
                pd.to_numeric(
                    mask[q_obs_col],
                    errors="coerce",
                )
            )
            mask_schema = (
                f"ROW_PRESENCE_WITH_FINITE_Q:{q_obs_col}"
            )
        else:
            mask["__use"] = True
            mask_schema = "ROW_PRESENCE_ALL_ROWS"

    if int(mask["__use"].sum()) == 0:
        raise RuntimeError(
            "Q-target mask resolved successfully but contains "
            "zero usable target rows."
        )

    if block_col is not None:
        mask["__block"] = (
            mask[block_col].astype(str)
        )
    else:
        mask["__block"] = None

    return mask[
        ["__time", "__block", "__use"]
    ]


def prepare_inputs(
    forcing_path,
    blocks_path,
    mask_path,
    dynamic_path,
    routing_model,
    expected,
):
    forcing = pd.read_csv(forcing_path)

    time_col = find_column(
        forcing,
        [
            "interval_end_utc",
            "time_utc",
            "timestamp_utc",
        ],
        description="forcing timestamp",
    )

    q_col = find_column(
        forcing,
        [
            "q_obs_m3s",
            "q_obs_mean_m3s",
            "q_mean_m3s",
        ],
        description="observed discharge",
    )

    forcing["__time"] = pd.to_datetime(
        forcing[time_col],
        utc=True,
        errors="raise",
    )

    forcing[q_col] = pd.to_numeric(
        forcing[q_col],
        errors="coerce",
    )

    rain_map = rainfall_columns(forcing)

    missing_rain = [
        sid
        for sid in routing_model["ids"]
        if sid not in rain_map
    ]

    if missing_rain:
        raise RuntimeError(
            f"Missing rainfall columns for: {missing_rain}"
        )

    if len(rain_map) != expected:
        raise RuntimeError(
            f"Expected {expected} subcatchment rainfall columns; "
            f"resolved {len(rain_map)}."
        )

    dynamic = pd.read_csv(dynamic_path)

    dyn_time_col = find_column(
        dynamic,
        ["interval_end_utc"],
        description="dynamic-state timestamp",
    )

    dyn_sc_col = find_column(
        dynamic,
        ["subcatchment_id"],
        description="dynamic-state subcatchment ID",
    )

    for required in [
        "root_zone_relative_wetness",
        "et0_mm_h",
    ]:
        if required not in dynamic.columns:
            raise RuntimeError(
                f"Dynamic-state forcing missing {required}."
            )

    dynamic["__time"] = pd.to_datetime(
        dynamic[dyn_time_col],
        utc=True,
        errors="raise",
    )

    dynamic["__sc"] = dynamic[dyn_sc_col].map(
        normalize_sc_id
    )

    dynamic[
        "root_zone_relative_wetness"
    ] = pd.to_numeric(
        dynamic[
            "root_zone_relative_wetness"
        ],
        errors="coerce",
    )

    dynamic["et0_mm_h"] = pd.to_numeric(
        dynamic["et0_mm_h"],
        errors="coerce",
    )

    duplicate_dynamic = int(
        dynamic.duplicated(
            ["__time", "__sc"]
        ).sum()
    )

    if duplicate_dynamic:
        raise RuntimeError(
            f"Dynamic-state forcing has {duplicate_dynamic} "
            "duplicate time/subcatchment rows."
        )

    blocks = pd.read_csv(blocks_path)

    for c in [
        "block_start_utc",
        "block_end_utc",
    ]:
        blocks[c] = pd.to_datetime(
            blocks[c],
            utc=True,
            errors="raise",
        )

    blocks["event_block_id"] = (
        blocks["event_block_id"].astype(str)
    )

    qmask = load_q_target_mask(mask_path)

    return {
        "forcing": forcing,
        "forcing_time_col": time_col,
        "q_col": q_col,
        "rain_map": rain_map,
        "dynamic": dynamic,
        "blocks": blocks,
        "qmask": qmask,
    }


def qmask_for_block(qmask, block_id, times):
    if qmask["__block"].notna().any():
        m = qmask[
            qmask["__block"] == str(block_id)
        ]
    else:
        m = qmask

    lookup = dict(
        zip(
            m["__time"],
            m["__use"],
        )
    )

    return np.asarray(
        [
            bool(lookup.get(t, False))
            for t in times
        ],
        dtype=bool,
    )


def build_event_blocks(data, routing_model, warmup_h):
    forcing = data["forcing"]
    dynamic = data["dynamic"]
    blocks = data["blocks"]
    qmask = data["qmask"]
    q_col = data["q_col"]
    rain_map = data["rain_map"]

    ids = routing_model["ids"]
    event_objects = []

    dyn_by_time = {
        t: g.set_index("__sc")
        for t, g in dynamic.groupby("__time")
    }

    forcing_by_time = forcing.set_index("__time")

    for _, b in blocks.iterrows():
        block_id = str(b["event_block_id"])
        phase = str(b["phase"])
        start = b["block_start_utc"]
        end = b["block_end_utc"]

        times = pd.date_range(
            start=start,
            end=end,
            freq="h",
            tz="UTC",
        )

        missing_forcing = times.difference(
            forcing_by_time.index
        )

        if len(missing_forcing):
            raise RuntimeError(
                f"{block_id}: missing {len(missing_forcing)} "
                "rain/Q forcing hours."
            )

        f = forcing_by_time.loc[times]

        rain = np.column_stack(
            [
                pd.to_numeric(
                    f[rain_map[sid]],
                    errors="coerce",
                ).to_numpy(float)
                for sid in ids
            ]
        )

        if not np.isfinite(rain).all():
            bad = int((~np.isfinite(rain)).sum())
            raise RuntimeError(
                f"{block_id}: rainfall contains {bad} missing "
                "subcatchment-hour values."
            )

        qobs = pd.to_numeric(
            f[q_col],
            errors="coerce",
        ).to_numpy(float)

        wet = np.full(
            (len(times), len(ids)),
            np.nan,
            dtype=float,
        )

        et0 = np.full_like(
            wet,
            np.nan,
        )

        for ti, t in enumerate(times):
            if t not in dyn_by_time:
                raise RuntimeError(
                    f"{block_id}: missing ERA5 dynamic forcing at {t}."
                )

            dg = dyn_by_time[t]

            missing_sc = [
                sid
                for sid in ids
                if sid not in dg.index
            ]

            if missing_sc:
                raise RuntimeError(
                    f"{block_id}: ERA5 forcing at {t} missing "
                    f"subcatchments {missing_sc}."
                )

            wet[ti] = [
                float(
                    dg.loc[
                        sid,
                        "root_zone_relative_wetness",
                    ]
                )
                for sid in ids
            ]

            et0[ti] = [
                float(
                    dg.loc[
                        sid,
                        "et0_mm_h",
                    ]
                )
                for sid in ids
            ]

        if not np.isfinite(wet).all():
            raise RuntimeError(
                f"{block_id}: ERA5 relative wetness is incomplete."
            )

        if not np.isfinite(et0).all():
            raise RuntimeError(
                f"{block_id}: ERA5 ET0 is incomplete."
            )

        target = qmask_for_block(
            qmask,
            block_id,
            times,
        )

        # If mask schema was block-agnostic and returned no targets, use
        # observed Q availability. This fallback is deliberately narrow.
        if target.sum() == 0:
            target = np.isfinite(qobs)

        # Always require observed target to be finite.
        target &= np.isfinite(qobs)

        # Exclude first N hours after the ORIGINAL event-block start.
        if warmup_h > 0:
            target[:min(warmup_h, len(target))] = False

        if target.sum() < 24:
            raise RuntimeError(
                f"{block_id}: only {target.sum()} scored Q hours "
                "after warm-up/masking."
            )

        event_objects.append(
            {
                "event_block_id": block_id,
                "phase": phase,
                "start": start,
                "end": end,
                "times": times,
                "rain": rain,
                "qobs": qobs,
                "target": target,
                "era5_wetness": wet,
                "et0": et0,
            }
        )

    return event_objects


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

PARAM_NAMES = [
    "soil_capacity_mm",
    "field_capacity_fraction",
    "runoff_beta",
    "soil_drainage_time_h",
    "quick_reservoir_time_h",
    "base_reservoir_time_h",
    "et_multiplier",
    "et_moisture_exponent",
    "channel_velocity_m_s",
]

PARAM_BOUNDS = [
    (80.0, 700.0),     # soil capacity mm
    (0.20, 0.85),      # field-capacity fraction
    (0.5, 12.0),       # runoff beta
    (48.0, 1200.0),    # soil drainage time h
    (2.0, 96.0),       # quick reservoir h
    (120.0, 2400.0),   # base reservoir h
    (0.20, 2.00),      # ET multiplier
    (0.25, 3.00),      # ET moisture exponent
    (0.20, 2.50),      # channel velocity m/s
]


def unpack_params(x):
    return dict(
        zip(
            PARAM_NAMES,
            map(float, x),
        )
    )


def reservoir_step(store, input_depth, k_h):
    """
    Exact-ish discrete first-order reservoir over dt=1h using:
        out = store * (1-exp(-1/K)) + input contribution
    We use a simple mass-conservative end-of-step form:
        store += input
        out = store * (1-exp(-1/K))
        store -= out
    """
    store = store + input_depth

    alpha = (
        1.0
        - math.exp(
            -1.0 / max(k_h, 1e-6)
        )
    )

    out = store * alpha
    store = store - out

    return store, out


def simulate_event(event, x, routing_model, return_states=False):
    p = unpack_params(x)

    ids = routing_model["ids"]
    order = routing_model["order"]
    pos = routing_model["position"]
    upstream_of = routing_model["upstream_of"]
    outlet = routing_model["outlet"]

    n = len(ids)
    nt = len(event["times"])

    area = np.array(
        [
            routing_model["area_km2"][sid]
            for sid in ids
        ],
        dtype=float,
    )

    segment_km = np.array(
        [
            routing_model["segment_km"][sid]
            for sid in ids
        ],
        dtype=float,
    )

    velocity = p["channel_velocity_m_s"]

    channel_k_h = np.maximum(
        segment_km * 1000.0
        / velocity
        / 3600.0,
        0.25,
    )

    C = p["soil_capacity_mm"]
    fc = p["field_capacity_fraction"]
    runoff_beta = p["runoff_beta"]
    k_drain = p["soil_drainage_time_h"]
    k_quick = p["quick_reservoir_time_h"]
    k_base = p["base_reservoir_time_h"]
    et_mult = p["et_multiplier"]
    et_exp = p["et_moisture_exponent"]

    # External state is used only here.
    initial_wetness = np.clip(
        event["era5_wetness"][0],
        0.0,
        1.0,
    )

    soil = initial_wetness * C
    quick_store = np.zeros(n, dtype=float)

    # Initial base reservoir equilibrium with initial soil drainage.
    threshold = fc * C
    initial_recharge = np.maximum(
        soil - threshold,
        0.0,
    ) / k_drain

    base_store = (
        initial_recharge * k_base
    )

    channel_store_m3 = np.zeros(n, dtype=float)

    qout = np.zeros(nt, dtype=float)

    if return_states:
        soil_hist = np.zeros((nt, n), dtype=float)
        aet_hist = np.zeros((nt, n), dtype=float)
        quick_hist = np.zeros((nt, n), dtype=float)
        recharge_hist = np.zeros((nt, n), dtype=float)

    # Convert 1 mm/h over km2 to m3/s:
    # 0.001 m * 1e6 m2 / 3600 s = 0.277777...
    mmh_km2_to_m3s = (
        area * 1000.0 / 3600.0
    )

    order_idx = [
        pos[sid]
        for sid in order
    ]

    outlet_idx = pos[outlet]

    for t in range(nt):
        P = np.maximum(
            event["rain"][t],
            0.0,
        )

        et0 = np.maximum(
            event["et0"][t],
            0.0,
        )

        relative_soil = np.clip(
            soil / C,
            0.0,
            1.0,
        )

        moisture_limitation = np.power(
            relative_soil,
            et_exp,
        )

        aet = np.minimum(
            soil,
            et0 * et_mult * moisture_limitation,
        )

        soil = np.maximum(
            soil - aet,
            0.0,
        )

        relative_soil = np.clip(
            soil / C,
            0.0,
            1.0,
        )

        quick_fraction = np.power(
            relative_soil,
            runoff_beta,
        )

        quick_input = (
            P * quick_fraction
        )

        infiltration = (
            P - quick_input
        )

        soil = soil + infiltration

        overflow = np.maximum(
            soil - C,
            0.0,
        )

        soil = np.minimum(
            soil,
            C,
        )

        quick_input = (
            quick_input + overflow
        )

        recharge = np.maximum(
            soil - threshold,
            0.0,
        ) / k_drain

        recharge = np.minimum(
            recharge,
            soil,
        )

        soil = np.maximum(
            soil - recharge,
            0.0,
        )

        local_q_m3s = np.zeros(
            n,
            dtype=float,
        )

        for i in range(n):
            quick_store[i], quick_out = reservoir_step(
                quick_store[i],
                quick_input[i],
                k_quick,
            )

            base_store[i], base_out = reservoir_step(
                base_store[i],
                recharge[i],
                k_base,
            )

            local_q_m3s[i] = (
                (quick_out + base_out)
                * mmh_km2_to_m3s[i]
            )

        routed_out = np.zeros(
            n,
            dtype=float,
        )

        for i in order_idx:
            sid = ids[i]

            upstream_q = 0.0

            for up_sid in upstream_of[sid]:
                upstream_q += routed_out[
                    pos[up_sid]
                ]

            inflow_q = (
                local_q_m3s[i]
                + upstream_q
            )

            # Channel reservoir in m3.
            channel_store_m3[i] += (
                inflow_q * 3600.0
            )

            alpha = (
                1.0
                - math.exp(
                    -1.0
                    / max(
                        channel_k_h[i],
                        1e-6,
                    )
                )
            )

            released_m3 = (
                channel_store_m3[i]
                * alpha
            )

            channel_store_m3[i] -= (
                released_m3
            )

            routed_out[i] = (
                released_m3 / 3600.0
            )

        qout[t] = routed_out[
            outlet_idx
        ]

        if return_states:
            soil_hist[t] = soil
            aet_hist[t] = aet
            quick_hist[t] = quick_input
            recharge_hist[t] = recharge

    result = {
        "q_sim_m3s": qout,
    }

    if return_states:
        result.update(
            {
                "soil_mm": soil_hist,
                "aet_mm_h": aet_hist,
                "quick_input_mm_h": quick_hist,
                "recharge_mm_h": recharge_hist,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Objective / reporting
# ---------------------------------------------------------------------------

def evaluate_events(events, x, routing_model):
    rows = []
    pooled_obs = []
    pooled_sim = []

    for event in events:
        sim = simulate_event(
            event,
            x,
            routing_model,
        )["q_sim_m3s"]

        target = event["target"]

        obs = event["qobs"][target]
        pred = sim[target]

        metrics = metric_bundle(
            obs,
            pred,
        )

        row = {
            "event_block_id": event["event_block_id"],
            "phase": event["phase"],
            "scored_hours": int(target.sum()),
            **metrics,
        }

        # Event-specific peak timing audit.  There is intentionally no
        # +/-96 h or other clipping here: reported offsets are the actual
        # argmax-to-argmax timing differences over scored target hours.
        valid_idx = np.where(target)[0]

        if len(valid_idx):
            local_obs = event["qobs"][valid_idx]
            local_sim = sim[valid_idx]

            obs_peak_i = valid_idx[
                int(np.nanargmax(local_obs))
            ]

            sim_peak_i = valid_idx[
                int(np.nanargmax(local_sim))
            ]

            obs_peak_time = event["times"][obs_peak_i]
            sim_peak_time = event["times"][sim_peak_i]
            scoring_start = event["times"][valid_idx[0]]
            scoring_end = event["times"][valid_idx[-1]]

            row["observed_peak_time_utc"] = obs_peak_time
            row["simulated_peak_time_utc"] = sim_peak_time
            row["scoring_start_utc"] = scoring_start
            row["scoring_end_utc"] = scoring_end
            row["observed_peak_q_m3s"] = float(event["qobs"][obs_peak_i])
            row["simulated_peak_q_m3s"] = float(sim[sim_peak_i])
            row["peak_timing_error_h"] = float(
                (sim_peak_time - obs_peak_time)
                / pd.Timedelta(hours=1)
            )
            row["observed_peak_hours_from_scoring_start"] = float(
                (obs_peak_time - scoring_start) / pd.Timedelta(hours=1)
            )
            row["simulated_peak_hours_from_scoring_start"] = float(
                (sim_peak_time - scoring_start) / pd.Timedelta(hours=1)
            )
            row["observed_peak_at_scoring_boundary"] = bool(
                obs_peak_i == valid_idx[0] or obs_peak_i == valid_idx[-1]
            )
            row["simulated_peak_at_scoring_boundary"] = bool(
                sim_peak_i == valid_idx[0] or sim_peak_i == valid_idx[-1]
            )
        else:
            row["observed_peak_time_utc"] = pd.NaT
            row["simulated_peak_time_utc"] = pd.NaT
            row["scoring_start_utc"] = pd.NaT
            row["scoring_end_utc"] = pd.NaT
            row["observed_peak_q_m3s"] = np.nan
            row["simulated_peak_q_m3s"] = np.nan
            row["peak_timing_error_h"] = np.nan
            row["observed_peak_hours_from_scoring_start"] = np.nan
            row["simulated_peak_hours_from_scoring_start"] = np.nan
            row["observed_peak_at_scoring_boundary"] = False
            row["simulated_peak_at_scoring_boundary"] = False

        rows.append(row)
        pooled_obs.append(obs)
        pooled_sim.append(pred)

    pooled_obs = np.concatenate(pooled_obs)
    pooled_sim = np.concatenate(pooled_sim)

    pooled_metrics = metric_bundle(
        pooled_obs,
        pooled_sim,
    )

    return pd.DataFrame(rows), pooled_metrics


def objective_factory(
    cal_events,
    routing_model,
    trace,
    *,
    block_weight,
    pooled_weight,
    metric_weights,
):
    counter = {"n": 0}

    total_mix = float(block_weight + pooled_weight)
    if not np.isfinite(total_mix) or total_mix <= 0:
        raise ValueError("Objective block/pooled weights must sum to a positive value.")
    block_weight = float(block_weight) / total_mix
    pooled_weight = float(pooled_weight) / total_mix

    def objective(x):
        counter["n"] += 1

        try:
            rows, pooled = evaluate_events(
                cal_events,
                x,
                routing_model,
            )

            block_penalties = [
                metric_penalty(r, metric_weights)
                for r in rows.to_dict("records")
            ]

            block_mean = float(
                np.mean(block_penalties)
            )

            pooled_penalty = metric_penalty(
                pooled,
                metric_weights,
            )

            value = (
                block_weight * block_mean
                + pooled_weight * pooled_penalty
            )

            if not np.isfinite(value):
                value = 1e6

        except Exception:
            value = 1e6

        trace.append(
            {
                "evaluation": counter["n"],
                "objective": float(value),
            }
        )

        return float(value)

    return objective


def near_bounds(x, bounds, fraction=0.01):
    rows = []
    count = 0

    for name, value, (low, high) in zip(
        PARAM_NAMES,
        x,
        bounds,
    ):
        span = high - low
        lower_frac = (
            (value - low) / span
        )
        upper_frac = (
            (high - value) / span
        )

        near = (
            lower_frac <= fraction
            or upper_frac <= fraction
        )

        if near:
            count += 1

        if lower_frac <= fraction:
            edge = "LOWER"
        elif upper_frac <= fraction:
            edge = "UPPER"
        else:
            edge = ""

        rows.append(
            {
                "parameter": name,
                "value": float(value),
                "lower_bound": low,
                "upper_bound": high,
                "near_bound": bool(near),
                "near_edge": edge,
            }
        )

    return count, pd.DataFrame(rows)


def main():
    a = parse_args()

    a.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = pd.read_csv(
        a.routing_features
    )

    routing = pd.read_csv(
        a.routing
    )

    routing_model = build_routing(
        features,
        routing,
        a.expected_subcatchments,
    )

    data = prepare_inputs(
        a.forcing,
        a.event_blocks,
        a.q_target_mask,
        a.dynamic_state_et0,
        routing_model,
        a.expected_subcatchments,
    )

    events = build_event_blocks(
        data,
        routing_model,
        a.objective_warmup_hours,
    )

    cal_events = [
        e for e in events
        if e["phase"].startswith(
            "CALIBRATION"
        )
    ]

    val_events = [
        e for e in events
        if e["phase"].startswith(
            "TEMPORAL_VALIDATION"
        )
    ]

    if len(cal_events) == 0:
        raise RuntimeError(
            "No calibration event blocks."
        )

    if len(val_events) == 0:
        raise RuntimeError(
            "No validation event blocks."
        )

    # Hard Florence guard.
    if max(
        e["end"]
        for e in events
    ) >= pd.Timestamp(
        "2018-01-01T00:00:00Z"
    ):
        raise RuntimeError(
            "Development event library includes 2018+ data."
        )

    print("=" * 100)
    print(
        f"SCRIPT BUILD                       : {BUILD}"
    )
    print(
        "PHYSICS V2 - "
        "ERA5 STATE + DYNAMIC ET SEMI-DISTRIBUTED MODEL"
    )
    print("=" * 100)
    print(
        f"Subcatchments                      : {len(routing_model['ids'])}"
    )
    print(
        f"Routing edges                      : {len(routing_model['edges'])}"
    )
    print(
        f"Basin outlet                       : {routing_model['outlet']}"
    )
    print(
        f"Channel segment method             : {routing_model['segment_method']}"
    )

    seg_stats = routing_model["segment_distance_stats"]

    print(
        f"Channel segment km min/median/max  : "
        f"{seg_stats['min_km']:.3f} / "
        f"{seg_stats['median_km']:.3f} / "
        f"{seg_stats['max_km']:.3f}"
    )

    cumulative_stats = routing_model.get(
        "cumulative_distance_stats"
    )

    if cumulative_stats is not None:
        print(
            f"Cumulative distance field           : "
            f"{cumulative_stats['column']}"
        )
        print(
            f"Cumulative km min/median/max        : "
            f"{cumulative_stats['min_km']:.3f} / "
            f"{cumulative_stats['median_km']:.3f} / "
            f"{cumulative_stats['max_km']:.3f}"
        )
    print(
        f"Calibration blocks                 : {len(cal_events)}"
    )
    print(
        f"Validation blocks                  : {len(val_events)}"
    )
    print(
        f"Objective warm-up                  : {a.objective_warmup_hours} h"
    )
    print(
        "Initial soil state                : ERA5-Land event-start wetness"
    )
    print(
        "Initial quick store               : 0 mm"
    )
    print(
        "Initial base store                : equilibrium with initial recharge"
    )
    print(
        "Soil-water atmospheric loss       : hourly ET0 driven"
    )
    print(
        "Observed Q used for state          : NO"
    )
    print(
        "Temporal interpolation             : NONE"
    )
    print(
        "Florence 2018 used                 : NO"
    )
    metric_weights = {
        "nse": a.metric_weight_nse,
        "kge": a.metric_weight_kge,
        "log_nse": a.metric_weight_log_nse,
        "pbias": a.metric_weight_pbias,
        "peak": a.metric_weight_peak,
    }

    print(
        f"Optimizer maxiter/popsize          : "
        f"{a.optimizer_maxiter}/{a.optimizer_popsize}"
    )
    print(
        f"Objective block/pooled weights     : "
        f"{a.objective_block_weight:.3f}/{a.objective_pooled_weight:.3f}"
    )
    print(
        "Metric weights NSE/KGE/log/PBIAS/peak: "
        f"{a.metric_weight_nse:.3f}/"
        f"{a.metric_weight_kge:.3f}/"
        f"{a.metric_weight_log_nse:.3f}/"
        f"{a.metric_weight_pbias:.3f}/"
        f"{a.metric_weight_peak:.3f}"
    )
    print()

    print("EVENT INVENTORY")
    print("-" * 100)

    for e in events:
        label = (
            "CAL"
            if e["phase"].startswith(
                "CALIBRATION"
            )
            else "VAL"
        )

        mean_wet = float(
            np.mean(
                e["era5_wetness"][0]
            )
        )

        event_et0 = float(
            np.mean(
                np.sum(
                    e["et0"],
                    axis=0,
                )
            )
        )

        print(
            f"{label} {e['event_block_id']} | "
            f"hours={len(e['times'])} | "
            f"targets={int(e['target'].sum())} | "
            f"initial ERA5 wetness mean={mean_wet:.3f} | "
            f"mean-SC event ET0={event_et0:.1f} mm"
        )

    print()
    print("OPTIMIZATION")
    print("-" * 100)

    trace = []
    objective = objective_factory(
        cal_events,
        routing_model,
        trace,
        block_weight=a.objective_block_weight,
        pooled_weight=a.objective_pooled_weight,
        metric_weights=metric_weights,
    )

    generation_rows = []
    best_seen = {"value": np.inf}

    # scipy callback receives xk and convergence for the legacy callback
    # signature used by differential_evolution.
    generation = {"n": 0}

    def callback(xk, convergence):
        generation["n"] += 1
        value = objective(xk)

        if value < best_seen["value"]:
            best_seen["value"] = value

        generation_rows.append(
            {
                "generation": generation["n"],
                "objective": value,
                "convergence": float(convergence),
            }
        )

        print(
            f"Optimizer generation "
            f"{generation['n']:02d} | "
            f"objective={value:.6f} | "
            f"convergence={float(convergence):.6f}"
        )

        return False

    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
        seed=a.seed,
        maxiter=a.optimizer_maxiter,
        popsize=a.optimizer_popsize,
        tol=a.optimizer_tol,
        polish=a.optimizer_polish,
        updating="immediate",
        workers=1,
        callback=callback,
    )

    xbest = np.asarray(
        result.x,
        dtype=float,
    )

    final_objective = float(
        objective(xbest)
    )

    param_dict = unpack_params(
        xbest
    )

    bound_count, bound_df = near_bounds(
        xbest,
        PARAM_BOUNDS,
    )

    cal_block_metrics, cal_pooled = evaluate_events(
        cal_events,
        xbest,
        routing_model,
    )

    val_block_metrics, val_pooled = evaluate_events(
        val_events,
        xbest,
        routing_model,
    )

    # Build hourly output for all development events.
    hourly_rows = []

    for e in events:
        sim = simulate_event(
            e,
            xbest,
            routing_model,
            return_states=False,
        )["q_sim_m3s"]

        for i, t in enumerate(e["times"]):
            hourly_rows.append(
                {
                    "event_block_id": e["event_block_id"],
                    "phase": e["phase"],
                    "interval_end_utc": t,
                    "q_obs_m3s": (
                        float(e["qobs"][i])
                        if np.isfinite(e["qobs"][i])
                        else np.nan
                    ),
                    "q_sim_m3s": float(sim[i]),
                    "q_target_scored": bool(
                        e["target"][i]
                    ),
                    "basin_mean_era5_relative_wetness": float(
                        np.mean(
                            e["era5_wetness"][i]
                        )
                    ),
                    "basin_mean_et0_mm_h": float(
                        np.mean(
                            e["et0"][i]
                        )
                    ),
                }
            )

    hourly_df = pd.DataFrame(
        hourly_rows
    )

    # Readiness gates.
    quality_failures = 0
    qc_rows = []

    gates = [
        (
            "validation_nse",
            val_pooled["nse"],
            a.min_validation_nse,
            ">=",
        ),
        (
            "validation_kge",
            val_pooled["kge"],
            a.min_validation_kge,
            ">=",
        ),
        (
            "validation_abs_pbias_percent",
            abs(
                val_pooled["pbias_percent"]
            ),
            a.max_validation_abs_pbias_percent,
            "<=",
        ),
    ]

    for name, value, threshold, op in gates:
        if op == ">=":
            passed = (
                np.isfinite(value)
                and value >= threshold
            )
        else:
            passed = (
                np.isfinite(value)
                and value <= threshold
            )

        if not passed:
            quality_failures += 1

        qc_rows.append(
            {
                "check": name,
                "value": value,
                "threshold": f"{op}{threshold}",
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
        )

    # Parameter-bound pressure is diagnostic, not an automatic failure.
    qc_rows.append(
        {
            "check": "parameters_near_bounds",
            "value": bound_count,
            "threshold": (
                "diagnostic only; validation gates control readiness"
            ),
            "status": (
                "WARN"
                if bound_count >= 3
                else "PASS"
            ),
        }
    )

    warnings = (
        1 if bound_count >= 3 else 0
    )

    blocking_failures = 0

    status = (
        "PASS_PHYSICS_V2_VALIDATION"
        if quality_failures == 0
        else "FAIL_PHYSICS_V2_VALIDATION_QUALITY"
    )

    safe_florence = (
        quality_failures == 0
        and blocking_failures == 0
    )

    # Outputs.
    params_path = (
        a.output_dir
        / "physics_v2_parameters.json"
    )

    hourly_path = (
        a.output_dir
        / "physics_v2_hourly.csv"
    )

    block_path = (
        a.output_dir
        / "physics_v2_block_metrics.csv"
    )

    summary_path = (
        a.output_dir
        / "physics_v2_summary_metrics.csv"
    )

    trace_path = (
        a.output_dir
        / "physics_v2_optimizer_trace.csv"
    )

    generation_path = (
        a.output_dir
        / "physics_v2_optimizer_generations.csv"
    )

    bounds_path = (
        a.output_dir
        / "physics_v2_parameter_bounds.csv"
    )

    qc_path = (
        a.output_dir
        / "physics_v2_qc.csv"
    )

    timing_audit_path = (
        a.output_dir
        / "physics_v2_peak_timing_audit.csv"
    )

    metadata_path = (
        a.output_dir
        / "physics_v2_metadata.json"
    )

    if params_path.exists() and not a.overwrite:
        raise FileExistsError(
            f"{params_path} exists. Use --overwrite."
        )

    params_payload = {
        "script_build": BUILD,
        "parameters": param_dict,
        "objective_value": final_objective,
        "optimizer_success": bool(
            result.success
        ),
        "optimizer_message": str(
            result.message
        ),
        "optimizer_nfev": int(
            result.nfev
        ),
        "parameters_near_bounds": int(
            bound_count
        ),
    }

    atomic_json(
        params_payload,
        params_path,
    )

    atomic_csv(
        hourly_df,
        hourly_path,
    )

    block_combined = pd.concat(
        [
            cal_block_metrics,
            val_block_metrics,
        ],
        ignore_index=True,
    )

    atomic_csv(
        block_combined,
        block_path,
    )

    summary_df = pd.DataFrame(
        [
            {
                "phase": "CALIBRATION_2015_2016",
                **cal_pooled,
            },
            {
                "phase": "VALIDATION_2017",
                **val_pooled,
            },
        ]
    )

    atomic_csv(
        summary_df,
        summary_path,
    )

    atomic_csv(
        pd.DataFrame(trace),
        trace_path,
    )

    atomic_csv(
        pd.DataFrame(generation_rows),
        generation_path,
    )

    atomic_csv(
        bound_df,
        bounds_path,
    )

    atomic_csv(
        pd.DataFrame(qc_rows),
        qc_path,
    )

    timing_cols = [
        "event_block_id",
        "phase",
        "scored_hours",
        "observed_peak_time_utc",
        "simulated_peak_time_utc",
        "observed_peak_q_m3s",
        "simulated_peak_q_m3s",
        "peak_timing_error_h",
        "scoring_start_utc",
        "scoring_end_utc",
        "observed_peak_hours_from_scoring_start",
        "simulated_peak_hours_from_scoring_start",
        "observed_peak_at_scoring_boundary",
        "simulated_peak_at_scoring_boundary",
    ]
    atomic_csv(
        block_combined[timing_cols],
        timing_audit_path,
    )

    metadata = {
        "script_build": BUILD,
        "status": status,
        "development_protocol": {
            "calibration": "2015-2016 accepted event blocks",
            "validation": "2017 accepted event blocks",
            "florence_used": False,
        },
        "forcing": {
            "rainfall": "MRMS spatial 37-subcatchment rainfall",
            "initial_soil_state": (
                "ERA5-Land calibration-scaled event-start relative wetness"
            ),
            "soil_loss": (
                "hourly FAO-56 ET0 multiplied by calibrated ET coefficient "
                "and conceptual soil-moisture limitation"
            ),
            "observed_q_used_for_initialization": False,
            "temporal_interpolation": False,
        },
        "model_structure": {
            "subcatchments": len(
                routing_model["ids"]
            ),
            "routing_edges": len(
                routing_model["edges"]
            ),
            "outlet": routing_model["outlet"],
            "channel_segment_method": routing_model[
                "segment_method"
            ],
            "channel_segment_distance_stats_km": routing_model[
                "segment_distance_stats"
            ],
            "cumulative_network_distance_stats_km": routing_model.get(
                "cumulative_distance_stats"
            ),
            "initial_quick_store_mm": 0.0,
            "initial_base_store": (
                "equilibrium with initial soil drainage rate"
            ),
            "objective_warmup_hours": (
                a.objective_warmup_hours
            ),
            "calibration_objective": {
                "block_weight": a.objective_block_weight,
                "pooled_weight": a.objective_pooled_weight,
                "metric_weights": metric_weights,
                "policy": (
                    "equal accepted-block penalties plus a small pooled regularizer; "
                    "validation and Florence are not used in optimization"
                ),
            },
        },
        "validation_gates": {
            "minimum_nse": a.min_validation_nse,
            "minimum_kge": a.min_validation_kge,
            "maximum_absolute_pbias_percent": (
                a.max_validation_abs_pbias_percent
            ),
        },
        "blocking_failures": blocking_failures,
        "quality_failures": quality_failures,
        "warnings": warnings,
        "safe_for_untouched_florence_test": (
            bool(safe_florence)
        ),
        "outputs": {
            "parameters": str(params_path),
            "hourly": str(hourly_path),
            "block_metrics": str(block_path),
            "summary_metrics": str(summary_path),
            "optimizer_trace": str(trace_path),
            "optimizer_generations": str(
                generation_path
            ),
            "parameter_bounds": str(bounds_path),
            "qc": str(qc_path),
            "peak_timing_audit": str(timing_audit_path),
        },
    }

    atomic_json(
        metadata,
        metadata_path,
    )

    # Console report.
    print()
    print("OPTIMIZED PARAMETERS")
    print("-" * 100)

    for name in PARAM_NAMES:
        print(
            f"{name:35s}: "
            f"{param_dict[name]:.6f}"
        )

    print(
        f"{'objective_value':35s}: "
        f"{final_objective:.6f}"
    )
    print(
        f"{'optimizer_success':35s}: "
        f"{bool(result.success)}"
    )
    print(
        f"{'parameters_near_bounds':35s}: "
        f"{bound_count}"
    )

    print()
    print("CALIBRATION METRICS - 2015/2016")
    print("-" * 100)
    print(
        f"NSE                                : "
        f"{cal_pooled['nse']:.6f}"
    )
    print(
        f"KGE                                : "
        f"{cal_pooled['kge']:.6f}"
    )
    print(
        f"log-NSE                            : "
        f"{cal_pooled['log_nse']:.6f}"
    )
    print(
        f"RMSE                               : "
        f"{cal_pooled['rmse_m3s']:.3f} m³/s"
    )
    print(
        f"MAE                                : "
        f"{cal_pooled['mae_m3s']:.3f} m³/s"
    )
    print(
        f"PBIAS                              : "
        f"{cal_pooled['pbias_percent']:.3f} %"
    )
    print(
        f"Peak error                         : "
        f"{cal_pooled['peak_error_percent']:.3f} %"
    )

    print()
    print("VALIDATION METRICS - 2017")
    print("-" * 100)
    print(
        f"NSE                                : "
        f"{val_pooled['nse']:.6f}"
    )
    print(
        f"KGE                                : "
        f"{val_pooled['kge']:.6f}"
    )
    print(
        f"log-NSE                            : "
        f"{val_pooled['log_nse']:.6f}"
    )
    print(
        f"RMSE                               : "
        f"{val_pooled['rmse_m3s']:.3f} m³/s"
    )
    print(
        f"MAE                                : "
        f"{val_pooled['mae_m3s']:.3f} m³/s"
    )
    print(
        f"PBIAS                              : "
        f"{val_pooled['pbias_percent']:.3f} %"
    )
    print(
        f"Peak error                         : "
        f"{val_pooled['peak_error_percent']:.3f} %"
    )

    print()
    print("VALIDATION BLOCK PEAK TIMING")
    print("-" * 100)

    for _, r in val_block_metrics.iterrows():
        print(
            f"{r['event_block_id']} | "
            f"NSE={r['nse']:.3f} | "
            f"KGE={r['kge']:.3f} | "
            f"PBIAS={r['pbias_percent']:+.1f}% | "
            f"peak timing={r['peak_timing_error_h']:+.1f} h"
        )

    print()
    print("READINESS")
    print("-" * 100)
    print(
        f"Blocking failures                  : "
        f"{blocking_failures}"
    )
    print(
        f"Quality failures                   : "
        f"{quality_failures}"
    )
    print(
        f"Warnings                           : "
        f"{warnings}"
    )
    print(
        f"Safe for untouched Florence test   : "
        f"{'YES' if safe_florence else 'NO'}"
    )
    print(
        f"Status                             : "
        f"{status}"
    )
    print(
        f"Parameters                         : "
        f"{params_path}"
    )
    print(
        f"Summary metrics                    : "
        f"{summary_path}"
    )
    print(
        f"Block metrics                      : "
        f"{block_path}"
    )
    print(
        f"Peak timing audit                  : "
        f"{timing_audit_path}"
    )
    print(
        f"Metadata                           : "
        f"{metadata_path}"
    )

    if not safe_florence:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
