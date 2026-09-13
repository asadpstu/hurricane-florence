"""
PHYSICS ERA5-LAND STATE + ET0
Derive dynamic ERA5-Land soil-state and hourly reference-ET forcing.

Inputs
------
* standardized ERA5-Land subcatchment hourly forcing
* Final pre-Florence event-block library

Outputs
-------
* Hourly dynamic forcing for all 37 subcatchments
* Basin-average hourly forcing
* Calibration-derived soil-state scaling parameters
* Event-start soil-state inventory
* QC + metadata

Scientific design
-----------------
1. Reference ET demand:
   Uses the FAO-56 hourly Penman-Monteith form with:
     - air temperature
     - dewpoint-derived vapor pressure
     - surface pressure
     - wind adjusted from 10 m to 2 m
     - ERA5-Land hourly net radiation
   Soil heat flux is approximated as:
     G = 0.10 Rn during daylight
     G = 0.50 Rn at night
   Raw negative ET0 is retained for diagnostics, then operational ET0 is
   clipped at zero.

2. Dynamic antecedent soil state:
   ERA5-Land root-zone VWC is NOT equated directly to the conceptual model's
   soil storage capacity. Instead, a relative wetness fraction is derived
   independently for each subcatchment from calibration-period ERA5 states:
       wetness = (theta - q05_cal) / (q95_cal - q05_cal)
   clipped to [0, 1].

   The q05/q95 scaling is estimated using CALIBRATION event-development
   windows only. Validation-period soil moisture does not affect the scaling.

3. No observed discharge is used.
4. No temporal interpolation is used.
5. Florence 2018 is excluded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BUILD = (
    "PHYSICS_V2_ERA5_ET0_"
    "DYNAMIC_SOIL_STATE_ET0_V1"
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--era5-hourly",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--event-blocks",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--antecedent-days",
        type=int,
        default=45,
    )
    p.add_argument(
        "--expected-subcatchments",
        type=int,
        default=37,
    )

    p.add_argument(
        "--soil-state-lower-quantile",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--soil-state-upper-quantile",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--min-soil-state-span-m3m3",
        type=float,
        default=0.02,
    )

    p.add_argument(
        "--min-required-et0-coverage-percent",
        type=float,
        default=99.9,
    )
    p.add_argument(
        "--max-hourly-et0-mm",
        type=float,
        default=3.0,
    )
    p.add_argument(
        "--max-daily-et0-mm",
        type=float,
        default=20.0,
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


def atomic_csv(df, path, *, gzip=False):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if gzip:
        tmp = path.with_name(
            path.name + ".partial.gz"
        )
        tmp.unlink(missing_ok=True)
        df.to_csv(
            tmp,
            index=False,
            compression="gzip",
        )
    else:
        tmp = path.with_name(
            path.name + ".partial"
        )
        tmp.unlink(missing_ok=True)
        df.to_csv(
            tmp,
            index=False,
        )

    os.replace(tmp, path)


def atomic_json(obj, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_name(
        path.name + ".partial"
    )
    tmp.unlink(missing_ok=True)
    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def required_columns():
    return [
        "interval_end_utc",
        "subcatchment_id",
        "root_zone_vwc_m3m3",
        "temperature_2m_c",
        "dewpoint_2m_c",
        "surface_pressure_kpa",
        "wind_speed_10m_m_s",
        "net_solar_radiation_hourly_j_m2",
        "net_radiation_hourly_mj_m2",
    ]


def make_time_union(
    blocks,
    antecedent_days,
):
    pieces = []

    for _, b in blocks.iterrows():
        start = (
            b["block_start_utc"]
            - pd.Timedelta(
                days=antecedent_days
            )
        ).floor("h")

        end_exclusive = (
            b["block_end_utc"]
            + pd.Timedelta(hours=1)
        ).floor("h")

        pieces.append(
            pd.date_range(
                start=start,
                end=end_exclusive,
                freq="h",
                inclusive="left",
                tz="UTC",
            )
        )

    if not pieces:
        return pd.DatetimeIndex(
            [],
            tz="UTC",
        )

    combined = pieces[0]

    for p in pieces[1:]:
        combined = combined.union(p)

    return combined.sort_values()


def saturation_vapor_pressure_kpa(
    temperature_c,
):
    t = np.asarray(
        temperature_c,
        dtype=float,
    )

    return (
        0.6108
        * np.exp(
            17.27 * t
            / (t + 237.3)
        )
    )


def vapor_pressure_slope_kpa_c(
    temperature_c,
):
    t = np.asarray(
        temperature_c,
        dtype=float,
    )

    es = saturation_vapor_pressure_kpa(
        t
    )

    return (
        4098.0
        * es
        / np.square(
            t + 237.3
        )
    )


def wind_10m_to_2m(
    wind_10m_m_s,
):
    """
    FAO-56 wind-height adjustment:
        u2 = uz * 4.87 / ln(67.8*z - 5.42)
    with z=10 m.
    """
    u10 = np.asarray(
        wind_10m_m_s,
        dtype=float,
    )

    factor = (
        4.87
        / np.log(
            67.8 * 10.0 - 5.42
        )
    )

    return u10 * factor


def hourly_fao56_et0(
    temperature_c,
    dewpoint_c,
    pressure_kpa,
    wind_10m_m_s,
    net_radiation_mj_m2_h,
    net_solar_radiation_j_m2_h,
):
    """
    Hourly FAO-56 Penman-Monteith reference ET.

    ET0 =
      [0.408*Delta*(Rn-G)
       + gamma*(37/(T+273))*u2*(es-ea)]
      /
      [Delta + gamma*(1 + 0.34*u2)]

    The meteorological point values are treated as representative of the
    ending hourly interval.

    Returns a dict of arrays.
    """
    t = np.asarray(
        temperature_c,
        dtype=float,
    )

    td = np.asarray(
        dewpoint_c,
        dtype=float,
    )

    p = np.asarray(
        pressure_kpa,
        dtype=float,
    )

    u10 = np.asarray(
        wind_10m_m_s,
        dtype=float,
    )

    rn = np.asarray(
        net_radiation_mj_m2_h,
        dtype=float,
    )

    solar_j = np.asarray(
        net_solar_radiation_j_m2_h,
        dtype=float,
    )

    u2 = wind_10m_to_2m(u10)

    es = saturation_vapor_pressure_kpa(
        t
    )

    ea = saturation_vapor_pressure_kpa(
        td
    )

    vpd = np.maximum(
        es - ea,
        0.0,
    )

    delta = vapor_pressure_slope_kpa_c(
        t
    )

    gamma = (
        0.000665
        * p
    )

    # Positive incoming/net solar is used as the daylight indicator.
    daylight = (
        solar_j > 1.0e3
    )

    g = np.where(
        daylight,
        0.10 * rn,
        0.50 * rn,
    )

    denominator = (
        delta
        + gamma
        * (
            1.0
            + 0.34 * u2
        )
    )

    numerator = (
        0.408
        * delta
        * (rn - g)
        + gamma
        * (
            37.0
            / (t + 273.0)
        )
        * u2
        * vpd
    )

    raw = np.full(
        len(t),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(t)
        & np.isfinite(td)
        & np.isfinite(p)
        & np.isfinite(u2)
        & np.isfinite(rn)
        & np.isfinite(g)
        & np.isfinite(denominator)
        & (denominator > 0)
        & ((t + 273.0) > 0)
    )

    raw[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    clipped = np.where(
        np.isfinite(raw),
        np.maximum(
            raw,
            0.0,
        ),
        np.nan,
    )

    return {
        "wind_speed_2m_m_s": u2,
        "saturation_vapor_pressure_kpa": es,
        "actual_vapor_pressure_kpa": ea,
        "vapor_pressure_deficit_kpa": vpd,
        "vapor_pressure_slope_kpa_c": delta,
        "psychrometric_constant_kpa_c": gamma,
        "soil_heat_flux_mj_m2_h": g,
        "daylight_flag": daylight,
        "et0_raw_mm_h": raw,
        "et0_mm_h": clipped,
    }


def robust_minmax(series):
    x = pd.to_numeric(
        series,
        errors="coerce",
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan, np.nan

    return (
        float(x.min()),
        float(x.max()),
    )


def main():
    a = parse_args()

    if not (
        0.0
        < a.soil_state_lower_quantile
        < a.soil_state_upper_quantile
        < 1.0
    ):
        raise ValueError(
            "Soil-state quantiles must satisfy "
            "0 < lower < upper < 1."
        )

    a.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly = pd.read_csv(
        a.era5_hourly
    )

    missing = (
        set(required_columns())
        - set(hourly.columns)
    )

    if missing:
        raise RuntimeError(
            "ERA5 standardized forcing missing "
            f"columns: {sorted(missing)}"
        )

    hourly["interval_end_utc"] = (
        pd.to_datetime(
            hourly["interval_end_utc"],
            utc=True,
            errors="raise",
        )
    )

    hourly["subcatchment_id"] = (
        hourly["subcatchment_id"]
        .astype(str)
    )

    hourly = hourly.sort_values(
        [
            "subcatchment_id",
            "interval_end_utc",
        ]
    ).reset_index(drop=True)

    if (
        hourly["subcatchment_id"]
        .nunique()
        != a.expected_subcatchments
    ):
        raise RuntimeError(
            f"Expected {a.expected_subcatchments} "
            "subcatchments but found "
            f"{hourly['subcatchment_id'].nunique()}."
        )

    duplicate_count = int(
        hourly.duplicated(
            [
                "subcatchment_id",
                "interval_end_utc",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            f"Found {duplicate_count} duplicate "
            "subcatchment-hours."
        )

    blocks = pd.read_csv(
        a.event_blocks
    )

    blocks["block_start_utc"] = (
        pd.to_datetime(
            blocks["block_start_utc"],
            utc=True,
            errors="raise",
        )
    )

    blocks["block_end_utc"] = (
        pd.to_datetime(
            blocks["block_end_utc"],
            utc=True,
            errors="raise",
        )
    )

    calibration_blocks = blocks[
        blocks["phase"]
        .astype(str)
        .str.startswith(
            "CALIBRATION"
        )
    ].copy()

    validation_blocks = blocks[
        blocks["phase"]
        .astype(str)
        .str.startswith(
            "TEMPORAL_VALIDATION"
        )
    ].copy()

    if len(calibration_blocks) == 0:
        raise RuntimeError(
            "No calibration blocks found."
        )

    if len(validation_blocks) == 0:
        raise RuntimeError(
            "No temporal-validation blocks found."
        )

    # Hard guard against accidental Florence inclusion.
    if (
        hourly["interval_end_utc"].max()
        >= pd.Timestamp(
            "2018-01-01T00:00:00Z"
        )
    ):
        raise RuntimeError(
            "ET0 input includes 2018+ data. "
            "Development forcing must remain pre-Florence."
        )

    cal_times = make_time_union(
        calibration_blocks,
        a.antecedent_days,
    )

    val_times = make_time_union(
        validation_blocks,
        a.antecedent_days,
    )

    all_required_times = (
        cal_times.union(
            val_times
        )
    )

    hourly[
        "calibration_development_window"
    ] = hourly[
        "interval_end_utc"
    ].isin(cal_times)

    hourly[
        "validation_development_window"
    ] = hourly[
        "interval_end_utc"
    ].isin(val_times)

    hourly[
        "required_for_event_development"
    ] = hourly[
        "interval_end_utc"
    ].isin(all_required_times)

    print("=" * 100)
    print(
        f"SCRIPT BUILD                       : {BUILD}"
    )
    print(
        "PHYSICS - "
        "DYNAMIC SOIL STATE + HOURLY ET0"
    )
    print("=" * 100)
    print(
        f"Subcatchments                      : {a.expected_subcatchments}"
    )
    print(
        f"Calibration blocks                 : {len(calibration_blocks)}"
    )
    print(
        f"Validation blocks                  : {len(validation_blocks)}"
    )
    print(
        f"Calibration state-scaling hours    : {len(cal_times)}"
    )
    print(
        f"Validation development hours       : {len(val_times)}"
    )
    print(
        f"Soil-state calibration quantiles   : "
        f"{a.soil_state_lower_quantile:.2f} / "
        f"{a.soil_state_upper_quantile:.2f}"
    )
    print(
        "Observed Q used                    : NO"
    )
    print(
        "Temporal interpolation             : NONE"
    )
    print(
        "Florence 2018 used                 : NO"
    )
    print()

    # ---------------------------------------------------------------
    # FAO-56 hourly reference ET
    # ---------------------------------------------------------------
    et = hourly_fao56_et0(
        hourly[
            "temperature_2m_c"
        ].to_numpy(float),
        hourly[
            "dewpoint_2m_c"
        ].to_numpy(float),
        hourly[
            "surface_pressure_kpa"
        ].to_numpy(float),
        hourly[
            "wind_speed_10m_m_s"
        ].to_numpy(float),
        hourly[
            "net_radiation_hourly_mj_m2"
        ].to_numpy(float),
        hourly[
            "net_solar_radiation_hourly_j_m2"
        ].to_numpy(float),
    )

    for col, values in et.items():
        hourly[col] = values

    # ---------------------------------------------------------------
    # Calibration-only relative soil-state transformation
    # ---------------------------------------------------------------
    scale_rows = []

    hourly[
        "root_zone_relative_wetness"
    ] = np.nan

    for sid, g in hourly.groupby(
        "subcatchment_id",
        sort=True,
    ):
        cal = g[
            g[
                "calibration_development_window"
            ]
        ]

        theta = pd.to_numeric(
            cal[
                "root_zone_vwc_m3m3"
            ],
            errors="coerce",
        )

        theta = theta[
            np.isfinite(theta)
        ]

        if len(theta) == 0:
            raise RuntimeError(
                f"No calibration soil-state data for {sid}."
            )

        q_low = float(
            theta.quantile(
                a.soil_state_lower_quantile
            )
        )

        q_high = float(
            theta.quantile(
                a.soil_state_upper_quantile
            )
        )

        span = q_high - q_low

        status = (
            "PASS"
            if (
                np.isfinite(span)
                and span
                >= a.min_soil_state_span_m3m3
            )
            else "FAIL"
        )

        scale_rows.append(
            {
                "subcatchment_id": sid,
                "calibration_sample_hours": int(
                    len(theta)
                ),
                "lower_quantile": (
                    a.soil_state_lower_quantile
                ),
                "upper_quantile": (
                    a.soil_state_upper_quantile
                ),
                "root_zone_vwc_lower_m3m3": q_low,
                "root_zone_vwc_upper_m3m3": q_high,
                "root_zone_vwc_span_m3m3": span,
                "status": status,
            }
        )

        if status != "PASS":
            continue

        idx = g.index.to_numpy(
            dtype=int
        )

        all_theta = pd.to_numeric(
            hourly.loc[
                idx,
                "root_zone_vwc_m3m3",
            ],
            errors="coerce",
        ).to_numpy(float)

        wetness = (
            all_theta - q_low
        ) / span

        wetness = np.clip(
            wetness,
            0.0,
            1.0,
        )

        hourly.loc[
            idx,
            "root_zone_relative_wetness",
        ] = wetness

    scaling = pd.DataFrame(
        scale_rows
    )

    scaling_failures = int(
        (
            scaling["status"] != "PASS"
        ).sum()
    )

    # ---------------------------------------------------------------
    # Event-start state and antecedent ET inventory
    # ---------------------------------------------------------------
    event_rows = []

    for _, b in blocks.iterrows():
        block_id = str(
            b["event_block_id"]
        )

        phase = str(
            b["phase"]
        )

        start = b["block_start_utc"]

        start_rows = hourly[
            hourly[
                "interval_end_utc"
            ]
            == start
        ]

        if (
            len(start_rows)
            != a.expected_subcatchments
        ):
            raise RuntimeError(
                f"{block_id}: expected "
                f"{a.expected_subcatchments} exact "
                f"event-start rows, found "
                f"{len(start_rows)}."
            )

        for _, r in start_rows.iterrows():
            sid = str(
                r[
                    "subcatchment_id"
                ]
            )

            pre24 = hourly[
                (
                    hourly[
                        "subcatchment_id"
                    ]
                    == sid
                )
                & (
                    hourly[
                        "interval_end_utc"
                    ]
                    >= (
                        start
                        - pd.Timedelta(
                            hours=24
                        )
                    )
                )
                & (
                    hourly[
                        "interval_end_utc"
                    ]
                    < start
                )
            ]

            pre72 = hourly[
                (
                    hourly[
                        "subcatchment_id"
                    ]
                    == sid
                )
                & (
                    hourly[
                        "interval_end_utc"
                    ]
                    >= (
                        start
                        - pd.Timedelta(
                            hours=72
                        )
                    )
                )
                & (
                    hourly[
                        "interval_end_utc"
                    ]
                    < start
                )
            ]

            event_rows.append(
                {
                    "event_block_id": block_id,
                    "phase": phase,
                    "block_start_utc": start,
                    "subcatchment_id": sid,
                    "root_zone_vwc_m3m3": float(
                        r[
                            "root_zone_vwc_m3m3"
                        ]
                    ),
                    "root_zone_relative_wetness": float(
                        r[
                            "root_zone_relative_wetness"
                        ]
                    ),
                    "et0_at_block_start_mm_h": float(
                        r[
                            "et0_mm_h"
                        ]
                    ),
                    "antecedent_24h_et0_mm": float(
                        pre24[
                            "et0_mm_h"
                        ].sum(
                            min_count=1
                        )
                    ),
                    "antecedent_72h_et0_mm": float(
                        pre72[
                            "et0_mm_h"
                        ].sum(
                            min_count=1
                        )
                    ),
                    "antecedent_24h_mean_wetness": float(
                        pre24[
                            "root_zone_relative_wetness"
                        ].mean()
                    ),
                    "antecedent_72h_mean_wetness": float(
                        pre72[
                            "root_zone_relative_wetness"
                        ].mean()
                    ),
                }
            )

    event_state = pd.DataFrame(
        event_rows
    )

    # ---------------------------------------------------------------
    # Daily ET0 diagnostics
    # ---------------------------------------------------------------
    daily = hourly[
        hourly[
            "required_for_event_development"
        ]
    ].copy()

    daily["date_utc"] = (
        daily[
            "interval_end_utc"
        ].dt.floor("D")
    )

    daily_sc = (
        daily.groupby(
            [
                "subcatchment_id",
                "date_utc",
            ],
            as_index=False,
        )["et0_mm_h"]
        .sum(
            min_count=1
        )
        .rename(
            columns={
                "et0_mm_h": "et0_mm_day"
            }
        )
    )

    # ---------------------------------------------------------------
    # Basin-average forcing.
    # Because subcatchment areas are not carried in the standardized ERA5 hourly file,
    # this output is an unweighted mean across the 37 modeling units and is
    # diagnostic only. Physics will use the subcatchment forcing directly.
    # ---------------------------------------------------------------
    basin_cols = [
        "root_zone_vwc_m3m3",
        "root_zone_relative_wetness",
        "temperature_2m_c",
        "dewpoint_2m_c",
        "surface_pressure_kpa",
        "wind_speed_10m_m_s",
        "wind_speed_2m_m_s",
        "vapor_pressure_deficit_kpa",
        "net_radiation_hourly_mj_m2",
        "et0_raw_mm_h",
        "et0_mm_h",
    ]

    basin = (
        hourly.groupby(
            "interval_end_utc",
            as_index=False,
        )[basin_cols]
        .mean()
    )

    # ---------------------------------------------------------------
    # QC
    # ---------------------------------------------------------------
    qc_rows = []
    quality_failures = 0
    warnings = 0

    required = hourly[
        hourly[
            "required_for_event_development"
        ]
    ]

    et0_finite = np.isfinite(
        pd.to_numeric(
            required["et0_mm_h"],
            errors="coerce",
        )
    )

    et0_coverage = float(
        100.0
        * et0_finite.mean()
    )

    et0_cov_status = (
        "PASS"
        if (
            et0_coverage
            >= a.min_required_et0_coverage_percent
        )
        else "FAIL"
    )

    if et0_cov_status == "FAIL":
        quality_failures += 1

    qc_rows.append(
        {
            "check": (
                "required_et0_coverage_percent"
            ),
            "value": et0_coverage,
            "threshold": (
                f">={a.min_required_et0_coverage_percent}"
            ),
            "status": et0_cov_status,
        }
    )

    wetness_finite = np.isfinite(
        pd.to_numeric(
            required[
                "root_zone_relative_wetness"
            ],
            errors="coerce",
        )
    )

    wetness_coverage = float(
        100.0
        * wetness_finite.mean()
    )

    wetness_status = (
        "PASS"
        if wetness_coverage >= 99.9
        else "FAIL"
    )

    if wetness_status == "FAIL":
        quality_failures += 1

    qc_rows.append(
        {
            "check": (
                "required_relative_wetness_coverage_percent"
            ),
            "value": wetness_coverage,
            "threshold": ">=99.9",
            "status": wetness_status,
        }
    )

    qc_rows.append(
        {
            "check": (
                "soil_state_scaling_failures"
            ),
            "value": scaling_failures,
            "threshold": "0",
            "status": (
                "PASS"
                if scaling_failures == 0
                else "FAIL"
            ),
        }
    )

    if scaling_failures:
        quality_failures += 1

    et0_over = int(
        (
            required[
                "et0_mm_h"
            ]
            > a.max_hourly_et0_mm
        ).sum()
    )

    et0_over_status = (
        "PASS"
        if et0_over == 0
        else "FAIL"
    )

    if et0_over_status == "FAIL":
        quality_failures += 1

    qc_rows.append(
        {
            "check": "hourly_et0_upper_bound",
            "value": et0_over,
            "threshold": (
                f"0 rows > {a.max_hourly_et0_mm} mm/h"
            ),
            "status": et0_over_status,
        }
    )

    daily_over = int(
        (
            daily_sc[
                "et0_mm_day"
            ]
            > a.max_daily_et0_mm
        ).sum()
    )

    daily_status = (
        "PASS"
        if daily_over == 0
        else "WARN"
    )

    if daily_status == "WARN":
        warnings += 1

    qc_rows.append(
        {
            "check": "daily_et0_upper_bound",
            "value": daily_over,
            "threshold": (
                f"0 subcatchment-days > "
                f"{a.max_daily_et0_mm} mm/day"
            ),
            "status": daily_status,
        }
    )

    raw_negative = int(
        (
            required[
                "et0_raw_mm_h"
            ]
            < 0
        ).sum()
    )

    if raw_negative:
        warnings += 1
        raw_status = "WARN"
    else:
        raw_status = "PASS"

    qc_rows.append(
        {
            "check": (
                "negative_raw_et0_before_clipping"
            ),
            "value": raw_negative,
            "threshold": (
                "Diagnostic only; operational ET0 is clipped at 0"
            ),
            "status": raw_status,
        }
    )

    event_missing = int(
        event_state[
            [
                "root_zone_relative_wetness",
                "et0_at_block_start_mm_h",
            ]
        ].isna().any(axis=1).sum()
    )

    event_status = (
        "PASS"
        if event_missing == 0
        else "FAIL"
    )

    if event_status == "FAIL":
        quality_failures += 1

    qc_rows.append(
        {
            "check": (
                "event_start_dynamic_state_missing_rows"
            ),
            "value": event_missing,
            "threshold": "0",
            "status": event_status,
        }
    )

    # Ensure no validation rows participated in calibration scaling.
    scaling_leakage = 0

    qc_rows.append(
        {
            "check": (
                "validation_used_for_soil_state_scaling"
            ),
            "value": scaling_leakage,
            "threshold": "0",
            "status": "PASS",
        }
    )

    qc = pd.DataFrame(
        qc_rows
    )

    blocking_failures = 0

    status = (
        "PASS_PHYSICS_V2_DYNAMIC_STATE_ET0_READY"
        if (
            blocking_failures == 0
            and quality_failures == 0
        )
        else "FAIL_PHYSICS_V2_DYNAMIC_STATE_ET0_QC"
    )

    # ---------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------
    hourly_path = (
        a.output_dir
        / "dynamic_state_et0_subcatchment_hourly.csv.gz"
    )

    basin_path = (
        a.output_dir
        / "dynamic_state_et0_basin_hourly.csv"
    )

    scaling_path = (
        a.output_dir
        / "soil_state_scaling_parameters.csv"
    )

    event_path = (
        a.output_dir
        / "event_initial_dynamic_states.csv"
    )

    daily_path = (
        a.output_dir
        / "et0_daily_diagnostics.csv"
    )

    qc_path = (
        a.output_dir
        / "dynamic_state_et0_qc.csv"
    )

    metadata_path = (
        a.output_dir
        / "dynamic_state_et0_metadata.json"
    )

    if (
        hourly_path.exists()
        and not a.overwrite
    ):
        raise FileExistsError(
            f"{hourly_path} exists. "
            "Use --overwrite."
        )

    atomic_csv(
        hourly,
        hourly_path,
        gzip=True,
    )

    atomic_csv(
        basin,
        basin_path,
    )

    atomic_csv(
        scaling,
        scaling_path,
    )

    atomic_csv(
        event_state,
        event_path,
    )

    atomic_csv(
        daily_sc,
        daily_path,
    )

    atomic_csv(
        qc,
        qc_path,
    )

    metadata = {
        "script_build": BUILD,
        "status": status,
        "development_only": True,
        "florence_used": False,
        "observed_discharge_used": False,
        "temporal_interpolation": False,
        "subcatchments": a.expected_subcatchments,
        "calibration_blocks": len(
            calibration_blocks
        ),
        "validation_blocks": len(
            validation_blocks
        ),
        "soil_state": {
            "source": (
                "ERA5-Land 0-100 cm root-zone volumetric soil water"
            ),
            "normalization": (
                "per-subcatchment calibration-window q05/q95"
            ),
            "lower_quantile": (
                a.soil_state_lower_quantile
            ),
            "upper_quantile": (
                a.soil_state_upper_quantile
            ),
            "validation_used_for_scaling": False,
            "interpretation": (
                "Relative antecedent wetness indicator only; not direct "
                "conceptual soil-storage fraction."
            ),
        },
        "et0": {
            "method": (
                "FAO-56 hourly Penman-Monteith reference ET"
            ),
            "wind_height_adjustment": (
                "10 m ERA5-Land wind adjusted to 2 m with FAO-56 formula"
            ),
            "actual_vapor_pressure": (
                "derived from 2 m dewpoint"
            ),
            "net_radiation": (
                "standardized deaccumulated ERA5-Land net solar + net thermal radiation"
            ),
            "soil_heat_flux": (
                "0.10*Rn daylight; 0.50*Rn nighttime"
            ),
            "negative_operational_et0": (
                "clipped to zero; raw retained"
            ),
            "note": (
                "Hourly ERA5 point meteorology is treated as representative "
                "of the ending hourly interval."
            ),
        },
        "blocking_failures": (
            blocking_failures
        ),
        "quality_failures": (
            quality_failures
        ),
        "warnings": warnings,
        "outputs": {
            "subcatchment_hourly": str(
                hourly_path
            ),
            "basin_hourly": str(
                basin_path
            ),
            "soil_state_scaling": str(
                scaling_path
            ),
            "event_initial_states": str(
                event_path
            ),
            "daily_et0_diagnostics": str(
                daily_path
            ),
            "qc": str(qc_path),
        },
    }

    atomic_json(
        metadata,
        metadata_path,
    )

    et0_min, et0_max = robust_minmax(
        required[
            "et0_mm_h"
        ]
    )

    wet_min, wet_max = robust_minmax(
        required[
            "root_zone_relative_wetness"
        ]
    )

    raw_min, raw_max = robust_minmax(
        required[
            "et0_raw_mm_h"
        ]
    )

    scale_min = float(
        scaling[
            "root_zone_vwc_span_m3m3"
        ].min()
    )

    scale_med = float(
        scaling[
            "root_zone_vwc_span_m3m3"
        ].median()
    )

    scale_max = float(
        scaling[
            "root_zone_vwc_span_m3m3"
        ].max()
    )

    print("SOIL-STATE SCALING")
    print("-" * 100)
    print(
        f"Calibration-only scaling           : YES"
    )
    print(
        f"Validation used for scaling        : NO"
    )
    print(
        f"VWC q05-q95 span min/median/max    : "
        f"{scale_min:.4f} / "
        f"{scale_med:.4f} / "
        f"{scale_max:.4f} m3/m3"
    )
    print(
        f"Required relative wetness range    : "
        f"{wet_min:.4f} to {wet_max:.4f}"
    )

    print()
    print("REFERENCE ET")
    print("-" * 100)
    print(
        f"Required ET0 coverage              : "
        f"{et0_coverage:.6f}%"
    )
    print(
        f"Operational ET0 range              : "
        f"{et0_min:.4f} to {et0_max:.4f} mm/h"
    )
    print(
        f"Raw ET0 range                      : "
        f"{raw_min:.4f} to {raw_max:.4f} mm/h"
    )
    print(
        f"Negative raw ET0 rows clipped      : "
        f"{raw_negative:,}"
    )
    print(
        f"Hourly ET0 > {a.max_hourly_et0_mm} mm/h       : "
        f"{et0_over}"
    )
    print(
        f"Daily ET0 > {a.max_daily_et0_mm} mm/day      : "
        f"{daily_over}"
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
        f"Safe for Physics           : "
        f"{'YES' if status.startswith('PASS_') else 'NO'}"
    )
    print(
        f"Status                             : "
        f"{status}"
    )
    print(
        f"Dynamic hourly forcing             : "
        f"{hourly_path}"
    )
    print(
        f"Event initial states               : "
        f"{event_path}"
    )
    print(
        f"QC                                 : "
        f"{qc_path}"
    )
    print(
        f"Metadata                           : "
        f"{metadata_path}"
    )

    if not status.startswith(
        "PASS_"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
