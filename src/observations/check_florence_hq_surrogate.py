"""
Step 1F - Validate the Florence 2018 H-Q surrogate against USGS field measurements.

Primary validation
------------------
Only field visits that occur within the Step 1 continuous event record are
treated as event-period independent anchors. For those visits, the script:

1. predicts discharge from the appropriate rising/falling surrogate branch
   using measured field gage height;
2. time-interpolates the continuous USGS event record to the field-visit time;
3. compares both estimates against field-measured discharge.

Other 2017-2019 field measurements are retained as contextual stability checks,
not as Florence-event validation observations.

Important
---------
The Florence high-flow field measurement is USGS Approved but rated Poor.
The script preserves and reports that quality rating rather than hiding it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CFS_TO_CMS = 0.028316846592
FT_TO_M = 0.3048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Florence H-Q surrogate against USGS field measurements."
    )
    parser.add_argument(
        "--event-observations",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--surrogate-h-to-q",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--surrogate-metadata",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--field-measurements",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-time-interpolation-gap-minutes",
        type=float,
        default=30.0,
        help="Maximum allowed bracketing gap for direct event-record interpolation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/observations/florence_2018_hq_validation"),
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


def linear_interp_inside(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    result = np.full(np.asarray(x).shape, np.nan, dtype=float)
    valid = np.isfinite(x)
    if len(xp) < 2:
        return result
    inside = valid & (x >= xp.min()) & (x <= xp.max())
    if inside.any():
        result[inside] = np.interp(x[inside], xp, fp)
    return result


def time_interpolate_event(
    event: pd.DataFrame,
    target_time: pd.Timestamp,
    max_gap_minutes: float,
) -> dict[str, Any]:
    times = event["datetime_utc"]
    if target_time < times.min() or target_time > times.max():
        return {
            "available": False,
            "reason": "FIELD_VISIT_OUTSIDE_EVENT_RECORD",
        }

    before = event.loc[times <= target_time].tail(1)
    after = event.loc[times >= target_time].head(1)

    if before.empty or after.empty:
        return {
            "available": False,
            "reason": "NO_BRACKETING_EVENT_OBSERVATIONS",
        }

    b = before.iloc[0]
    a = after.iloc[0]
    total_seconds = (a["datetime_utc"] - b["datetime_utc"]).total_seconds()

    if total_seconds == 0:
        weight = 0.0
        gap_minutes = 0.0
    else:
        gap_minutes = total_seconds / 60.0
        if gap_minutes > max_gap_minutes:
            return {
                "available": False,
                "reason": "BRACKETING_GAP_EXCEEDS_LIMIT",
                "gap_minutes": gap_minutes,
            }
        weight = (
            (target_time - b["datetime_utc"]).total_seconds() / total_seconds
        )

    q = float(b["discharge_cfs"] + weight * (a["discharge_cfs"] - b["discharge_cfs"]))
    h = float(b["gage_height_ft"] + weight * (a["gage_height_ft"] - b["gage_height_ft"]))

    return {
        "available": True,
        "reason": "OK",
        "gap_minutes": gap_minutes,
        "interpolated_discharge_cfs": q,
        "interpolated_gage_height_ft": h,
        "before_time_utc": b["datetime_utc"].isoformat(),
        "after_time_utc": a["datetime_utc"].isoformat(),
    }


def percent_error(pred: float, obs: float) -> float:
    if not np.isfinite(pred) or not np.isfinite(obs) or obs == 0:
        return float("nan")
    return (pred - obs) / obs * 100.0


def quality_weight(value: Any) -> float:
    mapping = {
        "Excellent": 1.0,
        "Good": 1.0,
        "Fair": 0.6,
        "Poor": 0.3,
    }
    return mapping.get(str(value), 0.5)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)

    event = pd.read_csv(args.event_observations)
    event["datetime_utc"] = pd.to_datetime(
        event["datetime_utc"], utc=True, errors="coerce"
    )
    event["discharge_cfs"] = pd.to_numeric(event["discharge_cfs"], errors="coerce")
    event["gage_height_ft"] = pd.to_numeric(event["gage_height_ft"], errors="coerce")
    event = (
        event.dropna(subset=["datetime_utc", "discharge_cfs", "gage_height_ft"])
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )

    with open(args.surrogate_metadata, "r", encoding="utf-8") as f:
        surrogate_meta = json.load(f)

    peak_time = pd.Timestamp(surrogate_meta["peak"]["datetime_utc"])
    if peak_time.tzinfo is None:
        peak_time = peak_time.tz_localize("UTC")
    else:
        peak_time = peak_time.tz_convert("UTC")

    hq = pd.read_csv(args.surrogate_h_to_q)
    hq["gage_height_ft"] = pd.to_numeric(hq["gage_height_ft"], errors="coerce")

    field = pd.read_csv(args.field_measurements)
    field["representative_time_utc"] = pd.to_datetime(
        field["representative_time_utc"], utc=True, errors="coerce"
    )
    field["gage_height_ft"] = pd.to_numeric(field["gage_height_ft"], errors="coerce")
    field["discharge_cfs"] = pd.to_numeric(field["discharge_cfs"], errors="coerce")
    field = field.dropna(
        subset=["representative_time_utc", "gage_height_ft", "discharge_cfs"]
    ).copy()

    event_start = event["datetime_utc"].min()
    event_end = event["datetime_utc"].max()

    stage_grid = hq["gage_height_ft"].to_numpy(float)
    rising_q = pd.to_numeric(hq["rising_discharge_cfs"], errors="coerce").to_numpy(float)
    falling_q = pd.to_numeric(hq["falling_discharge_cfs"], errors="coerce").to_numpy(float)

    results = []

    for _, row in field.iterrows():
        t = row["representative_time_utc"]
        h = float(row["gage_height_ft"])
        q_obs = float(row["discharge_cfs"])

        r_pred = linear_interp_inside(
            np.array([h]),
            stage_grid[np.isfinite(rising_q)],
            rising_q[np.isfinite(rising_q)],
        )[0]
        f_pred = linear_interp_inside(
            np.array([h]),
            stage_grid[np.isfinite(falling_q)],
            falling_q[np.isfinite(falling_q)],
        )[0]

        in_event = bool(event_start <= t <= event_end)

        if in_event:
            event_limb = "rising" if t <= peak_time else "falling"
            selected_pred = r_pred if event_limb == "rising" else f_pred
            direct = time_interpolate_event(
                event,
                t,
                args.max_time_interpolation_gap_minutes,
            )
        else:
            event_limb = None
            selected_pred = np.nan
            direct = {
                "available": False,
                "reason": "FIELD_VISIT_OUTSIDE_EVENT_RECORD",
            }

        contextual_predictions = [
            p for p in (r_pred, f_pred) if np.isfinite(p)
        ]
        if contextual_predictions:
            contextual_best = min(
                contextual_predictions,
                key=lambda p: abs(p - q_obs),
            )
            contextual_best_abs_error = abs(contextual_best - q_obs)
            contextual_best_pct_error = percent_error(contextual_best, q_obs)
        else:
            contextual_best = np.nan
            contextual_best_abs_error = np.nan
            contextual_best_pct_error = np.nan

        results.append(
            {
                **row.to_dict(),
                "is_within_continuous_event_record": in_event,
                "event_limb_if_applicable": event_limb,
                "rising_surrogate_q_cfs": r_pred,
                "falling_surrogate_q_cfs": f_pred,
                "selected_event_surrogate_q_cfs": selected_pred,
                "selected_event_surrogate_q_cms": (
                    selected_pred * CFS_TO_CMS
                    if np.isfinite(selected_pred)
                    else np.nan
                ),
                "selected_surrogate_error_cfs": (
                    selected_pred - q_obs
                    if np.isfinite(selected_pred)
                    else np.nan
                ),
                "selected_surrogate_absolute_error_cfs": (
                    abs(selected_pred - q_obs)
                    if np.isfinite(selected_pred)
                    else np.nan
                ),
                "selected_surrogate_percent_error": percent_error(
                    selected_pred, q_obs
                ),
                "direct_event_interpolation_available": direct.get("available", False),
                "direct_event_interpolation_reason": direct.get("reason"),
                "direct_event_interpolation_gap_minutes": direct.get("gap_minutes"),
                "continuous_time_interpolated_q_cfs": direct.get(
                    "interpolated_discharge_cfs"
                ),
                "continuous_time_interpolated_h_ft": direct.get(
                    "interpolated_gage_height_ft"
                ),
                "continuous_q_vs_field_error_cfs": (
                    direct.get("interpolated_discharge_cfs") - q_obs
                    if direct.get("available")
                    else np.nan
                ),
                "continuous_q_vs_field_percent_error": (
                    percent_error(
                        direct.get("interpolated_discharge_cfs"),
                        q_obs,
                    )
                    if direct.get("available")
                    else np.nan
                ),
                "continuous_h_vs_field_error_ft": (
                    direct.get("interpolated_gage_height_ft") - h
                    if direct.get("available")
                    else np.nan
                ),
                "contextual_best_branch_q_cfs": contextual_best,
                "contextual_best_branch_absolute_error_cfs": contextual_best_abs_error,
                "contextual_best_branch_percent_error": contextual_best_pct_error,
                "measurement_quality_weight": quality_weight(
                    row.get("discharge_measurement_rated")
                ),
                "validation_role": (
                    "PRIMARY_EVENT_PERIOD_FIELD_ANCHOR"
                    if in_event
                    else "CONTEXTUAL_2017_2019_STABILITY_CHECK"
                ),
            }
        )

    validation = pd.DataFrame(results).sort_values("representative_time_utc")

    primary = validation.loc[
        validation["validation_role"].eq("PRIMARY_EVENT_PERIOD_FIELD_ANCHOR")
    ].copy()
    contextual = validation.loc[
        validation["validation_role"].eq("CONTEXTUAL_2017_2019_STABILITY_CHECK")
    ].copy()

    metrics_rows = []

    if not primary.empty:
        valid_sur = primary.dropna(
            subset=["selected_event_surrogate_q_cfs", "discharge_cfs"]
        )
        if not valid_sur.empty:
            err = (
                valid_sur["selected_event_surrogate_q_cfs"]
                - valid_sur["discharge_cfs"]
            )
            metrics_rows.append(
                {
                    "scope": "PRIMARY_EVENT_FIELD_ANCHOR_SURROGATE",
                    "count": int(len(valid_sur)),
                    "mae_cfs": float(err.abs().mean()),
                    "rmse_cfs": float(np.sqrt(np.mean(err**2))),
                    "mean_percent_error": float(
                        valid_sur["selected_surrogate_percent_error"].mean()
                    ),
                    "mean_absolute_percent_error": float(
                        valid_sur["selected_surrogate_percent_error"].abs().mean()
                    ),
                    "interpretation": (
                        "Independent field-measurement check of the reconstructed "
                        "event-limb surrogate."
                    ),
                }
            )

        valid_direct = primary.loc[
            primary["direct_event_interpolation_available"].eq(True)
        ].dropna(subset=["continuous_time_interpolated_q_cfs", "discharge_cfs"])
        if not valid_direct.empty:
            err = (
                valid_direct["continuous_time_interpolated_q_cfs"]
                - valid_direct["discharge_cfs"]
            )
            metrics_rows.append(
                {
                    "scope": "PRIMARY_EVENT_FIELD_ANCHOR_DIRECT_CONTINUOUS",
                    "count": int(len(valid_direct)),
                    "mae_cfs": float(err.abs().mean()),
                    "rmse_cfs": float(np.sqrt(np.mean(err**2))),
                    "mean_percent_error": float(
                        valid_direct["continuous_q_vs_field_percent_error"].mean()
                    ),
                    "mean_absolute_percent_error": float(
                        valid_direct["continuous_q_vs_field_percent_error"].abs().mean()
                    ),
                    "interpretation": (
                        "Direct time-interpolated USGS continuous Q versus "
                        "field-measured Q at the same field-visit time."
                    ),
                }
            )

    contextual_valid = contextual.dropna(
        subset=["contextual_best_branch_q_cfs", "discharge_cfs"]
    )
    if not contextual_valid.empty:
        abs_pct = contextual_valid[
            "contextual_best_branch_percent_error"
        ].abs()
        weights = contextual_valid["measurement_quality_weight"].to_numpy(float)
        metrics_rows.append(
            {
                "scope": "CONTEXTUAL_2017_2019_BEST_BRANCH",
                "count": int(len(contextual_valid)),
                "mae_cfs": float(
                    contextual_valid[
                        "contextual_best_branch_absolute_error_cfs"
                    ].mean()
                ),
                "rmse_cfs": float(
                    np.sqrt(
                        np.mean(
                            (
                                contextual_valid["contextual_best_branch_q_cfs"]
                                - contextual_valid["discharge_cfs"]
                            )
                            ** 2
                        )
                    )
                ),
                "mean_percent_error": float(
                    contextual_valid[
                        "contextual_best_branch_percent_error"
                    ].mean()
                ),
                "mean_absolute_percent_error": float(abs_pct.mean()),
                "quality_weighted_mean_absolute_percent_error": float(
                    np.average(abs_pct.to_numpy(float), weights=weights)
                ),
                "interpretation": (
                    "Context only; these visits are outside the Florence continuous "
                    "event window and may reflect other rating shifts."
                ),
            }
        )

    metrics = pd.DataFrame(metrics_rows)

    qc_rows = []

    if primary.empty:
        qc_rows.append(
            {
                "severity": "BLOCKING",
                "issue": "NO_FIELD_MEASUREMENT_DURING_EVENT_RECORD",
                "detail": (
                    "No paired USGS field measurement falls inside the continuous "
                    "Florence event record."
                ),
            }
        )
    else:
        qc_rows.append(
            {
                "severity": "NOTE",
                "issue": "EVENT_FIELD_ANCHOR_COUNT",
                "detail": f"{len(primary)} field H-Q visit(s) fall inside the event record.",
            }
        )

    poor_primary = primary.loc[
        primary["discharge_measurement_rated"].astype(str).eq("Poor")
    ]
    if not poor_primary.empty:
        qc_rows.append(
            {
                "severity": "WARNING",
                "issue": "PRIMARY_HIGH_FLOW_FIELD_MEASUREMENT_RATED_POOR",
                "detail": (
                    f"{len(poor_primary)} primary event field measurement(s) are "
                    "USGS Approved but rated Poor. Treat the high-flow validation "
                    "as an uncertainty-bounded anchor, not exact truth."
                ),
            }
        )

    qc_rows.append(
        {
            "severity": "SCIENTIFIC_NOTE",
            "issue": "CONTEXTUAL_FIELD_VISITS_NOT_EVENT_VALIDATION",
            "detail": (
                "2017-2019 field visits outside the continuous Florence record are "
                "reported only as stability/context checks because rating shifts may "
                "differ outside the event."
            ),
        }
    )

    qc = pd.DataFrame(qc_rows)
    blocking = int((qc["severity"] == "BLOCKING").sum())

    # Plot
    fig_path = args.output_dir / "florence_hq_field_validation.png"
    fig, ax = plt.subplots(figsize=(9, 6))

    # Continuous event cloud
    ax.scatter(
        event["gage_height_ft"],
        event["discharge_cfs"],
        s=7,
        alpha=0.18,
        label="USGS continuous event H-Q",
    )

    rising_plot = hq.dropna(subset=["rising_discharge_cfs"])
    falling_plot = hq.dropna(subset=["falling_discharge_cfs"])

    ax.plot(
        rising_plot["gage_height_ft"],
        rising_plot["rising_discharge_cfs"],
        linewidth=2,
        label="Rising-limb surrogate",
    )
    ax.plot(
        falling_plot["gage_height_ft"],
        falling_plot["falling_discharge_cfs"],
        linewidth=2,
        label="Falling-limb surrogate",
    )

    if not contextual.empty:
        ax.scatter(
            contextual["gage_height_ft"],
            contextual["discharge_cfs"],
            marker="o",
            s=45,
            label="2017-2019 contextual field H-Q",
        )

    if not primary.empty:
        ax.scatter(
            primary["gage_height_ft"],
            primary["discharge_cfs"],
            marker="X",
            s=110,
            label="Florence-period field anchor",
        )

    ax.set_xlabel("Gage height (ft)")
    ax.set_ylabel("Discharge (ft³/s)")
    ax.set_title("USGS 02089000 — Florence 2018 H-Q surrogate and field validation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    validation_path = args.output_dir / "field_hq_validation_results.csv"
    metrics_path = args.output_dir / "field_hq_validation_metrics.csv"
    primary_path = args.output_dir / "florence_primary_field_anchor.csv"
    qc_path = args.output_dir / "field_hq_validation_qc.csv"
    report_path = args.output_dir / "field_hq_validation_report.md"
    metadata_path = args.output_dir / "field_hq_validation_metadata.json"

    atomic_write_csv(validation, validation_path)
    atomic_write_csv(metrics, metrics_path)
    atomic_write_csv(primary, primary_path)
    atomic_write_csv(qc, qc_path)

    lines = [
        "# Florence 2018 H-Q Field Validation",
        "",
        "## Scientific interpretation",
        "",
        "This validation evaluates the reconstructed Florence 2018 effective H-Q surrogate.",
        "It does not convert the surrogate into an official historical USGS rating curve.",
        "",
        f"- Continuous event period: {event_start.isoformat()} to {event_end.isoformat()}",
        f"- Peak stage time: {peak_time.isoformat()}",
        f"- Paired field visits supplied: {len(field)}",
        f"- Primary field visits inside event record: {len(primary)}",
        f"- Contextual field visits outside event record: {len(contextual)}",
        "",
    ]

    if not primary.empty:
        lines += [
            "## Primary event-period field anchor(s)",
            "",
        ]
        for _, row in primary.iterrows():
            lines += [
                f"- Time: {row['representative_time_utc'].isoformat()}",
                f"  - Field stage: {row['gage_height_ft']:.3f} ft",
                f"  - Field discharge: {row['discharge_cfs']:,.1f} ft³/s",
                f"  - USGS field quality rating: {row.get('discharge_measurement_rated')}",
                f"  - Event limb: {row['event_limb_if_applicable']}",
                f"  - Surrogate Q: {row['selected_event_surrogate_q_cfs']:,.1f} ft³/s",
                f"  - Surrogate percent error: {row['selected_surrogate_percent_error']:.2f}%",
            ]
            if bool(row["direct_event_interpolation_available"]):
                lines += [
                    f"  - Time-interpolated continuous Q: {row['continuous_time_interpolated_q_cfs']:,.1f} ft³/s",
                    f"  - Continuous-vs-field percent error: {row['continuous_q_vs_field_percent_error']:.2f}%",
                    f"  - Time-interpolated continuous H: {row['continuous_time_interpolated_h_ft']:.3f} ft",
                    f"  - Continuous-vs-field stage error: {row['continuous_h_vs_field_error_ft']:.3f} ft",
                ]
            lines.append("")

    lines += [
        "## Use boundary",
        "",
        "- The primary event-period field measurement is independent field evidence.",
        "- Its USGS measurement quality rating must be retained in uncertainty analysis.",
        "- Other 2017-2019 visits are contextual because the rating/shift may differ by date.",
        "- The reconstructed curves remain event-specific surrogates, not official historical ratings.",
        "",
    ]
    atomic_write_text("\n".join(lines), report_path)

    primary_summary = []
    for _, row in primary.iterrows():
        primary_summary.append(
            {
                "time_utc": row["representative_time_utc"].isoformat(),
                "field_gage_height_ft": float(row["gage_height_ft"]),
                "field_discharge_cfs": float(row["discharge_cfs"]),
                "field_measurement_rating": row.get("discharge_measurement_rated"),
                "event_limb": row["event_limb_if_applicable"],
                "surrogate_discharge_cfs": (
                    float(row["selected_event_surrogate_q_cfs"])
                    if pd.notna(row["selected_event_surrogate_q_cfs"])
                    else None
                ),
                "surrogate_percent_error": (
                    float(row["selected_surrogate_percent_error"])
                    if pd.notna(row["selected_surrogate_percent_error"])
                    else None
                ),
                "time_interpolated_continuous_discharge_cfs": (
                    float(row["continuous_time_interpolated_q_cfs"])
                    if pd.notna(row["continuous_time_interpolated_q_cfs"])
                    else None
                ),
                "continuous_vs_field_percent_error": (
                    float(row["continuous_q_vs_field_percent_error"])
                    if pd.notna(row["continuous_q_vs_field_percent_error"])
                    else None
                ),
            }
        )

    metadata = {
        "status": (
            "PASS_FLORENCE_HQ_SURROGATE_FIELD_VALIDATION_COMPLETED"
            if blocking == 0
            else "FAIL_FLORENCE_HQ_SURROGATE_FIELD_VALIDATION"
        ),
        "step": "STEP_1F",
        "created_utc": utc_now(),
        "scientific_target": "Florence 2018 effective H-Q surrogate",
        "official_usgs_historical_rating_validated": False,
        "event_record_start_utc": event_start.isoformat(),
        "event_record_end_utc": event_end.isoformat(),
        "peak_stage_time_utc": peak_time.isoformat(),
        "field_visit_count": int(len(field)),
        "primary_event_field_anchor_count": int(len(primary)),
        "contextual_field_visit_count": int(len(contextual)),
        "primary_event_field_anchors": primary_summary,
        "primary_measurement_quality_caveat": (
            "A USGS Approved field measurement can still carry a Poor measurement "
            "rating. That quality rating is retained explicitly."
        ),
        "blocking_qc_issue_count": blocking,
        "warning_count": int((qc["severity"] == "WARNING").sum()),
        "output_paths": {
            "validation_results": str(validation_path),
            "metrics": str(metrics_path),
            "primary_field_anchor": str(primary_path),
            "figure": str(fig_path),
            "qc": str(qc_path),
            "report": str(report_path),
            "metadata": str(metadata_path),
        },
    }
    atomic_write_json(metadata, metadata_path)

    print("=" * 100)
    print("STEP 1F - FLORENCE H-Q SURROGATE FIELD VALIDATION")
    print("=" * 100)
    print(f"Paired field visits supplied       : {len(field):,}")
    print(f"Primary event-period anchors       : {len(primary):,}")
    print(f"Contextual field visits            : {len(contextual):,}")

    if not primary.empty:
        for _, row in primary.iterrows():
            print()
            print(f"Primary field time                 : {row['representative_time_utc'].isoformat()}")
            print(f"Field stage                        : {row['gage_height_ft']:.3f} ft")
            print(f"Field discharge                    : {row['discharge_cfs']:,.1f} ft3/s")
            print(f"Field quality                      : {row.get('discharge_measurement_rated')}")
            print(f"Selected event limb                : {row['event_limb_if_applicable']}")
            if pd.notna(row["selected_event_surrogate_q_cfs"]):
                print(f"Surrogate discharge                : {row['selected_event_surrogate_q_cfs']:,.1f} ft3/s")
                print(f"Surrogate error                    : {row['selected_surrogate_percent_error']:.2f} %")
            if bool(row["direct_event_interpolation_available"]):
                print(f"Continuous interpolated Q          : {row['continuous_time_interpolated_q_cfs']:,.1f} ft3/s")
                print(f"Continuous-vs-field error          : {row['continuous_q_vs_field_percent_error']:.2f} %")
                print(f"Continuous interpolated H          : {row['continuous_time_interpolated_h_ft']:.3f} ft")
                print(f"Continuous-vs-field H error        : {row['continuous_h_vs_field_error_ft']:.3f} ft")

    print()
    print(f"Validation results                 : {validation_path}")
    print(f"Metrics                            : {metrics_path}")
    print(f"Figure                             : {fig_path}")
    print(f"Report                             : {report_path}")
    print(f"QC                                 : {qc_path}")
    print(f"Metadata                           : {metadata_path}")
    print()
    print(f"Status                             : {metadata['status']}")

    if blocking:
        raise RuntimeError(f"Step 1F produced {blocking} blocking QC issue(s).")


if __name__ == "__main__":
    main()
