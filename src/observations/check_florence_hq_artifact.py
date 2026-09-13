"""
Hard preflight for the Florence 2018 effective H-Q surrogate.

This script verifies the exact stage/discharge artifacts before they are used
by later hydrologic/flood-model stages.

It checks:
- expected SHA-256 fingerprints;
- Step 1E / Step 1F metadata status;
- required schemas;
- unit conversions;
- strictly ordered stage/discharge grids;
- rising/falling monotonicity;
- source-event peak and paired-count reproduction;
- reconstruction fidelity;
- independent Florence field anchor;
- supported H->Q and Q->H domains;
- explicit no-extrapolation boundaries.

Exit status:
  0 = PASS or PASS_WITH_WARNINGS
  1 = blocking validation failure

The surrogate is NOT an official historical USGS rating curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CFS_TO_CMS = 0.028316846592
FT_TO_M = 0.3048


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--event-observations", type=Path, required=True)
    p.add_argument("--field-measurements", type=Path, required=True)
    p.add_argument("--h-to-q", type=Path, required=True)
    p.add_argument("--q-to-h", type=Path, required=True)
    p.add_argument("--surrogate-metadata", type=Path, required=True)
    p.add_argument("--reconstruction-fidelity", type=Path, required=True)
    p.add_argument("--field-validation-results", type=Path, required=True)
    p.add_argument("--field-validation-metadata", type=Path, required=True)
    p.add_argument("--validated-manifest", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/observations/florence_2018_hq_preflight"),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def has_approved(v: Any) -> bool:
    if pd.isna(v):
        return False
    return "A" in {x.strip() for x in str(v).split("|")}


def main() -> None:
    a = parse_args()

    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()) and not a.overwrite:
        raise FileExistsError(f"{a.output_dir} is not empty; use --overwrite.")
    if a.overwrite:
        for p in a.output_dir.iterdir():
            if p.is_file():
                p.unlink()

    manifest = json.loads(a.validated_manifest.read_text(encoding="utf-8"))
    meta = json.loads(a.surrogate_metadata.read_text(encoding="utf-8"))
    val_meta = json.loads(a.field_validation_metadata.read_text(encoding="utf-8"))

    event = pd.read_csv(a.event_observations)
    field = pd.read_csv(a.field_measurements)
    h2q = pd.read_csv(a.h_to_q)
    q2h = pd.read_csv(a.q_to_h)
    fidelity = pd.read_csv(a.reconstruction_fidelity)
    validation = pd.read_csv(a.field_validation_results)

    rows: list[dict[str, Any]] = []

    def record(check: str, passed: bool, severity: str, detail: str) -> None:
        rows.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "severity": severity,
                "detail": detail,
            }
        )

    def warning(check: str, detail: str) -> None:
        rows.append(
            {
                "check": check,
                "status": "WARNING",
                "severity": "WARNING",
                "detail": detail,
            }
        )

    # ------------------------------------------------------------------
    # Fingerprints
    # ------------------------------------------------------------------
    actual_paths = {
        "usgs_event_observations.csv": a.event_observations,
        "usgs_field_measurements_paired_hq.csv": a.field_measurements,
        "florence_effective_h_to_q_surrogate.csv": a.h_to_q,
        "florence_effective_q_to_h_surrogate.csv": a.q_to_h,
        "florence_effective_hq_metadata.json": a.surrogate_metadata,
        "florence_hq_reconstruction_fidelity.csv": a.reconstruction_fidelity,
        "field_hq_validation_results.csv": a.field_validation_results,
        "field_hq_validation_metadata.json": a.field_validation_metadata,
    }

    for name, path in actual_paths.items():
        expected = manifest["artifacts"][name]["sha256"]
        actual = sha256_file(path)
        record(
            f"SHA256_{name}",
            actual == expected,
            "BLOCKING",
            f"expected={expected}; actual={actual}",
        )

    # ------------------------------------------------------------------
    # Metadata identity / status
    # ------------------------------------------------------------------
    record(
        "STEP1E_STATUS",
        meta.get("status") == "PASS_FLORENCE_EFFECTIVE_HQ_SURROGATE_RECONSTRUCTED",
        "BLOCKING",
        str(meta.get("status")),
    )
    record(
        "STEP1F_STATUS",
        val_meta.get("status") == "PASS_FLORENCE_HQ_SURROGATE_FIELD_VALIDATION_COMPLETED",
        "BLOCKING",
        str(val_meta.get("status")),
    )
    record(
        "NOT_OFFICIAL_HISTORICAL_USGS_RATING",
        meta.get("official_usgs_historical_rating") is False,
        "BLOCKING",
        f"official_usgs_historical_rating={meta.get('official_usgs_historical_rating')}",
    )
    record(
        "NO_STEP1E_BLOCKING_QC",
        int(meta.get("blocking_qc_issue_count", -1)) == 0,
        "BLOCKING",
        f"blocking={meta.get('blocking_qc_issue_count')}",
    )
    record(
        "NO_STEP1F_BLOCKING_QC",
        int(val_meta.get("blocking_qc_issue_count", -1)) == 0,
        "BLOCKING",
        f"blocking={val_meta.get('blocking_qc_issue_count')}",
    )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    required_h2q = {
        "gage_height_ft",
        "gage_height_m",
        "rising_discharge_cfs",
        "falling_discharge_cfs",
        "rising_discharge_cms",
        "falling_discharge_cms",
    }
    required_q2h = {
        "discharge_cfs",
        "discharge_cms",
        "rising_gage_height_ft",
        "rising_gage_height_m",
        "falling_gage_height_ft",
        "falling_gage_height_m",
    }
    required_event = {
        "datetime_utc",
        "discharge_cfs",
        "gage_height_ft",
        "00060_qualifiers",
        "00065_qualifiers",
    }

    record(
        "H2Q_SCHEMA",
        required_h2q.issubset(h2q.columns),
        "BLOCKING",
        ",".join(h2q.columns),
    )
    record(
        "Q2H_SCHEMA",
        required_q2h.issubset(q2h.columns),
        "BLOCKING",
        ",".join(q2h.columns),
    )
    record(
        "EVENT_SCHEMA",
        required_event.issubset(event.columns),
        "BLOCKING",
        ",".join(event.columns),
    )

    # ------------------------------------------------------------------
    # Grid and units
    # ------------------------------------------------------------------
    stage_grid = pd.to_numeric(h2q["gage_height_ft"], errors="coerce").to_numpy(float)
    q_grid = pd.to_numeric(q2h["discharge_cfs"], errors="coerce").to_numpy(float)

    record(
        "H2Q_STAGE_GRID_STRICTLY_INCREASING",
        np.all(np.diff(stage_grid) > 0),
        "BLOCKING",
        f"min_step={np.nanmin(np.diff(stage_grid))}",
    )
    record(
        "H2Q_STAGE_GRID_0P01_FT",
        np.allclose(np.diff(stage_grid), 0.01, atol=1e-9),
        "BLOCKING",
        f"median_step={np.nanmedian(np.diff(stage_grid))}",
    )
    record(
        "Q2H_DISCHARGE_GRID_STRICTLY_INCREASING",
        np.all(np.diff(q_grid) > 0),
        "BLOCKING",
        f"min_step={np.nanmin(np.diff(q_grid))}",
    )
    record(
        "Q2H_DISCHARGE_GRID_25_CFS",
        np.allclose(np.diff(q_grid), 25.0, atol=1e-9),
        "BLOCKING",
        f"median_step={np.nanmedian(np.diff(q_grid))}",
    )

    stage_m_err = np.nanmax(
        np.abs(
            pd.to_numeric(h2q["gage_height_m"], errors="coerce").to_numpy(float)
            - stage_grid * FT_TO_M
        )
    )
    record(
        "H2Q_STAGE_UNIT_CONVERSION",
        stage_m_err < 1e-10,
        "BLOCKING",
        f"max_abs_error_m={stage_m_err:.3e}",
    )

    q_cms_err = np.nanmax(
        np.abs(
            pd.to_numeric(q2h["discharge_cms"], errors="coerce").to_numpy(float)
            - q_grid * CFS_TO_CMS
        )
    )
    record(
        "Q2H_DISCHARGE_UNIT_CONVERSION",
        q_cms_err < 1e-8,
        "BLOCKING",
        f"max_abs_error_cms={q_cms_err:.3e}",
    )

    # ------------------------------------------------------------------
    # Branch monotonicity and ranges
    # ------------------------------------------------------------------
    finite_support: dict[str, dict[str, float]] = {}

    for limb in ("rising", "falling"):
        q = pd.to_numeric(h2q[f"{limb}_discharge_cfs"], errors="coerce").to_numpy(float)
        mask = np.isfinite(q)
        qv = q[mask]
        hv = stage_grid[mask]

        record(
            f"{limb.upper()}_H2Q_MONOTONIC",
            len(qv) > 1 and np.all(np.diff(qv) >= -1e-9),
            "BLOCKING",
            f"H={hv.min():.2f}-{hv.max():.2f} ft; Q={qv.min():.1f}-{qv.max():.1f} cfs",
        )

        inverse_h = pd.to_numeric(
            q2h[f"{limb}_gage_height_ft"], errors="coerce"
        ).to_numpy(float)
        imask = np.isfinite(inverse_h)
        inv_h = inverse_h[imask]
        inv_q = q_grid[imask]

        record(
            f"{limb.upper()}_Q2H_MONOTONIC",
            len(inv_h) > 1 and np.all(np.diff(inv_h) >= -1e-9),
            "BLOCKING",
            f"Q={inv_q.min():.1f}-{inv_q.max():.1f} cfs; H={inv_h.min():.3f}-{inv_h.max():.3f} ft",
        )

        finite_support[limb] = {
            "h_to_q_stage_min_ft": float(hv.min()),
            "h_to_q_stage_max_ft": float(hv.max()),
            "h_to_q_discharge_min_cfs": float(qv.min()),
            "h_to_q_discharge_max_cfs": float(qv.max()),
            "q_to_h_discharge_min_cfs": float(inv_q.min()),
            "q_to_h_discharge_max_cfs": float(inv_q.max()),
            "q_to_h_stage_min_ft": float(inv_h.min()),
            "q_to_h_stage_max_ft": float(inv_h.max()),
        }

    # ------------------------------------------------------------------
    # Independently reproduce source peak and paired count
    # ------------------------------------------------------------------
    event["datetime_utc"] = pd.to_datetime(event["datetime_utc"], utc=True, errors="coerce")
    event["discharge_cfs"] = pd.to_numeric(event["discharge_cfs"], errors="coerce")
    event["gage_height_ft"] = pd.to_numeric(event["gage_height_ft"], errors="coerce")

    valid = (
        event["datetime_utc"].notna()
        & event["discharge_cfs"].notna()
        & event["gage_height_ft"].notna()
        & event["00060_qualifiers"].map(has_approved)
        & event["00065_qualifiers"].map(has_approved)
    )
    paired = event.loc[valid].copy()

    record(
        "VALID_PAIRED_EVENT_COUNT",
        len(paired) == int(meta["valid_paired_event_observation_count"]),
        "BLOCKING",
        f"recomputed={len(paired)}; metadata={meta['valid_paired_event_observation_count']}",
    )

    peak_row = paired.loc[paired["gage_height_ft"].idxmax()]
    peak_meta = meta["peak"]
    peak_ok = (
        np.isclose(float(peak_row["gage_height_ft"]), float(peak_meta["gage_height_ft"]))
        and np.isclose(
            float(peak_row["discharge_cfs"]),
            float(peak_meta["discharge_cfs_at_peak_stage_time"]),
        )
        and peak_row["datetime_utc"].isoformat() == peak_meta["datetime_utc"]
    )
    record(
        "SOURCE_PEAK_REPRODUCED",
        peak_ok,
        "BLOCKING",
        (
            f"{peak_row['datetime_utc'].isoformat()}, "
            f"H={peak_row['gage_height_ft']:.3f} ft, "
            f"Q={peak_row['discharge_cfs']:.1f} cfs"
        ),
    )

    # ------------------------------------------------------------------
    # Reconstruction fidelity
    # ------------------------------------------------------------------
    for _, r in fidelity.iterrows():
        limb = str(r["limb"])
        record(
            f"{limb.upper()}_R2_GT_0P999",
            float(r["r2"]) > 0.999,
            "QUALITY",
            f"R2={float(r['r2']):.6f}",
        )
        record(
            f"{limb.upper()}_MAPE_LT_1_PERCENT",
            float(r["mape_percent"]) < 1.0,
            "QUALITY",
            f"MAPE={float(r['mape_percent']):.4f}%",
        )

    # ------------------------------------------------------------------
    # Independent Florence field anchor
    # ------------------------------------------------------------------
    primary = validation.loc[
        validation["validation_role"].eq("PRIMARY_EVENT_PERIOD_FIELD_ANCHOR")
    ].copy()

    record(
        "PRIMARY_EVENT_FIELD_ANCHOR_COUNT",
        len(primary) == 1,
        "QUALITY",
        f"count={len(primary)}",
    )

    if len(primary) == 1:
        r = primary.iloc[0]
        surrogate_pct = abs(float(r["selected_surrogate_percent_error"]))
        direct_pct = abs(float(r["continuous_q_vs_field_percent_error"]))
        stage_error = abs(float(r["continuous_h_vs_field_error_ft"]))

        record(
            "PRIMARY_FIELD_SURROGATE_Q_ERROR_LE_5_PERCENT",
            surrogate_pct <= 5.0,
            "QUALITY",
            f"absolute_error={surrogate_pct:.3f}%",
        )
        record(
            "PRIMARY_FIELD_DIRECT_Q_ERROR_LE_5_PERCENT",
            direct_pct <= 5.0,
            "QUALITY",
            f"absolute_error={direct_pct:.3f}%",
        )
        record(
            "PRIMARY_FIELD_STAGE_ERROR_LE_0P02_FT",
            stage_error <= 0.02,
            "QUALITY",
            f"absolute_error={stage_error:.4f} ft",
        )

        if str(r.get("discharge_measurement_rated")) == "Poor":
            warning(
                "PRIMARY_FIELD_MEASUREMENT_QUALITY",
                (
                    "The Florence field discharge is USGS Approved but rated Poor. "
                    "Retain it as an uncertainty-bounded high-flow anchor."
                ),
            )

    # ------------------------------------------------------------------
    # Operational use-boundary warning
    # ------------------------------------------------------------------
    observed_falling_max = float(
        paired.loc[
            paired["datetime_utc"] > pd.Timestamp(peak_meta["datetime_utc"]),
            "discharge_cfs",
        ].max()
    )
    falling_inverse_max = finite_support["falling"]["q_to_h_discharge_max_cfs"]

    if falling_inverse_max + 1e-9 < observed_falling_max:
        warning(
            "FALLING_Q2H_NEAR_CREST_SUPPORT",
            (
                f"Falling Q->H finite support ends at {falling_inverse_max:,.1f} cfs, "
                f"while post-peak source observations reach {observed_falling_max:,.1f} cfs. "
                "Do not extrapolate. At later implementation, use an explicit near-crest "
                "transition rule or rebuild a direct Q->H branch."
            ),
        )

    # Contextual points are not event validation.
    if "contextual_field_visit_count" in val_meta:
        warning(
            "CONTEXTUAL_FIELD_VISITS",
            (
                f"{val_meta['contextual_field_visit_count']} field visits are outside the "
                "Florence continuous event window. They are context/stability checks only."
            ),
        )

    checks = pd.DataFrame(rows)
    blocking_failures = checks[
        (checks["severity"] == "BLOCKING") & (checks["status"] == "FAIL")
    ]
    quality_failures = checks[
        (checks["severity"] == "QUALITY") & (checks["status"] == "FAIL")
    ]
    warnings = checks[checks["status"] == "WARNING"]

    if len(blocking_failures):
        status = "FAIL_HQ_PREFLIGHT_BLOCKING"
    elif len(quality_failures):
        status = "FAIL_HQ_PREFLIGHT_QUALITY"
    elif len(warnings):
        status = "PASS_HQ_PREFLIGHT_WITH_WARNINGS"
    else:
        status = "PASS_HQ_PREFLIGHT"

    output_csv = a.output_dir / "florence_hq_preflight_checks.csv"
    output_json = a.output_dir / "florence_hq_preflight_metadata.json"
    atomic_csv(checks, output_csv)

    summary = {
        "status": status,
        "scientific_name": meta.get("scientific_name"),
        "official_usgs_historical_rating": meta.get("official_usgs_historical_rating"),
        "site": manifest.get("site"),
        "blocking_failure_count": int(len(blocking_failures)),
        "quality_failure_count": int(len(quality_failures)),
        "warning_count": int(len(warnings)),
        "validated_sha256_manifest": str(a.validated_manifest),
        "finite_support": finite_support,
        "source_peak": {
            "time_utc": peak_row["datetime_utc"].isoformat(),
            "gage_height_ft": float(peak_row["gage_height_ft"]),
            "discharge_cfs": float(peak_row["discharge_cfs"]),
        },
        "primary_field_anchor_count": int(len(primary)),
        "safe_for_next_stage": bool(
            len(blocking_failures) == 0 and len(quality_failures) == 0
        ),
        "use_rule": (
            "Never extrapolate outside the finite branch support. Use rising branch "
            "before/at the hydrograph peak and falling branch after peak. The current "
            "falling Q->H file needs an explicit near-crest transition rule above its "
            "finite support before it is wired into automatic Q->H conversion."
        ),
        "output_checks": str(output_csv),
    }
    atomic_json(summary, output_json)

    print("=" * 100)
    print("FLORENCE 2018 H-Q ARTIFACT PREFLIGHT")
    print("=" * 100)
    print(f"Status                              : {status}")
    print(f"Blocking failures                   : {len(blocking_failures)}")
    print(f"Quality failures                    : {len(quality_failures)}")
    print(f"Warnings                            : {len(warnings)}")
    print()
    print("Finite Q->H support:")
    print(
        f"  Rising                            : "
        f"{finite_support['rising']['q_to_h_discharge_min_cfs']:,.1f} -> "
        f"{finite_support['rising']['q_to_h_discharge_max_cfs']:,.1f} ft3/s"
    )
    print(
        f"  Falling                           : "
        f"{finite_support['falling']['q_to_h_discharge_min_cfs']:,.1f} -> "
        f"{finite_support['falling']['q_to_h_discharge_max_cfs']:,.1f} ft3/s"
    )
    print()
    if len(primary) == 1:
        r = primary.iloc[0]
        print(f"Florence field-anchor Q error       : {abs(float(r['selected_surrogate_percent_error'])):.3f} %")
        print(f"Florence field-anchor stage error   : {abs(float(r['continuous_h_vs_field_error_ft'])):.4f} ft")
    print()
    print(f"Checks                              : {output_csv}")
    print(f"Metadata                            : {output_json}")
    print()
    print(
        "Safe for next scientific stage     : "
        + ("YES" if summary["safe_for_next_stage"] else "NO")
    )

    if len(blocking_failures) or len(quality_failures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
