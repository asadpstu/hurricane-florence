#!/usr/bin/env python3
"""Recovered P3.6B3 deep-groundwater calibration wrapper.

Hydrologic change relative to the parent core:
    initial_baseflow_mm_h = deep_baseflow_floor_mm_h
                          + deep_baseflow_range_mm_h * deep_relative_wetness

The local base reservoir and existing channel-routing stores are initialized
at the corresponding steady-flow state. Observed discharge is never used for
state initialization. Florence is not used in calibration or validation.

The deep-state hydrologic patch and parameter bounds are directly recovered
from the historical source audit. The optimization driver is the retained
parent event-balanced implementation in this cleaned codebase.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# The recovered deep-state parent already implements the exact two-parameter
# initial slow-flow state used by P3.6B3 and delegates all other equations to
# the retained real-A4-routing parent core.
import _physics_deep_state_parent as recovered

BUILD = "PHYSICS_P3_6B3_DEEP_GROUNDWATER_RECOVERED_V1"

ALIASES = {
    "physics_v2_parameters.json": "physics_p3_6b3_parameters.json",
    "physics_v2_metadata.json": "physics_p3_6b3_metadata.json",
    "physics_v2_summary_metrics.csv": "physics_p3_6b3_summary_metrics.csv",
    "physics_v2_block_metrics.csv": "physics_p3_6b3_block_metrics.csv",
    "physics_v2_peak_timing_audit.csv": "physics_p3_6b3_peak_timing_audit.csv",
    "physics_v2_hourly.csv": "physics_p3_6b3_hourly.csv",
    "physics_v2_optimizer_trace.csv": "physics_p3_6b3_optimizer_trace.csv",
    "physics_v2_optimizer_generations.csv": "physics_p3_6b3_optimizer_generations.csv",
    "physics_v2_parameter_bounds.csv": "physics_p3_6b3_parameter_bounds.csv",
    "physics_v2_qc.csv": "physics_p3_6b3_qc.csv",
}


def arg_value(flag: str) -> str | None:
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    return sys.argv[i + 1] if i + 1 < len(sys.argv) else None


def make_aliases(output_dir: Path) -> None:
    for src_name, dst_name in ALIASES.items():
        src = output_dir / src_name
        if src.exists():
            shutil.copy2(src, output_dir / dst_name)

    meta_path = output_dir / "physics_p3_6b3_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta.update(
            {
                "script_build": BUILD,
                "historical_branch": "P3.6B3 deep-groundwater/baseflow initialization",
                "hydrologic_change": "initial slow-flow state only",
                "deep_baseflow_equation": (
                    "deep_baseflow_floor_mm_h + deep_baseflow_range_mm_h * deep_relative_wetness"
                ),
                "deep_baseflow_parameter_bounds": {
                    "deep_baseflow_floor_mm_h": [0.0, 0.08],
                    "deep_baseflow_range_mm_h": [0.01, 0.22],
                },
                "observed_q_used_for_initialization": False,
                "florence_used": False,
                "recovery_note": (
                    "Hydrologic patch, bounds and historical event-window contract are recovered. "
                    "The cleaned retained parent supplies the event-balanced optimizer driver."
                ),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def main() -> None:
    out = arg_value("--output-dir")
    if out is None:
        raise RuntimeError("--output-dir is required")

    recovered.BUILD = BUILD

    exit_code = 0
    try:
        recovered.main()
    except SystemExit as exc:
        exit_code = 0 if exc.code is None else int(exc.code)
    finally:
        make_aliases(Path(out))

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
