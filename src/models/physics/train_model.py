#!/usr/bin/env python3
"""Public retained Physics trainer. Uses the preserved Physics V3.5 runtime and writes generic artifact names."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

ALIASES = {
    "physics_v2_parameters.json": "parameters.json",
    "physics_v2_metadata.json": "metadata.json",
    "physics_v2_block_metrics.csv": "physics_block_metrics.csv",
    "physics_v2_hourly.csv": "physics_hourly.csv",
    "physics_v2_optimizer_generations.csv": "physics_optimizer_generations.csv",
    "physics_v2_optimizer_trace.csv": "physics_optimizer_trace.csv",
    "physics_v2_parameter_bounds.csv": "physics_parameter_bounds.csv",
    "physics_v2_peak_timing_audit.csv": "physics_peak_timing_audit.csv",
    "physics_v2_qc.csv": "physics_qc.csv",
    "physics_v2_summary_metrics.csv": "physics_summary_metrics.csv",
}


def val(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    return Path(sys.argv[i + 1]) if i + 1 < len(sys.argv) else None


def load():
    p = Path(__file__).with_name("physics_core.py")
    spec = importlib.util.spec_from_file_location("physics_core_runtime", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _alias_outputs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for src, dst in ALIASES.items():
        s = out / src
        if not s.exists():
            continue
        d = out / dst
        tmp = d.with_name(d.name + ".partial")
        shutil.copy2(s, tmp)
        os.replace(tmp, d)
        s.unlink()


def main():
    runtime = load()
    ns, counts = runtime.load_v3_5_namespace()

    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "usage: train_model.py --deep-state PATH --forcing PATH "
            "--event-blocks PATH --q-target-mask PATH "
            "--dynamic-state-et0 PATH --routing-features PATH "
            "--routing PATH [training options] --output-dir PATH "
            "[--overwrite]"
        )
        print(
            "Retained Physics trainer: 2015-2016 calibration, 2017 holdout; "
            "Florence excluded."
        )
        return

    out = val("--output-dir")
    if out is None:
        raise RuntimeError("--output-dir is required")

    exit_code = 0
    try:
        ns["main"]()
    except SystemExit as exc:
        exit_code = 0 if exc.code is None else int(exc.code)
    finally:
        _alias_outputs(out)

    if exit_code != 0:
        raise SystemExit(exit_code)

    print("PASS_RETAINED_PHYSICS_MODEL_TRAINED")


if __name__ == "__main__":
    main()
