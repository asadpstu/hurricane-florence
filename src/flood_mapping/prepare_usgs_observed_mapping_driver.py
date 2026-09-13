#!/usr/bin/env python3
"""Prepare paired USGS observed discharge + observed stage for flood mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usgs-observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve(df, candidates, label):
    lookup = {str(c).casefold(): c for c in df.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    raise RuntimeError(
        f"Cannot resolve {label}; tried={candidates}; columns={list(df.columns)}"
    )


def main():
    a = parse_args()

    if not a.usgs_observations.exists():
        raise FileNotFoundError(a.usgs_observations)
    if a.output.exists() and not a.overwrite:
        raise FileExistsError(a.output)

    df = pd.read_csv(a.usgs_observations)

    tcol = resolve(
        df,
        ["datetime_utc", "interval_end_utc", "timestamp"],
        "timestamp",
    )
    qcol = resolve(
        df,
        ["discharge_cms", "q_obs_m3s", "q_mean_m3s"],
        "USGS discharge",
    )
    hcol = resolve(
        df,
        ["gage_height_ft", "stage_ft"],
        "USGS gage height",
    )

    out = pd.DataFrame({
        "interval_end_utc": pd.to_datetime(
            df[tcol], utc=True, errors="coerce"
        ),
        "q_pred_m3s": pd.to_numeric(df[qcol], errors="coerce"),
        "gage_height_ft": pd.to_numeric(df[hcol], errors="coerce"),
    })

    finite = (
        out["interval_end_utc"].notna()
        & np.isfinite(out["q_pred_m3s"].to_numpy(float))
        & np.isfinite(out["gage_height_ft"].to_numpy(float))
    )
    out = out[finite].copy()

    out = (
        out.sort_values("interval_end_utc")
        .drop_duplicates("interval_end_utc", keep="last")
    )

    if out.empty:
        raise RuntimeError("No finite paired USGS discharge-stage rows.")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)

    peak = out.loc[out["q_pred_m3s"].idxmax()]

    print("=" * 96)
    print("USGS OBSERVED DISCHARGE + STAGE FLOOD-MAPPING DRIVER")
    print("=" * 96)
    print(f"Rows                                : {len(out)}")
    print(f"Start                               : {out['interval_end_utc'].min()}")
    print(f"End                                 : {out['interval_end_utc'].max()}")
    print(f"Peak Q                              : {peak['q_pred_m3s']:.3f} m3/s")
    print(f"Observed stage at peak-Q timestamp  : {peak['gage_height_ft']:.3f} ft")
    print("Q->H surrogate used for USGS stage : NO")
    print(f"Output                              : {a.output}")
    print("Status                              : PASS_USGS_OBSERVED_MAPPING_DRIVER_READY")


if __name__ == "__main__":
    main()
