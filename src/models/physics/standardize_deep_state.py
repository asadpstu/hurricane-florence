#!/usr/bin/env python3
"""Aggregate ERA5-Land layer-4 soil moisture to subcatchments and freeze calibration-period scaling."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--event-blocks", type=Path, required=True)
    p.add_argument("--subcatchments", type=Path, required=True)
    p.add_argument("--expected-subcatchments", type=int, default=37)
    p.add_argument("--area-crs", default="EPSG:32618")
    p.add_argument("--lower-quantile", type=float, default=0.05)
    p.add_argument("--upper-quantile", type=float, default=0.95)
    p.add_argument("--min-scaling-span-m3m3", type=float, default=0.005)
    p.add_argument(
        "--min-subcatchment-grid-coverage-percent", type=float, default=99
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_std():
    p = Path(__file__).with_name("standardize_state.py")
    spec = importlib.util.spec_from_file_location("std_state", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    a = parse_args()
    m = load_std()
    man = pd.read_csv(a.manifest)

    good = {"DOWNLOADED", "EXISTS_SKIPPED"}
    man = man[
        (man.group.astype(str) == "DEEP")
        & man.status.astype(str).isin(good)
    ].copy()
    if man.empty:
        raise RuntimeError("No successful DEEP manifest entries")

    sub = gpd.read_file(a.subcatchments)
    idcol = "subcatchment_id"
    if idcol not in sub:
        raise RuntimeError("subcatchments missing subcatchment_id")
    if sub[idcol].nunique() != a.expected_subcatchments:
        raise RuntimeError("unexpected subcatchment count")

    pieces = []
    weights_ref = None
    for _, r in man.sort_values("month_id").iterrows():
        ds = m.open_standardized(Path(r.output))
        lat = np.asarray(
            ds[m.resolve_coord(ds, ["latitude", "lat"])].values, float
        )
        lon = np.asarray(
            ds[m.resolve_coord(ds, ["longitude", "lon"])].values, float
        )
        if weights_ref is None:
            weights_ref = m.build_weights(lat, lon, sub, idcol, a.area_crs)

        ids, W, weights, spatial, subm = weights_ref
        min_grid_coverage = float(spatial["era5_grid_coverage_percent"].min())
        if min_grid_coverage < a.min_subcatchment_grid_coverage_percent:
            raise RuntimeError(
                "ERA5 deep-state grid coverage below threshold: "
                f"{min_grid_coverage:.6f}% < "
                f"{a.min_subcatchment_grid_coverage_percent:.6f}%"
            )

        swvl4_cube, source_name, units = m.cube(ds, "swvl4")
        vals, den = m.aggregate(swvl4_cube, W)
        times = pd.to_datetime(
            ds[m.resolve_coord(ds, ["valid_time", "time"])].values,
            utc=True,
        )
        for j, sid in enumerate(ids):
            pieces.append(
                pd.DataFrame(
                    {
                        "interval_end_utc": times,
                        "subcatchment_id": sid,
                        "swvl4_m3m3": vals[:, j],
                    }
                )
            )
        ds.close()

    h = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(["interval_end_utc", "subcatchment_id"])
        .sort_values(["interval_end_utc", "subcatchment_id"])
    )

    blocks = pd.read_csv(a.event_blocks)
    phase_upper = blocks.phase.astype(str).str.upper().str.strip()
    cal = blocks[phase_upper.str.startswith("CALIBRATION")]
    if cal.empty:
        raise RuntimeError("No CALIBRATION event blocks")

    intervals = [
        (pd.Timestamp(x.block_start_utc), pd.Timestamp(x.block_end_utc))
        for _, x in cal.iterrows()
    ]
    t = pd.to_datetime(h.interval_end_utc, utc=True)
    use = np.zeros(len(h), bool)
    for s, e in intervals:
        use |= (t >= s) & (t <= e)

    scaling = []
    for sid, g in h[use].groupby("subcatchment_id"):
        v = pd.to_numeric(g.swvl4_m3m3, errors="coerce").dropna()
        lo = float(v.quantile(a.lower_quantile))
        hi = float(v.quantile(a.upper_quantile))
        span = hi - lo
        status = "PASS" if span >= a.min_scaling_span_m3m3 else "FAIL_SPAN"
        scaling.append(
            {
                "subcatchment_id": sid,
                "swvl4_lower_m3m3": lo,
                "swvl4_upper_m3m3": hi,
                "swvl4_span_m3m3": span,
                "status": status,
            }
        )

    sc = pd.DataFrame(scaling)
    if (
        len(sc) != a.expected_subcatchments
        or not sc.status.str.startswith("PASS").all()
    ):
        raise RuntimeError(
            "Deep-state scaling failed for one or more subcatchments"
        )

    mp = sc.set_index("subcatchment_id")
    h["deep_relative_wetness"] = [
        np.clip(
            (v - mp.loc[s, "swvl4_lower_m3m3"])
            / (
                mp.loc[s, "swvl4_upper_m3m3"]
                - mp.loc[s, "swvl4_lower_m3m3"]
            ),
            0,
            1,
        )
        for s, v in zip(h.subcatchment_id, h.swvl4_m3m3)
    ]

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    hp = out / "deep_state_subcatchment_hourly.csv.gz"
    sp = out / "deep_state_scaling_parameters.csv"
    meta = out / "deep_state_metadata.json"

    if not a.overwrite and (hp.exists() or sp.exists()):
        raise FileExistsError(hp)

    h.to_csv(hp, index=False, compression="gzip")
    sc.to_csv(sp, index=False)
    meta.write_text(
        json.dumps(
            {
                "status": "PASS_DEEP_STATE_READY",
                "scaling_period": "CALIBRATION events only (2015-2016)",
                "florence_used": False,
            },
            indent=2,
        )
    )
    print("PASS_DEEP_STATE_READY")


if __name__ == "__main__":
    main()
