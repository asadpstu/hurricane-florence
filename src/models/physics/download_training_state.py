#!/usr/bin/env python3
"""Download monthly ERA5-Land state/meteorology/deep-soil files needed by selected training events."""
from __future__ import annotations

import argparse
import calendar
import json
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

DATASET = "reanalysis-era5-land"
STATE = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
]
MET = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
]
DEEP = ["volumetric_soil_water_layer_4"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--event-blocks", type=Path, required=True)
    p.add_argument("--subcatchments", type=Path, required=True)
    p.add_argument("--antecedent-days", type=int, default=45)
    p.add_argument("--bbox-padding-deg", type=float, default=0.15)
    p.add_argument("--start-limit", default="2015-01-01T00:00:00Z")
    p.add_argument("--end-limit", default="2018-01-01T00:00:00Z")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite-manifests", action="store_true")
    return p.parse_args()


def months(a, b):
    return pd.period_range(a.to_period("M"), b.to_period("M"), freq="M")


def main():
    a = parse_args()
    for p in [a.event_blocks, a.subcatchments]:
        if not p.exists():
            raise FileNotFoundError(p)

    b = pd.read_csv(a.event_blocks)
    starts = pd.to_datetime(b.block_start_utc, utc=True)
    ends = pd.to_datetime(b.block_end_utc, utc=True)
    start = max(
        starts.min() - pd.Timedelta(days=a.antecedent_days),
        pd.Timestamp(a.start_limit),
    )
    end = min(
        ends.max(),
        pd.Timestamp(a.end_limit) - pd.Timedelta(seconds=1),
    )

    sub = gpd.read_file(a.subcatchments).to_crs(4326)
    minx, miny, maxx, maxy = sub.total_bounds
    pad = a.bbox_padding_deg
    area = [
        min(90, maxy + pad),
        max(-180, minx - pad),
        max(-90, miny - pad),
        min(180, maxx + pad),
    ]

    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    deep = []

    try:
        import cdsapi
    except ImportError as e:
        raise RuntimeError("cdsapi is required") from e

    client = cdsapi.Client()
    for per in months(start, end):
        y, m = per.year, per.month
        mid = f"{y:04d}-{m:02d}"
        nd = calendar.monthrange(y, m)[1]
        days = [f"{d:02d}" for d in range(1, nd + 1)]
        times = [f"{h:02d}:00" for h in range(24)]

        for group, vars_, destrows in [
            ("STATE", STATE, rows),
            ("MET", MET, rows),
            ("DEEP", DEEP, deep),
        ]:
            out = a.output_dir / f"{mid}_{group.lower()}.nc"
            status = "EXISTS_SKIPPED"
            if not out.exists():
                req = {
                    "variable": vars_,
                    "year": str(y),
                    "month": f"{m:02d}",
                    "day": days,
                    "time": times,
                    "area": area,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                }
                tmp = out.with_suffix(".nc.partial")
                tmp.unlink(missing_ok=True)
                client.retrieve(DATASET, req, str(tmp))
                os.replace(tmp, out)
                status = "DOWNLOADED"

            destrows.append(
                {
                    "month_id": mid,
                    "group": group,
                    "status": status,
                    "output": str(out),
                }
            )

    def save(name, data):
        p = a.output_dir / name
        if p.exists() and not a.overwrite_manifests:
            raise FileExistsError(
                f"{p} exists; use --overwrite-manifests"
            )
        pd.DataFrame(data).to_csv(p, index=False)

    save("era5_land_state_manifest.csv", rows)
    save("era5_land_deep_manifest.csv", deep)
    (a.output_dir / "era5_land_training_download_metadata.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "dataset": DATASET,
                "start": str(start),
                "end": str(end),
                "area_nwse": area,
            },
            indent=2,
        )
    )
    print("PASS_ERA5_LAND_TRAINING_DOWNLOAD")


if __name__ == "__main__":
    main()
