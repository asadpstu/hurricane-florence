#!/usr/bin/env python3
"""Rasterize matched NOAA FIM levels from comparison.py on a common projected grid."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--noaa-root", type=Path, required=True)
    p.add_argument("--crs", default="EPSG:32618")
    p.add_argument("--resolution-m", type=float, default=10)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def levels(root):
    x = {}
    for p in root.rglob("*.shp"):
        v = (
            float(p.stem)
            if re.fullmatch(r"\d+", p.stem)
            else (
                float(p.stem.replace("_5", ".5"))
                if re.fullmatch(r"\d+_5", p.stem)
                else None
            )
        )
        if v is not None:
            x[v] = p
    return x


def main():
    a = parse_args()
    d = pd.read_csv(a.comparison)
    lib = levels(a.noaa_root)
    if not lib:
        raise RuntimeError("No NOAA FIM stage shapefiles")

    geoms = {
        v: gpd.read_file(p).to_crs(a.crs).geometry.union_all()
        for v, p in lib.items()
    }
    union = gpd.GeoSeries(list(geoms.values()), crs=a.crs).union_all()
    minx, miny, maxx, maxy = union.bounds
    r = a.resolution_m
    minx = math.floor(minx / r) * r
    miny = math.floor(miny / r) * r
    maxx = math.ceil(maxx / r) * r
    maxy = math.ceil(maxy / r) * r
    W = int(round((maxx - minx) / r))
    H = int(round((maxy - miny) / r))
    tr = from_origin(minx, maxy, r, r)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in d.iterrows():
        date = str(row["date"])
        for source, pref in [
            ("USGS", "usgs"),
            ("Physics", "physics"),
            ("ML", "ml"),
        ]:
            lev = pd.to_numeric(
                pd.Series([row.get(f"{pref}_noaa_fim_level_ft")]),
                errors="coerce",
            ).iloc[0]
            if not np.isfinite(lev):
                continue

            lev = float(lev)
            geom = geoms.get(lev)
            if geom is None:
                raise RuntimeError(f"Missing NOAA FIM geometry {lev}")

            arr = rasterize(
                [(geom, 1)],
                out_shape=(H, W),
                transform=tr,
                fill=0,
                dtype="uint8",
            )
            out = a.output_dir / (
                f"{date}_{pref}_fim_{str(lev).replace('.', '_')}_ft.tif"
            )
            if out.exists() and not a.overwrite:
                raise FileExistsError(out)

            with rasterio.open(
                out,
                "w",
                driver="GTiff",
                height=H,
                width=W,
                count=1,
                dtype="uint8",
                crs=a.crs,
                transform=tr,
                nodata=0,
                compress="deflate",
                tiled=True,
            ) as ds:
                ds.write(arr, 1)

            rows.append(
                {
                    "date": date,
                    "source": source,
                    "noaa_fim_level_ft": lev,
                    "raster": str(out),
                    "flood_area_km2": float(arr.sum() * r * r / 1e6),
                }
            )

    pd.DataFrame(rows).to_csv(
        a.output_dir / "noaa_fim_map_inventory.csv", index=False
    )
    print("PASS_NOAA_FIM_FLOOD_MAPS_COMPLETE")


if __name__ == "__main__":
    main()
