#!/usr/bin/env python3
"""Extract the named Neuse River mainstem from NHDPlus HR across the full analysis AOI."""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gdb", type=Path, required=True)
    p.add_argument("--aoi", type=Path, required=True)
    p.add_argument("--river-name", default="Neuse River")
    p.add_argument("--gauge-lon", type=float, default=-77.9975)
    p.add_argument("--gauge-lat", type=float, default=35.3375)
    p.add_argument("--projected-crs", default="EPSG:32618")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    flow = gpd.read_file(a.gdb, layer="NHDFlowline")
    aoi = gpd.read_file(a.aoi)

    if flow.crs is None or aoi.crs is None:
        raise RuntimeError("Missing CRS")

    # NHDPlus HR geodatabases vary in field-name casing
    # (for example GNIS_Name vs GNIS_NAME). Resolve case-insensitively
    # without changing any hydrography-selection logic.
    column_lookup = {str(c).casefold(): c for c in flow.columns}
    name = next(
        (
            column_lookup[candidate.casefold()]
            for candidate in ["GNIS_Name", "GNIS_NAME", "gnis_name", "Name", "name"]
            if candidate.casefold() in column_lookup
        ),
        None,
    )
    if name is None:
        raise RuntimeError(
            f"Cannot resolve river-name field; columns={list(flow.columns)}"
        )

    x = flow[
        flow[name]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq(a.river_name.casefold())
    ].to_crs(a.projected_crs)
    poly = aoi.to_crs(a.projected_crs).geometry.union_all()
    x = x[x.geometry.intersects(poly)].copy()
    x["geometry"] = x.geometry.intersection(poly)
    x = x[~x.geometry.is_empty]

    if x.empty:
        raise RuntimeError(
            f"No {a.river_name} NHDFlowline segments intersect AOI"
        )

    gp = a.output_dir / "full_aoi_hydrography.gpkg"
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if gp.exists() and not a.overwrite:
        raise FileExistsError(gp)
    if gp.exists():
        gp.unlink()

    x.to_file(gp, layer="neuse_mainstem", driver="GPKG")
    gauge = gpd.GeoDataFrame(
        {"site": ["02089000"]},
        geometry=[Point(a.gauge_lon, a.gauge_lat)],
        crs=4326,
    ).to_crs(a.projected_crs)
    gauge.to_file(gp, layer="usgs_02089000", driver="GPKG")
    print("PASS_FULL_AOI_MAINSTEM_READY")


if __name__ == "__main__":
    main()
