#!/usr/bin/env python3
"""Compute daily and peak exposure for matched NOAA FIM polygons."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

NLCD = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--noaa-root", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--crs", default="EPSG:32618")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def lib(root, crs):
    out = {}
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
            out[v] = gpd.read_file(p).to_crs(crs).geometry.union_all()
    return out


def raster_sum(path, geom, mode):
    with rasterio.open(path) as src:
        g = gpd.GeoSeries([geom], crs="EPSG:32618").to_crs(src.crs).iloc[0]
        d, _ = mask(src, [mapping(g)], crop=True, filled=False)
        vals = d[0].compressed().astype(float)
        vals = vals[np.isfinite(vals)]

        if mode == "sum":
            return float(vals[vals > 0].sum())

        pix = abs(
            src.transform.a * src.transform.e
            - src.transform.b * src.transform.d
        ) / 1e6
        if mode == "imp":
            return float(np.clip(vals, 0, 100).sum() / 100 * pix)

        cls, cnt = np.unique(vals.astype(int), return_counts=True)
        return {
            int(c): float(n * pix)
            for c, n in zip(cls, cnt)
            if c != 0
        }


def main():
    a = parse_args()
    d = pd.read_csv(a.comparison)
    L = lib(a.noaa_root, a.crs)
    if not L:
        raise RuntimeError("No NOAA FIM polygons")

    wp = a.input_dir / "worldpop_usa_2018_population_100m.tif"
    lc = a.input_dir / "nlcd_2016_landcover_30m.tif"
    imp = a.input_dir / "nlcd_2016_impervious_30m.tif"
    osm = a.input_dir / "osm_current_exposure.gpkg"
    for p in [wp, lc, imp, osm]:
        if not p.exists():
            raise FileNotFoundError(p)

    b = gpd.read_file(osm, layer="buildings").to_crs(a.crs)
    roads = gpd.read_file(osm, layer="roads").to_crs(a.crs)
    crit = gpd.read_file(osm, layer="critical_facilities").to_crs(a.crs)
    cache = {}
    records = []
    lcrecs = []

    def impact(level):
        if level in cache:
            return cache[level]

        g = L[level]
        cands = roads[roads.geometry.intersects(g)]
        rr = (
            float(cands.geometry.intersection(g).length.sum() / 1000)
            if len(cands)
            else 0.0
        )
        x = {
            "flood_area_km2": float(g.area / 1e6),
            "affected_population_est": raster_sum(wp, g, "sum"),
            "flooded_buildings": int(b.geometry.intersects(g).sum()),
            "flooded_road_length_km": rr,
            "affected_critical_facilities": int(
                crit.geometry.intersects(g).sum()
            ),
            "impervious_equivalent_area_km2": raster_sum(imp, g, "imp"),
            "landcover": raster_sum(lc, g, "lc"),
        }
        cache[level] = x
        return x

    for _, r in d.iterrows():
        for source, pref in [
            ("USGS", "usgs"),
            ("Physics", "physics"),
            ("ML", "ml"),
        ]:
            lev = pd.to_numeric(
                pd.Series([r.get(f"{pref}_noaa_fim_level_ft")]),
                errors="coerce",
            ).iloc[0]
            rec = {
                "date": r.date,
                "source": source,
                "discharge_m3s": r.get(f"{pref}_q_m3s"),
                "stage_ft": r.get(f"{pref}_stage_ft"),
                "wse_navd88_ft": r.get(f"{pref}_wse_navd88_ft"),
                "noaa_fim_level_ft": lev,
                "noaa_status": r.get(f"{pref}_noaa_status"),
            }

            if np.isfinite(lev) and float(lev) in L:
                x = impact(float(lev))
                rec.update({k: v for k, v in x.items() if k != "landcover"})
                for code, area in x["landcover"].items():
                    lcrecs.append(
                        {
                            "date": r.date,
                            "source": source,
                            "noaa_fim_level_ft": float(lev),
                            "nlcd_class": code,
                            "nlcd_class_name": NLCD.get(code, f"Class {code}"),
                            "flooded_area_km2": area,
                        }
                    )
            else:
                for k in [
                    "flood_area_km2",
                    "affected_population_est",
                    "flooded_buildings",
                    "flooded_road_length_km",
                    "affected_critical_facilities",
                    "impervious_equivalent_area_km2",
                ]:
                    rec[k] = np.nan

            records.append(rec)

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame(records)
    lcdf = pd.DataFrame(lcrecs)
    daily.to_csv(out / "daily_impact.csv", index=False)
    lcdf.to_csv(out / "daily_landcover.csv", index=False)

    peaks = []
    groups = daily.dropna(subset=["flood_area_km2"]).groupby("source")
    for source, g in groups:
        row = g.loc[g.flood_area_km2.idxmax()].copy()
        for col, codes in {
            "cultivated_crops_km2": [82],
            "wetlands_km2": [90, 95],
            "open_water_km2": [11],
            "developed_km2": [21, 22, 23, 24],
            "forest_km2": [41, 42, 43],
        }.items():
            row[col] = lcdf[
                (lcdf.source == source)
                & (lcdf.date == row.date)
                & lcdf.nlcd_class.isin(codes)
            ].flooded_area_km2.sum()
        peaks.append(row)

    pd.DataFrame(peaks).to_csv(out / "peak_impact_summary.csv", index=False)
    print("PASS_NOAA_FIM_IMPACT_COMPLETE")


if __name__ == "__main__":
    main()
