#!/usr/bin/env python3
"""Create QGIS-ready 10 m impacted-asset rasters for each source's peak NOAA FIM polygon."""
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
    p.add_argument("--peak-summary", type=Path, required=True)
    p.add_argument("--noaa-root", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--resolution-m", type=float, default=10)
    p.add_argument("--crs", default="EPSG:32618")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def find(root, lev):
    stem = (
        str(int(lev))
        if float(lev).is_integer()
        else str(lev).replace(".5", "_5")
    )
    x = list(root.rglob(stem + ".shp"))
    return x[0] if x else None


def main():
    a = parse_args()
    s = pd.read_csv(a.peak_summary)
    osm = a.input_dir / "osm_current_exposure.gpkg"
    b = gpd.read_file(osm, layer="buildings").to_crs(a.crs)
    roads = gpd.read_file(osm, layer="roads").to_crs(a.crs)
    crit = gpd.read_file(osm, layer="critical_facilities").to_crs(a.crs)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in s.iterrows():
        lev = float(row.noaa_fim_level_ft)
        shp = find(a.noaa_root, lev)
        if shp is None:
            raise FileNotFoundError(f"FIM level {lev}")

        g = gpd.read_file(shp).to_crs(a.crs).geometry.union_all()
        minx, miny, maxx, maxy = g.bounds
        r = a.resolution_m
        minx = math.floor(minx / r) * r
        miny = math.floor(miny / r) * r
        maxx = math.ceil(maxx / r) * r
        maxy = math.ceil(maxy / r) * r
        W = int(round((maxx - minx) / r))
        H = int(round((maxy - miny) / r))
        tr = from_origin(minx, maxy, r, r)
        flood = rasterize(
            [(g, 1)],
            out_shape=(H, W),
            transform=tr,
            fill=0,
            dtype="uint8",
        )

        def rast(gdf, val):
            # IMPORTANT: select features that intersect the NOAA FIM polygon,
            # then CLIP their geometry to the FIM polygon before rasterizing.
            # The previous implementation rasterized the complete geometry of
            # every intersecting road/building, which allowed impacted-asset
            # pixels to extend outside the NOAA FIM inundation footprint.
            x = gdf[gdf.geometry.intersects(g)].copy()
            if not len(x):
                return np.zeros((H, W), dtype="uint8")

            x["geometry"] = x.geometry.intersection(g)
            x = x[
                x.geometry.notna()
                & ~x.geometry.is_empty
            ].copy()
            if not len(x):
                return np.zeros((H, W), dtype="uint8")

            arr = rasterize(
                [(z, val) for z in x.geometry],
                out_shape=(H, W),
                transform=tr,
                fill=0,
                dtype="uint8",
                all_touched=True,
            )

            # Enforce the exact rasterized NOAA FIM footprint as the final
            # visualization mask. This also prevents all_touched edge pixels
            # from appearing outside flood_extent.tif.
            arr[flood == 0] = 0
            return arr

        rb = rast(b, 3)
        rr = rast(roads, 2)
        rc = rast(crit, 4)
        combo = flood.copy()
        combo[rr > 0] = 2
        combo[rb > 0] = 3
        combo[rc > 0] = 4

        # Strict containment QA: no impacted-asset raster may contain a
        # non-zero pixel outside the NOAA FIM flood raster.
        for label, arr in [
            ("buildings", rb),
            ("roads", rr),
            ("critical_facilities", rc),
        ]:
            outside = int(np.count_nonzero((arr > 0) & (flood == 0)))
            if outside:
                raise RuntimeError(
                    f"{row.source} {label}: {outside} impacted pixels "
                    "fall outside the NOAA FIM flood extent"
                )

        slug = str(row.source).lower().replace(" ", "_")
        od = a.output_dir / slug
        od.mkdir(exist_ok=True)

        outputs = [
            ("flood_extent.tif", flood),
            ("impacted_buildings.tif", (rb > 0).astype("uint8")),
            ("impacted_roads.tif", (rr > 0).astype("uint8")),
            (
                "impacted_critical_facilities.tif",
                (rc > 0).astype("uint8"),
            ),
            ("impacted_assets_combined.tif", combo),
        ]
        for name, arr in outputs:
            p = od / name
            if p.exists() and not a.overwrite:
                raise FileExistsError(p)
            with rasterio.open(
                p,
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
                tiled=(W >= 16 and H >= 16),
            ) as ds:
                ds.write(arr, 1)

    print("PASS_NOAA_FIM_ASSET_RASTERS_COMPLETE")


if __name__ == "__main__":
    main()
