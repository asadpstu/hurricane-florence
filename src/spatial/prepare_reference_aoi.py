#!/usr/bin/env python3
"""Recover only the published Copernicus EMSR311 Goldsboro analysis-AOI boundary.

The Copernicus flood delineation is deliberately not exported or used for model validation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

DEFAULT_URL = (
    "https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/"
    "EMSR311/EMSR311_14GOLDSBORO_01DELINEATION_MAP_v1_vector.zip"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--product-url", default=DEFAULT_URL)
    p.add_argument("--gauge-lon", type=float, default=-77.9975)
    p.add_argument("--gauge-lat", type=float, default=35.3375)
    p.add_argument("--projected-crs", default="EPSG:32618")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def norm(s):
    return "".join(c.lower() for c in str(s) if c.isalnum())


def safe(z, d):
    root = d.resolve()
    with zipfile.ZipFile(z) as f:
        for x in f.infolist():
            t = (d / x.filename).resolve()
            if t != root and root not in t.parents:
                raise RuntimeError(f"Unsafe ZIP member {x.filename}")
        f.extractall(d)


def main():
    a = parse_args()
    a.raw_dir.mkdir(parents=True, exist_ok=True)
    z = a.raw_dir / "EMSR311_14GOLDSBORO_01DELINEATION_MAP_v1_vector.zip"
    ex = a.raw_dir / "extracted"

    if not z.exists():
        r = requests.get(
            a.product_url,
            timeout=180,
            headers={"User-Agent": "neuse-florence-research/1.0"},
        )
        r.raise_for_status()
        tmp = z.with_suffix(".zip.partial")
        tmp.write_bytes(r.content)
        os.replace(tmp, z)

    if ex.exists() and a.overwrite:
        shutil.rmtree(ex)
    if not ex.exists():
        ex.mkdir()
        safe(z, ex)

    candidates = []
    for shp in ex.rglob("*.shp"):
        if "areaofinterest" in norm(shp.stem):
            candidates.append((shp, None, shp.stem))

    for gpkg in ex.rglob("*.gpkg"):
        try:
            for lyr in gpd.list_layers(gpkg).name.astype(str):
                if "areaofinterest" in norm(lyr):
                    candidates.append((gpkg, lyr, lyr))
        except Exception:
            pass

    if not candidates:
        raise RuntimeError(
            "Could not locate Copernicus areaOfInterestA boundary in EMSR311 vector package"
        )

    path, layer, label = candidates[0]
    g = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if g.empty or g.crs is None:
        raise RuntimeError("AOI layer is empty or has no CRS")

    g = g[["geometry"]].copy()
    g["geometry"] = g.geometry.make_valid()
    g = g[~g.geometry.is_empty]
    g = g.dissolve().reset_index(drop=True)

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ao = out / "florence_analysis_aoi.geojson"
    meta = out / "reference_aoi_metadata.json"
    inv = out / "copernicus_vector_inventory.csv"

    if ao.exists() and not a.overwrite:
        raise FileExistsError(ao)

    g.to_file(ao, driver="GeoJSON")
    gm = g.to_crs(a.projected_crs)
    pd.DataFrame(
        [
            {
                "source_path": str(path),
                "source_layer": layer or "",
                "source_label": label,
                "area_km2": float(gm.geometry.area.sum() / 1e6),
            }
        ]
    ).to_csv(inv, index=False)

    meta.write_text(
        json.dumps(
            {
                "status": "PASS_REFERENCE_AOI_READY",
                "source": "Copernicus EMSR311 AOI14 Goldsboro areaOfInterestA",
                "role": (
                    "AOI boundary only; flood delineation not retained or used "
                    "for validation"
                ),
                "output": str(ao),
            },
            indent=2,
        )
    )
    print("PASS_REFERENCE_AOI_READY")


if __name__ == "__main__":
    main()
