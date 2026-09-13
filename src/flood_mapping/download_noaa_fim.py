#!/usr/bin/env python3
"""Download the NOAA/NWPS partner FIM shapefile archive for a gauge and normalize the polygon library."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import zipfile
from pathlib import Path

import requests


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    p.add_argument("--wfo", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_extract(z, d):
    root = d.resolve()
    with zipfile.ZipFile(z) as f:
        for x in f.infolist():
            t = (d / x.filename).resolve()
            if t != root and root not in t.parents:
                raise RuntimeError(f"Unsafe ZIP member: {x.filename}")
        f.extractall(d)


def level(stem):
    if re.fullmatch(r"\d+", stem):
        return float(stem)
    if re.fullmatch(r"\d+_5", stem):
        return float(stem.replace("_5", ".5"))
    return None


def main():
    a = parse_args()
    site = a.site.lower()
    wfo = a.wfo.lower()
    url = (
        f"https://water.noaa.gov/resources/downloads/fim/{wfo}/{site}/"
        f"shapefile/{site}_shapefiles.zip"
    )

    if a.output_dir.exists() and a.overwrite:
        shutil.rmtree(a.output_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    z = a.output_dir / f"{site}_shapefiles.zip"

    if not z.exists():
        tmp = z.with_suffix(".zip.partial")
        with requests.get(
            url,
            stream=True,
            timeout=180,
            headers={"User-Agent": "neuse-florence-research/1.0"},
        ) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for c in r.iter_content(1024 * 1024):
                    if c:
                        f.write(c)
        os.replace(tmp, z)

    extracted = a.output_dir / "shapefiles"
    safe_extract(z, extracted)
    shps = [
        p for p in extracted.rglob("*.shp") if level(p.stem) is not None
    ]
    if not shps:
        raise RuntimeError(
            f"NOAA archive contained no stage polygon shapefiles: {url}"
        )

    canonical = extracted / "shp" / "ahps" / "inundation" / site / "polygons"
    canonical.mkdir(parents=True, exist_ok=True)
    for shp in shps:
        if shp.parent.resolve() == canonical.resolve():
            continue
        for side in shp.parent.glob(shp.stem + ".*"):
            shutil.copy2(side, canonical / side.name)

    levels = sorted(
        {
            level(p.stem)
            for p in canonical.glob("*.shp")
            if level(p.stem) is not None
        }
    )
    if len(levels) < 2:
        raise RuntimeError("NOAA FIM library has too few levels after extraction")

    print(f"NOAA FIM source: {url}")
    print(f"Levels: {levels[0]:.1f}-{levels[-1]:.1f} ft ({len(levels)})")
    print("PASS_NOAA_FIM_LIBRARY_READY")


if __name__ == "__main__":
    main()
