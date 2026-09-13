#!/usr/bin/env python3
"""Download a seamless AOI DEM from the USGS 3DEP Elevation ArcGIS ImageServer.

Robustness behavior:
- preserves the requested final resolution/CRS;
- retries transient request/server failures;
- validates that responses are actual TIFF rasters;
- automatically subdivides a failed request into smaller pixel-aligned tiles.
"""
from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.mask import mask
from rasterio.merge import merge

SERVICE = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)

TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aoi", type=Path, required=True)
    p.add_argument("--resolution-m", type=float, default=10)
    p.add_argument("--target-crs", default="EPSG:32618")
    p.add_argument("--max-tile-pixels", type=int, default=4000)
    p.add_argument(
        "--min-tile-pixels",
        type=int,
        default=500,
        help="Smallest tile dimension used by automatic failure subdivision.",
    )
    p.add_argument(
        "--request-retries",
        type=int,
        default=3,
        help="Retries per tile before automatically subdividing it.",
    )
    p.add_argument("--request-timeout-seconds", type=int, default=180)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _is_tiff(content: bytes) -> bool:
    return (
        len(content) >= 4
        and (
            content[:4] == b"II*\x00"
            or content[:4] == b"MM\x00*"
            or content[:4] == b"II+\x00"   # BigTIFF little endian
            or content[:4] == b"MM\x00+"   # BigTIFF big endian
        )
    )


def _bbox_for_window(minx, maxy, res, c0, r0, w, h):
    x0 = minx + c0 * res
    x1 = x0 + w * res
    ytop = maxy - r0 * res
    ybot = ytop - h * res
    return x0, ybot, x1, ytop


def _split_window(c0, r0, w, h):
    """Split a pixel window into up to four non-overlapping children."""
    if w <= 1 and h <= 1:
        return []

    if w > 1:
        w1 = w // 2
        w2 = w - w1
    else:
        w1, w2 = w, 0

    if h > 1:
        h1 = h // 2
        h2 = h - h1
    else:
        h1, h2 = h, 0

    out = []
    for rr, hh in ((r0, h1), (r0 + h1, h2)):
        if hh <= 0:
            continue
        for cc, ww in ((c0, w1), (c0 + w1, w2)):
            if ww <= 0:
                continue
            out.append((cc, rr, ww, hh))
    return out


def _download_window(
    *,
    session,
    td,
    minx,
    maxy,
    res,
    epsg,
    c0,
    r0,
    w,
    h,
    min_tile_pixels,
    request_retries,
    timeout_seconds,
    counter,
):
    bbox = _bbox_for_window(minx, maxy, res, c0, r0, w, h)
    params = {
        "bbox": ",".join(f"{v:.10f}" for v in bbox),
        "bboxSR": epsg,
        "size": f"{w},{h}",
        "imageSR": epsg,
        "format": "tiff",
        "pixelType": "F32",
        "noData": -9999,
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    last_error = None
    for attempt in range(1, request_retries + 1):
        try:
            rr = session.get(
                SERVICE,
                params=params,
                timeout=timeout_seconds,
                headers={"User-Agent": "neuse-florence-research/1.0"},
            )

            if rr.status_code >= 400:
                detail = rr.text[:300].replace("\n", " ") if rr.content else ""
                err = RuntimeError(
                    f"HTTP {rr.status_code} for {w}x{h} tile "
                    f"(c0={c0}, r0={r0}); {detail}"
                )
                if rr.status_code not in TRANSIENT_HTTP:
                    raise err
                raise err

            if not _is_tiff(rr.content):
                detail = rr.text[:300].replace("\n", " ") if rr.content else ""
                raise RuntimeError(
                    f"USGS returned a non-TIFF response for {w}x{h} tile "
                    f"(c0={c0}, r0={r0}); {detail}"
                )

            counter[0] += 1
            p = td / (
                f"tile_{counter[0]:05d}_"
                f"r{r0}_c{c0}_{w}x{h}.tif"
            )
            p.write_bytes(rr.content)

            # Fail early if the response has a TIFF signature but is unreadable.
            with rasterio.open(p) as ds:
                if ds.width <= 0 or ds.height <= 0 or ds.count < 1:
                    raise RuntimeError(f"Unreadable/empty raster returned: {p}")
            return [p]

        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < request_retries:
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"Retry {attempt}/{request_retries - 1}: "
                    f"{w}x{h} tile after error: {exc}"
                )
                time.sleep(delay)

    # After retries, keep the requested 10 m grid but reduce server request size.
    can_split = (
        (w > min_tile_pixels or h > min_tile_pixels)
        and (w > 1 or h > 1)
    )
    if not can_split:
        raise RuntimeError(
            f"USGS 3DEP tile failed after retries and cannot be subdivided "
            f"further: {w}x{h}, c0={c0}, r0={r0}"
        ) from last_error

    children = _split_window(c0, r0, w, h)
    if not children:
        raise RuntimeError(
            f"Unable to subdivide failed tile {w}x{h}, c0={c0}, r0={r0}"
        ) from last_error

    print(
        f"USGS request failed for {w}x{h}; "
        f"subdividing into {len(children)} smaller pixel-aligned tile(s)."
    )
    paths = []
    for cc, rr0, ww, hh in children:
        paths.extend(
            _download_window(
                session=session,
                td=td,
                minx=minx,
                maxy=maxy,
                res=res,
                epsg=epsg,
                c0=cc,
                r0=rr0,
                w=ww,
                h=hh,
                min_tile_pixels=min_tile_pixels,
                request_retries=request_retries,
                timeout_seconds=timeout_seconds,
                counter=counter,
            )
        )
    return paths


def main():
    a = parse_args()

    if a.resolution_m <= 0:
        raise ValueError("--resolution-m must be > 0")
    if a.max_tile_pixels < 1:
        raise ValueError("--max-tile-pixels must be >= 1")
    if a.min_tile_pixels < 1:
        raise ValueError("--min-tile-pixels must be >= 1")
    if a.request_retries < 1:
        raise ValueError("--request-retries must be >= 1")

    g = gpd.read_file(a.aoi).to_crs(a.target_crs)
    geom = g.geometry.union_all()
    minx, miny, maxx, maxy = geom.bounds
    res = a.resolution_m
    nx = math.ceil((maxx - minx) / res)
    ny = math.ceil((maxy - miny) / res)
    epsg = g.crs.to_epsg()

    if epsg is None:
        raise RuntimeError("Target CRS must resolve to an EPSG code")

    out = a.output_dir / "full_aoi_dem_navd88_m.tif"
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.overwrite:
        raise FileExistsError(out)

    td = Path(tempfile.mkdtemp(prefix="3dep_", dir=a.output_dir))
    srcs = []
    tmp = out.with_suffix(".tif.partial")

    try:
        tiles = []
        session = requests.Session()
        counter = [0]

        for r0 in range(0, ny, a.max_tile_pixels):
            h = min(a.max_tile_pixels, ny - r0)
            for c0 in range(0, nx, a.max_tile_pixels):
                w = min(a.max_tile_pixels, nx - c0)
                tiles.extend(
                    _download_window(
                        session=session,
                        td=td,
                        minx=minx,
                        maxy=maxy,
                        res=res,
                        epsg=epsg,
                        c0=c0,
                        r0=r0,
                        w=w,
                        h=h,
                        min_tile_pixels=a.min_tile_pixels,
                        request_retries=a.request_retries,
                        timeout_seconds=a.request_timeout_seconds,
                        counter=counter,
                    )
                )

        if not tiles:
            raise RuntimeError("No DEM tiles were downloaded")

        srcs = [rasterio.open(p) for p in tiles]
        arr, tr = merge(srcs, nodata=-9999)
        prof = srcs[0].profile.copy()
        prof.update(
            height=arr.shape[1],
            width=arr.shape[2],
            transform=tr,
            crs=a.target_crs,
            nodata=-9999,
            dtype="float32",
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(tmp, "w", **prof) as ds:
            ds.write(arr.astype("float32"))

        for ds in srcs:
            ds.close()
        srcs = []

        # Crop to the actual AOI geometry; nodata remains outside AOI.
        with rasterio.open(tmp) as src:
            data, tr2 = mask(
                src,
                [geom],
                crop=True,
                nodata=-9999,
                filled=True,
            )
            prof = src.profile.copy()
            prof.update(
                height=data.shape[1],
                width=data.shape[2],
                transform=tr2,
                crs=a.target_crs,
                nodata=-9999,
                dtype="float32",
                compress="deflate",
                tiled=True,
                BIGTIFF="IF_SAFER",
            )

        with rasterio.open(out, "w", **prof) as ds:
            ds.write(data.astype("float32"))

        tmp.unlink(missing_ok=True)

        valid = data[np.isfinite(data) & (data != -9999)]
        if valid.size == 0:
            raise RuntimeError("Final AOI DEM contains no valid elevation cells")

        print(f"USGS 3DEP service: {SERVICE}")
        print(f"AOI grid: {nx} x {ny} pixels at {res:g} m")
        print(f"Downloaded raster tiles: {len(tiles)}")
        print(
            f"Valid elevation range: "
            f"{float(valid.min()):.3f} to {float(valid.max()):.3f} m"
        )
        print(f"DEM: {out}")
        print("PASS_FULL_AOI_DEM_READY")

    finally:
        for ds in srcs:
            try:
                ds.close()
            except Exception:
                pass
        tmp.unlink(missing_ok=True)
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
