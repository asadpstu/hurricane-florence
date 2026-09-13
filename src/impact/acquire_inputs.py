"""
Step 5C-1 — Acquire Florence impact/exposure reference inputs.

Inputs acquired
---------------
1. WorldPop 2018 population counts (~100 m), event-year population proxy.
2. NLCD 2016 land cover + imperviousness (30 m), closest pre-event NLCD epoch.
3. OpenStreetMap buildings, roads, and critical facilities (current snapshot).

Important temporal note
-----------------------
WorldPop 2018 and NLCD 2016 are event-era inputs.
OSM is acquired at run time and is NOT a historical 2018 snapshot. It must be
reported as a present-day infrastructure/building proxy unless replaced with a
historical OSM extract.

The script clips all products to the AOI supplied with --aoi.
For the full/original Copernicus AOI, use a separate output directory so the
smaller modeling-AOI inputs are never mistaken for full-domain exposure data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--aoi",
        type=Path,
        default=Path(
            "output/spatial/goldsboro_florence_modeling_aoi/"
            "goldsboro_florence_modeling_aoi.geojson"
        ),
    )
    p.add_argument(
        "--project",
        dest="gee_project",
        default="third-oarlock-496708-n7",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("input/impact/florence_2018"),
    )
    p.add_argument(
        "--overpass-endpoint",
        default="https://overpass-api.de/api/interpreter",
    )
    p.add_argument("--skip-worldpop", action="store_true")
    p.add_argument("--skip-nlcd", action="store_true")
    p.add_argument("--skip-osm", action="store_true")
    p.add_argument(
        "--osm-tile-deg",
        type=float,
        default=0.20,
        help=(
            "Maximum Overpass query tile width/height in degrees (default 0.20). "
            "Tiling avoids oversized OSM queries for the full/original AOI."
        ),
    )
    p.add_argument(
        "--osm-min-tile-deg",
        type=float,
        default=0.025,
        help=(
            "Smallest adaptive Overpass tile width/height in degrees "
            "after a server timeout (default 0.025)."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def load_aoi(path):
    import geopandas as gpd

    aoi = gpd.read_file(path)
    if aoi.empty:
        raise RuntimeError("AOI is empty.")
    if aoi.crs is None:
        raise RuntimeError("AOI has no CRS.")
    aoi = aoi.to_crs("EPSG:4326")
    geom = aoi.geometry.union_all()
    if geom.is_empty:
        raise RuntimeError("AOI geometry is empty.")
    return aoi, geom


def ee_initialize(project):
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError(
            "Earth Engine Python API is required: pip install earthengine-api"
        ) from exc
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate first with "
            "`earthengine authenticate`, then rerun."
        ) from exc
    return ee


def ee_region_from_geom(ee, geom):
    if geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
        return ee.Geometry.Polygon(coords, geodesic=False)
    if geom.geom_type == "MultiPolygon":
        coords = [
            [list(poly.exterior.coords)]
            for poly in geom.geoms
        ]
        return ee.Geometry.MultiPolygon(coords, geodesic=False)
    return ee.Geometry(geom.__geo_interface__)


def download_ee_image(image, region, out_path, *, scale, crs, name):
    """
    Download one Earth Engine image as GeoTIFF, handling ZIP responses.
    """
    params = {
        "name": name,
        "scale": scale,
        "crs": crs,
        "region": region,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    url = image.getDownloadURL(params)

    tmp_dir = Path(tempfile.mkdtemp(prefix="neuse_impact_ee_"))
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()

        tmp_payload = tmp_dir / "payload"
        tmp_payload.write_bytes(response.content)

        if response.content[:2] == b"PK" or "zip" in content_type:
            with zipfile.ZipFile(tmp_payload) as zf:
                tifs = [
                    n for n in zf.namelist()
                    if n.lower().endswith((".tif", ".tiff"))
                ]
                if not tifs:
                    raise RuntimeError(
                        f"Earth Engine ZIP for {name} contained no TIFF."
                    )
                extracted = tmp_dir / Path(tifs[0]).name
                with zf.open(tifs[0]) as src, extracted.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            extracted = tmp_dir / f"{name}.tif"
            extracted.write_bytes(response.content)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_name(out_path.name + ".partial")
        partial.unlink(missing_ok=True)
        shutil.copy2(extracted, partial)
        os.replace(partial, out_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def acquire_worldpop(ee, region, out_dir):
    collection = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("country", "USA"))
        .filter(ee.Filter.eq("year", 2018))
    )
    count = int(collection.size().getInfo())
    if count < 1:
        raise RuntimeError(
            "No WorldPop 2018 USA image found in WorldPop/GP/100m/pop."
        )
    image = collection.mosaic().select("population").clip(region)

    out = out_dir / "worldpop_usa_2018_population_100m.tif"
    download_ee_image(
        image,
        region,
        out,
        scale=92.77,
        crs="EPSG:4326",
        name="worldpop_usa_2018_population",
    )
    return out, {
        "dataset": "WorldPop/GP/100m/pop",
        "country": "USA",
        "year": 2018,
        "band": "population",
        "units": "estimated persons per grid cell",
        "nominal_resolution_m": 92.77,
        "license": "CC BY 4.0",
        "temporal_role": "event-year population proxy",
    }


def acquire_nlcd(ee, region, out_dir):
    collection = ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD")
    image = collection.filter(
        ee.Filter.eq("system:index", "2016")
    ).first()

    # Validate image presence by checking band names.
    bands = image.bandNames().getInfo()
    if "landcover" not in bands or "impervious" not in bands:
        raise RuntimeError(
            f"NLCD 2016 expected bands missing; got: {bands}"
        )

    landcover = image.select("landcover").clip(region)
    impervious = image.select("impervious").clip(region)

    lc_out = out_dir / "nlcd_2016_landcover_30m.tif"
    imp_out = out_dir / "nlcd_2016_impervious_30m.tif"

    download_ee_image(
        landcover,
        region,
        lc_out,
        scale=30,
        crs="EPSG:32618",
        name="nlcd_2016_landcover",
    )
    download_ee_image(
        impervious,
        region,
        imp_out,
        scale=30,
        crs="EPSG:32618",
        name="nlcd_2016_impervious",
    )

    return [lc_out, imp_out], {
        "dataset": "USGS/NLCD_RELEASES/2019_REL/NLCD",
        "epoch": 2016,
        "bands": ["landcover", "impervious"],
        "resolution_m": 30,
        "license": "USGS public domain",
        "temporal_role": "closest pre-event land-cover epoch",
    }


OVERPASS_FALLBACK_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def _unique_endpoints(primary):
    endpoints = [primary, *OVERPASS_FALLBACK_ENDPOINTS]
    out = []
    seen = set()
    for endpoint in endpoints:
        endpoint = str(endpoint).strip()
        if endpoint and endpoint not in seen:
            seen.add(endpoint)
            out.append(endpoint)
    return out


def request_overpass(endpoints, query, retries_per_endpoint=2):
    """Run one Overpass query with endpoint failover.

    Only transient HTTP/network failures are retried. If every endpoint fails,
    the caller can split the spatial tile and retry smaller queries.
    """
    last = None
    retryable = {429, 500, 502, 503, 504}

    for endpoint in endpoints:
        for attempt in range(1, retries_per_endpoint + 1):
            try:
                r = requests.post(
                    endpoint,
                    data={"data": query},
                    timeout=300,
                    headers={
                        "User-Agent": (
                            "neuse-flood-research/1.0 "
                            "(impact assessment; academic use)"
                        )
                    },
                )

                if r.status_code in retryable:
                    raise requests.HTTPError(
                        f"{r.status_code} {r.reason} for url: {endpoint}",
                        response=r,
                    )

                r.raise_for_status()
                return r.json(), endpoint

            except Exception as exc:
                last = exc
                if attempt < retries_per_endpoint:
                    time.sleep(3 * attempt)

        # Small pause before changing public instance.
        time.sleep(1)

    raise RuntimeError(
        "All Overpass endpoints failed for this tile. "
        f"Last error: {last}"
    )

def element_to_feature(el, kind):
    from shapely.geometry import LineString, Point, Polygon

    tags = dict(el.get("tags") or {})
    geom_data = el.get("geometry")

    geom = None
    if el.get("type") == "node":
        if "lon" in el and "lat" in el:
            geom = Point(float(el["lon"]), float(el["lat"]))
    elif geom_data:
        coords = [
            (float(p["lon"]), float(p["lat"]))
            for p in geom_data
        ]
        if len(coords) >= 2:
            if (
                kind == "buildings"
                and len(coords) >= 4
                and coords[0] == coords[-1]
            ):
                try:
                    geom = Polygon(coords)
                except Exception:
                    geom = LineString(coords)
            elif (
                kind == "critical_facilities"
                and len(coords) >= 4
                and coords[0] == coords[-1]
            ):
                try:
                    geom = Polygon(coords)
                except Exception:
                    geom = LineString(coords)
            else:
                geom = LineString(coords)

    if geom is None or geom.is_empty:
        return None

    props = {
        "osm_type": el.get("type"),
        "osm_id": el.get("id"),
        "name": tags.get("name"),
        "building": tags.get("building"),
        "highway": tags.get("highway"),
        "amenity": tags.get("amenity"),
        "emergency": tags.get("emergency"),
        "healthcare": tags.get("healthcare"),
        "bridge": tags.get("bridge"),
        "tunnel": tags.get("tunnel"),
        "surface": tags.get("surface"),
        "lanes": tags.get("lanes"),
        "tags_json": json.dumps(tags, ensure_ascii=False),
    }
    return props, geom


def _bbox_tiles(aoi_geom, tile_deg):
    """Yield bbox tiles that intersect the AOI."""
    from shapely.geometry import box

    if tile_deg <= 0:
        raise ValueError("--osm-tile-deg must be > 0")

    west, south, east, north = aoi_geom.bounds
    x0 = math.floor(west / tile_deg) * tile_deg
    y0 = math.floor(south / tile_deg) * tile_deg
    x1 = math.ceil(east / tile_deg) * tile_deg
    y1 = math.ceil(north / tile_deg) * tile_deg

    y = y0
    while y < y1 - 1e-12:
        x = x0
        while x < x1 - 1e-12:
            tile = box(x, y, min(x + tile_deg, x1), min(y + tile_deg, y1))
            if tile.intersects(aoi_geom):
                yield tile.bounds  # west, south, east, north
            x += tile_deg
        y += tile_deg


def _bbox_key(bounds):
    west, south, east, north = bounds
    return f"{south:.6f}_{west:.6f}_{north:.6f}_{east:.6f}".replace("-", "m")


def _split_bbox(bounds):
    west, south, east, north = bounds
    midx = (west + east) / 2.0
    midy = (south + north) / 2.0
    return [
        (west, south, midx, midy),
        (midx, south, east, midy),
        (west, midy, midx, north),
        (midx, midy, east, north),
    ]


def _query_tile_adaptive(
    *,
    kind,
    template,
    bounds,
    endpoints,
    raw_dir,
    min_tile_deg,
    depth=0,
):
    """Fetch a bbox, recursively splitting it on persistent Overpass failure."""
    west, south, east, north = bounds
    cache_path = raw_dir / f"{kind}_{_bbox_key(bounds)}.json"

    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return payload.get("elements", []), 0, True
        except Exception:
            # Corrupt/incomplete cache: query it again.
            cache_path.unlink(missing_ok=True)

    bbox = f"{south},{west},{north},{east}"
    query = template.format(bbox=bbox)

    try:
        payload, used_endpoint = request_overpass(endpoints, query)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        print(
            f"    {kind:20s} bbox={bbox}  "
            f"OK via {used_endpoint}"
        )
        return payload.get("elements", []), 1, False

    except Exception as exc:
        width = east - west
        height = north - south

        if max(width, height) / 2.0 < min_tile_deg - 1e-12:
            raise RuntimeError(
                f"Overpass failed even at minimum tile size for {kind}; "
                f"bbox={bbox}; error={exc}"
            ) from exc

        indent = "  " * min(depth, 4)
        print(
            f"  {indent}{kind}: tile failed ({width:.4f}° x {height:.4f}°); "
            "splitting into 4 smaller tiles ..."
        )

        all_elements = []
        query_count = 0
        all_cached = True
        for child in _split_bbox(bounds):
            elems, nqueries, cached = _query_tile_adaptive(
                kind=kind,
                template=template,
                bounds=child,
                endpoints=endpoints,
                raw_dir=raw_dir,
                min_tile_deg=min_tile_deg,
                depth=depth + 1,
            )
            all_elements.extend(elems)
            query_count += nqueries
            all_cached = all_cached and cached

        return all_elements, query_count, all_cached


def acquire_osm(
    aoi_geom,
    endpoint,
    out_dir,
    tile_deg=0.20,
    min_tile_deg=0.025,
):
    """Acquire OSM exposure with resume, endpoint failover, and adaptive tiles."""
    import geopandas as gpd

    tiles = list(_bbox_tiles(aoi_geom, tile_deg))
    if not tiles:
        raise RuntimeError("No OSM query tiles intersect the AOI.")
    if min_tile_deg <= 0 or min_tile_deg > tile_deg:
        raise ValueError("--osm-min-tile-deg must be > 0 and <= --osm-tile-deg")

    endpoints = _unique_endpoints(endpoint)

    query_templates = {
        "buildings": """
[out:json][timeout:180];
(
  way["building"]({bbox});
);
out tags geom;
""",
        "roads": """
[out:json][timeout:180];
(
  way["highway"]({bbox});
);
out tags geom;
""",
        "critical_facilities": """
[out:json][timeout:180];
(
  node["amenity"~"hospital|clinic|fire_station|police|school|kindergarten|shelter"]({bbox});
  way["amenity"~"hospital|clinic|fire_station|police|school|kindergarten|shelter"]({bbox});
  node["healthcare"]({bbox});
  way["healthcare"]({bbox});
  node["emergency"~"ambulance_station|fire_hydrant"]({bbox});
  way["emergency"~"ambulance_station"]({bbox});
);
out tags geom;
""",
    }

    gpkg = out_dir / "osm_current_exposure.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    counts = {}
    raw_dir = out_dir / "osm_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"  OSM initial tiles: {len(tiles)} tile(s), "
        f"initial size <= {tile_deg:.3f}°, adaptive minimum {min_tile_deg:.3f}°"
    )
    print("  Overpass endpoints:")
    for ep in endpoints:
        print(f"    {ep}")
    print("  Existing osm_raw JSON files will be reused automatically.")

    request_count = 0
    cache_hits = 0

    for kind, template in query_templates.items():
        elements = {}

        print(f"\n  OSM layer: {kind}")
        for tile_idx, bounds in enumerate(tiles, start=1):
            print(f"    initial tile {tile_idx}/{len(tiles)}")

            # Reuse raw files created by the previous downloader version.
            legacy_cache = raw_dir / f"{kind}_tile_{tile_idx:03d}.json"
            if legacy_cache.exists():
                try:
                    payload = json.loads(legacy_cache.read_text(encoding="utf-8"))
                    tile_elements = payload.get("elements", [])
                    nqueries = 0
                    cached = True
                    print(f"      reused legacy cache: {legacy_cache.name}")
                except Exception:
                    legacy_cache.unlink(missing_ok=True)
                    tile_elements, nqueries, cached = _query_tile_adaptive(
                        kind=kind,
                        template=template,
                        bounds=bounds,
                        endpoints=endpoints,
                        raw_dir=raw_dir,
                        min_tile_deg=min_tile_deg,
                    )
            else:
                tile_elements, nqueries, cached = _query_tile_adaptive(
                    kind=kind,
                    template=template,
                    bounds=bounds,
                    endpoints=endpoints,
                    raw_dir=raw_dir,
                    min_tile_deg=min_tile_deg,
                )

            request_count += nqueries
            cache_hits += int(cached)

            for el in tile_elements:
                key = (el.get("type"), el.get("id"))
                elements[key] = el

        records = []
        geoms = []
        for el in elements.values():
            feat = element_to_feature(el, kind)
            if feat is None:
                continue
            props, geom = feat
            records.append(props)
            geoms.append(geom)

        if records:
            gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
            try:
                gdf = gpd.clip(
                    gdf,
                    gpd.GeoSeries([aoi_geom], crs="EPSG:4326"),
                )
            except Exception:
                gdf = gdf[gdf.intersects(aoi_geom)]

            if not gdf.empty:
                gdf = gdf.drop_duplicates(subset=["osm_type", "osm_id"]).copy()
                gdf.to_file(gpkg, layer=kind, driver="GPKG")
            counts[kind] = int(len(gdf))
        else:
            counts[kind] = 0

        print(f"    retained {counts[kind]:,} feature(s)")

    return gpkg, {
        "source": "OpenStreetMap via public Overpass API instances",
        "snapshot": "current at acquisition time",
        "historical_2018": False,
        "license": "ODbL; attribution required",
        "layers": counts,
        "query_initial_tile_degrees": float(tile_deg),
        "query_min_tile_degrees": float(min_tile_deg),
        "query_initial_tile_count": int(len(tiles)),
        "http_queries_this_run": int(request_count),
        "cached_initial_tiles_reused": int(cache_hits),
        "overpass_endpoints": endpoints,
        "temporal_role": (
            "current building/infrastructure proxy; not event-date inventory"
        ),
    }


def main():
    a = parse_args()
    require(a.aoi)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = a.out_dir / "impact_input_acquisition_metadata.json"

    if metadata_path.exists() and not a.overwrite:
        raise FileExistsError(
            f"{metadata_path} exists; use --overwrite."
        )

    _, geom = load_aoi(a.aoi)

    metadata = {
        "status": "PASS_IMPACT_REFERENCE_INPUTS_ACQUIRED",
        "aoi": str(a.aoi),
        "gee_project": a.gee_project,
        "sources": {},
        "outputs": {},
        "warnings": [],
    }

    ee = None
    if not (a.skip_worldpop and a.skip_nlcd):
        ee = ee_initialize(a.gee_project)
        print("Earth Engine initialized")

    if not a.skip_worldpop:
        print("Acquiring WorldPop 2018 population ...")
        path, meta = acquire_worldpop(ee, ee_region_from_geom(ee, geom), a.out_dir)
        metadata["sources"]["population"] = meta
        metadata["outputs"]["population"] = str(path)
        print(f"  {path}")
    else:
        path = a.out_dir / "worldpop_usa_2018_population_100m.tif"
        require(path)
        metadata["sources"]["population"] = {
            "status": "reused_existing",
            "note": "Existing original-AOI WorldPop raster retained; download skipped.",
        }
        metadata["outputs"]["population"] = str(path)
        print(f"Reusing WorldPop 2018 population: {path}")

    if not a.skip_nlcd:
        print("Acquiring NLCD 2016 land cover + imperviousness ...")
        paths, meta = acquire_nlcd(ee, ee_region_from_geom(ee, geom), a.out_dir)
        metadata["sources"]["landcover"] = meta
        metadata["outputs"]["landcover"] = str(paths[0])
        metadata["outputs"]["impervious"] = str(paths[1])
        print(f"  {paths[0]}")
        print(f"  {paths[1]}")
    else:
        lc_path = a.out_dir / "nlcd_2016_landcover_30m.tif"
        imp_path = a.out_dir / "nlcd_2016_impervious_30m.tif"
        require(lc_path)
        require(imp_path)
        metadata["sources"]["landcover"] = {
            "status": "reused_existing",
            "note": "Existing original-AOI NLCD rasters retained; download skipped.",
        }
        metadata["outputs"]["landcover"] = str(lc_path)
        metadata["outputs"]["impervious"] = str(imp_path)
        print(f"Reusing NLCD 2016 land cover: {lc_path}")
        print(f"Reusing NLCD 2016 impervious: {imp_path}")

    if not a.skip_osm:
        print("Acquiring current OSM buildings/roads/critical facilities ...")
        gpkg, meta = acquire_osm(
            geom,
            a.overpass_endpoint,
            a.out_dir,
            a.osm_tile_deg,
            a.osm_min_tile_deg,
        )
        metadata["sources"]["osm"] = meta
        metadata["outputs"]["osm"] = str(gpkg)
        metadata["warnings"].append(
            "OSM exposure data are current at acquisition time, not a historical "
            "September 2018 snapshot."
        )
        print(f"  {gpkg}")
        print(f"  layer counts: {meta['layers']}")

    tmp = metadata_path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(tmp, metadata_path)

    print("=" * 96)
    print("IMPACT REFERENCE INPUT ACQUISITION")
    print("=" * 96)
    if "population" in metadata["outputs"]:
        print(f"WorldPop 2018 population : {metadata['outputs']['population']}")
    if "landcover" in metadata["outputs"]:
        print(f"NLCD 2016 land cover     : {metadata['outputs']['landcover']}")
    if "impervious" in metadata["outputs"]:
        print(f"NLCD 2016 impervious     : {metadata['outputs']['impervious']}")
    if "osm" in metadata["outputs"]:
        print(f"OSM exposure GPKG        : {metadata['outputs']['osm']}")
    print(f"Metadata                 : {metadata_path}")
    print("STATUS: PASS_IMPACT_REFERENCE_INPUTS_ACQUIRED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
