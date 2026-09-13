"""
STEP 2A - USGS control point and upstream watershed delineation.

Primary target
--------------
USGS 02089000 - Neuse River near Goldsboro, North Carolina.

Preferred source
----------------
USGS Network Linked Data Index (NLDI) / NHDPlusV2.

The preferred basin request is:
    simplified=false
    splitCatchment=true

Because the NLDI split-catchment operation can intermittently return server-side
5xx errors for large watersheds, this implementation uses a controlled fallback:

1. Core NLDI: full-resolution + splitCatchment=true
2. Core NLDI: simplified + splitCatchment=true
3. NLDI pygeoapi split-catchment: full-resolution, upstream=true
4. NLDI pygeoapi split-catchment: simplified, upstream=true
5. Core NLDI: full-resolution whole local catchment
6. Core NLDI: simplified whole local catchment

A fallback is accepted only if independent QC succeeds, especially the comparison
with the USGS-published drainage area (2,399 mi2 by default).

This step creates the watershed-scale hydrologic domain. It does NOT create an
arbitrary local flood-mapping AOI.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.validation import make_valid


NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
PYGEOAPI_SPLIT = (
    "https://api.water.usgs.gov/nldi/pygeoapi/processes/"
    "nldi-splitcatchment/execution?f=json"
)
SQM_PER_SQMI = 2_589_988.110336
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="02089000")
    p.add_argument("--expected-lon", type=float, default=-77.9975)
    p.add_argument("--expected-lat", type=float, default=35.3375)
    p.add_argument("--published-drainage-area-sqmi", type=float, default=2399.0)
    p.add_argument("--max-drainage-area-difference-percent", type=float, default=5.0)
    p.add_argument("--max-gauge-coordinate-offset-m", type=float, default=150.0)
    p.add_argument("--gauge-basin-tolerance-m", type=float, default=100.0)
    p.add_argument("--projected-crs", default="EPSG:32618")
    p.add_argument("--area-crs", default="EPSG:5070")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("input/spatial/usgs/02089000/nldi"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/spatial/usgs_02089000_domain"),
    )
    p.add_argument("--timeout-seconds", type=float, default=120.0)
    p.add_argument("--request-attempts", type=int, default=3)
    p.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.partial{path.suffix}")
    temp.unlink(missing_ok=True)
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(payload: Any, path: Path) -> None:
    atomic_text(json.dumps(payload, indent=2), path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.partial{path.suffix}")
    temp.unlink(missing_ok=True)
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(path.iterdir()):
        raise FileExistsError(f"{path} is not empty. Use --overwrite.")
    if overwrite:
        for p in path.iterdir():
            if p.is_file():
                p.unlink()


def _headers() -> dict[str, str]:
    return {
        "User-Agent": "neuse-flood-research/1.0",
        "Accept": "application/json, application/geo+json",
    }


def request_json_retry(
    method: str,
    url: str,
    timeout: float,
    attempts: int,
    backoff: float,
    json_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            if method.upper() == "GET":
                r = requests.get(url, headers=_headers(), timeout=timeout)
            elif method.upper() == "POST":
                headers = _headers()
                headers["Content-Type"] = "application/json"
                r = requests.post(
                    url,
                    headers=headers,
                    json=json_body,
                    timeout=timeout,
                )
            else:
                raise ValueError(method)

            logs.append(
                {
                    "attempt": attempt,
                    "method": method.upper(),
                    "url": url,
                    "status_code": r.status_code,
                    "content_type": r.headers.get("content-type"),
                }
            )

            if r.ok:
                payload = r.json()
                if not isinstance(payload, dict):
                    raise RuntimeError("Response JSON is not an object.")
                return payload, logs

            if r.status_code not in RETRYABLE_HTTP:
                r.raise_for_status()

            last_error = requests.HTTPError(
                f"{r.status_code} {r.reason} for {url}"
            )

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
            ValueError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            logs.append(
                {
                    "attempt": attempt,
                    "method": method.upper(),
                    "url": url,
                    "exception": repr(exc),
                }
            )

        if attempt < attempts:
            sleep_seconds = backoff * (2 ** (attempt - 1))
            print(
                f"  request attempt {attempt}/{attempts} failed; "
                f"retrying in {sleep_seconds:.1f} s"
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Request failed after {attempts} attempts: {url}; last_error={last_error}"
    )


def geojson_to_gdf(payload: dict[str, Any], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    kind = payload.get("type")
    if kind == "FeatureCollection":
        features = payload.get("features", [])
    elif kind == "Feature":
        features = [payload]
    elif kind in {"Polygon", "MultiPolygon", "GeometryCollection"}:
        features = [{"type": "Feature", "properties": {}, "geometry": payload}]
    else:
        raise ValueError(f"Unsupported GeoJSON type: {kind}")

    if not features:
        raise RuntimeError("GeoJSON contains no features.")

    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
    if gdf.empty or gdf.geometry.isna().all():
        raise RuntimeError("GeoJSON response contains no usable geometry.")
    return gdf


def looks_like_geojson(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and obj.get("type")
        in {"FeatureCollection", "Feature", "Polygon", "MultiPolygon", "GeometryCollection"}
    )


def find_named_output(payload: Any, target: str) -> dict[str, Any] | None:
    """
    Tolerate multiple pygeoapi response shapes:
      {"drainageBasin": <GeoJSON>}
      {"outputs": {"drainageBasin": <GeoJSON>}}
      [{"id": "drainageBasin", "value": <GeoJSON>}]
      nested equivalents.
    """
    if isinstance(payload, dict):
        if target in payload and looks_like_geojson(payload[target]):
            return payload[target]

        if payload.get("id") == target:
            for key in ("value", "data", "output"):
                value = payload.get(key)
                if looks_like_geojson(value):
                    return value

        for value in payload.values():
            found = find_named_output(value, target)
            if found is not None:
                return found

    elif isinstance(payload, list):
        for item in payload:
            found = find_named_output(item, target)
            if found is not None:
                return found

    return None


def clean_polygon_geometry(gdf: gpd.GeoDataFrame):
    geoms = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            geoms.append(geom)
        elif geom.geom_type == "GeometryCollection":
            for part in geom.geoms:
                if part.geom_type in ("Polygon", "MultiPolygon") and not part.is_empty:
                    geoms.append(part)

    if not geoms:
        raise RuntimeError("No polygonal basin geometry found.")

    merged = unary_union(geoms)
    if not merged.is_valid:
        merged = make_valid(merged)

    if merged.geom_type not in ("Polygon", "MultiPolygon"):
        raise RuntimeError(f"Unexpected basin geometry: {merged.geom_type}")

    return merged


def first_point_geometry(gdf: gpd.GeoDataFrame):
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Point":
            return geom
        if geom.geom_type == "MultiPoint":
            return list(geom.geoms)[0]
    raise RuntimeError("Site response has no point geometry.")


def retrieve_basin(
    args: argparse.Namespace,
    feature_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """
    Return:
      basin GeoJSON payload,
      strategy metadata,
      request logs.
    """
    all_logs: list[dict[str, Any]] = []
    failures: list[str] = []

    strategies = [
        {
            "name": "core_full_split",
            "kind": "GET",
            "url": (
                f"{NLDI_BASE}/nwissite/{feature_id}/basin"
                "?f=json&simplified=false&splitCatchment=true"
            ),
            "split_catchment": True,
            "simplified": False,
        },
        {
            "name": "core_simplified_split",
            "kind": "GET",
            "url": (
                f"{NLDI_BASE}/nwissite/{feature_id}/basin"
                "?f=json&simplified=true&splitCatchment=true"
            ),
            "split_catchment": True,
            "simplified": True,
        },
        {
            "name": "pygeoapi_full_split",
            "kind": "POST",
            "url": PYGEOAPI_SPLIT,
            "body": {
                "inputs": {
                    "lat": args.expected_lat,
                    "lon": args.expected_lon,
                    "upstream": True,
                    "simplified": False,
                }
            },
            "split_catchment": True,
            "simplified": False,
        },
        {
            "name": "pygeoapi_simplified_split",
            "kind": "POST",
            "url": PYGEOAPI_SPLIT,
            "body": {
                "inputs": {
                    "lat": args.expected_lat,
                    "lon": args.expected_lon,
                    "upstream": True,
                    "simplified": True,
                }
            },
            "split_catchment": True,
            "simplified": True,
        },
        {
            "name": "core_full_whole_catchment",
            "kind": "GET",
            "url": (
                f"{NLDI_BASE}/nwissite/{feature_id}/basin"
                "?f=json&simplified=false&splitCatchment=false"
            ),
            "split_catchment": False,
            "simplified": False,
        },
        {
            "name": "core_simplified_whole_catchment",
            "kind": "GET",
            "url": (
                f"{NLDI_BASE}/nwissite/{feature_id}/basin"
                "?f=json&simplified=true&splitCatchment=false"
            ),
            "split_catchment": False,
            "simplified": True,
        },
    ]

    for i, strategy in enumerate(strategies, start=1):
        print(
            f"Basin strategy {i}/{len(strategies)}"
            f"                   : {strategy['name']}"
        )
        try:
            payload, logs = request_json_retry(
                strategy["kind"],
                strategy["url"],
                args.timeout_seconds,
                args.request_attempts,
                args.retry_backoff_seconds,
                strategy.get("body"),
            )
            all_logs.extend(logs)

            if strategy["kind"] == "POST":
                drainage = find_named_output(payload, "drainageBasin")
                if drainage is None:
                    # Occasionally a process can return only splitCatchment if
                    # the full upstream drainage lies within the local catchment.
                    drainage = find_named_output(payload, "splitCatchment")
                if drainage is None:
                    raise RuntimeError(
                        "pygeoapi response did not expose drainageBasin/splitCatchment."
                    )
                basin_payload = drainage
            else:
                basin_payload = payload

            # Validate early that a polygon can actually be constructed.
            test_gdf = geojson_to_gdf(basin_payload)
            _ = clean_polygon_geometry(test_gdf)

            metadata = {
                **strategy,
                "fallback_level": i - 1,
                "preferred_strategy_used": i == 1,
                "prior_failures": failures,
            }
            metadata.pop("body", None)

            print(f"  basin strategy succeeded         : {strategy['name']}")
            return basin_payload, metadata, all_logs

        except Exception as exc:
            failures.append(f"{strategy['name']}: {exc}")
            print(f"  basin strategy failed            : {exc}")

    raise RuntimeError(
        "All NLDI basin strategies failed:\n- " + "\n- ".join(failures)
    )


def main() -> None:
    args = parse_args()
    if args.request_attempts < 1:
        raise ValueError("--request-attempts must be >= 1.")

    prepare_dir(args.raw_dir, args.overwrite)
    prepare_dir(args.output_dir, args.overwrite)

    feature_id = f"USGS-{args.site}"
    site_url = f"{NLDI_BASE}/nwissite/{feature_id}?f=json"

    print("=" * 100)
    print("STEP 2A - USGS CONTROL POINT + UPSTREAM WATERSHED")
    print("=" * 100)
    print(f"Monitoring location                : {feature_id}")
    print("Preferred basin mode               : full resolution + split catchment")
    print("Fallback policy                    : split alternatives -> whole-catchment basin")
    print()

    site_payload, site_logs = request_json_retry(
        "GET",
        site_url,
        args.timeout_seconds,
        args.request_attempts,
        args.retry_backoff_seconds,
    )

    basin_payload, basin_strategy, basin_logs = retrieve_basin(args, feature_id)

    raw_site_path = args.raw_dir / "nldi_site.json"
    raw_basin_path = args.raw_dir / "nldi_upstream_basin_selected.geojson"
    request_log_path = args.raw_dir / "nldi_request_log.json"
    atomic_json(site_payload, raw_site_path)
    atomic_json(basin_payload, raw_basin_path)
    atomic_json(
        {
            "site_requests": site_logs,
            "basin_requests": basin_logs,
            "selected_basin_strategy": basin_strategy,
        },
        request_log_path,
    )

    site_raw = geojson_to_gdf(site_payload)
    basin_raw = geojson_to_gdf(basin_payload)

    nldi_point = first_point_geometry(site_raw)
    basin_geom = clean_polygon_geometry(basin_raw)

    gauge = gpd.GeoDataFrame(
        [{
            "site": args.site,
            "feature_id": feature_id,
            "name": "NEUSE RIVER NEAR GOLDSBORO, NC",
            "source": "USGS NLDI / Water Data for the Nation",
        }],
        geometry=[nldi_point],
        crs="EPSG:4326",
    )

    basin = gpd.GeoDataFrame(
        [{
            "site": args.site,
            "feature_id": feature_id,
            "source": "USGS NLDI / NHDPlusV2",
            "retrieval_strategy": basin_strategy["name"],
            "simplified": basin_strategy["simplified"],
            "split_catchment": basin_strategy["split_catchment"],
        }],
        geometry=[basin_geom],
        crs="EPSG:4326",
    )

    expected_gauge = gpd.GeoDataFrame(
        [{"site": args.site}],
        geometry=[Point(args.expected_lon, args.expected_lat)],
        crs="EPSG:4326",
    )

    gauge_metric = gauge.to_crs(args.projected_crs)
    expected_metric = expected_gauge.to_crs(args.projected_crs)
    basin_metric = basin.to_crs(args.projected_crs)
    basin_area = basin.to_crs(args.area_crs)

    coordinate_offset_m = float(
        gauge_metric.geometry.iloc[0].distance(expected_metric.geometry.iloc[0])
    )
    gauge_to_basin_distance_m = float(
        gauge_metric.geometry.iloc[0].distance(basin_metric.geometry.iloc[0])
    )

    basin_area_m2 = float(basin_area.geometry.area.iloc[0])
    basin_area_km2 = basin_area_m2 / 1_000_000.0
    basin_area_sqmi = basin_area_m2 / SQM_PER_SQMI
    area_difference_sqmi = basin_area_sqmi - args.published_drainage_area_sqmi
    area_difference_percent = (
        area_difference_sqmi / args.published_drainage_area_sqmi * 100.0
    )

    minx, miny, maxx, maxy = basin.total_bounds

    qc_rows: list[dict[str, Any]] = []

    def qc(severity: str, check: str, passed: bool, detail: str) -> None:
        qc_rows.append({
            "severity": severity,
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    qc(
        "BLOCKING",
        "NLDI_GAUGE_COORDINATE_MATCH",
        coordinate_offset_m <= args.max_gauge_coordinate_offset_m,
        (
            f"NLDI=({nldi_point.x:.8f},{nldi_point.y:.8f}); "
            f"expected=({args.expected_lon:.8f},{args.expected_lat:.8f}); "
            f"offset={coordinate_offset_m:.2f} m"
        ),
    )
    qc(
        "BLOCKING",
        "BASIN_GEOMETRY_VALID",
        bool(basin.geometry.iloc[0].is_valid and not basin.geometry.iloc[0].is_empty),
        f"geometry_type={basin.geometry.iloc[0].geom_type}",
    )
    qc(
        "BLOCKING",
        "GAUGE_ON_OR_INSIDE_BASIN",
        gauge_to_basin_distance_m <= args.gauge_basin_tolerance_m,
        (
            f"distance={gauge_to_basin_distance_m:.3f} m; "
            f"tolerance={args.gauge_basin_tolerance_m:.1f} m"
        ),
    )
    qc(
        "BLOCKING",
        "DRAINAGE_AREA_MATCH_USGS",
        abs(area_difference_percent)
        <= args.max_drainage_area_difference_percent,
        (
            f"NLDI={basin_area_sqmi:,.2f} mi2 ({basin_area_km2:,.2f} km2); "
            f"USGS={args.published_drainage_area_sqmi:,.2f} mi2; "
            f"difference={area_difference_percent:+.2f}%"
        ),
    )

    if basin_strategy["preferred_strategy_used"]:
        qc(
            "NOTE",
            "BASIN_RETRIEVAL_STRATEGY",
            True,
            "Preferred full-resolution split-catchment NLDI request succeeded.",
        )
    elif basin_strategy["split_catchment"]:
        qc(
            "WARNING",
            "BASIN_RETRIEVAL_FALLBACK",
            True,
            (
                f"Preferred request failed; selected {basin_strategy['name']}. "
                "The selected result still preserves split-catchment delineation."
            ),
        )
    else:
        qc(
            "WARNING",
            "BASIN_RETRIEVAL_FALLBACK",
            True,
            (
                f"Split-catchment services failed; selected {basin_strategy['name']}. "
                "This includes the entire local NHDPlus catchment at the gauge. "
                "Acceptance depends on drainage-area QC."
            ),
        )

    qc_df = pd.DataFrame(qc_rows)
    blocking = qc_df[
        (qc_df["severity"] == "BLOCKING") & (qc_df["status"] == "FAIL")
    ]
    warnings = qc_df[qc_df["severity"] == "WARNING"]

    gpkg_path = args.output_dir / "usgs_02089000_spatial_domain.gpkg"
    geojson_path = args.output_dir / "usgs_02089000_upstream_basin.geojson"
    qc_path = args.output_dir / "usgs_02089000_spatial_domain_qc.csv"
    metadata_path = args.output_dir / "usgs_02089000_spatial_domain_metadata.json"
    figure_path = args.output_dir / "usgs_02089000_upstream_basin_overview.png"

    gpkg_path.unlink(missing_ok=True)
    gauge.to_file(gpkg_path, layer="gauge_wgs84", driver="GPKG")
    basin.to_file(gpkg_path, layer="upstream_basin_wgs84", driver="GPKG")
    gauge_metric.to_file(gpkg_path, layer="gauge_utm18n", driver="GPKG")
    basin_metric.to_file(gpkg_path, layer="upstream_basin_utm18n", driver="GPKG")
    basin.to_file(geojson_path, driver="GeoJSON")
    atomic_csv(qc_df, qc_path)

    fig, ax = plt.subplots(figsize=(8, 8))
    basin.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2)
    gauge.plot(ax=ax, marker="*", markersize=120, label=f"USGS {args.site}")
    ax.set_title("USGS 02089000 — Upstream Neuse Watershed")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    if len(blocking):
        status = "FAIL_USGS_UPSTREAM_WATERSHED_QC"
    elif len(warnings):
        status = "PASS_USGS_UPSTREAM_WATERSHED_READY_WITH_FALLBACK"
    else:
        status = "PASS_USGS_UPSTREAM_WATERSHED_READY"

    metadata = {
        "status": status,
        "step": "STEP_2A",
        "created_utc": utc_now(),
        "monitoring_location": {
            "site": args.site,
            "feature_id": feature_id,
            "name": "NEUSE RIVER NEAR GOLDSBORO, NC",
            "nldi_longitude": float(nldi_point.x),
            "nldi_latitude": float(nldi_point.y),
            "expected_longitude": args.expected_lon,
            "expected_latitude": args.expected_lat,
            "coordinate_offset_m": coordinate_offset_m,
        },
        "source": {
            "service": "USGS NLDI",
            "network": "NHDPlusV2",
            "site_request_url": site_url,
            "selected_basin_strategy": basin_strategy,
        },
        "crs": {
            "source": "EPSG:4326",
            "metric_working": args.projected_crs,
            "area_qc": args.area_crs,
        },
        "upstream_basin": {
            "area_m2": basin_area_m2,
            "area_km2": basin_area_km2,
            "area_sqmi": basin_area_sqmi,
            "published_usgs_area_sqmi": args.published_drainage_area_sqmi,
            "difference_sqmi": area_difference_sqmi,
            "difference_percent": area_difference_percent,
            "bbox_wgs84": [float(minx), float(miny), float(maxx), float(maxy)],
            "geometry_type": basin.geometry.iloc[0].geom_type,
            "geometry_valid": bool(basin.geometry.iloc[0].is_valid),
        },
        "gauge_basin_distance_m": gauge_to_basin_distance_m,
        "blocking_qc_issue_count": int(len(blocking)),
        "warning_count": int(len(warnings)),
        "local_flood_mapping_aoi_defined": False,
        "output_paths": {
            "raw_site_json": str(raw_site_path),
            "raw_selected_basin_geojson": str(raw_basin_path),
            "request_log": str(request_log_path),
            "spatial_domain_gpkg": str(gpkg_path),
            "upstream_basin_geojson": str(geojson_path),
            "qc": str(qc_path),
            "metadata": str(metadata_path),
            "overview_figure": str(figure_path),
        },
        "sha256": {
            "raw_site_json": sha256_file(raw_site_path),
            "raw_selected_basin_geojson": sha256_file(raw_basin_path),
            "spatial_domain_gpkg": sha256_file(gpkg_path),
            "upstream_basin_geojson": sha256_file(geojson_path),
        },
    }
    atomic_json(metadata, metadata_path)

    print()
    print(f"Selected basin strategy             : {basin_strategy['name']}")
    print(f"Split catchment                     : {basin_strategy['split_catchment']}")
    print(f"Simplified geometry                 : {basin_strategy['simplified']}")
    print(f"NLDI gauge coordinates              : {nldi_point.x:.8f}, {nldi_point.y:.8f}")
    print(f"Expected USGS coordinates           : {args.expected_lon:.8f}, {args.expected_lat:.8f}")
    print(f"Coordinate offset                   : {coordinate_offset_m:.2f} m")
    print()
    print(f"Upstream basin area                 : {basin_area_km2:,.2f} km2")
    print(f"Upstream basin area                 : {basin_area_sqmi:,.2f} mi2")
    print(f"USGS published drainage area        : {args.published_drainage_area_sqmi:,.2f} mi2")
    print(f"Area difference                     : {area_difference_percent:+.2f} %")
    print(f"Gauge -> basin distance             : {gauge_to_basin_distance_m:.3f} m")
    print()
    print(f"Spatial domain GPKG                 : {gpkg_path}")
    print(f"Upstream basin GeoJSON              : {geojson_path}")
    print(f"QC                                  : {qc_path}")
    print(f"Metadata                            : {metadata_path}")
    print(f"Request log                         : {request_log_path}")
    print(f"Overview                            : {figure_path}")
    print()
    print(f"Blocking QC issues                  : {len(blocking)}")
    print(f"Warnings                            : {len(warnings)}")
    print(f"Status                              : {status}")

    if len(blocking):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
