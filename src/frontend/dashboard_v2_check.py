#!/usr/bin/env python3
"""Preflight the generated Hurricane Florence Dashboard V2 bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check Hurricane Florence dashboard V2 outputs.")
    p.add_argument("--root", type=Path, default=Path("output/frontend_v2"))
    return p.parse_args()


def main() -> None:
    a = parse_args()
    root = a.root.resolve()
    bundle_path = root / "data_bundle.json"
    qa_path = root / "dashboard_v2_qa.json"
    required_static = [
        root / "dashboard/index.html",
        root / "dashboard/app.js",
        root / "dashboard/styles.css",
    ]

    failures: list[str] = []
    for p in [bundle_path, qa_path, *required_static]:
        if not p.exists():
            failures.append(f"Missing required file: {p}")

    if failures:
        raise SystemExit("\n".join(failures))

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    qa = json.loads(qa_path.read_text(encoding="utf-8"))

    index_text = (root / "dashboard/index.html").read_text(encoding="utf-8")
    app_text = (root / "dashboard/app.js").read_text(encoding="utf-8")
    styles_text = (root / "dashboard/styles.css").read_text(encoding="utf-8")
    for token in ["playDatesBtn", "showMapBtn", "overlayOpacity", "playDateBadge", "hydraulic-float", "analysis-switcher", "workspace-map", "landcoverClassLegend"]:
        if token not in index_text:
            failures.append(f"Dashboard UI is missing required control/layout token: {token}")
    for token in ["toggleDatePlayback", "replaceRasterLayer", "currentOverlayOpacity", "activateMapView", "activateGraphView", "analysis-mode", "2018-09-14", "renderLandcoverClassLegend", "flooded_road_length_km", "addImpactedRoadVector", "impactVectorPane"]:
        if token not in app_text:
            failures.append(f"Dashboard app.js is missing required behavior: {token}")
    if "FLORENCE_MAP_EDGE_TO_EDGE_V1" not in styles_text:
        failures.append("Dashboard styles are missing the edge-to-edge right-panel map layout.")
    if "FLORENCE_IMPACT_LEGEND_FIX_V1" not in styles_text:
        failures.append("Dashboard styles are missing the impacted land-cover/road legend fix.")
    if "FLORENCE_PLAYBACK_OPACITY_V1" not in styles_text:
        failures.append("Dashboard styles are missing smooth playback / opacity controls.")
    builder_text = Path(__file__).with_name("build_dashboard_v2.py").read_text(encoding="utf-8")
    if "create_date_matched_impact_preview" not in builder_text or "selected-day flood extent" not in builder_text:
        failures.append("Dashboard builder is missing selected-date flood clipping for impact previews.")

    if bundle.get("build") != "FLORENCE_DASHBOARD_V2_BUNDLE":
        failures.append(f"Unexpected bundle build: {bundle.get('build')}")
    if qa.get("status") != "PASS_FLORENCE_DASHBOARD_V2_BUNDLE_READY":
        failures.append(f"Unexpected QA status: {qa.get('status')}")

    compare_modes = bundle.get("map", {}).get("comparison_modes", [])
    for expected in ["single", "side_by_side"]:
        if expected not in compare_modes:
            failures.append(f"Missing comparison mode: {expected}")
    if "swipe" in compare_modes or 'value="swipe"' in index_text or "sideBySide" in app_text:
        failures.append("Swipe comparison must be removed from Dashboard V2.")

    source_keys = set(bundle.get("sources", {}))
    if source_keys != {"ml", "physics", "observed"}:
        failures.append(
            "Primary/secondary source domain must be exactly ML, Physics, and USGS observed-driver; "
            f"found: {sorted(source_keys)}"
        )

    impact_sources = set(qa.get("impact_sources", []))
    if not {"physics", "ml"}.issubset(impact_sources):
        failures.append(
            "Modeling-AOI impact bundle must include Physics and ML for the requested comparison. "
            f"Found: {sorted(impact_sources)}"
        )

    if int(qa.get("landcover_rows", 0)) < 12:
        failures.append(
            "Land-cover bundle is incomplete; expected at least Physics + ML across six categories."
        )

    landcover_legend = bundle.get("impact", {}).get("landcover_map_legend", [])
    if len(landcover_legend) < 10:
        failures.append(
            f"Impacted land-cover class legend is incomplete; found {len(landcover_legend)} classes."
        )

    hydraulics = bundle.get("hydraulics", [])
    required_hydraulic = {
        "discharge_m3s",
        "stage_ft",
        "wse_navd88_ft",
        "hand_equivalent_threshold_m",
        "hq_branch",
        "flood_area_km2",
    }
    if not hydraulics:
        failures.append("No hydraulic state rows in data_bundle.json")
    else:
        missing = required_hydraulic - set(hydraulics[0])
        if missing:
            failures.append(f"Hydraulic state rows missing fields: {sorted(missing)}")

    overlay_paths: list[str] = []
    mp = bundle.get("map", {})
    for source_map in mp.get("flood", {}).values():
        for daily in source_map.values():
            for key in ("extent", "depth"):
                if daily.get(key, {}).get("path"):
                    overlay_paths.append(daily[key]["path"])
    for source_map in mp.get("wse", {}).values():
        for spec in source_map.values():
            if spec.get("path"):
                overlay_paths.append(spec["path"])
    hand = mp.get("hand_equivalent")
    if hand and hand.get("path"):
        overlay_paths.append(hand["path"])
    for spec in mp.get("rainfall", {}).values():
        if spec.get("path"):
            overlay_paths.append(spec["path"])
    for source_map in mp.get("impact", {}).values():
        for day_map in source_map.values():
            if not isinstance(day_map, dict):
                continue
            for spec in day_map.values():
                if isinstance(spec, dict) and spec.get("path"):
                    overlay_paths.append(spec["path"])

    if not qa.get("upstream_watershed_ready", False):
        failures.append(
            "Upstream watershed context is missing. Expected the retained USGS 02089000 basin GeoJSON."
        )

    if int(qa.get("impact_map_preview_count", 0)) == 0:
        failures.append(
            "No modeling-AOI impact map previews were built. Expected retained impact GeoTIFFs under the impact source directories."
        )
    if not qa.get("impact_preview_strict_flood_mask", False):
        failures.append("Impact preview QA does not confirm strict flood masking.")
    if not qa.get("impact_preview_date_matched", False):
        failures.append("Impact preview QA does not confirm selected-date impact masking.")

    # Paths are authored relative to dashboard/index.html.
    dashboard_dir = root / "dashboard"
    missing_overlays = []
    for rel_path in overlay_paths:
        resolved = (dashboard_dir / rel_path).resolve()
        if not resolved.exists():
            missing_overlays.append(str(resolved))
    if missing_overlays:
        failures.append(
            f"Missing {len(missing_overlays)} referenced overlay(s); first: {missing_overlays[0]}"
        )


    road_vector_refs = mp.get("impact_vectors", {})
    road_vector_sources = []
    for source, products in road_vector_refs.items():
        spec = products.get("roads", {}) if isinstance(products, dict) else {}
        path = spec.get("path")
        if not path:
            continue
        road_vector_sources.append(source)
        resolved = (dashboard_dir / path).resolve()
        if not resolved.exists():
            failures.append(f"Missing impacted-road vector GeoJSON for {source}: {resolved}")
        elif resolved.stat().st_size <= 0:
            failures.append(f"Empty impacted-road vector file for {source}: {resolved}")

    if int(qa.get("impact_road_vector_source_count", 0)) < 2:
        failures.append(
            "Impacted-road vector bundle is incomplete; expected at least Physics and ML GeoJSON sources. "
            "Rerun the modeling-AOI impact assessment after applying the road-vector patch."
        )

    # Dry/no-valid-data rainfall previews must be completely transparent.
    transparent_dry_days = []
    for row in bundle.get("rainfall", []):
        if row.get("display_status") not in {"dry", "no_valid_data"}:
            continue
        path = row.get("overlay", {}).get("path")
        if not path:
            continue
        resolved = (dashboard_dir / path).resolve()
        if resolved.exists():
            rgba = np.asarray(Image.open(resolved).convert("RGBA"))
            if np.any(rgba[..., 3] > 0):
                failures.append(f"Dry/no-data rainfall preview is not transparent: {row.get('date')} -> {resolved}")
            else:
                transparent_dry_days.append(row.get("date"))

    # Affected-population preview must retain transparent zero/background pixels.
    population_specs = []
    for source, source_map in mp.get("impact", {}).items():
        for day, day_map in source_map.items():
            if not isinstance(day_map, dict):
                continue
            spec = day_map.get("population")
            if isinstance(spec, dict) and spec.get("path"):
                population_specs.append((source, day, spec))
    for source, day, spec in population_specs:
        resolved = (dashboard_dir / spec["path"]).resolve()
        if not resolved.exists():
            continue
        rgba = np.asarray(Image.open(resolved).convert("RGBA"))
        if not np.any(rgba[..., 3] == 0):
            failures.append(f"Affected-population preview has no transparent background: {resolved}")

    # Strong spatial QA: population, land cover, and imperviousness must never
    # have a visible pixel outside the SAME source/date flood extent preview.
    strict_checked = 0
    for source, source_map in mp.get("impact", {}).items():
        for day, day_map in source_map.items():
            if not isinstance(day_map, dict):
                continue
            flood_spec = day_map.get("flood_area")
            if not isinstance(flood_spec, dict) or not flood_spec.get("path"):
                continue
            flood_path = (dashboard_dir / flood_spec["path"]).resolve()
            if not flood_path.exists():
                continue
            flood_rgba = np.asarray(Image.open(flood_path).convert("RGBA"))
            flood_alpha = flood_rgba[..., 3] > 0
            for key in ("population", "landcover", "impervious"):
                spec = day_map.get(key)
                if not isinstance(spec, dict) or not spec.get("path"):
                    continue
                path = (dashboard_dir / spec["path"]).resolve()
                if not path.exists():
                    continue
                impact_rgba = np.asarray(Image.open(path).convert("RGBA"))
                if impact_rgba.shape[:2] != flood_rgba.shape[:2]:
                    failures.append(
                        f"Date-matched impact/flood preview grid mismatch: {source} {day} {key}"
                    )
                    continue
                outside = int(np.count_nonzero((impact_rgba[..., 3] > 0) & ~flood_alpha))
                strict_checked += 1
                if outside:
                    failures.append(
                        f"Impact preview crosses selected-date flood extent: {source} {day} {key} -> {outside} pixels"
                    )

    print("=" * 100)
    print("HURRICANE FLORENCE DASHBOARD V2 — PREFLIGHT")
    print("=" * 100)
    print(f"Bundle root                          : {root}")
    print(f"Comparison modes                     : {compare_modes}")
    print(f"Hydraulic rows                       : {len(hydraulics)}")
    print(f"Flood source/date previews           : {qa.get('flood_preview_source_date_count', 0)}")
    print(f"Spatial WSE previews                 : {qa.get('wse_preview_source_date_count', 0)}")
    print(f"HAND-equivalent preview              : {qa.get('hand_equivalent_ready', False)}")
    print(f"Daily rainfall previews              : {qa.get('rainfall_preview_days', 0)}")
    print(f"Transparent dry/no-data rainfall days: {transparent_dry_days}")
    print(f"Upstream watershed context           : {qa.get('upstream_watershed_ready', False)}")
    print(f"Impact sources                       : {sorted(impact_sources)}")
    print(f"Impact map previews                  : {qa.get('impact_map_preview_count', 0)}")
    print(f"Strict impact flood mask             : {qa.get('impact_preview_strict_flood_mask', False)}")
    print(f"Date-matched impact mask             : {qa.get('impact_preview_date_matched', False)}")
    print(f"Impact/flood pixel-mask checks       : {strict_checked}")
    print(f"Impacted-road vector sources         : {sorted(road_vector_sources)}")
    print(f"Land-cover rows                      : {qa.get('landcover_rows', 0)}")
    print(f"Referenced overlay files             : {len(overlay_paths)}")

    if failures:
        print("\nFAILURES")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\nStatus                               : PASS_FLORENCE_DASHBOARD_V2_PREFLIGHT")


if __name__ == "__main__":
    main()
