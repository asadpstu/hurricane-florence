"""
MODEL IMPROVEMENT A2.3
Aggregate raw upstream NHDPlus HR catchments into hydrologically connected
modeling subcatchments for semi-distributed physics + spatial ML.

Design
------
* Raw network: A2.2 upstream flowlines/catchments + routing edges.
* Build ONE downstream path per reach that is guaranteed to lead to the gauge.
* Prefer NHDPlus DnHydroSeq when it is a valid downstream step.
* Otherwise fall back to the routing edge that moves closest to the outlet.
* Partition the resulting tree into connected units near a target area.
* Dissolve raw catchment polygons by unit.
* Preserve a directed subcatchment routing graph.

Defaults
--------
Target area : 150 km2
Minimum area: 50 km2 (soft partition preference)
Maximum area: 300 km2 (soft partition preference)

The partition thresholds are not hydrologic calibration parameters. They define
the spatial discretization shared by Physics and ML.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


BUILD = "MODEL_IMPROVEMENT_A2_3_AGGREGATE_SUBCATCHMENTS_V1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--framework-gpkg", type=Path, required=True)
    p.add_argument("--routing-edges", type=Path, required=True)
    p.add_argument("--framework-metadata", type=Path, required=True)
    p.add_argument("--authoritative-basin", type=Path, required=True)
    p.add_argument("--target-area-km2", type=float, default=150.0)
    p.add_argument("--min-area-km2", type=float, default=50.0)
    p.add_argument("--max-area-km2", type=float, default=300.0)
    p.add_argument("--min-domain-coverage-percent", type=float, default=99.5)
    p.add_argument("--min-subcatchment-count", type=int, default=20)
    p.add_argument("--max-subcatchment-count", type=int, default=80)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/model_improvement/modeling_subcatchments"),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def findcol(df, *names, required=True):
    cmap = {str(c).lower(): c for c in df.columns}
    for name in names:
        c = cmap.get(name.lower())
        if c is not None:
            return c
    if required:
        raise RuntimeError(
            f"Missing any of {names}; available columns={list(df.columns)}"
        )
    return None


def as_int_or_none(v):
    if pd.isna(v):
        return None
    try:
        return int(round(float(v)))
    except Exception:
        return None


def atomic_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    if a.target_area_km2 <= 0:
        raise RuntimeError("Target area must be positive.")
    if not (0 < a.min_area_km2 <= a.target_area_km2 <= a.max_area_km2):
        raise RuntimeError(
            "Require min_area <= target_area <= max_area."
        )

    meta = json.loads(a.framework_metadata.read_text(encoding="utf-8"))
    if not meta.get("safe_for_a2_3", False):
        raise RuntimeError("A2.2 metadata does not authorize A2.3.")

    outlet_id = int(meta["outlet"]["nhdplusid"])

    flow = gpd.read_file(a.framework_gpkg, layer="upstream_flowlines")
    catch = gpd.read_file(a.framework_gpkg, layer="upstream_catchments")
    edges = pd.read_csv(a.routing_edges)

    idc = findcol(flow, "_id", "NHDPlusID")
    cid = findcol(catch, "_id", "NHDPlusID")
    hydro = findcol(flow, "HydroSeq")
    dnh = findcol(flow, "DnHydroSeq", required=False)
    total_da = findcol(flow, "TotDASqKm", required=False)
    order = findcol(flow, "StreamOrde", "StreamOrder", required=False)
    pathlen = findcol(flow, "PathLength", required=False)
    name = findcol(flow, "GNIS_Name", "GNISName", required=False)

    flow["_rid"] = pd.to_numeric(flow[idc], errors="coerce").round().astype("Int64")
    catch["_rid"] = pd.to_numeric(catch[cid], errors="coerce").round().astype("Int64")
    flow["_hydroseq"] = pd.to_numeric(flow[hydro], errors="coerce").round().astype("Int64")

    if dnh:
        flow["_dnhydroseq"] = pd.to_numeric(
            flow[dnh], errors="coerce"
        ).round().astype("Int64")
    else:
        flow["_dnhydroseq"] = pd.Series(
            pd.NA, index=flow.index, dtype="Int64"
        )

    flow = flow[flow["_rid"].notna()].copy()
    catch = catch[catch["_rid"].notna()].copy()

    nodes = set(flow["_rid"].astype("int64").tolist())
    if outlet_id not in nodes:
        raise RuntimeError(
            f"Outlet NHDPlusID {outlet_id} is absent from flowline framework."
        )

    ef = findcol(edges, "from_nhdplusid")
    et = findcol(edges, "to_nhdplusid")
    edges["_from"] = pd.to_numeric(edges[ef], errors="coerce").round().astype("Int64")
    edges["_to"] = pd.to_numeric(edges[et], errors="coerce").round().astype("Int64")
    edges = edges[
        edges["_from"].notna()
        & edges["_to"].notna()
        & edges["_from"].astype("int64").isin(nodes)
        & edges["_to"].astype("int64").isin(nodes)
    ].copy()

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {BUILD}")
    print("MODEL IMPROVEMENT A2.3 - HYDROLOGIC MODELING SUBCATCHMENTS")
    print("=" * 100)
    print(f"Raw upstream flowlines             : {len(flow):,}")
    print(f"Raw upstream catchments            : {len(catch):,}")
    print(f"Raw routing edges                  : {len(edges):,}")
    print(f"Gauge outlet NHDPlusID             : {outlet_id}")
    print(f"Target unit area                   : {a.target_area_km2:.1f} km²")
    print(f"Preferred min / max area           : {a.min_area_km2:.1f} / {a.max_area_km2:.1f} km²")
    print()

    # ------------------------------------------------------------------
    # Build all-edge network and exact graph distance to the basin outlet.
    # ------------------------------------------------------------------
    downstream_all = defaultdict(list)
    upstream_all = defaultdict(list)

    for f, t in zip(
        edges["_from"].astype("int64"),
        edges["_to"].astype("int64"),
    ):
        downstream_all[int(f)].append(int(t))
        upstream_all[int(t)].append(int(f))

    distance = {outlet_id: 0}
    q = deque([outlet_id])

    while q:
        d = q.popleft()
        for u in upstream_all.get(d, []):
            nd = distance[d] + 1
            if u not in distance or nd < distance[u]:
                distance[u] = nd
                q.append(u)

    unreachable = sorted(nodes - set(distance))
    if unreachable:
        raise RuntimeError(
            f"{len(unreachable)} upstream framework nodes cannot reach outlet "
            f"through A2.2 routing edges. Example={unreachable[:10]}"
        )

    # Lookup reach attributes.
    hydro_to_id = {}
    rid_to_row = {}

    for idx, row in flow.iterrows():
        rid = int(row["_rid"])
        rid_to_row[rid] = row

        hs = as_int_or_none(row["_hydroseq"])
        if hs is not None:
            hydro_to_id[hs] = rid

    # ------------------------------------------------------------------
    # Select one downstream edge per node.
    # This guarantees a tree-like routing structure to the gauge.
    # ------------------------------------------------------------------
    primary_downstream = {}
    primary_method = {}

    for rid in nodes:
        if rid == outlet_id:
            continue

        candidates = [
            v for v in downstream_all.get(rid, [])
            if v in distance and distance[v] == distance[rid] - 1
        ]

        if not candidates:
            raise RuntimeError(
                f"No downstream edge toward outlet for reach {rid}"
            )

        row = rid_to_row[rid]
        preferred = None

        if dnh:
            dnhs = as_int_or_none(row["_dnhydroseq"])
            if dnhs is not None:
                preferred = hydro_to_id.get(dnhs)

        if preferred in candidates:
            chosen = preferred
            method = "DnHydroSeq"
        else:
            # Prefer the candidate with the greatest drainage area.
            def candidate_key(v):
                rr = rid_to_row.get(v)
                da = (
                    float(rr[total_da])
                    if rr is not None
                    and total_da
                    and pd.notna(rr[total_da])
                    else -1.0
                )
                return (da, -v)

            chosen = max(candidates, key=candidate_key)
            method = "routing_distance_fallback"

        primary_downstream[rid] = chosen
        primary_method[rid] = method

    tree = nx.DiGraph()
    tree.add_nodes_from(nodes)
    tree.add_edges_from(primary_downstream.items())

    if not nx.is_directed_acyclic_graph(tree):
        raise RuntimeError("Primary downstream graph contains a cycle.")

    terminals = [n for n in tree.nodes if tree.out_degree(n) == 0]
    if terminals != [outlet_id]:
        raise RuntimeError(
            f"Expected exactly outlet terminal {outlet_id}; terminals={terminals[:20]}"
        )

    # ------------------------------------------------------------------
    # Local raw-catchment area for each reach.
    # Catchments are already clipped to the authoritative Step 2 basin.
    # ------------------------------------------------------------------
    if "model_area_km2" not in catch.columns:
        catch["model_area_km2"] = catch.geometry.area / 1e6

    local_area = defaultdict(float)

    grouped_area = (
        catch.groupby("_rid", dropna=True)["model_area_km2"]
        .sum()
    )
    for rid, area in grouped_area.items():
        local_area[int(rid)] = float(area)

    raw_area_total = float(catch["model_area_km2"].sum())

    print("PRIMARY ROUTING TREE")
    print("-" * 100)
    print(f"Nodes                               : {tree.number_of_nodes():,}")
    print(f"Edges                               : {tree.number_of_edges():,}")
    print(f"Terminals                           : {len(terminals)}")
    print(
        f"DnHydroSeq selections               : "
        f"{sum(v == 'DnHydroSeq' for v in primary_method.values()):,}"
    )
    print(
        f"Routing-distance fallbacks          : "
        f"{sum(v != 'DnHydroSeq' for v in primary_method.values()):,}"
    )
    print(f"Raw clipped catchment area          : {raw_area_total:,.3f} km²")
    print()

    # ------------------------------------------------------------------
    # Connected-area partition.
    # residual_nodes[n] is the unclosed connected drainage component
    # draining to n.
    # ------------------------------------------------------------------
    children = {
        n: list(tree.predecessors(n))
        for n in tree.nodes
    }

    topo = list(nx.topological_sort(tree))
    residual_nodes = {}
    residual_area = {}

    temp_units = []
    temp_assignment = {}
    next_unit = 1

    def close_unit(node_list, area, outlet_reach):
        nonlocal next_unit
        uid = next_unit
        next_unit += 1

        node_list = list(node_list)
        for n in node_list:
            if n in temp_assignment:
                raise RuntimeError(
                    f"Reach {n} assigned to more than one subcatchment."
                )
            temp_assignment[n] = uid

        temp_units.append({
            "temp_unit": uid,
            "outlet_reach": int(outlet_reach),
            "reach_ids": node_list,
            "area_km2": float(area),
        })
        return uid

    for node in topo:
        bags = []

        for child in children[node]:
            cnodes = residual_nodes.get(child, [])
            carea = residual_area.get(child, 0.0)

            if cnodes:
                bags.append({
                    "outlet": child,
                    "nodes": cnodes,
                    "area": carea,
                })

        total_area = local_area[node] + sum(b["area"] for b in bags)

        # If a confluence would create an overly large unit, close the
        # largest incoming connected components first.
        while total_area > a.max_area_km2 and bags:
            bags.sort(key=lambda b: b["area"], reverse=True)
            b = bags.pop(0)

            close_unit(
                b["nodes"],
                b["area"],
                b["outlet"],
            )
            total_area = local_area[node] + sum(
                x["area"] for x in bags
            )

        combined_nodes = [node]
        for b in bags:
            combined_nodes.extend(b["nodes"])

        combined_area = local_area[node] + sum(
            b["area"] for b in bags
        )

        if combined_area >= a.target_area_km2:
            close_unit(
                combined_nodes,
                combined_area,
                node,
            )
            residual_nodes[node] = []
            residual_area[node] = 0.0
        else:
            residual_nodes[node] = combined_nodes
            residual_area[node] = combined_area

    # Whatever remains at the gauge becomes the outlet modeling unit.
    outlet_residual = residual_nodes.get(outlet_id, [])
    outlet_residual_area = residual_area.get(outlet_id, 0.0)

    if outlet_residual:
        close_unit(
            outlet_residual,
            outlet_residual_area,
            outlet_id,
        )

    unassigned_nodes = sorted(nodes - set(temp_assignment))
    if unassigned_nodes:
        raise RuntimeError(
            f"{len(unassigned_nodes)} reaches were not assigned. "
            f"Example={unassigned_nodes[:20]}"
        )

    # ------------------------------------------------------------------
    # Aggregate-unit routing graph.
    # ------------------------------------------------------------------
    temp_edges = set()

    for u, v in primary_downstream.items():
        uu = temp_assignment[u]
        vv = temp_assignment[v]
        if uu != vv:
            temp_edges.add((uu, vv))

    unit_graph = nx.DiGraph()
    unit_graph.add_nodes_from([u["temp_unit"] for u in temp_units])
    unit_graph.add_edges_from(temp_edges)

    if not nx.is_directed_acyclic_graph(unit_graph):
        raise RuntimeError("Aggregated subcatchment graph contains a cycle.")

    unit_terminals = [
        n for n in unit_graph.nodes
        if unit_graph.out_degree(n) == 0
    ]

    outlet_temp_unit = temp_assignment[outlet_id]

    if unit_terminals != [outlet_temp_unit]:
        raise RuntimeError(
            f"Expected one aggregate outlet unit {outlet_temp_unit}; "
            f"got {unit_terminals}"
        )

    # Stable IDs: topological order from headwaters toward the gauge.
    unit_topo = list(nx.topological_sort(unit_graph))
    rename = {
        old: f"SC{i:03d}"
        for i, old in enumerate(unit_topo, 1)
    }

    node_to_sc = {
        rid: rename[uid]
        for rid, uid in temp_assignment.items()
    }

    # ------------------------------------------------------------------
    # Dissolve raw catchments into modeling polygons.
    # ------------------------------------------------------------------
    catch["subcatchment_id"] = (
        catch["_rid"].astype("int64").map(node_to_sc)
    )

    missing_catch_assignment = int(
        catch["subcatchment_id"].isna().sum()
    )

    if missing_catch_assignment:
        raise RuntimeError(
            f"{missing_catch_assignment} raw catchments lack subcatchment assignment."
        )

    polygons = catch[
        ["subcatchment_id", "model_area_km2", "geometry"]
    ].dissolve(
        by="subcatchment_id",
        aggfunc={"model_area_km2": "sum"},
        as_index=False,
    )

    polygons["geometry_area_km2"] = (
        polygons.geometry.area / 1e6
    )

    # Unit attributes.
    records = []

    for u in temp_units:
        old = u["temp_unit"]
        sc = rename[old]
        outlet = u["outlet_reach"]
        rr = rid_to_row[outlet]

        downstream_old = next(
            iter(unit_graph.successors(old)),
            None,
        )

        records.append({
            "subcatchment_id": sc,
            "temp_unit": old,
            "outlet_nhdplusid": outlet,
            "downstream_subcatchment_id": (
                rename[downstream_old]
                if downstream_old is not None
                else None
            ),
            "model_area_km2": u["area_km2"],
            "raw_reach_count": len(u["reach_ids"]),
            "outlet_hydroseq": as_int_or_none(rr["_hydroseq"]),
            "outlet_stream_order": (
                float(rr[order])
                if order and pd.notna(rr[order])
                else None
            ),
            "outlet_totdasqkm": (
                float(rr[total_da])
                if total_da and pd.notna(rr[total_da])
                else None
            ),
            "outlet_pathlength": (
                float(rr[pathlen])
                if pathlen and pd.notna(rr[pathlen])
                else None
            ),
            "outlet_name": (
                str(rr[name])
                if name and pd.notna(rr[name])
                else None
            ),
            "is_basin_outlet": bool(outlet == outlet_id),
        })

    attrs = pd.DataFrame(records)

    raw_counts = (
        catch.groupby("subcatchment_id")
        .size()
        .rename("raw_catchment_count")
        .reset_index()
    )

    attrs = attrs.merge(
        raw_counts,
        on="subcatchment_id",
        how="left",
        validate="one_to_one",
    )

    sub = polygons.merge(
        attrs,
        on="subcatchment_id",
        how="left",
        validate="one_to_one",
    )

    # Recompute exact area after dissolve.
    sub["model_area_km2"] = sub.geometry.area / 1e6
    sub["centroid_x_utm18"] = sub.geometry.centroid.x
    sub["centroid_y_utm18"] = sub.geometry.centroid.y

    sub = sub.sort_values("subcatchment_id").reset_index(drop=True)

    routing_out = []
    for old_u, old_v in unit_graph.edges():
        routing_out.append({
            "from_subcatchment_id": rename[old_u],
            "to_subcatchment_id": rename[old_v],
        })

    routing_out = pd.DataFrame(routing_out).sort_values(
        ["from_subcatchment_id", "to_subcatchment_id"]
    )

    # ------------------------------------------------------------------
    # Basin coverage + residual.
    # ------------------------------------------------------------------
    basin = gpd.read_file(a.authoritative_basin).to_crs(sub.crs)
    basin_geom = basin.geometry.union_all()
    sub_union = sub.geometry.union_all()

    coverage_pct = (
        100.0 * sub_union.intersection(basin_geom).area
        / basin_geom.area
    )

    residual_geom = basin_geom.difference(sub_union)
    residual_area_km2 = float(
        residual_geom.area / 1e6
        if not residual_geom.is_empty
        else 0.0
    )

    residual = gpd.GeoDataFrame(
        {
            "residual_area_km2": [residual_area_km2],
            "coverage_percent": [coverage_pct],
        },
        geometry=[residual_geom],
        crs=sub.crs,
    )

    # ------------------------------------------------------------------
    # QC.
    # ------------------------------------------------------------------
    areas = sub["model_area_km2"].to_numpy(float)
    n_units = len(sub)

    all_units_reach_outlet = True
    for old in unit_graph.nodes:
        if old == outlet_temp_unit:
            continue
        if not nx.has_path(unit_graph, old, outlet_temp_unit):
            all_units_reach_outlet = False
            break

    qrows = []

    def qc(sev, check, passed, detail):
        qrows.append({
            "severity": sev,
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    qc(
        "BLOCKING",
        "ALL_RAW_CATCHMENTS_ASSIGNED",
        missing_catch_assignment == 0,
        f"unassigned={missing_catch_assignment}",
    )
    qc(
        "BLOCKING",
        "AGGREGATED_GRAPH_ACYCLIC",
        nx.is_directed_acyclic_graph(unit_graph),
        f"nodes={unit_graph.number_of_nodes()}, edges={unit_graph.number_of_edges()}",
    )
    qc(
        "BLOCKING",
        "SINGLE_BASIN_OUTLET",
        unit_terminals == [outlet_temp_unit],
        f"terminal_count={len(unit_terminals)}",
    )
    qc(
        "BLOCKING",
        "ALL_UNITS_ROUTE_TO_GAUGE",
        all_units_reach_outlet,
        f"outlet={rename[outlet_temp_unit]}",
    )
    qc(
        "QUALITY",
        "DOMAIN_COVERAGE",
        coverage_pct >= a.min_domain_coverage_percent,
        f"{coverage_pct:.6f}% >= {a.min_domain_coverage_percent:.3f}%",
    )
    qc(
        "QUALITY",
        "SUBCATCHMENT_COUNT",
        a.min_subcatchment_count <= n_units <= a.max_subcatchment_count,
        (
            f"count={n_units}; expected range="
            f"{a.min_subcatchment_count}-{a.max_subcatchment_count}"
        ),
    )

    # Area-size checks are warnings because headwater/confluence geometry can
    # legitimately produce units outside the preferred range.
    qrows.append({
        "severity": "WARNING",
        "check": "SMALL_SUBCATCHMENTS",
        "status": "WARN" if np.any(areas < a.min_area_km2) else "PASS",
        "detail": (
            f"below {a.min_area_km2:.1f} km²="
            f"{int(np.sum(areas < a.min_area_km2))}; "
            f"minimum={areas.min():.3f} km²"
        ),
    })
    qrows.append({
        "severity": "WARNING",
        "check": "LARGE_SUBCATCHMENTS",
        "status": "WARN" if np.any(areas > a.max_area_km2) else "PASS",
        "detail": (
            f"above {a.max_area_km2:.1f} km²="
            f"{int(np.sum(areas > a.max_area_km2))}; "
            f"maximum={areas.max():.3f} km²"
        ),
    })

    qcdf = pd.DataFrame(qrows)
    bf = qcdf[
        (qcdf["severity"] == "BLOCKING")
        & (qcdf["status"] == "FAIL")
    ]
    qf = qcdf[
        (qcdf["severity"] == "QUALITY")
        & (qcdf["status"] == "FAIL")
    ]

    safe = len(bf) == 0 and len(qf) == 0
    status = (
        "PASS_A2_3_MODELING_SUBCATCHMENTS_READY"
        if safe
        else (
            "FAIL_A2_3_MODELING_SUBCATCHMENTS_BLOCKING"
            if len(bf)
            else "FAIL_A2_3_MODELING_SUBCATCHMENTS_QUALITY"
        )
    )

    # ------------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------------
    gpkg = a.output_dir / "modeling_subcatchments.gpkg"
    attrs_csv = a.output_dir / "modeling_subcatchment_attributes.csv"
    routing_csv = a.output_dir / "modeling_subcatchment_routing.csv"
    raw_map_csv = a.output_dir / "raw_reach_to_subcatchment.csv"
    qc_csv = a.output_dir / "modeling_subcatchment_qc.csv"
    meta_json = a.output_dir / "modeling_subcatchment_metadata.json"

    if gpkg.exists():
        if a.overwrite:
            gpkg.unlink()
        else:
            raise RuntimeError(f"{gpkg} exists; use --overwrite")

    sub.to_file(
        gpkg,
        layer="modeling_subcatchments",
        driver="GPKG",
    )

    residual.to_file(
        gpkg,
        layer="uncovered_residual",
        driver="GPKG",
        mode="a",
    )

    # Also save basin for visual QC.
    basin.to_file(
        gpkg,
        layer="authoritative_basin",
        driver="GPKG",
        mode="a",
    )

    attr_cols = [
        c for c in sub.columns
        if c != "geometry"
    ]
    atomic_csv(
        pd.DataFrame(sub[attr_cols]),
        attrs_csv,
    )
    atomic_csv(routing_out, routing_csv)

    raw_map = pd.DataFrame({
        "nhdplusid": list(node_to_sc.keys()),
        "subcatchment_id": list(node_to_sc.values()),
        "primary_downstream_nhdplusid": [
            primary_downstream.get(rid)
            for rid in node_to_sc.keys()
        ],
        "primary_downstream_method": [
            primary_method.get(rid, "BASIN_OUTLET")
            for rid in node_to_sc.keys()
        ],
        "routing_steps_to_gauge": [
            distance[rid]
            for rid in node_to_sc.keys()
        ],
    }).sort_values(
        ["subcatchment_id", "routing_steps_to_gauge"],
        ascending=[True, False],
    )

    atomic_csv(raw_map, raw_map_csv)
    atomic_csv(qcdf, qc_csv)

    metadata = {
        "script_build": BUILD,
        "status": status,
        "safe_for_a2_4": bool(safe),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spatial_discretization": {
            "target_area_km2": a.target_area_km2,
            "preferred_min_area_km2": a.min_area_km2,
            "preferred_max_area_km2": a.max_area_km2,
            "subcatchment_count": n_units,
            "area_min_km2": float(areas.min()),
            "area_median_km2": float(np.median(areas)),
            "area_mean_km2": float(np.mean(areas)),
            "area_max_km2": float(areas.max()),
            "total_subcatchment_area_km2": float(areas.sum()),
        },
        "routing": {
            "raw_node_count": tree.number_of_nodes(),
            "raw_primary_edge_count": tree.number_of_edges(),
            "aggregated_edge_count": unit_graph.number_of_edges(),
            "dn_hydroseq_selections": int(
                sum(v == "DnHydroSeq" for v in primary_method.values())
            ),
            "fallback_selections": int(
                sum(v != "DnHydroSeq" for v in primary_method.values())
            ),
            "basin_outlet_subcatchment_id": rename[outlet_temp_unit],
            "all_units_route_to_outlet": bool(all_units_reach_outlet),
        },
        "domain": {
            "coverage_percent": float(coverage_pct),
            "uncovered_residual_area_km2": residual_area_km2,
        },
        "blocking_failure_count": int(len(bf)),
        "quality_failure_count": int(len(qf)),
        "warning_count": int(
            ((qcdf["severity"] == "WARNING")
             & (qcdf["status"] == "WARN")).sum()
        ),
        "outputs": {
            "gpkg": str(gpkg),
            "attributes_csv": str(attrs_csv),
            "routing_csv": str(routing_csv),
            "raw_reach_map_csv": str(raw_map_csv),
            "qc_csv": str(qc_csv),
        },
    }

    atomic_json(metadata, meta_json)

    print("AGGREGATED MODELING SUBCATCHMENTS")
    print("-" * 100)
    print(f"Subcatchment count                  : {n_units}")
    print(f"Total modeled area                  : {areas.sum():,.3f} km²")
    print(f"Domain coverage                     : {coverage_pct:.6f} %")
    print(f"Uncovered residual                  : {residual_area_km2:.3f} km²")
    print(f"Area minimum                        : {areas.min():.3f} km²")
    print(f"Area median                         : {np.median(areas):.3f} km²")
    print(f"Area mean                           : {np.mean(areas):.3f} km²")
    print(f"Area maximum                        : {areas.max():.3f} km²")
    print(
        f"Below preferred minimum             : "
        f"{int(np.sum(areas < a.min_area_km2))}"
    )
    print(
        f"Above preferred maximum             : "
        f"{int(np.sum(areas > a.max_area_km2))}"
    )
    print(f"Aggregated routing edges            : {unit_graph.number_of_edges()}")
    print(f"Outlet subcatchment                 : {rename[outlet_temp_unit]}")
    print()
    print("READINESS")
    print("-" * 100)
    print(f"Blocking failures                   : {len(bf)}")
    print(f"Quality failures                    : {len(qf)}")
    print(
        f"Warnings                            : "
        f"{metadata['warning_count']}"
    )
    print(f"Safe for A2.4                       : {'YES' if safe else 'NO'}")
    print(f"Status                              : {status}")
    print(f"GeoPackage                          : {gpkg}")
    print(f"Routing                             : {routing_csv}")
    print(f"Metadata                            : {meta_json}")

    if not safe:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
