from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

BUILD = "MODEL_IMPROVEMENT_A4_COMMON_ROUTING_FEATURES_V1"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subcatchments-gpkg", type=Path, required=True)
    p.add_argument("--subcatchment-layer", default="modeling_subcatchments")
    p.add_argument("--routing", type=Path, required=True)
    p.add_argument("--raw-reach-map", type=Path, required=True)
    p.add_argument("--framework-gpkg", type=Path, required=True)
    p.add_argument("--flowline-layer", default="upstream_flowlines")
    p.add_argument("--expected-subcatchments", type=int, default=37)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def findcol(df, *names, required=True):
    cmap = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in cmap:
            return cmap[name.lower()]
    if required:
        raise RuntimeError(f"Missing columns {names}; available={list(df.columns)}")
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

    sub = gpd.read_file(a.subcatchments_gpkg, layer=a.subcatchment_layer)
    if len(sub) != a.expected_subcatchments:
        raise RuntimeError(
            f"Expected {a.expected_subcatchments} subcatchments, found {len(sub)}"
        )

    required_sub = {"subcatchment_id", "outlet_nhdplusid", "is_basin_outlet"}
    missing = required_sub - set(sub.columns)
    if missing:
        raise RuntimeError(f"Subcatchment layer missing {sorted(missing)}")

    sub = sub.to_crs("EPSG:32618").copy()
    sub["local_area_km2"] = sub.geometry.area / 1e6
    sub["subcatchment_id"] = sub["subcatchment_id"].astype(str)

    routing = pd.read_csv(a.routing)
    fcol = findcol(routing, "from_subcatchment_id")
    tcol = findcol(routing, "to_subcatchment_id")
    routing[fcol] = routing[fcol].astype(str)
    routing[tcol] = routing[tcol].astype(str)

    graph = nx.DiGraph()
    graph.add_nodes_from(sub["subcatchment_id"])
    graph.add_edges_from(zip(routing[fcol], routing[tcol]))

    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Subcatchment routing graph contains a cycle.")

    terminals = [n for n in graph.nodes if graph.out_degree(n) == 0]
    if len(terminals) != 1:
        raise RuntimeError(f"Expected one basin outlet; found terminals={terminals}")
    outlet_sc = terminals[0]

    flagged_outlets = sub.loc[
        sub["is_basin_outlet"].astype(bool), "subcatchment_id"
    ].tolist()
    if flagged_outlets != [outlet_sc]:
        raise RuntimeError(
            f"Routing outlet {outlet_sc} differs from polygon flag {flagged_outlets}"
        )

    raw = pd.read_csv(a.raw_reach_map)
    rid = findcol(raw, "nhdplusid")
    rdn = findcol(raw, "primary_downstream_nhdplusid")
    raw[rid] = pd.to_numeric(raw[rid], errors="coerce").round().astype("Int64")
    raw[rdn] = pd.to_numeric(raw[rdn], errors="coerce").round().astype("Int64")

    primary_down = {
        int(r[rid]): int(r[rdn])
        for _, r in raw.iterrows()
        if pd.notna(r[rid]) and pd.notna(r[rdn])
    }

    flow = gpd.read_file(a.framework_gpkg, layer=a.flowline_layer)
    flow_id = findcol(flow, "_id", "NHDPlusID")
    length_col = findcol(flow, "LengthKM", "LengthKm", required=False)

    flow["_rid"] = pd.to_numeric(flow[flow_id], errors="coerce").round().astype("Int64")

    if length_col is None:
        flow_metric = flow.to_crs("EPSG:32618")
        flow["_length_km"] = flow_metric.geometry.length / 1000.0
        length_source = "geometry_length_utm18"
    else:
        flow["_length_km"] = pd.to_numeric(flow[length_col], errors="coerce")
        length_source = length_col

    length_by_reach = {
        int(r["_rid"]): float(r["_length_km"])
        for _, r in flow.iterrows()
        if pd.notna(r["_rid"]) and pd.notna(r["_length_km"]) and float(r["_length_km"]) >= 0
    }

    outlet_reach_by_sc = {
        str(r["subcatchment_id"]): int(round(float(r["outlet_nhdplusid"])))
        for _, r in sub.iterrows()
    }
    basin_outlet_reach = outlet_reach_by_sc[outlet_sc]

    def distance_to_outlet(start_reach):
        if start_reach == basin_outlet_reach:
            return 0.0, 0
        seen = set()
        cur = start_reach
        km = 0.0
        steps = 0
        while cur != basin_outlet_reach:
            if cur in seen:
                raise RuntimeError(f"Cycle while tracing raw reach {start_reach}")
            seen.add(cur)
            nxt = primary_down.get(cur)
            if nxt is None:
                raise RuntimeError(
                    f"Reach {cur} has no primary downstream link while tracing {start_reach}"
                )
            if nxt != basin_outlet_reach:
                km += float(length_by_reach.get(nxt, 0.0))
            cur = nxt
            steps += 1
            if steps > len(primary_down) + 5:
                raise RuntimeError(f"Excessive path length tracing reach {start_reach}")
        return km, steps

    raw_sc = findcol(raw, "subcatchment_id")
    raw[raw_sc] = raw[raw_sc].astype(str)
    raw["_length_km"] = raw[rid].map(
        lambda x: length_by_reach.get(int(x), np.nan) if pd.notna(x) else np.nan
    )
    local_channel = raw.groupby(raw_sc)["_length_km"].sum(min_count=1).to_dict()

    topo = list(nx.topological_sort(graph))
    topo_index = {sc: i + 1 for i, sc in enumerate(topo)}
    local_area = dict(zip(sub["subcatchment_id"], sub["local_area_km2"].astype(float)))

    rows = []
    for sc in topo:
        ancestors = nx.ancestors(graph, sc)
        upstream_set = set(ancestors) | {sc}
        upstream_area = sum(local_area[x] for x in upstream_set)
        hops = nx.shortest_path_length(graph, source=sc, target=outlet_sc)
        dist_km, raw_steps = distance_to_outlet(outlet_reach_by_sc[sc])

        rows.append({
            "subcatchment_id": sc,
            "topological_index": topo_index[sc],
            "is_basin_outlet": sc == outlet_sc,
            "local_area_km2": local_area[sc],
            "upstream_area_km2": upstream_area,
            "upstream_subcatchment_count": len(upstream_set),
            "subcatchment_hops_to_gauge": int(hops),
            "raw_routing_steps_to_gauge": int(raw_steps),
            "network_distance_proxy_to_gauge_km": float(dist_km),
            "local_channel_length_km": float(local_channel.get(sc, np.nan)),
            "travel_time_prior_h_at_0p25_ms": dist_km * 1000.0 / 0.25 / 3600.0,
            "travel_time_prior_h_at_0p50_ms": dist_km * 1000.0 / 0.50 / 3600.0,
            "travel_time_prior_h_at_1p00_ms": dist_km * 1000.0 / 1.00 / 3600.0,
        })

    features = pd.DataFrame(rows)

    attach_cols = [
        c for c in [
            "subcatchment_id",
            "downstream_subcatchment_id",
            "outlet_nhdplusid",
            "outlet_hydroseq",
            "outlet_stream_order",
            "outlet_totdasqkm",
            "outlet_pathlength",
            "outlet_name",
            "raw_reach_count",
            "raw_catchment_count",
        ] if c in sub.columns
    ]
    if len(attach_cols) > 1:
        attrs = pd.DataFrame(sub[attach_cols]).copy()
        attrs["subcatchment_id"] = attrs["subcatchment_id"].astype(str)
        features = features.merge(
            attrs, on="subcatchment_id", how="left", validate="one_to_one"
        )

    ordered = features.sort_values("topological_index")["subcatchment_id"].tolist()
    matrix = pd.DataFrame(0, index=ordered, columns=ordered, dtype="uint8")
    for target in ordered:
        contributing = nx.ancestors(graph, target) | {target}
        matrix.loc[target, list(contributing)] = 1
    matrix.insert(0, "target_subcatchment_id", matrix.index)
    matrix = matrix.reset_index(drop=True)

    total_area = float(features["local_area_km2"].sum())
    outlet_upstream_area = float(
        features.loc[
            features["subcatchment_id"] == outlet_sc, "upstream_area_km2"
        ].iloc[0]
    )
    area_diff = outlet_upstream_area - total_area

    dmap = dict(zip(
        features["subcatchment_id"],
        features["network_distance_proxy_to_gauge_km"]
    ))
    monotonic_failures = [
        (u, v, dmap[u], dmap[v])
        for u, v in graph.edges
        if dmap[u] + 1e-9 < dmap[v]
    ]
    all_reach_outlet = all(nx.has_path(graph, sc, outlet_sc) for sc in graph.nodes)

    qrows = []
    def qc(severity, check, passed, detail):
        qrows.append({
            "severity": severity,
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    qc("BLOCKING", "SUBCATCHMENT_COUNT",
       len(features) == a.expected_subcatchments,
       f"{len(features)}/{a.expected_subcatchments}")
    qc("BLOCKING", "SINGLE_OUTLET",
       len(terminals) == 1, f"outlet={outlet_sc}")
    qc("BLOCKING", "ACYCLIC_ROUTING",
       nx.is_directed_acyclic_graph(graph),
       f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")
    qc("BLOCKING", "ALL_ROUTE_TO_GAUGE",
       all_reach_outlet, f"outlet={outlet_sc}")
    qc("QUALITY", "OUTLET_UPSTREAM_AREA_CLOSURE",
       abs(area_diff) <= 1e-6,
       f"outlet_upstream={outlet_upstream_area:.6f}, sum_local={total_area:.6f}, diff={area_diff:.9f}")
    qc("QUALITY", "ROUTING_DISTANCE_MONOTONIC",
       len(monotonic_failures) == 0,
       f"violations={len(monotonic_failures)}")
    qc("QUALITY", "OUTLET_DISTANCE_ZERO",
       abs(dmap[outlet_sc]) <= 1e-9,
       f"{outlet_sc} distance={dmap[outlet_sc]:.9f} km")

    qcdf = pd.DataFrame(qrows)
    bf = qcdf[(qcdf["severity"] == "BLOCKING") & (qcdf["status"] == "FAIL")]
    qf = qcdf[(qcdf["severity"] == "QUALITY") & (qcdf["status"] == "FAIL")]

    safe = len(bf) == 0 and len(qf) == 0
    status = (
        "PASS_A4_COMMON_ROUTING_FEATURES_READY"
        if safe else
        ("FAIL_A4_COMMON_ROUTING_FEATURES_BLOCKING"
         if len(bf) else "FAIL_A4_COMMON_ROUTING_FEATURES_QUALITY")
    )

    feature_csv = a.output_dir / "common_subcatchment_routing_features.csv"
    matrix_csv = a.output_dir / "upstream_connectivity_matrix.csv"
    qc_csv = a.output_dir / "common_routing_qc.csv"
    meta_json = a.output_dir / "common_routing_metadata.json"

    atomic_csv(features, feature_csv)
    atomic_csv(matrix, matrix_csv)
    atomic_csv(qcdf, qc_csv)

    metadata = {
        "script_build": BUILD,
        "status": status,
        "safe_to_split_into_physics_and_ml": bool(safe),
        "subcatchment_count": int(len(features)),
        "routing_edge_count": int(graph.number_of_edges()),
        "basin_outlet_subcatchment_id": outlet_sc,
        "total_modeled_area_km2": total_area,
        "maximum_network_distance_proxy_km": float(
            features["network_distance_proxy_to_gauge_km"].max()
        ),
        "maximum_subcatchment_hops_to_gauge": int(
            features["subcatchment_hops_to_gauge"].max()
        ),
        "flowline_length_source": length_source,
        "distance_definition": (
            "Sum of NHDPlus primary downstream reach lengths between each "
            "aggregated subcatchment outlet and the gauge outlet reach. "
            "This is a routing-distance proxy, not a surveyed centroid-to-gauge "
            "distance and not a calibrated travel time."
        ),
        "travel_time_prior_definition": (
            "Distance divided by illustrative velocities of 0.25, 0.50 and "
            "1.00 m/s. These are diagnostics/priors only."
        ),
        "blocking_failure_count": int(len(bf)),
        "quality_failure_count": int(len(qf)),
        "outputs": {
            "routing_features_csv": str(feature_csv),
            "upstream_connectivity_matrix_csv": str(matrix_csv),
            "qc_csv": str(qc_csv),
        },
    }
    atomic_json(metadata, meta_json)

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {BUILD}")
    print("MODEL IMPROVEMENT A4 - COMMON SUBCATCHMENT ROUTING FEATURES")
    print("=" * 100)
    print(f"Subcatchments                      : {len(features)}")
    print(f"Routing edges                      : {graph.number_of_edges()}")
    print(f"Basin outlet                       : {outlet_sc}")
    print(f"Total modeled area                 : {total_area:,.3f} km²")
    print(f"Max network-distance proxy         : {features['network_distance_proxy_to_gauge_km'].max():,.3f} km")
    print(f"Max routing hops                   : {features['subcatchment_hops_to_gauge'].max()}")
    print(f"Max 0.50 m/s travel-time prior     : {features['travel_time_prior_h_at_0p50_ms'].max():,.2f} h")
    print()
    print("READINESS")
    print("-" * 100)
    print(f"Blocking failures                  : {len(bf)}")
    print(f"Quality failures                   : {len(qf)}")
    print(f"Safe to split Physics / ML   : {'YES' if safe else 'NO'}")
    print(f"Status                             : {status}")
    print(f"Routing features                   : {feature_csv}")
    print(f"Connectivity matrix                : {matrix_csv}")
    print(f"Metadata                           : {meta_json}")

    if monotonic_failures:
        print()
        print("ROUTING DISTANCE VIOLATIONS")
        for item in monotonic_failures[:20]:
            print(item)

    if not safe:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
