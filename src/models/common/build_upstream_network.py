from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import Point

BUILD = "MODEL_IMPROVEMENT_A2_2_UPSTREAM_NHDPLUS_FRAMEWORK_V1"

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--gdb",type=Path,required=True)
    p.add_argument("--basin",type=Path,required=True)
    p.add_argument("--gauge-lon",type=float,default=-77.9975)
    p.add_argument("--gauge-lat",type=float,default=35.3375)
    p.add_argument("--expected-basin-area-km2",type=float,default=6232.005499)
    p.add_argument("--candidate-radius-m",type=float,default=1500)
    p.add_argument("--max-gauge-snap-distance-m",type=float,default=500)
    p.add_argument("--max-outlet-da-error-percent",type=float,default=5)
    p.add_argument("--min-catchment-coverage-percent",type=float,default=98)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()

def findcol(df,*names,required=True):
    m={str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in m:return m[n.lower()]
    if required: raise RuntimeError(f"Missing columns {names}; have {list(df.columns)}")
    return None

def ids(s):
    x=pd.to_numeric(s,errors="coerce")
    o=pd.Series(pd.NA,index=s.index,dtype="Int64")
    ok=x.notna()&np.isfinite(x)
    o.loc[ok]=np.rint(x.loc[ok]).astype("int64")
    return o

def main():
    a=args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    layers=set(pyogrio.list_layers(a.gdb)[:,0].tolist())
    req={"NHDFlowline","NHDPlusCatchment","NHDPlusFlowlineVAA","NHDPlusFlow"}
    miss=sorted(req-layers)
    if miss: raise RuntimeError(f"Missing required GDB layers/tables: {miss}")

    print("="*100); print(f"SCRIPT BUILD                       : {BUILD}")
    print("MODEL IMPROVEMENT A2.2 - FULL-UPSTREAM NHDPLUS FRAMEWORK"); print("="*100)

    flow=gpd.read_file(a.gdb,layer="NHDFlowline")
    catch=gpd.read_file(a.gdb,layer="NHDPlusCatchment")
    vaa=pyogrio.read_dataframe(a.gdb,layer="NHDPlusFlowlineVAA")
    pf=pyogrio.read_dataframe(a.gdb,layer="NHDPlusFlow")
    print(f"NHDFlowline                        : {len(flow):,}")
    print(f"NHDPlusCatchment                   : {len(catch):,}")
    print(f"NHDPlusFlowlineVAA                 : {len(vaa):,}")
    print(f"NHDPlusFlow                        : {len(pf):,}")

    fid=findcol(flow,"NHDPlusID"); cid=findcol(catch,"NHDPlusID")
    vid=findcol(vaa,"NHDPlusID"); fr=findcol(pf,"FromNHDPID","FromNHDPlusID")
    to=findcol(pf,"ToNHDPID","ToNHDPlusID")
    flow["_id"]=ids(flow[fid]); catch["_id"]=ids(catch[cid])
    vaa["_id"]=ids(vaa[vid]); pf["_from"]=ids(pf[fr]); pf["_to"]=ids(pf[to])
    vaa=vaa[vaa["_id"].notna()].drop_duplicates("_id")

    hydro=findcol(vaa,"HydroSeq")
    tot=findcol(vaa,"TotDASqKm","TotDASqKM")
    order=findcol(vaa,"StreamOrde","StreamOrder",required=False)
    dn=findcol(vaa,"DnHydroSeq",required=False)
    level=findcol(vaa,"LevelPathI","LevelPathId",required=False)
    div=findcol(vaa,"Divergence",required=False)
    plen=findcol(vaa,"PathLength",required=False)
    keep=["_id"]+[x for x in [hydro,tot,order,dn,level,div,plen] if x]
    keep=list(dict.fromkeys(keep))
    fj=flow.merge(vaa[keep],on="_id",how="left",validate="many_to_one")

    print("\nROUTING ATTRIBUTES")
    print("-"*100)
    for k,v in [("HydroSeq",hydro),("TotDASqKm",tot),("StreamOrder",order),
                ("DnHydroSeq",dn),("LevelPath",level),("Divergence",div),("PathLength",plen)]:
        print(f"{k:35s}: {v}")

    gauge=gpd.GeoDataFrame({"site_no":["02089000"]},
        geometry=[Point(a.gauge_lon,a.gauge_lat)],crs="EPSG:4326").to_crs("EPSG:32618")
    fu=fj.to_crs("EPSG:32618")
    fu["_dist_m"]=fu.geometry.distance(gauge.geometry.iloc[0])
    cand=fu[(fu["_dist_m"]<=a.candidate_radius_m)&fu[tot].notna()].copy()
    if cand.empty: raise RuntimeError("No outlet candidate near gauge with TotDASqKm.")
    cand["_da_err_abs_pct"]=100*(pd.to_numeric(cand[tot])-a.expected_basin_area_km2).abs()/a.expected_basin_area_km2
    cand["_score"]=cand["_da_err_abs_pct"]+0.002*cand["_dist_m"]
    cand=cand.sort_values(["_score","_da_err_abs_pct","_dist_m"])
    name=findcol(cand,"GNIS_Name","GNISName",required=False)

    print("\nOUTLET CANDIDATES")
    print("-"*100)
    for _,r in cand.head(10).iterrows():
        nm=f" | {r.get(name)}" if name else ""
        print(f"ID={int(r['_id'])} | dist={r['_dist_m']:.2f} m | "
              f"TotDA={float(r[tot]):,.3f} km² | DAerr={r['_da_err_abs_pct']:.3f}%{nm}")

    out=cand.iloc[0]; outlet=int(out["_id"]); snap=float(out["_dist_m"])
    oda=float(out[tot]); daerr=100*(oda-a.expected_basin_area_km2)/a.expected_basin_area_km2
    print("\nSELECTED OUTLET")
    print("-"*100)
    print(f"NHDPlusID                          : {outlet}")
    print(f"Gauge snap distance                : {snap:.3f} m")
    print(f"TotDASqKm                          : {oda:,.6f} km²")
    print(f"Drainage-area difference           : {daerr:+.6f} %")
    if name: print(f"GNIS name                           : {out.get(name)}")

    rev=defaultdict(list)
    edges=pf[pf["_from"].notna()&pf["_to"].notna()].copy()
    for f,t in zip(edges["_from"].astype("int64"),edges["_to"].astype("int64")):
        if f>0 and t>0: rev[int(t)].append(int(f))
    upstream={outlet}; q=deque([outlet])
    while q:
        d=q.popleft()
        for u in rev.get(d,[]):
            if u not in upstream: upstream.add(u); q.append(u)

    uf=fj[fj["_id"].isin(upstream)].copy()
    uc=catch[catch["_id"].isin(upstream)].copy()
    re=edges[edges["_from"].astype("Int64").isin(upstream)&edges["_to"].astype("Int64").isin(upstream)].copy()
    routing=pd.DataFrame({"from_nhdplusid":re["_from"].astype("int64"),
                          "to_nhdplusid":re["_to"].astype("int64")}).drop_duplicates()

    basin=gpd.read_file(a.basin).to_crs("EPSG:32618")
    bg=basin.geometry.union_all(); ba=bg.area/1e6
    ucu=uc.to_crs("EPSG:32618")
    raw=ucu.geometry.union_all()
    rawcov=100*raw.intersection(bg).area/bg.area
    clipped=gpd.clip(ucu,basin)
    clipped=clipped[~clipped.geometry.is_empty].copy()
    clipped["model_area_km2"]=clipped.geometry.area/1e6
    clipcov=100*clipped.geometry.union_all().area/bg.area

    print("\nUPSTREAM FRAMEWORK QC")
    print("-"*100)
    print(f"Upstream IDs                       : {len(upstream):,}")
    print(f"Upstream flowlines                 : {len(uf):,}")
    print(f"Upstream catchments                : {len(uc):,}")
    print(f"Routing edges                      : {len(routing):,}")
    print(f"Authoritative basin area (UTM18)   : {ba:,.6f} km²")
    print(f"Raw catchment coverage             : {rawcov:.6f} %")
    print(f"Clipped catchment coverage         : {clipcov:.6f} %")

    qc=[]
    def add(sev,ch,ok,detail):qc.append({"severity":sev,"check":ch,"status":"PASS" if ok else "FAIL","detail":detail})
    add("BLOCKING","GAUGE_SNAP",snap<=a.max_gauge_snap_distance_m,f"{snap:.3f} m")
    add("BLOCKING","OUTLET_DRAINAGE_AREA",abs(daerr)<=a.max_outlet_da_error_percent,f"{daerr:+.6f}%")
    add("BLOCKING","UPSTREAM_NETWORK",len(uf)>0 and len(uc)>0,f"flow={len(uf)}, catch={len(uc)}")
    add("QUALITY","RAW_CATCHMENT_COVERAGE",rawcov>=a.min_catchment_coverage_percent,f"{rawcov:.6f}%")
    add("QUALITY","CLIPPED_COVERAGE",clipcov>=a.min_catchment_coverage_percent,f"{clipcov:.6f}%")
    qcdf=pd.DataFrame(qc)
    bf=qcdf[(qcdf.severity=="BLOCKING")&(qcdf.status=="FAIL")]
    qf=qcdf[(qcdf.severity=="QUALITY")&(qcdf.status=="FAIL")]
    safe=len(bf)==0 and len(qf)==0
    status="PASS_A2_2_UPSTREAM_NHDPLUS_FRAMEWORK_READY" if safe else ("FAIL_A2_2_BLOCKING" if len(bf) else "FAIL_A2_2_QUALITY")

    gpkg=a.output_dir/"usgs_02089000_upstream_nhdplus_framework.gpkg"
    if gpkg.exists():
        if a.overwrite: gpkg.unlink()
        else: raise RuntimeError(f"{gpkg} exists; use --overwrite")
    uf.to_crs("EPSG:32618").to_file(gpkg,layer="upstream_flowlines",driver="GPKG")
    clipped.to_file(gpkg,layer="upstream_catchments",driver="GPKG",mode="a")
    gauge.to_file(gpkg,layer="gauge",driver="GPKG",mode="a")

    routing.to_csv(a.output_dir/"upstream_routing_edges.csv",index=False)
    pd.DataFrame(cand.head(50).drop(columns="geometry",errors="ignore")).to_csv(
        a.output_dir/"gauge_outlet_candidates.csv",index=False)
    qcdf.to_csv(a.output_dir/"framework_qc.csv",index=False)

    meta={"script_build":BUILD,"status":status,"safe_for_a2_3":safe,
          "created_utc":datetime.now(timezone.utc).isoformat(),
          "outlet":{"nhdplusid":outlet,"snap_distance_m":snap,
                    "total_drainage_area_km2":oda,"da_difference_percent":daerr},
          "network":{"ids":len(upstream),"flowlines":len(uf),"catchments":len(uc),"routing_edges":len(routing)},
          "coverage":{"basin_area_utm18_km2":ba,"raw_percent":rawcov,"clipped_percent":clipcov},
          "blocking_failures":len(bf),"quality_failures":len(qf)}
    (a.output_dir/"framework_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")

    print("\nREADINESS")
    print("-"*100)
    print(f"Blocking failures                  : {len(bf)}")
    print(f"Quality failures                   : {len(qf)}")
    print(f"Safe for A2.3                      : {'YES' if safe else 'NO'}")
    print(f"Status                             : {status}")
    print(f"GeoPackage                         : {gpkg}")
    if not safe: raise SystemExit(1)

if __name__=="__main__":
    main()
