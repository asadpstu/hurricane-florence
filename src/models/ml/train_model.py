#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

BUILD = "ML_RETAINED_EXOGENOUS_HIGHFLOW_EVENT_SELECTION"

# Small predeclared grid. No Florence-based selection.
CANDIDATES = [
    # id, depth, lr, trees, child, subsample, colsample, lambda, alpha, transform, weights
    ("V401", 2, 0.030, 700, 20, 0.85, 0.85, 10.0, 0.25, "sqrt",  "W1"),
    ("V402", 3, 0.025, 800, 15, 0.85, 0.85,  8.0, 0.10, "sqrt",  "W1"),
    ("V403", 3, 0.030, 700, 10, 0.90, 0.85,  5.0, 0.10, "sqrt",  "W2"),
    ("V404", 4, 0.020, 900, 10, 0.90, 0.80,  5.0, 0.05, "sqrt",  "W2"),
    ("V405", 2, 0.030, 700, 15, 0.85, 0.85,  8.0, 0.10, "raw",   "W1"),
    ("V406", 3, 0.025, 800, 10, 0.90, 0.85,  5.0, 0.05, "raw",   "W2"),
    ("V407", 4, 0.020, 900,  8, 0.90, 0.80,  3.0, 0.05, "raw",   "W3"),
    ("V408", 2, 0.035, 700, 15, 0.85, 0.85,  8.0, 0.10, "log1p", "W1"),
    ("V409", 3, 0.030, 750, 10, 0.90, 0.85,  5.0, 0.05, "log1p", "W2"),
    ("V410", 4, 0.020, 900,  8, 0.90, 0.80,  3.0, 0.05, "log1p", "W3"),
]

WEIGHT_SCHEMES = {
    "W1": (1.0, 1.5, 3.0, 6.0),
    "W2": (1.0, 2.0, 4.0, 8.0),
    "W3": (1.0, 2.5, 5.0, 10.0),
}

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--event-blocks", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument("--min-validation-nse", type=float, default=0.30)
    p.add_argument("--min-validation-kge", type=float, default=0.30)
    p.add_argument("--max-validation-abs-pbias-percent", type=float, default=30.0)
    p.add_argument("--max-validation-abs-peak-error-percent", type=float, default=40.0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def metric(obs, sim, times):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    times = pd.DatetimeIndex(times)
    valid = np.isfinite(obs) & np.isfinite(sim)
    obs, sim, times = obs[valid], sim[valid], times[valid]
    if len(obs) < 2:
        return {k: np.nan for k in [
            "nse","kge","rmse_m3s","mae_m3s","pbias_percent",
            "peak_error_percent","peak_timing_error_h","observed_peak_m3s",
            "predicted_peak_m3s"
        ]} | {"n": int(len(obs))}
    mo = float(np.mean(obs))
    denom = float(np.sum((obs-mo)**2))
    nse = 1.0 - float(np.sum((sim-obs)**2))/denom if denom > 0 else np.nan
    so, ss = float(np.std(obs)), float(np.std(sim))
    r = float(np.corrcoef(obs,sim)[0,1]) if so > 0 and ss > 0 else np.nan
    alpha = ss/so if so > 0 else np.nan
    beta = float(np.mean(sim)/np.mean(obs)) if abs(np.mean(obs)) > 1e-12 else np.nan
    kge = 1.0 - math.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2) if np.isfinite([r,alpha,beta]).all() else np.nan
    oi, si = int(np.argmax(obs)), int(np.argmax(sim))
    return {
        "n": int(len(obs)), "nse": float(nse), "kge": float(kge),
        "rmse_m3s": float(np.sqrt(np.mean((sim-obs)**2))),
        "mae_m3s": float(np.mean(np.abs(sim-obs))),
        "pbias_percent": float(100*np.sum(sim-obs)/np.sum(obs)),
        "peak_error_percent": float(100*(sim[si]-obs[oi])/obs[oi]),
        "peak_timing_error_h": float((times[si]-times[oi])/pd.Timedelta(hours=1)),
        "observed_peak_m3s": float(obs[oi]), "predicted_peak_m3s": float(sim[si]),
        "observed_peak_time_utc": str(times[oi]), "predicted_peak_time_utc": str(times[si]),
    }

def resolve_event_columns(df):
    lookup = {c.lower(): c for c in df.columns}
    def pick(names):
        for x in names:
            if x.lower() in lookup:
                return lookup[x.lower()]
        raise RuntimeError(f"Missing event column; tried {names}; available={list(df.columns)}")
    return (
        pick(["event_block_id","event_id","block_id"]),
        pick(["block_start_utc","start_utc","event_start_utc"]),
        pick(["block_end_utc","end_utc","event_end_utc"]),
    )

def candidate_dict(row):
    cid, depth, lr, trees, child, subsample, colsample, reg_lambda, reg_alpha, transform, weights = row
    return {
        "candidate_id": cid, "max_depth": depth, "learning_rate": lr,
        "n_estimators": trees, "min_child_weight": child, "subsample": subsample,
        "colsample_bytree": colsample, "reg_lambda": reg_lambda, "reg_alpha": reg_alpha,
        "target_transform": transform, "weight_scheme": weights,
    }

def model_for(c, seed, n_jobs):
    return XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse", random_state=seed,
        n_jobs=n_jobs, tree_method="hist", max_depth=c["max_depth"],
        learning_rate=c["learning_rate"], n_estimators=c["n_estimators"],
        min_child_weight=c["min_child_weight"], subsample=c["subsample"],
        colsample_bytree=c["colsample_bytree"], reg_lambda=c["reg_lambda"],
        reg_alpha=c["reg_alpha"],
    )

def transform_target(y, name):
    y = np.maximum(np.asarray(y,float),0.0)
    if name=="raw": return y
    if name=="sqrt": return np.sqrt(y)
    if name=="log1p": return np.log1p(y)
    raise ValueError(name)

def inverse_target(y, name):
    y = np.asarray(y,float)
    if name=="raw": return np.maximum(y,0.0)
    if name=="sqrt": return np.maximum(y,0.0)**2
    if name=="log1p": return np.maximum(np.expm1(y),0.0)
    raise ValueError(name)

def highflow_weights(y, scheme):
    y=np.asarray(y,float)
    q75,q90,q95=np.quantile(y,[0.75,0.90,0.95])
    vals=WEIGHT_SCHEMES[scheme]
    w=np.full(len(y),vals[0],float)
    w[y>=q75]=vals[1]; w[y>=q90]=vals[2]; w[y>=q95]=vals[3]
    return w, {"q75":float(q75),"q90":float(q90),"q95":float(q95),"weights":list(vals)}

def score_metric(m):
    if not np.isfinite([m["nse"],m["kge"],m["pbias_percent"],m["peak_error_percent"],m["peak_timing_error_h"]]).all():
        return -1e9
    return (
        0.30*m["nse"]
        + 0.25*m["kge"]
        - 0.15*abs(m["pbias_percent"])/100.0
        - 0.20*abs(m["peak_error_percent"])/100.0
        - 0.10*min(abs(m["peak_timing_error_h"])/72.0,2.0)
    )

def robust_score(pooled, events):
    vals=np.asarray([score_metric(r) for r in events.to_dict("records")],float)
    return float(0.35*score_metric(pooled)+0.45*np.mean(vals)+0.20*np.min(vals))

def event_metrics(pred, events):
    rows=[]
    for _,e in events.iterrows():
        z=pred[(pred["interval_end_utc"]>=e["start"]) & (pred["interval_end_utc"]<=e["end"])]
        m=metric(z["q_obs_m3s"],z["q_pred_m3s"],z["interval_end_utc"])
        rows.append({"event_block_id":str(e["event_block_id"]),**m})
    return pd.DataFrame(rows)

def main():
    a=args()
    meta=json.loads(a.metadata.read_text(encoding="utf-8"))
    static_features=meta.get("static_feature_columns") or meta.get("feature_columns")
    if not static_features:
        raise RuntimeError("No static feature list in metadata.")

    df=pd.read_csv(a.dataset)
    needed={"interval_end_utc","q_obs_m3s","split","static_feature_ready",*static_features}
    missing=needed-set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset missing columns: {sorted(missing)}")
    df["interval_end_utc"]=pd.to_datetime(df["interval_end_utc"],utc=True,errors="raise")
    if (df["interval_end_utc"]>=pd.Timestamp("2018-01-01T00:00:00Z")).any():
        raise RuntimeError("Development dataset contains 2018+ rows.")

    ready=df["static_feature_ready"].astype(str).str.lower().isin(("true","1"))
    finite_q=np.isfinite(pd.to_numeric(df["q_obs_m3s"],errors="coerce"))
    train=df[ready & finite_q & df["split"].eq("TRAIN_2015_2016")].copy()
    val=df[ready & finite_q & df["split"].eq("VALIDATION_2017")].copy()
    if len(train)<1000 or len(val)<1000:
        raise RuntimeError(f"Too few rows: train={len(train)}, validation={len(val)}")

    Xtr=train[static_features].to_numpy(np.float32)
    ytr=train["q_obs_m3s"].to_numpy(float)
    Xva=val[static_features].to_numpy(np.float32)

    blocks=pd.read_csv(a.event_blocks)
    idc,sc,ec=resolve_event_columns(blocks)
    blocks=blocks.rename(columns={idc:"event_block_id",sc:"start",ec:"end"})
    blocks["start"]=pd.to_datetime(blocks["start"],utc=True,errors="raise")
    blocks["end"]=pd.to_datetime(blocks["end"],utc=True,errors="raise")
    val_events=blocks[blocks["event_block_id"].astype(str).str.startswith("VAL_BLOCK_")].copy()
    if val_events.empty:
        raise RuntimeError("No VAL_BLOCK_* rows found.")

    print("="*112)
    print(f"SCRIPT BUILD                       : {BUILD}")
    print("ML - EXOGENOUS HIGH-FLOW-AWARE XGBOOST")
    print("="*112)
    print(f"Training rows                      : {len(train)}")
    print(f"Validation model-ready rows        : {len(val)}")
    print(f"Features                           : {len(static_features)}")
    print(f"Candidates                         : {len(CANDIDATES)}")
    print("Observed Q lags                    : NO")
    print("Recursive prediction               : NO")
    print("High-flow training weighting       : YES")
    print("2017 event-block selection         : YES")
    print("Florence 2018 used                 : NO")
    print()

    rows=[]; models={}; preds={}; evs={}; weight_meta={}
    for raw in CANDIDATES:
        c=candidate_dict(raw)
        w,wm=highflow_weights(ytr,c["weight_scheme"])
        model=model_for(c,a.seed,a.n_jobs)
        model.fit(Xtr,transform_target(ytr,c["target_transform"]),sample_weight=w)
        ptr=inverse_target(model.predict(Xtr),c["target_transform"])
        pva=inverse_target(model.predict(Xva),c["target_transform"])
        pred=val[["interval_end_utc","q_obs_m3s"]].copy()
        pred["q_pred_m3s"]=pva
        event_mask=np.zeros(len(pred),dtype=bool)
        for _,e in val_events.iterrows():
            event_mask |= ((pred["interval_end_utc"]>=e["start"]) & (pred["interval_end_utc"]<=e["end"])).to_numpy()
        pooled_df=pred.loc[event_mask]
        pooled=metric(pooled_df["q_obs_m3s"],pooled_df["q_pred_m3s"],pooled_df["interval_end_utc"])
        ev=event_metrics(pred,val_events)
        s=robust_score(pooled,ev)
        mtr=metric(ytr,ptr,train["interval_end_utc"])
        rows.append({
            **c,"train_nse":mtr["nse"],"train_kge":mtr["kge"],
            "validation_event_pooled_nse":pooled["nse"],
            "validation_event_pooled_kge":pooled["kge"],
            "validation_event_pooled_pbias_percent":pooled["pbias_percent"],
            "validation_event_pooled_peak_error_percent":pooled["peak_error_percent"],
            "validation_event_pooled_peak_timing_error_h":pooled["peak_timing_error_h"],
            "validation_event_min_nse":float(ev["nse"].min()),
            "validation_event_min_kge":float(ev["kge"].min()),
            "validation_event_max_abs_pbias_percent":float(ev["pbias_percent"].abs().max()),
            "validation_event_max_abs_peak_error_percent":float(ev["peak_error_percent"].abs().max()),
            "selection_score":s,
        })
        models[c["candidate_id"]]=model; preds[c["candidate_id"]]=pred; evs[c["candidate_id"]]=ev
        weight_meta[c["candidate_id"]]=wm
        print(f"{c['candidate_id']} | {c['target_transform']:5s} {c['weight_scheme']} | "
              f"NSE={pooled['nse']:.4f} KGE={pooled['kge']:.4f} "
              f"PBIAS={pooled['pbias_percent']:+.1f}% peak={pooled['peak_error_percent']:+.1f}% "
              f"timing={pooled['peak_timing_error_h']:+.0f}h minEvNSE={ev['nse'].min():+.3f} score={s:.4f}")

    results=pd.DataFrame(rows).sort_values(
        ["selection_score","validation_event_pooled_kge","validation_event_pooled_nse"],
        ascending=False
    ).reset_index(drop=True)
    sid=str(results.iloc[0]["candidate_id"])
    sel=next(candidate_dict(x) for x in CANDIDATES if x[0]==sid)
    model=models[sid]; pred=preds[sid]; ev=evs[sid]
    mask=np.zeros(len(pred),dtype=bool)
    for _,e in val_events.iterrows():
        mask |= ((pred["interval_end_utc"]>=e["start"]) & (pred["interval_end_utc"]<=e["end"])).to_numpy()
    pooled_df=pred.loc[mask]
    mva=metric(pooled_df["q_obs_m3s"],pooled_df["q_pred_m3s"],pooled_df["interval_end_utc"])
    ptr=inverse_target(model.predict(Xtr),sel["target_transform"])
    mtr=metric(ytr,ptr,train["interval_end_utc"])

    quality={
        "validation_event_pooled_nse":mva["nse"]>=a.min_validation_nse,
        "validation_event_pooled_kge":mva["kge"]>=a.min_validation_kge,
        "validation_event_pooled_abs_pbias":abs(mva["pbias_percent"])<=a.max_validation_abs_pbias_percent,
        "validation_event_pooled_abs_peak_error":abs(mva["peak_error_percent"])<=a.max_validation_abs_peak_error_percent,
    }
    failures=sum(not x for x in quality.values())
    status="PASS_ML_RETAINED_VALIDATION" if failures==0 else "FAIL_ML_RETAINED_VALIDATION_QUALITY"

    a.output_dir.mkdir(parents=True,exist_ok=True)
    model_path=a.output_dir/"model.joblib"
    if model_path.exists() and not a.overwrite:
        raise FileExistsError(f"{model_path} exists. Use --overwrite.")
    tmp=model_path.with_name(model_path.name+".partial")
    joblib.dump({
        "model":model,"feature_columns":static_features,
        "target_transform":sel["target_transform"],
        "selected_candidate":sel,"recursive_q_state":False,"script_build":BUILD,
    },tmp)
    os.replace(tmp,model_path)
    write_csv(results,a.output_dir/"candidate_metrics.csv")
    write_csv(pred,a.output_dir/"holdout_predictions.csv")
    write_csv(ev,a.output_dir/"holdout_event_metrics.csv")
    gain=model.get_booster().get_score(importance_type="gain")
    imp=pd.DataFrame([{"feature":name,"gain":float(gain.get(f"f{i}",0.0))} for i,name in enumerate(static_features)]).sort_values("gain",ascending=False)
    write_csv(imp,a.output_dir/"feature_importance.csv")
    metadata={
        "script_build":BUILD,"status":status,"quality_failures":int(failures),"quality_checks":quality,
        "protocol":{
            "training":"2015-2016","validation_model_selection":"2017 VAL_BLOCK events",
            "observed_q_lags":False,"recursive_prediction":False,
            "high_flow_weighting":True,"future_rainfall":False,
            "florence_2018_used_for_fit_or_selection":False,
            "architecture_revision_informed_by_prior_florence_v2_failure":True,
        },
        "selected_candidate":sel,"selected_weight_thresholds":weight_meta[sid],
        "training_metrics":mtr,"validation_event_pooled_metrics":mva,
        "validation_event_metrics":ev.to_dict("records"),
        "feature_count":len(static_features),"feature_columns":static_features,
        "input_dataset":str(a.dataset),"input_dataset_sha256":sha256(a.dataset),
        "feature_metadata":str(a.metadata),"event_blocks":str(a.event_blocks),
        "frozen_model":str(model_path),"frozen_model_sha256":sha256(model_path),
    }
    write_json(metadata,a.output_dir/"metadata.json")

    print("\nSELECTED MODEL")
    print("-"*112)
    print(f"Candidate                          : {sid}")
    print(f"Parameters                         : {sel}")
    print(f"High-flow thresholds/weights       : {weight_meta[sid]}")
    print("\nTRAINING METRICS - 2015/2016")
    print("-"*112)
    print(f"NSE                                : {mtr['nse']:.6f}")
    print(f"KGE                                : {mtr['kge']:.6f}")
    print(f"PBIAS                              : {mtr['pbias_percent']:+.3f}%")
    print("\nVALIDATION METRICS - 2017 EVENT WINDOWS")
    print("-"*112)
    print(f"NSE                                : {mva['nse']:.6f}")
    print(f"KGE                                : {mva['kge']:.6f}")
    print(f"PBIAS                              : {mva['pbias_percent']:+.3f}%")
    print(f"Peak error                         : {mva['peak_error_percent']:+.3f}%")
    print(f"Peak timing error                  : {mva['peak_timing_error_h']:+.1f} h")
    print("\nEVENT ROBUSTNESS")
    print("-"*112)
    for _,r in ev.iterrows():
        print(f"{r['event_block_id']} | NSE={r['nse']:.3f} KGE={r['kge']:.3f} "
              f"PBIAS={r['pbias_percent']:+.1f}% peak={r['peak_error_percent']:+.1f}% "
              f"timing={r['peak_timing_error_h']:+.0f}h")
    print("\nREADINESS")
    print("-"*112)
    print(f"Quality failures                   : {failures}")
    print("Florence 2018 used for selection   : NO")
    print("Observed Q lags                    : NO")
    print("Recursive prediction               : NO")
    print(f"Status                             : {status}")
    print(f"Model                              : {model_path}")

if __name__=="__main__":
    main()
