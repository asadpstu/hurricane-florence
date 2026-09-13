#!/usr/bin/env python3
"""
ML V4 Florence post-test improvement evaluation.
Architecture revision was motivated by the known ML V2 Florence failure.
No observed-Q lags or recursive discharge state are used.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ml_v3_rainfall_runtime import (
    atomic_csv, atomic_json, build_features, hydrologic_metrics,
    select_forcing, utc_timestamp,
)

BUILD="ML_V4_FLORENCE_EXOGENOUS_HIGHFLOW_POSTTEST"

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--spatial-rainfall",type=Path,required=True)
    p.add_argument("--routing-features",type=Path,required=True)
    p.add_argument("--feature-metadata",type=Path,required=True)
    p.add_argument("--frozen-model",type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--forcing",type=Path); g.add_argument("--forcing-dir",type=Path)
    p.add_argument("--feature-start",default="2018-06-01T00:00:00Z")
    p.add_argument("--feature-end",default="2018-09-27T00:00:00Z")
    p.add_argument("--evaluation-start",default="2018-09-10T00:00:00Z")
    p.add_argument("--evaluation-end",default="2018-09-26T00:00:00Z")
    p.add_argument("--expected-subcatchments",type=int,default=37)
    p.add_argument("--min-routed-area-coverage-percent",type=float,default=99.0)
    p.add_argument("--min-evaluation-q-coverage-percent",type=float,default=90.0)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()

def inverse_target(y,name):
    y=np.asarray(y,float)
    if name=="raw": return np.maximum(y,0.0)
    if name=="sqrt": return np.maximum(y,0.0)**2
    if name=="log1p": return np.maximum(np.expm1(y),0.0)
    raise ValueError(name)

def main():
    a=parse_args()
    fs,fe=utc_timestamp(a.feature_start),utc_timestamp(a.feature_end)
    es,ee=utc_timestamp(a.evaluation_start),utc_timestamp(a.evaluation_end)
    meta=json.loads(a.feature_metadata.read_text(encoding="utf-8"))
    static_features=meta.get("static_feature_columns") or meta.get("feature_columns")
    if not static_features:
        raise RuntimeError("No static feature columns in metadata.")

    bundle=joblib.load(a.frozen_model)
    if bundle.get("feature_columns")!=static_features:
        raise RuntimeError("Feature order mismatch between metadata and ML V4 model.")
    if bundle.get("recursive_q_state") is not False:
        raise RuntimeError("ML V4 model unexpectedly declares recursive Q state.")
    model=bundle["model"]; transform=bundle["target_transform"]

    feature, rebuilt, coverage_cols, scids, zones, basin_area, distcol = build_features(
        spatial_rainfall_path=a.spatial_rainfall,
        routing_features_path=a.routing_features,
        feature_start=fs, feature_end=fe,
        expected_subcatchments=a.expected_subcatchments,
        minimum_routed_coverage_percent=a.min_routed_area_coverage_percent,
    )
    if rebuilt!=static_features:
        raise RuntimeError(f"2018 static features mismatch: expected={len(static_features)} rebuilt={len(rebuilt)}")

    candidate=select_forcing(
        forcing_path=a.forcing, forcing_dir=a.forcing_dir,
        evaluation_start=es, evaluation_end=ee,
    )
    forcing=pd.read_csv(candidate["path"])
    forcing["interval_end_utc"]=pd.to_datetime(forcing[candidate["time_col"]],utc=True,errors="raise").dt.floor("h")
    forcing["q_obs_m3s"]=pd.to_numeric(forcing[candidate["q_col"]],errors="coerce")
    q=forcing.groupby("interval_end_utc")["q_obs_m3s"].mean().sort_index()

    times=pd.date_range(es,ee,freq="h",inclusive="left")
    x=feature.reindex(times)
    X=x[static_features].to_numpy(np.float32)
    ready=np.isfinite(X).all(axis=1)
    coverage=100.0*ready.mean()
    if coverage<99.0:
        bad=list(times[~ready][:10])
        raise RuntimeError(f"Florence static-feature coverage {coverage:.3f}% < 99%; first={bad}")
    ypred=np.full(len(times),np.nan)
    ypred[ready]=inverse_target(model.predict(X[ready]),transform)

    result=pd.DataFrame({
        "interval_end_utc":times,
        "q_obs_m3s":q.reindex(times).to_numpy(float),
        "q_pred_m3s":ypred,
    })
    common=np.isfinite(result["q_obs_m3s"]) & np.isfinite(result["q_pred_m3s"])
    qcov=100.0*np.isfinite(result["q_obs_m3s"]).mean()
    if qcov<a.min_evaluation_q_coverage_percent:
        raise RuntimeError(f"Observed-Q coverage {qcov:.3f}% below requirement.")
    metrics=hydrologic_metrics(
        result.loc[common,"q_obs_m3s"].to_numpy(float),
        result.loc[common,"q_pred_m3s"].to_numpy(float),
        result.loc[common,"interval_end_utc"],
    )

    a.output_dir.mkdir(parents=True,exist_ok=True)
    pred_path=a.output_dir/"ml_v4_florence_2018_predictions.csv"
    if pred_path.exists() and not a.overwrite:
        raise FileExistsError(f"{pred_path} exists. Use --overwrite.")
    atomic_csv(result,pred_path)
    atomic_csv(pd.DataFrame([metrics]),a.output_dir/"ml_v4_florence_2018_metrics.csv")
    atomic_json({
        "script_build":BUILD,
        "status":"PASS_ML_V4_FLORENCE_POSTTEST_EVALUATION",
        "scientific_label":"post_test_architecture_improvement_not_untouched_validation",
        "protocol":{
            "florence_used_for_model_fit":False,
            "florence_used_for_candidate_selection":False,
            "architecture_revision_informed_by_prior_florence_v2_failure":True,
            "observed_q_lags":False,
            "recursive_q_state":False,
            "future_rainfall":False,
        },
        "selected_candidate":bundle.get("selected_candidate"),
        "feature_count":len(static_features),
        "static_feature_coverage_percent":coverage,
        "observed_q_coverage_percent":qcov,
        "metrics":metrics,
        "outputs":{"predictions":str(pred_path)},
    },a.output_dir/"ml_v4_florence_metadata.json")

    print("="*108)
    print(f"SCRIPT BUILD                       : {BUILD}")
    print("ML V4 - FLORENCE POST-TEST EVALUATION")
    print("="*108)
    print("Observed Q lags                    : NO")
    print("Recursive prediction               : NO")
    print("Florence used for fit/selection    : NO")
    print(f"Feature coverage                   : {coverage:.3f}%")
    print(f"Observed Q coverage                : {qcov:.3f}%")
    print("\nFLORENCE METRICS")
    print("-"*108)
    for k in ["nse","kge","rmse_m3s","mae_m3s","pbias_percent","peak_error_percent","peak_timing_error_h"]:
        print(f"{k:35s}: {metrics.get(k)}")
    print(f"Predictions                        : {pred_path}")
    print("Status                             : PASS_ML_V4_FLORENCE_POSTTEST_EVALUATION")

if __name__=="__main__":
    main()
