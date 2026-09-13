#!/usr/bin/env python3
"""Retained Florence comparison: USGS vs Physics V3.5 vs ML V4.

Important invariants
--------------------
* Discharge metrics use only common timestamps, but daily maxima are computed
  independently for USGS, Physics and ML.  This preserves the native 15-minute
  USGS daily peak instead of silently downsampling it to model timestamps.
* USGS daily stage is taken directly from the observed row at daily maximum Q.
* Rising/falling H-Q branches are cleaned independently; their supports are not
  artificially intersected.
* Model stage uses the appropriate branch by peak timing, falls back to the
  other branch if needed, then to the nearest observed USGS H-Q pair.  The
  fallback is explicit in the detailed output.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

FT_TO_M = 0.3048
CFS_TO_CMS = 0.028316846592


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usgs-observations", type=Path, required=True)
    p.add_argument("--physics-predictions", type=Path, required=True)
    p.add_argument("--ml-predictions", type=Path, required=True)
    p.add_argument("--q-to-h", type=Path, required=True)
    p.add_argument("--noaa-root", type=Path, required=True)
    p.add_argument("--gage-datum-navd88-ft", type=float, default=41.926)
    p.add_argument("--daily-start", default="2018-09-14")
    p.add_argument("--daily-days", type=int, default=10)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve(df, candidates, label, required=True):
    m = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in m:
            return m[c.lower()]
    if required:
        raise RuntimeError(f"Cannot resolve {label}; columns={list(df.columns)}")
    return None


def read_usgs(path):
    d = pd.read_csv(path)
    t = resolve(d, ["interval_end_utc", "datetime_utc", "timestamp", "time"], "USGS time")
    q = resolve(d, ["q_obs_m3s", "discharge_cms", "discharge_m3s", "q_mean_m3s"], "USGS Q")
    h = resolve(d, ["gage_height_ft", "gage_height", "stage_ft", "gage_height_mean_ft"], "USGS stage", required=False)
    cols = [t, q] + ([h] if h else [])
    x = d[cols].copy()
    x.columns = ["time", "q"] + (["stage_ft"] if h else [])
    x["time"] = pd.to_datetime(x["time"], utc=True, errors="raise")
    x["q"] = pd.to_numeric(x["q"], errors="coerce")
    if "stage_ft" in x:
        x["stage_ft"] = pd.to_numeric(x["stage_ft"], errors="coerce")
    return x.dropna(subset=["time", "q"]).sort_values("time").reset_index(drop=True)


def read_model(path):
    d = pd.read_csv(path)
    t = resolve(d, ["interval_end_utc", "datetime_utc", "timestamp", "time"], "model time")
    q = resolve(d, ["q_sim_m3s", "q_pred_m3s", "predicted_q_m3s", "physics_q_m3s", "ml_q_m3s"], "model Q")
    x = d[[t, q]].copy()
    x.columns = ["time", "q"]
    x["time"] = pd.to_datetime(x["time"], utc=True, errors="raise")
    x["q"] = pd.to_numeric(x["q"], errors="coerce")
    return x.dropna().sort_values("time").reset_index(drop=True)


def metrics(obs_df, pred_df):
    z = obs_df.rename(columns={"q": "o"})[["time", "o"]].merge(
        pred_df.rename(columns={"q": "p"})[["time", "p"]], on="time", how="inner"
    ).dropna()
    o, p = z.o.to_numpy(float), z.p.to_numpy(float)
    if len(z) < 2:
        return {k: np.nan for k in ["nse", "kge", "pbias_percent", "rmse_m3s", "mae_m3s", "peak_error_percent", "peak_timing_hours"]} | {"n": len(z)}
    den = np.sum((o - o.mean()) ** 2)
    nse = 1 - np.sum((p - o) ** 2) / den if den > 0 else np.nan
    r = np.corrcoef(o, p)[0, 1] if np.std(o) > 0 and np.std(p) > 0 else np.nan
    alpha = np.std(p, ddof=1) / np.std(o, ddof=1) if np.std(o, ddof=1) > 0 else np.nan
    beta = p.mean() / o.mean() if o.mean() != 0 else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) if np.all(np.isfinite([r, alpha, beta])) else np.nan
    oi, pi = int(np.nanargmax(o)), int(np.nanargmax(p))
    return {
        "n": len(z), "nse": float(nse), "kge": float(kge),
        "pbias_percent": float(100 * np.sum(p - o) / np.sum(o)) if np.sum(o) else np.nan,
        "rmse_m3s": float(np.sqrt(np.mean((p - o) ** 2))),
        "mae_m3s": float(np.mean(np.abs(p - o))),
        "peak_error_percent": float(100 * (p[pi] - o[oi]) / o[oi]) if o[oi] else np.nan,
        "peak_timing_hours": float((z.iloc[pi].time - z.iloc[oi].time).total_seconds() / 3600),
    }


def load_hq(path):
    d = pd.read_csv(path)
    qcol = resolve(d, ["discharge_cfs"], "discharge_cfs")
    branches = {}
    for name, candidates in [
        ("rising", ["rising_gage_height_ft"]),
        ("falling", ["falling_gage_height_ft"]),
    ]:
        hcol = resolve(d, candidates, f"{name} stage")
        b = d[[qcol, hcol]].apply(pd.to_numeric, errors="coerce").dropna()
        b.columns = ["q_cfs", "stage_ft"]
        b = b.groupby("q_cfs", as_index=False)["stage_ft"].median().sort_values("q_cfs")
        if len(b) < 2:
            raise RuntimeError(f"H-Q {name} branch has fewer than 2 valid points")
        branches[name] = pd.DataFrame({
            "q_m3s": b.q_cfs.to_numpy(float) * CFS_TO_CMS,
            "stage_ft": b.stage_ft.to_numpy(float),
        })
    return branches


def stage_from_q(qv, preferred, branches, observed_pairs):
    if not np.isfinite(qv):
        return np.nan, "missing"
    order = [preferred, "falling" if preferred == "rising" else "rising"]
    for j, name in enumerate(order):
        b = branches[name]
        q = b.q_m3s.to_numpy(float)
        if q.min() <= qv <= q.max():
            h = float(np.interp(qv, q, b.stage_ft.to_numpy(float)))
            return h, f"hq_{name}" if j == 0 else f"hq_fallback_{name}"
    if observed_pairs is not None and not observed_pairs.empty:
        z = observed_pairs.dropna(subset=["q", "stage_ft"])
        if not z.empty:
            idx = (z.q - qv).abs().idxmin()
            return float(z.loc[idx, "stage_ft"]), "nearest_usgs_observed_q"
    # Last-resort nearest retained H-Q point, explicit rather than silent extrapolation.
    candidates = []
    for name, b in branches.items():
        j = int(np.argmin(np.abs(b.q_m3s.to_numpy(float) - qv)))
        candidates.append((abs(float(b.iloc[j].q_m3s) - qv), float(b.iloc[j].stage_ft), name))
    _, h, name = min(candidates, key=lambda x: x[0])
    return h, f"nearest_hq_{name}_endpoint"


def fim_library(root):
    out = {}
    for shp in root.rglob("*.shp"):
        n = shp.stem
        level = float(n) if re.fullmatch(r"\d+", n) else (float(n.replace("_5", ".5")) if re.fullmatch(r"\d+_5", n) else None)
        if level is None:
            continue
        try:
            geom = gpd.read_file(shp).to_crs(32618).geometry.union_all()
            out[level] = (shp, float(geom.area / 1e6))
        except Exception:
            continue
    if not out:
        raise RuntimeError(f"No NOAA FIM polygon levels found under {root}")
    return out


def daily_max(df, d0, d1):
    x = df[(df.time >= d0) & (df.time < d1)]
    if x.empty:
        return None
    return x.loc[x.q.idxmax()]


def main():
    a = parse_args()
    for p in [a.usgs_observations, a.physics_predictions, a.ml_predictions, a.q_to_h, a.noaa_root]:
        if not p.exists():
            raise FileNotFoundError(p)

    usgs = read_usgs(a.usgs_observations)
    physics = read_model(a.physics_predictions)
    ml = read_model(a.ml_predictions)
    branches = load_hq(a.q_to_h)
    lib = fim_library(a.noaa_root)
    levels = np.array(sorted(lib), dtype=float)

    metric_rows = []
    for name, pred in [("Physics V3.5", physics), ("ML V4", ml)]:
        m = metrics(usgs, pred)
        m["source"] = name
        metric_rows.append(m)

    peak_times = {
        "usgs": usgs.loc[usgs.q.idxmax(), "time"],
        "physics": physics.loc[physics.q.idxmax(), "time"],
        "ml": ml.loc[ml.q.idxmax(), "time"],
    }
    observed_pairs = usgs[["q", "stage_ft"]].dropna() if "stage_ft" in usgs else None
    start = pd.Timestamp(a.daily_start, tz="UTC")
    detailed = []
    compact = []

    for i in range(a.daily_days):
        d0, d1 = start + pd.Timedelta(days=i), start + pd.Timedelta(days=i+1)
        urow, prow, mrow = daily_max(usgs, d0, d1), daily_max(physics, d0, d1), daily_max(ml, d0, d1)
        rec = {"date": str(d0.date())}
        compact_rec = {"date": str(d0.date())}
        for pref, row, series_peak in [
            ("usgs", urow, peak_times["usgs"]),
            ("physics", prow, peak_times["physics"]),
            ("ml", mrow, peak_times["ml"]),
        ]:
            if row is None:
                qv, ts, stage, method = np.nan, pd.NaT, np.nan, "missing"
            else:
                qv, ts = float(row.q), row.time
                if pref == "usgs" and "stage_ft" in row.index and np.isfinite(row.stage_ft):
                    stage, method = float(row.stage_ft), "observed_usgs_stage_at_daily_max_q"
                else:
                    preferred = "rising" if ts <= series_peak else "falling"
                    stage, method = stage_from_q(qv, preferred, branches, observed_pairs)
            rec[f"{pref}_q_m3s"] = qv
            rec[f"{pref}_stage_ft"] = stage
            rec[f"{pref}_stage_method"] = method
            compact_rec[f"{pref}_q_m3s"] = qv
            compact_rec[f"{pref}_stage_ft"] = stage

            if np.isfinite(stage):
                wse = stage + a.gage_datum_navd88_ft
                rec[f"{pref}_wse_navd88_ft"] = wse
                if wse < levels.min():
                    lev, area, status = np.nan, np.nan, "below_FIM_range"
                elif wse > levels.max():
                    lev, area, status = np.nan, np.nan, "above_FIM_range"
                else:
                    lev = float(levels[np.argmin(np.abs(levels - wse))])
                    area = float(lib[lev][1])
                    status = "matched"
            else:
                rec[f"{pref}_wse_navd88_ft"] = np.nan
                lev, area, status = np.nan, np.nan, "missing_stage"
            rec[f"{pref}_noaa_fim_level_ft"] = lev
            rec[f"{pref}_noaa_fim_area_km2"] = area
            rec[f"{pref}_noaa_status"] = status

        compact.append(compact_rec)
        detailed.append(rec)

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    targets = [out/"model_metrics.csv", out/"daily_stage_discharge.csv", out/"daily_stage_discharge_noaa_fim.csv", out/"comparison_metadata.json"]
    if not a.overwrite:
        existing = next((p for p in targets if p.exists()), None)
        if existing is not None:
            raise FileExistsError(existing)

    pd.DataFrame(metric_rows)[["source", "n", "nse", "kge", "pbias_percent", "rmse_m3s", "mae_m3s", "peak_error_percent", "peak_timing_hours"]].to_csv(targets[0], index=False)
    compact_cols = ["date", "usgs_q_m3s", "physics_q_m3s", "ml_q_m3s", "usgs_stage_ft", "physics_stage_ft", "ml_stage_ft"]
    pd.DataFrame(compact)[compact_cols].to_csv(targets[1], index=False)
    pd.DataFrame(detailed).to_csv(targets[2], index=False)
    targets[3].write_text(json.dumps({
        "comparison_build": "RETAINED_V3_5_ML_V4_DAILY_MAX_V2",
        "daily_max_policy": "independent native-series daily maxima; USGS retains 15-minute observations",
        "usgs_stage_policy": "observed stage at row of daily maximum discharge",
        "model_stage_policy": "branch-aware H-Q interpolation with explicit fallback",
        "physics_label": "Physics V3.5",
        "ml_label": "ML V4",
        "noaa_fim_levels_ft_navd88": levels.tolist(),
    }, indent=2), encoding="utf-8")
    print("PASS_FLORENCE_MODEL_COMPARISON_COMPLETE")


if __name__ == "__main__":
    main()
