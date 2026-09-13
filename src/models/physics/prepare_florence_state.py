"""
PHYSICS - FLORENCE ERA5 PREPARATION

Purpose
-------
Prepare ONLY the ERA5-Land state/ET forcing required to run the already-frozen
frozen semi-distributed Physics model on Hurricane Florence 2018.

Scientific safeguards
---------------------
* Downloads Florence-era ERA5-Land forcing, but DOES NOT calibrate anything.
* Reuses the development-period (2015-2016) per-subcatchment q05/q95 scaling
  already saved by the development-period state standardization.
* Does NOT estimate any scaling from 2018.
* Does NOT use observed discharge.
* Does NOT interpolate missing values.
* Uses the same root-zone definition and hourly FAO-56 ET0 formulation as the
  development workflow.
* Uses exact polygon/grid overlap weighting for the same 37 subcatchments.

Default outputs
---------------
output/model_improvement/physics_v2_florence_era5/
    raw/<YYYYMM>/era5_land_state_<YYYYMM>.nc
    raw/<YYYYMM>/era5_land_met_<YYYYMM>.nc
    raw/<YYYYMM>/era5_land_swvl4_<YYYYMM>.nc
    florence_dynamic_state_et0_subcatchment_hourly.csv.gz
    florence_deep_state_subcatchment_hourly.csv.gz
    florence_era5_spatial_weights.csv
    florence_era5_qc.csv
    florence_era5_metadata.json
"""

from __future__ import annotations

import argparse
import calendar
import importlib.util
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


BUILD = "PHYSICS_V2_FLORENCE_ERA5_PREPARATION"
DATASET = "reanalysis-era5-land"

STATE_VARIABLES = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
]
MET_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
]
DEEP_VARIABLES = ["volumetric_soil_water_layer_4"]
HOURS = [f"{h:02d}:00" for h in range(24)]
ROOT_WEIGHTS = {"swvl1": 0.07, "swvl2": 0.21, "swvl3": 0.72}

ALIASES = {
    "swvl1": ["swvl1", "volumetric_soil_water_layer_1"],
    "swvl2": ["swvl2", "volumetric_soil_water_layer_2"],
    "swvl3": ["swvl3", "volumetric_soil_water_layer_3"],
    "swvl4": ["swvl4", "volumetric_soil_water_layer_4"],
    "t2m": ["t2m", "2m_temperature"],
    "d2m": ["d2m", "2m_dewpoint_temperature"],
    "sp": ["sp", "surface_pressure"],
    "u10": ["u10", "10m_u_component_of_wind"],
    "v10": ["v10", "10m_v_component_of_wind"],
    "ssr": ["ssr", "surface_net_solar_radiation"],
    "str": ["str", "surface_net_thermal_radiation"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subcatchments", type=Path, required=True)
    p.add_argument("--subcatchment-layer", default=None)
    p.add_argument("--subcatchment-id-column", default="subcatchment_id")
    p.add_argument("--soil-scaling", type=Path, required=True)
    p.add_argument("--deep-scaling", type=Path, required=True)
    p.add_argument("--start", default="2018-09-10T00:00:00Z")
    p.add_argument("--end", default="2018-09-26T00:00:00Z",
                   help="Final evaluation end timestamp. Required ERA5 output includes this endpoint.")
    p.add_argument("--bbox-padding-deg", type=float, default=0.15)
    p.add_argument("--expected-subcatchments", type=int, default=37)
    p.add_argument("--area-crs", default="EPSG:32618")
    p.add_argument("--min-subcatchment-grid-coverage-percent", type=float, default=99.0)
    p.add_argument("--min-required-variable-coverage-percent", type=float, default=99.9)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--redownload", action="store_true",
                   help="Redownload cached raw ERA5 files. Normally leave off.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite standardized outputs; cached raw downloads are reused.")
    return p.parse_args()


def atomic_csv(df, path, gzip=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".partial.gz" if gzip else ".partial"
    tmp = path.with_name(path.name + suffix)
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False, compression="gzip" if gzip else None)
    os.replace(tmp, path)


def atomic_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def parse_utc(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    if t.minute or t.second or t.microsecond:
        raise ValueError(f"Timestamp must be on an exact hour: {value}")
    return t


def normalize_sc_id(value):
    s = str(value).strip()
    if s.upper().startswith("SC"):
        try:
            return f"SC{int(s[2:]):03d}"
        except Exception:
            return s
    try:
        return f"SC{int(float(s)):03d}"
    except Exception:
        return s


def load_local_module(filename, module_name):
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required existing project module not found: {path}\n"
            "Place this script in src/model_improvement/ before running it."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def month_starts(start, end):
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while cur <= last:
        yield cur
        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1, tz="UTC")
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1, tz="UTC")


def request_payload(year, month, variables, area):
    ndays = calendar.monthrange(year, month)[1]
    return {
        "variable": variables,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "time": HOURS,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def download_one(client, payload, target, redownload):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not redownload:
        return "EXISTS_REUSED"

    target.unlink(missing_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)

    result = client.retrieve(DATASET, payload)
    result.download(str(partial))
    if not partial.exists() or partial.stat().st_size <= 0:
        raise RuntimeError(f"ERA5 download produced an empty file: {target}")
    os.replace(partial, target)
    return "DOWNLOADED"


def resolve_var(ds, canonical):
    lower = {str(k).lower(): k for k in ds.data_vars}
    for candidate in ALIASES[canonical]:
        if candidate in ds.data_vars:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if canonical.startswith("swvl"):
        layer = canonical[-1]
        for name, da in ds.data_vars.items():
            meta = f"{name} {da.attrs.get('long_name','')} {da.attrs.get('standard_name','')}".lower()
            if "soil" in meta and "water" in meta and f"layer {layer}" in meta:
                return name

    raise RuntimeError(f"Could not resolve {canonical}; data_vars={list(ds.data_vars)}")


def open_standardized(path):
    ds0 = xr.open_dataset(path, decode_times=True)
    try:
        names = {str(k).lower(): k for k in ds0.variables}

        def coord(candidates):
            for c in candidates:
                if c.lower() in names:
                    return names[c.lower()]
            raise RuntimeError(f"{path}: missing coordinate from {candidates}")

        t = coord(["valid_time", "time", "datetime"])
        y = coord(["latitude", "lat"])
        x = coord(["longitude", "lon"])

        rename = {}
        if t != "time":
            rename[t] = "time"
        if y != "latitude":
            rename[y] = "latitude"
        if x != "longitude":
            rename[x] = "longitude"

        ds = ds0.rename(rename).squeeze(drop=True)
        lon = np.asarray(ds.longitude.values, dtype=float)
        if np.nanmax(lon) > 180:
            ds = ds.assign_coords(longitude=((lon + 180.0) % 360.0) - 180.0)

        ds = ds.sortby("latitude").sortby("longitude")
        ds.load()
        return ds
    finally:
        ds0.close()


def cube(ds, canonical):
    name = resolve_var(ds, canonical)
    da = ds[name]
    extras = [d for d in da.dims if d not in ["time", "latitude", "longitude"]]
    for d in extras:
        if da.sizes[d] != 1:
            raise RuntimeError(f"{name}: unresolved dimension {d}={da.sizes[d]}")
    if extras:
        da = da.squeeze(extras, drop=True)
    da = da.transpose("time", "latitude", "longitude")
    return np.asarray(da.values, dtype=float)


def concat_month_cubes(files, canonicals):
    all_times = []
    values = {c: [] for c in canonicals}
    ref_lat = None
    ref_lon = None

    for path in files:
        ds = open_standardized(path)
        try:
            times = pd.to_datetime(ds.time.values, utc=True)
            lat = np.asarray(ds.latitude.values, dtype=float)
            lon = np.asarray(ds.longitude.values, dtype=float)

            if ref_lat is None:
                ref_lat, ref_lon = lat, lon
            elif not (np.allclose(lat, ref_lat) and np.allclose(lon, ref_lon)):
                raise RuntimeError(f"ERA5 grid changed between months: {path}")

            all_times.append(times)
            for c in canonicals:
                values[c].append(cube(ds, c))
        finally:
            ds.close()

    times = pd.to_datetime(
        np.concatenate([x.to_numpy() for x in all_times]),
        utc=True,
        errors="raise",
    )
    order = np.argsort(times.asi8)
    times = times[order]

    out = {}
    for c in canonicals:
        arr = np.concatenate(values[c], axis=0)[order]
        out[c] = arr

    if times.duplicated().any():
        raise RuntimeError("Duplicate ERA5 timestamps after monthly concatenation.")

    return times, ref_lat, ref_lon, out


def load_scaling(path, expected, kind):
    df = pd.read_csv(path)
    df["subcatchment_id"] = df["subcatchment_id"].map(normalize_sc_id)

    if df["subcatchment_id"].nunique() != expected:
        raise RuntimeError(
            f"{kind} scaling expected {expected} subcatchments; "
            f"found {df['subcatchment_id'].nunique()}."
        )
    if df["subcatchment_id"].duplicated().any():
        raise RuntimeError(f"{kind} scaling contains duplicate subcatchment IDs.")

    if "status" in df.columns and not df["status"].astype(str).str.startswith("PASS").all():
        bad = df.loc[~df["status"].astype(str).str.startswith("PASS"), "subcatchment_id"].tolist()
        raise RuntimeError(f"{kind} scaling contains failed rows: {bad}")

    return df.set_index("subcatchment_id")


def main():
    a = parse_args()
    start = parse_utc(a.start)
    end = parse_utc(a.end)

    if end <= start:
        raise ValueError("--end must be after --start.")
    if start.year != 2018 or end.year != 2018:
        raise RuntimeError("This final-test preparer is intentionally restricted to Florence-era 2018.")

    dynmod = load_local_module(
        "standardize_state.py",
        "physics_v2_dynamic_standardizer",
    )
    etmod = load_local_module(
        "calculate_et0.py",
        "physics_v2_et0_deriver",
    )

    if not a.subcatchments.exists():
        raise FileNotFoundError(a.subcatchments)
    if not a.soil_scaling.exists():
        raise FileNotFoundError(a.soil_scaling)
    if not a.deep_scaling.exists():
        raise FileNotFoundError(a.deep_scaling)

    layer = a.subcatchment_layer or dynmod.choose_layer(a.subcatchments, None)
    sub = gpd.read_file(a.subcatchments, layer=layer)
    if sub.crs is None:
        raise RuntimeError("Subcatchment CRS is missing.")

    id_col = a.subcatchment_id_column
    if id_col not in sub.columns:
        candidates = [c for c in ["subcatchment_id", "modeling_subcatchment_id", "sc_id"] if c in sub.columns]
        if not candidates:
            raise RuntimeError(f"Could not resolve subcatchment ID column. Columns={list(sub.columns)}")
        id_col = candidates[0]

    sub[id_col] = sub[id_col].map(normalize_sc_id)
    if sub[id_col].nunique() != a.expected_subcatchments:
        raise RuntimeError(
            f"Expected {a.expected_subcatchments} subcatchments; "
            f"found {sub[id_col].nunique()}."
        )

    sub4326 = sub.to_crs(4326)
    minx, miny, maxx, maxy = sub4326.total_bounds
    area = [
        round(min(90.0, float(maxy) + a.bbox_padding_deg), 4),
        round(max(-180.0, float(minx) - a.bbox_padding_deg), 4),
        round(max(-90.0, float(miny) - a.bbox_padding_deg), 4),
        round(min(180.0, float(maxx) + a.bbox_padding_deg), 4),
    ]

    months = list(month_starts(start, end))
    a.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {BUILD}")
    print("PHYSICS - FLORENCE ERA5 FINAL-TEST PREPARATION")
    print("=" * 100)
    print(f"Evaluation start                   : {start}")
    print(f"Evaluation end                     : {end}")
    print(f"Subcatchments                      : {a.expected_subcatchments}")
    print(f"ERA5-Land request area             : N={area[0]}, W={area[1]}, S={area[2]}, E={area[3]}")
    print("Observed Q used                    : NO")
    print("2018 used for scaling              : NO")
    print("Temporal interpolation             : NONE")
    print("Development scaling reused         : YES")
    print()

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("cdsapi is required. Install with: python -m pip install cdsapi") from exc

    client = cdsapi.Client()
    state_files, met_files, deep_files = [], [], []
    manifest_rows = []

    for m in months:
        month_id = f"{m:%Y%m}"
        raw = a.output_dir / "raw" / month_id
        state_path = raw / f"era5_land_state_{month_id}.nc"
        met_path = raw / f"era5_land_met_{month_id}.nc"
        deep_path = raw / f"era5_land_swvl4_{month_id}.nc"

        requests = [
            ("STATE", STATE_VARIABLES, state_path),
            ("MET", MET_VARIABLES, met_path),
            ("DEEP", DEEP_VARIABLES, deep_path),
        ]

        for group, variables, target in requests:
            payload = request_payload(m.year, m.month, variables, area)
            try:
                status = download_one(client, payload, target, a.redownload)
                error = None
            except Exception as exc:
                status = "ERROR"
                error = f"{type(exc).__name__}: {exc}"

            manifest_rows.append({
                "month_id": month_id,
                "group": group,
                "status": status,
                "path": str(target),
                "size_bytes": target.stat().st_size if target.exists() else 0,
                "variables": ",".join(variables),
                "error": error,
            })
            print(f"{month_id} | {group:5s} | {status:13s} | {target}")

            if status == "ERROR":
                raise RuntimeError(f"ERA5 download failed for {month_id} {group}: {error}")

        state_files.append(state_path)
        met_files.append(met_path)
        deep_files.append(deep_path)

    # Load and verify common ERA5 grid.
    state_times, lat, lon, state_cube = concat_month_cubes(
        state_files, ["swvl1", "swvl2", "swvl3"]
    )
    met_times, lat_m, lon_m, met_cube = concat_month_cubes(
        met_files, ["t2m", "d2m", "sp", "u10", "v10", "ssr", "str"]
    )
    deep_times, lat_d, lon_d, deep_cube = concat_month_cubes(
        deep_files, ["swvl4"]
    )

    if not state_times.equals(met_times) or not state_times.equals(deep_times):
        raise RuntimeError("STATE/MET/DEEP ERA5 timestamps do not match exactly.")
    if not (
        np.allclose(lat, lat_m) and np.allclose(lon, lon_m)
        and np.allclose(lat, lat_d) and np.allclose(lon, lon_d)
    ):
        raise RuntimeError("STATE/MET/DEEP ERA5 grids do not match.")

    ids, W, weights, spatial, _ = dynmod.build_weights(
        lat, lon, sub, id_col, a.area_crs
    )
    ids = [normalize_sc_id(x) for x in ids]

    if len(ids) != a.expected_subcatchments:
        raise RuntimeError(f"Weight matrix contains {len(ids)} subcatchments.")

    min_spatial = float(spatial["era5_grid_coverage_percent"].min())
    if min_spatial < a.min_subcatchment_grid_coverage_percent:
        raise RuntimeError(
            f"Minimum ERA5 polygon/grid coverage {min_spatial:.6f}% is below "
            f"{a.min_subcatchment_grid_coverage_percent}%."
        )

    # Aggregate each variable exactly with polygon/grid overlap weights.
    agg = {}
    den = {}
    for canonical, arr in {**state_cube, **met_cube, **deep_cube}.items():
        values, valid_fraction = dynmod.aggregate(arr, W)
        agg[canonical] = values
        den[canonical] = valid_fraction

    rows = []
    for ti, t in enumerate(state_times):
        for si, sid in enumerate(ids):
            rows.append({
                "interval_end_utc": t,
                "subcatchment_id": sid,
                "swvl1": agg["swvl1"][ti, si],
                "swvl2": agg["swvl2"][ti, si],
                "swvl3": agg["swvl3"][ti, si],
                "swvl4_m3m3": agg["swvl4"][ti, si],
                "t2m": agg["t2m"][ti, si],
                "d2m": agg["d2m"][ti, si],
                "sp": agg["sp"][ti, si],
                "u10": agg["u10"][ti, si],
                "v10": agg["v10"][ti, si],
                "ssr": agg["ssr"][ti, si],
                "str": agg["str"][ti, si],
            })

    h = pd.DataFrame(rows).sort_values(["subcatchment_id", "interval_end_utc"]).reset_index(drop=True)
    h["root_zone_vwc_m3m3"] = (
        ROOT_WEIGHTS["swvl1"] * h["swvl1"]
        + ROOT_WEIGHTS["swvl2"] * h["swvl2"]
        + ROOT_WEIGHTS["swvl3"] * h["swvl3"]
    )
    h["temperature_2m_c"] = h["t2m"] - 273.15
    h["dewpoint_2m_c"] = h["d2m"] - 273.15
    h["surface_pressure_kpa"] = h["sp"] / 1000.0
    h["wind_speed_10m_m_s"] = np.hypot(h["u10"], h["v10"])

    h["net_solar_radiation_hourly_j_m2"] = np.nan
    h["net_thermal_radiation_hourly_j_m2"] = np.nan

    # Same deaccumulation implementation used by the development standardizer.
    for sid, idx in h.groupby("subcatchment_id", sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        times = h.loc[loc, "interval_end_utc"]
        h.loc[loc, "net_solar_radiation_hourly_j_m2"] = dynmod.deaccumulate(
            times, h.loc[loc, "ssr"].to_numpy(float)
        )
        h.loc[loc, "net_thermal_radiation_hourly_j_m2"] = dynmod.deaccumulate(
            times, h.loc[loc, "str"].to_numpy(float)
        )

    h["net_radiation_hourly_j_m2"] = (
        h["net_solar_radiation_hourly_j_m2"]
        + h["net_thermal_radiation_hourly_j_m2"]
    )
    h["net_radiation_hourly_mj_m2"] = h["net_radiation_hourly_j_m2"] / 1e6

    et = etmod.hourly_fao56_et0(
        h["temperature_2m_c"].to_numpy(float),
        h["dewpoint_2m_c"].to_numpy(float),
        h["surface_pressure_kpa"].to_numpy(float),
        h["wind_speed_10m_m_s"].to_numpy(float),
        h["net_radiation_hourly_mj_m2"].to_numpy(float),
        h["net_solar_radiation_hourly_j_m2"].to_numpy(float),
    )
    for col, values in et.items():
        h[col] = values

    soil_scale = load_scaling(a.soil_scaling, a.expected_subcatchments, "root-zone")
    deep_scale = load_scaling(a.deep_scaling, a.expected_subcatchments, "deep-layer")

    root_low_col = "root_zone_vwc_lower_m3m3"
    root_high_col = "root_zone_vwc_upper_m3m3"
    deep_low_col = "swvl4_lower_m3m3"
    deep_high_col = "swvl4_upper_m3m3"

    for col, frame, label in [
        (root_low_col, soil_scale, "root-zone"),
        (root_high_col, soil_scale, "root-zone"),
        (deep_low_col, deep_scale, "deep"),
        (deep_high_col, deep_scale, "deep"),
    ]:
        if col not in frame.columns:
            raise RuntimeError(f"{label} scaling file missing {col}.")

    h["root_zone_relative_wetness"] = np.nan
    h["deep_relative_wetness"] = np.nan

    for sid, idx in h.groupby("subcatchment_id", sort=False).groups.items():
        if sid not in soil_scale.index or sid not in deep_scale.index:
            raise RuntimeError(f"Scaling missing subcatchment {sid}.")

        root_low = float(soil_scale.loc[sid, root_low_col])
        root_high = float(soil_scale.loc[sid, root_high_col])
        deep_low = float(deep_scale.loc[sid, deep_low_col])
        deep_high = float(deep_scale.loc[sid, deep_high_col])

        if not (np.isfinite(root_low) and np.isfinite(root_high) and root_high > root_low):
            raise RuntimeError(f"{sid}: invalid frozen root-zone scaling.")
        if not (np.isfinite(deep_low) and np.isfinite(deep_high) and deep_high > deep_low):
            raise RuntimeError(f"{sid}: invalid frozen deep-state scaling.")

        loc = np.asarray(list(idx), dtype=int)
        h.loc[loc, "root_zone_relative_wetness"] = np.clip(
            (h.loc[loc, "root_zone_vwc_m3m3"].to_numpy(float) - root_low)
            / (root_high - root_low),
            0.0, 1.0,
        )
        h.loc[loc, "deep_relative_wetness"] = np.clip(
            (h.loc[loc, "swvl4_m3m3"].to_numpy(float) - deep_low)
            / (deep_high - deep_low),
            0.0, 1.0,
        )

    # Keep state at evaluation start AND all interval-end states through evaluation end.
    required_times = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    required = h[h["interval_end_utc"].isin(required_times)].copy()

    expected_rows = len(required_times) * a.expected_subcatchments
    if len(required) != expected_rows:
        raise RuntimeError(
            f"Required Florence ERA5 rows={len(required):,}; expected={expected_rows:,}."
        )
    if required.duplicated(["interval_end_utc", "subcatchment_id"]).any():
        raise RuntimeError("Duplicate required Florence subcatchment-hours.")

    required_time_count = required["interval_end_utc"].nunique()
    if required_time_count != len(required_times):
        raise RuntimeError(
            f"Required Florence ERA5 timestamps={required_time_count}; expected={len(required_times)}."
        )

    critical_cols = [
        "root_zone_vwc_m3m3",
        "root_zone_relative_wetness",
        "et0_mm_h",
        "swvl4_m3m3",
        "deep_relative_wetness",
    ]
    coverage = {
        col: float(100.0 * required[col].notna().mean())
        for col in critical_cols
    }

    bad_cov = {
        col: pct
        for col, pct in coverage.items()
        if pct < a.min_required_variable_coverage_percent
    }
    if bad_cov:
        raise RuntimeError(f"Required Florence ERA5 coverage failed: {bad_cov}")

    start_rows = required[required["interval_end_utc"] == start]
    if len(start_rows) != a.expected_subcatchments:
        raise RuntimeError(
            f"Event-start state rows={len(start_rows)}; expected={a.expected_subcatchments}."
        )

    dynamic_cols = [
        "interval_end_utc",
        "subcatchment_id",
        "root_zone_vwc_m3m3",
        "root_zone_relative_wetness",
        "temperature_2m_c",
        "dewpoint_2m_c",
        "surface_pressure_kpa",
        "wind_speed_10m_m_s",
        "net_solar_radiation_hourly_j_m2",
        "net_thermal_radiation_hourly_j_m2",
        "net_radiation_hourly_mj_m2",
        "et0_raw_mm_h",
        "et0_mm_h",
    ]
    deep_cols = [
        "interval_end_utc",
        "subcatchment_id",
        "swvl4_m3m3",
        "deep_relative_wetness",
    ]

    dynamic_path = a.output_dir / "florence_dynamic_state_et0_subcatchment_hourly.csv.gz"
    deep_path = a.output_dir / "florence_deep_state_subcatchment_hourly.csv.gz"
    weights_path = a.output_dir / "florence_era5_spatial_weights.csv"
    manifest_path = a.output_dir / "florence_era5_download_manifest.csv"
    qc_path = a.output_dir / "florence_era5_qc.csv"
    metadata_path = a.output_dir / "florence_era5_metadata.json"

    for p in [dynamic_path, deep_path, weights_path, manifest_path, qc_path, metadata_path]:
        if p.exists() and not a.overwrite:
            raise FileExistsError(f"{p} exists. Use --overwrite.")

    weights_out = weights.copy()
    if "subcatchment_id" in weights_out.columns:
        weights_out["subcatchment_id"] = weights_out["subcatchment_id"].map(normalize_sc_id)

    qc_rows = [
        {"check": "subcatchment_count", "value": a.expected_subcatchments, "status": "PASS"},
        {"check": "required_hour_count_inclusive_state_start", "value": len(required_times), "status": "PASS"},
        {"check": "required_subcatchment_hours", "value": len(required), "status": "PASS"},
        {"check": "minimum_polygon_grid_coverage_percent", "value": min_spatial, "status": "PASS"},
        {"check": "event_start_state_rows", "value": len(start_rows), "status": "PASS"},
        {"check": "temporal_interpolation_used", "value": 0, "status": "PASS"},
        {"check": "observed_q_used", "value": 0, "status": "PASS"},
        {"check": "florence_used_for_scaling", "value": 0, "status": "PASS"},
    ]
    for col, pct in coverage.items():
        qc_rows.append({
            "check": f"{col}_coverage_percent",
            "value": pct,
            "status": "PASS" if pct >= a.min_required_variable_coverage_percent else "FAIL",
        })

    atomic_csv(required[dynamic_cols], dynamic_path, gzip=True)
    atomic_csv(required[deep_cols], deep_path, gzip=True)
    atomic_csv(weights_out, weights_path)
    atomic_csv(pd.DataFrame(manifest_rows), manifest_path)
    atomic_csv(pd.DataFrame(qc_rows), qc_path)

    metadata = {
        "script_build": BUILD,
        "status": "PASS_PHYSICS_V2_FLORENCE_ERA5_FINAL_TEST_FORCING_READY",
        "dataset": DATASET,
        "source": "Copernicus Climate Data Store / ERA5-Land",
        "evaluation_start_utc": start,
        "evaluation_end_utc": end,
        "required_timestamp_semantics": (
            "ERA5 state is retained at evaluation_start for model initialization; "
            "rainfall/discharge final-test intervals are evaluated separately as (start, end]."
        ),
        "subcatchments": a.expected_subcatchments,
        "subcatchment_layer": layer,
        "request_area_nwse": area,
        "development_scaling_reused": True,
        "soil_scaling_source": str(a.soil_scaling),
        "deep_scaling_source": str(a.deep_scaling),
        "florence_used_for_scaling": False,
        "observed_discharge_used": False,
        "temporal_interpolation": False,
        "root_zone_vwc_equation": "0.07*swvl1 + 0.21*swvl2 + 0.72*swvl3",
        "root_zone_relative_wetness": (
            "clip((2018 theta - frozen development q05)/(frozen development q95-q05), 0, 1)"
        ),
        "deep_relative_wetness": (
            "clip((2018 swvl4 - frozen development q05)/(frozen development q95-q05), 0, 1)"
        ),
        "et0_method": "the same hourly FAO-56 Penman-Monteith helper used during development",
        "minimum_polygon_grid_coverage_percent": min_spatial,
        "coverage_percent": coverage,
        "outputs": {
            "dynamic_state_et0": str(dynamic_path),
            "deep_state": str(deep_path),
            "weights": str(weights_path),
            "download_manifest": str(manifest_path),
            "qc": str(qc_path),
        },
    }
    atomic_json(metadata, metadata_path)

    print()
    print("READINESS")
    print("-" * 100)
    print(f"Required timestamps (start..end)   : {len(required_times)}")
    print(f"Required subcatchment-hours        : {len(required):,}")
    print(f"Minimum spatial coverage           : {min_spatial:.6f}%")
    print(f"Root wetness coverage              : {coverage['root_zone_relative_wetness']:.6f}%")
    print(f"ET0 coverage                       : {coverage['et0_mm_h']:.6f}%")
    print(f"Deep wetness coverage              : {coverage['deep_relative_wetness']:.6f}%")
    print("Observed Q used                    : NO")
    print("2018 scaling fitted                : NO")
    print("Temporal interpolation             : NONE")
    print("Status                             : PASS_PHYSICS_V2_FLORENCE_ERA5_FINAL_TEST_FORCING_READY")
    print(f"Dynamic forcing                    : {dynamic_path}")
    print(f"Deep state                         : {deep_path}")
    print(f"Metadata                           : {metadata_path}")


if __name__ == "__main__":
    main()
