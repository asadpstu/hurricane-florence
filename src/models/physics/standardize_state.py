"""
PHYSICS ERA5-LAND STATE STANDARDIZATION
QC, spatially aggregate, and standardize ERA5-Land dynamic-state forcing.

No model calibration, no interpolation, and no Florence 2018 data are used.

Outputs include:
- hourly ERA5-Land forcing for each of the 37 modeling subcatchments;
- 0-100 cm root-zone volumetric soil-water content;
- deaccumulated hourly net solar/thermal radiation;
- basin-average diagnostics;
- exact ERA5-grid/subcatchment overlap weights;
- event-start soil-state inventory and QC metadata.

ET0 is calculated in the dedicated ET0 script.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

BUILD = "PHYSICS_V2_ERA5_STATE_STANDARDIZATION"

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

STATE_VARS = ["swvl1", "swvl2", "swvl3"]
MET_VARS = ["t2m", "d2m", "sp", "u10", "v10", "ssr", "str"]
ROOT_WEIGHTS = {"swvl1": 0.07, "swvl2": 0.21, "swvl3": 0.72}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--event-blocks", type=Path, required=True)
    p.add_argument("--antecedent-days", type=int, default=45)
    p.add_argument("--subcatchments", type=Path, required=True)
    p.add_argument("--subcatchment-layer", default=None)
    p.add_argument("--subcatchment-id-column", default="subcatchment_id")
    p.add_argument("--expected-subcatchments", type=int, default=37)
    p.add_argument("--area-crs", default="EPSG:32618")
    p.add_argument("--min-subcatchment-grid-coverage-percent", type=float, default=99.0)
    p.add_argument("--min-required-variable-coverage-percent", type=float, default=99.9)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def atomic_csv(df: pd.DataFrame, path: Path, gzip: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + (".partial.gz" if gzip else ".partial"))
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False, compression="gzip" if gzip else None)
    os.replace(tmp, path)


def atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def choose_layer(gpkg: Path, requested: str | None):
    if requested:
        return requested
    import fiona
    layers = fiona.listlayers(gpkg)
    candidates = [x for x in layers if "subcatch" in x.lower() and "routing" not in x.lower()]
    if candidates:
        return candidates[0]
    if len(layers) == 1:
        return layers[0]
    raise RuntimeError(f"Could not choose subcatchment layer. Available: {layers}")


def resolve_coord(ds: xr.Dataset, names):
    available = {str(k).lower(): k for k in ds.variables}
    for n in names:
        if n.lower() in available:
            return available[n.lower()]
    raise RuntimeError(f"Could not resolve coordinate {names}; variables={list(ds.variables)}")


def resolve_var(ds: xr.Dataset, canonical: str):
    lower = {str(k).lower(): k for k in ds.data_vars}
    for name in ALIASES[canonical]:
        if name in ds.data_vars:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    if canonical.startswith("swvl"):
        layer = canonical[-1]
        for name, da in ds.data_vars.items():
            meta = f"{name} {da.attrs.get('long_name','')} {da.attrs.get('standard_name','')}".lower()
            if "soil" in meta and "water" in meta and f"layer {layer}" in meta:
                return name
    raise RuntimeError(f"Could not resolve {canonical}; data_vars={list(ds.data_vars)}")


def open_standardized(path: Path):
    ds0 = xr.open_dataset(path, decode_times=True)
    try:
        t = resolve_coord(ds0, ["valid_time", "time", "datetime"])
        y = resolve_coord(ds0, ["latitude", "lat"])
        x = resolve_coord(ds0, ["longitude", "lon"])
        rename = {}
        if t != "time": rename[t] = "time"
        if y != "latitude": rename[y] = "latitude"
        if x != "longitude": rename[x] = "longitude"
        ds = ds0.rename(rename).squeeze(drop=True)
        if not all(d in ds.dims for d in ["time", "latitude", "longitude"]):
            raise RuntimeError(f"{path}: expected time/latitude/longitude dimensions; dims={dict(ds.dims)}")
        lon = np.asarray(ds.longitude.values, dtype=float)
        if np.nanmax(lon) > 180:
            ds = ds.assign_coords(longitude=((lon + 180.0) % 360.0) - 180.0)
        ds = ds.sortby("latitude").sortby("longitude")
        ds.load()
        return ds
    finally:
        ds0.close()


def cube(ds: xr.Dataset, canonical: str):
    name = resolve_var(ds, canonical)
    da = ds[name]
    extras = [d for d in da.dims if d not in ["time", "latitude", "longitude"]]
    for d in extras:
        if da.sizes[d] != 1:
            raise RuntimeError(f"{name}: unresolved non-singleton dimension {d}={da.sizes[d]}")
    if extras:
        da = da.squeeze(extras, drop=True)
    da = da.transpose("time", "latitude", "longitude")
    return np.asarray(da.values, dtype=float), name, str(da.attrs.get("units", ""))


def coord_edges(v):
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or len(v) < 2 or not np.all(np.diff(v) > 0):
        raise RuntimeError("Grid coordinate must be strictly increasing 1-D array")
    mids = 0.5 * (v[:-1] + v[1:])
    return np.r_[v[0] - 0.5 * (v[1] - v[0]), mids, v[-1] + 0.5 * (v[-1] - v[-2])]


def grid_signature(lat, lon):
    h = hashlib.sha256()
    h.update(np.asarray(lat, dtype=np.float64).tobytes())
    h.update(np.asarray(lon, dtype=np.float64).tobytes())
    return h.hexdigest()


def build_weights(lat, lon, sub, id_col, area_crs):
    lat_edges, lon_edges = coord_edges(lat), coord_edges(lon)
    nlon = len(lon)
    records = []
    for i in range(len(lat)):
        for j in range(len(lon)):
            records.append({
                "cell_index": i * nlon + j,
                "lat_index": i,
                "lon_index": j,
                "latitude": float(lat[i]),
                "longitude": float(lon[j]),
                "geometry": box(lon_edges[j], lat_edges[i], lon_edges[j+1], lat_edges[i+1]),
            })
    cells = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326").to_crs(area_crs)
    subm = sub.to_crs(area_crs).copy()
    subm["subcatchment_area_m2"] = subm.geometry.area
    sindex = cells.sindex
    weight_rows, spatial_rows = [], []
    for _, sc in subm.iterrows():
        sid = str(sc[id_col])
        geom = sc.geometry
        area = float(sc["subcatchment_area_m2"])
        cand = cells.iloc[list(sindex.intersection(geom.bounds))].copy()
        iarea = np.asarray(cand.geometry.intersection(geom).area, dtype=float)
        keep = iarea > 0
        cand, iarea = cand.loc[keep], iarea[keep]
        overlap = float(iarea.sum())
        coverage = 100.0 * overlap / area if area > 0 else np.nan
        spatial_rows.append({
            "subcatchment_id": sid,
            "subcatchment_area_m2": area,
            "era5_overlap_area_m2": overlap,
            "era5_grid_coverage_percent": coverage,
            "overlapping_grid_cell_count": int(len(cand)),
        })
        if overlap <= 0:
            continue
        for (_, cell), a, w in zip(cand.iterrows(), iarea, iarea / overlap):
            weight_rows.append({
                "subcatchment_id": sid,
                "cell_index": int(cell.cell_index),
                "lat_index": int(cell.lat_index),
                "lon_index": int(cell.lon_index),
                "latitude": float(cell.latitude),
                "longitude": float(cell.longitude),
                "overlap_area_m2": float(a),
                "normalized_weight": float(w),
            })
    weights = pd.DataFrame(weight_rows)
    spatial = pd.DataFrame(spatial_rows)
    ids = sorted(sub[id_col].astype(str).unique())
    idrow = {sid: i for i, sid in enumerate(ids)}
    W = np.zeros((len(ids), len(lat) * len(lon)), dtype=float)
    for _, r in weights.iterrows():
        W[idrow[str(r.subcatchment_id)], int(r.cell_index)] = float(r.normalized_weight)
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-9):
        raise RuntimeError("One or more spatial-weight rows do not sum to 1")
    return ids, W, weights, spatial, subm


def aggregate(cube3d, W):
    flat = cube3d.reshape(cube3d.shape[0], -1)
    valid = np.isfinite(flat)
    num = np.where(valid, flat, 0.0) @ W.T
    den = valid.astype(float) @ W.T
    out = np.full_like(num, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out, den


def deaccumulate(times, values):
    """ERA5-Land daily accumulation/reset convention; no interpolation."""
    t = pd.DatetimeIndex(times)
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    one = pd.Timedelta(hours=1)
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        h = t[i].hour
        if h == 1:  # first hourly accumulation after daily reset at 00 UTC
            out[i] = x[i]
            continue
        if i == 0 or t[i] - t[i-1] != one or not np.isfinite(x[i-1]):
            continue
        ph = t[i-1].hour
        if h == 0 and ph == 23:
            out[i] = x[i] - x[i-1]
        elif h >= 2 and ph == h - 1:
            out[i] = x[i] - x[i-1]
    return out


def required_times(blocks, antecedent_days):
    """Build the union of required UTC hourly timestamps without integer epoch conversion.

    Important: pandas 2.x/3.x may retain datetime resolutions such as seconds or
    microseconds. DatetimeIndex.asi8 follows that native resolution, so converting
    those integers back with pd.to_datetime() (which assumes nanoseconds by default)
    can silently shift dates toward 1970. Keep timestamps as timestamps instead.
    """
    pieces = []
    for _, b in blocks.iterrows():
        start = (b.block_start_utc - pd.Timedelta(days=antecedent_days)).floor("h")
        end = (b.block_end_utc + pd.Timedelta(hours=1)).floor("h")
        pieces.append(pd.date_range(start, end, freq="h", inclusive="left"))

    if not pieces:
        return pd.DatetimeIndex([], tz="UTC")

    out = pieces[0]
    for piece in pieces[1:]:
        out = out.union(piece)

    out = out.sort_values()
    if hasattr(out, "as_unit"):
        out = out.as_unit("ns")
    return out


def minmax(s):
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    return (float(x.min()), float(x.max())) if len(x) else (np.nan, np.nan)


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = a.manifest or (a.input_dir / "era5_land_state_manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path)
    required_cols = {"month_id", "group", "status", "output"}
    if not required_cols.issubset(manifest.columns):
        raise RuntimeError(f"Manifest missing columns: {sorted(required_cols-set(manifest.columns))}")
    good_status = {"DOWNLOADED", "EXISTS_SKIPPED"}
    bad = manifest[~manifest.status.isin(good_status)]
    if len(bad):
        raise RuntimeError("Manifest contains unsuccessful entries:\n" + bad[["month_id","group","status"]].to_string(index=False))
    # download_training_state.py writes month_id as YYYY-MM (for example, 2015-08).
    # Normalize both YYYY-MM and legacy YYYYMM forms to YYYY-MM so the
    # standardizer is compatible with either manifest representation.
    raw_month_id = manifest.month_id.astype(str).str.strip()
    month_digits = raw_month_id.str.replace("-", "", regex=False)
    if not month_digits.str.fullmatch(r"\d{6}").all():
        bad_months = raw_month_id[~month_digits.str.fullmatch(r"\d{6}")].unique().tolist()
        raise RuntimeError(f"Invalid month_id value(s) in manifest: {bad_months}")
    years = month_digits.str[:4].astype(int)
    month_numbers = month_digits.str[4:6].astype(int)
    invalid_month = ~month_numbers.between(1, 12)
    if invalid_month.any():
        bad_months = raw_month_id[invalid_month].unique().tolist()
        raise RuntimeError(f"Invalid calendar month in manifest month_id: {bad_months}")
    manifest["month_id"] = [f"{y:04d}-{mo:02d}" for y, mo in zip(years, month_numbers)]
    if manifest.duplicated(["month_id", "group"]).any():
        raise RuntimeError("Duplicate month/group entries in manifest")
    months = sorted(manifest.month_id.unique())
    for m in months:
        groups = set(manifest.loc[manifest.month_id == m, "group"].astype(str))
        if groups != {"STATE", "MET"}:
            raise RuntimeError(f"{m}: expected STATE+MET, found {groups}")

    def resolve_file(row):
        p = Path(str(row.output))
        if p.exists():
            return p
        fallback = a.input_dir / "raw" / str(row.month_id) / p.name
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Missing ERA5 file: {p}; fallback={fallback}")

    blocks = pd.read_csv(a.event_blocks)
    blocks["block_start_utc"] = pd.to_datetime(blocks.block_start_utc, utc=True, errors="raise")
    blocks["block_end_utc"] = pd.to_datetime(blocks.block_end_utc, utc=True, errors="raise")
    req_times = required_times(blocks, a.antecedent_days)

    layer = choose_layer(a.subcatchments, a.subcatchment_layer)
    sub = gpd.read_file(a.subcatchments, layer=layer)
    if a.subcatchment_id_column not in sub.columns or sub.crs is None:
        raise RuntimeError("Subcatchment ID column or CRS missing")
    sub[a.subcatchment_id_column] = sub[a.subcatchment_id_column].astype(str)
    if sub[a.subcatchment_id_column].nunique() != a.expected_subcatchments:
        raise RuntimeError(f"Expected {a.expected_subcatchments} subcatchments")

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {BUILD}")
    print("PHYSICS - ERA5-LAND QC + SPATIAL STANDARDIZATION")
    print("=" * 100)
    print(f"Monthly STATE/MET pairs            : {len(months)}")
    print(f"Subcatchments                      : {a.expected_subcatchments}")
    print(f"Subcatchment layer                 : {layer}")
    print(f"Antecedent QC window               : {a.antecedent_days} days")
    print(f"Required event/state hours         : {len(req_times)}")
    if len(req_times):
        print(f"Required time range                 : {req_times.min()} -> {req_times.max()}")
    print("Temporal interpolation             : NONE")
    print("Florence 2018 used                 : NO\n")

    sig0 = lat0 = lon0 = None
    ids = W = weights = spatial = subm = None
    frames, month_qc, var_inventory = [], [], []

    for m in months:
        sr = manifest[(manifest.month_id == m) & (manifest.group == "STATE")].iloc[0]
        mr = manifest[(manifest.month_id == m) & (manifest.group == "MET")].iloc[0]
        spath, mpath = resolve_file(sr), resolve_file(mr)
        dss, dsm = open_standardized(spath), open_standardized(mpath)
        try:
            ts = pd.to_datetime(dss.time.values, utc=True, errors="raise")
            tm = pd.to_datetime(dsm.time.values, utc=True, errors="raise")
            if ts.duplicated().any() or tm.duplicated().any() or not ts.equals(tm):
                raise RuntimeError(f"{m}: duplicate or mismatched STATE/MET timestamps")
            y_text, mo_text = str(m).split("-", maxsplit=1)
            y, mo = int(y_text), int(mo_text)
            expected_h = calendar.monthrange(y, mo)[1] * 24
            expected_idx = pd.date_range(pd.Timestamp(y, mo, 1, tz="UTC"), periods=expected_h, freq="h")
            if len(ts) != expected_h or not ts.equals(expected_idx):
                raise RuntimeError(f"{m}: incomplete monthly hourly axis ({len(ts)} vs {expected_h})")

            lat = np.asarray(dss.latitude.values, dtype=float)
            lon = np.asarray(dss.longitude.values, dtype=float)
            if not np.array_equal(lat, np.asarray(dsm.latitude.values, dtype=float)) or not np.array_equal(lon, np.asarray(dsm.longitude.values, dtype=float)):
                raise RuntimeError(f"{m}: STATE/MET grids differ")
            sig = grid_signature(lat, lon)
            if sig0 is None:
                lat0, lon0, sig0 = lat.copy(), lon.copy(), sig
                ids, W, weights, spatial, subm = build_weights(lat0, lon0, sub, a.subcatchment_id_column, a.area_crs)
                min_spatial = float(spatial.era5_grid_coverage_percent.min())
                if min_spatial < a.min_subcatchment_grid_coverage_percent:
                    raise RuntimeError(f"Minimum ERA5 spatial coverage only {min_spatial:.4f}%")
            elif sig != sig0:
                raise RuntimeError(f"{m}: ERA5 grid signature changed")

            nt, ns = len(ts), len(ids)
            data = {
                "interval_end_utc": np.repeat(ts.to_numpy(), ns),
                "subcatchment_id": np.tile(np.asarray(ids, dtype=object), nt),
            }
            min_valid = 1.0
            for canonical, ds, group in [(v, dss, "STATE") for v in STATE_VARS] + [(v, dsm, "MET") for v in MET_VARS]:
                arr, source_name, units = cube(ds, canonical)
                agg, valid_fraction = aggregate(arr, W)
                data[canonical] = agg.reshape(-1)
                min_valid = min(min_valid, float(np.nanmin(valid_fraction)))
                var_inventory.append({
                    "month_id": m,
                    "group": group,
                    "canonical_variable": canonical,
                    "source_variable": source_name,
                    "source_units": units,
                    "finite_grid_fraction": float(np.isfinite(arr).mean()),
                    "min_subcatchment_valid_weight_percent": float(100 * np.nanmin(valid_fraction)),
                })
            frames.append(pd.DataFrame(data))
            month_qc.append({
                "month_id": m,
                "expected_hours": expected_h,
                "actual_hours": nt,
                "state_file": str(spath),
                "met_file": str(mpath),
                "grid_lat_count": len(lat0),
                "grid_lon_count": len(lon0),
                "grid_signature_sha256": sig0,
                "minimum_subcatchment_valid_grid_weight_percent": 100 * min_valid,
                "status": "PASS",
            })
            print(f"{m} | hours={nt:4d} | grid={len(lat0)}x{len(lon0)} | min weighted valid={100*min_valid:.3f}% | PASS")
        finally:
            dss.close(); dsm.close()

    hourly = pd.concat(frames, ignore_index=True)
    hourly["interval_end_utc"] = pd.to_datetime(hourly.interval_end_utc, utc=True, errors="raise")
    hourly = hourly.sort_values(["subcatchment_id", "interval_end_utc"]).reset_index(drop=True)
    if hourly.duplicated(["subcatchment_id", "interval_end_utc"]).any():
        raise RuntimeError("Duplicate subcatchment-hours after concatenation")

    # Root-zone water state: official ERA5-Land layer thicknesses total 1 m.
    hourly["root_zone_vwc_m3m3"] = sum(ROOT_WEIGHTS[v] * hourly[v] for v in STATE_VARS)
    hourly["root_zone_total_water_equivalent_mm"] = 1000.0 * hourly.root_zone_vwc_m3m3

    hourly["temperature_2m_c"] = hourly.t2m - 273.15
    hourly["dewpoint_2m_c"] = hourly.d2m - 273.15
    hourly["surface_pressure_kpa"] = hourly.sp / 1000.0
    hourly["wind_speed_10m_m_s"] = np.hypot(hourly.u10, hourly.v10)

    hourly["net_solar_radiation_hourly_j_m2"] = np.nan
    hourly["net_thermal_radiation_hourly_j_m2"] = np.nan
    for sid, idx in hourly.groupby("subcatchment_id", sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        times = hourly.loc[loc, "interval_end_utc"]
        hourly.loc[loc, "net_solar_radiation_hourly_j_m2"] = deaccumulate(times, hourly.loc[loc, "ssr"].to_numpy(float))
        hourly.loc[loc, "net_thermal_radiation_hourly_j_m2"] = deaccumulate(times, hourly.loc[loc, "str"].to_numpy(float))
    hourly["net_radiation_hourly_j_m2"] = hourly.net_solar_radiation_hourly_j_m2 + hourly.net_thermal_radiation_hourly_j_m2
    hourly["net_radiation_hourly_mj_m2"] = hourly.net_radiation_hourly_j_m2 / 1e6

    # Normalize both sides to nanosecond-resolution UTC timestamps and compare
    # timestamps directly. Do not compare raw integer epochs because pandas may
    # retain different datetime units (s/us/ns) depending on the NetCDF decoder.
    hourly["interval_end_utc"] = hourly["interval_end_utc"].astype("datetime64[ns, UTC]")
    if hasattr(req_times, "as_unit"):
        req_times = req_times.as_unit("ns")

    hourly["required_for_event_development"] = hourly["interval_end_utc"].isin(req_times)
    available_times = pd.DatetimeIndex(hourly.interval_end_utc.drop_duplicates())
    if hasattr(available_times, "as_unit"):
        available_times = available_times.as_unit("ns")
    missing_req_times = req_times.difference(available_times)
    required = hourly[hourly.required_for_event_development].copy()
    expected_req_rows = len(req_times) * a.expected_subcatchments
    if len(required) != expected_req_rows:
        raise RuntimeError(f"Required row mismatch: {len(required)} vs {expected_req_rows}; missing times={len(missing_req_times)}")

    qc_rows, quality_failures, warnings = [], 0, 0
    required_vars = [
        "swvl1", "swvl2", "swvl3", "root_zone_vwc_m3m3", "temperature_2m_c", "dewpoint_2m_c",
        "surface_pressure_kpa", "u10", "v10", "wind_speed_10m_m_s",
        "net_solar_radiation_hourly_j_m2", "net_thermal_radiation_hourly_j_m2", "net_radiation_hourly_mj_m2",
    ]
    for col in required_vars:
        x = pd.to_numeric(required[col], errors="coerce")
        coverage = float(100 * np.isfinite(x).mean())
        mn, mx = minmax(x)
        status = "PASS" if coverage >= a.min_required_variable_coverage_percent else "FAIL"
        quality_failures += (status == "FAIL")
        qc_rows.append({"check": f"required_coverage_{col}", "value": coverage, "threshold": f">={a.min_required_variable_coverage_percent}", "status": status, "detail": f"min={mn}, max={mx}"})

    range_checks = [
        ("root_zone_vwc_m3m3", 0.0, 1.0),
        ("temperature_2m_c", -60.0, 60.0),
        ("dewpoint_2m_c", -80.0, 50.0),
        ("surface_pressure_kpa", 70.0, 110.0),
        ("wind_speed_10m_m_s", 0.0, 60.0),
        ("net_solar_radiation_hourly_j_m2", -1e6, 6e6),
        ("net_thermal_radiation_hourly_j_m2", -6e6, 3e6),
        ("net_radiation_hourly_mj_m2", -6.0, 6.0),
    ]
    for col, lo, hi in range_checks:
        x = pd.to_numeric(required[col], errors="coerce")
        bad_count = int((np.isfinite(x) & ((x < lo) | (x > hi))).sum())
        status = "PASS" if bad_count == 0 else "FAIL"
        quality_failures += (status == "FAIL")
        qc_rows.append({"check": f"physical_range_{col}", "value": bad_count, "threshold": "0", "status": status, "detail": f"allowed=[{lo}, {hi}]"})

    dew_bad = int(((required.dewpoint_2m_c - required.temperature_2m_c) > 1.0).sum())
    dew_status = "PASS" if dew_bad == 0 else "WARN"
    warnings += (dew_status == "WARN")
    qc_rows.append({"check": "dewpoint_not_above_temperature", "value": dew_bad, "threshold": "0 rows with Td > T + 1C", "status": dew_status, "detail": "Meteorological consistency diagnostic"})

    min_spatial = float(spatial.era5_grid_coverage_percent.min())
    spatial_status = "PASS" if min_spatial >= a.min_subcatchment_grid_coverage_percent else "FAIL"
    quality_failures += (spatial_status == "FAIL")
    qc_rows.append({"check": "minimum_subcatchment_spatial_coverage", "value": min_spatial, "threshold": f">={a.min_subcatchment_grid_coverage_percent}%", "status": spatial_status, "detail": "Exact polygon-overlap coverage"})

    missing_time_status = "PASS" if len(missing_req_times) == 0 else "FAIL"
    quality_failures += (missing_time_status == "FAIL")
    qc_rows.append({"check": "missing_required_hourly_timestamps", "value": len(missing_req_times), "threshold": "0", "status": missing_time_status, "detail": "No temporal interpolation allowed"})

    rad_missing_all = int(hourly.net_radiation_hourly_mj_m2.isna().sum())
    rad_missing_req = int(required.net_radiation_hourly_mj_m2.isna().sum())
    if rad_missing_req > 0:
        rad_status = "FAIL"; quality_failures += 1
    elif rad_missing_all > 0:
        rad_status = "WARN"; warnings += 1
    else:
        rad_status = "PASS"
    qc_rows.append({"check": "radiation_deaccumulation_boundary_gaps", "value": rad_missing_all, "threshold": "0 within required windows", "status": rad_status, "detail": f"required_missing={rad_missing_req}; disconnected-month first 00 UTC can be undefined"})
    qc = pd.DataFrame(qc_rows)

    # Basin-area weighted diagnostics.
    area_col = "subcatchment_area_m2"
    area_map = dict(zip(subm[a.subcatchment_id_column].astype(str), subm[area_col].astype(float)))
    basin_vars = [
        "swvl1", "swvl2", "swvl3", "root_zone_vwc_m3m3", "root_zone_total_water_equivalent_mm",
        "temperature_2m_c", "dewpoint_2m_c", "surface_pressure_kpa", "u10", "v10", "wind_speed_10m_m_s",
        "net_solar_radiation_hourly_j_m2", "net_thermal_radiation_hourly_j_m2", "net_radiation_hourly_mj_m2",
    ]
    basin_rows = []
    for ts, g in hourly.groupby("interval_end_utc", sort=True):
        areas = np.asarray([area_map[str(s)] for s in g.subcatchment_id], dtype=float)
        row = {"interval_end_utc": ts, "required_for_event_development": bool(g.required_for_event_development.iloc[0])}
        for col in basin_vars:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            valid = np.isfinite(vals)
            row[col] = float(np.average(vals[valid], weights=areas[valid])) if valid.any() else np.nan
        basin_rows.append(row)
    basin = pd.DataFrame(basin_rows)

    # Exact event-start ERA5 soil state for each subcatchment.
    event_rows = []
    for _, b in blocks.iterrows():
        g = hourly[hourly.interval_end_utc == b.block_start_utc]
        if len(g) != a.expected_subcatchments:
            raise RuntimeError(f"No exact event-start ERA5 state for {b.event_block_id} at {b.block_start_utc}")
        for _, r in g.iterrows():
            event_rows.append({
                "event_block_id": str(b.event_block_id),
                "phase": str(getattr(b, "phase", "")),
                "block_start_utc": b.block_start_utc,
                "subcatchment_id": str(r.subcatchment_id),
                "swvl1_m3m3": float(r.swvl1),
                "swvl2_m3m3": float(r.swvl2),
                "swvl3_m3m3": float(r.swvl3),
                "root_zone_vwc_m3m3": float(r.root_zone_vwc_m3m3),
                "root_zone_total_water_equivalent_mm": float(r.root_zone_total_water_equivalent_mm),
            })
    event_state = pd.DataFrame(event_rows)

    outputs = {
        "subcatchment_hourly": a.output_dir / "era5_land_subcatchment_hourly.csv.gz",
        "basin_hourly": a.output_dir / "era5_land_basin_hourly.csv",
        "grid_weights": a.output_dir / "era5_land_subcatchment_grid_weights.csv",
        "spatial_qc": a.output_dir / "era5_land_spatial_qc.csv",
        "monthly_qc": a.output_dir / "era5_land_monthly_qc.csv",
        "variable_inventory": a.output_dir / "era5_land_variable_inventory.csv",
        "event_initial_states": a.output_dir / "era5_land_event_initial_states.csv",
        "qc": a.output_dir / "era5_land_state_qc.csv",
        "metadata": a.output_dir / "era5_land_state_metadata.json",
    }
    if outputs["subcatchment_hourly"].exists() and not a.overwrite:
        raise FileExistsError(f"{outputs['subcatchment_hourly']} exists; use --overwrite")

    atomic_csv(hourly, outputs["subcatchment_hourly"], gzip=True)
    atomic_csv(basin, outputs["basin_hourly"])
    atomic_csv(weights, outputs["grid_weights"])
    atomic_csv(spatial, outputs["spatial_qc"])
    atomic_csv(pd.DataFrame(month_qc), outputs["monthly_qc"])
    atomic_csv(pd.DataFrame(var_inventory), outputs["variable_inventory"])
    atomic_csv(event_state, outputs["event_initial_states"])
    atomic_csv(qc, outputs["qc"])

    blocking_failures = 0
    status = "PASS_PHYSICS_V2_ERA5_STATE_STANDARDIZED" if quality_failures == 0 else "FAIL_PHYSICS_V2_ERA5_STATE_QC"
    metadata = {
        "script_build": BUILD,
        "status": status,
        "florence_used": False,
        "temporal_interpolation": False,
        "monthly_pairs": len(months),
        "subcatchments": len(ids),
        "grid": {"latitude_count": len(lat0), "longitude_count": len(lon0), "signature_sha256": sig0, "area_crs": a.area_crs, "minimum_subcatchment_coverage_percent": min_spatial},
        "soil_layers": {
            "layer_1_cm": [0, 7], "layer_2_cm": [7, 28], "layer_3_cm": [28, 100],
            "root_zone_formula": "0.07*swvl1 + 0.21*swvl2 + 0.72*swvl3",
            "root_zone_total_depth_mm": 1000,
            "note": "Root-zone water-equivalent is total water in the 0-100 cm column; it is not plant-available water and is not the conceptual model soil capacity."
        },
        "radiation": {"deaccumulated_to_hourly_intervals": True, "no_interpolation": True},
        "et0": {"calculated_in_this_step": False, "next": "calculate_era5_et0.py"},
        "required_event_development_hours": len(req_times),
        "required_rows": len(required),
        "blocking_failures": blocking_failures,
        "quality_failures": quality_failures,
        "warnings": warnings,
        "outputs": {k: str(v) for k, v in outputs.items() if k != "metadata"},
    }
    atomic_json(metadata, outputs["metadata"])

    root_min, root_max = minmax(required.root_zone_vwc_m3m3)
    temp_min, temp_max = minmax(required.temperature_2m_c)
    rad_min, rad_max = minmax(required.net_radiation_hourly_mj_m2)
    print("\nSTANDARDIZED DEVELOPMENT FORCING")
    print("-" * 100)
    print(f"Subcatchment-hours                 : {len(hourly):,}")
    print(f"Required development rows          : {len(required):,}")
    print(f"Minimum ERA5 spatial coverage      : {min_spatial:.6f}%")
    print(f"Root-zone VWC range                : {root_min:.4f} to {root_max:.4f} m3/m3")
    print(f"2 m temperature range              : {temp_min:.2f} to {temp_max:.2f} C")
    print(f"Hourly net radiation range         : {rad_min:.4f} to {rad_max:.4f} MJ/m2")
    print(f"Radiation missing in required rows : {rad_missing_req}")
    print("\nREADINESS")
    print("-" * 100)
    print(f"Blocking failures                  : {blocking_failures}")
    print(f"Quality failures                   : {quality_failures}")
    print(f"Warnings                           : {warnings}")
    print(f"Safe for ET0/state forcing    : {'YES' if status.startswith('PASS_') else 'NO'}")
    print(f"Status                             : {status}")
    print(f"Hourly forcing                     : {outputs['subcatchment_hourly']}")
    print(f"Event initial states               : {outputs['event_initial_states']}")
    print(f"QC                                 : {outputs['qc']}")
    print(f"Metadata                           : {outputs['metadata']}")
    if not status.startswith("PASS_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
