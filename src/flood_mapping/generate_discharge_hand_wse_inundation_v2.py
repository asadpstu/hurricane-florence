"""
Production Hurricane Florence inundation mapper (Physics/HAND V2 replacement).

The legacy workflow applied one scalar HAND threshold to every drainage cell in
an AOI.  That transfers the Goldsboro mainstem stage to tributaries and to
hydraulically disconnected low terrain.  This module instead treats HAND as a
terrain/provenance concept and computes flood water from an absolute water
surface:

    Physics discharge -> limb-aware Q->H stage -> NAVD88 WSE at gauge
    -> longitudinal-only mainstem WSE profile -> 1 m unconditioned DEM
    -> mainstem-connected wet cells -> depth = WSE - DEM

No Copernicus, Landsat, Sentinel-1, or NOAA FIM footprint is used to create the
production flood map.  Those datasets are validation references only.

The script produces one map for each day in the requested event window plus a
summary CSV/JSON.  A uint8 colour table is embedded in the inundation TIFF so
QGIS opens it as transparent dry / blue flood without manual symbology.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import Window

BUILD = "FLORENCE_DISCHARGE_HAND_WSE_INUNDATION_V2"
CFS_TO_CMS = 0.028316846592
FT_TO_M = 0.3048
BYTE_NODATA = 255
FLOAT_NODATA = -9999.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--physics-predictions", type=Path, required=True)
    p.add_argument("--q-to-h", type=Path, required=True)
    p.add_argument(
        "--dem-1m",
        type=Path,
        required=True,
        help="Original unconditioned 1 m lidar DEM in metres NAVD88.",
    )
    p.add_argument("--mainstem-gpkg", type=Path, required=True)
    p.add_argument("--mainstem-layer", default="named_neuse_river")
    p.add_argument("--gage-datum-navd88-ft", type=float, default=41.926)
    p.add_argument("--gauge-lon", type=float, default=-77.9975)
    p.add_argument("--gauge-lat", type=float, default=35.3375)
    p.add_argument("--daily-start", default="2018-09-14")
    p.add_argument("--daily-days", type=int, default=10)
    p.add_argument(
        "--snapshot-time-utc",
        action="append",
        default=[],
        help=(
            "Optional exact validation timestamp; may be repeated. The nearest "
            "Physics V2 row is mapped in addition to daily maxima. Example: "
            "2018-09-18T15:45:05Z for Landsat-7."
        ),
    )
    p.add_argument(
        "--maximum-snapshot-offset-minutes",
        type=float,
        default=60.0,
    )
    p.add_argument(
        "--profile-spacing-m",
        type=float,
        default=10.0,
        help="Spacing of mainstem points used to infer the longitudinal WSE slope.",
    )
    p.add_argument("--mad-sigma", type=float, default=3.0)
    p.add_argument("--minimum-residual-window-m", type=float, default=0.35)
    p.add_argument("--max-fit-iterations", type=int, default=8)
    p.add_argument(
        "--maximum-longitudinal-slope-m-per-km",
        type=float,
        default=5.0,
        help="Safety bound; this is NOT a calibration knob.",
    )
    p.add_argument(
        "--seed-buffer-m",
        type=float,
        default=3.0,
        help="Small buffer around the mainstem used only as hydraulic connectivity seed.",
    )
    p.add_argument("--connectivity", choices=("4", "8"), default="8")
    p.add_argument(
        "--chunk-rows",
        type=int,
        default=1024,
        help="Rows per DEM block while constructing candidate/depth rasters.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def require(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Required files missing:\n- " + "\n- ".join(missing))


def resolve_column(df, candidates, label):
    lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise RuntimeError(
        f"Could not resolve {label}. Tried {candidates}. Available={list(df.columns)}"
    )


def prepare_curve(path, branch):
    df = pd.read_csv(path)
    q_col = resolve_column(df, ["discharge_cfs"], "Q->H discharge")
    h_col = resolve_column(df, [f"{branch}_gage_height_ft"], f"{branch} stage")
    work = df[[q_col, h_col]].copy()
    work[q_col] = pd.to_numeric(work[q_col], errors="coerce")
    work[h_col] = pd.to_numeric(work[h_col], errors="coerce")
    work = work.dropna().groupby(q_col, as_index=False)[h_col].median().sort_values(q_col)
    if len(work) < 2:
        raise RuntimeError(f"Too few finite rows for {branch} Q->H curve.")
    q = work[q_col].to_numpy(float)
    h = work[h_col].to_numpy(float)
    if np.any(np.diff(q) <= 0):
        raise RuntimeError(f"{branch} Q->H discharge must be strictly increasing.")
    return q, h


def stage_from_q(q_m3s, curve, branch):
    q_cfs = float(q_m3s) / CFS_TO_CMS
    q_curve, h_curve = curve
    if q_cfs < q_curve.min() or q_cfs > q_curve.max():
        raise RuntimeError(
            f"{branch} Q={q_m3s:.3f} m3/s ({q_cfs:.1f} cfs) is outside "
            f"Q->H support [{q_curve.min():.1f}, {q_curve.max():.1f}] cfs. "
            "No extrapolation is allowed."
        )
    return q_cfs, float(np.interp(q_cfs, q_curve, h_curve))


def _read_prediction_frame(path):
    df = pd.read_csv(path)
    tcol = resolve_column(df, ["interval_end_utc", "timestamp", "time"], "timestamp")
    qcol = resolve_column(df, ["q_sim_m3s", "q_pred_m3s", "predicted_q_m3s"], "predicted discharge")
    df["time_utc"] = pd.to_datetime(df[tcol], utc=True, errors="raise")
    df["q_m3s"] = pd.to_numeric(df[qcol], errors="coerce")
    df = df[np.isfinite(df["q_m3s"].to_numpy(float))].copy().sort_values("time_utc")
    if df.empty:
        raise RuntimeError("Physics prediction CSV has no finite discharge rows.")
    return df


def _utc_timestamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def read_mapping_states(path, start, days, snapshot_times, max_snapshot_offset_minutes):
    df = _read_prediction_frame(path)
    peak_idx = df["q_m3s"].idxmax()
    peak_time = df.loc[peak_idx, "time_utc"]
    peak_q = float(df.loc[peak_idx, "q_m3s"])

    start_ts = _utc_timestamp(start)
    out = []
    for i in range(days):
        d0 = start_ts + pd.Timedelta(days=i)
        d1 = d0 + pd.Timedelta(days=1)
        part = df[(df["time_utc"] >= d0) & (df["time_utc"] < d1)]
        if part.empty:
            raise RuntimeError(f"No finite predicted discharge for day {i+1}: {d0.date()}")
        idx = part["q_m3s"].idxmax()
        time = df.loc[idx, "time_utc"]
        q = float(df.loc[idx, "q_m3s"])
        branch = "rising" if time <= peak_time else "falling"
        out.append({
            "state_type": "daily_max",
            "day": i + 1,
            "date": str(d0.date()),
            "label": f"Day {i+1:02d} {d0.date()}",
            "stem": f"day_{i+1:02d}_{d0.date()}",
            "requested_time_utc": None,
            "time_utc": time,
            "time_offset_minutes": 0.0,
            "q_m3s": q,
            "branch": branch,
        })

    for value in snapshot_times:
        target = _utc_timestamp(value)
        offsets = (df["time_utc"] - target).abs()
        idx = offsets.idxmin()
        matched = df.loc[idx, "time_utc"]
        offset_min = float(abs((matched - target).total_seconds()) / 60.0)
        if offset_min > max_snapshot_offset_minutes:
            raise RuntimeError(
                f"Nearest Physics V2 state to {target.isoformat()} is {offset_min:.1f} min away; "
                f"limit={max_snapshot_offset_minutes:.1f} min."
            )
        q = float(df.loc[idx, "q_m3s"])
        branch = "rising" if matched <= peak_time else "falling"
        stamp = target.strftime("%Y%m%dT%H%M%SZ")
        out.append({
            "state_type": "snapshot",
            "day": None,
            "date": str(target.date()),
            "label": f"Snapshot {target.isoformat()}",
            "stem": f"snapshot_{stamp}",
            "requested_time_utc": target,
            "time_utc": matched,
            "time_offset_minutes": offset_min,
            "q_m3s": q,
            "branch": branch,
        })

    return out, peak_time, peak_q


def iter_lines(geom):
    if geom is None or geom.is_empty:
        return
    gt = geom.geom_type
    if gt == "LineString":
        yield geom
    elif gt == "MultiLineString":
        yield from geom.geoms
    elif gt == "GeometryCollection":
        for g in geom.geoms:
            yield from iter_lines(g)


def sample_mainstem_profile(gdf, dem_ds, gauge_x, gauge_y, spacing_m):
    points_xy = []
    for geom in gdf.geometry:
        for line in iter_lines(geom):
            if line.length <= 0:
                continue
            distances = np.arange(0.0, line.length + spacing_m * 0.5, spacing_m)
            distances = np.minimum(distances, line.length)
            for d in np.unique(distances):
                p = line.interpolate(float(d))
                points_xy.append((p.x, p.y))
    if len(points_xy) < 20:
        raise RuntimeError(f"Too few mainstem profile points: {len(points_xy)}")

    z = np.asarray([v[0] for v in dem_ds.sample(points_xy)], dtype=np.float64)
    xy = np.asarray(points_xy, dtype=np.float64)
    valid = np.isfinite(z)
    if dem_ds.nodata is not None:
        valid &= z != dem_ds.nodata
    xy = xy[valid]
    z = z[valid]
    if len(z) < 20:
        raise RuntimeError("Too few valid 1 m DEM samples on Neuse mainstem.")

    # First principal component defines a longitudinal coordinate only.  The WSE
    # later varies along this coordinate, never across it, avoiding the old 2-D
    # plane's artificial cross-floodplain tilt.
    centered = xy - np.asarray([gauge_x, gauge_y])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(np.float64)
    axis /= np.linalg.norm(axis)
    s = centered @ axis
    return xy, z, s, axis


def robust_longitudinal_fit(s, z, sigma, min_window, max_iter):
    keep = np.isfinite(s) & np.isfinite(z)
    history = []
    beta = None
    for iteration in range(max_iter):
        A = np.column_stack([s[keep], np.ones(int(keep.sum()))])
        beta, *_ = np.linalg.lstsq(A, z[keep], rcond=None)
        pred = beta[0] * s + beta[1]
        resid = z - pred
        med = float(np.median(resid[keep]))
        mad = float(np.median(np.abs(resid[keep] - med)))
        robust_sigma = 1.4826 * mad
        window = max(float(min_window), float(sigma) * robust_sigma)
        new_keep = np.isfinite(resid) & (np.abs(resid - med) <= window)
        history.append({
            "iteration": iteration + 1,
            "kept": int(new_keep.sum()),
            "residual_median_m": med,
            "residual_mad_m": mad,
            "window_m": window,
        })
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep
        if keep.sum() < 20:
            raise RuntimeError("Longitudinal profile fit rejected too many samples.")
    slope = float(beta[0])
    resid = z - (slope * s + float(beta[1]))
    return {
        "slope_m_per_m": slope,
        "slope_m_per_km": slope * 1000.0,
        "intercept_dem_m": float(beta[1]),
        "fit_count": int(keep.sum()),
        "sample_count": int(len(z)),
        "rmse_m": float(np.sqrt(np.mean(resid[keep] ** 2))),
        "median_abs_residual_m": float(np.median(np.abs(resid[keep]))),
        "history": history,
        "keep": keep,
    }


def build_seed(gdf, shape, transform, buffer_m):
    geoms = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        geoms.append(geom.buffer(buffer_m) if buffer_m > 0 else geom)
    if not geoms:
        raise RuntimeError("Mainstem layer contains no usable geometry.")
    return rasterize(
        [(g, 1) for g in geoms], out_shape=shape, transform=transform,
        fill=0, default_value=1, dtype="uint8", all_touched=True,
    ).astype(bool)


def wse_block(transform, row0, height, width, gauge_x, gauge_y, axis, slope, gauge_wse_m):
    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(row0, row0 + height, dtype=np.float64)
    xs = transform.c + (cols + 0.5) * transform.a + 0.5 * transform.b
    ys = transform.f + (rows + 0.5) * transform.e + 0.5 * transform.d
    # North-up DEMs have b=d=0.  Keep the formula separable and fail clearly
    # for a rotated grid rather than silently producing a wrong WSE.
    if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
        raise RuntimeError("Rotated DEM grids are not supported by the V2 WSE mapper.")
    longitudinal = (
        axis[0] * (xs[None, :] - gauge_x)
        + axis[1] * (ys[:, None] - gauge_y)
    )
    return (gauge_wse_m + slope * longitudinal).astype(np.float32)


def write_flood(path, connected, dem_ds, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    profile = dem_ds.profile.copy()
    profile.update(dtype="uint8", count=1, nodata=BYTE_NODATA, compress="DEFLATE", predictor=2, BIGTIFF="IF_SAFER")
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    with rasterio.open(tmp, "w", **profile) as dst:
        for _, window in dem_ds.block_windows(1):
            dem = dem_ds.read(1, window=window)
            valid = np.isfinite(dem)
            if dem_ds.nodata is not None:
                valid &= dem != dem_ds.nodata
            rs, cs = window.toslices()
            out = np.full(dem.shape, BYTE_NODATA, dtype=np.uint8)
            out[valid] = 0
            out[valid & connected[rs, cs]] = 1
            dst.write(out, 1, window=window)
        dst.set_band_description(1, "0=dry,1=flooded,255=nodata")
        dst.write_colormap(1, {
            0: (255, 255, 255, 0),
            1: (0, 105, 255, 220),
            255: (255, 255, 255, 0),
        })
    os.replace(tmp, path)


def write_depth(path, connected, dem_ds, gauge_wse_m, gauge_x, gauge_y, axis, slope, chunk_rows, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    p = dem_ds.profile.copy()
    p.update(dtype="float32", count=1, nodata=FLOAT_NODATA, compress="DEFLATE", predictor=3, BIGTIFF="IF_SAFER")
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    tmp.unlink(missing_ok=True)
    with rasterio.open(tmp, "w", **p) as dst:
        for row0 in range(0, dem_ds.height, chunk_rows):
            h = min(chunk_rows, dem_ds.height - row0)
            window = Window(0, row0, dem_ds.width, h)
            dem = dem_ds.read(1, window=window).astype(np.float32)
            valid = np.isfinite(dem)
            if dem_ds.nodata is not None:
                valid &= dem != dem_ds.nodata
            wse = wse_block(dem_ds.transform, row0, h, dem_ds.width, gauge_x, gauge_y, axis, slope, gauge_wse_m)
            out = np.full(dem.shape, FLOAT_NODATA, dtype=np.float32)
            out[valid] = 0.0
            rs, cs = window.toslices()
            wet = valid & connected[rs, cs]
            out[wet] = np.maximum(0.0, wse[wet] - dem[wet])
            dst.write(out, 1, window=window)
        dst.set_band_description(1, "connected_flood_depth_m; dry=0")
    os.replace(tmp, path)



def write_wse_surface(path, dem_path, gauge_wse_m, gauge_x, gauge_y, axis, slope, chunk_rows, overwrite):
    """Write the absolute WSE surface on valid DEM cells for provenance/QC."""
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    with rasterio.open(dem_path) as dem_ds:
        p = dem_ds.profile.copy()
        p.update(dtype="float32", count=1, nodata=FLOAT_NODATA,
                 compress="DEFLATE", predictor=3, BIGTIFF="IF_SAFER")
        tmp = path.with_name(path.stem + ".partial" + path.suffix)
        tmp.unlink(missing_ok=True)
        with rasterio.open(tmp, "w", **p) as dst:
            for row0 in range(0, dem_ds.height, chunk_rows):
                h = min(chunk_rows, dem_ds.height - row0)
                window = Window(0, row0, dem_ds.width, h)
                dem = dem_ds.read(1, window=window)
                valid = np.isfinite(dem)
                if dem_ds.nodata is not None:
                    valid &= dem != dem_ds.nodata
                wse = wse_block(
                    dem_ds.transform, row0, h, dem_ds.width, gauge_x, gauge_y,
                    axis, slope, gauge_wse_m
                )
                out = np.full(wse.shape, FLOAT_NODATA, dtype=np.float32)
                out[valid] = wse[valid]
                dst.write(out, 1, window=window)
            dst.set_band_description(1, "absolute_water_surface_elevation_m_NAVD88")
        os.replace(tmp, path)

def main():
    a = parse_args()
    require([a.physics_predictions, a.q_to_h, a.dem_1m, a.mainstem_gpkg])
    if a.daily_days < 1:
        raise ValueError("--daily-days must be >= 1")
    if a.profile_spacing_m <= 0 or a.chunk_rows < 1:
        raise ValueError("Profile spacing and chunk rows must be positive.")

    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("SciPy is required: pip install scipy") from exc

    states, predicted_peak_time, predicted_peak_q = read_mapping_states(
        a.physics_predictions,
        a.daily_start,
        a.daily_days,
        a.snapshot_time_utc,
        a.maximum_snapshot_offset_minutes,
    )
    curves = {b: prepare_curve(a.q_to_h, b) for b in ("rising", "falling")}

    a.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = a.output_dir / "daily_flood_summary.csv"
    meta_path = a.output_dir / "daily_flood_metadata.json"
    profile_csv = a.output_dir / "mainstem_longitudinal_profile_samples.csv"
    for p in (summary_path, meta_path, profile_csv):
        if p.exists() and not a.overwrite:
            raise FileExistsError(p)

    mainstem = gpd.read_file(a.mainstem_gpkg, layer=a.mainstem_layer)
    if mainstem.empty or mainstem.crs is None:
        raise RuntimeError("Mainstem layer is empty or has no CRS.")

    rows_out = []
    with rasterio.open(a.dem_1m) as dem_ds:
        if dem_ds.crs is None:
            raise RuntimeError("1 m DEM has no CRS.")
        res_x = abs(dem_ds.transform.a)
        res_y = abs(dem_ds.transform.e)
        if max(abs(res_x - 1.0), abs(res_y - 1.0)) > 0.05:
            raise RuntimeError(
                f"Expected ~1 m production DEM, got {res_x:.3f} x {res_y:.3f} m. "
                "Use the original unconditioned lidar DEM, not the averaged 5 m grid."
            )
        mainstem = mainstem.to_crs(dem_ds.crs)
        tx = Transformer.from_crs("EPSG:4326", dem_ds.crs, always_xy=True)
        gauge_x, gauge_y = tx.transform(a.gauge_lon, a.gauge_lat)
        xy, z, s, axis = sample_mainstem_profile(
            mainstem, dem_ds, gauge_x, gauge_y, a.profile_spacing_m
        )
        fit = robust_longitudinal_fit(
            s, z, a.mad_sigma, a.minimum_residual_window_m, a.max_fit_iterations
        )
        slope = fit["slope_m_per_m"]
        if abs(fit["slope_m_per_km"]) > a.maximum_longitudinal_slope_m_per_km:
            raise RuntimeError(
                f"Fitted mainstem slope {fit['slope_m_per_km']:.3f} m/km exceeds safety "
                f"bound {a.maximum_longitudinal_slope_m_per_km:.3f} m/km. Inspect DEM/vector alignment."
            )

        pd.DataFrame({
            "x": xy[:, 0], "y": xy[:, 1], "longitudinal_s_m": s,
            "dem_m_navd88": z, "fit_kept": fit["keep"].astype(np.uint8),
        }).to_csv(profile_csv, index=False)

        seed = build_seed(mainstem, (dem_ds.height, dem_ds.width), dem_ds.transform, a.seed_buffer_m)
        structure = ndimage.generate_binary_structure(2, 2 if a.connectivity == "8" else 1)
        pixel_area_m2 = abs(dem_ds.transform.a * dem_ds.transform.e)

        for state in states:
            q_cfs, stage_ft = stage_from_q(state["q_m3s"], curves[state["branch"]], state["branch"])
            gauge_wse_m = (a.gage_datum_navd88_ft + stage_ft) * FT_TO_M
            candidate = np.zeros((dem_ds.height, dem_ds.width), dtype=bool)

            for row0 in range(0, dem_ds.height, a.chunk_rows):
                h = min(a.chunk_rows, dem_ds.height - row0)
                window = Window(0, row0, dem_ds.width, h)
                dem = dem_ds.read(1, window=window).astype(np.float32)
                valid = np.isfinite(dem)
                if dem_ds.nodata is not None:
                    valid &= dem != dem_ds.nodata
                wse = wse_block(dem_ds.transform, row0, h, dem_ds.width, gauge_x, gauge_y, axis, slope, gauge_wse_m)
                candidate[row0:row0+h, :] = valid & (dem <= wse)

            seed_wet = seed & candidate
            if seed_wet.any():
                connected = ndimage.binary_propagation(
                    seed_wet, structure=structure, mask=candidate
                )
                seed_status = "wet_mainstem_seed_present"
            else:
                # A low-flow day can legitimately have no DEM cell below the
                # event WSE after permanent channel water is ignored. Emit a
                # dry flood map instead of aborting the entire 10-day series.
                connected = np.zeros_like(candidate, dtype=bool)
                seed_status = "no_wet_mainstem_seed_dry_map"

            stem = state["stem"]
            flood_path = a.output_dir / f"{stem}_inundation.tif"
            depth_path = a.output_dir / f"{stem}_depth_m.tif"
            write_flood(flood_path, connected, dem_ds, a.overwrite)
            write_depth(depth_path, connected, dem_ds, gauge_wse_m, gauge_x, gauge_y, axis, slope, a.chunk_rows, a.overwrite)

            area_km2 = float(connected.sum() * pixel_area_m2 / 1_000_000.0)
            rows_out.append({
                "state_type": state["state_type"],
                "day": state["day"],
                "date": state["date"],
                "requested_time_utc": (
                    state["requested_time_utc"].isoformat()
                    if state["requested_time_utc"] is not None else None
                ),
                "state_time_utc": state["time_utc"].isoformat(),
                "time_offset_minutes": state["time_offset_minutes"],
                "hydrograph_branch": state["branch"],
                "predicted_q_m3s": state["q_m3s"], "predicted_q_cfs": q_cfs,
                "predicted_gage_height_ft": stage_ft,
                "gauge_wse_navd88_m": gauge_wse_m,
                "mainstem_hand_equivalent_threshold_m": gauge_wse_m - float(fit["intercept_dem_m"]),
                "connectivity_seed_status": seed_status,
                "flood_area_km2": area_km2,
                "inundation_raster": str(flood_path), "depth_raster": str(depth_path),
            })
            print(
                f"{state['label']}: Q={state['q_m3s']:.2f} m3/s, "
                f"stage={stage_ft:.2f} ft ({state['branch']}), area={area_km2:.3f} km2",
                flush=True,
            )

    summary_df = pd.DataFrame(rows_out)
    summary_df.to_csv(summary_path, index=False)

    # Stable peak aliases keep downstream packaging commands independent of the
    # exact crest date in the requested daily window.
    daily_summary = summary_df[summary_df["state_type"] == "daily_max"]
    peak_row = daily_summary.loc[daily_summary["predicted_q_m3s"].idxmax()]
    peak_inundation_path = a.output_dir / "florence_physics_v2_peak_inundation.tif"
    peak_depth_path = a.output_dir / "florence_physics_v2_peak_depth_m.tif"
    peak_wse_path = a.output_dir / "florence_physics_v2_peak_wse_navd88_m.tif"
    for src_name, dst_path in (
        (peak_row["inundation_raster"], peak_inundation_path),
        (peak_row["depth_raster"], peak_depth_path),
    ):
        if dst_path.exists() and not a.overwrite:
            raise FileExistsError(dst_path)
        tmp = dst_path.with_name(dst_path.stem + ".partial" + dst_path.suffix)
        tmp.unlink(missing_ok=True)
        shutil.copy2(src_name, tmp)
        os.replace(tmp, dst_path)

    write_wse_surface(
        peak_wse_path,
        a.dem_1m,
        float(peak_row["gauge_wse_navd88_m"]),
        gauge_x,
        gauge_y,
        axis,
        slope,
        a.chunk_rows,
        a.overwrite,
    )

    metadata = {
        "script_build": BUILD,
        "status": "PASS_DISCHARGE_DRIVEN_MAINSTEM_CONNECTED_DAILY_FLOOD_MAPS",
        "method": {
            "forcing": "Physics V2 predicted discharge",
            "stage_translation": "Florence effective Q->H surrogate; rising before model crest, falling after crest",
            "vertical_reference": "NAVD88 absolute WSE = gage datum + predicted gage height",
            "terrain": "original unconditioned ~1 m lidar DEM",
            "wse_profile": "robust longitudinal-only mainstem slope; no lateral WSE plane tilt",
            "inundation": "DEM <= WSE followed by mainstem-seeded hydraulic connectivity",
            "hand_role": "mainstem-relative height is applied through the longitudinal WSE surface; no AOI-wide generic HAND threshold",
            "hand_equivalence": "candidate DEM<=WSE is algebraically equivalent to [DEM - fitted mainstem reference elevation] <= [gauge WSE - fitted mainstem reference elevation at gauge]",
            "reference_extent_used_to_generate_map": False,
        },
        "predicted_peak": {"time_utc": predicted_peak_time.isoformat(), "q_m3s": predicted_peak_q},
        "gauge": {
            "lon": a.gauge_lon, "lat": a.gauge_lat,
            "gage_datum_navd88_ft": a.gage_datum_navd88_ft,
            "peak_predicted_gage_height_ft": float(peak_row["predicted_gage_height_ft"]),
            "event_wse_navd88_m": float(peak_row["gauge_wse_navd88_m"]),
            "event_wse_navd88_ft": float(peak_row["gauge_wse_navd88_m"]) / FT_TO_M,
        },
        "mainstem_profile_fit": {
            k: v for k, v in fit.items() if k not in {"keep"}
        },
        "longitudinal_axis_xy": axis.tolist(),
        "daily_start": a.daily_start,
        "daily_days": a.daily_days,
        "snapshot_times_utc": a.snapshot_time_utc,
        "generated_states": rows_out,
        "outputs": {
            "summary_csv": str(summary_path),
            "profile_samples_csv": str(profile_csv),
            "peak_inundation": str(peak_inundation_path),
            "peak_depth": str(peak_depth_path),
            "peak_wse_navd88_m": str(peak_wse_path),
            "peak_day": int(peak_row["day"]),
            "peak_date": str(peak_row["date"]),
        },
        "limitations": [
            "This is a terrain-based connected-bathtub / HAND-style approximation, not 2-D shallow-water hydraulics.",
            "No terrain-only method can guarantee 90% spatial accuracy; validate against time-matched imagery on a common valid domain.",
            "Bridges/culverts and unresolved sub-grid drainage can still affect connectivity even at 1 m.",
        ],
    }
    tmp = meta_path.with_name(meta_path.name + ".partial")
    tmp.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, meta_path)
    print(f"Summary : {summary_path}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
