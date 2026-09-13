"""
MODEL IMPROVEMENT A3.2/A3.3
Build spatial NHDPlus-subcatchment MRMS GaugeCorr_QPE_01H rainfall from the
Iowa State IEM historical archive and validate it against the existing
basin-average MRMS archive.

Use first as a short smoke test (A3.2), then with the full development window
2015-05-10 -> 2018-01-01 (A3.3).

Scientific invariants
---------------------
* Primary product: GaugeCorr_QPE_01H. If the reference basin archive explicitly marks an hour as RadarOnly_QPE_01H fallback, the same product is used for that hour so spatial and basin forcing remain source-consistent.
* Timestamp T represents hourly accumulation ending at T: (T-1h, T].
* Same exact 1-km MRMS target grid and 37 subcatchment rainfall zones as A2.4.
* No rainfall interpolation.
* Hours absent from the existing reference basin archive remain missing.
* Each successfully rebuilt hour is checked against the original basin mean.
* GRIB files are streamed through a temporary directory and deleted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import reproject, Resampling
import requests


BUILD = "MODEL_IMPROVEMENT_A3_SPATIAL_MRMS_ARCHIVE_V6_RESUME_DTYPE_SAFE"
BASE = "https://mtarchive.geol.iastate.edu"
PRIMARY_PRODUCT = "GaugeCorr_QPE_01H"
FALLBACK_PRODUCT = "RadarOnly_QPE_01H"
MANUAL_DRY_PRODUCT = "DRY_HOUR_INFERRED_ZERO"

_TLS = threading.local()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True,
                   help="UTC inclusive, e.g. 2015-12-20T00:00:00Z")
    p.add_argument("--end", required=True,
                   help="UTC exclusive, e.g. 2015-12-22T00:00:00Z")
    p.add_argument("--reference-basin-rainfall", type=Path, required=True)
    p.add_argument("--zone-raster", type=Path, required=True)
    p.add_argument("--zone-lookup", type=Path, required=True)
    p.add_argument("--zone-metadata", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--timeout-seconds", type=int, default=90)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--resampling", choices=["bilinear", "nearest"], default="bilinear")
    p.add_argument("--max-basin-mean-abs-error-mm", type=float, default=0.01)
    p.add_argument("--min-zone-valid-coverage-percent", type=float, default=99.0)
    p.add_argument("--min-available-hour-success-percent", type=float, default=99.0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--resume", action="store_true",
        help=(
            "Reuse existing OK rows and process only reference-available hours "
            "that are missing/non-OK. Intended for surgical MRMS gap repair."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def get_session():
    if not hasattr(_TLS, "session"):
        s = requests.Session()
        s.headers.update({
            "User-Agent": "neuse-flood-research-spatial-mrms/1.0",
            "Accept-Encoding": "identity",
        })
        _TLS.session = s
    return _TLS.session


def _csv_safe_metadata_value(column, value):
    """Normalize structured source metadata for safe CSV resume updates.

    Existing resume CSVs may be inferred as pandas StringDtype. Assigning a
    tuple (notably raster transform/bounds) into such a column raises a
    TypeError in recent pandas. Structured values are therefore serialized as
    deterministic JSON strings before point assignment.
    """
    if value is None:
        return None
    if column in {"source_transform", "source_bounds"}:
        if isinstance(value, (tuple, list, np.ndarray)):
            return json.dumps([float(x) for x in value], separators=(",", ":"))
        return str(value)
    return value


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


def detect_reference_columns(df):
    time_candidates = [
        "interval_end_utc", "timestamp_utc", "time_utc",
        "valid_time_utc", "time", "timestamp",
    ]
    rain_candidates = [
        "basin_mean_rainfall_mm", "basin_mean_mm",
        "rainfall_mm", "rain_mm", "basin_rainfall_mm",
    ]

    cmap = {str(c).lower(): c for c in df.columns}

    tcol = next((cmap[x.lower()] for x in time_candidates if x.lower() in cmap), None)
    rcol = next((cmap[x.lower()] for x in rain_candidates if x.lower() in cmap), None)

    if tcol is None:
        raise RuntimeError(
            f"Cannot detect reference time column. Columns={list(df.columns)}"
        )
    if rcol is None:
        raise RuntimeError(
            f"Cannot detect reference basin-rainfall column. Columns={list(df.columns)}"
        )

    return tcol, rcol


def url_candidates(t, product):
    day = t.strftime("%Y/%m/%d")
    stamp = t.strftime("%Y%m%d-%H0000")
    stem = f"{product}_00.00_{stamp}.grib2"
    base = f"{BASE}/{day}/mrms/ncep/{product}"
    return [
        (f"{base}/{stem}.gz", True),
        (f"{base}/{stem}", False),
    ]


def download_source(t, out_dir, timeout, retries, product):
    session = get_session()
    last_detail = None

    for url, compressed in url_candidates(t, product):
        suffix = ".grib2.gz" if compressed else ".grib2"
        target = out_dir / f"{t.strftime('%Y%m%d%H')}{suffix}"

        for attempt in range(1, retries + 1):
            try:
                with session.get(url, stream=True, timeout=timeout) as r:
                    if r.status_code == 404:
                        last_detail = f"404 {url}"
                        break
                    r.raise_for_status()
                    with target.open("wb") as f:
                        for chunk in r.iter_content(1024 * 1024):
                            if chunk:
                                f.write(chunk)
                return target, compressed, url, None
            except Exception as exc:
                last_detail = f"{type(exc).__name__}: {exc}"
                target.unlink(missing_ok=True)
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 5))

    return None, None, None, last_detail


def decompress_if_needed(path, compressed):
    if not compressed:
        return path

    out = path.with_suffix("")  # strip .gz
    with gzip.open(path, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return out


def process_hour(task):
    (
        t_iso,
        reference_rain,
        temp_root,
        zone_raster_path,
        zone_ids,
        zone_cell_counts,
        resampling_name,
        timeout,
        retries,
        source_product,
    ) = task

    t = pd.Timestamp(t_iso)
    work = Path(temp_root) / t.strftime("%Y%m%d%H")
    work.mkdir(parents=True, exist_ok=True)

    result = {
        "interval_end_utc": t.isoformat(),
        "reference_basin_mean_rainfall_mm": reference_rain,
        "source_product": source_product,
        "status": "ERROR",
        "source_url": None,
        "source_band_count": None,
        "source_crs": None,
        "source_transform": None,
        "source_width": None,
        "source_height": None,
        "source_bounds": None,
        "reproject_notgeoref_warning_count": 0,
        "reproject_notgeoref_warning": None,
        "min_zone_valid_coverage_percent": np.nan,
        "reconstructed_basin_mean_rainfall_mm": np.nan,
        "basin_mean_difference_mm": np.nan,
        "error": None,
    }

    try:
        if source_product == MANUAL_DRY_PRODUCT:
            result.update({
                "status": "OK",
                "source_url": "manual://documented-dry-hour",
                "min_zone_valid_coverage_percent": 100.0,
                "reconstructed_basin_mean_rainfall_mm": 0.0,
                "basin_mean_difference_mm": 0.0 - float(reference_rain),
                "zone_means": {int(zid): 0.0 for zid in zone_ids},
            })
            return result

        downloaded, compressed, url, err = download_source(
            t, work, timeout, retries, source_product
        )
        if downloaded is None:
            result["status"] = "DOWNLOAD_MISSING_OR_ERROR"
            result["error"] = err
            return result

        result["source_url"] = url
        grib = decompress_if_needed(downloaded, compressed)

        with rasterio.open(zone_raster_path) as zsrc:
            zones = zsrc.read(1)
            dst_crs = zsrc.crs
            dst_transform = zsrc.transform
            dst_height = zsrc.height
            dst_width = zsrc.width

        with rasterio.open(grib) as src:
            result["source_band_count"] = src.count

            # Historical archives can occasionally contain malformed GRIBs.
            # v1 allowed Rasterio/GDAL to fall back to an identity transform,
            # which is unsafe because it can silently produce spatially invalid
            # rainfall. Treat missing georeferencing as an explicit hour failure.
            result["source_crs"] = str(src.crs) if src.crs is not None else None

            if src.crs is None:
                raise RuntimeError(
                    "SOURCE_NOT_GEOREFERENCED: source CRS is missing"
                )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", NotGeoreferencedWarning)
                    src_transform = src.transform
            except NotGeoreferencedWarning as exc:
                raise RuntimeError(
                    "SOURCE_NOT_GEOREFERENCED: "
                    "dataset has no valid geotransform/GCP/RPC information"
                ) from exc

            result["source_transform"] = tuple(float(x) for x in src_transform)[:6]

            if src_transform.is_identity:
                raise RuntimeError(
                    "SOURCE_NOT_GEOREFERENCED: identity transform detected"
                )

            # Read the source band into memory before reprojection.
            # This is intentional: passing rasterio.band(src, 1) can cause
            # GDAL/Rasterio to re-query georeferencing internally during
            # _reproject, which produced NotGeoreferencedWarning for some
            # historical GRIBs even after src.transform/src.crs were inspected.
            source_array = src.read(1).astype("float32")

            if source_array.size == 0:
                raise RuntimeError(
                    "SOURCE_EMPTY: band 1 contains no cells"
                )

            if source_array.shape != (src.height, src.width):
                raise RuntimeError(
                    "SOURCE_SHAPE_MISMATCH: "
                    f"array={source_array.shape}, dataset={(src.height, src.width)}"
                )

            result["source_width"] = int(src.width)
            result["source_height"] = int(src.height)
            result["source_bounds"] = tuple(float(x) for x in src.bounds)

            dst = np.full(
                (dst_height, dst_width),
                -9999.0,
                dtype="float32",
            )

            if resampling_name == "bilinear":
                rs = Resampling.bilinear
            else:
                rs = Resampling.nearest

            src_nodata = src.nodata

            # Some valid historical MRMS GRIBs emit Rasterio's
            # NotGeoreferencedWarning *inside* reproject() even though their
            # dataset CRS, transform, bounds and dimensions are valid and
            # identical to adjacent MRMS hours. We therefore record, rather
            # than reject, that warning after validating the source metadata.
            #
            # Scientific acceptance is determined downstream by:
            #   1) subcatchment valid-cell coverage, and
            #   2) exact reconstruction of the pre-existing basin-mean MRMS
            #      rainfall archive.
            with warnings.catch_warnings(record=True) as caught_reproject:
                warnings.simplefilter("always", NotGeoreferencedWarning)
                reproject(
                    source=source_array,
                    destination=dst,
                    src_transform=src_transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=-9999.0,
                    resampling=rs,
                )

            reproject_ng_warnings = [
                str(w.message)
                for w in caught_reproject
                if issubclass(w.category, NotGeoreferencedWarning)
            ]
            result["reproject_notgeoref_warning_count"] = len(
                reproject_ng_warnings
            )
            result["reproject_notgeoref_warning"] = (
                " | ".join(reproject_ng_warnings)
                if reproject_ng_warnings
                else None
            )

        basin = zones > 0
        valid = basin & np.isfinite(dst) & (dst != -9999.0) & (dst >= 0.0)

        zone_means = {}
        min_cov = 100.0
        basin_sum = 0.0
        basin_valid = 0

        for zid in zone_ids:
            zmask = zones == zid
            zvalid = zmask & valid
            nvalid = int(zvalid.sum())
            ntotal = int(zone_cell_counts[zid])

            cov = 100.0 * nvalid / ntotal if ntotal else 0.0
            min_cov = min(min_cov, cov)

            mean = float(np.mean(dst[zvalid])) if nvalid else np.nan
            zone_means[int(zid)] = mean

            if nvalid:
                basin_sum += float(dst[zvalid].sum(dtype="float64"))
                basin_valid += nvalid

        reconstructed = (
            basin_sum / basin_valid
            if basin_valid else np.nan
        )

        diff = (
            reconstructed - float(reference_rain)
            if np.isfinite(reconstructed) and np.isfinite(reference_rain)
            else np.nan
        )

        result.update({
            "status": "OK",
            "min_zone_valid_coverage_percent": min_cov,
            "reconstructed_basin_mean_rainfall_mm": reconstructed,
            "basin_mean_difference_mm": diff,
            "zone_means": zone_means,
        })

        return result

    except Exception as exc:
        result["status"] = "PROCESSING_ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    a = parse_args()

    start = pd.Timestamp(a.start)
    end = pd.Timestamp(a.end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    if end <= start:
        raise RuntimeError("--end must be after --start.")
    if a.resume and a.overwrite:
        raise RuntimeError("Use either --resume or --overwrite, not both.")

    a.output_dir.mkdir(parents=True, exist_ok=True)
    wide_csv = a.output_dir / "spatial_subcatchment_rainfall_wide.csv"
    missing_csv = a.output_dir / "spatial_subcatchment_missing_hours.csv"
    qc_csv = a.output_dir / "spatial_subcatchment_rainfall_qc.csv"
    metadata_json = a.output_dir / "spatial_subcatchment_rainfall_metadata.json"

    if wide_csv.exists() and not (a.overwrite or a.resume):
        raise RuntimeError(
            f"{wide_csv} exists; use --resume for gap repair or --overwrite for a full rebuild"
        )

    zmeta = json.loads(a.zone_metadata.read_text(encoding="utf-8"))
    if not zmeta.get("safe_for_a3", False):
        raise RuntimeError("A2.4 metadata does not authorize spatial MRMS.")

    lookup = pd.read_csv(a.zone_lookup)
    lookup["zone_id"] = lookup["zone_id"].astype(int)
    required_lookup = {"zone_id", "subcatchment_id", "raster_cell_count"}
    miss_lookup = required_lookup - set(lookup.columns)
    if miss_lookup:
        raise RuntimeError(f"Zone lookup missing {sorted(miss_lookup)}")

    zone_ids = lookup["zone_id"].astype(int).tolist()
    zone_cell_counts = dict(zip(
        lookup["zone_id"].astype(int),
        lookup["raster_cell_count"].astype(int),
    ))
    zone_to_sc = dict(zip(
        lookup["zone_id"].astype(int),
        lookup["subcatchment_id"].astype(str),
    ))
    rain_cols = [f"rain_{zone_to_sc[zid]}_mm" for zid in zone_ids]

    ref = pd.read_csv(a.reference_basin_rainfall)
    tcol, rcol = detect_reference_columns(ref)
    ref["_time"] = pd.to_datetime(ref[tcol], utc=True, errors="coerce")
    ref["_rain"] = pd.to_numeric(ref[rcol], errors="coerce")

    if "source_product" in ref.columns:
        ref["_product"] = ref["source_product"].fillna(PRIMARY_PRODUCT).astype(str)
    else:
        ref["_product"] = PRIMARY_PRODUCT

    allowed_products = {PRIMARY_PRODUCT, FALLBACK_PRODUCT, MANUAL_DRY_PRODUCT}
    bad_products = sorted(set(ref["_product"].dropna().unique()) - allowed_products)
    if bad_products:
        raise RuntimeError(
            f"Unsupported reference source_product values: {bad_products}. "
            f"Allowed={sorted(allowed_products)}"
        )

    ref = ref[(ref["_time"] >= start) & (ref["_time"] < end)].copy()
    expected_hours = pd.date_range(start=start, end=end, freq="1h", inclusive="left")
    frame = pd.DataFrame({"interval_end_utc": expected_hours})
    frame = frame.merge(
        ref[["_time", "_rain", "_product"]].rename(columns={
            "_time": "interval_end_utc",
            "_rain": "reference_basin_mean_rainfall_mm",
            "_product": "reference_source_product",
        }),
        on="interval_end_utc",
        how="left",
        validate="one_to_one",
    )
    frame["reference_source_product"] = frame["reference_source_product"].fillna(PRIMARY_PRODUCT)

    available = frame[frame["reference_basin_mean_rainfall_mm"].notna()].copy()
    reference_missing = frame[frame["reference_basin_mean_rainfall_mm"].isna()].copy()

    existing = None
    completed_ok_times = set()
    if a.resume and wide_csv.exists():
        existing = pd.read_csv(wide_csv, low_memory=False)
        if "interval_end_utc" not in existing.columns or "status" not in existing.columns:
            raise RuntimeError("Existing spatial archive is incompatible with --resume.")
        existing["interval_end_utc"] = pd.to_datetime(
            existing["interval_end_utc"], utc=True, errors="raise"
        )
        completed_ok_times = set(
            existing.loc[existing["status"].astype(str).eq("OK"), "interval_end_utc"]
        )

    pending = available[
        ~available["interval_end_utc"].isin(completed_ok_times)
    ].copy()

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {BUILD}")
    print("MODEL IMPROVEMENT A3 - HISTORICAL SPATIAL MRMS")
    print("=" * 100)
    print(f"Window                             : {start} -> {end} end-exclusive")
    print(f"Requested hours                    : {len(frame):,}")
    print(f"Reference-available hours          : {len(available):,}")
    print(f"Reference-missing hours            : {len(reference_missing):,}")
    print(f"Existing OK rows reused            : {len(completed_ok_times):,}")
    print(f"Pending hours to process           : {len(pending):,}")
    print(f"Subcatchments                      : {len(zone_ids)}")
    print(f"Workers                            : {a.workers}")
    print(f"Primary product                    : {PRIMARY_PRODUCT}")
    print(f"Reference-selected fallback        : {FALLBACK_PRODUCT}")
    print(f"Documented dry-hour product        : {MANUAL_DRY_PRODUCT}")
    print(f"Resampling                         : {a.resampling}")
    print()

    new_rows = []
    task_errors = []
    with tempfile.TemporaryDirectory(prefix="neuse_spatial_mrms_") as temp_root:
        tasks = [
            (
                r.interval_end_utc.isoformat(),
                float(r.reference_basin_mean_rainfall_mm),
                temp_root,
                str(a.zone_raster),
                zone_ids,
                zone_cell_counts,
                a.resampling,
                a.timeout_seconds,
                a.retries,
                str(r.reference_source_product),
            )
            for r in pending.itertuples(index=False)
        ]

        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futures = {ex.submit(process_hour, task): task[0] for task in tasks}
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                res = fut.result()
                new_rows.append(res)
                if res["status"] != "OK":
                    task_errors.append(res)
                    print(
                        "\nFAILED HOUR                        : "
                        f"{res.get('interval_end_utc')} | {res.get('status')} | "
                        f"{res.get('source_product')} | {res.get('source_url')} | {res.get('error')}"
                    )
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"Processed                           : {done:,}/{total:,} (errors={len(task_errors):,})")

    # Start with the current reference frame and reuse prior spatial values where possible.
    out = frame.copy().set_index("interval_end_utc")
    if existing is not None:
        old = existing.set_index("interval_end_utc")
        for c in old.columns:
            if c in {"reference_basin_mean_rainfall_mm", "reference_source_product"}:
                continue
            out[c] = old[c].reindex(out.index)

    base_cols = [
        "status", "source_product", "source_url", "source_band_count",
        "source_crs", "source_transform", "source_width", "source_height",
        "source_bounds", "reproject_notgeoref_warning_count",
        "reproject_notgeoref_warning", "min_zone_valid_coverage_percent",
        "reconstructed_basin_mean_rainfall_mm", "basin_mean_difference_mm", "error",
    ]
    text_metadata_cols = {
        "status", "source_product", "source_url", "source_crs",
        "source_transform", "source_bounds", "reproject_notgeoref_warning",
        "error",
    }
    numeric_metadata_cols = set(base_cols) - text_metadata_cols

    for c in base_cols + rain_cols:
        if c not in out.columns:
            out[c] = None if c in text_metadata_cols else np.nan

    # Resume archives are loaded from CSV and recent pandas versions can infer
    # text columns as StringDtype. Convert provenance columns to ordinary object
    # before scalar assignment so mixed missing/text values are safe.
    for c in text_metadata_cols:
        out[c] = out[c].astype("object")
    for c in numeric_metadata_cols | set(rain_cols):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    for res in new_rows:
        t = pd.Timestamp(res["interval_end_utc"])
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        for c in base_cols:
            if c in res:
                out.at[t, c] = _csv_safe_metadata_value(c, res.get(c))
        if res.get("status") == "OK":
            for zid in zone_ids:
                out.at[t, f"rain_{zone_to_sc[zid]}_mm"] = res["zone_means"].get(zid, np.nan)

    # Older V4 archives did not store source_product explicitly. For reused
    # OK rows, inherit it from the current basin reference so provenance remains
    # complete after a resume repair.
    inherited_mask = (
        out["status"].astype(str).eq("OK")
        & out["source_product"].isna()
    )
    out.loc[inherited_mask, "source_product"] = out.loc[
        inherited_mask, "reference_source_product"
    ]

    # Reference-missing hours remain explicitly missing, with no interpolation.
    ref_missing_idx = frame.loc[
        frame["reference_basin_mean_rainfall_mm"].isna(), "interval_end_utc"
    ]
    out.loc[ref_missing_idx, "status"] = "REFERENCE_MISSING_NO_INTERPOLATION"
    out.loc[ref_missing_idx, rain_cols] = np.nan

    out = out.reset_index().sort_values("interval_end_utc").reset_index(drop=True)

    available_times = set(available["interval_end_utc"])
    ok = out[
        out["interval_end_utc"].isin(available_times)
        & out["status"].astype(str).eq("OK")
    ].copy()
    failed = out[
        out["interval_end_utc"].isin(available_times)
        & ~out["status"].astype(str).eq("OK")
    ].copy()

    reproject_warning_hours = int(
        pd.to_numeric(ok.get("reproject_notgeoref_warning_count", 0), errors="coerce")
        .fillna(0).gt(0).sum()
    ) if len(ok) else 0

    requested = len(frame)
    ref_available_n = len(available)
    success_n = len(ok)
    success_pct = 100.0 * success_n / ref_available_n if ref_available_n else 0.0

    if len(ok):
        min_zone_cov = float(pd.to_numeric(ok["min_zone_valid_coverage_percent"], errors="coerce").min())
        diffs = pd.to_numeric(ok["basin_mean_difference_mm"], errors="coerce").to_numpy(float)
        max_abs_diff = float(np.nanmax(np.abs(diffs)))
        rmse_diff = float(np.sqrt(np.nanmean(diffs ** 2)))
        mae_diff = float(np.nanmean(np.abs(diffs)))
        band_counts = ok["source_band_count"].value_counts(dropna=False).to_dict()
        source_product_counts = ok["source_product"].value_counts(dropna=False).to_dict()
    else:
        min_zone_cov = 0.0
        max_abs_diff = rmse_diff = mae_diff = np.inf
        band_counts = {}
        source_product_counts = {}

    qrows = []
    def qc(sev, check, passed, detail):
        qrows.append({"severity": sev, "check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    qc("BLOCKING", "REFERENCE_WINDOW_PRESENT", requested > 0 and ref_available_n > 0,
       f"requested={requested}, reference_available={ref_available_n}")
    qc("QUALITY", "AVAILABLE_HOUR_SUCCESS_RATE", success_pct >= a.min_available_hour_success_percent,
       f"{success_pct:.6f}% >= {a.min_available_hour_success_percent:.3f}%")
    qc("QUALITY", "MIN_ZONE_VALID_COVERAGE", min_zone_cov >= a.min_zone_valid_coverage_percent,
       f"{min_zone_cov:.6f}% >= {a.min_zone_valid_coverage_percent:.3f}%")
    qc("QUALITY", "REFERENCE_BASIN_MEAN_RECONSTRUCTION", max_abs_diff <= a.max_basin_mean_abs_error_mm,
       f"max_abs_error={max_abs_diff:.9f} mm <= {a.max_basin_mean_abs_error_mm:.6f} mm")

    qcdf = pd.DataFrame(qrows)
    bf = qcdf[(qcdf["severity"] == "BLOCKING") & (qcdf["status"] == "FAIL")]
    qf = qcdf[(qcdf["severity"] == "QUALITY") & (qcdf["status"] == "FAIL")]
    safe = len(bf) == 0 and len(qf) == 0
    is_smoke = requested <= 168
    status = (
        "PASS_A3_2_SPATIAL_MRMS_SMOKE_TEST" if safe and is_smoke else
        "PASS_A3_3_SPATIAL_MRMS_DEVELOPMENT_ARCHIVE_READY" if safe else
        "FAIL_A3_SPATIAL_MRMS_BLOCKING" if len(bf) else
        "FAIL_A3_SPATIAL_MRMS_QUALITY"
    )

    missing_rows = out[out["status"].fillna("UNKNOWN") != "OK"][[
        "interval_end_utc", "reference_basin_mean_rainfall_mm",
        "reference_source_product", "status", "error",
    ]].copy()

    atomic_csv(out, wide_csv)
    atomic_csv(missing_rows, missing_csv)
    atomic_csv(qcdf, qc_csv)

    metadata = {
        "script_build": BUILD,
        "status": status,
        "safe_for_next_step": bool(safe),
        "mode": "SMOKE_TEST" if is_smoke else "DEVELOPMENT_ARCHIVE",
        "execution": {
            "resume": bool(a.resume),
            "existing_ok_rows_reused": int(len(completed_ok_times)),
            "hours_processed_this_run": int(len(new_rows)),
        },
        "window": {
            "start_utc": start.isoformat(),
            "end_utc_exclusive": end.isoformat(),
            "requested_hours": requested,
            "reference_available_hours": ref_available_n,
            "reference_missing_hours": int(len(reference_missing)),
            "successfully_rebuilt_hours": success_n,
            "failed_rebuild_hours": int(len(failed)),
            "available_hour_success_percent": success_pct,
        },
        "source": {
            "archive": BASE,
            "primary_product": PRIMARY_PRODUCT,
            "reference_selected_fallback_product": FALLBACK_PRODUCT,
            "source_product_counts": {str(k): int(v) for k, v in source_product_counts.items()},
            "time_semantics": "(T-1h, T]",
            "rainfall_interpolation": False,
            "resampling": a.resampling,
            "source_band_counts": {str(k): int(v) for k, v in band_counts.items()},
            "reproject_notgeoref_warning_hours": reproject_warning_hours,
        },
        "spatial": {
            "subcatchment_count": len(zone_ids),
            "minimum_zone_valid_coverage_percent": min_zone_cov,
        },
        "reference_reconstruction": {
            "max_abs_difference_mm": max_abs_diff,
            "rmse_mm": rmse_diff,
            "mae_mm": mae_diff,
        },
        "blocking_failure_count": int(len(bf)),
        "quality_failure_count": int(len(qf)),
        "outputs": {
            "wide_csv": str(wide_csv),
            "missing_csv": str(missing_csv),
            "qc_csv": str(qc_csv),
        },
    }
    atomic_json(metadata, metadata_json)

    print()
    print("SPATIAL ARCHIVE QC")
    print("-" * 100)
    print(f"Successfully available             : {success_n:,}/{ref_available_n:,}")
    print(f"Available-hour success             : {success_pct:.6f} %")
    print(f"Minimum zone valid coverage        : {min_zone_cov:.6f} %")
    print(f"Source product counts              : {source_product_counts}")
    print(f"Source band counts                 : {band_counts}")
    print(f"Reproject warning hours            : {reproject_warning_hours:,}")
    print()
    print("REFERENCE BASIN-MEAN RECONSTRUCTION")
    print("-" * 100)
    print(f"Max absolute difference            : {max_abs_diff:.9f} mm")
    print(f"RMSE                               : {rmse_diff:.9f} mm")
    print(f"MAE                                : {mae_diff:.9f} mm")
    print()
    print("READINESS")
    print("-" * 100)
    print(f"Blocking failures                  : {len(bf)}")
    print(f"Quality failures                   : {len(qf)}")
    print(f"Safe for next step                 : {'YES' if safe else 'NO'}")
    print(f"Status                             : {status}")
    print(f"Wide rainfall                      : {wide_csv}")
    print(f"Missing/error hours                : {missing_csv}")
    print(f"Metadata                           : {metadata_json}")

    if not safe:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
