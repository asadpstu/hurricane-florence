"""
MODEL IMPROVEMENT A2.1
Download and validate the official NHDPlus HR vector package for HU4 0302.

Why HU4 0302?
-------------
USGS station 02089000 is in HUC8 03020202, therefore its containing HU4 is 0302.

Data source
-----------
Official USGS The National Map S3 staged-products directory:
StagedProducts/Hydrography/NHDPlusHR/VPU/Current/GDB/

The script:
1. Lists the official S3 prefix.
2. Finds NHDPlus HR HU4 0302 GDB ZIP candidates.
3. Selects the newest dated current product.
4. Downloads it.
5. Extracts the File Geodatabase.
6. Verifies expected layers, especially NHDFlowline and NHDPlusCatchment.
7. Reports feature counts and CRS.
8. Writes an acquisition manifest for A2.2.

No raster download is required for this step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import fiona
import geopandas as gpd
import requests


SCRIPT_BUILD = "MODEL_IMPROVEMENT_A2_1_NHDPLUSHR_HU4_ACQUISITION_V1"

S3_BUCKET = "https://prd-tnm.s3.amazonaws.com"
S3_PREFIX = "StagedProducts/Hydrography/NHDPlusHR/VPU/Current/GDB/"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hu4", default="0302")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input/hydrography/nhdplus_hr_hu4_0302"),
    )
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def list_s3_keys(prefix: str, timeout: int):
    """
    Public S3 ListObjectsV2 pagination.
    """
    keys = []
    continuation = None

    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if continuation:
            params["continuation-token"] = continuation

        r = requests.get(
            S3_BUCKET + "/",
            params=params,
            timeout=timeout,
            headers={"Accept-Encoding": "identity"},
        )
        r.raise_for_status()

        root = ET.fromstring(r.content)

        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        # Some S3-compatible responses omit namespace.
        content_nodes = root.findall("s3:Contents", ns)
        if not content_nodes:
            content_nodes = root.findall("Contents")

        for node in content_nodes:
            key_node = node.find("s3:Key", ns)
            if key_node is None:
                key_node = node.find("Key")
            if key_node is not None and key_node.text:
                keys.append(key_node.text)

        trunc_node = root.find("s3:IsTruncated", ns)
        if trunc_node is None:
            trunc_node = root.find("IsTruncated")
        truncated = (
            trunc_node is not None
            and str(trunc_node.text).lower() == "true"
        )

        if not truncated:
            break

        next_node = root.find("s3:NextContinuationToken", ns)
        if next_node is None:
            next_node = root.find("NextContinuationToken")

        if next_node is None or not next_node.text:
            raise RuntimeError(
                "S3 listing reports truncation but no continuation token."
            )

        continuation = next_node.text

    return keys


def select_hu4_zip(keys, hu4):
    """
    Prefer names such as:
    NHDPLUS_H_0302_HU4_2025xxxx_GDB.zip
    but also accept undated current package names.
    """
    pattern = re.compile(
        rf"/NHDPLUS_H_{re.escape(hu4)}_HU4(?:_(\d{{8}}))?_GDB\.zip$",
        re.IGNORECASE,
    )

    candidates = []

    for key in keys:
        m = pattern.search(key)
        if not m:
            continue

        date_text = m.group(1)
        date_value = int(date_text) if date_text else 0

        candidates.append(
            {
                "key": key,
                "date_text": date_text,
                "date_value": date_value,
            }
        )

    if not candidates:
        # Slightly broader diagnostic fallback.
        broad = [
            k for k in keys
            if f"_{hu4}_HU4" in k.upper()
            and k.lower().endswith(".zip")
        ]
        raise RuntimeError(
            "No NHDPlus HR HU4 GDB ZIP matched expected naming. "
            f"Broad HU4 matches: {broad[:20]}"
        )

    candidates.sort(
        key=lambda x: (x["date_value"], x["key"]),
        reverse=True,
    )

    return candidates[0], candidates


def download(url: str, target: Path, timeout: int):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)

    with requests.get(
        url,
        stream=True,
        timeout=timeout,
        headers={"Accept-Encoding": "identity"},
    ) as r:
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        done = 0

        with partial.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)

                if total:
                    pct = 100.0 * done / total
                    print(
                        f"\rDownload                           : "
                        f"{done / 1024**2:,.1f} / "
                        f"{total / 1024**2:,.1f} MiB "
                        f"({pct:6.2f}%)",
                        end="",
                        flush=True,
                    )

    print()
    partial.replace(target)


def find_gdb(root: Path):
    gdbs = [
        p for p in root.rglob("*.gdb")
        if p.is_dir()
    ]
    if len(gdbs) != 1:
        raise RuntimeError(
            f"Expected exactly one .gdb after extraction; found {gdbs}"
        )
    return gdbs[0]


def inspect_layer(gdb: Path, layer: str):
    gdf = gpd.read_file(gdb, layer=layer)

    result = {
        "layer": layer,
        "feature_count": int(len(gdf)),
        "crs": str(gdf.crs),
        "geometry_types": sorted(
            set(gdf.geom_type.dropna().astype(str))
        ),
        "columns": list(gdf.columns),
        "bounds": (
            [float(x) for x in gdf.total_bounds]
            if len(gdf) else None
        ),
    }

    # Capture some important NHDPlus attributes without assuming every
    # release uses identical capitalization.
    interesting = [
        "NHDPlusID",
        "Hydroseq",
        "DnHydroseq",
        "LevelPathI",
        "TerminalPa",
        "StreamOrde",
        "StreamCalc",
        "TotDASqKm",
        "AreaSqKm",
        "LengthKM",
        "FCode",
        "GNIS_Name",
    ]

    found = {}
    colmap = {c.lower(): c for c in gdf.columns}

    for name in interesting:
        actual = colmap.get(name.lower())
        if actual:
            found[name] = actual

    result["important_columns_found"] = found

    if "NHDPlusID" in found:
        result["nhdplusid_nonnull"] = int(
            gdf[found["NHDPlusID"]].notna().sum()
        )

    return result


def main():
    args = parse_args()

    if not re.fullmatch(r"\d{4}", args.hu4):
        raise SystemExit("--hu4 must be exactly four digits.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"SCRIPT BUILD                       : {SCRIPT_BUILD}")
    print("MODEL IMPROVEMENT A2.1 - NHDPLUS HR HU4 ACQUISITION")
    print("=" * 100)
    print(f"HU4                                 : {args.hu4}")
    print(f"Official S3 prefix                  : {S3_PREFIX}")
    print(f"Output directory                    : {args.output_dir}")
    print()

    print("Discovering current USGS package ...")
    keys = list_s3_keys(S3_PREFIX, args.timeout_seconds)
    selected, candidates = select_hu4_zip(keys, args.hu4)

    print(f"S3 objects scanned                  : {len(keys):,}")
    print(f"HU4 candidate ZIPs                  : {len(candidates)}")
    for i, c in enumerate(candidates[:10], 1):
        print(
            f"  {i:02d}. "
            f"{c['date_text'] or 'undated':8s}  {c['key']}"
        )

    key = selected["key"]
    filename = Path(key).name
    url = S3_BUCKET + "/" + quote(key, safe="/")

    zip_path = args.output_dir / filename

    print()
    print(f"Selected current package            : {filename}")
    print(f"Selected embedded date              : {selected['date_text'] or 'NONE'}")

    if zip_path.exists() and not args.overwrite:
        print(f"ZIP already exists                  : {zip_path}")
    else:
        if zip_path.exists():
            zip_path.unlink()
        download(url, zip_path, args.timeout_seconds)

    digest = sha256(zip_path)

    extract_dir = args.output_dir / "extracted"

    if extract_dir.exists() and args.overwrite:
        shutil.rmtree(extract_dir)

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        print("Extracting ZIP ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    gdb = find_gdb(extract_dir)
    layers = list(fiona.listlayers(gdb))

    print()
    print("FILE GEODATABASE VALIDATION")
    print("-" * 100)
    print(f"Geodatabase                         : {gdb}")
    print(f"Layer count                         : {len(layers)}")

    required_layers = [
        "NHDFlowline",
        "NHDPlusCatchment",
    ]

    missing_layers = [
        layer for layer in required_layers
        if layer not in layers
    ]

    print(f"Required layers                     : {required_layers}")
    print(f"Missing required layers             : {missing_layers if missing_layers else 'NONE'}")

    layer_results = {}

    for layer in required_layers:
        if layer not in layers:
            continue

        info = inspect_layer(gdb, layer)
        layer_results[layer] = info

        print()
        print(f"[{layer}]")
        print(f"Features                            : {info['feature_count']:,}")
        print(f"CRS                                 : {info['crs']}")
        print(f"Geometry                            : {info['geometry_types']}")
        print(
            f"Important attributes                : "
            f"{sorted(info['important_columns_found'].keys())}"
        )

    flow_count = (
        layer_results.get("NHDFlowline", {})
        .get("feature_count", 0)
    )
    catch_count = (
        layer_results.get("NHDPlusCatchment", {})
        .get("feature_count", 0)
    )

    blocking = []

    if missing_layers:
        blocking.append(
            f"Missing required layers: {missing_layers}"
        )

    if flow_count <= 0:
        blocking.append("NHDFlowline is empty.")

    if catch_count <= 0:
        blocking.append("NHDPlusCatchment is empty.")

    safe = len(blocking) == 0

    status = (
        "PASS_A2_1_NHDPLUSHR_HU4_READY"
        if safe
        else "FAIL_A2_1_NHDPLUSHR_HU4_ACQUISITION"
    )

    manifest = {
        "script_build": SCRIPT_BUILD,
        "status": status,
        "safe_for_a2_2": safe,
        "hu4": args.hu4,
        "source": {
            "provider": "U.S. Geological Survey - The National Map",
            "dataset": "NHDPlus High Resolution",
            "s3_prefix": S3_PREFIX,
            "selected_key": key,
            "selected_url": url,
            "embedded_product_date": selected["date_text"],
        },
        "download": {
            "zip_path": str(zip_path),
            "zip_size_bytes": zip_path.stat().st_size,
            "sha256": digest,
            "gdb_path": str(gdb),
        },
        "layers": layer_results,
        "blocking_failures": blocking,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = (
        args.output_dir / "nhdplus_hr_hu4_acquisition_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("READINESS")
    print("-" * 100)
    print(f"NHDFlowline features                : {flow_count:,}")
    print(f"NHDPlusCatchment features           : {catch_count:,}")
    print(f"Blocking failures                   : {len(blocking)}")
    print(f"Safe for A2.2                       : {'YES' if safe else 'NO'}")
    print(f"Status                              : {status}")
    print(f"Manifest                            : {manifest_path}")

    if blocking:
        for item in blocking:
            print(f"  BLOCKING: {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
