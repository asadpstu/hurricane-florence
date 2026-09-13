"""
Step 1B - Download currently published USGS rating-curve files via WDFN STAC.

Default site:
    USGS-02089000 - Neuse River near Goldsboro, NC

Downloads every rating artifact currently exposed for the site:
    base : base stage-discharge rating
    exsa : expanded rating including current SHIFT values
    corr : expanded-stage correction table

Important scientific boundary
-----------------------------
These are the rating files CURRENTLY published by USGS at retrieval time.
They are NOT automatically assumed to be the rating/shift in effect during
Hurricane Florence in September 2018.

The script therefore saves:
- raw STAC search JSON
- raw RDB rating files
- parsed CSV copies where possible
- a manifest
- metadata with event_specific_rating_confirmed = false

A later step should compare the 2018 paired observed stage/discharge records
against the downloaded rating and investigate event-era rating/shift metadata.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


STAC_SEARCH_URL = "https://api.waterdata.usgs.gov/stac/v0/search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download USGS rating files from the Water Data STAC API."
    )
    parser.add_argument("--site", default="02089000")
    parser.add_argument(
        "--event-date",
        default="2018-09-18",
        help=(
            "Reference event date recorded in metadata only. "
            "It is NOT used to assert historical rating applicability."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input/observations/usgs/02089000/ratings"),
    )
    parser.add_argument("--http-timeout-seconds", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    tmp.unlink(missing_ok=True)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(json.dumps(payload, indent=2), path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    tmp.unlink(missing_ok=True)
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = list(output_dir.glob("*"))
        if existing:
            raise FileExistsError(
                f"{output_dir} is not empty. Use --overwrite."
            )


def request_json(
    url: str,
    *,
    params: dict[str, Any] | None,
    timeout: int,
) -> tuple[dict[str, Any], str]:
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={
            "User-Agent": (
                "neuse-flood-validation/1.0 "
                "(academic USGS rating acquisition)"
            )
        },
    )
    response.raise_for_status()
    return response.json(), response.url


def request_text(url: str, timeout: int) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "neuse-flood-validation/1.0 "
                "(academic USGS rating acquisition)"
            )
        },
    )
    response.raise_for_status()
    return response.text


def parse_rdb(text: str) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Parse a standard USGS tab-delimited RDB file.

    Returns:
        dataframe or None
        comment/header lines
    """
    lines = text.splitlines()
    comments = [line for line in lines if line.startswith("#")]
    data_lines = [line for line in lines if line and not line.startswith("#")]

    if len(data_lines) < 2:
        return None, comments

    # USGS RDB normally has:
    # line 1: column names
    # line 2: field format/type declarations
    # remaining lines: data
    header = data_lines[0]
    data = data_lines[2:]

    if not data:
        return None, comments

    body = header + "\n" + "\n".join(data)

    try:
        frame = pd.read_csv(
            StringIO(body),
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
    except Exception:
        return None, comments

    # Convert known numeric rating columns when possible.
    for column in ["INDEP", "DEP", "SHIFT", "CORR", "CORRINDEP"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame, comments


def main() -> None:
    args = parse_args()

    site_number = args.site.replace("USGS-", "")
    monitoring_location_id = f"USGS-{site_number}"
    output_dir = args.output_dir

    prepare_output_dir(output_dir, args.overwrite)

    print("=" * 96)
    print("STEP 1B - USGS RATING-CURVE ACQUISITION")
    print("=" * 96)
    print(f"Monitoring location               : {monitoring_location_id}")
    print(f"Reference event date              : {args.event_date}")
    print()

    params = {
        "collection": "ratings",
        "filter": f"monitoring_location_id='{monitoring_location_id}'",
        "limit": 100,
    }

    stac, search_url = request_json(
        STAC_SEARCH_URL,
        params=params,
        timeout=args.http_timeout_seconds,
    )

    stac_path = output_dir / "usgs_rating_stac_search.json"
    atomic_write_json(stac, stac_path)

    features = stac.get("features", [])
    if not features:
        raise RuntimeError(
            f"No USGS rating files were returned for {monitoring_location_id}."
        )

    manifest_rows: list[dict[str, Any]] = []
    downloaded_types: list[str] = []

    for feature in features:
        properties = feature.get("properties", {})
        file_type = str(properties.get("file_type", "unknown")).lower()
        feature_id = str(feature.get("id", "unknown"))

        asset = feature.get("assets", {}).get("data", {})
        href = asset.get("href")
        if not href:
            manifest_rows.append(
                {
                    "feature_id": feature_id,
                    "file_type": file_type,
                    "stac_datetime": properties.get("datetime"),
                    "download_status": "NO_DATA_ASSET",
                    "raw_rdb": None,
                    "parsed_csv": None,
                    "asset_url": None,
                }
            )
            continue

        text = request_text(href, args.http_timeout_seconds)

        raw_path = output_dir / f"usgs_02089000_rating_{file_type}.rdb"
        atomic_write_text(text, raw_path)

        frame, comments = parse_rdb(text)

        parsed_path: Path | None = None
        if frame is not None:
            parsed_path = output_dir / f"usgs_02089000_rating_{file_type}.csv"
            atomic_write_csv(frame, parsed_path)

        header_path = output_dir / f"usgs_02089000_rating_{file_type}_header.txt"
        atomic_write_text("\n".join(comments) + "\n", header_path)

        downloaded_types.append(file_type)
        manifest_rows.append(
            {
                "feature_id": feature_id,
                "file_type": file_type,
                "stac_datetime": properties.get("datetime"),
                "download_status": "DOWNLOADED",
                "raw_rdb": str(raw_path),
                "parsed_csv": str(parsed_path) if parsed_path else None,
                "header_txt": str(header_path),
                "asset_url": href,
                "row_count": 0 if frame is None else int(len(frame)),
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("file_type")
    manifest_path = output_dir / "usgs_rating_manifest.csv"
    atomic_write_csv(manifest, manifest_path)

    metadata = {
        "status": "PASS_CURRENT_USGS_RATING_FILES_ACQUIRED",
        "step": "STEP_1B",
        "created_utc": utc_now(),
        "monitoring_location_id": monitoring_location_id,
        "site_number": site_number,
        "reference_event_date": args.event_date,
        "source": "USGS Water Data for the Nation STAC ratings collection",
        "stac_search_url": search_url,
        "downloaded_file_types": sorted(set(downloaded_types)),
        "rating_interpretation": {
            "base": "Base stage-discharge rating.",
            "exsa": (
                "Expanded rating; normally includes INDEP, SHIFT, DEP and "
                "therefore reflects currently published shift information."
            ),
            "corr": "Expanded-stage correction table when available.",
        },
        "event_specific_rating_confirmed": False,
        "event_specific_rating_reason": (
            "The STAC files are the currently published rating artifacts. "
            "Their applicability to September 2018 must be verified separately."
        ),
        "safe_use_now": (
            "Use these files for inspection and comparison with the paired "
            "2018 observed stage/discharge series. Do not yet label them the "
            "Florence-2018 rating curve."
        ),
        "output_paths": {
            "stac_search": str(stac_path),
            "manifest": str(manifest_path),
            "output_dir": str(output_dir),
        },
    }

    metadata_path = output_dir / "usgs_rating_metadata.json"
    atomic_write_json(metadata, metadata_path)

    print(f"Rating artifacts returned          : {len(features):,}")
    print(
        f"File types downloaded              : "
        f"{', '.join(sorted(set(downloaded_types)))}"
    )
    print(f"Manifest                           : {manifest_path}")
    print(f"Metadata                           : {metadata_path}")
    print()
    print("Event-specific 2018 rating         : NOT YET CONFIRMED")
    print(
        "Next scientific check              : compare 2018 paired H/Q against "
        "the downloaded base/EXSA relation and inspect rating headers."
    )
    print()
    print("Status                             : PASS_CURRENT_USGS_RATING_FILES_ACQUIRED")


if __name__ == "__main__":
    main()
