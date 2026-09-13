"""
Step 1C - Discover the historical USGS stage-discharge rating applicable to
Hurricane Florence 2018 at USGS 02089000.

Purpose
-------
We need the rating and any shift/correction that were actually applicable
around the event date, not the currently published 2023+ rating.

This script deliberately separates:
1. Authoritative current USGS STAC products.
2. An event-date STAC query (diagnostic only).
3. Archived snapshots of the former USGS NWIS rating endpoint.
4. A ready-to-send USGS historical-data request if no authoritative historical
   rating can be recovered publicly.

Scientific safeguards
---------------------
- A Wayback snapshot is treated only as a HISTORICAL CANDIDATE, not automatically
  as an authoritative event-era rating.
- The script never labels a rating "Florence 2018 rating" unless its own header
  contains an effective date interval that includes the event date.
- Shift applicability is checked separately from base-rating applicability.
- If public recovery fails, the correct next action is to request the historical
  rating/shift from the USGS office managing the station.

Dependencies
------------
requests
pandas
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests


USGS_STAC_SEARCH = "https://api.waterdata.usgs.gov/stac/v0/search"
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK_REPLAY = "https://web.archive.org/web"

LEGACY_RATING_ENDPOINTS = (
    "https://waterdata.usgs.gov/nwisweb/get_ratings",
    "http://waterdata.usgs.gov/nwisweb/get_ratings",
)

USER_AGENT = (
    "neuse-flood-validation/1.0 "
    "(academic historical USGS rating discovery)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover historical USGS rating/shift evidence for a flood event."
        )
    )
    parser.add_argument("--site", default="02089000")
    parser.add_argument("--event-date", default="2018-09-18")
    parser.add_argument("--archive-from-year", type=int, default=2017)
    parser.add_argument("--archive-to-year", type=int, default=2019)
    parser.add_argument(
        "--current-rating-dir",
        type=Path,
        default=Path("input/observations/usgs/02089000/ratings"),
        help=(
            "Directory created by Step 1B. Existing current rating headers are "
            "inspected but never modified."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "input/observations/usgs/02089000/historical_rating_discovery"
        ),
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


def atomic_write_json(payload: Any, path: Path) -> None:
    atomic_write_text(json.dumps(payload, indent=2), path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    tmp.unlink(missing_ok=True)
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty. Use --overwrite."
        )
    if overwrite:
        for path in existing:
            if path.is_file():
                path.unlink()


def get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int,
    allow_failure: bool = False,
) -> requests.Response | None:
    try:
        response = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        return response
    except requests.RequestException:
        if allow_failure:
            return None
        raise


def parse_usgs_datetime_token(token: str | None) -> datetime | None:
    if not token or token.startswith("-"):
        return None
    token = token.strip()
    try:
        return datetime.strptime(token, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def extract_header_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "rating_id": None,
        "rating_type": None,
        "rating_begin": None,
        "rating_end": None,
        "rating_comment": None,
        "rating_shifted_timestamp": None,
        "shift_prev_begin": None,
        "shift_prev_end": None,
        "shift_prev_stage1": None,
        "shift_prev_shift1": None,
        "shift_prev_stage2": None,
        "shift_prev_shift2": None,
        "shift_prev_stage3": None,
        "shift_prev_shift3": None,
        "shift_prev_comment": None,
    }

    rating_match = re.search(
        r'//RATING ID="([^"]+)" TYPE="([^"]+)"',
        text,
    )
    if rating_match:
        metadata["rating_id"] = rating_match.group(1).strip()
        metadata["rating_type"] = rating_match.group(2).strip()

    period_match = re.search(
        r'//RATING_DATETIME BEGIN=([^\s]+)\s+BZONE="[^"]*"\s+'
        r'END=([^\s]+)',
        text,
    )
    if period_match:
        metadata["rating_begin"] = period_match.group(1)
        metadata["rating_end"] = period_match.group(2)

    comment_match = re.search(
        r'//RATING_DATETIME COMMENT="([^"]*)"',
        text,
    )
    if comment_match:
        metadata["rating_comment"] = comment_match.group(1).strip()

    shifted_match = re.search(r'//RATING SHIFTED="([^"]+)"', text)
    if shifted_match:
        metadata["rating_shifted_timestamp"] = shifted_match.group(1).strip()

    shift_period = re.search(
        r'//SHIFT_PREV BEGIN="([^"]+)"\s+BZONE="[^"]*"\s+'
        r'END="([^"]+)"',
        text,
    )
    if shift_period:
        metadata["shift_prev_begin"] = shift_period.group(1).strip()
        metadata["shift_prev_end"] = shift_period.group(2).strip()

    shift_values = re.search(
        r'//SHIFT_PREV STAGE1="([^"]+)" SHIFT1="([^"]+)" '
        r'STAGE2="([^"]+)" SHIFT2="([^"]+)" '
        r'STAGE3="([^"]+)" SHIFT3="([^"]+)"',
        text,
    )
    if shift_values:
        (
            metadata["shift_prev_stage1"],
            metadata["shift_prev_shift1"],
            metadata["shift_prev_stage2"],
            metadata["shift_prev_shift2"],
            metadata["shift_prev_stage3"],
            metadata["shift_prev_shift3"],
        ) = [v.strip() for v in shift_values.groups()]

    shift_comment = re.search(
        r'//SHIFT_PREV COMMENT="([^"]*)"',
        text,
    )
    if shift_comment:
        metadata["shift_prev_comment"] = shift_comment.group(1).strip()

    return metadata


def event_within_rating_period(
    event_date: datetime,
    header: dict[str, Any],
) -> bool:
    begin = parse_usgs_datetime_token(header.get("rating_begin"))
    end = parse_usgs_datetime_token(header.get("rating_end"))

    if begin is None:
        return False

    event_naive = event_date.replace(tzinfo=None)
    if event_naive < begin:
        return False
    if end is not None and event_naive > end:
        return False
    return True


def archive_distance_seconds(
    timestamp: str,
    event_date: datetime,
) -> float:
    capture = datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )
    return abs((capture - event_date).total_seconds())


def query_wayback(
    original_url: str,
    *,
    from_year: int,
    to_year: int,
    timeout: int,
) -> list[dict[str, Any]]:
    params = {
        "url": original_url,
        "from": str(from_year),
        "to": str(to_year),
        "output": "json",
        "filter": "statuscode:200",
        "fl": "timestamp,original,statuscode,mimetype,digest",
        "collapse": "digest",
    }
    response = get(
        WAYBACK_CDX,
        params=params,
        timeout=timeout,
        allow_failure=True,
    )
    if response is None:
        return []

    try:
        payload = response.json()
    except ValueError:
        return []

    if not payload or len(payload) < 2:
        return []

    columns = payload[0]
    results = []
    for row in payload[1:]:
        if len(row) != len(columns):
            continue
        results.append(dict(zip(columns, row)))
    return results


def build_legacy_url(endpoint: str, site: str, file_type: str) -> str:
    return endpoint + "?" + urlencode(
        {"site_no": site, "file_type": file_type}
    )


def download_wayback_candidate(
    candidate: dict[str, Any],
    *,
    file_type: str,
    output_dir: Path,
    timeout: int,
) -> tuple[Path | None, dict[str, Any] | None]:
    timestamp = candidate["timestamp"]
    original = candidate["original"]
    replay_url = f"{WAYBACK_REPLAY}/{timestamp}id_/{original}"

    response = get(
        replay_url,
        timeout=timeout,
        allow_failure=True,
    )
    if response is None:
        return None, None

    text = response.text
    if "RATING" not in text.upper():
        return None, None

    raw_path = output_dir / (
        f"archived_{file_type}_{timestamp}.rdb"
    )
    atomic_write_text(text, raw_path)

    header_lines = [
        line for line in text.splitlines()
        if line.startswith("#")
    ]
    header_path = output_dir / (
        f"archived_{file_type}_{timestamp}_header.txt"
    )
    atomic_write_text("\n".join(header_lines) + "\n", header_path)

    metadata = extract_header_metadata(text)
    metadata["wayback_timestamp"] = timestamp
    metadata["original_url"] = original
    metadata["replay_url"] = replay_url
    metadata["raw_path"] = str(raw_path)
    metadata["header_path"] = str(header_path)

    return raw_path, metadata


def inspect_current_header(
    current_rating_dir: Path,
    file_type: str,
) -> dict[str, Any] | None:
    path = current_rating_dir / (
        f"usgs_02089000_rating_{file_type}_header.txt"
    )
    if not path.exists():
        # Also support non-hardcoded site filenames by looking for one match.
        matches = list(current_rating_dir.glob(
            f"usgs_*_rating_{file_type}_header.txt"
        ))
        if not matches:
            return None
        path = matches[0]

    text = path.read_text(encoding="utf-8", errors="replace")
    metadata = extract_header_metadata(text)
    metadata["source_path"] = str(path)
    return metadata


def write_request_template(
    *,
    site: str,
    event_date: str,
    output_path: Path,
) -> None:
    text = f"""Subject: Request for historical stage-discharge rating and shift history for USGS {site}

Hello,

I am conducting an academic flood-model validation study using USGS monitoring
location {site} (Neuse River near Goldsboro, North Carolina).

Could you please provide, if available, the historical streamflow rating
information applicable around Hurricane Florence, specifically {event_date}?

Requested items:
1. The base stage-discharge rating ID/table in effect on {event_date}.
2. The effective begin/end dates for that rating.
3. Any shift corrections in effect during the Florence flood period.
4. Any stage corrections relevant to the event.
5. If distributable, the corresponding station analysis or rating-history
   documentation covering water year 2018.

Machine-readable RDB/CSV files are preferred if available.

This request is specifically for historical scientific validation; I already
have the currently published USGS rating files and the 2018 continuous
gage-height/discharge time series.

Thank you.
"""
    atomic_write_text(text, output_path)


def main() -> None:
    args = parse_args()

    event_date = datetime.strptime(
        args.event_date, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)

    if args.archive_from_year > event_date.year:
        raise ValueError("--archive-from-year must not be after event year.")
    if args.archive_to_year < event_date.year:
        raise ValueError("--archive-to-year must not be before event year.")

    prepare_output_dir(args.output_dir, args.overwrite)

    monitoring_location_id = f"USGS-{args.site}"

    print("=" * 100)
    print("STEP 1C - HISTORICAL USGS RATING / SHIFT DISCOVERY")
    print("=" * 100)
    print(f"Monitoring location                : {monitoring_location_id}")
    print(f"Target event date                  : {args.event_date}")
    print(
        f"Archive search years                : "
        f"{args.archive_from_year}-{args.archive_to_year}"
    )
    print()

    # ------------------------------------------------------------------
    # A. Inspect current Step 1B headers
    # ------------------------------------------------------------------
    current_rows = []
    for file_type in ("base", "exsa", "corr"):
        metadata = inspect_current_header(
            args.current_rating_dir, file_type
        )
        if metadata is None:
            current_rows.append(
                {
                    "file_type": file_type,
                    "available": False,
                    "rating_id": None,
                    "rating_begin": None,
                    "event_within_rating_period": False,
                }
            )
            continue

        current_rows.append(
            {
                "file_type": file_type,
                "available": True,
                "rating_id": metadata.get("rating_id"),
                "rating_begin": metadata.get("rating_begin"),
                "rating_end": metadata.get("rating_end"),
                "shift_prev_begin": metadata.get("shift_prev_begin"),
                "event_within_rating_period": event_within_rating_period(
                    event_date, metadata
                ),
                "source_path": metadata.get("source_path"),
            }
        )

    current_df = pd.DataFrame(current_rows)
    current_check_path = (
        args.output_dir / "current_rating_applicability_check.csv"
    )
    atomic_write_csv(current_df, current_check_path)

    # ------------------------------------------------------------------
    # B. Query STAC using the event date.
    # This is diagnostic: STAC item datetime describes the published file
    # artifact timestamp, not necessarily rating effective time.
    # ------------------------------------------------------------------
    stac_params = {
        "collection": "ratings",
        "filter": (
            f"monitoring_location_id='{monitoring_location_id}'"
        ),
        "datetime": (
            f"{args.event_date}T00:00:00Z/"
            f"{args.event_date}T23:59:59Z"
        ),
        "limit": 100,
    }
    stac_response = get(
        USGS_STAC_SEARCH,
        params=stac_params,
        timeout=args.http_timeout_seconds,
        allow_failure=True,
    )
    if stac_response is not None:
        try:
            stac_event = stac_response.json()
        except ValueError:
            stac_event = {
                "error": "USGS STAC event-date response was not JSON."
            }
    else:
        stac_event = {
            "error": "USGS STAC event-date query failed."
        }

    stac_event_path = (
        args.output_dir / "usgs_stac_event_date_query.json"
    )
    atomic_write_json(stac_event, stac_event_path)

    # ------------------------------------------------------------------
    # C. Search archived former NWIS rating files.
    # ------------------------------------------------------------------
    archive_manifest: list[dict[str, Any]] = []
    selected_candidates: dict[str, dict[str, Any]] = {}

    for file_type in ("base", "exsa", "corr"):
        all_candidates: list[dict[str, Any]] = []

        for endpoint in LEGACY_RATING_ENDPOINTS:
            original_url = build_legacy_url(
                endpoint, args.site, file_type
            )
            results = query_wayback(
                original_url,
                from_year=args.archive_from_year,
                to_year=args.archive_to_year,
                timeout=args.http_timeout_seconds,
            )
            for result in results:
                result = dict(result)
                result["queried_url"] = original_url
                result["file_type"] = file_type
                try:
                    result["distance_seconds"] = archive_distance_seconds(
                        result["timestamp"], event_date
                    )
                except Exception:
                    result["distance_seconds"] = float("inf")
                all_candidates.append(result)

        # Deduplicate by timestamp + original.
        unique = {}
        for item in all_candidates:
            key = (item.get("timestamp"), item.get("original"))
            unique[key] = item
        all_candidates = list(unique.values())
        all_candidates.sort(key=lambda x: x["distance_seconds"])

        cdx_path = args.output_dir / (
            f"wayback_{file_type}_candidate_index.json"
        )
        atomic_write_json(all_candidates, cdx_path)

        # Try closest captures until a valid-looking rating file is obtained.
        selected_meta = None
        for candidate in all_candidates[:20]:
            _, candidate_meta = download_wayback_candidate(
                candidate,
                file_type=file_type,
                output_dir=args.output_dir,
                timeout=args.http_timeout_seconds,
            )
            if candidate_meta is None:
                continue

            candidate_meta["event_within_rating_period"] = (
                event_within_rating_period(
                    event_date, candidate_meta
                )
            )
            selected_meta = candidate_meta
            break

        if selected_meta is not None:
            selected_candidates[file_type] = selected_meta
            archive_manifest.append(
                {
                    "file_type": file_type,
                    "candidate_found": True,
                    "wayback_timestamp": selected_meta.get(
                        "wayback_timestamp"
                    ),
                    "rating_id": selected_meta.get("rating_id"),
                    "rating_begin": selected_meta.get("rating_begin"),
                    "rating_end": selected_meta.get("rating_end"),
                    "shift_prev_begin": selected_meta.get(
                        "shift_prev_begin"
                    ),
                    "event_within_rating_period": selected_meta.get(
                        "event_within_rating_period"
                    ),
                    "raw_path": selected_meta.get("raw_path"),
                    "header_path": selected_meta.get("header_path"),
                    "replay_url": selected_meta.get("replay_url"),
                }
            )
        else:
            archive_manifest.append(
                {
                    "file_type": file_type,
                    "candidate_found": False,
                    "wayback_timestamp": None,
                    "rating_id": None,
                    "rating_begin": None,
                    "rating_end": None,
                    "shift_prev_begin": None,
                    "event_within_rating_period": False,
                    "raw_path": None,
                    "header_path": None,
                    "replay_url": None,
                }
            )

    archive_df = pd.DataFrame(archive_manifest)
    archive_manifest_path = (
        args.output_dir / "historical_rating_candidate_manifest.csv"
    )
    atomic_write_csv(archive_df, archive_manifest_path)

    # ------------------------------------------------------------------
    # D. Classification
    # ------------------------------------------------------------------
    base = selected_candidates.get("base")
    exsa = selected_candidates.get("exsa")

    base_applicable = bool(
        base and base.get("event_within_rating_period")
    )
    exsa_applicable = bool(
        exsa and exsa.get("event_within_rating_period")
    )

    if base_applicable and exsa_applicable:
        status = (
            "PASS_HISTORICAL_BASE_AND_SHIFT_CANDIDATES_FOUND_REVIEW_REQUIRED"
        )
        authoritative_historical_rating_found = False
        next_step = (
            "Manually review archived base/EXSA headers and verify against "
            "USGS field measurements or an official USGS historical-data response."
        )
    elif base_applicable:
        status = (
            "PASS_HISTORICAL_BASE_CANDIDATE_FOUND_SHIFT_NOT_CONFIRMED"
        )
        authoritative_historical_rating_found = False
        next_step = (
            "Base-rating candidate covers event date, but Florence-period shift "
            "is not confirmed. Request shift history from USGS and/or inspect "
            "field measurements."
        )
    else:
        status = (
            "PASS_DISCOVERY_COMPLETED_USGS_HISTORICAL_REQUEST_REQUIRED"
        )
        authoritative_historical_rating_found = False
        next_step = (
            "No publicly recovered rating candidate is proven applicable to "
            "2018-09-18. Send the generated request to USGS, then use field "
            "measurements as the fallback reconstruction dataset."
        )

    request_path = (
        args.output_dir / "usgs_historical_rating_request.txt"
    )
    write_request_template(
        site=args.site,
        event_date=args.event_date,
        output_path=request_path,
    )

    metadata = {
        "status": status,
        "step": "STEP_1C",
        "created_utc": utc_now(),
        "monitoring_location_id": monitoring_location_id,
        "target_event_date": args.event_date,
        "archive_search_years": {
            "from": args.archive_from_year,
            "to": args.archive_to_year,
        },
        "current_rating_event_applicable": bool(
            current_df["event_within_rating_period"].fillna(False).any()
        ),
        "stac_event_date_feature_count": (
            len(stac_event.get("features", []))
            if isinstance(stac_event, dict)
            else 0
        ),
        "historical_archive_candidate_types": sorted(
            selected_candidates.keys()
        ),
        "historical_base_candidate_covers_event": base_applicable,
        "historical_exsa_candidate_covers_event": exsa_applicable,
        "authoritative_historical_rating_found": (
            authoritative_historical_rating_found
        ),
        "scientific_interpretation": (
            "Archived web snapshots are discovery evidence only. They are not "
            "treated as authoritative until their effective dates and shift "
            "history are corroborated by USGS documentation or independent "
            "field measurements."
        ),
        "next_step": next_step,
        "output_paths": {
            "current_applicability_check": str(current_check_path),
            "stac_event_query": str(stac_event_path),
            "archive_candidate_manifest": str(
                archive_manifest_path
            ),
            "usgs_request_template": str(request_path),
            "output_dir": str(args.output_dir),
        },
    }

    metadata_path = (
        args.output_dir / "historical_rating_discovery_metadata.json"
    )
    atomic_write_json(metadata, metadata_path)

    print(f"Current rating valid on event date : {metadata['current_rating_event_applicable']}")
    print(f"STAC event-date artifacts          : {metadata['stac_event_date_feature_count']}")
    print(
        "Archived base candidate covers event: "
        f"{base_applicable}"
    )
    print(
        "Archived EXSA candidate covers event: "
        f"{exsa_applicable}"
    )
    print(
        "Authoritative historical rating found: "
        f"{authoritative_historical_rating_found}"
    )
    print()
    print(f"Candidate manifest                 : {archive_manifest_path}")
    print(f"USGS request template              : {request_path}")
    print(f"Metadata                           : {metadata_path}")
    print()
    print(f"Status                             : {status}")
    print(f"Next step                          : {next_step}")


if __name__ == "__main__":
    main()
