"""
Step 1 - Download observed USGS discharge and gage height for a flood event.

Default study case
------------------
River   : Neuse River near Goldsboro, North Carolina, USA
USGS ID : 02089000
Event   : Hurricane Florence, September 2018
Window  : 2018-09-10 through 2018-09-25

Downloaded variables
--------------------
00060 : Discharge, cubic feet per second
00065 : Gage height, feet

Outputs are standardized to both source units and SI units:
- discharge_cfs
- discharge_cms
- gage_height_ft
- gage_height_m

Important:
Gage height is a local gauge-stage measurement. This script deliberately does
NOT add a vertical datum to turn gage height into absolute WSE. Datum handling
will be implemented as a separate, explicitly verified step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
CFS_TO_CMS = 0.028316846592
FT_TO_M = 0.3048

PARAMETERS = {
    "00060": {
        "source_name": "Discharge",
        "source_unit": "ft3/s",
        "output_source_column": "discharge_cfs",
        "output_si_column": "discharge_cms",
        "si_factor": CFS_TO_CMS,
        "si_unit": "m3/s",
    },
    "00065": {
        "source_name": "Gage height",
        "source_unit": "ft",
        "output_source_column": "gage_height_ft",
        "output_si_column": "gage_height_m",
        "si_factor": FT_TO_M,
        "si_unit": "m",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download USGS discharge and gage-height observations."
    )
    parser.add_argument("--site", default="02089000")
    parser.add_argument("--start", default="2018-09-10")
    parser.add_argument("--end", default="2018-09-25")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input/observations/usgs/02089000/florence_2018"),
    )
    parser.add_argument("--http-timeout-seconds", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.partial{path.suffix}")
    temp.unlink(missing_ok=True)
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(json.dumps(payload, indent=2), path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.partial{path.suffix}")
    temp.unlink(missing_ok=True)
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def prepare_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist. Use --overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )
    if overwrite:
        for path in existing:
            path.unlink()


def request_usgs(site: str, start: str, end: str, timeout: int) -> tuple[dict[str, Any], str]:
    params = {
        "format": "json",
        "sites": site,
        "startDT": start,
        "endDT": end,
        "parameterCd": ",".join(PARAMETERS),
        "siteStatus": "all",
    }

    response = requests.get(
        USGS_IV_URL,
        params=params,
        timeout=timeout,
        headers={
            "User-Agent": (
                "flood-validation-research/1.0 "
                "(academic hydrology data acquisition)"
            )
        },
    )
    response.raise_for_status()
    return response.json(), response.url


def parse_timeseries(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    series = payload.get("value", {}).get("timeSeries", [])
    if not series:
        raise RuntimeError("USGS returned no time-series records.")

    frames: list[pd.DataFrame] = []
    site_metadata: dict[str, Any] = {}

    for item in series:
        source_info = item.get("sourceInfo", {})
        variable = item.get("variable", {})
        variable_code_items = variable.get("variableCode", [])
        if not variable_code_items:
            continue

        parameter = str(variable_code_items[0].get("value", "")).strip()
        if parameter not in PARAMETERS:
            continue

        values_groups = item.get("values", [])
        observations: list[dict[str, Any]] = []

        for group in values_groups:
            for value in group.get("value", []):
                raw_value = value.get("value")
                if raw_value in (None, "", "-999999"):
                    continue

                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    continue

                observations.append(
                    {
                        "datetime": value.get("dateTime"),
                        "value": numeric_value,
                        "qualifiers": "|".join(value.get("qualifiers", [])),
                    }
                )

        if not observations:
            continue

        spec = PARAMETERS[parameter]
        frame = pd.DataFrame(observations)
        frame["datetime"] = pd.to_datetime(
            frame["datetime"], utc=True, errors="raise"
        )
        frame = frame.drop_duplicates(subset=["datetime"], keep="last")
        frame = frame.sort_values("datetime")

        frame = frame.rename(columns={"value": spec["output_source_column"]})
        frame[spec["output_si_column"]] = (
            frame[spec["output_source_column"]].astype(float)
            * spec["si_factor"]
        )
        frame = frame.rename(columns={"qualifiers": f"{parameter}_qualifiers"})
        frames.append(frame)

        if not site_metadata:
            geo = source_info.get("geoLocation", {}).get("geogLocation", {})
            site_codes = source_info.get("siteCode", [])
            site_metadata = {
                "site_name": source_info.get("siteName"),
                "site_code": site_codes[0].get("value") if site_codes else None,
                "latitude": geo.get("latitude"),
                "longitude": geo.get("longitude"),
                "srs": geo.get("srs"),
                "site_property": source_info.get("siteProperty", []),
            }

    if not frames:
        raise RuntimeError("USGS response contained no usable records.")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="datetime", how="outer")

    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["datetime_utc"] = merged["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    merged = merged.drop(columns=["datetime"])

    ordered = [
        "datetime_utc",
        "discharge_cfs",
        "discharge_cms",
        "00060_qualifiers",
        "gage_height_ft",
        "gage_height_m",
        "00065_qualifiers",
    ]
    merged = merged[[c for c in ordered if c in merged.columns]]
    return merged, site_metadata


def build_qc(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in ["discharge_cfs", "gage_height_ft"]:
        if column not in frame.columns:
            rows.append(
                {
                    "severity": "BLOCKING",
                    "issue": "MISSING_REQUIRED_VARIABLE",
                    "item": column,
                    "detail": f"{column} was not returned by USGS.",
                }
            )
            continue

        valid = pd.to_numeric(frame[column], errors="coerce").notna()
        rows.append(
            {
                "severity": "NOTE" if valid.any() else "BLOCKING",
                "issue": "VARIABLE_RECORD_COUNT",
                "item": column,
                "detail": f"{int(valid.sum())} valid observations.",
            }
        )

    rows.append(
        {
            "severity": "SCIENTIFIC_NOTE",
            "issue": "GAGE_HEIGHT_IS_NOT_ABSOLUTE_WSE",
            "item": "gage_height",
            "detail": (
                "Gage height is retained as gauge-relative stage. "
                "Absolute WSE requires separately verified gauge-datum metadata."
            ),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if args.start >= args.end:
        raise ValueError("--start must be earlier than --end.")

    output_dir = args.output_dir
    timeseries_path = output_dir / "usgs_event_observations.csv"
    qc_path = output_dir / "usgs_event_observations_qc.csv"
    metadata_path = output_dir / "usgs_event_observations_metadata.json"
    raw_path = output_dir / "usgs_event_observations_raw.json"

    prepare_outputs(
        [timeseries_path, qc_path, metadata_path, raw_path],
        args.overwrite,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("STEP 1 - USGS EVENT OBSERVATION ACQUISITION")
    print("=" * 96)
    print(f"USGS site                         : {args.site}")
    print(f"Event window                      : {args.start} -> {args.end}")
    print()

    payload, request_url = request_usgs(
        args.site,
        args.start,
        args.end,
        args.http_timeout_seconds,
    )
    atomic_write_json(payload, raw_path)

    frame, site_metadata = parse_timeseries(payload)
    qc = build_qc(frame)

    blocking = int((qc["severity"] == "BLOCKING").sum())
    if blocking:
        raise RuntimeError(
            f"USGS acquisition produced {blocking} blocking QC issue(s)."
        )

    atomic_write_csv(frame, timeseries_path)
    atomic_write_csv(qc, qc_path)

    q = pd.to_numeric(frame["discharge_cms"], errors="coerce")
    h = pd.to_numeric(frame["gage_height_m"], errors="coerce")

    metadata = {
        "status": "PASS_USGS_EVENT_OBSERVATIONS_ACQUIRED",
        "step": "STEP_1",
        "created_utc": utc_now(),
        "source": "USGS NWIS Instantaneous Values",
        "request_url": request_url,
        "site": args.site,
        "event_name": "Hurricane Florence 2018",
        "event_window": {"start": args.start, "end": args.end},
        "site_metadata_from_response": site_metadata,
        "record_count": int(len(frame)),
        "discharge": {
            "valid_count": int(q.notna().sum()),
            "minimum_cms": float(q.min()) if q.notna().any() else None,
            "maximum_cms": float(q.max()) if q.notna().any() else None,
        },
        "gage_height": {
            "valid_count": int(h.notna().sum()),
            "minimum_m": float(h.min()) if h.notna().any() else None,
            "maximum_m": float(h.max()) if h.notna().any() else None,
            "absolute_wse_created": False,
        },
        "blocking_qc_issue_count": blocking,
        "warning_count": int((qc["severity"] == "WARNING").sum()),
        "output_paths": {
            "timeseries": str(timeseries_path),
            "qc": str(qc_path),
            "metadata": str(metadata_path),
            "raw_response": str(raw_path),
        },
    }
    atomic_write_json(metadata, metadata_path)

    print(f"Records                           : {len(frame):,}")
    print(f"Valid discharge observations      : {int(q.notna().sum()):,}")
    print(f"Valid stage observations          : {int(h.notna().sum()):,}")
    if q.notna().any():
        print(f"Peak observed discharge           : {q.max():,.3f} m3/s")
    if h.notna().any():
        print(f"Peak observed gage height         : {h.max():,.3f} m")
    print()
    print("Status                            : PASS_USGS_EVENT_OBSERVATIONS_ACQUIRED")
    print(f"Timeseries                        : {timeseries_path}")
    print(f"QC                                : {qc_path}")
    print(f"Metadata                          : {metadata_path}")
    print(f"Raw response                      : {raw_path}")


if __name__ == "__main__":
    main()
