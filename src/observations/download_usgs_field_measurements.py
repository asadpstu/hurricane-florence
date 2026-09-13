"""
Step 1D - Download and pair USGS field measurements around Hurricane Florence.

Default study case
------------------
USGS site : 02089000
River     : Neuse River near Goldsboro, NC
Years     : 2017-2019
Event     : Hurricane Florence, reference date 2018-09-18

Why this exists
---------------
The currently published USGS rating is not the rating that was in effect in
September 2018. USGS field measurements provide physically measured discharge
and gage-height observations collected during site visits. These are the most
useful public observations for reconstructing or checking the event-era
stage-discharge relationship.

USGS field-measurements API behavior
------------------------------------
Each measurement is a separate record. Measurements from the same field visit
share a `field_visit_id`; this script uses that ID to pair:
- reading_type = Discharge
- reading_type = MeanGageHeight

Scientific safeguards
---------------------
- No rating curve is fitted in this step.
- Only observations from the same field_visit_id are paired.
- Approval status, measurement rating, control condition, procedures, and
  qualifiers are preserved.
- Gage height remains local gauge stage; it is not converted to absolute WSE.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


FIELD_MEASUREMENTS_ITEMS_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/"
    "collections/field-measurements/items"
)

CFS_TO_CMS = 0.028316846592
FT_TO_M = 0.3048

USER_AGENT = (
    "neuse-flood-validation/1.0 "
    "(academic USGS field-measurement acquisition)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and pair USGS field discharge and gage-height "
            "measurements."
        )
    )
    parser.add_argument("--site", default="02089000")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--event-date", default="2018-09-18")
    parser.add_argument(
        "--event-window-days",
        type=int,
        default=180,
        help=(
            "Flag paired field visits within +/- this many days of the "
            "reference event date."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "input/observations/usgs/02089000/field_measurements_2017_2019"
        ),
    )
    parser.add_argument("--http-timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Optional USGS Water Data API key. A few low-rate requests usually "
            "work without one."
        ),
    )
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


def atomic_write_json(payload: Any, path: Path) -> None:
    atomic_write_text(json.dumps(payload, indent=2), path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.partial{path.suffix}")
    temp.unlink(missing_ok=True)
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


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


def request_json(
    url: str,
    *,
    params: dict[str, Any] | None,
    timeout: int,
    api_key: str | None,
) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["X-Api-Key"] = api_key

    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers=headers,
    )
    response.raise_for_status()
    return response.json()


def fetch_year(
    *,
    monitoring_location_id: str,
    year: int,
    timeout: int,
    api_key: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Fetch every field-measurement feature for a site/year, following OGC
    pagination links when needed.
    """
    params = {
        "f": "json",
        "monitoring_location_id": monitoring_location_id,
        "year": year,
        "limit": 10000,
    }

    features: list[dict[str, Any]] = []
    request_urls: list[str] = []

    url = FIELD_MEASUREMENTS_ITEMS_URL
    first = True

    while url:
        headers = {"User-Agent": USER_AGENT}
        if api_key:
            headers["X-Api-Key"] = api_key

        response = requests.get(
            url,
            params=params if first else None,
            timeout=timeout,
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        request_urls.append(response.url)

        year_features = payload.get("features", [])
        features.extend(year_features)

        next_url = None
        for link in payload.get("links", []):
            if link.get("rel") == "next" and link.get("href"):
                next_url = link["href"]
                break

        url = next_url
        first = False

    return features, request_urls


def feature_to_row(feature: dict[str, Any]) -> dict[str, Any]:
    p = feature.get("properties", {})
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or [None, None]

    value_raw = p.get("value")
    try:
        value_numeric = float(value_raw)
    except (TypeError, ValueError):
        value_numeric = np.nan

    return {
        "record_id": feature.get("id"),
        "field_measurements_series_id": p.get(
            "field_measurements_series_id"
        ),
        "field_visit_id": p.get("field_visit_id"),
        "reading_type": p.get("reading_type"),
        "parameter_code": p.get("parameter_code"),
        "monitoring_location_id": p.get("monitoring_location_id"),
        "time": p.get("time"),
        "value": value_numeric,
        "unit_of_measure": p.get("unit_of_measure"),
        "approval_status": p.get("approval_status"),
        "measurement_rated": p.get("measurement_rated"),
        "control_condition": p.get("control_condition"),
        "observing_procedure_code": p.get(
            "observing_procedure_code"
        ),
        "observing_procedure": p.get("observing_procedure"),
        "qualifier": json.dumps(p.get("qualifier")),
        "vertical_datum": p.get("vertical_datum"),
        "measuring_agency": p.get("measuring_agency"),
        "year": p.get("year"),
        "month": p.get("month"),
        "day": p.get("day"),
        "time_of_day": p.get("time_of_day"),
        "longitude": coords[0] if len(coords) >= 1 else None,
        "latitude": coords[1] if len(coords) >= 2 else None,
        "last_modified": p.get("last_modified"),
    }


def normalize_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["time_utc"] = pd.to_datetime(
        frame["time"], utc=True, errors="coerce"
    )

    # Retain only the two measurement types needed for H-Q pairing.
    wanted = frame["reading_type"].isin(
        ["Discharge", "MeanGageHeight"]
    )
    frame = frame.loc[wanted].copy()

    # Convert to standard scientific columns without discarding source units.
    frame["discharge_cfs"] = np.nan
    frame["discharge_cms"] = np.nan
    frame["gage_height_ft"] = np.nan
    frame["gage_height_m"] = np.nan

    qmask = frame["reading_type"].eq("Discharge")
    hmask = frame["reading_type"].eq("MeanGageHeight")

    # The expected units for USGS stream field measurements are ft3/s and ft.
    q_units = frame.loc[qmask, "unit_of_measure"].astype(str)
    h_units = frame.loc[hmask, "unit_of_measure"].astype(str)

    # Be permissive about common spelling variants while preserving originals.
    q_cfs_mask = qmask & frame["unit_of_measure"].astype(str).str.lower().isin(
        ["ft3/s", "ft^3/s", "cfs", "ft³/s"]
    )
    h_ft_mask = hmask & frame["unit_of_measure"].astype(str).str.lower().eq("ft")

    frame.loc[q_cfs_mask, "discharge_cfs"] = frame.loc[
        q_cfs_mask, "value"
    ]
    frame.loc[q_cfs_mask, "discharge_cms"] = (
        frame.loc[q_cfs_mask, "value"] * CFS_TO_CMS
    )

    frame.loc[h_ft_mask, "gage_height_ft"] = frame.loc[
        h_ft_mask, "value"
    ]
    frame.loc[h_ft_mask, "gage_height_m"] = (
        frame.loc[h_ft_mask, "value"] * FT_TO_M
    )

    return frame.sort_values(
        ["time_utc", "field_visit_id", "reading_type"]
    ).reset_index(drop=True)


def choose_best_record(group: pd.DataFrame) -> pd.Series:
    """
    Pick one record when a field visit contains duplicate records of the same
    reading type. Preference:
      1. Approved
      2. measurement_rated Good
      3. non-null numeric value
      4. latest last_modified
    """
    scored = group.copy()
    scored["_approved"] = scored["approval_status"].eq("Approved").astype(int)
    scored["_good"] = scored["measurement_rated"].eq("Good").astype(int)
    scored["_numeric"] = scored["value"].notna().astype(int)
    scored["_modified"] = pd.to_datetime(
        scored["last_modified"], utc=True, errors="coerce"
    )

    scored = scored.sort_values(
        ["_approved", "_good", "_numeric", "_modified"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return scored.iloc[0]


def pair_field_visits(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    paired_rows: list[dict[str, Any]] = []

    for visit_id, visit in frame.groupby("field_visit_id", dropna=False):
        q_records = visit.loc[visit["reading_type"].eq("Discharge")]
        h_records = visit.loc[visit["reading_type"].eq("MeanGageHeight")]

        if q_records.empty or h_records.empty:
            continue

        q = choose_best_record(q_records)
        h = choose_best_record(h_records)

        q_time = q["time_utc"]
        h_time = h["time_utc"]

        if pd.notna(q_time) and pd.notna(h_time):
            delta_minutes = abs(
                (q_time - h_time).total_seconds()
            ) / 60.0
            representative_time = q_time + (h_time - q_time) / 2
        else:
            delta_minutes = np.nan
            representative_time = q_time if pd.notna(q_time) else h_time

        paired_rows.append(
            {
                "field_visit_id": visit_id,
                "representative_time_utc": (
                    representative_time.isoformat()
                    if pd.notna(representative_time)
                    else None
                ),
                "discharge_time_utc": (
                    q_time.isoformat() if pd.notna(q_time) else None
                ),
                "gage_height_time_utc": (
                    h_time.isoformat() if pd.notna(h_time) else None
                ),
                "pair_time_difference_minutes": delta_minutes,
                "discharge_cfs": q["discharge_cfs"],
                "discharge_cms": q["discharge_cms"],
                "gage_height_ft": h["gage_height_ft"],
                "gage_height_m": h["gage_height_m"],
                "discharge_approval_status": q["approval_status"],
                "gage_height_approval_status": h["approval_status"],
                "discharge_measurement_rated": q["measurement_rated"],
                "gage_height_measurement_rated": h["measurement_rated"],
                "discharge_control_condition": q["control_condition"],
                "gage_height_control_condition": h["control_condition"],
                "discharge_observing_procedure": q[
                    "observing_procedure"
                ],
                "gage_height_observing_procedure": h[
                    "observing_procedure"
                ],
                "discharge_qualifier": q["qualifier"],
                "gage_height_qualifier": h["qualifier"],
                "discharge_record_id": q["record_id"],
                "gage_height_record_id": h["record_id"],
                "latitude": q["latitude"],
                "longitude": q["longitude"],
            }
        )

    paired = pd.DataFrame(paired_rows)
    if paired.empty:
        return paired

    paired["_time"] = pd.to_datetime(
        paired["representative_time_utc"],
        utc=True,
        errors="coerce",
    )
    paired = paired.sort_values("_time").drop(columns=["_time"])
    return paired.reset_index(drop=True)


def add_event_proximity(
    paired: pd.DataFrame,
    event_date: datetime,
    event_window_days: int,
) -> pd.DataFrame:
    if paired.empty:
        return paired

    paired = paired.copy()
    times = pd.to_datetime(
        paired["representative_time_utc"],
        utc=True,
        errors="coerce",
    )
    distance_days = (
        (times - event_date).abs().dt.total_seconds() / 86400.0
    )
    paired["days_from_event"] = distance_days
    paired["within_event_window"] = (
        distance_days <= float(event_window_days)
    )
    paired["is_nearest_field_visit_to_event"] = False

    valid = distance_days.dropna()
    if not valid.empty:
        idx = valid.idxmin()
        paired.loc[idx, "is_nearest_field_visit_to_event"] = True

    return paired


def build_qc(
    raw: pd.DataFrame,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    discharge_count = int(
        raw["reading_type"].eq("Discharge").sum()
    ) if not raw.empty else 0
    stage_count = int(
        raw["reading_type"].eq("MeanGageHeight").sum()
    ) if not raw.empty else 0

    rows.append({
        "severity": "NOTE" if discharge_count else "BLOCKING",
        "issue": "FIELD_DISCHARGE_COUNT",
        "detail": f"{discharge_count} discharge measurement records found.",
    })
    rows.append({
        "severity": "NOTE" if stage_count else "BLOCKING",
        "issue": "FIELD_GAGE_HEIGHT_COUNT",
        "detail": f"{stage_count} mean-gage-height records found.",
    })
    rows.append({
        "severity": "NOTE" if len(paired) else "BLOCKING",
        "issue": "PAIRED_FIELD_VISIT_COUNT",
        "detail": (
            f"{len(paired)} field visits contain both discharge and "
            "gage-height measurements."
        ),
    })

    if not paired.empty:
        unapproved = ~(
            paired["discharge_approval_status"].eq("Approved")
            & paired["gage_height_approval_status"].eq("Approved")
        )
        if unapproved.any():
            rows.append({
                "severity": "WARNING",
                "issue": "UNAPPROVED_PAIRED_MEASUREMENTS",
                "detail": (
                    f"{int(unapproved.sum())} paired visits contain at least "
                    "one record not marked Approved."
                ),
            })

        large_time_delta = (
            pd.to_numeric(
                paired["pair_time_difference_minutes"],
                errors="coerce",
            ) > 120
        )
        if large_time_delta.any():
            rows.append({
                "severity": "WARNING",
                "issue": "LARGE_PAIR_TIME_DIFFERENCE",
                "detail": (
                    f"{int(large_time_delta.sum())} paired visits have "
                    "discharge/stage timestamps more than 120 minutes apart."
                ),
            })

    rows.append({
        "severity": "SCIENTIFIC_NOTE",
        "issue": "NO_RATING_FIT_IN_THIS_STEP",
        "detail": (
            "This step only acquires and pairs field measurements. "
            "Historical rating reconstruction is performed separately."
        ),
    })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year.")

    event_date = datetime.strptime(
        args.event_date, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)

    prepare_output_dir(args.output_dir, args.overwrite)

    monitoring_location_id = f"USGS-{args.site}"

    print("=" * 100)
    print("STEP 1D - USGS FIELD-MEASUREMENT ACQUISITION AND H-Q PAIRING")
    print("=" * 100)
    print(f"Monitoring location                : {monitoring_location_id}")
    print(
        f"Measurement years                  : "
        f"{args.start_year}-{args.end_year}"
    )
    print(f"Reference event date               : {args.event_date}")
    print()

    all_features: list[dict[str, Any]] = []
    request_urls: list[str] = []
    yearly_counts: dict[str, int] = {}

    raw_by_year_dir = args.output_dir / "raw_by_year"
    raw_by_year_dir.mkdir(parents=True, exist_ok=True)

    for year in range(args.start_year, args.end_year + 1):
        features, urls = fetch_year(
            monitoring_location_id=monitoring_location_id,
            year=year,
            timeout=args.http_timeout_seconds,
            api_key=args.api_key,
        )
        yearly_counts[str(year)] = len(features)
        all_features.extend(features)
        request_urls.extend(urls)
        atomic_write_json(
            {
                "type": "FeatureCollection",
                "features": features,
                "numberReturned": len(features),
            },
            raw_by_year_dir / f"field_measurements_{year}.json",
        )
        print(f"{year} raw field records              : {len(features):,}")

    rows = [feature_to_row(f) for f in all_features]
    raw = pd.DataFrame(rows)

    if raw.empty:
        raise RuntimeError(
            "USGS returned no field-measurement records for the requested "
            "site/year range."
        )

    normalized = normalize_measurements(raw)
    paired = pair_field_visits(normalized)
    paired = add_event_proximity(
        paired,
        event_date,
        args.event_window_days,
    )

    qc = build_qc(normalized, paired)
    blocking_count = int((qc["severity"] == "BLOCKING").sum())

    raw_path = args.output_dir / "usgs_field_measurements_long.csv"
    paired_path = args.output_dir / "usgs_field_measurements_paired_hq.csv"
    event_path = (
        args.output_dir / "usgs_field_measurements_near_florence.csv"
    )
    qc_path = args.output_dir / "usgs_field_measurements_qc.csv"
    metadata_path = args.output_dir / "usgs_field_measurements_metadata.json"

    atomic_write_csv(normalized, raw_path)
    atomic_write_csv(paired, paired_path)

    event_subset = (
        paired.loc[paired["within_event_window"]].copy()
        if not paired.empty
        else pd.DataFrame()
    )
    atomic_write_csv(event_subset, event_path)
    atomic_write_csv(qc, qc_path)

    nearest = None
    if not paired.empty:
        nearest_rows = paired.loc[
            paired["is_nearest_field_visit_to_event"]
        ]
        if not nearest_rows.empty:
            nearest = nearest_rows.iloc[0].to_dict()

    metadata = {
        "status": (
            "PASS_USGS_FIELD_MEASUREMENTS_PAIRED"
            if blocking_count == 0
            else "FAIL_USGS_FIELD_MEASUREMENTS_INCOMPLETE"
        ),
        "step": "STEP_1D",
        "created_utc": utc_now(),
        "source": "USGS Water Data for the Nation OGC field-measurements API",
        "monitoring_location_id": monitoring_location_id,
        "requested_years": {
            "start": args.start_year,
            "end": args.end_year,
        },
        "event_reference": {
            "date": args.event_date,
            "window_days": args.event_window_days,
        },
        "yearly_raw_record_counts": yearly_counts,
        "request_urls": request_urls,
        "measurement_counts": {
            "relevant_long_records": int(len(normalized)),
            "discharge_records": int(
                normalized["reading_type"].eq("Discharge").sum()
            ),
            "gage_height_records": int(
                normalized["reading_type"].eq("MeanGageHeight").sum()
            ),
            "paired_hq_field_visits": int(len(paired)),
            "paired_hq_visits_near_event": int(len(event_subset)),
        },
        "nearest_field_visit_to_event": nearest,
        "scientific_use": (
            "Use paired field-measured H-Q observations to evaluate or "
            "reconstruct the stage-discharge relation applicable near 2018. "
            "Do not treat this step itself as a fitted historical rating."
        ),
        "gage_height_is_absolute_wse": False,
        "blocking_qc_issue_count": blocking_count,
        "warning_count": int((qc["severity"] == "WARNING").sum()),
        "output_paths": {
            "long_measurements": str(raw_path),
            "paired_hq": str(paired_path),
            "near_event_hq": str(event_path),
            "qc": str(qc_path),
            "metadata": str(metadata_path),
            "raw_by_year_dir": str(raw_by_year_dir),
        },
    }
    atomic_write_json(metadata, metadata_path)

    print()
    print(
        f"Relevant discharge/stage records   : {len(normalized):,}"
    )
    print(
        "Discharge measurement records      : "
        f"{int(normalized['reading_type'].eq('Discharge').sum()):,}"
    )
    print(
        "Gage-height measurement records     : "
        f"{int(normalized['reading_type'].eq('MeanGageHeight').sum()):,}"
    )
    print(f"Paired H-Q field visits             : {len(paired):,}")
    print(
        "Paired visits near Florence         : "
        f"{len(event_subset):,}"
    )

    if nearest:
        print(
            "Nearest paired field visit          : "
            f"{nearest.get('representative_time_utc')}"
        )
        print(
            "Days from event                     : "
            f"{nearest.get('days_from_event'):.2f}"
        )
        if pd.notna(nearest.get("gage_height_ft")):
            print(
                "Nearest measured gage height        : "
                f"{nearest.get('gage_height_ft'):.3f} ft"
            )
        if pd.notna(nearest.get("discharge_cfs")):
            print(
                "Nearest measured discharge          : "
                f"{nearest.get('discharge_cfs'):,.1f} ft3/s"
            )

    print()
    print(f"Paired H-Q CSV                      : {paired_path}")
    print(f"Near-event H-Q CSV                  : {event_path}")
    print(f"QC                                  : {qc_path}")
    print(f"Metadata                            : {metadata_path}")
    print()
    print(f"Status                              : {metadata['status']}")

    if blocking_count:
        raise RuntimeError(
            f"Step 1D produced {blocking_count} blocking QC issue(s)."
        )


if __name__ == "__main__":
    main()
