"""
Create a frozen SHA-256 manifest for the validated Florence H-Q artifacts.

Run this after the Florence H-Q surrogate is reconstructed and validated.
The final H-Q preflight uses this manifest to verify that none of the
validated artifacts changed between validation and use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


BUILD = "FLORENCE_HQ_VALIDATED_MANIFEST_V1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="02089000")
    p.add_argument("--event-observations", type=Path, required=True)
    p.add_argument("--field-measurements", type=Path, required=True)
    p.add_argument("--h-to-q", type=Path, required=True)
    p.add_argument("--q-to-h", type=Path, required=True)
    p.add_argument("--surrogate-metadata", type=Path, required=True)
    p.add_argument("--reconstruction-fidelity", type=Path, required=True)
    p.add_argument("--field-validation-results", type=Path, required=True)
    p.add_argument("--field-validation-metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    a = parse_args()

    if a.output.exists() and not a.overwrite:
        raise FileExistsError(f"{a.output} exists. Use --overwrite.")

    artifacts = {
        "usgs_event_observations.csv": a.event_observations,
        "usgs_field_measurements_paired_hq.csv": a.field_measurements,
        "florence_effective_h_to_q_surrogate.csv": a.h_to_q,
        "florence_effective_q_to_h_surrogate.csv": a.q_to_h,
        "florence_effective_hq_metadata.json": a.surrogate_metadata,
        "florence_hq_reconstruction_fidelity.csv": a.reconstruction_fidelity,
        "field_hq_validation_results.csv": a.field_validation_results,
        "field_hq_validation_metadata.json": a.field_validation_metadata,
    }

    missing = [str(path) for path in artifacts.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot create validated H-Q manifest because required artifacts "
            "are missing:\n- " + "\n- ".join(missing)
        )

    manifest_artifacts = {
        name: {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for name, path in artifacts.items()
    }

    payload = {
        "script_build": BUILD,
        "site": str(a.site),
        "event": "Hurricane Florence 2018",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Freeze fingerprints of the validated Florence effective H-Q "
            "surrogate and its source/validation artifacts before final preflight."
        ),
        "artifacts": manifest_artifacts,
    }

    atomic_json(payload, a.output)

    print("=" * 100)
    print("FLORENCE VALIDATED H-Q MANIFEST")
    print("=" * 100)
    print(f"Site                               : {a.site}")
    print(f"Artifacts fingerprinted            : {len(manifest_artifacts)}")
    print(f"Output                             : {a.output}")
    print("Status                             : PASS_FLORENCE_HQ_VALIDATED_MANIFEST_CREATED")


if __name__ == "__main__":
    main()
