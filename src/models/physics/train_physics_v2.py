
"""
PHYSICS V2 FINAL TRAINING
Deep-groundwater/baseflow initialization using ERA5-Land layer-4 state.

Parent model
------------
Physics V2 core real-A4-routing implementation:
    physics_v2_model.py

Physics V2 changes ONE structural component:
    initial slow-flow / groundwater state.

It does NOT change:
* rainfall-runoff partition equation
* soil-store formulation
* ERA5 root-zone initialization
* hourly ET0 forcing
* quickflow reservoir
* channel-routing equation
* A4 network distances
* calibration/validation event library
* validation gates
* Florence holdout

Deep-state initialization
-------------------------
The standardized ERA5-Land layer-4 dataset supplies a calibration-only normalized:
    deep_relative_wetness in [0, 1]

For each event and subcatchment:

    initial_baseflow_flux_mm_h =
        deep_baseflow_floor_mm_h
        + deep_baseflow_range_mm_h * deep_relative_wetness

Two new globally calibrated parameters are introduced:

    deep_baseflow_floor_mm_h
    deep_baseflow_range_mm_h

This is deliberately parsimonious: no extra exponent is added yet.

The deep state initializes:
1. the local slow/base reservoir, and
2. the pre-existing channel-routing storage implied by that slow flow.

The second point is essential. If only the local base reservoir were
initialized while every channel reservoir started empty, event-start flow
would still take many routing hours to reach the gauge. Physics V2 therefore
initializes channel reservoirs at a steady-flow state consistent with the
deep-state-derived local baseflow fluxes.

Observed discharge is NOT used to initialize any event state.
It remains only the calibration/validation target.

Development protocol
--------------------
Calibration : 2015-2016
Validation  : 2017 untouched
Florence    : NOT USED

Calibration-policy revision
---------------------------
The parent model now uses a strongly event-balanced objective (95% equal
accepted-block penalty, 5% pooled regularizer by default) and writes an
explicit peak-timing audit. Hydrologic equations and the Florence holdout
remain unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BUILD = "PHYSICS_V2_FINAL_EVENT_BALANCED_V2"

EXPECTED_PARENT_TOKEN = "PHYSICS_V2_SEMIDISTRIBUTED_ERA5_ROUTING"

NEW_PARAM_NAMES = [
    "deep_baseflow_floor_mm_h",
    "deep_baseflow_range_mm_h",
]

NEW_PARAM_BOUNDS = [
    # Minimum slow-flow contribution under the driest normalized deep state.
    (0.000, 0.080),

    # Additional slow-flow flux between normalized deep wetness 0 and 1.
    (0.010, 0.220),
]


def _extract_custom_arg(flag: str) -> str:
    """
    Remove a training-only argument from sys.argv before the parent parser
    sees it and return its value.
    """
    if flag not in sys.argv:
        raise RuntimeError(
            f"{flag} is required."
        )

    idx = sys.argv.index(flag)

    if idx + 1 >= len(sys.argv):
        raise RuntimeError(
            f"{flag} requires a value."
        )

    value = sys.argv[idx + 1]

    del sys.argv[
        idx: idx + 2
    ]

    return value


def _arg_value(flag: str) -> str | None:
    if flag not in sys.argv:
        return None

    idx = sys.argv.index(flag)

    if idx + 1 >= len(sys.argv):
        return None

    return sys.argv[idx + 1]


def _load_parent():
    parent_path = (
        Path(__file__).resolve().parent
        / "physics_v2_model.py"
    )

    if not parent_path.exists():
        raise FileNotFoundError(
            "Required Physics V2 core parent model not found:\n"
            f"  {parent_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "physics_v2_parent",
        parent_path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not import {parent_path}"
        )

    parent = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        parent
    )

    parent_build = str(
        getattr(
            parent,
            "BUILD",
            "",
        )
    )

    if EXPECTED_PARENT_TOKEN not in parent_build:
        raise RuntimeError(
            "Physics V2 requires the Physics V2 core "
            "real-A4-routing parent.\n"
            f"Current parent BUILD: {parent_build}\n"
            f"Expected token: {EXPECTED_PARENT_TOKEN}"
        )

    return (
        parent,
        parent_path,
        parent_build,
    )


def _normalize_sc_id(value):
    text = str(value).strip()

    if text.upper().startswith(
        "SC"
    ):
        try:
            return (
                f"SC{int(text[2:]):03d}"
            )
        except Exception:
            return text

    try:
        return (
            f"SC{int(float(text)):03d}"
        )
    except Exception:
        return text


def _load_deep_state(
    path: Path,
    expected_subcatchments: int,
):
    deep = pd.read_csv(
        path
    )

    required = {
        "interval_end_utc",
        "subcatchment_id",
        "deep_relative_wetness",
    }

    missing = (
        required
        - set(deep.columns)
    )

    if missing:
        raise RuntimeError(
            "Deep-state forcing missing columns: "
            f"{sorted(missing)}"
        )

    deep[
        "interval_end_utc"
    ] = pd.to_datetime(
        deep[
            "interval_end_utc"
        ],
        utc=True,
        errors="raise",
    )

    deep[
        "subcatchment_id"
    ] = deep[
        "subcatchment_id"
    ].map(
        _normalize_sc_id
    )

    deep[
        "deep_relative_wetness"
    ] = pd.to_numeric(
        deep[
            "deep_relative_wetness"
        ],
        errors="coerce",
    )

    duplicate_count = int(
        deep.duplicated(
            [
                "interval_end_utc",
                "subcatchment_id",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            "Deep-state forcing contains "
            f"{duplicate_count} duplicate "
            "time/subcatchment rows."
        )

    if (
        deep[
            "subcatchment_id"
        ].nunique()
        != expected_subcatchments
    ):
        raise RuntimeError(
            f"Expected {expected_subcatchments} "
            "deep-state subcatchments; found "
            f"{deep['subcatchment_id'].nunique()}."
        )

    bad = int(
        (
            ~np.isfinite(
                deep[
                    "deep_relative_wetness"
                ]
            )
            |
            (
                deep[
                    "deep_relative_wetness"
                ]
                < 0
            )
            |
            (
                deep[
                    "deep_relative_wetness"
                ]
                > 1
            )
        ).sum()
    )

    if bad:
        raise RuntimeError(
            f"Deep-state forcing contains {bad} "
            "invalid normalized wetness values."
        )

    # Hard guard against accidental Florence-state use.
    if (
        deep[
            "interval_end_utc"
        ].max()
        >= pd.Timestamp(
            "2018-01-01T00:00:00Z"
        )
    ):
        raise RuntimeError(
            "Physics V2 deep-state input contains 2018+ data."
        )

    return deep


def _make_build_event_blocks(
    parent,
    original_build_event_blocks,
    deep_state,
):
    """
    Wrap the parent event builder and attach one layer-4 deep-state vector at
    each event start. ERA5 layer-4 is NOT imposed hourly.
    """

    deep_by_time = {
        timestamp: group.set_index(
            "subcatchment_id"
        )
        for timestamp, group
        in deep_state.groupby(
            "interval_end_utc"
        )
    }

    def build_event_blocks(
        data,
        routing_model,
        warmup_h,
    ):
        events = original_build_event_blocks(
            data,
            routing_model,
            warmup_h,
        )

        ids = routing_model[
            "ids"
        ]

        for event in events:
            start = pd.Timestamp(
                event[
                    "times"
                ][0]
            )

            if start not in deep_by_time:
                raise RuntimeError(
                    f"{event['event_block_id']}: "
                    f"no layer-4 deep state at {start}."
                )

            group = deep_by_time[
                start
            ]

            missing_ids = [
                sid
                for sid in ids
                if sid not in group.index
            ]

            if missing_ids:
                raise RuntimeError(
                    f"{event['event_block_id']}: "
                    "deep state missing "
                    f"subcatchments {missing_ids}."
                )

            deep_vector = np.asarray(
                [
                    float(
                        group.loc[
                            sid,
                            "deep_relative_wetness",
                        ]
                    )
                    for sid in ids
                ],
                dtype=float,
            )

            if not np.isfinite(
                deep_vector
            ).all():
                raise RuntimeError(
                    f"{event['event_block_id']}: "
                    "non-finite initial deep state."
                )

            event[
                "deep_wetness_initial"
            ] = np.clip(
                deep_vector,
                0.0,
                1.0,
            )

        return events

    return build_event_blocks


def _make_unpack_params(
    parent_param_names,
):
    names = (
        list(
            parent_param_names
        )
        + NEW_PARAM_NAMES
    )

    def unpack_params(x):
        if len(x) != len(names):
            raise RuntimeError(
                f"Expected {len(names)} parameters; "
                f"received {len(x)}."
            )

        return dict(
            zip(
                names,
                map(
                    float,
                    x,
                ),
            )
        )

    return (
        names,
        unpack_params,
    )


def _make_simulate_event(
    parent,
):
    """
    Physics V2 core model equations with only the initial base/channel state changed.
    """

    def simulate_event(
        event,
        x,
        routing_model,
        return_states=False,
    ):
        p = parent.unpack_params(
            x
        )

        ids = routing_model[
            "ids"
        ]

        order = routing_model[
            "order"
        ]

        pos = routing_model[
            "position"
        ]

        upstream_of = routing_model[
            "upstream_of"
        ]

        outlet = routing_model[
            "outlet"
        ]

        n = len(
            ids
        )

        nt = len(
            event[
                "times"
            ]
        )

        if (
            "deep_wetness_initial"
            not in event
        ):
            raise RuntimeError(
                f"{event['event_block_id']}: "
                "deep_wetness_initial is missing."
            )

        area = np.asarray(
            [
                routing_model[
                    "area_km2"
                ][sid]
                for sid in ids
            ],
            dtype=float,
        )

        segment_km = np.asarray(
            [
                routing_model[
                    "segment_km"
                ][sid]
                for sid in ids
            ],
            dtype=float,
        )

        velocity = p[
            "channel_velocity_m_s"
        ]

        channel_k_h = np.maximum(
            (
                segment_km
                * 1000.0
                / velocity
                / 3600.0
            ),
            0.25,
        )

        soil_capacity = p[
            "soil_capacity_mm"
        ]

        field_capacity_fraction = p[
            "field_capacity_fraction"
        ]

        runoff_beta = p[
            "runoff_beta"
        ]

        soil_drainage_time = p[
            "soil_drainage_time_h"
        ]

        quick_k = p[
            "quick_reservoir_time_h"
        ]

        base_k = p[
            "base_reservoir_time_h"
        ]

        et_multiplier = p[
            "et_multiplier"
        ]

        et_exponent = p[
            "et_moisture_exponent"
        ]

        deep_floor = p[
            "deep_baseflow_floor_mm_h"
        ]

        deep_range = p[
            "deep_baseflow_range_mm_h"
        ]

        # ---------------------------------------------------------------
        # Root-zone event initialization: unchanged from Physics V2 core.
        # ---------------------------------------------------------------
        initial_root_wetness = np.clip(
            event[
                "era5_wetness"
            ][0],
            0.0,
            1.0,
        )

        soil = (
            initial_root_wetness
            * soil_capacity
        )

        quick_store = np.zeros(
            n,
            dtype=float,
        )

        # ---------------------------------------------------------------
        # NEW Physics V2 slow-flow initialization.
        # Layer-4 deep wetness is independent of the root-zone store.
        # ---------------------------------------------------------------
        deep_wetness = np.clip(
            np.asarray(
                event[
                    "deep_wetness_initial"
                ],
                dtype=float,
            ),
            0.0,
            1.0,
        )

        initial_base_flux_mm_h = (
            deep_floor
            + deep_range
            * deep_wetness
        )

        # Local base reservoir stores depth [mm].
        #
        # reservoir_step() releases:
        #   out = (store + current_input) * alpha
        #
        # Initializing store as target_flux/alpha makes the first release
        # approximately the deep-state target flux before considering the
        # current hour's recharge.
        base_alpha = (
            1.0
            - math.exp(
                -1.0
                / max(
                    base_k,
                    1e-6,
                )
            )
        )

        base_store = (
            initial_base_flux_mm_h
            / max(
                base_alpha,
                1e-12,
            )
        )

        threshold = (
            field_capacity_fraction
            * soil_capacity
        )

        # Conversion:
        #   1 mm/h over 1 km2
        # = 0.001 m * 1e6 m2 / 3600 s
        # = 0.277777... m3/s
        mmh_km2_to_m3s = (
            area
            * 1000.0
            / 3600.0
        )

        initial_local_base_q = (
            initial_base_flux_mm_h
            * mmh_km2_to_m3s
        )

        # ---------------------------------------------------------------
        # NEW Physics V2 channel-state initialization.
        #
        # The event begins with groundwater-supported flow already present
        # throughout the river network. Starting every channel store at zero
        # would force that antecedent flow to propagate from scratch.
        #
        # For the channel update:
        #   S += I*3600
        #   O = alpha*S
        #
        # steady state with O = I requires:
        #   S_before_input = I*3600*(1/alpha - 1)
        # ---------------------------------------------------------------
        initial_routed_q = np.zeros(
            n,
            dtype=float,
        )

        channel_store_m3 = np.zeros(
            n,
            dtype=float,
        )

        order_idx = [
            pos[
                sid
            ]
            for sid in order
        ]

        for i in order_idx:
            sid = ids[
                i
            ]

            upstream_q = sum(
                initial_routed_q[
                    pos[
                        up_sid
                    ]
                ]
                for up_sid
                in upstream_of[
                    sid
                ]
            )

            steady_inflow_q = (
                initial_local_base_q[
                    i
                ]
                + upstream_q
            )

            channel_alpha = (
                1.0
                - math.exp(
                    -1.0
                    / max(
                        channel_k_h[
                            i
                        ],
                        1e-6,
                    )
                )
            )

            channel_store_m3[
                i
            ] = (
                steady_inflow_q
                * 3600.0
                * (
                    1.0
                    / max(
                        channel_alpha,
                        1e-12,
                    )
                    - 1.0
                )
            )

            initial_routed_q[
                i
            ] = (
                steady_inflow_q
            )

        outlet_idx = pos[
            outlet
        ]

        qout = np.zeros(
            nt,
            dtype=float,
        )

        if return_states:
            soil_hist = np.zeros(
                (
                    nt,
                    n,
                ),
                dtype=float,
            )

            aet_hist = np.zeros(
                (
                    nt,
                    n,
                ),
                dtype=float,
            )

            quick_hist = np.zeros(
                (
                    nt,
                    n,
                ),
                dtype=float,
            )

            recharge_hist = np.zeros(
                (
                    nt,
                    n,
                ),
                dtype=float,
            )

            base_flux_hist = np.zeros(
                (
                    nt,
                    n,
                ),
                dtype=float,
            )

        # ---------------------------------------------------------------
        # Time integration: same equations as Physics V2 core.
        # ---------------------------------------------------------------
        for t in range(
            nt
        ):
            rainfall = np.maximum(
                event[
                    "rain"
                ][t],
                0.0,
            )

            et0 = np.maximum(
                event[
                    "et0"
                ][t],
                0.0,
            )

            relative_soil = np.clip(
                soil
                / soil_capacity,
                0.0,
                1.0,
            )

            moisture_limitation = np.power(
                relative_soil,
                et_exponent,
            )

            aet = np.minimum(
                soil,
                (
                    et0
                    * et_multiplier
                    * moisture_limitation
                ),
            )

            soil = np.maximum(
                soil
                - aet,
                0.0,
            )

            relative_soil = np.clip(
                soil
                / soil_capacity,
                0.0,
                1.0,
            )

            quick_fraction = np.power(
                relative_soil,
                runoff_beta,
            )

            quick_input = (
                rainfall
                * quick_fraction
            )

            infiltration = (
                rainfall
                - quick_input
            )

            soil = (
                soil
                + infiltration
            )

            overflow = np.maximum(
                soil
                - soil_capacity,
                0.0,
            )

            soil = np.minimum(
                soil,
                soil_capacity,
            )

            quick_input = (
                quick_input
                + overflow
            )

            recharge = np.maximum(
                soil
                - threshold,
                0.0,
            ) / soil_drainage_time

            recharge = np.minimum(
                recharge,
                soil,
            )

            soil = np.maximum(
                soil
                - recharge,
                0.0,
            )

            local_q_m3s = np.zeros(
                n,
                dtype=float,
            )

            base_out_depth = np.zeros(
                n,
                dtype=float,
            )

            for i in range(
                n
            ):
                (
                    quick_store[
                        i
                    ],
                    quick_out,
                ) = parent.reservoir_step(
                    quick_store[
                        i
                    ],
                    quick_input[
                        i
                    ],
                    quick_k,
                )

                (
                    base_store[
                        i
                    ],
                    base_out,
                ) = parent.reservoir_step(
                    base_store[
                        i
                    ],
                    recharge[
                        i
                    ],
                    base_k,
                )

                base_out_depth[
                    i
                ] = (
                    base_out
                )

                local_q_m3s[
                    i
                ] = (
                    (
                        quick_out
                        + base_out
                    )
                    * mmh_km2_to_m3s[
                        i
                    ]
                )

            routed_out = np.zeros(
                n,
                dtype=float,
            )

            for i in order_idx:
                sid = ids[
                    i
                ]

                upstream_q = sum(
                    routed_out[
                        pos[
                            up_sid
                        ]
                    ]
                    for up_sid
                    in upstream_of[
                        sid
                    ]
                )

                inflow_q = (
                    local_q_m3s[
                        i
                    ]
                    + upstream_q
                )

                channel_store_m3[
                    i
                ] += (
                    inflow_q
                    * 3600.0
                )

                channel_alpha = (
                    1.0
                    - math.exp(
                        -1.0
                        / max(
                            channel_k_h[
                                i
                            ],
                            1e-6,
                        )
                    )
                )

                released_m3 = (
                    channel_store_m3[
                        i
                    ]
                    * channel_alpha
                )

                channel_store_m3[
                    i
                ] -= (
                    released_m3
                )

                routed_out[
                    i
                ] = (
                    released_m3
                    / 3600.0
                )

            qout[
                t
            ] = routed_out[
                outlet_idx
            ]

            if return_states:
                soil_hist[
                    t
                ] = soil

                aet_hist[
                    t
                ] = aet

                quick_hist[
                    t
                ] = quick_input

                recharge_hist[
                    t
                ] = recharge

                base_flux_hist[
                    t
                ] = base_out_depth

        result = {
            "q_sim_m3s": qout,
        }

        if return_states:
            result.update(
                {
                    "soil_mm": (
                        soil_hist
                    ),
                    "aet_mm_h": (
                        aet_hist
                    ),
                    "quick_input_mm_h": (
                        quick_hist
                    ),
                    "recharge_mm_h": (
                        recharge_hist
                    ),
                    "base_out_mm_h": (
                        base_flux_hist
                    ),
                    "initial_deep_wetness": (
                        deep_wetness
                    ),
                    "initial_base_flux_mm_h": (
                        initial_base_flux_mm_h
                    ),
                    "initial_routed_base_q_m3s": (
                        initial_routed_q
                    ),
                }
            )

        return result

    return simulate_event


def _create_aliases(
    output_dir: Path,
    parent_build: str,
    exit_code: int,
):
    """Augment the core-model metadata with final deep-state configuration."""
    metadata_path = output_dir / "physics_v2_metadata.json"
    metadata = {}

    if metadata_path.exists():
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except Exception:
            metadata = {}

    metadata.update({
        "script_build": BUILD,
        "parent_build": parent_build,
        "model_variant": "final_deep_state_initialization",
        "deep_baseflow_mapping": {
            "equation": (
                "initial_baseflow_mm_h = "
                "deep_baseflow_floor_mm_h + "
                "deep_baseflow_range_mm_h * deep_relative_wetness"
            ),
            "parameters": NEW_PARAM_NAMES,
            "observed_q_used_for_initial_state": False,
        },
        "development_protocol": {
            "calibration": "2015-2016",
            "validation": "2017",
            "florence_used": False,
        },
        "parent_exit_code": exit_code,
    })

    tmp = metadata_path.with_name(metadata_path.name + ".partial")
    tmp.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, metadata_path)
    return metadata_path

def main():
    deep_state_path = Path(
        _extract_custom_arg(
            "--deep-state"
        )
    )

    parent, parent_path, parent_build = (
        _load_parent()
    )

    # Read expected count from the parent CLI if supplied, otherwise use
    # the study's established 37 units.
    expected_subcatchments = 37

    if (
        "--expected-subcatchments"
        in sys.argv
    ):
        idx = sys.argv.index(
            "--expected-subcatchments"
        )

        expected_subcatchments = int(
            sys.argv[
                idx + 1
            ]
        )

    deep_state = _load_deep_state(
        deep_state_path,
        expected_subcatchments,
    )

    original_build_event_blocks = (
        parent.build_event_blocks
    )

    parent.build_event_blocks = (
        _make_build_event_blocks(
            parent,
            original_build_event_blocks,
            deep_state,
        )
    )

    original_param_names = list(
        parent.PARAM_NAMES
    )

    original_param_bounds = list(
        parent.PARAM_BOUNDS
    )

    (
        extended_names,
        extended_unpack,
    ) = _make_unpack_params(
        original_param_names
    )

    parent.PARAM_NAMES = (
        extended_names
    )

    parent.PARAM_BOUNDS = (
        original_param_bounds
        + NEW_PARAM_BOUNDS
    )

    parent.unpack_params = (
        extended_unpack
    )

    parent.simulate_event = (
        _make_simulate_event(
            parent
        )
    )

    parent.BUILD = BUILD

    output_dir_value = _arg_value(
        "--output-dir"
    )

    if output_dir_value is None:
        raise RuntimeError(
            "--output-dir is required."
        )

    output_dir = Path(
        output_dir_value
    )

    print("=" * 100)
    print(
        f"PHYSICS V2 TRAIN BUILD               : {BUILD}"
    )
    print(
        f"Parent model                       : {parent_build}"
    )
    print(
        f"Parent script                      : {parent_path}"
    )
    print(
        f"Deep-state forcing                 : {deep_state_path}"
    )
    print(
        "Hydrologic equations changed       : "
        "ONLY INITIAL SLOW-FLOW STATE"
    )
    print(
        "Calibration objective changed      : "
        "YES - EVENT BALANCED / SMALL POOLED REGULARIZER"
    )
    print(
        "Peak timing diagnostic changed     : "
        "YES - EXACT TIMESTAMPS / NO +/-96 H CLIP"
    )
    print(
        "Runoff partition changed           : NO"
    )
    print(
        "Root-zone initialization changed   : NO"
    )
    print(
        "ET0 forcing changed                : NO"
    )
    print(
        "A4 routing distances changed       : NO"
    )
    print(
        "Calibration/validation split changed: NO"
    )
    print(
        "Observed Q used for initialization : NO"
    )
    print(
        "Florence 2018 used                 : NO"
    )
    print(
        "New calibrated parameters          : "
        "2"
    )
    print(
        "  deep_baseflow_floor_mm_h         : "
        f"{NEW_PARAM_BOUNDS[0]}"
    )
    print(
        "  deep_baseflow_range_mm_h         : "
        f"{NEW_PARAM_BOUNDS[1]}"
    )
    print("=" * 100)
    print()

    exit_code = 0

    try:
        parent.main()

    except SystemExit as exc:
        exit_code = (
            0
            if exc.code is None
            else int(
                exc.code
            )
        )

    finally:
        metadata_path = (
            _create_aliases(
                output_dir,
                parent_build,
                exit_code,
            )
        )

        print()
        print(
            "PHYSICS V2 OUTPUT"
        )
        print(
            "-" * 100
        )
        print(
            f"Physics V2 metadata                    : {metadata_path}"
        )

        parameters_path = (
            output_dir
            / "physics_v2_parameters.json"
        )

        if parameters_path.exists():
            try:
                payload = json.loads(
                    parameters_path.read_text(
                        encoding="utf-8"
                    )
                )

                params = payload.get(
                    "parameters",
                    {}
                )

                print(
                    f"Deep baseflow floor                : "
                    f"{params.get('deep_baseflow_floor_mm_h')} mm/h"
                )
                print(
                    f"Deep baseflow range                : "
                    f"{params.get('deep_baseflow_range_mm_h')} mm/h"
                )

            except Exception:
                pass

    if exit_code != 0:
        raise SystemExit(
            exit_code
        )


if __name__ == "__main__":
    main()
