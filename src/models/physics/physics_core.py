#!/usr/bin/env python3
"""
Physics V3.5 — moisture-dependent preferential interflow runtime.

Only structural change relative to retained Physics V3.1:
----------------------------------------------------------
V3.1:
    interflow_input = infiltration * f_pref

V3.5:
    dryness = clip(1 - S/C, 0, 1)
    f_pref_effective = f_pref_max * dryness
    interflow_input = infiltration * f_pref_effective

So preferential bypass is strongest in dry soil and progressively suppressed
as the soil becomes wetter.

Why:
----
Robust V3.1R calibration repaired the dry-regime event but pushed the constant
preferential-interflow fraction to its upper bound and then overpredicted a
wetter validation event. A constant bypass is therefore too rigid across
moisture regimes.

No new parameter is added. The existing preferential_interflow_fraction is
reinterpreted as a MAXIMUM dry-state fraction.

The final September-4 run was explicitly labelled CONSTRAINED interflow.
The preserved run history does not contain the final source line for this
bound, but the immediately preceding constrained lineage used 0.075 and the
final optimum (0.065875) is consistent with that cap.  This recovery therefore
uses 0.075 and records that provenance in RECOVERY_NOTES_TIMING_AWARE.txt.

All other retained equations/state/routing behavior are unchanged.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path


V3_BUILD = "PHYSICS_CONSTRAINED_INTERFLOW_TIMING_AWARE"

V3_PARAM_NAMES = [
    "preferential_interflow_fraction",
    "interflow_reservoir_time_h",
]

# f_pref is a maximum dry-state fraction.  The actual hourly fraction is
# f_pref * (1 - relative_soil).
#
# IMPORTANT RECOVERY NOTE:
# The September-4 final source line containing this bound was not preserved.
# The final run changed from MOISTURE_DEPENDENT_INTERFLOW_TIMING_AWARE to
# CONSTRAINED_INTERFLOW_TIMING_AWARE, after an unconstrained optimum of 0.145161.
# The established constrained lineage used 0.075, and the final optimum was
# 0.065875.  We therefore recover the constrained cap as 0.075.
V3_PARAM_BOUNDS = [
    (0.000, 0.075),
    (24.0, 120.0),
]

# Final September-4 calibration-only peak-timing regularizer.
# The constants below are exactly preserved in the retained training audit.
# The original source body of the penalty was not preserved; the function
# below is a conservative reconstruction from those printed constants and
# the retained exact-timestamp/boundary diagnostics.  See recovery notes.
TIMING_OBJECTIVE_WEIGHT = 0.200
TIMING_TOLERANCE_H = 12.0
TIMING_SCALE_H = 60.0
TIMING_MAX_PENALTY = 2.0
SCORING_BOUNDARY_PEAK_PENALTY = 0.5


def timing_penalty_from_rows(rows):
    """Return the mean calibration peak-timing penalty.

    Reconstructed interpretation of the preserved September-4 settings:
    exact peak offsets are used (no +/-96 h clipping); a 12-hour tolerance is
    free; excess absolute timing error is scaled by 60 hours and capped at
    2.0; a simulated peak on the scored-window boundary adds 0.5.

    The constants are exact from the audit.  The algebraic combination is a
    best-evidence reconstruction because the final source body is unavailable.
    """
    import numpy as np

    penalties = []
    for row in rows.to_dict("records"):
        err = row.get("peak_timing_error_h")
        if err is None or not np.isfinite(float(err)):
            penalties.append(TIMING_MAX_PENALTY + SCORING_BOUNDARY_PEAK_PENALTY)
            continue

        excess = max(abs(float(err)) - TIMING_TOLERANCE_H, 0.0)
        penalty = min(excess / TIMING_SCALE_H, TIMING_MAX_PENALTY)
        if bool(row.get("simulated_peak_at_scoring_boundary", False)):
            penalty += SCORING_BOUNDARY_PEAK_PENALTY
        penalties.append(float(penalty))

    if not penalties:
        return TIMING_MAX_PENALTY
    return float(np.mean(penalties))


def _install_timing_objective_on_parent(parent):
    """Patch the core parent's calibration objective with retained timing term."""
    import numpy as np

    def objective_factory(
        cal_events,
        routing_model,
        trace,
        *,
        block_weight,
        pooled_weight,
        metric_weights,
    ):
        counter = {"n": 0}
        total_mix = float(block_weight + pooled_weight)
        if not np.isfinite(total_mix) or total_mix <= 0:
            raise ValueError(
                "Objective block/pooled weights must sum to a positive value."
            )
        bw = float(block_weight) / total_mix
        pw = float(pooled_weight) / total_mix

        def objective(x):
            counter["n"] += 1
            hydrologic_value = 1e6
            timing_value = TIMING_MAX_PENALTY
            value = 1e6
            try:
                rows, pooled = parent.evaluate_events(cal_events, x, routing_model)
                block_penalties = [
                    parent.metric_penalty(r, metric_weights)
                    for r in rows.to_dict("records")
                ]
                block_mean = float(np.mean(block_penalties))
                pooled_penalty = parent.metric_penalty(pooled, metric_weights)
                hydrologic_value = bw * block_mean + pw * pooled_penalty
                timing_value = timing_penalty_from_rows(rows)
                value = hydrologic_value + TIMING_OBJECTIVE_WEIGHT * timing_value
                if not np.isfinite(value):
                    value = 1e6
            except Exception:
                value = 1e6

            trace.append({
                "evaluation": counter["n"],
                "objective": float(value),
                "hydrologic_objective": float(hydrologic_value),
                "calibration_peak_timing_penalty": float(timing_value),
                "timing_objective_weight": TIMING_OBJECTIVE_WEIGHT,
            })
            return float(value)

        return objective

    parent.objective_factory = objective_factory


def _install_final_parent_loader(ns):
    """Ensure the transformed deep-state runtime installs the timing objective."""
    original_load_parent = ns["_load_parent"]

    def load_parent():
        parent, parent_path, parent_build = original_load_parent()
        _install_timing_objective_on_parent(parent)
        # The September-4 public build reported this generic parent label.
        public_parent_build = "PHYSICS_SEMIDISTRIBUTED_ERA5_ROUTING"
        return parent, parent_path, public_parent_build

    ns["_load_parent"] = load_parent


def _target_name(target):
    return target.id if isinstance(target, ast.Name) else None


def _is_subscript_of(target, name):
    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == name
    )


def _contains_name(node, name):
    return any(
        isinstance(x, ast.Name) and x.id == name
        for x in ast.walk(node)
    )


def _contains_quick_out_target(target):
    return (
        _contains_name(target, "quick_out")
        and _contains_name(target, "quick_store")
    )


class _SimulationPatcher:
    def __init__(self):
        self.counts = {
            "params": 0,
            "state": 0,
            "dynamic_interflow": 0,
            "interflow_reservoir": 0,
            "local_q": 0,
        }

    @staticmethod
    def _parsed(code):
        return ast.parse(code).body

    def process_block(self, body):
        out = []

        for stmt in body:
            # Recurse first.
            for attr in ("body", "orelse", "finalbody"):
                child = getattr(stmt, attr, None)
                if isinstance(child, list):
                    setattr(stmt, attr, self.process_block(child))

            if isinstance(stmt, ast.Try):
                for h in stmt.handlers:
                    h.body = self.process_block(h.body)

            # Extract V3.5 parameters.
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and _target_name(stmt.targets[0]) == "deep_range"
            ):
                out.append(stmt)
                out.extend(
                    self._parsed(
                        'preferential_interflow_fraction = '
                        'p["preferential_interflow_fraction"]\n'
                        'interflow_reservoir_time_h = '
                        'p["interflow_reservoir_time_h"]\n'
                    )
                )
                self.counts["params"] += 1
                continue

            # Empty interflow reservoir, same as V3.1.
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and _target_name(stmt.targets[0]) == "quick_store"
            ):
                out.append(stmt)
                out.extend(
                    self._parsed(
                        "interflow_store = np.zeros(n, dtype=float)\n"
                    )
                )
                self.counts["state"] += 1
                continue

            # Replace constant preferential fraction with moisture dependence.
            #
            # At this point in the parent model:
            #   relative_soil = soil / soil_capacity
            # has already been recomputed after ET and before rainfall
            # partitioning. It is therefore the pre-rainfall soil state.
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and _target_name(stmt.targets[0]) == "infiltration"
            ):
                out.append(stmt)
                out.extend(
                    self._parsed(
                        "preferential_dryness_factor = np.clip(\n"
                        "    1.0 - relative_soil,\n"
                        "    0.0,\n"
                        "    1.0,\n"
                        ")\n"
                        "preferential_interflow_fraction_effective = (\n"
                        "    preferential_interflow_fraction\n"
                        "    * preferential_dryness_factor\n"
                        ")\n"
                        "interflow_input = (\n"
                        "    infiltration\n"
                        "    * preferential_interflow_fraction_effective\n"
                        ")\n"
                        "infiltration = infiltration - interflow_input\n"
                    )
                )
                self.counts["dynamic_interflow"] += 1
                continue

            # Same delayed interflow reservoir as V3.1.
            if isinstance(stmt, ast.Assign):
                if any(
                    _contains_quick_out_target(t)
                    for t in stmt.targets
                ):
                    out.append(stmt)
                    out.extend(
                        self._parsed(
                            "interflow_store[i], interflow_out = "
                            "parent.reservoir_step(\n"
                            "    interflow_store[i],\n"
                            "    interflow_input[i],\n"
                            "    interflow_reservoir_time_h,\n"
                            ")\n"
                        )
                    )
                    self.counts["interflow_reservoir"] += 1
                    continue

            # Same local discharge addition as V3.1.
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and _is_subscript_of(stmt.targets[0], "local_q_m3s")
            ):
                idx = copy.deepcopy(stmt.targets[0].slice)

                extra = ast.BinOp(
                    left=ast.Name(
                        id="interflow_out",
                        ctx=ast.Load(),
                    ),
                    op=ast.Mult(),
                    right=ast.Subscript(
                        value=ast.Name(
                            id="mmh_km2_to_m3s",
                            ctx=ast.Load(),
                        ),
                        slice=idx,
                        ctx=ast.Load(),
                    ),
                )

                stmt.value = ast.BinOp(
                    left=stmt.value,
                    op=ast.Add(),
                    right=extra,
                )

                self.counts["local_q"] += 1
                out.append(stmt)
                continue

            out.append(stmt)

        return out


class _ModulePatcher(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.sim_patcher = _SimulationPatcher()
        self.found_build = False
        self.found_names = False
        self.found_bounds = False
        self.found_factory = False

    def visit_Assign(self, node):
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "BUILD"
        ):
            node.value = ast.Constant(V3_BUILD)
            self.found_build = True
            return node

        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "NEW_PARAM_NAMES"
            and isinstance(node.value, ast.List)
        ):
            existing = [
                e.value
                for e in node.value.elts
                if isinstance(e, ast.Constant)
                and isinstance(e.value, str)
            ]

            for name in V3_PARAM_NAMES:
                if name not in existing:
                    node.value.elts.append(ast.Constant(name))

            self.found_names = True
            return node

        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "NEW_PARAM_BOUNDS"
            and isinstance(node.value, ast.List)
        ):
            for lo, hi in V3_PARAM_BOUNDS:
                node.value.elts.append(
                    ast.Tuple(
                        elts=[
                            ast.Constant(float(lo)),
                            ast.Constant(float(hi)),
                        ],
                        ctx=ast.Load(),
                    )
                )

            self.found_bounds = True
            return node

        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "_make_simulate_event":
            self.found_factory = True
            node.body = self.sim_patcher.process_block(node.body)
            return node

        return self.generic_visit(node)


def build_transformed_tree(v2_trainer_path: Path):
    source = v2_trainer_path.read_text(encoding="utf-8")

    tree = ast.parse(
        source,
        filename=str(v2_trainer_path),
    )

    patcher = _ModulePatcher()
    tree = patcher.visit(tree)
    ast.fix_missing_locations(tree)

    required = {
        "BUILD": patcher.found_build,
        "NEW_PARAM_NAMES": patcher.found_names,
        "NEW_PARAM_BOUNDS": patcher.found_bounds,
        "_make_simulate_event": patcher.found_factory,
    }

    missing = [
        key
        for key, ok in required.items()
        if not ok
    ]

    if missing:
        raise RuntimeError(
            "Physics V3.5 could not locate required final-V2 "
            f"trainer structures: {missing}"
        )

    counts = patcher.sim_patcher.counts

    bad = {
        key: value
        for key, value in counts.items()
        if value != 1
    }

    if bad:
        raise RuntimeError(
            "Physics V3.5 expected exactly one structural patch "
            f"for every anchor; got {counts}. "
            "No training was started."
        )

    transformed = ast.unparse(tree)

    required_snippets = [
        "preferential_dryness_factor",
        "1.0 - relative_soil",
        "preferential_interflow_fraction_effective",
        "interflow_input",
    ]

    absent = [
        text
        for text in required_snippets
        if text not in transformed
    ]

    if absent:
        raise RuntimeError(
            "Physics V3.5 semantic preflight failed. "
            f"Missing transformed code markers: {absent}"
        )

    compile(
        tree,
        str(v2_trainer_path),
        "exec",
    )

    return tree, counts


def load_v3_5_namespace():
    here = Path(__file__).resolve().parent
    v2_path = here / "_physics_deep_state_parent.py"

    if not v2_path.exists():
        raise FileNotFoundError(
            "Required existing final Physics V2 trainer not found:\n"
            f"  {v2_path}"
        )

    tree, counts = build_transformed_tree(v2_path)

    ns = {
        "__name__": "physics_v3_5_moisture_interflow_transformed",
        "__file__": str(v2_path),
        "__package__": None,
    }

    exec(
        compile(
            tree,
            str(v2_path),
            "exec",
        ),
        ns,
    )

    _install_final_parent_loader(ns)
    ns["BUILD"] = V3_BUILD

    return ns, counts
