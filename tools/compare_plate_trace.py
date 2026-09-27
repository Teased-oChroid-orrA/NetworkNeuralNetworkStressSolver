#!/usr/bin/env python3
"""Compare a plate audit trace with the same finite-domain FEM problem.

Usage: python tools/compare_plate_trace.py TRACE_DIR --levels 48x6x0.8,64x8x0.6,96x12x0.4
Creates new ``fem_reference.json`` and ``fem_comparison.json`` in TRACE_DIR. Existing
files are never overwritten. Supports the current plane-stress, uniaxial-x plate FEM;
unsupported loads fail closed rather than producing an unlike-for-like comparison.
"""
import argparse
import json
import math
import pathlib
import tomllib

import numpy as np

try:
    from .multi_hole_reference import solve
except ImportError:  # direct script invocation
    from multi_hole_reference import solve


def profile_metrics(reference, candidate):
    """Scale-free full-field comparison, retaining peak and angular-shape evidence."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 1 or not len(reference):
        raise ValueError("profiles must be nonempty, one-dimensional, and angle-aligned")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("profiles must be finite")
    scale = float(np.sqrt(np.mean(reference ** 2)))
    if scale <= 0:
        raise ValueError("reference profile has zero RMS")
    correlation = float(np.corrcoef(reference, candidate)[0, 1])
    return {
        "peak_relative_error": float(abs(candidate.max() - reference.max()) / abs(reference.max())),
        "rms_error_over_reference_rms": float(np.sqrt(np.mean((candidate - reference) ** 2)) / scale),
        "correlation": correlation if math.isfinite(correlation) else None,
    }


def reference_quality(levels):
    """Fail closed unless recovered von-Mises and stress components stabilize and agree with FD."""
    failures = []
    if len(levels) < 3:
        failures.append("at least three refinement levels required")
    if not levels:
        return {"passed": False, "failures": failures}
    for i, hole in enumerate(levels[-1]["holes"]):
        # CST displacement gradients and element stress jump across mesh edges.
        # Their fixed-location profiles oscillate as ring positions cross the probe.
        # Recovery must stabilize; independent FD stress must agree at this level.
        change = hole["recovered"].get("fem_profile_change_over_current_rms")
        if change is None or not math.isfinite(change) or change > 0.02:
            failures.append(f"hole {i}: recovered full-profile mesh change exceeds 2% or is absent")
        disagreement = hole["fd_vs_recovered_rms_over_recovered_rms"]
        if not math.isfinite(disagreement) or disagreement > 0.03:
            failures.append(f"hole {i}: FD/recovered stress disagreement exceeds 3%")
        for component in ("sxx", "syy", "sxy"):
            quality = hole.get("reference_components", {}).get(component, {})
            change = quality.get("fem_profile_change_over_current_rms")
            disagreement = quality.get("fd_vs_recovered_rms_over_recovered_rms")
            if change is None or not math.isfinite(change) or change > 0.02:
                failures.append(f"hole {i}: {component} FEM full-profile mesh change exceeds 2% or is absent")
            if disagreement is None or not math.isfinite(disagreement) or disagreement > 0.03:
                failures.append(f"hole {i}: {component} FD/recovered disagreement exceeds 3% or is absent")
    return {"passed": not failures, "failures": failures}


def physical_acceptance(spec, update, comparison, companion=None, run_summary=None):
    """Run-specific plate gate. No signed objective or legacy mixed-unit RMS enters it."""
    failures = list(comparison["reference_quality"]["failures"])
    if (run_summary is None or not run_summary.get("completed")
            or not run_summary.get("worker_ok") or run_summary.get("nonfinite")):
        failures.append("trace lacks a completed finite run summary")
    if (companion is None or not companion.get("passed") or not companion.get("l0_passed")
            or companion.get("operational_status") != "PASS"):
        failures.append("matched no-hole L0/L4 companion is absent or failed")
    complete_profile = all(all(key in point and math.isfinite(point[key]) for key in
                               ("x", "y", "ux", "uy", "sxx", "syy", "sxy", "von_mises"))
                           for hole in update["holes"] for point in hole["profile"])
    if not complete_profile:
        failures.append("trace lacks complete coordinates, displacement, or stress")
    energy = update.get("energy_balance")
    if not energy or not math.isfinite(energy.get("energy_balance_error", math.nan)) or energy["energy_balance_error"] > 0.02:
        failures.append("energy-balance error exceeds 2% or is absent")
    fem_energy_error = comparison.get("energy_vs_fem_relative_error")
    if fem_energy_error is None or not math.isfinite(fem_energy_error) or fem_energy_error > 0.02:
        failures.append("internal energy differs >2% from matched FEM or is absent")
    force = update.get("reaction_force")
    if not force or not math.isfinite(force.get("equilibrium_error", math.nan)) or force["equilibrium_error"] > 0.02:
        failures.append("force-balance error exceeds 2% or is absent")
    boundaries = update.get("physical_boundary_residuals")
    load = spec["load"]["px"]
    if not boundaries or not math.isfinite(boundaries.get("outer_traction_pa", {}).get("rms", math.nan)) or boundaries["outer_traction_pa"]["rms"] / load > 0.02:
        failures.append("outer traction RMS exceeds 2% of applied stress or is absent")
    if boundaries:
        free = dict(boundaries.get("free_hole_traction_pa", []))
        fixed = dict(boundaries.get("fixed_hole_displacement_m", []))
        for i, hole in enumerate(spec["geometry"]["holes"]):
            if hole["bc"].lower() == "free":
                reading = free.get(i)
                if reading is None or not math.isfinite(reading.get("rms", math.nan)):
                    failures.append(f"hole {i}: offset-ring traction reading absent")
            else:
                reading = fixed.get(i)
                # E*displacement/offset is a local stress-sensitivity scale, not a
                # plate-wide displacement percentage. Fixed support needs this gate.
                probe_offset = 4 * spec["training"]["fd_h"] * max(
                    spec["geometry"]["half_w"], spec["geometry"]["half_h"])
                if reading is None or not math.isfinite(reading.get("rms", math.nan)) or spec["material"]["e"] * reading["rms"] / probe_offset / load > 0.05:
                    failures.append(f"hole {i}: Fixed displacement implies >5% local stress scale or is absent")
    if complete_profile and comparison["reference_quality"]["passed"]:
        finest = comparison["levels"][-1]
        for i, hole in enumerate(finest["holes"]):
            result = hole["recovered"]
            if result["peak_relative_error"] > 0.05 or result["rms_error_over_reference_rms"] > 0.05:
                failures.append(f"hole {i}: full von-Mises profile or peak differs >5% from FEM")
            for component in ("sxx", "syy", "sxy"):
                if hole["stress_components"][component]["rms_error_over_reference_rms"] > 0.05:
                    failures.append(f"hole {i}: {component} full profile differs >5% from FEM")
            if spec["geometry"]["holes"][i]["bc"].lower() == "free":
                if hole["offset_traction_rms_difference_over_load"] > 0.05:
                    failures.append(f"hole {i}: matched offset-ring traction differs >5% of applied stress")
    stationarity = comparison.get("training_stationarity")
    if stationarity is None:
        failures.append("100-step whole-profile training trend is unavailable")
    else:
        for i, change in enumerate(stationarity):
            if not math.isfinite(change) or change > 0.03:
                failures.append(f"hole {i}: full profile changed >3% over last 100 steps")
    return {"passed": not failures, "failures": failures}


def last_hole_update(path):
    last = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("holes"):
                last = row
    if last is None:
        raise ValueError("trace has no hole-profile update")
    return last


def hole_profile_stationarity(path, latest, window=100):
    """Measure field change over real training updates, not different physical radii."""
    earlier = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("holes") and row["step"] <= latest["step"] - window:
                earlier = row
    if earlier is None or len(earlier["holes"]) != len(latest["holes"]):
        return None
    return [profile_metrics([p["von_mises"] for p in recent["profile"]],
                            [p["von_mises"] for p in old["profile"]])[
                                "rms_error_over_reference_rms"]
            for recent, old in zip(latest["holes"], earlier["holes"])]


def compare(trace_dir, levels, companion_dir=None):
    spec = tomllib.loads((trace_dir / "effective.toml").read_text())
    geometry, material, load, training = (spec[key] for key in ("geometry", "material", "load", "training"))
    if load["px"] <= 0 or load["py"] != 0:
        raise ValueError("FEM comparator currently supports positive uniaxial x traction only")
    holes = [(h["center"][0], h["center"][1], h["radius"]) for h in geometry["holes"]]
    bcs = [h["bc"].lower() for h in geometry["holes"]]
    if any(bc not in ("free", "fixed") for bc in bcs):
        raise ValueError("FEM comparator supports only Free and Fixed circular holes")
    half_w, half_h, fd_h = geometry["half_w"], geometry["half_h"], training["fd_h"]
    fd_step = (fd_h * half_w, fd_h * half_h)
    radii = [r + 4 * fd_h * max(half_w, half_h) for _, _, r in holes]
    update = last_hole_update(trace_dir / "updates.jsonl")
    if len(update["holes"]) != len(holes):
        raise ValueError("trace and specification have different hole counts")
    n_angles = len(update["holes"][0]["profile"])
    if n_angles < 12 or any(len(h["profile"]) != n_angles for h in update["holes"]):
        raise ValueError("trace requires equal-length profiles with at least 12 angles")
    expected_angles = np.arange(n_angles) * 360 / n_angles
    for i, hole in enumerate(update["holes"]):
        if hole["index"] != i:
            raise ValueError("trace hole order differs from specification")
        angles = np.array([p["theta_deg"] for p in hole["profile"]])
        if not np.allclose(angles, expected_angles, rtol=0, atol=1e-8):
            raise ValueError("trace angles differ from FEM comparison angles")
        for point, angle in zip(hole["profile"], expected_angles):
            if "x" not in point or "y" not in point:
                continue  # historical traces remain comparable, but cannot be accepted
            theta = math.radians(angle)
            expected = (holes[i][0] + radii[i] * math.cos(theta),
                        holes[i][1] + radii[i] * math.sin(theta))
            if not np.allclose([point["x"], point["y"]], expected, rtol=0, atol=1e-7):
                raise ValueError(f"hole {i}: trace coordinates differ from matched FEM probe")

    reference = {"spec": spec, "trace_step": update["step"], "levels": []}
    comparison = {"trace_step": update["step"], "levels": []}
    previous = None
    for n_theta, n_rings, background in levels:
        solution = solve(half_w, half_h, holes, material["e"], material["nu"], load["px"],
                         n_theta, n_rings, background, bcs=bcs)
        fields = [{mode: solution.profile_field(i, n_angles=n_angles, fd_step=fd_step,
                                                radius=radii[i], stress_mode=mode)
                   for mode in ("fd", "element", "recovered")}
                  for i in range(len(holes))]
        level_ref = {
            "mesh": {"n_theta": n_theta, "n_ring_layers": n_rings, "bg_spacing_factor": background,
                     "nodes": len(solution.nodes), "elements": len(solution.triangles)},
            "pcg_relative_residual": solution.relative_residual,
            "relative_work_error": solution.relative_work_error,
            "fem_internal_energy_j": solution.strain_energy_per_thickness * geometry["thickness"],
            "fd_step_m": fd_step,
            "probe_radii_m": radii,
            "holes": [{mode: {key: value.tolist() for key, value in field.items()}
                       for mode, field in by_mode.items()} for by_mode in fields],
        }
        level_cmp = {"mesh": level_ref["mesh"], "probe_radii_m": radii,
                     "fem_internal_energy_j": level_ref["fem_internal_energy_j"], "holes": []}
        for i, by_mode in enumerate(fields):
            measured = np.array([p["von_mises"] for p in update["holes"][i]["profile"]])
            metrics = {mode: profile_metrics(field["von_mises"], measured)
                       for mode, field in by_mode.items()}
            if previous is not None:
                for mode, field in by_mode.items():
                    metrics[mode]["fem_profile_change_over_current_rms"] = profile_metrics(
                        field["von_mises"], previous[i][mode]["von_mises"])["rms_error_over_reference_rms"]
            metrics["fd_vs_recovered_rms_over_recovered_rms"] = profile_metrics(
                by_mode["recovered"]["von_mises"], by_mode["fd"]["von_mises"])["rms_error_over_reference_rms"]
            metrics["element_vs_recovered_rms_over_recovered_rms"] = profile_metrics(
                by_mode["recovered"]["von_mises"], by_mode["element"]["von_mises"])["rms_error_over_reference_rms"]
            metrics["reference_components"] = {}
            for component in ("sxx", "syy", "sxy"):
                quality = {
                    "fd_vs_recovered_rms_over_recovered_rms": profile_metrics(
                        by_mode["recovered"][component], by_mode["fd"][component]
                    )["rms_error_over_reference_rms"],
                }
                if previous is not None:
                    quality["fem_profile_change_over_current_rms"] = profile_metrics(
                        by_mode["recovered"][component], previous[i]["recovered"][component]
                    )["rms_error_over_reference_rms"]
                metrics["reference_components"][component] = quality
            if all(all(component in point for component in ("sxx", "syy", "sxy"))
                   for point in update["holes"][i]["profile"]):
                metrics["stress_components"] = {
                    component: profile_metrics(by_mode["recovered"][component],
                                               [p[component] for p in update["holes"][i]["profile"]])
                    for component in ("sxx", "syy", "sxy")
                }
                theta = np.deg2rad(expected_angles)
                c, s = np.cos(theta), np.sin(theta)
                points = update["holes"][i]["profile"]
                sx = np.array([p["sxx"] for p in points])
                sy = np.array([p["syy"] for p in points])
                shear = np.array([p["sxy"] for p in points])
                candidate_traction = np.hypot(sx * c + shear * s, shear * c + sy * s)
                metrics["offset_traction_rms_difference_over_load"] = float(np.sqrt(np.mean(
                    (candidate_traction - by_mode["recovered"]["traction"]) ** 2)) / load["px"])
            if all("ux" in p and "uy" in p for p in update["holes"][i]["profile"]):
                predicted = np.array([[p["ux"], p["uy"]] for p in update["holes"][i]["profile"]])
                reference_displacement = np.column_stack((by_mode["fd"]["ux"], by_mode["fd"]["uy"]))
                metrics["displacement_rms_error_m"] = float(np.sqrt(np.mean(np.sum(
                    (predicted - reference_displacement) ** 2, axis=1))))
            level_cmp["holes"].append(metrics)
        reference["levels"].append(level_ref)
        comparison["levels"].append(level_cmp)
        previous = fields
    comparison["reference_quality"] = reference_quality(comparison["levels"])
    energy = update.get("energy_balance")
    comparison["energy_vs_fem_relative_error"] = (
        abs(energy["internal_energy"] - comparison["levels"][-1]["fem_internal_energy_j"])
        / comparison["levels"][-1]["fem_internal_energy_j"]
        if energy and math.isfinite(energy.get("internal_energy", math.nan)) else None)
    comparison["training_stationarity"] = hole_profile_stationarity(trace_dir / "updates.jsonl", update)
    companion = None
    if companion_dir is not None:
        companion_summary = json.loads((companion_dir / "summary.json").read_text())
        if (not companion_summary.get("completed") or not companion_summary.get("worker_ok")
                or companion_summary.get("nonfinite")):
            raise ValueError("no-hole companion must be complete and finite")
        companion_spec = tomllib.loads((companion_dir / "effective.toml").read_text())
        for key in ("material", "load", "formulation"):
            if companion_spec[key] != spec[key]:
                raise ValueError(f"no-hole companion has different {key}")
        if companion_spec["geometry"]["holes"] or any(
                companion_spec["geometry"][key] != geometry[key]
                for key in ("half_w", "half_h", "thickness")):
            raise ValueError("no-hole companion has different plate or contains holes")
        with (companion_dir / "updates.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("no_hole_benchmark") is not None:
                    companion = row["no_hole_benchmark"]
    summary_path = trace_dir / "summary.json"
    run_summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    comparison["physical_acceptance"] = physical_acceptance(
        spec, update, comparison, companion, run_summary)
    return reference, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=pathlib.Path)
    parser.add_argument("--levels", required=True,
                        help="comma-separated n_theta x n_ring_layers x bg_spacing_factor levels")
    parser.add_argument("--companion-no-hole", type=pathlib.Path,
                        help="completed no-hole trace with identical plate, material, load, and formulation")
    parser.add_argument("--output-suffix", default="",
                        help="suffix for new-only FEM JSON files when repeating a comparison")
    args = parser.parse_args()
    levels = []
    for part in args.levels.split(","):
        theta, rings, background = part.split("x")
        levels.append((int(theta), int(rings), float(background)))
    if len(levels) < 2:
        parser.error("at least two FEM levels are required")
    reference, comparison = compare(args.trace_dir, levels, args.companion_no_hole)
    if args.output_suffix and (not args.output_suffix.startswith("_") or not args.output_suffix[1:].replace("_", "").isalnum()):
        parser.error("--output-suffix must start with underscore and contain only letters, digits, or underscores")
    for name, value in ((f"fem_reference{args.output_suffix}.json", reference),
                        (f"fem_comparison{args.output_suffix}.json", comparison)):
        with (args.trace_dir / name).open("x") as output:
            json.dump(value, output, indent=2)
    print(json.dumps(comparison["levels"][-1], indent=2))


if __name__ == "__main__":
    main()
