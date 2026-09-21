#!/usr/bin/env python3
"""Phase B (docs/multi-hole-fem-ground-truth-investigation.md): a first-order superposition
estimate of multi-hole Kt, checked against `multi_hole_reference.py`'s real FEM ground truth.

Pure NumPy, no PINN code, no scipy needed (closed-form evaluation only) - deliberately scoped
this way per the investigation's own plan: this answers "is superposition even in the right
ballpark," not "build the production approximation."

Method: each hole's own classical isolated-hole Kirsch solution (infinite plate, remote
uniaxial tension S) is evaluated EXACTLY at its own boundary (satisfies traction-free there by
construction). Every OTHER hole's own Kirsch PERTURBATION field (its solution minus the
uniform remote field, so the far-field term is never double-counted) is added on top, evaluated
at that same point. This is the classical "zeroth iteration" of the method of successive
images/corrections (see the investigation doc's own "Approach 1"): a single evaluation, no
self-consistent correction pass - if this lands close to the real FEM Kt, the self-consistent
version (which WOULD account for each hole perturbing what the other "sees" as its own remote
field) has a real chance of being production-accurate; if it's already far off, that's evidence
against superposition being worth pursuing further for this project's actual geometries.
"""
import argparse
import json

import numpy as np


def kirsch_stress_cartesian(x, y, a, S):
    """Classical isolated circular-hole (radius a, centered at the origin of these x,y
    coordinates) stress field under remote uniaxial tension S in +x, evaluated at physical
    point (x,y) - Kirsch's closed-form solution in polar coordinates, transformed back to
    Cartesian. Valid for r=hypot(x,y) >= a (points inside the hole are undefined/unphysical)."""
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    ratio = (a / r) ** 2
    ratio2 = ratio ** 2
    srr = S / 2 * ((1 - ratio) + (1 - 4 * ratio + 3 * ratio2) * np.cos(2 * theta))
    stt = S / 2 * ((1 + ratio) - (1 + 3 * ratio2) * np.cos(2 * theta))
    srt = -S / 2 * (1 + 2 * ratio - 3 * ratio2) * np.sin(2 * theta)
    c, s = np.cos(theta), np.sin(theta)
    sxx = srr * c * c + stt * s * s - 2 * srt * s * c
    syy = srr * s * s + stt * c * c + 2 * srt * s * c
    sxy = (srr - stt) * s * c + srt * (c * c - s * s)
    return sxx, syy, sxy


def kirsch_perturbation_cartesian(x, y, a, S):
    """Same field, with the uniform remote term (S, 0, 0) subtracted - the LOCAL correction
    this hole alone contributes, safe to superpose onto another hole's own exact solution
    without double-counting the far field."""
    sxx, syy, sxy = kirsch_stress_cartesian(x, y, a, S)
    return sxx - S, syy, sxy


def superposition_kt(holes, S, n_angles=720):
    """First-order superposition Kt estimate for every hole in `holes` (list of (cx, cy, a)).
    Returns {hole_index: {"kt_hoop": ..., "kt_vm": ...}}."""
    results = {}
    theta = np.arange(n_angles) * 2 * np.pi / n_angles
    for i, (cxi, cyi, a) in enumerate(holes):
        # Points on hole i's own boundary, in GLOBAL coordinates.
        px = cxi + a * np.cos(theta)
        py = cyi + a * np.sin(theta)
        # Hole i's own exact Kirsch solution (local coordinates relative to its own center) -
        # already traction-free at r=a by construction.
        sxx, syy, sxy = kirsch_stress_cartesian(a * np.cos(theta), a * np.sin(theta), a, S)
        # Add every OTHER hole's perturbation, evaluated at these same global points.
        for j, (cxj, cyj, aj) in enumerate(holes):
            if j == i:
                continue
            dx, dy = px - cxj, py - cyj
            pxx, pyy, pxy = kirsch_perturbation_cartesian(dx, dy, aj, S)
            sxx = sxx + pxx
            syy = syy + pyy
            sxy = sxy + pxy
        # Hoop stress at hole i's own boundary: local normal is radial (cos(theta),sin(theta))
        # relative to hole i's OWN center - transform the combined Cartesian stress back to
        # this hole's own polar hoop component.
        c, s = np.cos(theta), np.sin(theta)
        hoop = sxx * s * s - 2 * sxy * s * c + syy * c * c
        vm = np.sqrt(sxx * sxx - sxx * syy + syy * syy + 3 * sxy * sxy)
        results[i] = {"kt_hoop": float(hoop.max() / S), "kt_vm": float(vm.max() / S)}
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--holes", required=True, help="semicolon-separated cx,cy,radius triples")
    p.add_argument("--traction", type=float, default=69e6)
    p.add_argument("--fem-reference", default=None,
                    help="optional JSON, e.g. '{\"0\": 3.063, \"1\": 2.937}' of real FEM kt_vm "
                         "per hole index (from multi_hole_reference.py), to report relative error against")
    args = p.parse_args()

    holes = []
    for triple in args.holes.split(";"):
        cx, cy, r = (float(v) for v in triple.split(","))
        holes.append((cx, cy, r))

    estimate = superposition_kt(holes, args.traction)
    fem = json.loads(args.fem_reference) if args.fem_reference else None
    report = {}
    for i, metrics in estimate.items():
        report[f"hole{i}"] = metrics
        if fem and str(i) in fem:
            report[f"hole{i}"]["fem_kt_vm"] = fem[str(i)]
            report[f"hole{i}"]["relative_error_vs_fem"] = abs(metrics["kt_vm"] - fem[str(i)]) / fem[str(i)]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
