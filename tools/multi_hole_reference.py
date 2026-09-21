#!/usr/bin/env python3
"""Independent plane-stress CST reference for a rectangular plate with N arbitrary circular
holes under uniaxial tension - no symmetry assumption.

This is a deliberately SEPARATE tool from `finite_plate_reference.py`, not a generalization of
it in place: that tool's quarter-plate, double-symmetry mesh is exact only for one centered
hole, and its own hardcoded reference numbers (issue #76/#77's `FEM_KT=2.460638516` and every
comparison built on it) must stay byte-reproducible - this file never imports or modifies it,
only reuses its independently-testable numerical core (`cst_matrix`, `element_pcg`) via import.

Unstructured triangulation (`scipy.spatial.Delaunay`) over a point cloud combining: graded
rings around EACH hole boundary (exact circular geometry at every hole, mirroring
`finite_plate_reference.py`'s own radial grading idea, just local to each hole instead of
extending to the rectangle edge), the outer rectangle boundary, and a background fill grid.
Triangles are kept only if their centroid lies inside the plate and outside every hole (with a
small safety margin) - the same "keep only structurally valid elements" discipline the existing
tool's degenerate-triangle rejection already established.

New dependency (scipy), by design, not by oversight: `finite_plate_reference.py`'s own "only
NumPy" choice is safe there because its mesh is a simple structured polar grid; a general
N-arbitrary-hole domain genuinely needs a real Delaunay triangulation, and scipy's (wrapping
Qhull, an extremely well-tested industry-standard implementation) is far more trustworthy for a
GROUND-TRUTH tool than a hand-rolled one would be - the entire point of this file is to be an
independent check, so its own correctness risk has to be lower than what it's checking, not
merely "no new pip installs."

No symmetry to exploit means two real differences from `finite_plate_reference.py`: traction
must be applied on BOTH the left and right edges (there marks only the right edge, relying on
the reflection embedded in its own quarter-plate BCs to cover the left), and rigid-body motion
(2 translations + 1 rotation, vs. that tool's 2 symmetry-fixed translations only) needs explicit
pinning - this file pins ux=uy=0 at the midpoint of the left edge and uy=0 at the midpoint of
the right edge (a standard statically-determinate "pin + roller" tension-specimen support),
chosen on the OUTER boundary, far from every hole, so it cannot contaminate a near-hole Kt
reading.
"""
import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import Delaunay
except ImportError as exc:  # pragma: no cover - environment-dependent, not a logic path
    raise SystemExit(
        "multi_hole_reference.py requires scipy (Delaunay triangulation) - "
        "install with `pip install scipy` (or `pip install --user scipy` / a venv "
        "if your Python is externally managed). This is a deliberate, disclosed "
        "dependency - see this file's own module doc comment for why."
    ) from exc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from finite_plate_reference import cst_matrix, element_pcg  # noqa: E402


def validate_holes(half_w, half_h, holes):
    if not holes:
        raise ValueError("at least one hole is required (use finite_plate_reference.py for the no-hole/single-centered-hole case)")
    for i, (cx, cy, r) in enumerate(holes):
        if not all(math.isfinite(v) for v in (cx, cy, r)) or r <= 0:
            raise ValueError(f"hole {i}: center and radius must be finite, radius must be positive")
        if not (-half_w + r <= cx <= half_w - r and -half_h + r <= cy <= half_h - r):
            raise ValueError(f"hole {i}: extends outside the plate (half_w={half_w}, half_h={half_h})")
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            cxi, cyi, ri = holes[i]
            cxj, cyj, rj = holes[j]
            dist = math.hypot(cxi - cxj, cyi - cyj)
            if dist < ri + rj:
                raise ValueError(f"holes {i} and {j} overlap")


def mesh(half_w, half_h, holes, n_theta, n_ring_layers, bg_spacing_factor=1.0):
    """Build a full-plate, N-hole unstructured CST mesh - `(nodes, triangles)` only. Thin
    wrapper around `_mesh_with_delaunay` (which additionally returns the `Delaunay` object and
    index remap `solve()`/`MultiHoleSolution.sample` need for point location) - kept as its own
    function since tests/callers that only need the mesh itself shouldn't have to know about
    the remap plumbing.

    `n_theta` points per hole ring (matches `finite_plate_reference.py`'s own `n_theta`
    convention), `n_ring_layers` radially-graded layers per hole extending out to
    `ring_extent = 3 * radius` (mirrors this project's own `ANNULAR_INTERFACE_RADIUS_FACTOR`,
    `pinn-core/src/user_geometry.rs` - not imported, just the same well-motivated constant, kept
    independent since this Python tool has no Rust dependency). `bg_spacing_factor` scales the
    background fill grid spacing relative to the smallest hole radius - smaller is finer/slower.
    """
    _, _, nodes, triangles = _mesh_with_delaunay(half_w, half_h, holes, n_theta, n_ring_layers, bg_spacing_factor)
    return nodes, triangles


@dataclass
class MultiHoleSolution:
    nodes: np.ndarray
    triangles: np.ndarray
    displacement: np.ndarray
    stress: np.ndarray
    constitutive: np.ndarray
    traction: float
    holes: list
    bcs: list
    delaunay: "Delaunay"
    # Maps a `Delaunay` simplex index -> this solution's own `triangles`/`stress` row index,
    # or -1 if that simplex was rejected (a hole interior, outside the rectangle, or a
    # sliver). NOT vertex-index content matching (`(self.triangles == kept_tri).all(axis=1)`) -
    # that approach breaks silently whenever `_mesh_with_delaunay`'s CW->CCW winding fix
    # reorders a triangle's vertex columns relative to `delaunay.simplices`' own order, since
    # then no row of `self.triangles` matches `node_remap[orig_tri]` by content even though the
    # triangle itself WAS kept - a real bug this exact field exists to make structurally
    # impossible (direct index lookup, no equality search over reordered rows).
    simplex_to_kept: np.ndarray
    iterations: int
    relative_residual: float
    relative_work_error: float

    def sample(self, points):
        """Piecewise-linear displacement and element stress at arbitrary physical points, via
        the SAME Delaunay object's own point-location (`find_simplex`) - no bespoke candidate
        search needed, unlike `finite_plate_reference.py`'s polar-structured-mesh version."""
        points = np.asarray(points, dtype=float)
        simplex_idx = self.delaunay.find_simplex(points)
        displacements, stresses = [], []
        for point, orig_simplex in zip(points, simplex_idx):
            if orig_simplex < 0:
                raise ValueError(f"point {point.tolist()} lies outside the triangulated hull")
            kept_idx = self.simplex_to_kept[orig_simplex]
            if kept_idx < 0:
                raise ValueError(f"point {point.tolist()} lies in a rejected (hole/outside) element")
            tri = self.triangles[kept_idx]
            vertices = self.nodes[tri]
            origin = vertices[0]
            edge1, edge2 = vertices[1] - origin, vertices[2] - origin
            det = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            diff = point - origin
            w1 = (diff[0] * edge2[1] - diff[1] * edge2[0]) / det
            w2 = (edge1[0] * diff[1] - edge1[1] * diff[0]) / det
            weights = np.array([1 - w1 - w2, w1, w2])
            uv = weights @ self.displacement[tri]
            stress = self.stress[kept_idx]
            displacements.append(uv)
            stresses.append(stress)
        return np.asarray(displacements), np.asarray(stresses)

    def profile(self, hole_index, n_angles=360, fd_step=None, radius=None):
        """`radius=None` (default) reads exactly at this hole's own true boundary - only valid
        with `fd_step=None`, since a central-FD stencil straddling the exact hole edge would
        need points strictly closer than the boundary itself. When `fd_step` is given, the
        CALLER must pass a `radius` with enough margin for the stencil to clear the hole on
        every side (mirrors `finite_plate_reference.py`'s own `probe_radius = radius +
        4*fd_h*max(half_w,half_h)` convention exactly - see `main()`'s own call site)."""
        cx, cy, hole_r = self.holes[hole_index]
        r = hole_r if radius is None else radius
        if fd_step is not None and r <= hole_r:
            raise ValueError(f"hole {hole_index}: an FD profile needs radius > the hole's own radius ({hole_r}) for stencil clearance, got {r}")
        theta = np.arange(n_angles) * 2 * math.pi / n_angles
        points = np.stack((cx + r * np.cos(theta), cy + r * np.sin(theta)), axis=1)
        _, stress = self.sample(points)
        if fd_step is not None:
            hx, hy = fd_step
            xp, _ = self.sample(points + [hx, 0])
            xm, _ = self.sample(points - [hx, 0])
            yp, _ = self.sample(points + [0, hy])
            ym, _ = self.sample(points - [0, hy])
            dx, dy = (xp - xm) / (2 * hx), (yp - ym) / (2 * hy)
            strain = np.stack((dx[:, 0], dy[:, 1], dy[:, 0] + dx[:, 1]), axis=1)
            stress = strain @ self.constitutive.T
        sx, sy, shear = stress.T
        hoop = sx * np.sin(theta) ** 2 - 2 * shear * np.sin(theta) * np.cos(theta) + sy * np.cos(theta) ** 2
        vm = np.sqrt(sx * sx - sx * sy + sy * sy + 3 * shear * shear)
        return {"kt_hoop": float(hoop.max() / self.traction), "kt_vm": float(vm.max() / self.traction)}


def solve(half_w, half_h, holes, young, nu, traction, n_theta, n_ring_layers, bg_spacing_factor=1.0, bcs=None):
    """`bcs`: optional list of `"free"`/`"fixed"`, one per hole (default: every hole `"free"`,
    byte-identical to this function's pre-BC-support behavior). `"fixed"` applies a real
    Dirichlet (zero-displacement) constraint at every mesh node on that hole's own exact
    boundary ring - the FEM equivalent of `pinn_core::user_geometry::HoleBc::Fixed`, added so a
    PINN spec with a genuinely mixed Free/Fixed hole set (e.g. `triple_hole_plate.toml`'s own
    real production BCs) can be compared against ground truth solved under the SAME boundary
    conditions, not the "every hole traction-free" approximation this tool used before - see
    `docs/multi-hole-fem-ground-truth-investigation.md`'s own disclosed BC-mismatch caveat this
    closes."""
    validate_holes(half_w, half_h, holes)
    if bcs is None:
        bcs = ["free"] * len(holes)
    if len(bcs) != len(holes):
        raise ValueError(f"bcs must have exactly one entry per hole ({len(holes)} holes, {len(bcs)} bcs given)")
    for i, bc in enumerate(bcs):
        if bc not in ("free", "fixed"):
            raise ValueError(f"hole {i}: bc must be 'free' or 'fixed', got {bc!r}")
    if not all(math.isfinite(v) for v in (half_w, half_h, young, nu, traction)):
        raise ValueError("geometry, material, and traction must be finite")
    if half_w <= 0 or half_h <= 0 or young <= 0 or not -1 < nu < 0.5 or traction <= 0:
        raise ValueError("require half_w>0, half_h>0, E>0, -1<nu<0.5, positive uniaxial tensile traction")

    # `_mesh_with_delaunay` (not the public `mesh()` wrapper) - `MultiHoleSolution.sample`
    # needs the `Delaunay` object itself (for `find_simplex`-based point location) and the
    # simplex->kept-triangle index map, neither of which the plain `(nodes, triangles)`
    # wrapper exposes.
    delaunay, simplex_to_kept, nodes, triangles = _mesh_with_delaunay(half_w, half_h, holes, n_theta, n_ring_layers, bg_spacing_factor)

    B, D, areas, stiffness = cst_matrix(nodes[triangles], young, nu)
    dofs = np.stack((2 * triangles, 2 * triangles + 1), axis=-1).reshape(-1, 6)
    force = np.zeros(2 * len(nodes))

    for side, x_target in ((-1, -half_w), (1, half_w)):
        on_edge = np.flatnonzero(np.abs(nodes[:, 0] - x_target) < 1e-9 * half_w)
        if len(on_edge) < 2:
            raise ValueError(f"too few mesh nodes on the {'left' if side < 0 else 'right'} edge to apply traction")
        order = np.argsort(nodes[on_edge, 1])
        edge_nodes = on_edge[order]
        ys = nodes[edge_nodes, 1]
        seg_len = np.diff(ys)
        nodal_force = np.zeros(len(edge_nodes))
        nodal_force[:-1] += seg_len / 2
        nodal_force[1:] += seg_len / 2
        force[2 * edge_nodes] += side * traction * nodal_force

    # Rigid-body-motion removal, chosen to itself be mirror-symmetric (x -> -x) - a REAL bug
    # fix, not a style choice. The original scheme (full ux=uy=0 pin at the left edge midpoint,
    # uy=0-only roller at the right) is asymmetric BY CONSTRUCTION. That was invisible for
    # every all-`Free` geometry this tool shipped with before Fixed-BC support: a self-
    # equilibrated external load (uniaxial tension, zero net force/moment) plus zero-reaction
    # Free-hole boundaries left the pins themselves carrying zero reaction force regardless of
    # where they sat, so an asymmetric CHOICE of zero-force gauge-fix never perturbed the
    # solution. A `Fixed` hole is a genuine internal support that - for a hole set that is
    # itself mirror-symmetric about x=0 under symmetric loading, e.g.
    # `triple_hole_plate.toml`'s own two off-center Free holes flanking one on-axis Fixed hole
    # - DOES generally exert a nonzero net Y reaction (never a net X reaction, by that same
    # mirror symmetry) on the plate. The OLD asymmetric pin then had to carry part of that
    # reaction asymmetrically, measurably breaking left/right symmetry: a real, reproducible
    # ~30-50% Kt difference between two holes that MUST be identical by symmetry
    # (`multi-hole-fem-ground-truth-investigation.md`'s own real regression proof of this bug,
    # confirmed via a geometry-mirroring cross-check that ruled out mesh/solve nondeterminism
    # before this fix was identified).
    #
    # Fix: `uy=0` at BOTH the left and right edge midpoints (a mirror-symmetric PAIR - `uy` is
    # even under x-mirror, so this constraint set is itself invariant), `ux=0` at a single node
    # as close as the mesh allows to the bottom edge's own midpoint (`x≈0`, the mirror axis
    # itself - `ux` is odd under x-mirror, so a constraint AT the axis is the only single-point
    # choice that doesn't privilege a side). Together removes exactly the 3 rigid-body DOFs
    # (2 translation + 1 rotation) the old scheme did, just via a symmetric set instead of an
    # asymmetric one.
    left_edge = np.flatnonzero(np.abs(nodes[:, 0] + half_w) < 1e-9 * half_w)
    right_edge = np.flatnonzero(np.abs(nodes[:, 0] - half_w) < 1e-9 * half_w)
    bottom_edge = np.flatnonzero(np.abs(nodes[:, 1] + half_h) < 1e-9 * half_h)
    if len(bottom_edge) == 0:
        raise ValueError("no mesh nodes found on the bottom edge to anchor x-translation")
    pin_left = left_edge[np.argmin(np.abs(nodes[left_edge, 1]))]
    pin_right = right_edge[np.argmin(np.abs(nodes[right_edge, 1]))]
    pin_bottom = bottom_edge[np.argmin(np.abs(nodes[bottom_edge, 0]))]
    free = np.ones(len(force), dtype=bool)
    free[2 * pin_left + 1] = False
    free[2 * pin_right + 1] = False
    free[2 * pin_bottom] = False

    # `bc == "fixed"`: zero-displacement Dirichlet constraint at every mesh node exactly on
    # that hole's own boundary ring (`_mesh_with_delaunay` always places the first ring layer
    # AT the hole's true radius - see `test_mesh_has_exact_circular_boundary_at_every_hole`).
    # Matched by distance rather than tracking indices through the point-cloud construction
    # pipeline - robust to any future reordering/deduplication there, and mirrors the existing
    # test's own verification method exactly.
    for (cx, cy, r), bc in zip(holes, bcs):
        if bc != "fixed":
            continue
        dist = np.hypot(nodes[:, 0] - cx, nodes[:, 1] - cy)
        on_boundary = np.flatnonzero(np.abs(dist - r) < 1e-9 * max(half_w, half_h))
        if len(on_boundary) == 0:
            raise ValueError(f"fixed hole at ({cx},{cy},{r}): no mesh nodes found exactly on its boundary")
        free[2 * on_boundary] = False
        free[2 * on_boundary + 1] = False

    displacement, reaction, iterations, residual = element_pcg(stiffness, dofs, force, free)
    strains = np.einsum("eij,ej->ei", B, displacement[dofs])
    stress = strains @ D.T
    twice_energy = float(np.sum(areas * np.einsum("ei,ei->e", strains, stress)))
    work = float(displacement @ force)
    work_error = abs(twice_energy - work) / abs(work)
    balance = reaction.reshape(-1, 2).sum(axis=0) + force.reshape(-1, 2).sum(axis=0)
    ref_force = traction * 2 * half_h
    if np.linalg.norm(balance) / ref_force > 1e-6 or work_error > 1e-6:
        raise RuntimeError("FEM reaction balance or strain-energy identity failed")

    return MultiHoleSolution(nodes, triangles, displacement.reshape(-1, 2), stress, D, traction,
                              holes, bcs, delaunay, simplex_to_kept, iterations, residual, work_error)


def _mesh_with_delaunay(half_w, half_h, holes, n_theta, n_ring_layers, bg_spacing_factor):
    """Same point-cloud construction as `mesh()`, but also returns the `Delaunay` object and a
    simplex-index -> kept-triangle-index map, so `MultiHoleSolution.sample` can locate arbitrary
    probe points via `Delaunay.find_simplex` (O(1) lookup after that, not a vertex-content
    search - see `MultiHoleSolution.simplex_to_kept`'s own doc comment for why a content search
    is a real, silent-failure-prone bug here) without re-triangulating."""
    if n_theta < 12 or n_ring_layers < 2:
        raise ValueError("require n_theta>=12, n_ring_layers>=2")
    points = []
    ring_extent_factor = 3.0
    for cx, cy, r in holes:
        extent = ring_extent_factor * r
        layer_radii = r * (extent / r) ** np.linspace(0, 1, n_ring_layers + 1)
        thetas = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
        for layer_r in layer_radii:
            points.append(np.stack((cx + layer_r * np.cos(thetas), cy + layer_r * np.sin(thetas)), axis=-1))
    min_r = min(r for _, _, r in holes)
    spacing = bg_spacing_factor * min_r
    n_x = max(4, int(round(2 * half_w / spacing)))
    n_y = max(4, int(round(2 * half_h / spacing)))
    xs = np.linspace(-half_w, half_w, n_x + 1)
    ys = np.linspace(-half_h, half_h, n_y + 1)
    points.append(np.stack((xs, np.full_like(xs, -half_h)), axis=-1))
    points.append(np.stack((xs, np.full_like(xs, half_h)), axis=-1))
    points.append(np.stack((np.full_like(ys, -half_w), ys), axis=-1))
    points.append(np.stack((np.full_like(ys, half_w), ys), axis=-1))
    gx, gy = np.meshgrid(xs[1:-1], ys[1:-1])
    bg = np.stack((gx.ravel(), gy.ravel()), axis=-1)
    keep = np.ones(len(bg), dtype=bool)
    for cx, cy, r in holes:
        keep &= (bg[:, 0] - cx) ** 2 + (bg[:, 1] - cy) ** 2 > (ring_extent_factor * r) ** 2
    points.append(bg[keep])
    all_points = np.concatenate(points, axis=0)
    delaunay = Delaunay(all_points)
    simplices = delaunay.simplices
    centroids = all_points[simplices].mean(axis=1)
    inside = (np.abs(centroids[:, 0]) < half_w) & (np.abs(centroids[:, 1]) < half_h)
    for cx, cy, r in holes:
        inside &= (centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2 > r * r
    v = all_points[simplices]
    twice_area = (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1])
    min_area = 1e-8 * (2 * half_w) * (2 * half_h) / len(simplices)
    inside &= np.abs(twice_area) / 2 > min_area
    kept = simplices[inside].copy()
    v_kept = all_points[kept]
    signed = (v_kept[:, 1, 0] - v_kept[:, 0, 0]) * (v_kept[:, 2, 1] - v_kept[:, 0, 1]) - (v_kept[:, 2, 0] - v_kept[:, 0, 0]) * (v_kept[:, 1, 1] - v_kept[:, 0, 1])
    kept[signed < 0] = kept[signed < 0][:, [0, 2, 1]]
    used = np.unique(kept)
    node_remap = np.full(len(all_points), -1, dtype=int)
    node_remap[used] = np.arange(len(used))
    nodes = all_points[used]
    triangles = node_remap[kept]
    # `inside` is a boolean mask over ALL simplices (Delaunay's own order); `kept`/`triangles`
    # only cover the True entries, in the SAME relative order `np.flatnonzero`/boolean indexing
    # preserves - so simplex i's kept-triangle row is simply its rank among the True entries.
    simplex_to_kept = np.full(len(simplices), -1, dtype=int)
    simplex_to_kept[inside] = np.arange(inside.sum())
    return delaunay, simplex_to_kept, nodes, triangles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--half-width", type=float, required=True)
    p.add_argument("--half-height", type=float, required=True)
    p.add_argument("--holes", required=True,
                    help="semicolon-separated cx,cy,radius triples, e.g. '-0.03,0,0.01;0.03,0,0.008'")
    p.add_argument("--bcs", default=None,
                    help="semicolon-separated 'free'/'fixed' per hole, e.g. 'free;fixed' - "
                         "default: every hole free (this tool's pre-existing behavior)")
    p.add_argument("--young", type=float, default=71.7e9)
    p.add_argument("--poisson", type=float, default=0.33)
    p.add_argument("--traction", type=float, default=69e6)
    p.add_argument("--fd-h", type=float, default=1e-3)
    p.add_argument("--levels", default="24x3x1.2,32x4x0.9,48x6x0.6",
                    help="comma-separated n_theta x n_ring_layers x bg_spacing_factor levels")
    p.add_argument("--max-relative-change", type=float, default=0.03)
    args = p.parse_args()

    holes = []
    for triple in args.holes.split(";"):
        cx, cy, r = (float(v) for v in triple.split(","))
        holes.append((cx, cy, r))
    bcs = args.bcs.split(";") if args.bcs is not None else None
    if bcs is not None and len(bcs) != len(holes):
        p.error(f"--bcs must have exactly one entry per hole ({len(holes)} holes, {len(bcs)} bcs given)")

    levels = []
    for level in args.levels.split(","):
        nt, nr, bg = level.split("x")
        levels.append((int(nt), int(nr), float(bg)))
    if len(levels) < 2:
        p.error("at least two mesh refinement levels are required for a convergence check")

    fd_step = (args.fd_h * args.half_width, args.fd_h * args.half_height)
    # Same safety-margin convention as finite_plate_reference.py's own `probe_radius` - an FD
    # central-difference stencil straddling the hole's EXACT boundary would need points closer
    # to the hole than the mesh (or the physical stencil offset itself) can resolve.
    probe_radii = [r + 4 * args.fd_h * max(args.half_width, args.half_height) for _, _, r in holes]
    for i, (probe_r, (_, _, hole_r)) in enumerate(zip(probe_radii, holes)):
        if probe_r + max(fd_step) >= 3.0 * hole_r:
            p.error(f"hole {i}: the FD-safe probe ring and its stencil must clear the hole's own ring-extent mesh region (3x radius)")
    previous, changes = None, []
    for n_theta, n_ring_layers, bg_factor in levels:
        start = time.monotonic()
        solution = solve(args.half_width, args.half_height, holes, args.young, args.poisson,
                          args.traction, n_theta, n_ring_layers, bg_factor, bcs=bcs)
        metrics = {}
        for hole_index in range(len(holes)):
            for key, value in solution.profile(hole_index, fd_step=fd_step, radius=probe_radii[hole_index]).items():
                metrics[f"hole{hole_index}_{key}"] = value
        changes.append(None if previous is None else max(abs(metrics[k] - previous[k]) / max(abs(metrics[k]), 1e-30) for k in metrics))
        print(json.dumps(dict(mesh=f"{n_theta}x{n_ring_layers}x{bg_factor}", nodes=len(solution.nodes),
                              elements=len(solution.triangles), **metrics,
                              max_relative_change=changes[-1], pcg_iterations=solution.iterations,
                              relative_work_error=solution.relative_work_error,
                              seconds=time.monotonic() - start)), flush=True)
        previous = metrics
    if any(change is not None and change > args.max_relative_change for change in changes[-2:]):
        raise SystemExit("reference not converged on the last two refinements; refine before using it")
    print("Every hole's Kt metrics pass the mesh-change gate; this is not a rigorous error bound.")


if __name__ == "__main__":
    main()
