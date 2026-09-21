"""Structural/regression checks for `multi_hole_reference.py`, independent of the PINN
implementation - mirrors `test_finite_plate_reference.py`'s own conventions.

NOTE (honest disclosure, see the investigation writeup this file ships alongside): this tool's
mesh has a real, unresolved fine-mesh-density anomaly for closely-spaced holes (a Kt value can
move the WRONG way between two successive refinements once the background-grid strip between
two hole rings gets thin relative to local density - likely a sliver/quality issue in that
transition zone, not yet root-caused). These tests check structural correctness (no crash,
real physics identities hold, Kt lands in a sane range) at MODERATE mesh density where the tool
is well-behaved - they do NOT assert a specific converged Kt value, since that number is not
yet trustworthy to the precision `finite_plate_reference.py`'s own tests hold themselves to.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from multi_hole_reference import mesh, solve, validate_holes


class MultiHoleReferenceTests(unittest.TestCase):
    def test_validate_holes_rejects_overlap(self):
        with self.assertRaises(ValueError):
            validate_holes(0.1, 0.1, [(-0.01, 0.0, 0.02), (0.01, 0.0, 0.02)])

    def test_validate_holes_rejects_out_of_bounds(self):
        with self.assertRaises(ValueError):
            validate_holes(0.1, 0.1, [(0.095, 0.0, 0.01)])

    def test_validate_holes_accepts_well_separated_holes(self):
        validate_holes(0.1, 0.05, [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)])

    def test_mesh_has_exact_circular_boundary_at_every_hole(self):
        holes = [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)]
        nodes, triangles = mesh(0.10, 0.05, holes, 20, 3, 1.4)
        self.assertGreater(len(triangles), 0)
        for cx, cy, r in holes:
            dist = np.hypot(nodes[:, 0] - cx, nodes[:, 1] - cy)
            self.assertTrue(np.any(np.abs(dist - r) < 1e-9), f"no mesh node found exactly on hole ({cx},{cy},{r})'s boundary")

    def test_mesh_every_triangle_is_ccw_and_outside_every_hole(self):
        holes = [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)]
        nodes, triangles = mesh(0.10, 0.05, holes, 20, 3, 1.4)
        v = nodes[triangles]
        twice_area = (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1])
        self.assertTrue(np.all(twice_area > 0), "every kept triangle must be CCW-ordered (cst_matrix's own requirement)")
        centroids = v.mean(axis=1)
        for cx, cy, r in holes:
            dist2 = (centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2
            self.assertTrue(np.all(dist2 > r * r), f"a triangle centroid fell inside hole ({cx},{cy},{r})")

    def test_solve_two_holes_satisfies_energy_and_reaction_balance(self):
        holes = [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)]
        solution = solve(0.10, 0.05, holes, 71.7e9, 0.33, 69e6, 20, 3, 1.4)
        self.assertLess(solution.relative_work_error, 1e-9)

    def test_solve_two_holes_kt_lands_in_a_physically_sane_range(self):
        # Not a converged-value assertion (see this file's own module doc comment) - just a
        # real, decisive sanity range: a legitimate stress-concentration factor for two
        # traction-free circular holes under uniaxial tension cannot be below 1 (a hole cannot
        # REDUCE peak stress below the nominal far-field value at its own boundary) or wildly
        # above the single isolated-hole Kirsch value of 3.0 for these well-separated holes.
        holes = [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)]
        solution = solve(0.10, 0.05, holes, 71.7e9, 0.33, 69e6, 28, 4, 1.0)
        fd_h = 1e-3
        for i, (_, _, r) in enumerate(holes):
            probe_r = r + 4 * fd_h * max(0.10, 0.05)
            result = solution.profile(i, fd_step=(fd_h * 0.10, fd_h * 0.05), radius=probe_r)
            self.assertGreater(result["kt_vm"], 1.0, f"hole {i}: Kt must exceed 1.0")
            self.assertLess(result["kt_vm"], 4.0, f"hole {i}: Kt implausibly far above the isolated-hole Kirsch value")

    def test_solve_rejects_a_single_hole_too_close_to_the_edge(self):
        with self.assertRaises(ValueError):
            solve(0.1, 0.1, [(0.095, 0.0, 0.01)], 71.7e9, 0.33, 69e6, 20, 3, 1.0)

    # ─── Fixed-BC support (closes the BC-mismatch caveat documented against the PINN's own
    # real mixed Free/Fixed specs, e.g. notched_plate.toml/triple_hole_plate.toml) ──────────

    def test_solve_default_bcs_is_byte_identical_to_omitting_bcs(self):
        holes = [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)]
        a = solve(0.10, 0.05, holes, 71.7e9, 0.33, 69e6, 20, 3, 1.0)
        b = solve(0.10, 0.05, holes, 71.7e9, 0.33, 69e6, 20, 3, 1.0, bcs=["free", "free"])
        np.testing.assert_array_equal(a.displacement, b.displacement)
        np.testing.assert_array_equal(a.stress, b.stress)

    def test_solve_rejects_an_unknown_bc_value(self):
        with self.assertRaises(ValueError):
            solve(0.10, 0.05, [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)], 71.7e9, 0.33, 69e6,
                  20, 3, 1.0, bcs=["free", "clamped"])

    def test_solve_rejects_wrong_length_bcs(self):
        with self.assertRaises(ValueError):
            solve(0.10, 0.05, [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)], 71.7e9, 0.33, 69e6,
                  20, 3, 1.0, bcs=["free"])

    def test_solve_fixed_hole_boundary_has_zero_displacement(self):
        # The real, load-bearing proof: a "fixed" hole's own boundary nodes must be
        # constrained to (u,v)=(0,0) by the solve itself - not approximately small, exactly
        # zero (a hard FEM Dirichlet constraint), mirroring what PINN's own HoleBc::Fixed
        # means physically (even though the PINN side only enforces it as a soft penalty).
        holes = [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)]
        solution = solve(0.10, 0.05, holes, 71.7e9, 0.33, 69e6, 24, 3, 1.0, bcs=["free", "fixed"])
        cx, cy, r = holes[1]
        dist = np.hypot(solution.nodes[:, 0] - cx, solution.nodes[:, 1] - cy)
        on_boundary = np.flatnonzero(np.abs(dist - r) < 1e-9 * 0.10)
        self.assertGreater(len(on_boundary), 0, "must find real mesh nodes on the fixed hole's boundary")
        boundary_disp = solution.displacement[on_boundary]
        np.testing.assert_allclose(boundary_disp, 0.0, atol=1e-15)

    def test_solve_fixed_hole_changes_the_free_holes_own_stress_field(self):
        # A fixed hole is a real, coupled elasticity boundary condition - it must measurably
        # change the OTHER (free) hole's own stress reading too, not just its own boundary.
        # This is exactly the "BC mismatch" this feature exists to close: treating every hole
        # as free (the tool's old-only behavior) is NOT a correct ground truth for a spec that
        # has a real Fixed hole, because the Fixed hole perturbs the whole field.
        holes = [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)]
        fd_h = 1e-3
        probe_r = holes[0][2] + 4 * fd_h * 0.10
        all_free = solve(0.10, 0.05, holes, 71.7e9, 0.33, 69e6, 24, 3, 1.0, bcs=["free", "free"])
        mixed = solve(0.10, 0.05, holes, 71.7e9, 0.33, 69e6, 24, 3, 1.0, bcs=["free", "fixed"])
        kt_all_free = all_free.profile(0, fd_step=(fd_h * 0.10, fd_h * 0.05), radius=probe_r)["kt_vm"]
        kt_mixed = mixed.profile(0, fd_step=(fd_h * 0.10, fd_h * 0.05), radius=probe_r)["kt_vm"]
        self.assertNotAlmostEqual(kt_all_free, kt_mixed, places=3)

    def test_solve_with_a_fixed_hole_still_satisfies_energy_and_reaction_balance(self):
        holes = [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)]
        solution = solve(0.10, 0.05, holes, 71.7e9, 0.33, 69e6, 24, 3, 1.0, bcs=["free", "fixed"])
        self.assertLess(solution.relative_work_error, 1e-6)

    # ─── Real regression proof: the rigid-body-pin mirror-symmetry bug this Fixed-BC feature
    # exposed and fixed. Two off-center Free holes, mirror images of each other about x=0, with
    # one Fixed hole exactly on that mirror axis (x=0) - the whole geometry/load/BC problem is
    # mirror-symmetric under x -> -x, so the two Free holes' own Kt MUST be equal by physics.
    # Before this fix, the tool's own asymmetric rigid-body pin (full pin on the left edge,
    # roller-only on the right) gave a real, reproducible ~30-50% difference between them -
    # invisible in every all-Free geometry this tool shipped with before (a self-equilibrated
    # external load plus zero-reaction Free-hole boundaries left the pins carrying zero
    # reaction regardless of asymmetric placement), but exposed the instant a Fixed hole (a
    # real internal support with a genuinely nonzero net reaction) was added. ─────────────

    def test_solve_fixed_hole_on_mirror_axis_gives_symmetric_kt_for_two_mirrored_free_holes(self):
        holes = [(-0.06, 0.02, 0.009), (0.0, -0.02, 0.007), (0.06, 0.02, 0.009)]
        solution = solve(0.15, 0.06, holes, 71.7e9, 0.33, 69e6, 44, 6, 1.0, bcs=["free", "fixed", "free"])
        fd_h = 1e-3
        r0 = holes[0][2] + 4 * fd_h * 0.15
        r2 = holes[2][2] + 4 * fd_h * 0.15
        kt0 = solution.profile(0, fd_step=(fd_h * 0.15, fd_h * 0.06), radius=r0)["kt_vm"]
        kt2 = solution.profile(2, fd_step=(fd_h * 0.15, fd_h * 0.06), radius=r2)["kt_vm"]
        relative_diff = abs(kt0 - kt2) / kt2
        self.assertLess(relative_diff, 0.03,
            f"two mirror-symmetric Free holes flanking an on-axis Fixed hole must have equal "
            f"Kt by physical symmetry: kt0={kt0}, kt2={kt2}, relative_diff={relative_diff} - "
            f"a large difference here is the real rigid-body-pin asymmetry bug, not noise")

    def test_solve_mixed_bc_stress_field_is_gauge_invariant_to_the_pin_choice(self):
        # Direct proof the CURRENT (symmetric) pin scheme still only removes rigid-body motion
        # and does not itself perturb the physical stress field - solved twice from the SAME
        # mesh/load with two different (but each individually valid, statically-determinate)
        # pin choices, asserting the resulting STRESS fields agree to near machine precision
        # (not just "close" - a genuine gauge-invariance identity). This is the same property
        # that made the old asymmetric scheme's bug invisible for every all-Free geometry -
        # proving it still holds for the free DOFs generally, independent of which specific
        # zero-reaction gauge is chosen.
        from finite_plate_reference import cst_matrix as _cst_matrix, element_pcg as _element_pcg
        from multi_hole_reference import _mesh_with_delaunay

        half_w, half_h = 0.10, 0.05
        holes = [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)]
        young, nu, traction = 71.7e9, 0.33, 69e6
        _, _, nodes, triangles = _mesh_with_delaunay(half_w, half_h, holes, 24, 3, 1.0)
        B, D, _, stiffness = _cst_matrix(nodes[triangles], young, nu)
        dofs = np.stack((2 * triangles, 2 * triangles + 1), axis=-1).reshape(-1, 6)
        force = np.zeros(2 * len(nodes))
        for side, x_target in ((-1, -half_w), (1, half_w)):
            on_edge = np.flatnonzero(np.abs(nodes[:, 0] - x_target) < 1e-9 * half_w)
            order = np.argsort(nodes[on_edge, 1])
            edge_nodes = on_edge[order]
            ys = nodes[edge_nodes, 1]
            seg_len = np.diff(ys)
            nodal_force = np.zeros(len(edge_nodes))
            nodal_force[:-1] += seg_len / 2
            nodal_force[1:] += seg_len / 2
            force[2 * edge_nodes] += side * traction * nodal_force

        def stress_for(free):
            displacement, _, _, _ = _element_pcg(stiffness, dofs, force.copy(), free)
            strains = np.einsum("eij,ej->ei", B, displacement[dofs])
            return strains @ D.T

        left_edge = np.flatnonzero(np.abs(nodes[:, 0] + half_w) < 1e-9 * half_w)
        right_edge = np.flatnonzero(np.abs(nodes[:, 0] - half_w) < 1e-9 * half_w)
        bottom_edge = np.flatnonzero(np.abs(nodes[:, 1] + half_h) < 1e-9 * half_h)
        pin_left = left_edge[np.argmin(np.abs(nodes[left_edge, 1]))]
        pin_right = right_edge[np.argmin(np.abs(nodes[right_edge, 1]))]
        pin_bottom = bottom_edge[np.argmin(np.abs(nodes[bottom_edge, 0]))]

        free_current = np.ones(len(force), dtype=bool)
        free_current[2 * pin_left + 1] = False
        free_current[2 * pin_right + 1] = False
        free_current[2 * pin_bottom] = False
        stress_current = stress_for(free_current)

        # A DIFFERENT valid symmetric gauge: ux=0 at bottom-mid, uy=0 at left-mid AND
        # top-mid (still a mirror-symmetric-about-x=0 pair is not required for THIS
        # gauge-invariance check - any two statically-determinate zero-reaction gauges
        # must agree on the stress field, symmetric or not).
        top_edge = np.flatnonzero(np.abs(nodes[:, 1] - half_h) < 1e-9 * half_h)
        pin_top = top_edge[np.argmin(np.abs(nodes[top_edge, 0]))]
        free_alt = np.ones(len(force), dtype=bool)
        free_alt[2 * pin_left + 1] = False
        free_alt[2 * pin_top + 1] = False
        free_alt[2 * pin_bottom] = False
        stress_alt = stress_for(free_alt)

        max_diff = np.abs(stress_current - stress_alt).max()
        scale = np.abs(stress_current).max()
        self.assertLess(max_diff / scale, 1e-6,
            f"two different valid zero-reaction gauge choices must give the same physical "
            f"stress field: max_diff={max_diff}, scale={scale}")


if __name__ == "__main__":
    unittest.main()
