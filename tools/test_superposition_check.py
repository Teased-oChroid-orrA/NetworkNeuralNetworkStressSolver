"""Structural/regression checks for `superposition_check.py`."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from superposition_check import kirsch_stress_cartesian, superposition_kt


class SuperpositionCheckTests(unittest.TestCase):
    def test_single_hole_recovers_the_classical_kt_three(self):
        # No other hole to perturb the field - this must reduce to the exact isolated-hole
        # Kirsch solution, whose textbook peak hoop/von-Mises stress ratio is exactly 3.0.
        result = superposition_kt([(0.0, 0.0, 0.01)], S=69e6)
        self.assertAlmostEqual(result[0]["kt_hoop"], 3.0, places=3)
        self.assertAlmostEqual(result[0]["kt_vm"], 3.0, places=3)

    def test_isolated_hole_boundary_is_traction_free(self):
        # sigma_rr and sigma_r_theta must vanish exactly at r=a for the isolated Kirsch
        # solution (the boundary condition this whole formula is built to satisfy).
        a, S = 0.01, 69e6
        theta = np.linspace(0, 2 * np.pi, 37)
        x, y = a * np.cos(theta), a * np.sin(theta)
        sxx, syy, sxy = kirsch_stress_cartesian(x, y, a, S)
        c, s = np.cos(theta), np.sin(theta)
        srr = sxx * c * c + syy * s * s + 2 * sxy * s * c
        srt = (syy - sxx) * s * c + sxy * (c * c - s * s)
        np.testing.assert_allclose(srr, 0, atol=1e-6 * S)
        np.testing.assert_allclose(srt, 0, atol=1e-6 * S)

    def test_two_well_separated_holes_stay_close_to_isolated_kt(self):
        # Real, measured regression guard: this exact geometry (notched_plate.toml's own),
        # checked against multi_hole_reference.py's real FEM ground truth in this session,
        # landed within ~3.4% relative error at zeroth-order superposition (see
        # docs/multi-hole-fem-ground-truth-investigation.md's Phase B result). This test only
        # re-asserts the superposition estimate itself stays stable, not the FEM comparison
        # (that needs scipy - see the doc for the real cross-checked numbers).
        holes = [(-0.03, 0.0, 0.01), (0.03, 0.0, 0.008)]
        result = superposition_kt(holes, S=69e6)
        self.assertAlmostEqual(result[0]["kt_vm"], 2.9598, places=3)
        self.assertAlmostEqual(result[1]["kt_vm"], 2.9336, places=3)

    def test_moving_holes_farther_apart_reduces_the_perturbation(self):
        close = superposition_kt([(-0.02, 0.0, 0.01), (0.02, 0.0, 0.01)], S=69e6)
        far = superposition_kt([(-0.10, 0.0, 0.01), (0.10, 0.0, 0.01)], S=69e6)
        # Farther apart -> each hole's cross-perturbation on the other shrinks -> closer to the
        # isolated Kt=3.0 than the close case.
        self.assertGreater(abs(close[0]["kt_vm"] - 3.0), abs(far[0]["kt_vm"] - 3.0))


if __name__ == "__main__":
    unittest.main()
