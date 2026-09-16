"""Mechanics and geometry checks independent of the PINN implementation."""
import math
import unittest

import numpy as np

from finite_plate_reference import cst_matrix, mesh, solve, validate


class FinitePlateReferenceTests(unittest.TestCase):
    def test_affine_patch_and_rigid_rotation(self):
        points = np.array([[0.1, -0.4], [1.3, 0.2], [-0.2, 0.8]])
        B, D, area, stiffness = cst_matrix(points, 71.7e9, 0.33)
        # Unequal cross derivatives expose a broken engineering-shear row.
        gradient = np.array([[0.012, -0.003], [0.008, -0.007]])
        displacement = (points @ gradient.T + [0.4, -0.7]).ravel()
        expected = np.array([0.012, -0.007, 0.005])
        np.testing.assert_allclose(B @ displacement, expected, rtol=1e-13, atol=1e-15)
        self.assertAlmostEqual(float(displacement @ stiffness @ displacement) / float(area * expected @ D @ expected), 1.0, places=10)
        rotation = 0.017 * np.stack((-points[:, 1], points[:, 0]), axis=1)
        np.testing.assert_allclose(B @ rotation.ravel(), 0, atol=1e-17)
        np.testing.assert_allclose(B @ np.tile([0.2, -0.7], 3), 0, atol=1e-15)

    def test_nonsquare_rectangle_has_exact_corner_and_positive_elements(self):
        nodes, triangles, angles = mesh(0.13, 0.07, 0.004, 32, 8)
        self.assertTrue(np.any(np.linalg.norm(nodes - [0.13, 0.07], axis=1) < 1e-14))
        self.assertTrue(np.any(np.isclose(angles, math.atan2(0.07, 0.13))))
        _, _, areas, _ = cst_matrix(nodes[triangles], 1, 0.33)
        expected = 0.13 * 0.07 - 0.5 * 0.004**2 * np.sin(np.diff(angles)).sum()
        self.assertAlmostEqual(areas.sum(), expected, places=15)

    def test_solution_balance_symmetry_and_material_scaling(self):
        first = solve(0.1, 0.1, 0.005, 71.7e9, 0.33, 69e6, 32, 12)
        second = solve(0.1, 0.1, 0.005, 2 * 71.7e9, 0.33, 2 * 69e6, 32, 12)
        np.testing.assert_allclose(first.displacement, second.displacement, atol=1e-14)
        np.testing.assert_allclose(first.stress * 2, second.stress, rtol=1e-8, atol=1e-4)
        self.assertLess(first.relative_residual, 1e-9)
        self.assertLess(first.relative_work_error, 1e-9)
        self.assertLess(first.relative_load_error, 1e-12)
        uv, stress = first.sample([[0.02, 0.03], [-0.02, 0.03], [0.02, -0.03]])
        np.testing.assert_allclose(uv[1], uv[0] * [-1, 1])
        np.testing.assert_allclose(uv[2], uv[0] * [1, -1])
        np.testing.assert_allclose(stress[1], stress[0] * [1, 1, -1])
        self.assertGreater(first.profile(0.0054, fd_step=(0.0001, 0.0001))["kt_vm"], 2)

    def test_invalid_geometry_material_and_load_are_rejected(self):
        valid = [0.1, 0.1, 0.005, 71.7e9, 0.33, 69e6, 32, 8]
        for index, value in ((0, 0), (2, 0), (2, 0.1), (3, -1), (4, 0.5),
                             (5, 0), (5, -1), (6, 18), (7, 1), (0, float("nan"))):
            args = valid.copy()
            args[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                validate(*args)


if __name__ == "__main__":
    unittest.main()
