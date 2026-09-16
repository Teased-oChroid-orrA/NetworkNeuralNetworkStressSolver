#!/usr/bin/env python3
"""Independent plane-stress CST reference for a centered hole under uniaxial tension.

The quarter plate uses exact rectangular outer edges, an inscribed polygonal hole,
and symmetry constraints (ux=0 on x=0, uy=0 on y=0). Geometric radial grading
resolves the hole. Only NumPy is required: a coalesced sparse matrix and Jacobi
preconditioned conjugate gradients avoid quadratic dense storage.

Boundary stress, stress at the production FD-safe ring, and central-FD stress
from interpolated FEM displacement are distinct reported quantities. The last
uses the same physical offsets as L5, without importing PINN code. Mesh
convergence is required for every Kt; it is evidence, not an error bound.
"""
import argparse
import json
import math
import time
from dataclasses import dataclass

import numpy as np


def validate(half_w, half_h, radius, young, nu, traction, n_theta, n_radial):
    if not all(math.isfinite(v) for v in (half_w, half_h, radius, young, nu, traction)):
        raise ValueError("geometry, material, and traction must be finite")
    if half_w <= 0 or half_h <= 0 or not 0 < radius < min(half_w, half_h):
        raise ValueError("the centered hole must be strictly inside a positive rectangle")
    if young <= 0 or not -1 < nu < 0.5 or traction <= 0:
        raise ValueError("require E>0, -1<nu<0.5, and positive uniaxial tensile traction")
    if n_theta < 16 or n_theta % 4 or n_radial < 2:
        raise ValueError("require n_theta>=16 divisible by four, n_radial>=2")


def ray_box_radius(theta, half_w, half_h):
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    return min(half_w / max(c, 1e-15), half_h / max(s, 1e-15))


def mesh(half_w, half_h, radius, n_theta, n_radial):
    """Quarter mesh; insert the rectangle corner even for non-square plates."""
    validate(half_w, half_h, radius, 1.0, 0.3, 1.0, n_theta, n_radial)
    angles = np.linspace(0, math.pi / 2, n_theta // 4 + 1)
    corner = math.atan2(half_h, half_w)
    if not np.any(np.isclose(angles, corner, atol=1e-13, rtol=0)):
        angles = np.sort(np.append(angles, corner))
    outer = np.asarray([ray_box_radius(t, half_w, half_h) for t in angles])
    radii = radius * (outer[None, :] / radius) ** np.linspace(0, 1, n_radial + 1)[:, None]
    nodes = np.stack((radii * np.cos(angles), radii * np.sin(angles)), axis=-1).reshape(-1, 2)
    nodes[np.abs(nodes) < 1e-14 * max(half_w, half_h)] = 0
    width = len(angles)
    triangles = []
    for j in range(n_radial):
        for i in range(width - 1):
            a, b = j * width + i, j * width + i + 1
            c, d = (j + 1) * width + i, (j + 1) * width + i + 1
            # Alternate diagonals to avoid a single preferred shear direction.
            triangles.extend(((a, c, d), (a, d, b)) if (i + j) % 2 == 0 else ((a, c, b), (b, c, d)))
    return nodes, np.asarray(triangles), angles


def cst_matrix(points, young, nu):
    """Engineering shear gamma_xy=du/dy+dv/dx; supports batched elements."""
    points = np.asarray(points, dtype=float)
    x, y = points[..., :, 0], points[..., :, 1]
    twice_area = (x[..., 1] - x[..., 0]) * (y[..., 2] - y[..., 0]) - (x[..., 2] - x[..., 0]) * (y[..., 1] - y[..., 0])
    if np.any(twice_area <= 0):
        raise ValueError("mesh contains an inverted or degenerate triangle")
    b = np.stack((y[..., 1] - y[..., 2], y[..., 2] - y[..., 0], y[..., 0] - y[..., 1]), axis=-1) / twice_area[..., None]
    c = np.stack((x[..., 2] - x[..., 1], x[..., 0] - x[..., 2], x[..., 1] - x[..., 0]), axis=-1) / twice_area[..., None]
    B = np.zeros(points.shape[:-2] + (3, 6))
    B[..., 0, 0::2], B[..., 1, 1::2] = b, c
    B[..., 2, 0::2], B[..., 2, 1::2] = c, b
    D = young / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    area = twice_area / 2
    return B, D, area, area[..., None, None] * (np.swapaxes(B, -1, -2) @ D @ B)


def sparse_pcg(rows, columns, values, force, free, rtol=1e-10):
    """Coalesce element entries, then solve only free DOFs; O(nonzeros) storage."""
    n = len(force)
    keys = rows.astype(np.int64) * n + columns
    order = np.argsort(keys)
    keys, values = keys[order], values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    values = np.add.reduceat(values, starts)
    rows, columns = keys[starts] // n, keys[starts] % n
    diagonal = np.bincount(rows[rows == columns], weights=values[rows == columns], minlength=n)
    if np.any(diagonal[free] <= 0):
        raise ValueError("non-positive stiffness diagonal")

    def matvec(vector):
        return np.bincount(rows, weights=values * vector[columns], minlength=n)

    x, residual = np.zeros(n), force.copy()
    residual[~free] = 0
    rhs_norm = np.linalg.norm(residual)
    if rhs_norm == 0:
        raise ValueError("no nonzero load on free DOFs")
    inv_diagonal = np.zeros(n)
    inv_diagonal[free] = 1 / diagonal[free]
    z = residual * inv_diagonal
    direction, rz = z.copy(), float(residual @ z)
    for iteration in range(1, 20 * n + 1):
        product = matvec(direction)
        product[~free] = 0
        curvature = float(direction @ product)
        if curvature <= 0 or not math.isfinite(curvature):
            raise RuntimeError("PCG encountered non-positive curvature")
        alpha = rz / curvature
        x += alpha * direction
        residual -= alpha * product
        relative = np.linalg.norm(residual) / rhs_norm
        if relative < rtol:
            # Check the actual residual rather than trusting the recurrence.
            actual = matvec(x) - force
            actual_relative = np.linalg.norm(actual[free]) / rhs_norm
            if actual_relative > 10 * rtol:
                raise RuntimeError(f"PCG true relative residual too large: {actual_relative}")
            return x, actual, iteration, actual_relative
        z = residual * inv_diagonal
        next_rz = float(residual @ z)
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    raise RuntimeError("PCG failed to converge")


@dataclass
class Solution:
    nodes: np.ndarray
    triangles: np.ndarray
    displacement: np.ndarray
    stress: np.ndarray
    constitutive: np.ndarray
    traction: float
    radius: float
    iterations: int
    relative_residual: float
    relative_load_error: float
    relative_work_error: float

    def sample(self, points):
        """Piecewise linear displacement and element stress at physical points.

        Reflection uses the quarter-plate symmetries. On an element interface,
        average incident stresses; displacement is continuous there.
        """
        vertices = self.nodes[self.triangles]
        minimum, maximum = vertices.min(axis=1), vertices.max(axis=1)
        origin = vertices[:, 0]
        edge1, edge2 = vertices[:, 1] - origin, vertices[:, 2] - origin
        det = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
        displacements, stresses = [], []
        for point in np.asarray(points):
            sign = np.where(point < 0, -1, 1)
            p = np.abs(point)
            candidates = np.flatnonzero(np.all((p >= minimum - 1e-14) & (p <= maximum + 1e-14), axis=1))
            diff = p - origin[candidates]
            w1 = (diff[:, 0] * edge2[candidates, 1] - diff[:, 1] * edge2[candidates, 0]) / det[candidates]
            w2 = (edge1[candidates, 0] * diff[:, 1] - edge1[candidates, 1] * diff[:, 0]) / det[candidates]
            weights = np.stack((1 - w1 - w2, w1, w2), axis=-1)
            inside = np.all(weights >= -1e-10, axis=1)
            selected, weights = candidates[inside], weights[inside]
            if not len(selected):
                raise ValueError(f"point {point.tolist()} lies outside the FEM domain")
            uv = np.einsum("ei,eij->ej", weights, self.displacement[self.triangles[selected]]).mean(axis=0)
            stress = self.stress[selected].mean(axis=0)
            displacements.append(uv * sign)
            stresses.append(stress * [1, 1, sign.prod()])
        return np.asarray(displacements), np.asarray(stresses)

    def profile(self, radius, n_angles=360, fd_step=None):
        theta = np.arange(n_angles) * 2 * math.pi / n_angles
        points = radius * np.stack((np.cos(theta), np.sin(theta)), axis=1)
        _, stress = self.sample(points)
        if fd_step is not None:
            hx, hy = fd_step
            if min(hx, hy) <= 0 or radius - max(hx, hy) <= self.radius:
                raise ValueError("FD stencil must have positive offsets and clear the hole")
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


def solve(half_w, half_h, radius, young, nu, traction, n_theta, n_radial):
    validate(half_w, half_h, radius, young, nu, traction, n_theta, n_radial)
    nodes, triangles, angles = mesh(half_w, half_h, radius, n_theta, n_radial)
    B, D, areas, stiffness = cst_matrix(nodes[triangles], young, nu)
    dofs = np.stack((2 * triangles, 2 * triangles + 1), axis=-1).reshape(-1, 6)
    rows = np.broadcast_to(dofs[:, :, None], stiffness.shape).ravel()
    columns = np.broadcast_to(dofs[:, None, :], stiffness.shape).ravel()
    force = np.zeros(2 * len(nodes))
    width, load_length = len(angles), 0.0
    for i in range(width - 1):
        a, b = n_radial * width + i, n_radial * width + i + 1
        p, q = nodes[a], nodes[b]
        if np.all(np.abs([p[0] - half_w, q[0] - half_w]) < 1e-12 * half_w):
            length = np.linalg.norm(q - p)
            force[2 * a] += traction * length / 2
            force[2 * b] += traction * length / 2
            load_length += length
    load_error = abs(load_length - half_h) / half_h
    if load_error > 1e-12:
        raise ValueError("right boundary load does not cover its exact physical length")
    free = np.ones(len(force), dtype=bool)
    free[2 * np.flatnonzero(nodes[:, 0] == 0)] = False
    free[2 * np.flatnonzero(nodes[:, 1] == 0) + 1] = False
    displacement, reaction, iterations, residual = sparse_pcg(rows, columns, stiffness.ravel(), force, free)
    strains = np.einsum("eij,ej->ei", B, displacement[dofs])
    stress = strains @ D.T
    twice_energy = float(np.sum(areas * np.einsum("ei,ei->e", strains, stress)))
    work = float(displacement @ force)
    work_error = abs(twice_energy - work) / abs(work)
    balance = reaction.reshape(-1, 2).sum(axis=0) + force.reshape(-1, 2).sum(axis=0)
    if np.linalg.norm(balance) / (traction * half_h) > 1e-7 or work_error > 1e-7:
        raise RuntimeError("FEM reaction balance or strain-energy identity failed")
    return Solution(nodes, triangles, displacement.reshape(-1, 2), stress, D, traction,
                    radius, iterations, residual, load_error, work_error)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--half-width", type=float, default=0.10)
    p.add_argument("--half-height", type=float, default=0.10)
    p.add_argument("--radius", type=float, default=0.005)
    p.add_argument("--young", type=float, default=71.7e9)
    p.add_argument("--poisson", type=float, default=0.33)
    p.add_argument("--traction", type=float, default=69e6)
    p.add_argument("--fd-h", type=float, default=1e-3, help="normalized central FD offset used by L5")
    p.add_argument("--levels", default="128x32,256x64,512x128")
    p.add_argument("--max-relative-change", type=float, default=0.02)
    args = p.parse_args()
    if not math.isfinite(args.fd_h) or args.fd_h <= 0 or not 0 < args.max_relative_change < 1:
        p.error("require finite positive fd-h and max-relative-change in (0,1)")
    probe_radius = args.radius + 4 * args.fd_h * max(args.half_width, args.half_height)
    fd_step = (args.fd_h * args.half_width, args.fd_h * args.half_height)
    if probe_radius + max(fd_step) >= min(args.half_width, args.half_height):
        p.error("the FD-safe ring and its stencils must fit inside the rectangle")
    levels = [tuple(int(x) for x in level.split("x")) for level in args.levels.split(",")]
    if len(levels) < 3 or any(len(level) != 2 for level in levels) or any(
            b[0] <= a[0] or b[1] <= a[1] for a, b in zip(levels, levels[1:])):
        p.error("at least three strictly increasing angular and radial mesh levels are required")
    previous, changes = None, []
    for nt, nr in levels:
        start = time.monotonic()
        solution = solve(args.half_width, args.half_height, args.radius, args.young,
                         args.poisson, args.traction, nt, nr)
        metrics = {}
        for name, rad, fd in (("boundary", args.radius, None), ("probe", probe_radius, None),
                              ("probe_fd", probe_radius, fd_step)):
            for key, value in solution.profile(rad, fd_step=fd).items():
                metrics[f"{name}_{key}"] = value
        changes.append(None if previous is None else max(abs(metrics[k] - previous[k]) / abs(metrics[k]) for k in metrics))
        print(json.dumps(dict(mesh=f"{nt}x{nr}", nodes=len(solution.nodes),
                              probe_radius_m=probe_radius, **metrics,
                              max_relative_change=changes[-1], pcg_iterations=solution.iterations,
                              relative_linear_residual=solution.relative_residual,
                              relative_work_error=solution.relative_work_error,
                              seconds=time.monotonic() - start)), flush=True)
        previous = metrics
    if any(change > args.max_relative_change for change in changes[-2:]):
        raise SystemExit("reference not converged on two consecutive refinements; refine before using it")
    print("All six Kt metrics pass two successive mesh-change gates; this is not a rigorous error bound.")


if __name__ == "__main__":
    main()
