//! Issue #77: exact, closed-form hard-constraint ansatz for the hole traction-free condition.
//!
//! Ten independently-evidenced training-time interventions (see
//! `docs/PHASE_4_IMPLEMENTATION_MANIFEST.md` PH4-24 through PH4-34b) failed to close the L5
//! Kt gap. The common thread across every failure: `hole_free` (the SOFT penalty enforcing
//! traction-free-at-the-hole-boundary) competes with the domain's own energy terms for
//! gradient budget, and rebalancing that competition (weight changes, added local residuals,
//! richer input/activation representations) never worked, because the competition itself —
//! not any tunable inside it — is the problem. This module removes the competition entirely:
//! the traction-free condition is satisfied EXACTLY, for any network weights, by construction
//! — a hard constraint, not a soft target.
//!
//! ## The construction
//!
//! `u_hole(x,y) = kirsch_hole_displacement(x,y; px,py,a,E,nu)  +  traction_free_envelope(x,y;
//! a) * NN(x,y)`
//!
//! - [`kirsch_hole_displacement`] is the EXACT, closed-form classical Kirsch hole-correction
//!   displacement field (Timoshenko & Goodier's stress solution, independently re-derived and
//!   verified here — not copied from a remembered displacement formula, which is easy to get
//!   subtly wrong). Combined with `u_affine` (already added elsewhere in this codebase, see
//!   `user_problem::affine_strain`'s doc comment), the TOTAL field `u_affine +
//!   kirsch_hole_displacement` is traction-free at `r=a` EXACTLY, for the specific `px,py`
//!   this problem's load applies — by construction of the classical solution, not by training.
//! - [`traction_free_envelope`] is a scalar multiplier on the network's OWN free output `NN`,
//!   equal to `0` AND with zero radial derivative at `r=a` — so `NN`'s contribution to STRAIN
//!   (a first derivative of displacement) at `r=a` is exactly zero too, meaning `NN` cannot
//!   disturb the exact traction-free condition no matter what it computes. `NN` is not wasted:
//!   away from the hole it is free to represent whatever correction is needed to satisfy the
//!   ANNULUS/OUTER interface continuity terms (unaffected by this module) and any deviation of
//!   this FINITE plate's true solution from the idealized-infinite-plate Kirsch correction.
//!
//! ## Why this is not "hand the network the answer"
//!
//! `kirsch_hole_displacement` is the exact solution for an INFINITE plate. This plate is
//! FINITE (see PH4-33's own real result: the finite-plate FEM reference, 2.4606, differs from
//! the idealized infinite-plate Kirsch value, 3.0). The gap between "idealized infinite-plate
//! correction" and "the true finite-plate solution" — plus whatever is needed to satisfy
//! interface continuity with the outer domain at `r=3a` — is exactly what `NN` still has to
//! learn. The hard constraint removes ONLY the traction-free condition from gradient
//! competition; it does not remove the actual physics problem.
//!
//! ## Verification discipline
//!
//! Every closed-form formula below was derived via independent symbolic computation (sympy),
//! not recalled from memory or a textbook displacement formula: (1) start from the STANDARD,
//! independently-checkable Kirsch STRESS solution; (2) integrate strain-displacement relations
//! to get displacement; (3) verify the displacement satisfies strain COMPATIBILITY exactly
//! (this is a real, independent check — an invalid stress field would fail it); (4)
//! re-differentiate the resulting displacement and confirm it reproduces the ORIGINAL stress
//! formula exactly; (5) confirm zero traction at `r=a` via the re-derived stress, not just the
//! original. The general-biaxial (`px`, `py`) closed form was obtained by superposing two
//! orthogonal single-axis solutions and independently checked against the known equal-biaxial
//! (`px=py`) special case. See `docs/PHASE_4_IMPLEMENTATION_MANIFEST.md` PH4-35 for the full
//! derivation record. [`tests`] below cross-checks this Rust transcription against numeric
//! reference values computed independently in Python/sympy — a transcription bug here would
//! silently fake the entire hard constraint, so this is treated as load-bearing, not optional.

/// Exact closed-form hole-correction displacement, superposed on `u_affine`
/// (`user_problem::affine_strain`'s corresponding displacement) to make the TOTAL field
/// traction-free at `r=a` for a centered circular hole under remote biaxial tension
/// (`px` along x, `py` along y, no shear — matching this codebase's `LoadConfig`).
///
/// `x,y`: position relative to the hole center, meters. `a`: hole radius, meters. `e,nu`:
/// material Young's modulus (Pa) and Poisson's ratio. `px,py`: remote/far-field normal
/// tractions, Pa (same convention as `affine_strain`/`LoadConfig`).
///
/// Returns `(u_hole_x, u_hole_y)` in meters. Singular at `x=y=0` (the hole center itself,
/// inside the hole — never a valid collocation point in this codebase, which excludes the
/// hole's interior and keeps every stencil arm outside it by construction).
pub fn kirsch_hole_displacement(x: f64, y: f64, a: f64, e: f64, nu: f64, px: f64, py: f64) -> (f64, f64) {
    let a2 = a * a;
    let r2 = x * x + y * y;
    let px_a2 = px * a2;
    let py_a2 = py * a2;
    let nu_px_a2 = nu * px_a2;
    let nu_py_a2 = nu * py_a2;
    let px_r2 = px * r2;
    let py_r2 = py * r2;

    // u_hole_x: the "4*(...)" cross terms use 4*y^2 (the x-branch); u_hole_y's use 4*x^2 (the
    // y-branch) - these are genuinely different quantities, kept as separately named locals
    // rather than reusing one shared "4*coord^2" variable, to avoid a silent x/y mixup.
    let four_y2 = 4.0 * y * y;
    let px_4y2 = px * four_y2;
    let py_4y2 = py * four_y2;
    let nu_px_4y2 = nu * px_4y2;
    let nu_py_4y2 = nu * py_4y2;
    let u_hole_x = 0.5 * x * a2
        * (r2 * (nu * px_r2 + nu * py_r2 - nu_px_4y2 + nu_py_4y2 + 5.0 * px_r2 - 3.0 * py_r2
                 - px_a2 + py_a2 - nu_px_a2 + nu_py_a2 - px_4y2 + py_4y2)
            + px_a2 * four_y2 - py_a2 * four_y2 + nu_px_a2 * four_y2 - nu_py_a2 * four_y2)
        / (e * r2 * r2 * r2);

    let four_x2 = 4.0 * x * x;
    let px_4x2 = px * four_x2;
    let py_4x2 = py * four_x2;
    let nu_px_4x2 = nu * px_4x2;
    let nu_py_4x2 = nu * py_4x2;
    let u_hole_y = 0.5 * a2 * y
        * (r2 * (nu * px_r2 + nu * py_r2 + nu_px_4x2 - nu_py_4x2 - 3.0 * px_r2 + 5.0 * py_r2
                 + px_a2 - py_a2 + nu_px_a2 - nu_py_a2 + px_4x2 - py_4x2)
            - px_a2 * four_x2 + py_a2 * four_x2 - nu_px_a2 * four_x2 + nu_py_a2 * four_x2)
        / (e * r2 * r2 * r2);

    (u_hole_x, u_hole_y)
}

/// Scalar envelope multiplying the network's own free output before it is added to the
/// closed-form correction — `0` AND zero radial derivative at `r=a`, so the network's own
/// contribution to STRAIN at the hole boundary is exactly zero regardless of what it predicts
/// (a first derivative of `0*NN` at a point where the multiplier's own derivative is also zero
/// vanishes by the product rule: `d(phi*NN)/dr = phi'*NN + phi*NN'`, and both `phi(a)=0` and
/// `phi'(a)=0` here). Saturates smoothly to `1` (no suppression) away from the hole, using the
/// hole's own radius as the natural length scale — `phi(3a) ~= 0.98`, so the network is
/// essentially unconstrained by the time it reaches the annulus/outer interface at `r=3a`.
///
/// `phi(r) = 1 - exp(-((r-a)/a)^2)`. `phi(a)=1-exp(0)=0`.
/// `phi'(r) = (2*(r-a)/a^2) * exp(-((r-a)/a)^2)`, which has an explicit `(r-a)` factor, so
/// `phi'(a)=0` too — confirmed directly from the closed form, not just asserted.
pub fn traction_free_envelope(x: f64, y: f64, a: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    let u = (r - a) / a;
    1.0 - (-(u * u)).exp()
}

/// [`pinn_core::problem::DirichletAnsatz`] for the annulus domain's hard-constraint mode:
/// `eval` returns [`traction_free_envelope`] (same value for both u,v columns — a single
/// scalar suppression, not independently tuned per axis) as the multiplicative scale on the
/// network's own free output; `additive` returns [`kirsch_hole_displacement`] converted from
/// this ansatz's stored `u_ref` back to the "pre-u_ref-rescale" units `compute_domain_forwards`
/// operates in at the point this is applied (see that function's own `u_col`/`v_col`
/// construction — the multiplicative and additive contributions are summed there BEFORE the
/// later `.mul_scalar(u_ref)` converts the whole displacement column back to physical meters,
/// so this struct must pre-divide by the SAME `u_ref` to land in physical meters correctly
/// after that rescale, not before it).
///
/// `(xn, yn)` arrive normalized to `[-1,1]^2` (`DirichletAnsatz`'s own contract); this struct
/// converts back to physical coordinates relative to the hole center using `half_w`/`half_h`
/// (`UserGeometry::to_placeholder`'s own normalization, `plate_normalize_point`'s inverse)
/// before evaluating either closed form.
pub struct HoleTractionFreeAnsatz {
    pub hole_center: [f64; 2],
    pub hole_radius: f64,
    pub half_w: f64,
    pub half_h: f64,
    pub px: f64,
    pub py: f64,
    pub e: f64,
    pub nu: f64,
    pub u_ref: f64,
}

impl HoleTractionFreeAnsatz {
    fn physical_xy(&self, xn: f32, yn: f32) -> (f64, f64) {
        (xn as f64 * self.half_w - self.hole_center[0], yn as f64 * self.half_h - self.hole_center[1])
    }
}

impl pinn_core::problem::DirichletAnsatz for HoleTractionFreeAnsatz {
    fn eval(&self, xn: f32, yn: f32, _k: f32) -> (f32, f32) {
        let (x, y) = self.physical_xy(xn, yn);
        let phi = traction_free_envelope(x, y, self.hole_radius) as f32;
        (phi, phi)
    }
    fn additive(&self, xn: f32, yn: f32) -> (f32, f32) {
        let (x, y) = self.physical_xy(xn, yn);
        let (ux, uy) = kirsch_hole_displacement(x, y, self.hole_radius, self.e, self.nu, self.px, self.py);
        ((ux / self.u_ref) as f32, (uy / self.u_ref) as f32)
    }
}

/// The annulus domain's ansatz, selectable between the byte-identical default (`Identity`,
/// no behavior change from every pre-PH4-35 caller) and the new hard-constraint mode. An enum
/// rather than a trait object (`Box<dyn DirichletAnsatz>`) — matches this codebase's existing
/// preference for concrete types where a small, closed set of variants is all that's needed.
pub enum AnnulusAnsatz {
    Identity,
    HardConstraint(HoleTractionFreeAnsatz),
}

impl pinn_core::problem::DirichletAnsatz for AnnulusAnsatz {
    fn eval(&self, xn: f32, yn: f32, k: f32) -> (f32, f32) {
        match self {
            AnnulusAnsatz::Identity => (1.0, 1.0),
            AnnulusAnsatz::HardConstraint(a) => a.eval(xn, yn, k),
        }
    }
    fn additive(&self, xn: f32, yn: f32) -> (f32, f32) {
        match self {
            AnnulusAnsatz::Identity => (0.0, 0.0),
            AnnulusAnsatz::HardConstraint(a) => a.additive(xn, yn),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const E: f64 = 71.7e9;
    const NU: f64 = 0.33;

    /// Cross-check against independently-computed sympy reference values (see PH4-35's
    /// derivation record) — catches a transcription bug in the CSE'd Rust formula above, which
    /// would otherwise silently fake the entire hard constraint with no visible symptom.
    #[test]
    fn kirsch_hole_displacement_matches_sympy_reference_values() {
        let cases: &[(f64, f64, f64, f64, f64, f64, f64)] = &[
            // (x, y, a, e, nu, px, py, expected handled below)
            (0.00955336489125606, 0.0029552020666133954, 0.005, E, NU, 6.9e7, 0.0),
            (0.0045359612142557735, 0.008912073600614355, 0.005, E, NU, 6.9e7, 3.0e7),
            (-0.011060905733118681, 0.010131947708267265, 0.005, E, NU, 5.0e7, -2.0e7),
            (0.005, 0.0, 0.005, E, NU, 6.9e7, 0.0),
            (3.061616997868383e-19, 0.005, 0.005, E, NU, 6.9e7, 0.0),
        ];
        let expected: &[(f64, f64)] = &[
            (5.342690569906143e-06, 4.635745216959231e-07),
            (1.194953516096086e-06, 3.2090566300007345e-07),
            (-1.3586290525068016e-06, -3.9066320945596856e-07),
            (9.623430962343097e-06, -0.0),
            (5.892651962424502e-22, -3.2238493723849374e-06),
        ];
        for (i, &(x, y, a, e, nu, px, py)) in cases.iter().enumerate() {
            let (ux, uy) = kirsch_hole_displacement(x, y, a, e, nu, px, py);
            let (ex, ey) = expected[i];
            assert!((ux - ex).abs() < 1e-9 * ex.abs().max(1e-12),
                "case {i}: u_hole_x mismatch: got {ux:e}, expected {ex:e}");
            assert!((uy - ey).abs() < 1e-9 * ey.abs().max(1e-12),
                "case {i}: u_hole_y mismatch: got {uy:e}, expected {ey:e}");
        }
    }

    /// Direct FD re-derivation of stress from `kirsch_hole_displacement` + the SAME
    /// `affine_strain` this codebase already superposes elsewhere, confirming the TOTAL field
    /// is traction-free at r=a to FD-truncation precision (not just symbolically, in Python —
    /// an independent, in-Rust numeric confirmation that this transcription is correct end to
    /// end, using plain central differences exactly like the FD stencils used everywhere else
    /// in this codebase).
    #[test]
    fn total_field_is_traction_free_at_hole_boundary_numerically() {
        let a = 0.005_f64;
        let px = 6.9e7_f64;
        let py = 2.0e7_f64;
        let h = 1e-7_f64; // tiny physical FD step, independent of this codebase's own fd_h convention

        // Same closed-form plane-stress inverse Hooke's law `user_problem::affine_strain`
        // uses (verified identical there): eps_xx=(px-nu*py)/E, eps_yy=(py-nu*px)/E, eps_xy=0
        // for this codebase's shear-free LoadConfig. Inlined here rather than depending on
        // that private function, since this test only needs the trivial linear-strain
        // displacement (u=eps_xx*x, v=eps_yy*y), not a cross-module dependency.
        let a_exx = (px - NU * py) / E;
        let a_eyy = (py - NU * px) / E;

        let total_u = |x: f64, y: f64| -> (f64, f64) {
            let (hx, hy) = kirsch_hole_displacement(x, y, a, E, NU, px, py);
            (a_exx * x + hx, a_eyy * y + hy)
        };

        // Central-difference strain at several points on the hole boundary r=a.
        for &theta in &[0.0_f64, 0.7, 1.3, std::f64::consts::FRAC_PI_2, 2.1, 3.0, 4.2, 5.5] {
            let x0 = a * theta.cos();
            let y0 = a * theta.sin();
            let (u_c, v_c) = total_u(x0, y0);
            let (u_xp, v_xp) = total_u(x0 + h, y0);
            let (u_xm, v_xm) = total_u(x0 - h, y0);
            let (u_yp, v_yp) = total_u(x0, y0 + h);
            let (u_ym, v_ym) = total_u(x0, y0 - h);
            let _ = (u_c, v_c);

            let exx = (u_xp - u_xm) / (2.0 * h);
            let eyy = (v_yp - v_ym) / (2.0 * h);
            let exy = 0.5 * ((u_yp - u_ym) / (2.0 * h) + (v_xp - v_xm) / (2.0 * h));

            let sxx = E / (1.0 - NU * NU) * (exx + NU * eyy);
            let syy = E / (1.0 - NU * NU) * (eyy + NU * exx);
            let sxy = E / (2.0 * (1.0 + NU)) * (2.0 * exy);

            let nx = theta.cos();
            let ny = theta.sin();
            let traction_x = sxx * nx + sxy * ny;
            let traction_y = sxy * nx + syy * ny;
            let traction_mag = (traction_x * traction_x + traction_y * traction_y).sqrt();

            // Reference scale: px itself (Pa) - traction should be a tiny FRACTION of it,
            // limited only by O(h) central-difference truncation error at h=1e-7.
            assert!(traction_mag / px.abs() < 1e-3,
                "theta={theta}: traction magnitude {traction_mag:e} Pa is not negligible vs px={px:e} Pa");
        }
    }

    #[test]
    fn traction_free_envelope_is_zero_with_zero_derivative_at_hole_boundary() {
        let a = 0.005_f64;
        let h = 1e-9_f64;
        for &theta in &[0.0_f64, 1.0, 2.5, 4.0] {
            let x0 = a * theta.cos();
            let y0 = a * theta.sin();
            let phi_a = traction_free_envelope(x0, y0, a);
            assert!(phi_a.abs() < 1e-12, "phi(a) should be exactly 0, got {phi_a}");

            // Radial derivative via central difference along the outward normal direction.
            let nx = theta.cos();
            let ny = theta.sin();
            let phi_p = traction_free_envelope(x0 + h * nx, y0 + h * ny, a);
            let phi_m = traction_free_envelope(x0 - h * nx, y0 - h * ny, a);
            let dphi_dr = (phi_p - phi_m) / (2.0 * h);
            assert!(dphi_dr.abs() < 1e-3,
                "phi'(a) should be ~0 (a first-order-degenerate root), got {dphi_dr} at theta={theta}");
        }
    }

    #[test]
    fn traction_free_envelope_saturates_toward_one_away_from_hole() {
        let a = 0.005_f64;
        // At r=3a (the annulus/outer interface radius), envelope should be close to 1 - the
        // network's own output should be almost entirely unconstrained there.
        let phi_3a = traction_free_envelope(3.0 * a, 0.0, a);
        assert!(phi_3a > 0.97, "expected phi(3a) close to 1, got {phi_3a}");
        // Far away, envelope -> 1 exactly (within f64 precision).
        let phi_far = traction_free_envelope(100.0 * a, 0.0, a);
        assert!((phi_far - 1.0).abs() < 1e-9, "expected phi(100a) ~= 1, got {phi_far}");
    }

    #[test]
    fn traction_free_envelope_is_bounded_in_zero_one() {
        let a = 0.005_f64;
        for r_mult in [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0, 1000.0] {
            let phi = traction_free_envelope(r_mult * a, 0.0, a);
            assert!((0.0..=1.0).contains(&phi), "phi out of [0,1] at r={r_mult}*a: {phi}");
        }
    }
}
