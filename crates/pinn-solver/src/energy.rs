/// Deep Energy Method (DEM) loss functions.
///
/// Total potential energy (plane stress):
///   Π = (1/N) Σᵢ [½ ε:C:ε] dΩ  −  Neumann work
///
/// The equilibrium residual ∇·σ = 0 is enforced as a soft penalty at near-hole
/// ring points using a 4-meta-position stress-divergence stencil.  This provides
/// non-zero gradient toward the Kirsch solution even when the displacement field
/// has u_net ≈ 0 at the near-hole apex — where DEM energy gradients vanish.

use burn::tensor::{backend::Backend, Tensor};
use pinn_core::material::MaterialProps;

/// Compute plane-stress Cauchy stress components from strains (Hooke's law).
///
/// Returns (σ_xx, σ_yy, σ_xy) as [N] tensors. Single source of truth for the plane-stress
/// constitutive law — used by every energy/loss function below instead of each
/// re-deriving it.
pub fn compute_stress<B: Backend>(
    eps_xx: Tensor<B, 1>,
    eps_yy: Tensor<B, 1>,
    eps_xy: Tensor<B, 1>,
    material: &MaterialProps,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let e  = material.e  as f64;
    let nu = material.nu as f64;
    let factor = e / (1.0 - nu * nu);
    let sxx = (eps_xx.clone() + eps_yy.clone().mul_scalar(nu)).mul_scalar(factor);
    let syy = (eps_yy         + eps_xx        .mul_scalar(nu)).mul_scalar(factor);
    let sxy = eps_xy.mul_scalar(e / (1.0 + nu));
    (sxx, syy, sxy)
}

/// Plane-stress strain energy density: ½(σ_xx·ε_xx + σ_yy·ε_yy + 2·σ_xy·ε_xy).
fn strain_energy_density<B: Backend>(
    eps_xx: Tensor<B, 1>,
    eps_yy: Tensor<B, 1>,
    eps_xy: Tensor<B, 1>,
    material: &MaterialProps,
) -> Tensor<B, 1> {
    let (sxx, syy, sxy) = compute_stress(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), material);
    (sxx * eps_xx + syy * eps_yy + sxy * eps_xy.mul_scalar(2.0)).mul_scalar(0.5)
}

/// Plane-stress strain energy density per point (not averaged).
/// Returns a [N] tensor — used by AMR to identify high-energy (high-residual) regions.
pub fn dem_energy_per_point<B: Backend>(
    eps_xx: Tensor<B, 1>,
    eps_yy: Tensor<B, 1>,
    eps_xy: Tensor<B, 1>,
    material: &MaterialProps,
) -> Tensor<B, 1> {
    strain_energy_density(eps_xx, eps_yy, eps_xy, material)
}

/// Plane-stress strain energy density at N interior collocation points.
pub fn dem_energy_loss<B: Backend>(
    eps_xx: Tensor<B, 1>,
    eps_yy: Tensor<B, 1>,
    eps_xy: Tensor<B, 1>,
    material: &MaterialProps,
) -> Tensor<B, 1> {
    strain_energy_density(eps_xx, eps_yy, eps_xy, material).mean()
}

/// Equilibrium residual loss ‖∇·σ‖² at N_eq near-hole ring points.
///
/// Takes σ components pre-computed at 4 meta-positions (ring shifted ±h):
///   (sxx_xp, syy_xp, sxy_xp) = σ at (ring + hx, ring_y) — for ∂/∂x
///   (sxx_xm, syy_xm, sxy_xm) = σ at (ring - hx, ring_y)
///   (sxx_yp, syy_yp, sxy_yp) = σ at (ring_x, ring + hy) — for ∂/∂y
///   (sxx_ym, syy_ym, sxy_ym) = σ at (ring_x, ring - hy)
///
/// `cx, cy`: 1/(2*h_phys) coefficients for the FD divergence (units: m⁻¹).
/// `ref_div2`: normalisation = (Px·cx)² so output is dimensionless O(1).
pub fn equilibrium_residual_loss<B: Backend>(
    sxx_xp: Tensor<B, 1>, sxy_xp: Tensor<B, 1>,
    sxx_xm: Tensor<B, 1>, sxy_xm: Tensor<B, 1>,
    sxy_yp: Tensor<B, 1>, syy_yp: Tensor<B, 1>,
    sxy_ym: Tensor<B, 1>, syy_ym: Tensor<B, 1>,
    cx: f64,
    cy: f64,
    ref_div2: f64,
) -> Tensor<B, 1> {
    // ∇·σ in x: ∂σ_xx/∂x + ∂σ_xy/∂y
    let eq_x = (sxx_xp - sxx_xm).mul_scalar(cx)
             + (sxy_yp.clone() - sxy_ym.clone()).mul_scalar(cy);
    // ∇·σ in y: ∂σ_xy/∂x + ∂σ_yy/∂y
    let eq_y = (sxy_xp - sxy_xm).mul_scalar(cx)
             + (syy_yp - syy_ym).mul_scalar(cy);

    (eq_x.clone() * eq_x + eq_y.clone() * eq_y)
        .mean()
        .mul_scalar(1.0 / ref_div2)
}

/// Equilibrium residual ‖∇·σ‖² computed from the DISPLACEMENT HESSIAN via σ=C:ε(u), not from
/// the network's direct σ output (see `equilibrium_residual_loss` above, which does the latter
/// and was found functionally inert - bugSource-New #12: the plate's direct-σ output never
/// develops real spatial structure during training, so constraining ∇·σ_direct=0 is trivially
/// already satisfied and provides no real gradient pressure). This is the derived-stress
/// alternative: substitute σ=C:ε(u) into ∇·σ=0 for standard plane-stress elasticity to get,
/// directly in terms of the Hessian of u,v:
///   r_x = factor·(u_xx + ν·v_xy) + G·(u_yy + v_xy)
///   r_y = factor·(v_yy + ν·u_xy) + G·(u_xy + v_xx)
/// `factor = E/(1-ν²)`, `G = E/(2(1+ν))` - identical constants to `compute_stress`, reused not
/// re-derived. `ref_div2`: normalisation so output is dimensionless O(1), same convention as
/// `equilibrium_residual_loss`.
pub fn equilibrium_from_displacement_hessian_loss<B: Backend>(
    u_xx: Tensor<B, 1>, u_yy: Tensor<B, 1>, u_xy: Tensor<B, 1>,
    v_xx: Tensor<B, 1>, v_yy: Tensor<B, 1>, v_xy: Tensor<B, 1>,
    material: &MaterialProps,
    ref_div2: f64,
) -> Tensor<B, 1> {
    let e  = material.e  as f64;
    let nu = material.nu as f64;
    let factor = e / (1.0 - nu * nu);
    let g = e / (2.0 * (1.0 + nu));

    let r_x = (u_xx + v_xy.clone().mul_scalar(nu)).mul_scalar(factor)
            + (u_yy + v_xy.clone()).mul_scalar(g);
    let r_y = (v_yy + u_xy.clone().mul_scalar(nu)).mul_scalar(factor)
            + (u_xy + v_xx).mul_scalar(g);

    (r_x.clone() * r_x + r_y.clone() * r_y)
        .mean()
        .mul_scalar(1.0 / ref_div2)
}

/// Neumann BC penalty: squared traction residual on the load boundary.
pub fn neumann_loss<B: Backend>(
    eps_xx_b: Tensor<B, 1>,
    eps_yy_b: Tensor<B, 1>,
    eps_xy_b: Tensor<B, 1>,
    nx_b: Tensor<B, 1>,
    ny_b: Tensor<B, 1>,
    tx_target: Tensor<B, 1>,
    ty_target: Tensor<B, 1>,
    material: &MaterialProps,
) -> Tensor<B, 1> {
    let (sxx, syy, sxy) = compute_stress(eps_xx_b, eps_yy_b, eps_xy_b, material);

    let tx_pred = sxx * nx_b.clone() + sxy.clone() * ny_b.clone();
    let ty_pred = sxy * nx_b         + syy          * ny_b;

    let ex = tx_pred - tx_target;
    let ey = ty_pred - ty_target;
    (ex.clone() * ex + ey.clone() * ey).mean()
}

/// Hole (free surface) penalty via FD strains: traction on hole boundary must be zero.
pub fn hole_traction_loss<B: Backend>(
    eps_xx_h: Tensor<B, 1>,
    eps_yy_h: Tensor<B, 1>,
    eps_xy_h: Tensor<B, 1>,
    nx_h: Tensor<B, 1>,
    ny_h: Tensor<B, 1>,
    material: &MaterialProps,
) -> Tensor<B, 1> {
    let zero_tx = eps_xx_h.clone().mul_scalar(0.0_f64);
    let zero_ty = eps_yy_h.clone().mul_scalar(0.0_f64);
    neumann_loss(eps_xx_h, eps_yy_h, eps_xy_h, nx_h, ny_h, zero_tx, zero_ty, material)
}

/// Hole traction-free loss using mDEM direct stress outputs (no FD stencil).
///
/// In mDEM mode the network outputs σ directly; no FD strains needed at the hole surface.
/// This eliminates FD stencil artifacts that arise when stencil points cross the hole boundary.
///
/// Returns mean(|σ·n|²) in [Pa²]; caller normalises by ref_stress2 = Px².
pub fn hole_traction_loss_direct<B: Backend>(
    sxx_h: Tensor<B, 1>,
    syy_h: Tensor<B, 1>,
    sxy_h: Tensor<B, 1>,
    nx_h:  Tensor<B, 1>,
    ny_h:  Tensor<B, 1>,
) -> Tensor<B, 1> {
    let tx = sxx_h * nx_h.clone() + sxy_h.clone() * ny_h.clone();
    let ty = sxy_h * nx_h + syy_h * ny_h;
    (tx.clone() * tx + ty.clone() * ty).mean()
}

/// Constitutive consistency: MSE between σ_net and C:ε_fd at interior collocation points.
///
/// Used only in mDEM mode to keep the network's direct stress outputs aligned with
/// the plane-stress constitutive law, preventing σ_net from drifting away from Hooke's law.
///
/// Returns mean(|σ_net − C:ε_fd|²) in [Pa²]; caller normalises by ref_stress2 = Px².
pub fn constitutive_consistency_loss<B: Backend>(
    sxx_net: Tensor<B, 1>,
    syy_net: Tensor<B, 1>,
    sxy_net: Tensor<B, 1>,
    eps_xx:  Tensor<B, 1>,
    eps_yy:  Tensor<B, 1>,
    eps_xy:  Tensor<B, 1>,
    material: &MaterialProps,
) -> Tensor<B, 1> {
    let (sxx_fd, syy_fd, sxy_fd) = compute_stress(eps_xx, eps_yy, eps_xy, material);
    let ex = sxx_net - sxx_fd;
    let ey = syy_net - syy_fd;
    let ez = sxy_net - sxy_fd;
    (ex.clone() * ex + ey.clone() * ey + ez.clone() * ez).mean()
}

/// General-PINN architecture recommendations §31 ("automatic cheating-solution detection"):
/// compare two mathematically-equivalent representations of the same physical quantity and
/// flag divergence — here, direct mDEM σ vs. derived (Hooke) σ at the same point, the exact
/// dual-representation gap this session's Kt investigation found and fixed by hand (bugSource-
/// New #2/#11/#12). Computes the SAME quantity `constitutive_consistency_loss` does
/// (`‖direct − derived‖² / ref_stress2`), but as a plain, non-differentiable, single-point f32
/// function — a standing diagnostic usable even for a problem that never registers
/// `constitutive_consistency` as a training loss at all (§28's "residuals and losses are
/// different concepts": a problem can still ask "are these two representations consistent?" as
/// a health check without training against the answer).
pub fn representation_consistency_check(
    direct: (f32, f32, f32),
    derived: (f32, f32, f32),
    ref_stress2: f32,
) -> f32 {
    let (dxx, dyy, dxy) = (direct.0 - derived.0, direct.1 - derived.1, direct.2 - derived.2);
    (dxx * dxx + dyy * dyy + dxy * dxy) / ref_stress2.max(1e-30)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Wgpu};

    type TB = Autodiff<Wgpu>;

    fn material(e: f64, nu: f64) -> MaterialProps {
        MaterialProps { e, nu, density: 0.0, ultimate_strength_pa: 1.0 }
    }

    fn t1(v: f32) -> Tensor<TB, 1> {
        Tensor::<TB, 1>::from_floats([v], &Default::default())
    }

    /// Zero-cost analytical sanity check (no training, no network) - recommended as the
    /// single highest-value diagnostic in the Kt investigation
    /// (`powershell_tool/CLAUDE.md`): feed the EXACT uniform-uniaxial-tension elasticity
    /// solution through every plate physics loss function and confirm each one reports
    /// what it should. If this test fails, there's an implementation bug independent of
    /// training/optimization; if it passes (it does), the loss math itself is confirmed
    /// correct and the Kt/no-hole convergence problem is an optimization-dynamics question,
    /// not an implementation bug in these functions.
    ///
    /// Exact plane-stress uniaxial tension solution (σ0 along x, free in y):
    ///   u(x,y) = (σ0/E) x,  v(x,y) = -ν(σ0/E) y
    ///   ε_xx = σ0/E,  ε_yy = -ν·σ0/E,  ε_xy = 0
    ///   σ_xx = σ0,  σ_yy = 0,  σ_xy = 0  (uniform - zero divergence everywhere)
    #[test]
    fn analytical_uniform_uniaxial_tension_satisfies_every_plate_loss_term() {
        use crate::fd_stencil::{compute_strains, FdConfig};
        use burn::tensor::TensorData;

        let device = Default::default();
        let e = 71.7e9_f64;
        let nu = 0.33_f64;
        let sigma0 = 69e6_f64;
        let half_w = 0.1_f64;
        let half_h = 0.1_f64;
        let fd_h = 1e-3_f32;
        let mat = material(e, nu);

        let hx_phys = fd_h as f64 * half_w;
        let hy_phys = fd_h as f64 * half_h;
        // Arbitrary non-origin interior point - catches any point-dependent bug an
        // origin-only check (where linear terms vanish) would miss.
        let (x0, y0) = (0.03_f64, 0.02_f64);
        let u = |x: f64| sigma0 / e * x;
        let v = |y: f64| -nu * sigma0 / e * y;
        let (sxx_exact, syy_exact, sxy_exact) = (sigma0, 0.0_f64, 0.0_f64);

        // [5,5] tensor matching `compute_domain_forwards`'s own row layout (center, x+hx,
        // x-hx, y+hy, y-hy) - already-physical-unit values, exactly as `raw` is by the time
        // it reaches `compute_strains` in production. Direct-σ columns are the exact constant
        // stress (the ideal case: ideally the network's direct output IS the true field).
        let mut data = Vec::with_capacity(25);
        for &(dx, dy) in &[(0.0, 0.0), (hx_phys, 0.0), (-hx_phys, 0.0), (0.0, hy_phys), (0.0, -hy_phys)] {
            let (x, y) = (x0 + dx, y0 + dy);
            data.extend_from_slice(&[u(x) as f32, v(y) as f32, sxx_exact as f32, syy_exact as f32, sxy_exact as f32]);
        }
        let raw = Tensor::<TB, 2>::from_data(TensorData::new(data, vec![5, 5]), &device);

        let fd = FdConfig::new(fd_h, 2.0 * half_w, 2.0 * half_h);
        let (eps_xx, eps_yy, eps_xy) = compute_strains::<TB>(raw.clone(), 1, &fd);

        let get = |t: Tensor<TB, 1>| -> f64 { t.into_data().to_vec::<f32>().unwrap()[0] as f64 };
        let (exx, eyy, exy) = (get(eps_xx.clone()), get(eps_yy.clone()), get(eps_xy.clone()));
        let (exx_expected, eyy_expected) = (sigma0 / e, -nu * sigma0 / e);
        assert!((exx - exx_expected).abs() / exx_expected.abs() < 1e-3, "eps_xx {exx} vs expected {exx_expected}");
        assert!((eyy - eyy_expected).abs() / eyy_expected.abs() < 1e-3, "eps_yy {eyy} vs expected {eyy_expected}");
        assert!(exy.abs() < 1e-9, "eps_xy {exy} should be ~0");

        let (sxx, syy, sxy) = compute_stress::<TB>(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), &mat);
        let (sxx_v, syy_v, sxy_v) = (get(sxx), get(syy), get(sxy));
        assert!((sxx_v - sigma0).abs() / sigma0 < 1e-3, "sigma_xx {sxx_v} vs {sigma0}");
        assert!(syy_v.abs() / sigma0 < 1e-3, "sigma_yy {syy_v} should be ~0");
        assert!(sxy_v.abs() / sigma0 < 1e-3, "sigma_xy {sxy_v} should be ~0");

        // Interior energy: finite, matches 0.5*sigma:epsilon by hand (real, expected nonzero).
        let energy_v = get(dem_energy_loss::<TB>(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), &mat));
        let energy_expected = 0.5 * (sxx_exact * exx_expected + syy_exact * eyy_expected);
        assert!((energy_v - energy_expected).abs() / energy_expected.abs() < 1e-2, "energy {energy_v} vs {energy_expected}");

        // Outer traction at a right-edge-like point (normal=(1,0)): predicted traction must
        // exactly match the target (sigma0, 0) - this is `OuterTractionTerm`'s exact math.
        let t1v = |v: f32| Tensor::<TB, 1>::from_data(TensorData::new(vec![v], vec![1]), &device);
        let outer_v = get(neumann_loss::<TB>(
            eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), t1v(1.0), t1v(0.0), t1v(sigma0 as f32), t1v(0.0), &mat,
        ));
        assert!(outer_v.abs() / (sigma0 * sigma0) < 1e-4, "outer_traction {outer_v} should be ~0 (normalized by sigma0^2)");

        // Equilibrium: constant direct-sigma field everywhere -> exactly zero divergence.
        // Same (sxx_xp,sxy_xp,sxx_xm,sxy_xm,sxy_yp,syy_yp,sxy_ym,syy_ym) argument order/row
        // layout `EquilibriumTerm` uses in production (user_problem.rs).
        let col = |r: usize, c: usize| -> Tensor<TB, 1> { raw.clone().slice([r..r + 1, c..c + 1]).reshape([1]) };
        let cx = fd.sx / (2.0 * fd.hx as f64);
        let cy = fd.sy / (2.0 * fd.hy as f64);
        let ref_div2 = (sigma0 * cx).powi(2).max(1.0);
        let eq_v = get(equilibrium_residual_loss::<TB>(
            col(1, 2), col(1, 4), col(2, 2), col(2, 4), col(3, 4), col(3, 3), col(4, 4), col(4, 3), cx, cy, ref_div2,
        ));
        assert!(eq_v < 1e-6, "equilibrium residual {eq_v} should be ~machine-zero for a spatially constant stress field");

        // Constitutive consistency: direct sigma exactly equals Hooke(epsilon) by construction.
        let cc_v = get(constitutive_consistency_loss::<TB>(
            col(0, 2), col(0, 3), col(0, 4), eps_xx, eps_yy, eps_xy, &mat,
        ));
        assert!(cc_v.abs() / (sigma0 * sigma0) < 1e-6, "constitutive_consistency {cc_v} should be ~0");
    }

    /// Zero-cost analytical Hessian check (no training, no network) - gate required by the
    /// #12 plan before `compute_hessian` is trusted in any real loss term. The uniform-tension
    /// field above has zero curvature everywhere and can't distinguish "correct second-
    /// derivative code" from "always returns zero"; this uses a manufactured field with
    /// nonzero, distinct curvature in every one of the 6 Hessian components instead:
    ///   u(x,y) = a x² + c xy   ->  u_xx=2a, u_yy=0,  u_xy=c
    ///   v(x,y) = b y² + d xy   ->  v_xx=0,  v_yy=2b, v_xy=d
    /// Central FD is exact (no truncation error) for polynomials up to degree 2 in each
    /// stencil direction, so this is checked to near machine precision, not just "close".
    #[test]
    fn hessian_recovers_exact_second_derivatives_of_a_manufactured_quadratic_field() {
        // Priority 7 (General-PINN §23, manufactured-solution framework): this test previously
        // hand-built the quadratic field, its exact derivatives, and the 9-point stencil-offset
        // tensor inline - all of that is now `crate::manufactured`'s job (a reusable module,
        // not a one-off), used here as its first real consumer via a straight refactor (same
        // field, same tolerances, same assertions - zero behavior change).
        use crate::manufactured::{verify_hessian, ManufacturedField};

        let device = Default::default();
        let half_w = 0.1_f64;
        let half_h = 0.1_f64;
        // See `verify_hessian`'s own doc comment for why this goes through `hessian_fd_config`
        // rather than production's raw `fd_h` directly (f32 second-order FD cancellation error).
        let base_fd_h = 1e-3_f32;
        let (x0, y0) = (0.03_f64, 0.02_f64);
        let (a, b, c, d) = (5.0_f64, -3.0_f64, 2.0_f64, -1.5_f64);
        let field = ManufacturedField::quadratic(a, b, c, d);

        let (u_xx_v, u_yy_v, u_xy_v, v_xx_v, v_yy_v, v_xy_v) =
            verify_hessian::<TB>(&field, half_w, half_h, base_fd_h, x0, y0, &device);

        let tol = 1e-3; // relative
        assert!((u_xx_v - 2.0 * a).abs() / (2.0 * a).abs() < tol, "u_xx {u_xx_v} vs {}", 2.0 * a);
        assert!(u_yy_v.abs() < 1e-6, "u_yy {u_yy_v} should be ~0");
        assert!((u_xy_v - c).abs() / c.abs() < tol, "u_xy {u_xy_v} vs {c}");
        assert!(v_xx_v.abs() < 1e-6, "v_xx {v_xx_v} should be ~0");
        assert!((v_yy_v - 2.0 * b).abs() / (2.0 * b).abs() < tol, "v_yy {v_yy_v} vs {}", 2.0 * b);
        assert!((v_xy_v - d).abs() / d.abs() < tol, "v_xy {v_xy_v} vs {d}");
    }

    #[test]
    fn equilibrium_from_displacement_hessian_loss_matches_hand_derived_residual() {
        // Pure arithmetic transcription check (mirrors compute_stress_matches_plane_stress_
        // hookes_law below): arbitrary nonzero Hessian values, hand-computed r_x/r_y, verify
        // the function's output matches - independent of whether the residual is physically
        // zero for any real field (that's `hessian_recovers_exact_second_derivatives_...`'s
        // and later the full-pipeline diagnostic's job).
        let mat = material(100.0, 0.25); // factor = 106.666.., G = 100/2.5 = 40
        let (u_xx, u_yy, u_xy) = (0.01_f64, 0.02_f64, 0.005_f64);
        let (v_xx, v_yy, v_xy) = (-0.01_f64, 0.03_f64, -0.002_f64);
        let factor = 100.0 / (1.0 - 0.25 * 0.25);
        let g = 100.0 / (2.0 * 1.25);
        let r_x_expected = factor * (u_xx + 0.25 * v_xy) + g * (u_yy + v_xy);
        let r_y_expected = factor * (v_yy + 0.25 * u_xy) + g * (u_xy + v_xx);
        let expected = r_x_expected * r_x_expected + r_y_expected * r_y_expected;

        let loss = equilibrium_from_displacement_hessian_loss(
            t1(u_xx as f32), t1(u_yy as f32), t1(u_xy as f32),
            t1(v_xx as f32), t1(v_yy as f32), t1(v_xy as f32),
            &mat, 1.0,
        );
        let loss_v = loss.into_data().to_vec::<f32>().unwrap()[0] as f64;
        assert!((loss_v - expected).abs() / expected < 1e-4, "expected {expected}, got {loss_v}");
    }

    #[test]
    fn compute_stress_matches_plane_stress_hookes_law() {
        // E=100, nu=0.25 -> factor = E/(1-nu^2) = 100/0.9375 = 106.666...
        let mat = material(100.0, 0.25);
        let (sxx, syy, sxy) = compute_stress(t1(0.01), t1(0.02), t1(0.005), &mat);

        let sxx_v = sxx.into_data().to_vec::<f32>().unwrap()[0];
        let syy_v = syy.into_data().to_vec::<f32>().unwrap()[0];
        let sxy_v = sxy.into_data().to_vec::<f32>().unwrap()[0];

        assert!((sxx_v - 1.6).abs() < 1e-4, "sxx: expected 1.6, got {sxx_v}");
        assert!((syy_v - 2.4).abs() < 1e-4, "syy: expected 2.4, got {syy_v}");
        assert!((sxy_v - 0.4).abs() < 1e-4, "sxy: expected 0.4, got {sxy_v}");
    }

    #[test]
    fn representation_consistency_check_matches_hand_computed_value() {
        // direct-derived = (3,4,0) -> ||.||^2 = 25; ref_stress2=5 -> 25/5=5.0
        let v = representation_consistency_check((13.0, 4.0, 10.0), (10.0, 0.0, 10.0), 5.0);
        assert!((v - 5.0).abs() < 1e-6, "expected 5.0, got {v}");
    }

    #[test]
    fn representation_consistency_check_zero_when_representations_agree() {
        let v = representation_consistency_check((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 1.0);
        assert_eq!(v, 0.0);
    }

    /// Proves the "computes the SAME quantity as `constitutive_consistency_loss`" claim in
    /// `representation_consistency_check`'s doc comment, rather than just asserting it - runs
    /// both on the same single-point (N=1) inputs (a plain scalar strain state, so
    /// `compute_stress`'s derived output is unambiguous) and confirms they agree to float
    /// precision.
    #[test]
    fn representation_consistency_check_matches_constitutive_consistency_loss_for_a_single_point() {
        let get = |t: Tensor<TB, 1>| -> f64 { t.into_data().to_vec::<f32>().unwrap()[0] as f64 };
        let mat = material(71.7e9, 0.33);
        let (eps_xx, eps_yy, eps_xy) = (1e-3_f64, -2e-4_f64, 5e-4_f64);
        let (direct_sxx, direct_syy, direct_sxy) = (6.9e7_f32, 1.0e6_f32, -2.0e6_f32);

        let tensor_loss = get(constitutive_consistency_loss::<TB>(
            t1(direct_sxx), t1(direct_syy), t1(direct_sxy),
            t1(eps_xx as f32), t1(eps_yy as f32), t1(eps_xy as f32),
            &mat,
        ));

        let (derived_sxx, derived_syy, derived_sxy) = {
            let (sxx, syy, sxy) = compute_stress::<TB>(t1(eps_xx as f32), t1(eps_yy as f32), t1(eps_xy as f32), &mat);
            (get(sxx) as f32, get(syy) as f32, get(sxy) as f32)
        };
        let check = representation_consistency_check(
            (direct_sxx, direct_syy, direct_sxy),
            (derived_sxx, derived_syy, derived_sxy),
            1.0,
        );
        assert!((check as f64 - tensor_loss).abs() / tensor_loss.max(1.0) < 1e-4,
            "representation_consistency_check={check} vs constitutive_consistency_loss={tensor_loss}");
    }

    #[test]
    fn dem_energy_loss_matches_half_sigma_dot_epsilon() {
        let mat = material(100.0, 0.25);
        let energy = dem_energy_loss(t1(0.01), t1(0.02), t1(0.005), &mat);
        let energy_v = energy.into_data().to_vec::<f32>().unwrap()[0];
        // 0.5 * (sxx*exx + syy*eyy + 2*sxy*exy) = 0.5*(1.6*0.01 + 2.4*0.02 + 2*0.4*0.005) = 0.034
        assert!((energy_v - 0.034).abs() < 1e-4, "expected 0.034, got {energy_v}");
    }

    #[test]
    fn dem_energy_per_point_matches_dem_energy_loss_for_single_point() {
        let mat = material(71.7e9, 0.33);
        let per_point = dem_energy_per_point(t1(1e-4), t1(-5e-5), t1(2e-5), &mat);
        let loss = dem_energy_loss(t1(1e-4), t1(-5e-5), t1(2e-5), &mat);
        let pp_v = per_point.into_data().to_vec::<f32>().unwrap()[0];
        let loss_v = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!((pp_v - loss_v).abs() < 1e-6, "per-point and mean should agree for N=1: {pp_v} vs {loss_v}");
    }
}
