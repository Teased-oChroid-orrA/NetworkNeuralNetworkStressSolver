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
