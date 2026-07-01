/// Kirsch analytical solution for a circular hole in an infinite plate
/// under biaxial far-field stress (σ_∞_x = px, σ_∞_y = py).
///
/// Returns (σ_rr, σ_θθ, σ_rθ) at polar coordinates (r, θ) from the hole centre.
/// Reference: Kirsch (1898), validated by Timoshenko & Goodier §32.
pub fn kirsch_stress(
    r: f64,
    theta: f64,
    hole_radius: f64,
    px: f64,
    py: f64,
) -> (f64, f64, f64) {
    let a = hole_radius;
    let a2 = a * a;
    let r2 = r * r;
    let a2r2 = a2 / r2;
    let a4r4 = a2r2 * a2r2;

    let cos2t = (2.0 * theta).cos();
    let sin2t = (2.0 * theta).sin();

    // Average and deviatoric
    let p_avg = 0.5 * (px + py);
    let p_dev = 0.5 * (px - py);

    let s_rr = p_avg * (1.0 - a2r2)
        + p_dev * (1.0 - 4.0 * a2r2 + 3.0 * a4r4) * cos2t;

    let s_tt = p_avg * (1.0 + a2r2)
        - p_dev * (1.0 + 3.0 * a4r4) * cos2t;

    let s_rt = -p_dev * (1.0 + 2.0 * a2r2 - 3.0 * a4r4) * sin2t;

    (s_rr, s_tt, s_rt)
}

/// Peak hoop stress at the hole edge (r = a).
/// For uniaxial px: peak at θ=90° → σ_θθ = 3·px (K_t = 3.0)
/// For biaxial: σ_θθ_max depends on px/py ratio.
pub fn peak_hoop_stress(_hole_radius: f64, px: f64, py: f64) -> f64 {
    // At r = a: σ_θθ = (px + py) - 2*(px - py)*cos(2θ)
    // Maximum over θ:
    let p_avg = px + py;
    let p_dev = 2.0 * (px - py).abs();
    (p_avg + p_dev).max(p_avg - p_dev)
}

/// Theoretical stress concentration factor for uniaxial tension
pub fn theoretical_kt_uniaxial() -> f64 {
    3.0
}

/// Estimate K_t from PINN-computed stress field.
/// Scans the hole boundary to find peak σ_θθ / σ_remote.
pub fn stress_concentration_factor(peak_sigma_yy: f32, sigma_remote: f32) -> f32 {
    if sigma_remote.abs() < 1e-12 {
        return 0.0;
    }
    peak_sigma_yy / sigma_remote
}

/// Kirsch displacement field (plane stress, uniaxial px):
/// Returns (u_r, u_θ) displacement in polar coords relative to hole center.
/// Useful for setting far-field displacement BCs.
pub fn kirsch_displacement(
    r: f64,
    theta: f64,
    hole_radius: f64,
    px: f64,
    py: f64,
    e: f64,
    nu: f64,
) -> (f64, f64) {
    let a = hole_radius;
    let a2r = a * a / r;
    let kappa = (3.0 - nu) / (1.0 + nu); // plane stress

    let p_avg = 0.5 * (px + py);
    let p_dev = 0.5 * (px - py);

    let mu = e / (2.0 * (1.0 + nu));

    let ur = p_avg / (2.0 * mu) * r
        + p_dev / (2.0 * mu) * (
            (kappa + 1.0) * r / 2.0 - 2.0 * a2r + a2r * (a * a / r / r) / 2.0
        ) * (2.0 * theta).cos();
    // Only radial component simplified here; tangential for completeness
    let ut = -p_dev / (2.0 * mu) * (
        (kappa - 1.0) * r / 2.0 + a2r + a2r * (a * a / r / r) / 2.0
    ) * (2.0 * theta).sin();

    (ur, ut)
}
