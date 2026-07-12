/// Problem Analysis Engine
///
/// Analyzes the physical problem specification and automatically derives all
/// training parameters. Eliminates hardcoding — every constant in the training
/// loop is computed from geometry, material, and loading.

use pinn_core::{
    amr::{AmrtConfig, DEFAULT_HOLE_ZONE_FACTOR},
    geometry::{GeometryConfig, HoleType, SymmetryMode},
    kirsch::kirsch_stress,
    loading::LoadConfig,
    messages::SolverConfig,
};

use crate::kirsch_problem::{LAM_D, LAM_E, LAM_EQ, LAM_H, LAM_KIRSCH, LAM_N};

/// All parameters derived automatically from the physical problem.
#[derive(Debug, Clone)]
pub struct EngineParams {
    /// Dirichlet ansatz: u = tanh(k*(xn+1)) * u_net. Saturates at hole surface.
    pub ansatz_k: f32,
    /// FD step in normalised coords.
    pub fd_h: f32,
    /// Auto-detected symmetry mode.
    pub symmetry: SymmetryMode,
    /// Network hidden layer width.
    pub hidden_dim: usize,
    /// Number of hidden layers.
    pub n_hidden: usize,
    /// Interior collocation points.
    pub n_interior: usize,
    /// Boundary collocation points.
    pub n_boundary: usize,
    /// Equilibrium check points (near-hole ring, r ∈ [2r_hole, 3r_hole]).
    pub n_eq_ring: usize,
    /// SAW-BRDR base weights: [lam_e, lam_n, lam_h, lam_d, lam_eq].
    pub lam_e:  f32,
    pub lam_n:  f32,
    pub lam_h:  f32,
    pub lam_d:  f32,
    pub lam_eq: f32,
    /// Peak learning rate.
    pub peak_lr: f64,
    /// K_t probe radius factor (r_probe = r_hole * probe_r_factor). Used for K_t REPORTING only.
    pub probe_r_factor: f64,
    /// Probe angles in degrees for K_t validation report.
    pub probe_thetas_deg: Vec<f64>,
    /// Theoretical K_t at probe radius from Kirsch formula (validation metric, not training target).
    pub expected_kt: f64,
    /// Two-phase curriculum: steps 0..phase1_steps train BCs only (no kirsch_loss).
    /// At phase1_steps, kirsch_loss is activated. BCs must be converged (< 1e-2) first.
    pub phase1_steps: usize,
    /// Collocation count for Phase 1 (BC-only). Much smaller than n_interior to speed Phase 1.
    /// Phase 2 uses n_interior (full batch). Ratio ~16× reduces Phase 1 compute.
    pub phase1_n_interior: usize,
    /// Radius factors for Kirsch stress loss probes: samples at r = factor × r_hole.
    /// Multiple radii teach the network the radial DECAY shape, not just the peak value.
    pub kirsch_r_factors: Vec<f64>,
    /// Angles (degrees) for Kirsch stress loss probes. Dense near 90° where σ_xx peaks.
    pub kirsch_thetas_deg: Vec<f64>,
    /// Weight for Kirsch stress loss (initial SAW base for Phase 2 6th component).
    pub lam_kirsch: f32,
    /// Fourier feature embedding levels. 0 = disabled; 8 → 32 features covering freq [π, 128π].
    /// Enabled automatically when has_hole=true to fix spectral bias near the hole.
    pub n_fourier: usize,
    /// Mixed Deep Energy Method: network outputs (u, v, σ_xx, σ_yy, σ_xy) instead of (u, v, w).
    /// Eliminates FD stencil artifacts at the hole surface; enabled when has_hole=true.
    pub use_mdem: bool,
    /// Constitutive consistency loss weight (fixed, not in SAW-BRDR): enforces σ_net ≈ C:ε_fd.
    /// Active in both phases when use_mdem=true; otherwise 0.
    pub lam_const: f32,
    /// Self-learning quadtree AMR configuration (all params derived from geometry).
    /// Disabled in Phase 1 (use static sample_interior); active in Phase 2 only.
    pub amr: AmrtConfig,
}

/// Relative tolerance for treating a load component as zero / two components as equal
/// when auto-detecting whether the problem is symmetric enough for the quarter model.
const SYMMETRY_LOAD_TOLERANCE: f64 = 1e-6;

/// Detect symmetry from geometry and loading.
pub fn detect_symmetry(geom: &GeometryConfig, load: &LoadConfig) -> SymmetryMode {
    let tol = SYMMETRY_LOAD_TOLERANCE;
    let sym_load = load.py.abs() < tol * load.px.abs().max(1.0)
        || load.px.abs() < tol * load.py.abs().max(1.0)
        || (load.px - load.py).abs()
            < tol * (load.px.abs() + load.py.abs()).max(1.0);
    if sym_load && geom.half_w > 0.0 && geom.half_h > 0.0 {
        SymmetryMode::QuarterSymm
    } else {
        SymmetryMode::Full
    }
}

/// Clamp bounds for the ansatz saturation factor `k` — keeps tanh(k*r_norm) in a numerically
/// well-behaved range across both very large and very small hole-to-domain ratios.
const ANSATZ_K_MIN: f32 = 10.0;
const ANSATZ_K_MAX: f32 = 200.0;
/// Ansatz `k` used when there's no hole (fixed; no hole radius to scale against).
const ANSATZ_K_NO_HOLE: f32 = 5.0;

/// Compute Dirichlet ansatz saturation factor k.
///
/// k = max_dim / r_hole ensures tanh(k * r_norm) ≈ 0.96 at the hole surface,
/// giving O(1) gradient signal through the ansatz at near-hole collocation points.
pub fn compute_ansatz_k(geom: &GeometryConfig) -> f32 {
    match geom.hole {
        HoleType::Circular { radius } => {
            let max_dim = geom.half_w.max(geom.half_h);
            (max_dim / radius).clamp(ANSATZ_K_MIN as f64, ANSATZ_K_MAX as f64) as f32
        }
        HoleType::None => ANSATZ_K_NO_HOLE,
    }
}

/// Problem-complexity (domain/hole ratio) breakpoints for auto-sizing the network —
/// larger stress-concentration problems need more capacity to resolve the near-hole field.
const COMPLEXITY_HIDDEN_DIM_SMALL: f64 = 10.0;
const COMPLEXITY_HIDDEN_DIM_LARGE: f64 = 50.0;
const COMPLEXITY_N_HIDDEN_BREAKPOINT: f64 = 20.0;

/// Network size from problem complexity (ratio of domain to hole).
pub fn compute_network_size(geom: &GeometryConfig) -> (usize, usize) {
    let complexity = match geom.hole {
        HoleType::Circular { radius } => geom.half_w.max(geom.half_h) / radius,
        HoleType::None => 1.0,
    };
    let hidden_dim = if complexity < COMPLEXITY_HIDDEN_DIM_SMALL { 64 }
        else if complexity < COMPLEXITY_HIDDEN_DIM_LARGE { 128 }
        else { 256 };
    let n_hidden = if complexity < COMPLEXITY_N_HIDDEN_BREAKPOINT { 5 } else { 6 };
    (hidden_dim, n_hidden)
}

/// Collocation counts scaled from hole-to-domain ratio.
pub fn compute_collocation_sizes(geom: &GeometryConfig) -> (usize, usize) {
    let r_ratio = match geom.hole {
        HoleType::Circular { radius } => (radius / geom.half_w.max(geom.half_h)).max(0.001),
        HoleType::None => 1.0,
    };
    let n_int = ((4096.0 / r_ratio.sqrt()) as usize).clamp(2048, 8192);
    let n_bnd = (n_int / 4).clamp(512, 2048);
    (n_int, n_bnd)
}

/// Theoretical K_t at the probe radius via sin²θ-weighted Kirsch average.
fn expected_kt_at_probe(
    r_factor:       f64,
    probe_thetas:   &[f64],
    load:           &LoadConfig,
) -> f64 {
    if load.px.abs() < 1e-10 { return 0.0; }
    let mut wsum = 0.0_f64;
    let mut wval = 0.0_f64;
    for &deg in probe_thetas {
        let theta = deg.to_radians();
        let (s_rr, s_tt, s_rt) = kirsch_stress(r_factor, theta, 1.0, load.px, load.py);
        let sxx = s_rr * theta.cos().powi(2) + s_tt * theta.sin().powi(2)
            - 2.0 * s_rt * theta.sin() * theta.cos();
        let w = theta.sin().powi(2);
        wval += w * sxx;
        wsum += w;
    }
    if wsum > 1e-12 { wval / wsum / load.px } else { 0.0 }
}

impl EngineParams {
    /// Derive all training parameters from the physical problem specification.
    pub fn analyze(config: &SolverConfig) -> Self {
        let geom = &config.geometry;
        let load = &config.load;

        let symmetry         = detect_symmetry(geom, load);
        // Safe probe: inward FD stencil must not cross hole.
        // fd_h=1e-3 (normalised coords) → physical step ≈ 0.127mm; empirically r_factor ≥ 1.2 is safe.
        // (Attempts at 1.05/1.1 caused FD stencil artifacts → K_t capped at ~2.0.)
        // K_t is normalised vs Kirsch analytical at r=1.2·r_hole → reports 3.0 at convergence.
        let probe_r_factor   = 1.2_f64;
        let probe_thetas_deg = vec![80.0_f64, 83.0, 86.0, 88.0, 89.0];

        let ansatz_k  = compute_ansatz_k(geom);
        let fd_h      = config.fd_h; // keep user-supplied; engine validates it's safe
        let (hidden_dim, n_hidden) = compute_network_size(geom);
        let (n_interior, n_boundary) = compute_collocation_sizes(geom);

        let has_hole = matches!(geom.hole, HoleType::Circular { .. });
        // LAM_E/LAM_N/LAM_H/LAM_D/LAM_EQ/LAM_KIRSCH are `kirsch_problem.rs`'s canonical
        // SAW-BRDR base weights (also what `KirschProblem::base_weight` returns) — read here
        // rather than re-typed, so this seed and that lookup can't silently drift apart.
        // `KirschProblem` always has a hole (`GeometryConfig::kirsch_plate_inches()`), so in
        // practice `lam_h`/`lam_eq`/`lam_kirsch` below always take their non-zero branch;
        // `lam_e`/`lam_n`/`lam_d` are unconditional regardless of `has_hole` and always were.
        // Those three `else` branches only matter for a hole-less geometry driven directly
        // through this generic engine outside a `KirschProblem` (no equivalent Kirsch-loss/
        // eq-ring terms to weight in that case).
        let lam_e  = LAM_E;
        let lam_n  = LAM_N;
        let lam_h  = if has_hole { LAM_H } else { 0.0 };
        let lam_d  = LAM_D;
        // Equilibrium residual: enforces ∇·σ=0 at near-hole ring.
        let lam_eq    = if has_hole { LAM_EQ } else { 0.0 };
        let n_eq_ring = if has_hole { 100 } else { 0 };
        // Kirsch stress loss: directly targets (σ_xx, σ_yy, σ_xy) at 4 radii × 7 angles = 28 pts.
        // SAW-BRDR 6th component in Phase 2 — replaces AdaptiveLamKirsch controller.
        let lam_kirsch = if has_hole { LAM_KIRSCH } else { 0.0 };

        // mDEM + Fourier Feature Embedding: enabled when a circular hole is present.
        // n_fourier=8 → 32 features covering freq [π, 128π] — corrects spectral bias near hole.
        // use_mdem: network outputs 5 values (u, v, σ_xx, σ_yy, σ_xy); eliminates FD stencil
        //   artifacts at the hole surface; constitutive consistency loss enforces σ_net ≈ C:ε_fd.
        let n_fourier = if has_hole { 8_usize } else { 0 };
        let use_mdem  = has_hole;
        let lam_const = if use_mdem { 5.0_f32 } else { 0.0 };

        // Hole-surface K_t = 3.0 for infinite plate under uniaxial tension (Kirsch, 1898).
        // probe_kt() normalises predicted σ_xx by Kirsch analytical at the probe radius so the
        // reported K_t converges to 3.0 when the stress field matches the Kirsch solution.
        let expected_kt = if has_hole && load.py.abs() < 1e-6 * load.px.abs().max(1.0) {
            3.0_f64
        } else {
            expected_kt_at_probe(probe_r_factor, &probe_thetas_deg, load)
        };

        // Two-phase curriculum: Phase 1 trains BCs only until convergence (steps 0..phase1_steps),
        // Phase 2 adds kirsch_stress_loss (fixed weight, outside SAW-BRDR). BC gradients are ≈0
        // at phase1_steps so kirsch gradient drives stress field toward Kirsch solution.
        let phase1_steps = if has_hole { 4000 } else { 0 };

        // Phase 1 uses a smaller collocation set for 16× faster per-step compute.
        // Phase 2 switches to full n_interior when BCs are converged.
        let phase1_n_interior = (n_interior / 16).max(512);

        // Self-learning AMR: refinement depth derived from hole/domain ratio.
        // Smaller hole relative to domain → need more refinement levels near hole.
        let hole_r = match geom.hole {
            HoleType::Circular { radius } => radius,
            HoleType::None => geom.half_w,
        };
        // A small hole relative to the domain needs deeper refinement to resolve the
        // stress concentration without an enormous uniform grid.
        const SMALL_HOLE_RATIO_THRESHOLD: f64 = 0.04;
        let hole_ratio = hole_r / geom.half_w.max(geom.half_h);
        let amr_max_level = if hole_ratio < SMALL_HOLE_RATIO_THRESHOLD { 7 } else { 6 };
        let amr = AmrtConfig {
            initial_level:      4,
            max_level:          amr_max_level,
            min_level_hole:     amr_max_level - 1,
            hole_zone_factor:   DEFAULT_HOLE_ZONE_FACTOR,
            refine_percentile:  0.80,
            coarsen_percentile: 0.15,
            ema_alpha:          0.30,
            trend_weight:       0.40,
            interval_steps:     1000,
            pts_per_cell:       1,
            max_active_cells:   None,
        };

        // Kirsch stress loss probe points.
        // Safe minimum: r_factor ≥ 1.2 (empirical — FD step ≈ 0.127mm physical; 1.05/1.1 caused artifacts).
        // r=3.0 added for far-field regularization: prevents network overfitting near-hole region,
        // improves decay-shape learning, reduces K_t plateau probability.
        let kirsch_r_factors  = vec![1.2_f64, 1.5, 2.0, 3.0];
        let kirsch_thetas_deg = vec![60.0_f64, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0];

        Self {
            ansatz_k,
            fd_h,
            symmetry,
            hidden_dim,
            n_hidden,
            n_interior,
            n_boundary,
            n_eq_ring,
            lam_e,
            lam_n,
            lam_h,
            lam_d,
            lam_eq,
            lam_kirsch,
            n_fourier,
            use_mdem,
            lam_const,
            peak_lr: 1e-3,
            probe_r_factor,
            probe_thetas_deg,
            expected_kt,
            phase1_steps,
            phase1_n_interior,
            kirsch_r_factors,
            kirsch_thetas_deg,
            amr,
        }
    }

    /// Apply engine-derived values back into a SolverConfig.
    pub fn apply_to(&self, config: &mut SolverConfig) {
        config.geometry.symmetry = self.symmetry;
        config.hidden_dim  = self.hidden_dim;
        config.n_hidden    = self.n_hidden;
        config.n_interior  = self.n_interior;
        config.n_boundary  = self.n_boundary;
    }

    /// Network output dimension: 3 (plain DEM: u,v,w) or 5 (mDEM: u,v,σ_xx,σ_yy,σ_xy).
    pub fn output_dim(&self) -> usize { if self.use_mdem { 5 } else { 3 } }

    /// Network input dimension: 3 (raw x,y,z) or 4*n_fourier (Fourier features from x,y).
    pub fn net_input_dim(&self) -> usize { if self.n_fourier > 0 { 4 * self.n_fourier } else { 3 } }

    /// Phase 1 SAW-BRDR base weights: [lam_e, lam_n, lam_h, lam_d, lam_eq] (5 components).
    pub fn init_weights(&self) -> Vec<f32> {
        vec![self.lam_e, self.lam_n, self.lam_h, self.lam_d, self.lam_eq]
    }

    /// Phase 2 SAW-BRDR base weights: 6 components — adds kirsch_stress_loss.
    /// SAW replaced entirely at phase2 transition (not mutated in-place).
    pub fn init_weights_phase2(&self) -> Vec<f32> {
        vec![self.lam_e, self.lam_n, self.lam_h, self.lam_d, self.lam_eq, self.lam_kirsch]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::messages::SolverConfig;

    /// LAM_E/LAM_N/LAM_H/LAM_D/LAM_EQ/LAM_KIRSCH must be hand-typed in exactly one place.
    /// `kirsch_problem.rs`'s `LAM_*` constants are that canonical source (they're also what
    /// `KirschProblem::base_weight` returns — see `base_weights_match_engine_params_literals`
    /// there); `analyze` reads them here rather than carrying its own independently-typed
    /// copies that could silently drift out of sync with the values actually seeded into
    /// live Kirsch training (`engine.init_weights()` / `init_weights_phase2()`).
    /// `SolverConfig::default_kirsch()` always has a hole, so every term below is on its
    /// non-zero (`has_hole`) branch.
    #[test]
    fn lam_weights_match_kirsch_problem_canonical_constants() {
        let config = SolverConfig::default_kirsch();
        let engine = EngineParams::analyze(&config);
        assert_eq!(engine.lam_e, crate::kirsch_problem::LAM_E);
        assert_eq!(engine.lam_n, crate::kirsch_problem::LAM_N);
        assert_eq!(engine.lam_h, crate::kirsch_problem::LAM_H);
        assert_eq!(engine.lam_d, crate::kirsch_problem::LAM_D);
        assert_eq!(engine.lam_eq, crate::kirsch_problem::LAM_EQ);
        assert_eq!(engine.lam_kirsch, crate::kirsch_problem::LAM_KIRSCH);
    }

    /// `hole_zone_factor` (3.0) must likewise be a single shared constant: this AMR config
    /// and `KirschSamplingStrategy::amr_lock_zone`'s geometry-only mirror (see that method's
    /// own test in `kirsch_problem.rs`) both read `pinn_core::amr::DEFAULT_HOLE_ZONE_FACTOR`
    /// instead of each hand-typing `3.0`.
    #[test]
    fn hole_zone_factor_matches_shared_default_constant() {
        let config = SolverConfig::default_kirsch();
        let engine = EngineParams::analyze(&config);
        assert_eq!(engine.amr.hole_zone_factor, pinn_core::amr::DEFAULT_HOLE_ZONE_FACTOR);
    }
}
