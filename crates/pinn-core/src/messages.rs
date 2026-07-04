use ndarray::Array2;
use crate::geometry::GeometryConfig;
use crate::loading::LoadConfig;
use crate::material::MaterialProps;

/// Command sent from GUI thread → solver thread
pub enum ControlMsg {
    Stop,
    Pause,
    Resume,
    /// Trigger a warm-start with new configuration
    WarmStart { config: SolverConfig, geometry_changed: bool },
}

/// Data sent from solver thread → GUI thread (bounded channel capacity=1)
pub enum TrainingMsg {
    Update(Box<TrainingUpdate>),
    Done,
    Error(String),
}

pub struct TrainingUpdate {
    pub step: usize,
    pub total_loss:   f32,
    pub energy_loss:  f32,
    pub neumann_loss: f32,
    pub lr:           f32,
    pub lam_energy:   f32,
    pub lam_neumann:  f32,
    pub n_colloc:     usize,
    pub kt_estimate:  Option<f32>,
    pub vis: Option<VisFields>,
}

/// Visualization fields — sent every 10 steps (not every step, to keep channel fast)
pub struct VisFields {
    pub von_mises: Array2<f32>,
    pub sigma_xx:  Array2<f32>,
    pub sigma_yy:  Array2<f32>,
    pub sigma_xy:  Array2<f32>,
    pub disp_u:    Array2<f32>,
    pub disp_v:    Array2<f32>,
}

/// Configuration for the meta-optimizer decision maker (opt-in, disabled by default).
///
/// When `enabled = false`, the existing training loop runs unchanged (SOAP-Muon for
/// weights, AdamW for biases, for the entire run). Set `enabled = true` to activate
/// the three-tier optimizer state machine.
///
/// Architecture invariant: K_t is a post-hoc verification metric. It is **never** used
/// as a transition trigger here — all gates are pure gradient-signal metrics so the
/// decision maker works identically on problems with no closed-form analytical solution.
#[derive(Clone, Debug)]
pub struct DecisionMakerConfig {
    /// Enable the three-tier optimizer state machine (default: false — opt-in).
    pub enabled: bool,
    /// Steps between gradient conflict evaluations (default: 50).
    pub check_interval: usize,
    /// Cosine similarity below which Explore → Align (default: 0.60).
    pub conflict_threshold: f32,
    /// Cosine similarity above which Align → Explore (hysteresis, default: 0.25).
    pub alignment_threshold: f32,
    /// Minimum cosine similarity to enter Converge tier (default: 0.75).
    pub converge_cosine_min: f32,
    /// g_total_norm (g_pde_norm + g_bc_norm) threshold for Converge entry (default: 5.0).
    /// Dimensionless, relative to O(1)-normalized losses on a 25% collocation subset.
    pub converge_grad_threshold: f32,
    /// Use exact dual-pass cosine similarity; if false, use cheap proxy ratio (default: true).
    /// Converge tier (L-BFGS) is only entered when this is true.
    pub use_exact_cosine: bool,
    /// Minimum steps to spend in any tier before allowing a transition (default: 50).
    pub min_dwell_steps: usize,
    /// L-BFGS max inner iterations per outer step (default: 5).
    pub lbfgs_max_iter: usize,
}

impl Default for DecisionMakerConfig {
    fn default() -> Self {
        Self {
            enabled:                 false,
            check_interval:          50,
            conflict_threshold:      0.60,
            alignment_threshold:     0.25,
            converge_cosine_min:     0.75,
            converge_grad_threshold: 5.0,
            use_exact_cosine:        true,
            min_dwell_steps:         50,
            lbfgs_max_iter:          5,
        }
    }
}

/// Configuration for the stiffness-coupled SAW-BRDR / PirateNet-gate accelerator
/// (opt-in, disabled by default). When `enabled = false`, no extra gradient-conflict
/// computation is scheduled by this subsystem and `step_physics()` receives
/// `physics_boost = 1.0`, `alpha_lr_mult = 1.0` (both no-ops).
///
/// Architecture invariant: like [`DecisionMakerConfig`], this is driven purely by the
/// real-time gradient-conflict cosine-similarity metric — K_t is never read here.
#[derive(Clone, Debug)]
pub struct StiffnessConfig {
    /// Enable the stiffness controller (default: false — opt-in).
    pub enabled: bool,
    /// Steps between gradient-conflict evaluations (default: 50).
    pub check_interval: usize,
    /// EMA smoothing factor for the held stiffness value (default: 0.7).
    pub ema_beta: f32,
    /// Gain for the SAW-BRDR physics-loss boost; boost = `1 + gain * stiffness`,
    /// hard-clamped to `[1, 4]` regardless of this value (default: 1.0).
    pub physics_boost_gain: f32,
    /// Gain for the PirateNet gate-LR multiplier; mult = `1 + gain * stiffness`,
    /// hard-clamped to `[1, 5]` regardless of this value (default: 2.0).
    pub alpha_accel_gain: f32,
    /// Gate magnitude above which a PirateNet block is considered "awake" and its
    /// weights are included in the SOAP-Muon optimizer step (default: 1e-4).
    pub gate_awake_epsilon: f32,
}

impl Default for StiffnessConfig {
    fn default() -> Self {
        Self {
            enabled:            false,
            check_interval:     50,
            ema_beta:           0.7,
            physics_boost_gain: 1.0,
            alpha_accel_gain:   2.0,
            gate_awake_epsilon: 1e-4,
        }
    }
}

/// Complete solver configuration (passed when spawning the solver thread)
#[derive(Clone)]
pub struct SolverConfig {
    pub material: MaterialProps,
    pub geometry: GeometryConfig,
    pub load:     LoadConfig,
    pub n_interior: usize,
    pub n_boundary: usize,
    pub max_steps:  usize,
    pub vis_grid:   [usize; 2],   // [Nx, Ny]
    pub hidden_dim: usize,
    pub n_hidden:   usize,
    /// FD step size in normalized coordinates [−1,1]²
    pub fd_h: f32,
    /// If true (default), 2D weight matrices are trained with the SOAP-Muon hybrid
    /// optimizer. If false, falls back to plain AdamW for all parameters — kept as an
    /// escape hatch in case the hybrid proves unstable on a given configuration.
    pub use_soap_muon: bool,
    /// Meta-optimizer decision maker configuration (disabled by default).
    pub decision_maker: DecisionMakerConfig,
    /// Opt-in PirateNet adaptive-residual gating (disabled by default). See
    /// [`ElasticityNetConfig::use_piratenet`] in `pinn-solver`.
    pub use_piratenet: bool,
    /// Stiffness-coupled SAW-BRDR / gate-LR accelerator configuration (disabled by
    /// default).
    pub stiffness: StiffnessConfig,
}

impl SolverConfig {
    pub fn default_kirsch() -> Self {
        Self {
            material:   MaterialProps::al7075_t6(),
            geometry:   GeometryConfig::kirsch_plate_inches(),
            load:       LoadConfig::default_10ksi(),
            n_interior: 4096,
            n_boundary: 1024,
            max_steps:  28000,
            vis_grid:   [64, 64],
            hidden_dim: 128,
            n_hidden:   5,
            fd_h:       1e-3,
            use_soap_muon:  true,
            decision_maker: DecisionMakerConfig::default(),
            use_piratenet:  false,
            stiffness:      StiffnessConfig::default(),
        }
    }

    /// Pin-in-lug contact problem defaults. This single-domain `SolverConfig` shape can't
    /// carry two domains' geometry/material — it's populated here with the LUG domain's
    /// values (the driven/output-of-interest domain) purely so CLI/env plumbing that reads
    /// `config.geometry`/`config.material`/`config.load` for display (see
    /// `pinn-app/src/main.rs`, `headless.rs`'s startup banner) has *something* sensible to
    /// show; the actual two-domain geometry/material/load setup used for training lives in
    /// `pinn_solver::pinlug_problem::PinLugProblem::new`, which is the single source of
    /// truth for both domains.
    ///
    /// Force→traction conversion for the driving load (see `PinLugProblem::new`'s doc
    /// comment for the full derivation): `load.px` here is set to the SAME equivalent
    /// traction magnitude used for the pin's driving boundary condition, expressed as a
    /// far-field-style stress purely for display consistency with `default_kirsch()`.
    pub fn default_pinlug() -> Self {
        use crate::units::{IN_TO_M, LBF_TO_N};
        let pin_radius = 0.5 * IN_TO_M;
        let thickness = 0.4 * IN_TO_M;
        // P = 20,000 lbf total axial force / (projected diametral contact area = 2*r*t).
        // See PinLugProblem::new doc comment for why diametral projection is the right
        // denominator (Hertzian/pin-bearing convention: the resultant force is reacted by
        // the pressure distribution's projection onto the loading axis, whose max extent is
        // the pin diameter times thickness).
        let total_force_lbf = 20_000.0;
        let total_force_n = total_force_lbf * LBF_TO_N;
        let projected_area_m2 = 2.0 * pin_radius * thickness;
        let equivalent_traction_pa = total_force_n / projected_area_m2;
        Self {
            material:   MaterialProps::steel_4340(),
            geometry:   GeometryConfig::pinlug_lug_inches(),
            load:       LoadConfig::uniaxial_x(equivalent_traction_pa),
            n_interior: 2048,
            n_boundary: 512,
            max_steps:  20000,
            vis_grid:   [64, 64],
            hidden_dim: 128,
            n_hidden:   5,
            fd_h:       1e-3,
            use_soap_muon:  true,
            decision_maker: DecisionMakerConfig::default(),
            use_piratenet:  false,
            stiffness:      StiffnessConfig::default(),
        }
    }
}
