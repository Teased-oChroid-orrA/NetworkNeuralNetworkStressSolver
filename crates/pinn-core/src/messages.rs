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
    /// Request that the pin-in-lug training loop export the current contact-pressure
    /// profile to CSV (see `pinn_solver::contact_export`). No-op (treated as `Continue`)
    /// on the single-domain Kirsch path — there is nothing to export there.
    ExportContactPressure,
}

/// Data sent from solver thread → GUI thread (bounded channel capacity=1)
pub enum TrainingMsg {
    Update(Box<TrainingUpdate>),
    /// Pin-in-lug analogue of `Update` — carries both domains' visualization fields and a
    /// generic convergence metric instead of Kirsch's K_t.
    PinLugUpdate(Box<PinLugTrainingUpdate>),
    Done,
    Error(String),
    /// Contact-pressure CSV export finished successfully; carries the written file path.
    ExportComplete(String),
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

/// Pin-in-lug analogue of `TrainingUpdate` — one entry per domain's visualization fields,
/// plus a generic (non-K_t) convergence metric.
pub struct PinLugTrainingUpdate {
    pub step: usize,
    pub total_loss:   f32,
    /// Documented approximation: sum of both domains' interior-energy scalars.
    pub energy_loss:  f32,
    /// Documented approximation: sum of all non-energy BC-term scalars.
    pub neumann_loss: f32,
    pub lr:           f32,
    pub lam_energy:   f32,
    pub lam_neumann:  f32,
    /// Pin + lug interior point counts, summed.
    pub n_colloc:     usize,
    /// Interface-gap RMS (see `PinLugProblem::convergence_metric`) — deliberately NOT named
    /// `kt_estimate`; pin-in-lug has no closed-form K_t.
    pub convergence_metric: Option<f32>,
    pub vis: Option<PinLugVisFields>,
}

/// Visualization fields — sent every 10 steps (not every step, to keep channel fast)
#[derive(Debug, Clone)]
pub struct VisFields {
    pub von_mises: Array2<f32>,
    pub sigma_xx:  Array2<f32>,
    pub sigma_yy:  Array2<f32>,
    pub sigma_xy:  Array2<f32>,
    pub disp_u:    Array2<f32>,
    pub disp_v:    Array2<f32>,
}

/// Per-domain visualization fields for the pin-in-lug 2-domain problem.
pub struct PinLugVisFields {
    pub pin: VisFields,
    pub lug: VisFields,
}

/// Which `BoundaryValueProblem` a `SolverConfig`/GUI session is driving. Single source of
/// truth shared by `pinn-app`'s CLI parsing and `pinn-gui`'s problem selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ProblemKind {
    #[default]
    Kirsch,
    PinLug,
}

/// Which execution strategy the training loop should use for host-side work (resampling
/// today; a future CPU-parallel/GPU-dispatch executor later — see `pinn_solver::execution`).
/// Phase 1 of the hardware-adaptive-execution epic: `Serial` is what every code path already
/// does, and `Auto` currently resolves to `Serial` unconditionally
/// (`pinn_solver::execution::ExecutionPlanner` is a deliberate stub until a real workload-aware
/// decision is warranted by profiling data — see that module's own doc comment). No
/// `CpuParallel`/`Gpu` variants yet: adding them with nothing behind them would invite dead-code
/// noise and a false impression of capability that doesn't exist yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    #[default]
    Auto,
    Serial,
}

/// Coarse hardware/resource-usage target (Eco/Balanced/Performance/Maximum), independent of
/// `ExecutionMode` (mode picks *how* work executes; profile is meant to eventually cap *how
/// much* — batch size, thread count). Phase 1 only accepts, validates, and threads this value
/// through `SolverConfig`/`pinn.env`; nothing reads it yet to change behavior (see
/// `pinn_solver::execution`'s module doc for why: profiling-driven optimization, not a guess,
/// decides what `Eco`/`Performance`/`Maximum` should each concretely do). Never alters the
/// mathematical formulation being solved — only ever execution strategy, once wired.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PerformanceProfile {
    Eco,
    #[default]
    Balanced,
    Performance,
    Maximum,
}

/// `ExecutionMode` + `PerformanceProfile` bundled onto `SolverConfig`, following the same
/// opt-in-subsystem-config shape as `DecisionMakerConfig`/`StiffnessConfig`/`WidthGrowthConfig`.
/// Phase 1: read from `pinn.env`'s `EXEC_MODE`/`EXEC_PROFILE` keys (`pinn-app/src/main.rs`'s
/// `apply_env`), round-tripped and validated, not yet load-bearing on any executed code path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ExecutionConfig {
    pub mode: ExecutionMode,
    pub profile: PerformanceProfile,
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
    ///
    /// Kirsch-derived; reused verbatim for pin-lug (`PinnDecisionMaker::new`'s `allow_converge`
    /// arm, see CLAUDE.md's Multi-domain Converge-tier L-BFGS section) with no problem-specific
    /// derivation (issue #42, untuned). Pin-lug's Signorini KKT complementarity terms
    /// (`interface_penetration`/`interface_non_tension`) have discontinuous curvature at the
    /// active-set boundary — a structurally different gradient-conflict regime from Kirsch's
    /// smooth energy landscape — so this threshold may gate Converge entry too early or too
    /// late for pin-lug specifically. Retuning requires a real multi-thousand-step pin-lug run
    /// (tracked in issue #42), not a code-only change.
    pub converge_cosine_min: f32,
    /// g_total_norm (g_pde_norm + g_bc_norm) threshold for Converge entry (default: 5.0).
    /// Dimensionless, relative to O(1)-normalized losses on a 25% collocation subset.
    ///
    /// Same Kirsch-derived-but-untuned-for-pin-lug caveat as `converge_cosine_min` above
    /// (issue #42) — shared verbatim across both problems' `DecisionMakerConfig`.
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

/// Configuration for one-shot, fixed-step-count, function-preserving network width growth
/// (Net2WiderNet-style — see `pinn_solver::network::ElasticityNet::grow_width`), opt-in and
/// disabled by default. Unlike `DecisionMakerConfig`/`StiffnessConfig`, growth is triggered by
/// a plain step-count comparison (`step == trigger_step`), not a plateau/gradient-conflict
/// signal — a deliberate v1 scope cut (see the issue #50 design doc), not an oversight.
///
/// v1 scope: only read by the Kirsch headless path (`pinn_solver::headless::run_headless`) —
/// `run_headless_pinlug_inner`, `runner.rs`'s GUI-driving paths, and pin-lug entirely do not
/// read this field yet, mirroring `use_piratenet_compute_skip`'s existing GUI-absence
/// precedent. Present on both `default_kirsch()`/`default_pinlug()` regardless (matching this
/// struct's existing flag-uniformity convention) so `SolverConfig` stays a single shared shape
/// across both problems even though only one problem's training loop currently acts on it.
#[derive(Clone, Default)]
pub struct WidthGrowthConfig {
    /// Enable the one-shot width-growth event (default: false — opt-in).
    pub enabled: bool,
    /// The training step at which growth fires (compared with plain integer equality against
    /// the training loop's own step counter — zero GPU sync). Default: 0.
    pub trigger_step: usize,
    /// `hidden_dim` to grow to. Must be strictly greater than `SolverConfig::hidden_dim` when
    /// `enabled = true` (`ElasticityNet::grow_width` panics otherwise). Default: 0.
    pub target_hidden_dim: usize,
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
    /// If true, `training_core::compute_reference_scales` (and `PinLugProblem::new`'s
    /// internal equivalent) normalizes stress by `config.material.ultimate_strength_pa`
    /// instead of the applied load (`config.load.px`). Disabled by default — the applied-
    /// load normalization is what the K_t=3.0 Kirsch validation and pin-lug's tuned
    /// SAW-BRDR/LR/ConvergenceTracker thresholds were established against; switching the
    /// stress reference changes every loss term's O(1) magnitude by roughly
    /// `(Px/ultimate_strength_pa)^2` and must be an explicit, informed choice.
    pub use_ultimate_strength_scaling: bool,
    /// Skip forward/backward compute (not just SOAP-Muon preconditioning) for PirateNet
    /// hidden blocks whose gate magnitude is below `stiffness.gate_awake_epsilon`. No-op
    /// when `use_piratenet=false` (gates are empty). Default false — opt-in, matching
    /// `use_ultimate_strength_scaling`'s convention: a numerically-provable-lossless
    /// optimization (see network.rs's `dormant_block_gradient_is_exactly_zero`) that still
    /// ships behind a kill-switch because it changes autodiff-graph structure per step.
    pub use_piratenet_compute_skip: bool,
    /// One-shot function-preserving network width growth (issue #50). Disabled by default —
    /// see [`WidthGrowthConfig`]'s doc comment for scope (Kirsch headless only in v1).
    pub width_growth: WidthGrowthConfig,
    /// Execution-mode/performance-profile selection (hardware-adaptive-execution epic, Phase
    /// 1). See [`ExecutionConfig`]'s doc comment — accepted/validated, not yet load-bearing.
    pub execution: ExecutionConfig,
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
            use_ultimate_strength_scaling: false,
            use_piratenet_compute_skip: false,
            width_growth: WidthGrowthConfig::default(),
            execution: ExecutionConfig::default(),
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
            use_ultimate_strength_scaling: false,
            use_piratenet_compute_skip: false,
            width_growth: WidthGrowthConfig::default(),
            execution: ExecutionConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_kirsch_has_ultimate_strength_scaling_disabled() {
        assert!(!SolverConfig::default_kirsch().use_ultimate_strength_scaling);
    }

    #[test]
    fn default_pinlug_has_ultimate_strength_scaling_disabled() {
        assert!(!SolverConfig::default_pinlug().use_ultimate_strength_scaling);
    }

    #[test]
    fn solver_config_use_piratenet_compute_skip_defaults_to_false() {
        assert!(!SolverConfig::default_kirsch().use_piratenet_compute_skip);
    }
}
