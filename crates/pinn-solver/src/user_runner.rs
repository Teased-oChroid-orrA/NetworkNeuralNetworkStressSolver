//! Minimal headless runner for a [`UserDefinedProblem`] (`user_problem.rs`) — the third,
//! generic entry point alongside `headless::run_headless`/`run_headless_pinlug`. Deliberately
//! v1-simple: no `PinnDecisionMaker`/`ConvergenceTracker`/L-BFGS-Converge/warm-restart
//! cascade (all confirmed optional extras layered around `step_physics_multi`, not required
//! by it) — a plain constant/scheduled-LR AdamW loop, matching `toy_beam`'s own "prove the
//! formulation converges before adding curriculum machinery" scope discipline.

use burn::tensor::backend::Backend;
use serde::Serialize;
use pinn_core::messages::SolverConfig;
use pinn_core::problem_spec::ProblemSpec;

use crate::{
    fd_stencil::FdConfig,
    lr_schedule::LrSchedule,
    network::ElasticityNetConfig,
    optim::{make_bias_optim, make_gate_optim, WeightOptim},
    problem::{BoundaryValueProblem, DomainOptim},
    saw_brdr::SawBrdr,
    training_core::{step_physics_multi, sync_device, BDevice, B, StepOutput},
    user_problem::{
        plate_normalize_point as normalize_point, resample_plate_step_data, plate_multi_step_ctx,
        plate_multi_domain_step_ctx, resample_domain_step_data, AnnularDecompositionProblem,
        UserDefinedProblem, OuterStageProblem, AnnulusStageProblem, phase2_interface_parametrization,
        ANNULUS_DOMAIN, OUTER_DOMAIN,
    },
};

/// One deterministic L5 checkpoint. Values are deliberately derived from displacement or
/// loss tensors; direct mDEM stress never supplies the acceptance Kt.
#[derive(Debug, Clone, Serialize)]
pub struct AnnularL5Diagnostic {
    pub step: usize,
    pub total_loss: f32,
    /// Issue #77 Step 4: kept for backward compatibility with earlier diagnostic JSON files -
    /// this is `out.lr`, the now-decorative SHARED `LrSchedule`'s own result. It does NOT
    /// drive either domain's optimizer once `annulus_lr`/`outer_lr` below are populated (see
    /// `PHASE_4_IMPLEMENTATION_MANIFEST.md`'s PH4-27). Prefer those two fields.
    pub learning_rate: f64,
    /// The annulus domain's REAL, effective learning rate this step (from its own
    /// `LrSchedule`, actually applied by `step_physics_multi` via `MultiStepCtx::
    /// per_domain_lr`) - added specifically to let a future run distinguish "the annulus
    /// domain's own schedule is still decaying prematurely" from other hypotheses, per
    /// PH4-27's own open-question list.
    pub annulus_lr: f64,
    /// Same as `annulus_lr`, for the outer domain.
    pub outer_lr: f64,
    pub kt_derived_fd_vm: f64,
    pub direct_stress_rms: f64,
    pub derived_stress_rms: f64,
    pub direct_derived_mismatch_rms: f64,
    pub derived_traction_rms: f64,
    pub terms: Vec<AnnularTermDiagnostic>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnnularTermDiagnostic {
    pub name: String,
    pub raw: f32,
    pub effective_weight: f64,
    pub gradient_norm: Option<f32>,
    pub gradient_share: Option<f32>,
}

#[allow(clippy::too_many_arguments)]
fn annular_l5_diagnostic(
    step: usize,
    out: &StepOutput,
    annulus_lr: f64,
    outer_lr: f64,
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    fd: &FdConfig,
    device: &BDevice,
    ansatz: &dyn pinn_core::problem::DirichletAnsatz,
) -> AnnularL5Diagnostic {
    let scales = crate::training_core::compute_reference_scales_for_plate(spec);
    let hole = &spec.geometry.holes[0];
    let margin = crate::user_problem::ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
    // Issue #77 PH4-41: `ansatz` is the annulus domain's OWN real ansatz (`AnnularDecompositionProblem::
    // ansatz(0)` - `Identity` or the hard-constraint one), and `affine` mirrors the SAME
    // `decomposition_applicable` gate `AnnularDecompositionProblem::loss_terms()` itself uses -
    // every real Kt number this whole investigation (PH4-24 through PH4-40) reported came from
    // this function, previously missing both.
    let affine = crate::user_problem::decomposition_applicable(spec).then_some((spec.load.px, spec.load.py));
    let profile = crate::user_problem::probe_hole_boundary_profile_derived(
        model, &spec.geometry, hole, 144, fd, scales.u_ref, scales.stress_ref,
        &spec.material, margin, device, ansatz, affine,
    );
    let kt = crate::user_problem::stress_concentration_from_profile(&profile, spec.load.px.abs()).kt;
    let stress = crate::user_problem::probe_hole_stress_diagnostic(
        model, &spec.geometry, hole, 144, fd, scales.u_ref, scales.stress_ref,
        &spec.material, margin, device, ansatz,
    );
    let raw = out.raw_scalar_by_name.as_ref();
    let weights = out.lam_by_name.as_ref();
    let norms = out.term_grad_norms.as_ref();
    let shares = out.gradient_share_report.as_ref();
    let mut names: Vec<&str> = raw.into_iter().flat_map(|m| m.keys().copied()).collect();
    names.sort_unstable();
    let terms = names.into_iter().map(|name| AnnularTermDiagnostic {
        name: name.to_owned(),
        raw: raw.and_then(|m| m.get(name)).copied().unwrap_or(f32::NAN),
        effective_weight: weights.and_then(|m| m.get(name)).copied().unwrap_or(f64::NAN),
        gradient_norm: norms.and_then(|m| m.get(name)).copied(),
        gradient_share: shares.and_then(|s| s.shares.get(name)).copied(),
    }).collect();
    AnnularL5Diagnostic {
        step, total_loss: out.total_scalar, learning_rate: out.lr, annulus_lr, outer_lr, kt_derived_fd_vm: kt,
        direct_stress_rms: stress.direct_stress_rms, derived_stress_rms: stress.derived_stress_rms,
        direct_derived_mismatch_rms: stress.stress_mismatch_rms,
        derived_traction_rms: stress.derived_traction_rms, terms,
    }
}

/// Emit deterministic L5 diagnostics as JSON. Caller owns output path; normal training never
/// writes files implicitly.
pub fn write_annular_l5_diagnostics_json(
    path: impl AsRef<std::path::Path>,
    diagnostics: &[AnnularL5Diagnostic],
) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(diagnostics)
        .expect("AnnularL5Diagnostic contains only serializable finite-or-null fields");
    std::fs::write(path, text)
}

/// Train #77's bonded annular/global model pair. Kept public so GUI production dispatch and
/// L5 use identical optimizer, sampling, and loss wiring rather than maintaining two loops.
/// Issue #77 root-cause fix, Step 1 (see the investigation-branch plan and
/// `saw_brdr::grad_norm_damping_factors`'s own doc comment for the full mechanism): `0` means
/// the mechanism is fully disabled — every existing call site passes `0`, byte-identical to
/// before this parameter existed. A nonzero value is the step interval at which
/// `run_annular_decomposition_training_inner`'s loop, below, re-probes real per-term gradient
/// norms and recalibrates `SawBrdr`'s base weights via `grad_norm_damping_factors` — pulling a
/// gradient-dominant CONSTRAINT term (e.g. `interface_traction_continuity`) back down toward
/// the pinned physical-functional terms' own gradient scale, never amplifying anything.
///
/// PH4-36 Step 1 real result (`issue_77_grad_norm_rescale_l5_trace`, `period=150`,
/// `floor=0.1`): the mechanism DID activate (`hole_free`'s weight was damped from its default
/// `100.0` to `39.62` by the final checkpoint, combined `physical_potential`+`annulus_potential`
/// gradient share rose from PH4-34b's baseline ~18% to ~30%) but `hole_free` STILL dominated at
/// 64.2% share and Kt (1.261) did not meaningfully move past baseline (1.231, PH4-28) - within
/// this whole investigation's established noise band. `default_floor` (below) stays `0.1` for
/// every pre-existing caller; a caller MAY now pass a lower floor for a genuinely more
/// aggressive damping test, per the plan's own Step 1 escalation before moving to Step 2.
const GRAD_NORM_DAMPING_FLOOR: f32 = 0.1;

fn run_annular_decomposition_training_inner(
    spec: ProblemSpec,
    device: BDevice,
    mut on_step: impl FnMut(usize, f32, f64, usize) -> bool,
    diagnostic_steps: &[usize],
    // Issue #77 PH4-45: a sink callback (not an accumulator) so a live GUI/headless caller can
    // read the diagnostic's `kt` AND both models at the exact checkpoint it was computed -
    // `AnnularL5Diagnostic` alone doesn't carry a field-reconstructible heatmap. Every existing
    // caller below passes `&mut |d, _a, _o| diagnostics.push(d)`, reproducing the prior
    // accumulator behavior byte-identically (same values, same order, same cadence).
    on_diagnostic: &mut dyn FnMut(AnnularL5Diagnostic, &crate::network::ElasticityNet<crate::training_core::BInner>, &crate::network::ElasticityNet<crate::training_core::BInner>),
    interface_weight: f32,
    include_annulus_equilibrium: bool,
    annulus_n_fourier: usize,
    annulus_use_siren: bool,
    hole_free_weight: f32,
    annulus_use_hard_constraint: bool,
    grad_norm_rescale_period: usize,
    grad_norm_damping_floor: f32,
    annulus_use_log_polar: bool,
) -> (crate::network::ElasticityNet<B>, crate::network::ElasticityNet<B>, f32) {
    let problem = AnnularDecompositionProblem::new_experimental(
        spec.clone(), interface_weight, include_annulus_equilibrium, hole_free_weight, annulus_use_hard_constraint,
        annulus_use_log_polar,
    );
    crate::problem::validate_loss_terms(&problem);
    // Issue #77 spectral-bias fix (PH4-31): `annulus_n_fourier=0` (every existing caller)
    // gives the exact pre-existing `spec.geometry.coordinate_embedding()` value - this is
    // purely additive. Computed once and reused for both the annulus model's own input width
    // AND the per-step ctx's embedding (they must always agree, or the forward pass panics on
    // a width mismatch inside `compute_domain_forwards`).
    //
    // Issue #77 Phase 3 architectural redesign: `annulus_use_log_polar` (default `false`,
    // byte-identical) overrides to the log-polar embedding instead - deliberately NOT combined
    // with `annulus_n_fourier` (Fourier features are themselves a rejected candidate, PH4-31/32;
    // stacking two representation changes would confound which one caused any observed effect).
    let annulus_embedding = if annulus_use_log_polar {
        spec.geometry.log_polar_embedding()
    } else {
        spec.geometry.coordinate_embedding_with_fourier(annulus_n_fourier)
    };
    let mut config = SolverConfig::default_kirsch();
    config.load = spec.load;
    // Issue #77 SIREN hypothesis test: opt-in to the annulus domain only, same scope as every
    // other candidate this investigation has tested - the outer domain (already representing
    // only a small residual correction post-kinematic-decomposition) stays plain-tanh.
    let chart_cfg = ElasticityNetConfig::new()
        .with_input_dim(annulus_embedding.input_dim())
        .with_hidden_dim(spec.network.hidden_dim)
        .with_n_hidden(spec.network.n_hidden)
        .with_output_dim(5)
        .with_use_siren(annulus_use_siren);
    let raw_cfg = ElasticityNetConfig::new()
        .with_input_dim(3)
        .with_hidden_dim(spec.network.hidden_dim)
        .with_n_hidden(spec.network.n_hidden)
        .with_output_dim(5);
    B::seed(&device, spec.network.model_init_seed);
    let annulus_model = chart_cfg.init(&device);
    B::seed(&device, spec.network.model_init_seed ^ 0xA77A_0001);
    let outer_model = raw_cfg.init(&device);
    let mut models = vec![annulus_model, outer_model];
    let mut optims = (0..2).map(|_| DomainOptim {
        weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim(),
    }).collect::<Vec<_>>();
    let term_order: Vec<&'static str> = problem.loss_terms().iter().map(|t| t.name()).collect();
    // Kept separately from `saw.base_weights` (see `SawBrdr::base_weights`'s own doc comment):
    // issue #77's grad-norm rescale below always recalibrates from this ORIGINAL set, never
    // from the previous refresh's already-rescaled weights, so repeated refreshes don't
    // compound into runaway shrink.
    let base_weights: Vec<f32> = term_order.iter().map(|&n| problem.base_weight(n)).collect();
    let mut saw = SawBrdr::with_base(base_weights.clone(), 0.95);
    // Issue #77 Step 4 (`PHASE_4_IMPLEMENTATION_MANIFEST.md`'s PH4-26): a single shared
    // `LrSchedule` reading TOTAL loss was found to collapse LR for BOTH domains once
    // `physical_potential` (outer) plateaus almost immediately post-decomposition — starving
    // the annulus domain right around a real, measured Kt peak (a 12000-step run showed Kt
    // rise to 1.287 at step 3000, then DECLINE to 0.977 by step 11999). Each domain now gets
    // its OWN schedule, fed its OWN weighted-loss aggregate (`domain_weighted_loss`) instead
    // of the shared total. `lr_sched_shared` is kept only so `step_physics_multi`'s own
    // (unconditionally-called) `lr_sched.step(total_scalar)` bookkeeping still runs and
    // populates `StepOutput.lr` consistently — its result no longer drives either domain's
    // optimizer once `ctx.per_domain_lr` is set below.
    let mut lr_sched_annulus = LrSchedule::new(spec.training.lr, 100, 500);
    let mut lr_sched_outer = LrSchedule::new(spec.training.lr, 100, 500);
    let mut lr_sched_shared = LrSchedule::new(spec.training.lr, 100, 500);
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * spec.geometry.half_w, 2.0 * spec.geometry.half_h);
    let hole_fd = crate::user_problem::hole_fd_config_for_geometry(&fd, &spec.geometry);
    let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
    let placeholder = spec.geometry.to_placeholder();
    let n_annulus = (spec.training.n_interior / 2).max(256);
    let n_outer = spec.training.n_interior.saturating_sub(n_annulus).max(256);
    let mut last_total = f32::NAN;
    let mut prev_out: Option<crate::training_core::StepOutput> = None;
    for step in 0..spec.training.max_steps {
        let annulus_data = resample_domain_step_data(
            problem.domains()[0].id, problem.sampling_strategy(0), &placeholder, &spec.load,
            n_annulus, 0, spec.geometry.half_w, spec.geometry.half_h,
        );
        let outer_data = resample_domain_step_data(
            problem.domains()[1].id, problem.sampling_strategy(1), &placeholder, &spec.load,
            n_outer, spec.training.n_boundary, spec.geometry.half_w, spec.geometry.half_h,
        );
        // Issue #77 grad-norm rescale (Step 1): `period=0` (every existing caller) makes this
        // always `false`, so `probe_term_gradients` reduces to exactly its pre-existing
        // expression below — byte-identical. A nonzero period ORs in an extra probe step on
        // its own cadence, independent of `diagnostic_steps`.
        let grad_norm_probe_step = grad_norm_rescale_period > 0 && step % grad_norm_rescale_period == 0;
        let mut ctx = plate_multi_domain_step_ctx(
            &config, &problem, &fd, &hole_fd,
            &annulus_data, &outer_data,
            scales.u_ref, scales.ref_energy, scales.ref_stress2, spec.geometry.n_fourier(),
            annulus_embedding.clone(), diagnostic_steps.contains(&step) || grad_norm_probe_step, step,
        );
        // One-step lag: this step's own per-term losses aren't known until `step_physics_multi`
        // runs below, so (like every LR schedule) this reacts to the LAST observed reading.
        // Step 0 has no previous reading — `f64::MAX` matches `LrSchedule::best_loss`'s own
        // "nothing observed yet" sentinel, so it never spuriously triggers a plateau decay.
        let annulus_loss = prev_out.as_ref()
            .map(|o| crate::training_core::domain_weighted_loss(&problem, problem.domains()[0].id, true, o))
            .unwrap_or(f64::MAX);
        let outer_loss = prev_out.as_ref()
            .map(|o| crate::training_core::domain_weighted_loss(&problem, problem.domains()[1].id, true, o))
            .unwrap_or(f64::MAX);
        let lr_annulus = lr_sched_annulus.step(annulus_loss);
        let lr_outer = lr_sched_outer.step(outer_loss);
        ctx.per_domain_lr = Some(vec![lr_annulus, lr_outer]);
        let (new_models, out) = step_physics_multi(
            models, &mut optims, &ctx, &mut saw, &mut lr_sched_shared, &device, 0, 1.0, 1.0,
        );
        models = new_models;
        last_total = out.total_scalar;
        if grad_norm_probe_step {
            if let Some(norms) = out.term_grad_norms.as_ref() {
                let factors = crate::saw_brdr::grad_norm_damping_factors(
                    norms, &["physical_potential", "annulus_potential"], grad_norm_damping_floor,
                );
                let rescaled: Vec<f32> = term_order.iter().zip(base_weights.iter())
                    .map(|(&name, &orig)| orig * factors.get(name).copied().unwrap_or(1.0))
                    .collect();
                saw.set_base_weights(rescaled);
            }
        }
        if diagnostic_steps.contains(&step) {
            use burn::module::AutodiffModule;
            let annulus_valid = models[0].valid();
            let outer_valid = models[1].valid();
            let diag = annular_l5_diagnostic(step, &out, lr_annulus, lr_outer, &annulus_valid, &spec, &fd, &device, problem.ansatz(0));
            on_diagnostic(diag, &annulus_valid, &outer_valid);
        }
        // Reports the annulus domain's own LR (the one whose starvation this fix addresses) -
        // a logging/callback choice only, does not affect either domain's optimizer step.
        if on_step(step, out.total_scalar, lr_annulus, annulus_data.int_norm.len() + outer_data.int_norm.len()) {
            break;
        }
        prev_out = Some(out);
    }
    sync_device(&device);
    let mut models = models.into_iter();
    (models.next().expect("annulus model"), models.next().expect("outer model"), last_total)
}

/// Issue #77 PH4-45: production entry point exposing `use_hard_constraint`/`use_log_polar`
/// (PH4-42/44's corrected architectures) together with a live diagnostic sink, for a caller
/// (the GUI/headless dispatch) that wants real Kt/field data as training proceeds - not just
/// the loss/lr telemetry `on_step` alone carries. `diagnostic_steps` controls both the Kt
/// probe's AND `on_diagnostic`'s cadence; the caller is expected to compute
/// `evaluate_annular_vis_grid` itself from the models `on_diagnostic` hands it (this function
/// stays free of any GUI-specific `VisFields`/grid-size concern).
pub fn run_annular_decomposition_training_with_architecture(
    spec: ProblemSpec,
    device: BDevice,
    use_hard_constraint: bool,
    use_log_polar: bool,
    diagnostic_steps: &[usize],
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
    on_diagnostic: &mut dyn FnMut(AnnularL5Diagnostic, &crate::network::ElasticityNet<crate::training_core::BInner>, &crate::network::ElasticityNet<crate::training_core::BInner>),
) -> (crate::network::ElasticityNet<B>, crate::network::ElasticityNet<B>, f32) {
    run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, on_diagnostic, 100.0, false, 0, false, 100.0,
        use_hard_constraint, 0, GRAD_NORM_DAMPING_FLOOR, use_log_polar,
    )
}

/// Train #77's bonded annular/global model pair. Kept public so GUI production dispatch and
/// L5 use identical optimizer, sampling, and loss wiring rather than maintaining two loops.
pub fn run_annular_decomposition_training(
    spec: ProblemSpec,
    device: BDevice,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (crate::network::ElasticityNet<B>, crate::network::ElasticityNet<B>, f32) {
    run_annular_decomposition_training_inner(spec, device, on_step, &[], &mut |_d, _a, _o| {}, 100.0, false, 0, false, 100.0, false, 0, GRAD_NORM_DAMPING_FLOOR, false)
}

/// Same production runner with opt-in, deterministic diagnostic checkpoints. No output file is
/// created here; callers explicitly persist the returned records with
/// [`write_annular_l5_diagnostics_json`].
pub fn run_annular_decomposition_training_with_diagnostics(
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (
    crate::network::ElasticityNet<B>,
    crate::network::ElasticityNet<B>,
    f32,
    Vec<AnnularL5Diagnostic>,
) {
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let (annulus, outer, loss) = run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, &mut |d, _a, _o| diagnostics.push(d), 100.0, false, 0, false, 100.0, false, 0, GRAD_NORM_DAMPING_FLOOR, false,
    );
    (annulus, outer, loss, diagnostics)
}

/// Issue #77 candidate (b): same as [`run_annular_decomposition_training_with_diagnostics`],
/// but with `interface_weight` exposed for a real, controlled A/B comparison against the
/// default `100.0` — see `AnnularDecompositionProblem::interface_weight`'s own doc comment
/// and `PHASE_4_IMPLEMENTATION_MANIFEST.md`'s PH4-28 open-question list for why this exists.
/// Not used by any production entry point; experimental-comparison callers only.
pub fn run_annular_decomposition_training_with_diagnostics_and_interface_weight(
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    interface_weight: f32,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (
    crate::network::ElasticityNet<B>,
    crate::network::ElasticityNet<B>,
    f32,
    Vec<AnnularL5Diagnostic>,
) {
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let (annulus, outer, loss) = run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, &mut |d, _a, _o| diagnostics.push(d), interface_weight, false, 0, false, 100.0, false, 0, GRAD_NORM_DAMPING_FLOOR, false,
    );
    (annulus, outer, loss, diagnostics)
}

/// Issue #77 next candidate (PH4-29): same as
/// [`run_annular_decomposition_training_with_diagnostics`], but with
/// `include_annulus_equilibrium` exposed to test whether a strong-form residual on the
/// annulus domain closes (or narrows) the Kt gap that pure variational energy minimization,
/// collocation margin, sampling variance, training-dynamics/LR, and interface-continuity
/// weight have all failed to close. See `AnnularDecompositionProblem::
/// include_annulus_equilibrium`'s own doc comment. Not used by any production entry point.
pub fn run_annular_decomposition_training_with_diagnostics_and_annulus_equilibrium(
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    include_annulus_equilibrium: bool,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (
    crate::network::ElasticityNet<B>,
    crate::network::ElasticityNet<B>,
    f32,
    Vec<AnnularL5Diagnostic>,
) {
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let (annulus, outer, loss) = run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, &mut |d, _a, _o| diagnostics.push(d), 100.0, include_annulus_equilibrium, 0, false, 100.0, false, 0, GRAD_NORM_DAMPING_FLOOR, false,
    );
    (annulus, outer, loss, diagnostics)
}

/// Issue #77 spectral-bias fix (PH4-31): same as
/// [`run_annular_decomposition_training_with_diagnostics`], but with `annulus_n_fourier`
/// exposed to test whether multi-scale Fourier features on the annulus domain's hole-relative
/// coordinates close (or narrow) the Kt gap - the working hypothesis after six other tested
/// mechanisms (representation, collocation margin, sampling variance, training-dynamics/LR,
/// interface-continuity weight, a strong-form residual) were each fixed or ruled out without
/// closing it. See `CoordinateEmbedding::SingleHoleChart::n_fourier`'s own doc comment. Not
/// used by any production entry point.
pub fn run_annular_decomposition_training_with_diagnostics_and_annulus_fourier(
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    annulus_n_fourier: usize,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (
    crate::network::ElasticityNet<B>,
    crate::network::ElasticityNet<B>,
    f32,
    Vec<AnnularL5Diagnostic>,
) {
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let (annulus, outer, loss) = run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, &mut |d, _a, _o| diagnostics.push(d), 100.0, false, annulus_n_fourier, false, 100.0, false, 0, GRAD_NORM_DAMPING_FLOOR, false,
    );
    (annulus, outer, loss, diagnostics)
}

/// Issue #77 SIREN hypothesis test: same as
/// [`run_annular_decomposition_training_with_diagnostics`], but with `use_siren` exposed on the
/// annulus domain's own network to test whether SIREN's sinusoidal activation (Sitzmann et al.
/// 2020) - a network-wide representational change, not just an input-feature addition like the
/// Fourier-feature candidate - closes or narrows the Kt gap that eight other tested mechanisms
/// (representation, collocation margin, sampling variance, training-dynamics/LR,
/// interface-continuity weight, a strong-form residual, spectral-bias input features at two
/// frequency counts, and finite-domain scale) have each been fixed or ruled out without
/// closing. Not used by any production entry point; experimental-comparison callers only.
pub fn run_annular_decomposition_training_with_diagnostics_and_siren(
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    use_siren: bool,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (
    crate::network::ElasticityNet<B>,
    crate::network::ElasticityNet<B>,
    f32,
    Vec<AnnularL5Diagnostic>,
) {
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let (annulus, outer, loss) = run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, &mut |d, _a, _o| diagnostics.push(d), 100.0, false, 0, use_siren, 100.0, false, 0, GRAD_NORM_DAMPING_FLOOR, false,
    );
    (annulus, outer, loss, diagnostics)
}

/// Issue #77 gradient-share hypothesis (PH4-34): same as
/// [`run_annular_decomposition_training_with_diagnostics`], but with `hole_free_weight` exposed
/// on the annulus domain's own `hole_free` (traction-free boundary condition) term. Real
/// diagnostic evidence from every prior comparison run shows `hole_free`'s raw loss converges
/// to a tiny residual (~1e-4) by step ~1500 yet its gradient share climbs to 70-90% of the
/// entire optimization's gradient budget by step 2999, while `physical_potential` (far from
/// converged) gets only ~15%. Tests whether reducing this weight frees gradient budget for the
/// energy terms that actually shape the stress field, without meaningfully un-satisfying the
/// already-small traction residual. See `AnnularDecompositionProblem::hole_free_weight`'s own
/// doc comment. Not used by any production entry point.
pub fn run_annular_decomposition_training_with_diagnostics_and_hole_free_weight(
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    hole_free_weight: f32,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (
    crate::network::ElasticityNet<B>,
    crate::network::ElasticityNet<B>,
    f32,
    Vec<AnnularL5Diagnostic>,
) {
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let (annulus, outer, loss) = run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, &mut |d, _a, _o| diagnostics.push(d), 100.0, false, 0, false, hole_free_weight, false, 0, GRAD_NORM_DAMPING_FLOOR, false,
    );
    (annulus, outer, loss, diagnostics)
}

/// Issue #77 PH4-35: same as [`run_annular_decomposition_training_with_diagnostics`], but with
/// the exact, closed-form hard-constraint hole ansatz active on the annulus domain
/// (`use_hard_constraint=true`) instead of the soft `hole_free` penalty. See
/// `AnnularDecompositionProblem::new_with_hard_constraint_ansatz`'s and
/// `kirsch_hole_correction`'s own doc comments. Not used by any production entry point.
pub fn run_annular_decomposition_training_with_diagnostics_and_hard_constraint(
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    use_hard_constraint: bool,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (
    crate::network::ElasticityNet<B>,
    crate::network::ElasticityNet<B>,
    f32,
    Vec<AnnularL5Diagnostic>,
) {
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let (annulus, outer, loss) = run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, &mut |d, _a, _o| diagnostics.push(d), 100.0, false, 0, false, 100.0, use_hard_constraint, 0, GRAD_NORM_DAMPING_FLOOR, false,
    );
    (annulus, outer, loss, diagnostics)
}

/// Issue #77 Phase 3 architectural redesign: same as
/// [`run_annular_decomposition_training_with_diagnostics`], but with the annulus domain's own
/// coordinate embedding switched to `CoordinateEmbedding::LogPolar` instead of `SingleHoleChart`
/// — a true reparameterization of the network's input coordinate system (not another feature
/// added on top of raw x,y, which PH4-31/32 already tried as Fourier features and rejected).
/// `use_hard_constraint` lets this combine with Phase 1/PH4-35's exact hole ansatz (orthogonal
/// axes — representation vs. traction-free enforcement); not combined with `annulus_n_fourier`
/// or SIREN in this pass, to keep which representation change caused any observed effect
/// unambiguous. Not used by any production entry point; experimental-comparison callers only.
pub fn run_annular_decomposition_training_with_diagnostics_and_log_polar_embedding(
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    use_log_polar: bool,
    use_hard_constraint: bool,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (
    crate::network::ElasticityNet<B>,
    crate::network::ElasticityNet<B>,
    f32,
    Vec<AnnularL5Diagnostic>,
) {
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let (annulus, outer, loss) = run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, &mut |d, _a, _o| diagnostics.push(d), 100.0, false, 0, false, 100.0, use_hard_constraint, 0, GRAD_NORM_DAMPING_FLOOR, use_log_polar,
    );
    (annulus, outer, loss, diagnostics)
}

/// Issue #77 PH4-36 Step 2: combines PH4-35's hard-constraint hole ansatz WITH the grad-norm
/// rescale mechanism (Step 1). PH4-35's own real diagnostic showed that removing `hole_free`
/// does NOT hand the freed gradient budget to `physical_potential`/`annulus_potential` - it
/// hands it to whichever term is next structurally privileged, `interface_traction_continuity`
/// (62.8% share, unaddressed by PH4-35 itself). Since `grad_norm_damping_factors` is generic
/// over "every active term except the reference set" (`saw_brdr.rs`'s own doc comment), it
/// applies automatically to `interface_traction_continuity`/`interface_displacement_
/// continuity`/`translation_gauge`/`rotation_gauge` once `hole_free` is absent under
/// `use_hard_constraint=true` - no new rescale logic needed, only this combined entry point.
pub fn run_annular_decomposition_training_with_diagnostics_and_hard_constraint_and_grad_norm_rescale(
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    use_hard_constraint: bool,
    grad_norm_rescale_period: usize,
    grad_norm_damping_floor: f32,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (
    crate::network::ElasticityNet<B>,
    crate::network::ElasticityNet<B>,
    f32,
    Vec<AnnularL5Diagnostic>,
) {
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let (annulus, outer, loss) = run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, &mut |d, _a, _o| diagnostics.push(d), 100.0, false, 0, false, 100.0, use_hard_constraint,
        grad_norm_rescale_period, grad_norm_damping_floor, false,
    );
    (annulus, outer, loss, diagnostics)
}

/// Issue #77 root-cause fix, Step 1: same as
/// [`run_annular_decomposition_training_with_diagnostics`], but with `grad_norm_rescale_period`
/// exposed — the gradient-norm-aware base-weight recalibration this investigation's own plan
/// document identifies as the untried mechanism behind all twelve prior candidates' failures
/// (`SawBrdr::update` reacts only to loss-VALUE decay rate, never to a term's real
/// backpropagated gradient magnitude; PH4-34b's own nominal-weight-reduction test proved the
/// two are only loosely coupled). `0` disables the mechanism entirely (byte-identical to every
/// other entry point above). A nonzero value re-probes real per-term gradient norms every that
/// many steps and rescales `SawBrdr`'s base weights via `saw_brdr::grad_norm_damping_factors`
/// — damping a gradient-dominant CONSTRAINT term back toward the pinned physical-functional
/// terms' own scale, never amplifying anything and never touching `physical_potential`/
/// `annulus_potential` themselves (that pin is enforced independently, in `step_physics_multi`'s
/// own hardcoded match arm — see `grad_norm_damping_factors`'s doc comment). Not used by any
/// production entry point; experimental-comparison callers only. `grad_norm_damping_floor`
/// lets a caller escalate to a MORE aggressive damping than the module default
/// (`GRAD_NORM_DAMPING_FLOOR`, `0.1`) - see that constant's own doc comment for the real
/// `floor=0.1` result that motivated exposing this as a real, separately-testable parameter
/// rather than a hardcoded constant.
pub fn run_annular_decomposition_training_with_diagnostics_and_grad_norm_rescale(
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    grad_norm_rescale_period: usize,
    grad_norm_damping_floor: f32,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (
    crate::network::ElasticityNet<B>,
    crate::network::ElasticityNet<B>,
    f32,
    Vec<AnnularL5Diagnostic>,
) {
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let (annulus, outer, loss) = run_annular_decomposition_training_inner(
        spec, device, on_step, diagnostic_steps, &mut |d, _a, _o| diagnostics.push(d), 100.0, false, 0, false, 100.0, false, grad_norm_rescale_period, grad_norm_damping_floor, false,
    );
    (annulus, outer, loss, diagnostics)
}

/// Issue #77 Phase 2 architectural redesign: real, tiny-forward-pass evaluation of a FROZEN
/// (non-autodiff, `BInner`) model's displacement at a fixed set of physical points — used to
/// read Stage A's trained outer model's own interface trace once, before Stage B starts (the
/// model never changes during Stage B, so its interface trace is provably invariant across
/// every one of Stage B's steps — computing it once, not "fresh each step" as a naive
/// re-evaluation would, is a real efficiency this frozen-model design affords, not an
/// approximation). No FD stencil needed (a value read, not a derivative) — a plain
/// `fwd_embedded` forward pass, `u_ref`-rescaled by hand exactly like `probe_hole_boundary_
/// profile_derived`'s own convention (`fwd_embedded`'s raw output is in NORMALIZED units).
fn evaluate_frozen_outer_interface_displacement(
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    geometry: &pinn_core::user_geometry::UserGeometry,
    interface_radius: f64,
    thetas: &[f64],
    u_ref: f32,
    device: &BDevice,
) -> (Vec<f32>, Vec<f32>) {
    use crate::network::fwd_embedded;
    use crate::fd_stencil::norm_pts_to_tensor;
    use crate::training_core::BInner;
    let n = thetas.len();
    let pts_norm: Vec<[f32; 2]> = thetas.iter().map(|&theta| {
        let (x, y) = (interface_radius * theta.cos(), interface_radius * theta.sin());
        [(x / geometry.half_w) as f32, (y / geometry.half_h) as f32]
    }).collect();
    let pts_t = norm_pts_to_tensor::<BInner>(&pts_norm, device);
    // The frozen model here is always the OUTER domain's own model, which `OuterStageProblem`
    // always constructs raw (`with_input_dim(3)`, matching `AnnularDecompositionProblem`'s own
    // outer model convention) - NOT `geometry.coordinate_embedding()` (the chart embedding,
    // sized for the ANNULUS model), which would silently mismatch this model's actual 3-column
    // width and panic deep inside the matmul, not with a clear "wrong embedding" message.
    let raw = fwd_embedded::<BInner>(model, pts_t, pinn_core::user_geometry::CoordinateEmbedding::Raw, device);
    let u: Vec<f32> = raw.clone().slice([0..n, 0..1]).reshape([n]).mul_scalar(u_ref as f64)
        .into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    let v: Vec<f32> = raw.slice([0..n, 1..2]).reshape([n]).mul_scalar(u_ref as f64)
        .into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    (u, v)
}

/// Issue #77 Phase 2 architectural redesign: sequential two-stage training with ONE-DIRECTIONAL
/// domain coupling — see `OuterStageProblem`/`AnnulusStageProblem`'s own doc comments
/// (`user_problem.rs`) and the investigation-branch plan's Phase 2 design for the full
/// rationale. A third, independent training function — matches this codebase's own established
/// "two step-driver functions by design" precedent for structurally different loops (frozen
/// `step_physics`/additive `step_physics_multi`) rather than forcing a third shape through
/// `run_annular_decomposition_training_inner`'s existing simultaneous-joint loop, which stays
/// completely unmodified.
///
/// Stage A trains the outer domain alone (against the EXACT closed-form interface trace, no
/// live annulus network) for `stage_a_steps`; the resulting model is frozen (`AutodiffModule::
/// valid()`) and its own interface displacement evaluated ONCE. Stage B trains the annulus
/// domain alone for `stage_b_steps`, anchored to that frozen, fixed target — no term in either
/// stage has an adaptively-reweighted coefficient; every registered term keeps its plain static
/// `base_weight` (`OuterStageProblem`/`AnnulusStageProblem::base_weight`), matching this
/// redesign's explicit "no more reweighting" scope.
#[allow(clippy::too_many_arguments)]
pub fn run_annular_decomposition_training_sequential(
    spec: ProblemSpec,
    device: BDevice,
    stage_a_steps: usize,
    stage_b_steps: usize,
    stage_b_hard_constraint: bool,
    diagnostic_steps: &[usize],
    mut on_step: impl FnMut(&str, usize, f32) -> bool,
    // Issue #77 PH4-45: additive live-streaming hook, called at the same point/cadence as the
    // `diagnostics.push` below - lets a GUI/headless caller read Stage B's real per-checkpoint
    // Kt AND both models (Stage B's own live annulus model, Stage A's frozen outer model)
    // without waiting for this function to return. Every existing test caller passes
    // `&mut |_d, _a, _o| {}` and keeps reading the returned `Vec<UserProblemL5Diagnostic>`
    // exactly as before - purely additive, no existing behavior changed.
    on_diagnostic: &mut dyn FnMut(&UserProblemL5Diagnostic, &crate::network::ElasticityNet<crate::training_core::BInner>, &crate::network::ElasticityNet<crate::training_core::BInner>),
) -> (
    crate::network::ElasticityNet<crate::training_core::BInner>,
    crate::network::ElasticityNet<crate::training_core::BInner>,
    f32,
    Vec<UserProblemL5Diagnostic>,
) {
    use burn::module::AutodiffModule;

    let mut config = SolverConfig::default_kirsch();
    config.load = spec.load;
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * spec.geometry.half_w, 2.0 * spec.geometry.half_h);
    let hole_fd = crate::user_problem::hole_fd_config_for_geometry(&fd, &spec.geometry);
    let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
    let placeholder = spec.geometry.to_placeholder();

    // --- Stage A: outer domain alone, anchored to the exact closed-form interface trace.
    let outer_problem = OuterStageProblem::new(spec.clone());
    crate::problem::validate_loss_terms(&outer_problem);
    let raw_cfg = ElasticityNetConfig::new().with_input_dim(3)
        .with_hidden_dim(spec.network.hidden_dim).with_n_hidden(spec.network.n_hidden).with_output_dim(5);
    // Same outer-model seed convention `run_annular_decomposition_training_inner` already
    // established, so Stage A's outer model starts from the identical init every other
    // annular-decomposition comparison's own outer model does.
    B::seed(&device, spec.network.model_init_seed ^ 0xA77A_0001);
    let mut outer_model = raw_cfg.init(&device);
    let mut outer_optim = DomainOptim { weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim() };
    let mut outer_saw = SawBrdr::with_base(outer_problem.loss_terms().iter().map(|t| outer_problem.base_weight(t.name())).collect(), 0.95);
    let mut outer_lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
    for step in 0..stage_a_steps {
        let data = resample_domain_step_data(
            OUTER_DOMAIN, outer_problem.sampling_strategy(0), &placeholder, &spec.load,
            spec.training.n_interior, spec.training.n_boundary, spec.geometry.half_w, spec.geometry.half_h,
        );
        let ctx = plate_multi_step_ctx(
            &config, &outer_problem, &fd, &hole_fd, &data, scales.u_ref, scales.ref_energy, scales.ref_stress2,
            spec.geometry.n_fourier(), spec.geometry.coordinate_embedding(), false, step,
        );
        let (new_model, out) = step_physics_multi(
            vec![outer_model], std::slice::from_mut(&mut outer_optim), &ctx, &mut outer_saw, &mut outer_lr_sched, &device, 0, 1.0, 1.0,
        );
        outer_model = new_model.into_iter().next().expect("single model");
        if on_step("stage_a", step, out.total_scalar) {
            break;
        }
    }
    let frozen_outer = outer_model.valid();

    // --- Stage B: annulus domain alone, anchored to the frozen outer model's OWN interface
    // trace, evaluated once (see `evaluate_frozen_outer_interface_displacement`'s own doc
    // comment for why once is correct, not an approximation).
    let partition = spec.geometry.annular_partition().expect("validated by OuterStageProblem/AnnulusStageProblem constructors");
    let interface = phase2_interface_parametrization();
    let (target_u, target_v) = evaluate_frozen_outer_interface_displacement(
        &frozen_outer, &spec.geometry, partition.interface_radius, &interface.thetas, scales.u_ref, &device,
    );
    let annulus_problem = AnnulusStageProblem::new(spec.clone(), stage_b_hard_constraint);
    annulus_problem.set_frozen_interface_target(target_u, target_v);
    crate::problem::validate_loss_terms(&annulus_problem);
    let chart_cfg = ElasticityNetConfig::new()
        .with_input_dim(spec.geometry.coordinate_embedding().input_dim())
        .with_hidden_dim(spec.network.hidden_dim).with_n_hidden(spec.network.n_hidden).with_output_dim(5);
    B::seed(&device, spec.network.model_init_seed);
    let mut annulus_model = chart_cfg.init(&device);
    let mut annulus_optim = DomainOptim { weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim() };
    let mut annulus_saw = SawBrdr::with_base(annulus_problem.loss_terms().iter().map(|t| annulus_problem.base_weight(t.name())).collect(), 0.95);
    let mut annulus_lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let mut last_loss = f32::NAN;
    for step in 0..stage_b_steps {
        let data = resample_domain_step_data(
            ANNULUS_DOMAIN, annulus_problem.sampling_strategy(0), &placeholder, &spec.load,
            spec.training.n_interior, 0, spec.geometry.half_w, spec.geometry.half_h,
        );
        let ctx = plate_multi_step_ctx(
            &config, &annulus_problem, &fd, &hole_fd, &data, scales.u_ref, scales.ref_energy, scales.ref_stress2,
            spec.geometry.n_fourier(), spec.geometry.coordinate_embedding(), diagnostic_steps.contains(&step), step,
        );
        let (new_model, out) = step_physics_multi(
            vec![annulus_model], std::slice::from_mut(&mut annulus_optim), &ctx, &mut annulus_saw, &mut annulus_lr_sched, &device, 0, 1.0, 1.0,
        );
        annulus_model = new_model.into_iter().next().expect("single model");
        last_loss = out.total_scalar;
        if diagnostic_steps.contains(&step) {
            let annulus_valid = annulus_model.valid();
            let diag = user_problem_l5_diagnostic(step, &out, &annulus_valid, &spec, &fd, &device, annulus_problem.ansatz(0));
            on_diagnostic(&diag, &annulus_valid, &frozen_outer);
            diagnostics.push(diag);
        }
        if on_step("stage_b", step, out.total_scalar) {
            break;
        }
    }
    sync_device(&device);
    (frozen_outer, annulus_model.valid(), last_loss, diagnostics)
}

/// Minimal deterministic single-model loop for benchmark companions. Production UI/headless
/// retain their richer reporting loops; this avoids copying that training math into L5 tests.
/// `#[allow(dead_code)]`: only ever called from `#[cfg(test)]` code (`user_problem.rs`'s own
/// benchmark tests), so a plain `cargo build`/`cargo build --lib` (which doesn't compile test
/// code) genuinely sees no call site - a false-positive `dead_code`, not actually unused.
#[allow(dead_code)]
pub(crate) fn train_single_user_problem_for_benchmark(
    spec: ProblemSpec,
    device: &BDevice,
) -> crate::network::ElasticityNet<crate::training_core::BInner> {
    assert!(!AnnularDecompositionProblem::supports(&spec), "benchmark helper is single-model only");
    let problem = UserDefinedProblem::new(spec.clone());
    train_user_problem_for_benchmark(&problem, spec, device).0
}

/// Issue #77 Phase 1: same minimal training loop as
/// [`train_single_user_problem_for_benchmark`], but accepting an already-constructed `problem`
/// (e.g. one built via `UserDefinedProblem::new_with_hard_constraint_ansatz`) instead of always
/// building a plain `UserDefinedProblem::new(spec)` internally — lets test code exercise a real,
/// tiny end-to-end training run for any `UserDefinedProblem` configuration, not only the plain
/// default. Returns the trained model AND its final step's total loss (the benchmark caller
/// only ever needed the model; Phase 1's smoke test needs to assert the loss stayed finite).
/// `#[allow(dead_code)]`: same "test-only call site" false-positive as
/// [`train_single_user_problem_for_benchmark`]'s own attribute - see that doc comment.
#[allow(dead_code)]
pub(crate) fn train_user_problem_for_benchmark(
    problem: &UserDefinedProblem,
    spec: ProblemSpec,
    device: &BDevice,
) -> (crate::network::ElasticityNet<crate::training_core::BInner>, f32) {
    crate::problem::validate_loss_terms(problem);
    let mut config = SolverConfig::default_kirsch();
    config.load = spec.load;
    let net_cfg = ElasticityNetConfig::new().with_input_dim(spec.geometry.net_input_dim())
        .with_hidden_dim(spec.network.hidden_dim).with_n_hidden(spec.network.n_hidden).with_output_dim(5);
    B::seed(device, spec.network.model_init_seed);
    let mut model = net_cfg.init(device);
    let mut optim = DomainOptim { weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim() };
    let mut saw = SawBrdr::with_base(problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect(), 0.95);
    let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * spec.geometry.half_w, 2.0 * spec.geometry.half_h);
    let hole_fd = crate::user_problem::hole_fd_config_for_geometry(&fd, &spec.geometry);
    let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
    let placeholder = spec.geometry.to_placeholder();
    let mut last_loss = f32::NAN;
    for step in 0..spec.training.max_steps {
        let data = resample_plate_step_data(
            problem.sampling_strategy(0), &placeholder, &spec.load, spec.training.n_interior,
            spec.training.n_boundary, spec.geometry.half_w, spec.geometry.half_h,
        );
        // Issue #77 PH4-41 fix (finding 2) - see `run_user_problem_training_with_diagnostics`'s
        // identical comment for the full rationale.
        let weights = crate::user_problem::hole_bias_quadrature_weights(
            &data.int_norm, &spec.geometry, problem.hole_bias_fraction(),
        );
        problem.set_interior_weights(Some(weights));
        let ctx = plate_multi_step_ctx(
            &config, problem, &fd, &hole_fd, &data, scales.u_ref, scales.ref_energy, scales.ref_stress2,
            spec.geometry.n_fourier(), spec.geometry.coordinate_embedding(), false, step,
        );
        let (new_model, out) = step_physics_multi(
            vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, device, 0, 1.0, 1.0,
        );
        model = new_model.into_iter().next().expect("single model");
        last_loss = out.total_scalar;
    }
    use burn::module::AutodiffModule;
    (model.valid(), last_loss)
}

/// Issue #77 Phase 1 architectural redesign: single-domain analogue of `AnnularL5Diagnostic` —
/// no `annulus_lr`/`outer_lr` (there is only one model, one learning rate) or direct-vs-derived
/// stress-mismatch fields (single-domain models don't have a second domain to disagree with);
/// otherwise the same real, deterministic per-checkpoint record (`AnnularTermDiagnostic` is
/// already generic over term name/raw/weight/gradient-norm/gradient-share and reused verbatim).
#[derive(Debug, Clone, Serialize)]
pub struct UserProblemL5Diagnostic {
    pub step: usize,
    pub total_loss: f32,
    pub learning_rate: f64,
    pub kt_derived_fd_vm: f64,
    pub terms: Vec<AnnularTermDiagnostic>,
}

fn user_problem_l5_diagnostic(
    step: usize,
    out: &StepOutput,
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    fd: &FdConfig,
    device: &BDevice,
    ansatz: &dyn pinn_core::problem::DirichletAnsatz,
) -> UserProblemL5Diagnostic {
    let scales = crate::training_core::compute_reference_scales_for_plate(spec);
    let hole = &spec.geometry.holes[0];
    let margin = crate::user_problem::ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
    // Issue #77 PH4-41: same fix as `annular_l5_diagnostic` - `ansatz` is the caller's own
    // domain ansatz (Phase 1's `UserDefinedProblem` or Phase 2's `AnnulusStageProblem`), and
    // `affine` mirrors the SAME `decomposition_applicable` gate both problems' own
    // `loss_terms()` use.
    let affine = crate::user_problem::decomposition_applicable(spec).then_some((spec.load.px, spec.load.py));
    let profile = crate::user_problem::probe_hole_boundary_profile_derived(
        model, &spec.geometry, hole, 144, fd, scales.u_ref, scales.stress_ref,
        &spec.material, margin, device, ansatz, affine,
    );
    let kt = crate::user_problem::stress_concentration_from_profile(&profile, spec.load.px.abs()).kt;
    let raw = out.raw_scalar_by_name.as_ref();
    let weights = out.lam_by_name.as_ref();
    let norms = out.term_grad_norms.as_ref();
    let shares = out.gradient_share_report.as_ref();
    let mut names: Vec<&str> = raw.into_iter().flat_map(|m| m.keys().copied()).collect();
    names.sort_unstable();
    let terms = names.into_iter().map(|name| AnnularTermDiagnostic {
        name: name.to_owned(),
        raw: raw.and_then(|m| m.get(name)).copied().unwrap_or(f32::NAN),
        effective_weight: weights.and_then(|m| m.get(name)).copied().unwrap_or(f64::NAN),
        gradient_norm: norms.and_then(|m| m.get(name)).copied(),
        gradient_share: shares.and_then(|s| s.shares.get(name)).copied(),
    }).collect();
    UserProblemL5Diagnostic { step, total_loss: out.total_scalar, learning_rate: out.lr, kt_derived_fd_vm: kt, terms }
}

/// Issue #77 Phase 1: real Kt-measuring training loop for an already-constructed
/// `UserDefinedProblem` (e.g. via `new_with_hard_constraint_ansatz`) — the single-domain
/// counterpart of `run_annular_decomposition_training_with_diagnostics`, giving Phase 1's real
/// L5 comparison test the same checkpoint/diagnostic machinery every PH4-28..37 comparison test
/// already relies on, instead of a bespoke inline loop. Requires
/// `decomposition_applicable(&spec)` (one centered, traction-free hole — same scope
/// `probe_hole_boundary_profile_derived`'s own `hole = &spec.geometry.holes[0]` assumes).
pub fn run_user_problem_training_with_diagnostics(
    problem: UserDefinedProblem,
    spec: ProblemSpec,
    device: BDevice,
    diagnostic_steps: &[usize],
    mut on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (crate::network::ElasticityNet<crate::training_core::BInner>, f32, Vec<UserProblemL5Diagnostic>) {
    crate::problem::validate_loss_terms(&problem);
    let mut config = SolverConfig::default_kirsch();
    config.load = spec.load;
    let net_cfg = ElasticityNetConfig::new().with_input_dim(spec.geometry.net_input_dim())
        .with_hidden_dim(spec.network.hidden_dim).with_n_hidden(spec.network.n_hidden).with_output_dim(5);
    B::seed(&device, spec.network.model_init_seed);
    let mut model = net_cfg.init(&device);
    let mut optim = DomainOptim { weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim() };
    let mut saw = SawBrdr::with_base(problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect(), 0.95);
    let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * spec.geometry.half_w, 2.0 * spec.geometry.half_h);
    let hole_fd = crate::user_problem::hole_fd_config_for_geometry(&fd, &spec.geometry);
    let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
    let placeholder = spec.geometry.to_placeholder();
    let mut diagnostics = Vec::with_capacity(diagnostic_steps.len());
    let mut last_loss = f32::NAN;
    for step in 0..spec.training.max_steps {
        let data = resample_plate_step_data(
            problem.sampling_strategy(0), &placeholder, &spec.load, spec.training.n_interior,
            spec.training.n_boundary, spec.geometry.half_w, spec.geometry.half_h,
        );
        // Issue #77 PH4-41 fix (finding 2): `UserSamplingStrategy`'s hole-biased stratified
        // sampling (opt-in, Phase 1's own addition) draws a deliberately non-uniform point
        // density - `PhysicalPotentialEnergyTerm`'s own unweighted-mean energy integral
        // (`domain_integral_tensor`) is only a correct Monte-Carlo estimator under UNIFORM
        // density, so it must be told the real per-point weight whenever bias is active.
        // `hole_bias_fraction()<=0.0` (every pre-Phase-1 caller) makes `hole_bias_quadrature_
        // weights` return all-`1.0`, which `set_interior_weights(Some(...))` treats identically
        // to `None` (both feed `domain_integral_tensor`'s own uniform-mean path) - byte-
        // identical for every existing caller.
        let weights = crate::user_problem::hole_bias_quadrature_weights(
            &data.int_norm, &spec.geometry, problem.hole_bias_fraction(),
        );
        problem.set_interior_weights(Some(weights));
        let ctx = plate_multi_step_ctx(
            &config, &problem, &fd, &hole_fd, &data, scales.u_ref, scales.ref_energy, scales.ref_stress2,
            spec.geometry.n_fourier(), spec.geometry.coordinate_embedding(), diagnostic_steps.contains(&step), step,
        );
        let (new_model, out) = step_physics_multi(
            vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );
        model = new_model.into_iter().next().expect("single model");
        last_loss = out.total_scalar;
        if diagnostic_steps.contains(&step) {
            use burn::module::AutodiffModule;
            diagnostics.push(user_problem_l5_diagnostic(step, &out, &model.valid(), &spec, &fd, &device, problem.ansatz(0)));
        }
        if on_step(step, out.total_scalar, out.lr, data.int_norm.len()) {
            break;
        }
    }
    sync_device(&device);
    use burn::module::AutodiffModule;
    (model.valid(), last_loss, diagnostics)
}

/// Trains a [`UserDefinedProblem`] built from `spec` headlessly, printing progress. Returns
/// `true` if the final step's loss is finite (the only generic "did this not blow up"
/// signal available — there's no closed-form convergence target for an arbitrary
/// user-defined geometry, unlike Kirsch's K_t).
pub fn run_headless_user_problem(spec: ProblemSpec) -> bool {
    use pinn_core::problem_spec::{CoordinateEmbeddingSelection, TrainingProcedure};

    // Issue #77 PH4-45: same architecture dispatch as `runner::run_training_user_problem`
    // (GUI), see that function's own comment - kept as two independent call sites (headless
    // has no `Sender<TrainingMsg>`/GUI vis grid to stream into), same precedent as every other
    // headless/GUI pair in this codebase (`run_headless`/`run_training`,
    // `run_headless_pinlug`/`run_training_pinlug`).
    if let TrainingProcedure::SequentialTwoStage { stage_a_steps, stage_b_steps } = spec.architecture.training_procedure {
        assert!(
            spec.architecture.coordinate_embedding == CoordinateEmbeddingSelection::Cartesian,
            "SequentialTwoStage + LogPolar is an untested combination - see runner::run_training_user_problem's identical assertion"
        );
        assert!(AnnularDecompositionProblem::supports(&spec), "SequentialTwoStage requires the same geometry Joint annular decomposition would need");
        let device = BDevice::default();
        let use_hard_constraint = spec.architecture.hard_constraint_ansatz;
        let (_outer, _annulus, total, diagnostics) = run_annular_decomposition_training_sequential(
            spec, device, stage_a_steps, stage_b_steps, use_hard_constraint,
            &[stage_b_steps.saturating_sub(1)],
            |stage, step, loss| {
                if step % 300 == 0 || step + 1 == stage_a_steps.max(stage_b_steps) {
                    println!("  [#77 sequential] {stage} step={step:>6} loss={loss:.6e}");
                }
                false
            },
            &mut |_d, _a, _o| {},
        );
        if let Some(last) = diagnostics.last() {
            println!("  [#77 sequential] done final_total_loss={total:.6e} kt={:.9}", last.kt_derived_fd_vm);
        } else {
            println!("  [#77 sequential] done final_total_loss={total:.6e}");
        }
        return total.is_finite();
    }
    // Issue #77 PH4-45: same `SingleDomain` override as `runner::run_training_user_problem`,
    // see that call site's own comment.
    let force_single_domain = spec.architecture.training_procedure == TrainingProcedure::SingleDomain;
    if force_single_domain {
        assert!(
            spec.architecture.coordinate_embedding == CoordinateEmbeddingSelection::Cartesian,
            "SingleDomain + LogPolar is unsupported - see runner::run_training_user_problem's identical assertion"
        );
    }
    if !force_single_domain && AnnularDecompositionProblem::supports(&spec) {
        let device = BDevice::default();
        let steps = spec.training.max_steps;
        let use_hard_constraint = spec.architecture.hard_constraint_ansatz;
        let use_log_polar = spec.architecture.coordinate_embedding == CoordinateEmbeddingSelection::LogPolar;
        let diagnostic_steps = [steps.saturating_sub(1)];
        let mut final_kt = None;
        let (_annulus, _outer, total) = run_annular_decomposition_training_with_architecture(
            spec, device, use_hard_constraint, use_log_polar, &diagnostic_steps,
            |step, loss, lr, n| {
                if step % (steps / 10).max(1) == 0 || step + 1 == steps {
                    println!("  [#77 annular] step {step:>6} total_loss={loss:.6e} lr={lr:.3e} points={n}");
                }
                false
            },
            &mut |diag, _a, _o| final_kt = Some(diag.kt_derived_fd_vm),
        );
        match final_kt {
            Some(kt) => println!("  [#77 annular] done final_total_loss={total:.6e} kt={kt:.9}"),
            None => println!("  [#77 annular] done final_total_loss={total:.6e}"),
        }
        return total.is_finite();
    }
    let device = BDevice::default();
    let half_w = spec.geometry.half_w;
    let half_h = spec.geometry.half_h;
    let n_holes = spec.geometry.holes.len();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   PINN Structural Stress Solver — User-Defined Problem   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("  Plate    : {:.4}×{:.4} m, {} hole(s)", 2.0 * half_w, 2.0 * half_h, n_holes);
    println!("  Material : E={:.3e} Pa  ν={:.3}", spec.material.e, spec.material.nu);
    println!("  Load     : Px={:.3e} Pa  Py={:.3e} Pa", spec.load.px, spec.load.py);
    println!(
        "  Steps    : {}   Interior: {}  Boundary: {}",
        spec.training.max_steps, spec.training.n_interior, spec.training.n_boundary
    );

    // Issue #61 EPIC P2-08: MANDATORY L0 affine-amplitude gate, run BEFORE any neural
    // optimization begins - the issue's own wording. Cheap (a single scalar parameter, 100
    // gradient-descent steps through the measure-aware variational functional) - aborts
    // loudly rather than proceeding to train a network on top of a functional that can't even
    // recover the textbook exact solution for the simplest possible case.
    {
        let result = crate::verification_ladder::run_affine_amplitude_test(
            &spec.material, spec.load.px, half_w, half_h, spec.geometry.thickness, 100, 1e-4,
        );
        if !result.passed {
            panic!(
                "P2-08 L0 gate FAILED: affine-amplitude test did not recover a_exact - \
                 a_recovered={:.6e} a_exact={:.6e} relative_error={:.3e} (tolerance 1e-4). \
                 Refusing to proceed to neural training on top of a functional that fails the \
                 most basic analytic sanity check.",
                result.a_recovered, result.a_exact, result.relative_error,
            );
        }
        println!("  [diag] P2-08 L0 gate PASSED: affine amplitude relative_error={:.3e}", result.relative_error);
    }

    // Issue #77 PH4-45: same architecture selection as `runner::run_training_user_problem`'s
    // single-domain branch, see that call site's own comment.
    let problem = UserDefinedProblem::new_with_hard_constraint_ansatz(
        spec.clone(), spec.architecture.hard_constraint_ansatz, spec.architecture.hole_bias_fraction,
    );
    crate::problem::validate_loss_terms(&problem);

    let mut config = SolverConfig::default_kirsch();
    config.load = spec.load;

    let net_cfg = ElasticityNetConfig::new()
        // Same real fix as `runner::run_training_user_problem` - see `UserGeometry::
        // n_fourier`'s doc comment.
        .with_input_dim(spec.geometry.net_input_dim())
        .with_hidden_dim(spec.network.hidden_dim)
        .with_n_hidden(spec.network.n_hidden)
        .with_output_dim(5); // mDEM: u, v, sigma_xx, sigma_yy, sigma_xy
    // Issue #62 PH3-11 - same fix as `runner::run_training_user_problem`, see that call site's
    // own comment.
    B::seed(&device, spec.network.model_init_seed);
    let mut model = net_cfg.init(&device);
    let mut optim = DomainOptim {
        weight: WeightOptim::new(config.use_soap_muon),
        bias: make_bias_optim(),
        gate: make_gate_optim(),
    };

    let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
    let mut saw = SawBrdr::with_base(base_weights, 0.95);
    let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);
    let hole_fd = crate::user_problem::hole_fd_config_for_geometry(&fd, &spec.geometry);

    let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
    let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);

    let sampling = problem.sampling_strategy(0);
    let placeholder_geom = pinn_core::geometry::GeometryConfig::kirsch_plate_inches(); // ignored by UserSamplingStrategy

    // Issue #61 P2-06: one-time startup diagnostic (no effect on training - `sample_interior`
    // is deterministic, re-seeded identically every call, so this just previews the same
    // point set the training loop's own first-step call will reproduce). Reports how many of
    // this run's real interior collocation points have a fully valid 5-point FD stencil vs.
    // need `UserSamplingStrategy`'s own margin-based fallback - makes that margin's real,
    // per-run effect a visible number instead of an invisible property of the sampling
    // process. See `stencil_quality_report`'s own doc comment.
    {
        let preview_pts = sampling.sample_interior(&placeholder_geom, spec.training.n_interior);
        let fd_h = spec.training.fd_h as f64;
        let report = crate::user_problem::stencil_quality_report(&spec.geometry, &preview_pts, fd_h, fd_h);
        println!(
            "  [diag] stencil quality: {}/{} fully valid, {} fallback-needed, {} invalid-center",
            report.fully_valid, report.total, report.fallback_needed, report.invalid_center,
        );
    }

    // Issue #78 investigation: real per-step Kt evidence, not just the final value. Every
    // prior headless run only measured Kt ONCE, after training completed - meaning "total_loss
    // goes flat around step 300 but training continues to step 3000" could not be distinguished
    // from "Kt itself is still improving while total_loss looks flat" vs "training has
    // genuinely stalled" without a NEW run. Computed once here (static for the whole run, same
    // values `loss_terms()`/the final diagnostic block use) so the periodic print below is
    // cheap (a real small forward pass per checkpoint, at the same 10-checkpoint cadence the
    // loss line already uses - not the per-step hot loop).
    let live_kt_ansatz = problem.ansatz(0);
    let live_kt_affine = (crate::user_problem::decomposition_applicable(&spec) || problem.hard_constraint_active())
        .then_some((spec.load.px, spec.load.py));
    let live_kt_margin = crate::user_problem::ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
    let live_kt_nominal_stress = spec.load.px.abs().max(spec.load.py.abs());
    let live_kt_free_holes: Vec<usize> = spec.geometry.holes.iter().enumerate()
        .filter(|(_, h)| h.bc == pinn_core::user_geometry::HoleBc::Free)
        .map(|(i, _)| i).collect();

    let mut last_total = f32::NAN;
    for step in 0..spec.training.max_steps {
        // Issue #73: shared with `runner::run_user_problem_training_from` - see
        // `resample_plate_step_data`/`plate_multi_step_ctx`'s own doc comments (user_problem.rs)
        // for why this must stay a real per-step call, not something cached across steps.
        let data = resample_plate_step_data(
            sampling, &placeholder_geom, &spec.load,
            spec.training.n_interior, spec.training.n_boundary, half_w, half_h,
        );
        // Issue #77 PH4-41 fix (finding 2) - same fix as `run_user_problem_training_with_
        // diagnostics`, see that call site's own comment. `hole_bias_fraction()<=0.0` (every
        // pre-#77 spec) makes this a no-op.
        let weights = crate::user_problem::hole_bias_quadrature_weights(
            &data.int_norm, &spec.geometry, problem.hole_bias_fraction(),
        );
        problem.set_interior_weights(Some(weights));
        let ctx = plate_multi_step_ctx(
            &config, &problem, &fd, &hole_fd, &data, u_ref, ref_energy, ref_stress2,
            spec.geometry.n_fourier(),
            spec.geometry.coordinate_embedding(),
            // PH4-06: one final live gradient ledger is cheap enough for headless controlled
            // ladders and prevents a falling total loss from being mistaken for physical
            // convergence. Earlier steps keep the normal no-extra-backward-pass path.
            step + 1 == spec.training.max_steps,
            step,
        );

        let (new_model, out) = step_physics_multi(
            vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, &device,
            0, 1.0, 1.0,
        );
        model = new_model.into_iter().next().unwrap();
        last_total = out.total_scalar;

        if step + 1 == spec.training.max_steps {
            if let (Some(raw), Some(weights), Some(norms)) = (
                out.raw_scalar_by_name.as_ref(), out.lam_by_name.as_ref(), out.term_grad_norms.as_ref(),
            ) {
                let mut names: Vec<&&str> = raw.keys().collect();
                names.sort();
                for name in names {
                    println!(
                        "  [diag] PH4 term: {name} raw={:.6e} lambda={:.6e} grad_norm={:.6e}",
                        raw[name], weights.get(name).copied().unwrap_or(f64::NAN),
                        norms.get(name).copied().unwrap_or(f32::NAN),
                    );
                }
            }
        }

        if step % (spec.training.max_steps / 10).max(1) == 0 || step + 1 == spec.training.max_steps {
            println!("  step {step:>6}   total_loss={:.6e}   lr={:.3e}   grad_norm={:.6e}", out.total_scalar, out.lr, out.grad_norm.unwrap_or(f32::NAN));
            if matches!(spec.formulation, pinn_core::problem_spec::FormulationSelection::Variational) {
                let raw = out.raw_scalar_by_name.as_ref()
                    .and_then(|values| values.get("physical_potential")).copied()
                    .unwrap_or(f32::NAN);
                let physical_weight = out.lam_by_name.as_ref()
                    .and_then(|weights| weights.get("physical_potential")).copied()
                    .unwrap_or(f64::NAN);
                let translation = out.raw_scalar_by_name.as_ref()
                    .and_then(|values| values.get("translation_gauge")).copied()
                    .unwrap_or(f32::NAN);
                println!(
                    "  [diag] PH4 trend: normalized_Pi={raw:.6e} physical_weight={physical_weight:.6e} translation_gauge={translation:.6e}"
                );
            }
            // Issue #78 investigation: real per-step Kt, not just the final one - see this
            // block's own setup comment above `for step in 0..` for why this exists. A real
            // small forward pass per Free hole, only at this same 10-checkpoint cadence.
            if !live_kt_free_holes.is_empty() {
                use burn::module::AutodiffModule;
                let model_val: crate::network::ElasticityNet<crate::training_core::BInner> = model.valid();
                let kts: Vec<String> = live_kt_free_holes.iter().map(|&i| {
                    let hole = &spec.geometry.holes[i];
                    let profile = crate::user_problem::probe_hole_boundary_profile_derived(
                        &model_val, &spec.geometry, hole, 36, &fd, u_ref, spec.load.px, &spec.material,
                        live_kt_margin, &device, live_kt_ansatz, live_kt_affine,
                    );
                    let kt = crate::user_problem::stress_concentration_from_profile(&profile, live_kt_nominal_stress).kt;
                    format!("hole{i}={kt:.4}")
                }).collect();
                println!("  [diag] live Kt (Free holes): {}", kts.join(" "));
            }
        }
    }
    sync_device(&device);

    // Qualitative "not a trivial collapse" readout — no closed-form oracle exists for an
    // arbitrary user-defined geometry, unlike toy_beam's beam or Kirsch's K_t, so this
    // checks max|displacement| over a handful of interior points is non-negligible rather
    // than asserting a specific value. IdentityAnsatz means raw network columns 0,1 (u,v)
    // need only the same u_ref physical-unit scaling `compute_domain_forwards` applies —
    // no ansatz dx/dy factor to replicate (always (1.0, 1.0)).
    //
    // Issue #77 PH4-45: a real, pre-existing bug caught by this session's own headless smoke
    // test of a holed geometry - this block used to call `model.forward(eval_t)` directly on a
    // bare 3-column `[xn, yn, 0.0]` tensor, bypassing the embedding transform entirely. That
    // panicked on any single-hole geometry (`net_cfg` above builds the model with
    // `net_input_dim()` = 10 for `SingleHoleChart`, not 3) - it happened to go unexercised
    // until now because every real PH4-24..44 run trained through a different function
    // (`run_user_problem_training_with_diagnostics`/`run_annular_decomposition_training*`),
    // never this plain headless path against a REAL hole. Fixed by routing through
    // `fwd_embedded`, the same embedding-aware forward every other real call site uses.
    let eval_pts = sampling.sample_interior(&placeholder_geom, 32);
    let eval_norm: Vec<f32> = eval_pts.iter()
        .flat_map(|&[x, y]| { let [nx, ny] = normalize_point(x, y, half_w, half_h); [nx, ny, 0.0f32] })
        .collect();
    let n_eval = eval_pts.len();
    let eval_t = burn::tensor::Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(eval_norm, vec![n_eval, 3]), &device,
    );
    let raw = crate::network::fwd_embedded::<B>(&model, eval_t, spec.geometry.coordinate_embedding(), &device);
    let max_abs_disp: f32 = raw.slice([0..n_eval, 0..2])
        .mul_scalar(u_ref as f64)
        .abs()
        .reshape([n_eval * 2])
        .max()
        .into_scalar();

    println!("  Done! final total_loss={last_total:.6e}  max|displacement|~{max_abs_disp:.3e} m");
    if max_abs_disp < 1e-12 {
        println!("  [!] max|displacement| is suspiciously small — possible trivial-solution collapse.");
    } else {
        println!("  [ok] displacement is non-trivial.");
    }

    // Real, permanent diagnostic readout (same probes the GUI's vis-cadence block uses, so
    // these numbers are directly comparable to what the app shows): PDE residual RMS/max and
    // per-hole Kt, letting a headless CLI run self-report solution quality without needing
    // the GUI.
    {
        use burn::module::AutodiffModule;
        let model_val: crate::network::ElasticityNet<crate::training_core::BInner> = model.valid();
        let diag_int_norm: Vec<[f32; 2]> = sampling.sample_interior(&placeholder_geom, 512).iter()
            .map(|&[x, y]| normalize_point(x, y, half_w, half_h)).collect();
        // Issue #77 PH4-45: same field-reconstruction fix as the GUI's own vis-cadence block -
        // `problem.ansatz(0)` is this model's own real training-time ansatz, and the affine
        // gate mirrors `UserDefinedProblem::loss_terms()`'s own generalized (issue #78)
        // `decomposed || hard_constraint_active()` condition - see that gate's own doc comment.
        let affine = (crate::user_problem::decomposition_applicable(&spec) || problem.hard_constraint_active())
            .then_some((spec.load.px, spec.load.py));
        let vis = crate::user_problem::evaluate_user_vis_grid(
            &model_val, &spec.geometry, [96, 96], u_ref, spec.load.px, &spec.material, &fd, &diag_int_norm, &device,
            problem.ansatz(0), affine,
        );
        let pde_vals: Vec<f32> = vis.pde_residual.iter().copied().filter(|v| v.is_finite()).collect();
        let (pde_rms, pde_max) = crate::training_core::residual_stats(&pde_vals);
        // "Constitutive residual" - not "PDE residual" - see `EquilibriumTerm`'s doc comment
        // (user_problem.rs) for why that distinction matters.
        println!("  [diag] constitutive residual RMS={pde_rms:.4e}  max={pde_max:.4e} Pa");

        // Issue #61 EPIC P2-09: predicted-vs-prescribed load transfer + generic trivial-
        // solution warning - directly motivated by the real Debug_runs evidence (a collapsed
        // solution with nonzero-but-far-too-small stress, which a bare displacement check
        // alone would have missed).
        let load_transfer = crate::user_problem::probe_load_transfer(&model_val, &spec, &device, problem.ansatz(0), affine);
        println!(
            "  [diag] load transfer ratio={:.4}  predicted=({:.3e},{:.3e}) N  prescribed=({:.3e},{:.3e}) N",
            load_transfer.load_transfer_ratio, load_transfer.predicted_load_x, load_transfer.predicted_load_y,
            load_transfer.prescribed_load_x, load_transfer.prescribed_load_y,
        );
        if load_transfer.trivial_solution_warning {
            println!("  [!] P2-09 trivial-solution warning: only {:.1}% of the prescribed load is being transferred - likely a collapsed/trivial solution.", load_transfer.load_transfer_ratio * 100.0);
        }

        // Issue #63 sub-issue #69 (PH4-15 hole-side): a real, independent field-validation
        // signal that works for ANY geometry, hole or no-hole, with no closed-form reference
        // needed - for a valid elastic solution, the net resultant traction integrated around
        // the WHOLE closed outer boundary must be zero (far-field loading self-cancels around a
        // closed rectangle, hole(s) or not). `probe_reaction_force` already existed (tested,
        // never wired into any printed output) - this closes that gap, giving hole geometries
        // the same "independently-computed, not training-loss-derived" validation no-hole
        // geometries already get from `validate_no_hole_fields`.
        let reaction_force = crate::user_problem::probe_reaction_force(&model_val, &spec, &device, problem.ansatz(0), affine);
        println!(
            "  [diag] reaction force (closed-boundary equilibrium): net=({:.3e},{:.3e}) N  reference={:.3e} N  equilibrium_error={:.4e}",
            reaction_force.net_fx, reaction_force.net_fy, reaction_force.reference_force, reaction_force.equilibrium_error,
        );

        let nominal_stress = spec.load.px.abs().max(spec.load.py.abs());
        // Derived-stress-at-margin, not direct σ at the exact boundary - see
        // `probe_hole_boundary_profile_derived`'s doc comment.
        let hole_margin = crate::user_problem::ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
        // Issue #78 real bug fix: this comment used to claim "this headless path always trains
        // a plain UserDefinedProblem::new(spec) (IdentityAnsatz)" - true when PH4-41 wrote it,
        // FALSE since PH4-45 wired `spec.architecture.hard_constraint_ansatz` through to
        // `new_with_hard_constraint_ansatz` above (this function's own `problem` construction).
        // This diagnostic loop kept reconstructing the field with a hardcoded `IdentityAnsatz`
        // regardless - silently wrong for ANY hard-constraint-ansatz spec, undetected until this
        // session's own real `--headless` verification run on an off-center multi-hole geometry
        // measured Kt≈0.46-0.59 against a real FEM ground truth of ≈2.9-3.1 and traced the gap
        // to this exact call site reconstructing "raw network output" as if it WERE the physical
        // field, discarding both the multiplicative envelope and the additive closed-form
        // correction the model was actually trained with. `evaluate_user_vis_grid` (the sibling
        // call directly above) was already correctly fixed to `problem.ansatz(0)` during
        // PH4-45 - this loop was the one call site that migration missed. Now uses the model's
        // own real training-time ansatz and the same generalized affine gate as `loss_terms()`.
        let affine = (crate::user_problem::decomposition_applicable(&spec) || problem.hard_constraint_active())
            .then_some((spec.load.px, spec.load.py));
        let ansatz = problem.ansatz(0);
        for (i, hole) in spec.geometry.holes.iter().enumerate() {
            let profile = crate::user_problem::probe_hole_boundary_profile_derived(
                &model_val, &spec.geometry, hole, 72, &fd, u_ref, spec.load.px, &spec.material, hole_margin, &device,
                ansatz, affine,
            );
            let sc = crate::user_problem::stress_concentration_from_profile(&profile, nominal_stress);
            println!("  [diag] hole {i}: max_von_mises={:.4e} Pa  nominal={:.4e} Pa  Kt={:.4}", sc.max_von_mises, sc.nominal_stress, sc.kt);

            // Issue #61 EPIC P2-10: angular/radial convergence support - is this Kt value
            // actually converged, or still drifting with resolution/margin?
            let convergence = crate::user_problem::kt_convergence_check(
                &model_val, &spec.geometry, hole, 72, &fd, u_ref, spec.load.px, &spec.material,
                hole_margin, nominal_stress, 0.1, &device, ansatz, affine,
            );
            if convergence.converged {
                println!("  [diag] hole {i}: Kt convergence OK (angular Δ={:.3}, radial Δ={:.3})", convergence.angular_relative_change, convergence.radial_relative_change);
            } else {
                println!("  [!] hole {i}: Kt NOT converged (angular Δ={:.3}, radial Δ={:.3}) - Kt value may not be trustworthy yet", convergence.angular_relative_change, convergence.radial_relative_change);
            }
        }

        // Issue #61 EPIC P2-08: L4 no-hole neural training health gate. Only meaningful for a
        // no-hole geometry itself (the "companion no-hole run" a hole run would need per the
        // epic's own "MANDATORY no-hole gate before Kt/hole results accepted" - the actual
        // cross-run enforcement tying a hole run's Kt acceptance to a verified no-hole PASS is
        // P2-14's job, which owns the full benchmark protocol + P2-13's provenance).
        if n_holes == 0 {
            let energy_balance = crate::user_problem::probe_energy_balance(&model_val, &spec, &device);
            // Phase 4: print the same physical objective ledger checkpoint/GUI reports
            // persist. `EnergyBalance.external_work` is half work for U=W/2 validation;
            // snapshot exposes full prescribed W_ext and Pi=U-W_ext separately.
            let objective = crate::provenance::mathematical_objective_snapshot(
                &spec, Some(&energy_balance), None,
            );
            println!(
                "  [diag] PH4 objective: U={:.6e} J  W_ext={:.6e} J  Pi={:.6e} J  normalized_Pi={:.6e}",
                objective.physical_u.unwrap_or(f64::NAN),
                objective.physical_w_ext.unwrap_or(f64::NAN),
                objective.physical_pi.unwrap_or(f64::NAN),
                objective.normalized_pi.unwrap_or(f64::NAN),
            );
            let check = crate::verification_ladder::no_hole_health_check(&energy_balance, max_abs_disp as f64);
            if check.passed {
                println!("  [diag] P2-08 L4 no-hole health check PASSED (energy_balance_error={:.4e})", check.energy_balance_error);
            } else {
                println!(
                    "  [!] P2-08 L4 no-hole health check FAILED: {} (energy_balance_error={:.4e}, max_abs_displacement={:.4e})",
                    check.failure_reason.unwrap_or("unknown"), check.energy_balance_error, check.max_abs_displacement,
                );
            }

            // Issue #61 EPIC P2-14: the FINAL, hard-numeric-threshold no-hole gate (as opposed
            // to P2-08's looser sanity check above). This is the gate a hole run's Kt would
            // need a PASSING companion run of to be accepted - see the hole-side message below.
            let benchmark = crate::user_problem::run_no_hole_benchmark(&model_val, &spec, &device);
            if benchmark.passed {
                println!("  [diag] P2-14 no-hole BENCHMARK PASSED (all hard thresholds met)");
            } else {
                println!("  [!] P2-14 no-hole BENCHMARK FAILED: {:?}", benchmark.failures);
            }
            println!(
                "  [diag] P2-14 no-hole benchmark: sigma_xx_err={:.4}  sigma_yy/ref={:.4}  sigma_xy/ref={:.4}  traction_rms/ref={:.4}  load_transfer={:.4}",
                benchmark.sigma_xx_relative_error, benchmark.sigma_yy_over_ref, benchmark.sigma_xy_over_ref,
                benchmark.traction_rms_over_ref, benchmark.load_transfer_ratio,
            );

            // Issue #64 / PH4-15: independent field validation, wired into the headless path
            // now that a corrected-Variational L4 pass actually exists to validate. Reuses the
            // `vis` grid already computed above (no extra forward pass) — a SEPARATE, regular
            // 96x96 grid, not the training collocation points, so this cannot pass merely
            // because the network overfit the points it was trained on. Checks published
            // displacement/strain/constitutive-stress fields against the exact affine no-hole
            // solution directly, and confirms no residual rigid-body mode is hiding in the
            // published fields.
            let field_check = crate::user_problem::validate_no_hole_fields(&vis, &spec);
            let field_ok = field_check.sigma_xx_relative_error < crate::user_problem::SIGMA_XX_RELATIVE_ERROR_MAX
                && field_check.sigma_yy_over_ref < crate::user_problem::SIGMA_YY_OVER_REF_MAX
                && field_check.sigma_xy_over_ref < crate::user_problem::SIGMA_XY_OVER_REF_MAX;
            if field_ok {
                println!("  [diag] P2-14 independent field validation PASSED (separate grid, not training points)");
            } else {
                println!("  [!] P2-14 independent field validation FAILED (separate grid, not training points)");
            }
            println!(
                "  [diag] independent field validation: sigma_xx_err={:.4}  sigma_yy/ref={:.4}  sigma_xy/ref={:.4}  u_l2={:.4e}  v_l2={:.4e}  rigid_translation={:.4e}  rigid_rotation={:.4e}",
                field_check.sigma_xx_relative_error, field_check.sigma_yy_over_ref, field_check.sigma_xy_over_ref,
                field_check.u_l2, field_check.v_l2, field_check.rigid_translation_residual, field_check.rigid_rotation_residual,
            );
        } else {
            // Issue #61 EPIC P2-14: a hole run's Kt is only accepted after a companion no-hole
            // run PASSES the benchmark above - this single CLI invocation trains only the holed
            // geometry, so no such companion result is available here. Honestly reported as
            // unavailable (matching P2-13's own discipline), NOT silently skipped or assumed.
            println!("  [!] P2-14 hole benchmark gate: NOT EVALUATED - run the companion no-hole config and check its P2-14 benchmark PASSES before accepting this run's Kt value(s).");
        }
    }

    last_total.is_finite()
}
