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
        UserDefinedProblem,
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

fn annular_l5_diagnostic(
    step: usize,
    out: &StepOutput,
    annulus_lr: f64,
    outer_lr: f64,
    model: &crate::network::ElasticityNet<crate::training_core::BInner>,
    spec: &ProblemSpec,
    fd: &FdConfig,
    device: &BDevice,
) -> AnnularL5Diagnostic {
    let scales = crate::training_core::compute_reference_scales_for_plate(spec);
    let hole = &spec.geometry.holes[0];
    let margin = crate::user_problem::ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
    let profile = crate::user_problem::probe_hole_boundary_profile_derived(
        model, &spec.geometry, hole, 144, fd, scales.u_ref, scales.stress_ref,
        &spec.material, margin, device,
    );
    let kt = crate::user_problem::stress_concentration_from_profile(&profile, spec.load.px.abs()).kt;
    let stress = crate::user_problem::probe_hole_stress_diagnostic(
        model, &spec.geometry, hole, 144, fd, scales.u_ref, scales.stress_ref,
        &spec.material, margin, device,
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
fn run_annular_decomposition_training_inner(
    spec: ProblemSpec,
    device: BDevice,
    mut on_step: impl FnMut(usize, f32, f64, usize) -> bool,
    diagnostic_steps: &[usize],
    diagnostics: &mut Vec<AnnularL5Diagnostic>,
) -> (crate::network::ElasticityNet<B>, crate::network::ElasticityNet<B>, f32) {
    let problem = AnnularDecompositionProblem::new(spec.clone());
    crate::problem::validate_loss_terms(&problem);
    let mut config = SolverConfig::default_kirsch();
    config.load = spec.load;
    let chart_cfg = ElasticityNetConfig::new()
        .with_input_dim(spec.geometry.coordinate_embedding().input_dim())
        .with_hidden_dim(spec.network.hidden_dim)
        .with_n_hidden(spec.network.n_hidden)
        .with_output_dim(5);
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
    let base_weights = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
    let mut saw = SawBrdr::with_base(base_weights, 0.95);
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
        let mut ctx = plate_multi_domain_step_ctx(
            &config, &problem, &fd, &hole_fd,
            &annulus_data, &outer_data,
            scales.u_ref, scales.ref_energy, scales.ref_stress2, spec.geometry.n_fourier(),
            spec.geometry.coordinate_embedding(), diagnostic_steps.contains(&step), step,
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
        if diagnostic_steps.contains(&step) {
            use burn::module::AutodiffModule;
            diagnostics.push(annular_l5_diagnostic(step, &out, lr_annulus, lr_outer, &models[0].valid(), &spec, &fd, &device));
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

/// Train #77's bonded annular/global model pair. Kept public so GUI production dispatch and
/// L5 use identical optimizer, sampling, and loss wiring rather than maintaining two loops.
pub fn run_annular_decomposition_training(
    spec: ProblemSpec,
    device: BDevice,
    on_step: impl FnMut(usize, f32, f64, usize) -> bool,
) -> (crate::network::ElasticityNet<B>, crate::network::ElasticityNet<B>, f32) {
    run_annular_decomposition_training_inner(spec, device, on_step, &[], &mut Vec::new())
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
        spec, device, on_step, diagnostic_steps, &mut diagnostics,
    );
    (annulus, outer, loss, diagnostics)
}

/// Minimal deterministic single-model loop for benchmark companions. Production UI/headless
/// retain their richer reporting loops; this avoids copying that training math into L5 tests.
pub(crate) fn train_single_user_problem_for_benchmark(
    spec: ProblemSpec,
    device: &BDevice,
) -> crate::network::ElasticityNet<crate::training_core::BInner> {
    assert!(!AnnularDecompositionProblem::supports(&spec), "benchmark helper is single-model only");
    let problem = UserDefinedProblem::new(spec.clone());
    crate::problem::validate_loss_terms(&problem);
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
    for step in 0..spec.training.max_steps {
        let data = resample_plate_step_data(
            problem.sampling_strategy(0), &placeholder, &spec.load, spec.training.n_interior,
            spec.training.n_boundary, spec.geometry.half_w, spec.geometry.half_h,
        );
        let ctx = plate_multi_step_ctx(
            &config, &problem, &fd, &hole_fd, &data, scales.u_ref, scales.ref_energy, scales.ref_stress2,
            spec.geometry.n_fourier(), spec.geometry.coordinate_embedding(), false, step,
        );
        let (new_model, _) = step_physics_multi(
            vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, device, 0, 1.0, 1.0,
        );
        model = new_model.into_iter().next().expect("single model");
    }
    use burn::module::AutodiffModule;
    model.valid()
}

/// Trains a [`UserDefinedProblem`] built from `spec` headlessly, printing progress. Returns
/// `true` if the final step's loss is finite (the only generic "did this not blow up"
/// signal available — there's no closed-form convergence target for an arbitrary
/// user-defined geometry, unlike Kirsch's K_t).
pub fn run_headless_user_problem(spec: ProblemSpec) -> bool {
    if AnnularDecompositionProblem::supports(&spec) {
        let device = BDevice::default();
        let steps = spec.training.max_steps;
        let (_annulus, _outer, total) = run_annular_decomposition_training(spec, device, |step, loss, lr, n| {
            if step % (steps / 10).max(1) == 0 || step + 1 == steps {
                println!("  [#77 annular] step {step:>6} total_loss={loss:.6e} lr={lr:.3e} points={n}");
            }
            false
        });
        println!("  [#77 annular] done final_total_loss={total:.6e}");
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

    let problem = UserDefinedProblem::new(spec.clone());
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

    let mut last_total = f32::NAN;
    for step in 0..spec.training.max_steps {
        // Issue #73: shared with `runner::run_user_problem_training_from` - see
        // `resample_plate_step_data`/`plate_multi_step_ctx`'s own doc comments (user_problem.rs)
        // for why this must stay a real per-step call, not something cached across steps.
        let data = resample_plate_step_data(
            sampling, &placeholder_geom, &spec.load,
            spec.training.n_interior, spec.training.n_boundary, half_w, half_h,
        );
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
            println!("  step {step:>6}   total_loss={:.6e}   lr={:.3e}", out.total_scalar, out.lr);
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
        }
    }
    sync_device(&device);

    // Qualitative "not a trivial collapse" readout — no closed-form oracle exists for an
    // arbitrary user-defined geometry, unlike toy_beam's beam or Kirsch's K_t, so this
    // checks max|displacement| over a handful of interior points is non-negligible rather
    // than asserting a specific value. IdentityAnsatz means raw network columns 0,1 (u,v)
    // need only the same u_ref physical-unit scaling `compute_domain_forwards` applies —
    // no ansatz dx/dy factor to replicate (always (1.0, 1.0)).
    let eval_pts = sampling.sample_interior(&placeholder_geom, 32);
    let eval_norm: Vec<f32> = eval_pts.iter()
        .flat_map(|&[x, y]| { let [nx, ny] = normalize_point(x, y, half_w, half_h); [nx, ny, 0.0f32] })
        .collect();
    let n_eval = eval_pts.len();
    let eval_t = burn::tensor::Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(eval_norm, vec![n_eval, 3]), &device,
    );
    let raw = model.forward(eval_t);
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
        let vis = crate::user_problem::evaluate_user_vis_grid(
            &model_val, &spec.geometry, [96, 96], u_ref, spec.load.px, &spec.material, &fd, &diag_int_norm, &device,
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
        let load_transfer = crate::user_problem::probe_load_transfer(&model_val, &spec, &device);
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
        let reaction_force = crate::user_problem::probe_reaction_force(&model_val, &spec, &device);
        println!(
            "  [diag] reaction force (closed-boundary equilibrium): net=({:.3e},{:.3e}) N  reference={:.3e} N  equilibrium_error={:.4e}",
            reaction_force.net_fx, reaction_force.net_fy, reaction_force.reference_force, reaction_force.equilibrium_error,
        );

        let nominal_stress = spec.load.px.abs().max(spec.load.py.abs());
        // Derived-stress-at-margin, not direct σ at the exact boundary - see
        // `probe_hole_boundary_profile_derived`'s doc comment.
        let hole_margin = crate::user_problem::ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
        for (i, hole) in spec.geometry.holes.iter().enumerate() {
            let profile = crate::user_problem::probe_hole_boundary_profile_derived(
                &model_val, &spec.geometry, hole, 72, &fd, u_ref, spec.load.px, &spec.material, hole_margin, &device,
            );
            let sc = crate::user_problem::stress_concentration_from_profile(&profile, nominal_stress);
            println!("  [diag] hole {i}: max_von_mises={:.4e} Pa  nominal={:.4e} Pa  Kt={:.4}", sc.max_von_mises, sc.nominal_stress, sc.kt);

            // Issue #61 EPIC P2-10: angular/radial convergence support - is this Kt value
            // actually converged, or still drifting with resolution/margin?
            let convergence = crate::user_problem::kt_convergence_check(
                &model_val, &spec.geometry, hole, 72, &fd, u_ref, spec.load.px, &spec.material,
                hole_margin, nominal_stress, 0.1, &device,
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
