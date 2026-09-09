//! Minimal headless runner for a [`UserDefinedProblem`] (`user_problem.rs`) — the third,
//! generic entry point alongside `headless::run_headless`/`run_headless_pinlug`. Deliberately
//! v1-simple: no `PinnDecisionMaker`/`ConvergenceTracker`/L-BFGS-Converge/warm-restart
//! cascade (all confirmed optional extras layered around `step_physics_multi`, not required
//! by it) — a plain constant/scheduled-LR AdamW loop, matching `toy_beam`'s own "prove the
//! formulation converges before adding curriculum machinery" scope discipline.

use std::collections::HashMap;

use pinn_core::messages::SolverConfig;
use pinn_core::problem_spec::ProblemSpec;

use crate::{
    fd_stencil::FdConfig,
    lr_schedule::LrSchedule,
    network::ElasticityNetConfig,
    optim::{make_bias_optim, make_gate_optim, WeightOptim},
    problem::{BoundaryValueProblem, DomainOptim, DomainStepCtx, DomainStepData, MultiStepCtx, PointSetData},
    saw_brdr::SawBrdr,
    training_core::{step_physics_multi, sync_device, BDevice, B},
    user_problem::{UserDefinedProblem, USER_DOMAIN},
};

fn normalize_point(x: f64, y: f64, half_w: f64, half_h: f64) -> [f32; 2] {
    [(x / half_w) as f32, (y / half_h) as f32]
}

fn build_pointset(pts: &[pinn_core::loading::BoundaryPoint], half_w: f64, half_h: f64) -> PointSetData {
    PointSetData {
        norm: pts.iter().map(|p| normalize_point(p.x, p.y, half_w, half_h)).collect(),
        nx: pts.iter().map(|p| p.nx as f32).collect(),
        ny: pts.iter().map(|p| p.ny as f32).collect(),
        tx: pts.iter().map(|p| p.tx as f32).collect(),
        ty: pts.iter().map(|p| p.ty as f32).collect(),
    }
}

/// Trains a [`UserDefinedProblem`] built from `spec` headlessly, printing progress. Returns
/// `true` if the final step's loss is finite (the only generic "did this not blow up"
/// signal available — there's no closed-form convergence target for an arbitrary
/// user-defined geometry, unlike Kirsch's K_t).
pub fn run_headless_user_problem(spec: ProblemSpec) -> bool {
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

    let stress_ref = spec.load.px.abs().max(spec.load.py.abs()).max(1.0);
    let u_ref = ((stress_ref / spec.material.e) * half_w) as f32;
    let ref_energy = (0.5 * stress_ref * stress_ref / spec.material.e).max(1.0) as f32;
    let ref_stress2 = (stress_ref * stress_ref).max(1.0) as f32;

    let sampling = problem.sampling_strategy(0);
    let placeholder_geom = pinn_core::geometry::GeometryConfig::kirsch_plate_inches(); // ignored by UserSamplingStrategy

    let mut last_total = f32::NAN;
    for step in 0..spec.training.max_steps {
        let int_pts = sampling.sample_interior(&placeholder_geom, spec.training.n_interior);
        let bnd_pts = sampling.sample_boundary(&placeholder_geom, &spec.load, spec.training.n_boundary);

        let int_norm: Vec<[f32; 2]> = int_pts.iter().map(|&[x, y]| normalize_point(x, y, half_w, half_h)).collect();

        let mut named = HashMap::with_capacity(1 + n_holes);
        named.insert("outer_boundary", build_pointset(&bnd_pts, half_w, half_h));
        for set in sampling.named_point_sets(&[]) {
            named.insert(set.name, build_pointset(&set.points, half_w, half_h));
        }

        let data = DomainStepData { id: USER_DOMAIN, int_norm, extra_ring_norm: Vec::new(), named };
        let ctx = MultiStepCtx {
            config: &config,
            problem: &problem,
            fd: &fd,
            k: 1.0, // IdentityAnsatz ignores k entirely — value is inert
            domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
            // Same real fix as `runner::run_user_problem_training_from`'s per-step `MultiStepCtx`
            // (this headless CLI path mirrors that GUI path's training loop) - see that call
            // site's doc comment for the full root-cause explanation.
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: f64::MAX,
            dynamic_lam_non_tension_cap: f64::MAX,
            // Same real fix as `runner::run_user_problem_training_from`'s per-step
            // `MultiStepCtx` - see that call site's doc comment for the full root-cause
            // explanation of why this matches `dynamic_lam_h_cap` at 50.0.
            constitutive_consistency_weight: 50.0,
            // Same real fix as `runner::run_user_problem_training_from`'s per-step
            // `MultiStepCtx` - see `UserGeometry::n_fourier`'s doc comment for the full
            // root-cause story. `net_cfg`'s `input_dim` (this function's model-construction
            // site) MUST use the same value.
            n_fourier: spec.geometry.n_fourier(),
            phase2_active: true,
            step,
        };

        let (new_model, out) = step_physics_multi(
            vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, &device,
            0, 1.0, 1.0,
        );
        model = new_model.into_iter().next().unwrap();
        last_total = out.total_scalar;

        if step % (spec.training.max_steps / 10).max(1) == 0 || step + 1 == spec.training.max_steps {
            println!("  step {step:>6}   total_loss={:.6e}   lr={:.3e}", out.total_scalar, out.lr);
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
        println!("  [diag] PDE residual RMS={pde_rms:.4e}  max={pde_max:.4e} Pa");
        let nominal_stress = spec.load.px.abs().max(spec.load.py.abs());
        for (i, hole) in spec.geometry.holes.iter().enumerate() {
            let profile = crate::user_problem::probe_hole_boundary_profile(
                &model_val, &spec.geometry, hole, 72, &fd, u_ref, spec.load.px, &device,
            );
            let sc = crate::user_problem::stress_concentration_from_profile(&profile, nominal_stress);
            println!("  [diag] hole {i}: max_von_mises={:.4e} Pa  nominal={:.4e} Pa  Kt={:.4}", sc.max_von_mises, sc.nominal_stress, sc.kt);
        }
    }

    last_total.is_finite()
}
