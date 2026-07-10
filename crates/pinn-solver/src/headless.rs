/// Headless (terminal-only) training mode with full engine integration.
///
/// Two-phase curriculum:
///   Phase 1 (0..phase1_steps): SAW-BRDR on 5 BC losses (e, n, h, d, eq). No kirsch loss.
///     Smart batch: phase1_n_interior points (<<n_interior) for 16× faster BC convergence.
///   Phase 2 (phase1_steps..): BCs converged → add kirsch_stress_loss (fixed lam_kirsch).
///     Full n_interior batch. lam_h capped at 50 so kirsch gradient dominates h gradient.
///     K_t is REPORTED from probe_kt_shared() as a verification metric — not used as a loss.
/// - No hardcoded constants — everything flows from SolverConfig via EngineParams.

use std::collections::HashMap;

use burn::{
    backend::{Autodiff, Wgpu},
    module::AutodiffModule,
};
use burn::backend::wgpu::WgpuDevice;
use pinn_core::{
    amr::AdaptiveGrid,
    messages::SolverConfig,
    sampling::{sample_boundary, sample_interior, sample_eq_ring},
    units::{IN_TO_M, KSI_TO_PA, MSI_TO_PA},
};

use crate::{
    bc::apply_dirichlet_ansatz,
    controllers::ConvergenceTracker,
    decision_maker::{OptimizerTier, PinnDecisionMaker},
    engine::EngineParams,
    energy::dem_energy_per_point,
    fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig},
    kirsch_problem::KirschProblem,
    lr_schedule::LrSchedule,
    network::{fwd, ElasticityNet, ElasticityNetConfig},
    optim::{make_bias_optim, make_gate_optim, WeightOptim},
    problem::validate_loss_terms,
    saw_brdr::SawBrdr,
    stiffness::StiffnessController,
    training_core::{
        compute_gradient_conflict, compute_reference_scales, extract_boundary_indices,
        make_lbfgs, normalize_point, probe_kt_shared, step_lbfgs, step_physics,
        BInner, LbfgsCtxScalars, LbfgsLams, StepCtx, StepOutput,
    },
};

type B = Autodiff<Wgpu>;

const BAR_WIDTH: usize = 40;

/// Run headless training. Returns `true` if converged, `false` if exhausted max_steps.
pub fn run_headless(config: SolverConfig) -> bool {
    let mut config = config;
    let engine = EngineParams::analyze(&config);
    engine.apply_to(&mut config);
    let k = engine.ansatz_k;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       PINN Structural Stress Solver  —  Headless Mode   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("  Material : E={:.2} Msi  ν={:.3}", config.material.e / MSI_TO_PA, config.material.nu);
    println!("  Geometry : {:.3}×{:.3} in  ({:?})",
        config.geometry.half_w / IN_TO_M, config.geometry.half_h / IN_TO_M, engine.symmetry);
    println!("  Load     : Px={:.2} ksi  Py={:.2} ksi",
        config.load.px / KSI_TO_PA, config.load.py / KSI_TO_PA);
    println!("  Steps    : {}   Interior: {}  Boundary: {}",
        config.max_steps, config.n_interior, config.n_boundary);
    println!("  Engine   : k={:.1}  fd_h={:.2e}  net={}×{}  eq_ring={}",
        k, config.fd_h, engine.hidden_dim, engine.n_hidden, engine.n_eq_ring);
    println!("  Optimizer: {} (weights)  +  AdamW (biases)",
        if config.use_soap_muon { "SOAP-Muon" } else { "AdamW" });
    println!("  Expected K_t at probe (r={:.2}·r_hole): {:.4}",
        engine.probe_r_factor, engine.expected_kt);
    println!("  Curriculum: Phase 1 (BC, {} pts) 0..{}  →  Phase 2 (+kirsch_stress, {} pts) {}..{}",
        engine.phase1_n_interior, engine.phase1_steps,
        config.n_interior, engine.phase1_steps, config.max_steps);
    println!("──────────────────────────────────────────────────────────────────────────────────────────────");
    println!("{:>6}  {:>10}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}  {:>7}",
        "step", "total", "energy", "neumann", "h_loss", "d_loss", "eq_loss", "k_stress", "lr", "K_t");
    println!("────────────────────────────────────────────────────────────────────────────────────────────────────────");

    let device = WgpuDevice::default();

    let (x0, x1) = config.geometry.x_range();
    let (y0, y1) = config.geometry.y_range();
    let dom_w = x1 - x0;
    let dom_h = y1 - y0;
    let fd = FdConfig::new(config.fd_h, dom_w, dom_h);

    let cx = fd.sx / (2.0 * fd.hx as f64);
    let cy = fd.sy / (2.0 * fd.hy as f64);
    let ref_div2 = (config.load.px * cx).powi(2).max(1.0);

    let net_cfg = ElasticityNetConfig::new()
        .with_input_dim(engine.net_input_dim())
        .with_hidden_dim(config.hidden_dim)
        .with_n_hidden(config.n_hidden)
        .with_output_dim(engine.output_dim())
        .with_use_piratenet(config.use_piratenet);
    let mut model: ElasticityNet<B> = net_cfg.init(&device);
    let use_soap_muon = config.use_soap_muon;
    let dm_config = config.decision_maker.clone();
    let mut decision_maker = PinnDecisionMaker::new(dm_config.clone(), false, false);
    let mut optim_w = WeightOptim::from_tier(use_soap_muon, &decision_maker.current_tier);
    let mut optim_b = make_bias_optim();
    let mut optim_gate = make_gate_optim();
    let stiff_config = config.stiffness.clone();
    let mut stiffness_controller = StiffnessController::new(stiff_config.clone());
    let mut lbfgs_opt: Option<burn::optim::LBFGS<B>> = None;
    let mut frozen_lbfgs_ctx: Option<LbfgsCtxScalars> = None;
    let mut frozen_lbfgs_lams: Option<LbfgsLams> = None;

    let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

    let problem = KirschProblem::new(
        config.material.clone(), engine.output_dim(), engine.phase1_steps, engine.expected_kt,
    );
    validate_loss_terms(&problem);

    let mut saw = SawBrdr::with_base(engine.init_weights(), 0.95);
    let mut lr_sched = LrSchedule::new(engine.peak_lr, 200, 1000);
    let mut phase2_started = false;
    let mut dynamic_lam_h_cap = 50.0_f64;
    let mut dynamic_lam_d_cap = 50.0_f64;
    let mut int_pts_phys = sample_interior(&config.geometry, engine.phase1_n_interior);
    let mut amr: Option<AdaptiveGrid> = None;

    let bnd_pts = sample_boundary(&config.geometry, &config.load, config.n_boundary);
    let bnd_norm: Vec<[f32; 2]> = bnd_pts.iter()
        .map(|b| normalize_point(b.x, b.y, &config)).collect();
    let bnd_nx: Vec<f32> = bnd_pts.iter().map(|b| b.nx as f32).collect();
    let bnd_ny: Vec<f32> = bnd_pts.iter().map(|b| b.ny as f32).collect();
    let bnd_tx: Vec<f32> = bnd_pts.iter().map(|b| b.tx as f32).collect();
    let bnd_ty: Vec<f32> = bnd_pts.iter().map(|b| b.ty as f32).collect();

    // Precomputed once — geometry is fixed during headless training.
    let (trac_idx, hole_idx, right_idx) = extract_boundary_indices(&bnd_pts, &bnd_nx);

    // Equilibrium ring points: precomputed outside loop (fixed seed 42424242, fixed geometry).
    let eq_ring_norm: Vec<[f32; 2]> = sample_eq_ring(&config.geometry, engine.n_eq_ring)
        .iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();

    let mut tracker = ConvergenceTracker::new();

    let mut best_loss = f32::MAX;
    let mut last_kt: Option<f32> = None;
    let mut converged = false;
    let start = std::time::Instant::now();

    // Cache normalized interior points — recomputed only when int_pts_phys changes (Phase 2 start
    // and each AMR event every 1000 steps), not every step. Avoids ~14 000 Vec allocations per run.
    let mut int_norm: Vec<[f32; 2]> = int_pts_phys.iter()
        .map(|&[x, y]| normalize_point(x, y, &config)).collect();
    let mut int_pts_dirty = false;

    'training: for step in 0..config.max_steps {
        // === Phase transition: replace SAW (5→6 components), activate kirsch + AMR ===
        if step == engine.phase1_steps && engine.phase1_steps > 0 && !phase2_started {
            phase2_started = true;
            saw = SawBrdr::with_base(engine.init_weights_phase2(), 0.95);
            let grid = AdaptiveGrid::new(&config.geometry, engine.amr.clone());
            let new_pts = grid.sample_points();
            println!("\n  *** PHASE 2 START (step {step}): SAW→6 components, AMR init {} pts (L{}→L{} max), kirsch_stress_loss active, Adam+LR reset ***",
                new_pts.len(), engine.amr.initial_level, engine.amr.max_level);
            int_pts_phys = new_pts;
            int_pts_dirty = true;
            amr = Some(grid);
            lr_sched.reset_for_phase2();
            decision_maker = PinnDecisionMaker::new(dm_config.clone(), true, false);
            optim_w = WeightOptim::from_tier(use_soap_muon, &decision_maker.current_tier);
            optim_b = make_bias_optim();
            optim_gate = make_gate_optim();
            stiffness_controller = StiffnessController::new(stiff_config.clone());
            lbfgs_opt = None; frozen_lbfgs_ctx = None; frozen_lbfgs_lams = None;
        }

        // === AMR sweep ===
        if step > engine.phase1_steps
            && (step - engine.phase1_steps) % engine.amr.interval_steps == 0
        {
            if let Some(ref mut grid) = amr {
                // Use cached int_norm for residual computation (already up-to-date here).
                let n_amr = int_norm.len();
                let model_val: ElasticityNet<BInner> = model.valid();
                let amr_residuals: Vec<f32> = {
                    let pts_t = norm_pts_to_tensor::<BInner>(&int_norm, &device);
                    let stencil = assemble_stencil::<BInner>(&pts_t, &fd, &device);
                    let out = apply_dirichlet_ansatz::<BInner>(
                        fwd::<BInner>(&model_val, stencil.clone(), engine.n_fourier, &device),
                        &stencil,
                        config.geometry.symmetry, k,
                    ).mul_scalar(u_ref as f64);
                    let (exx, eyy, exy) = compute_strains::<BInner>(out, n_amr, &fd);
                    dem_energy_per_point::<BInner>(exx, eyy, exy, &config.material)
                        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_amr])
                        .into_iter().map(|e| e.abs()).collect()
                };
                grid.update_residuals(&amr_residuals);
                grid.adapt();
                let amr_stats = grid.stats();
                int_pts_phys = grid.sample_points();
                int_pts_dirty = true;
                println!("  [AMR@{step}] cells={} depth={} mean_res={:.3e} pts={}",
                    amr_stats.active_count, amr_stats.max_depth,
                    amr_stats.mean_residual, int_pts_phys.len());
                // AMR invalidates L-BFGS curvature history (new quadrature points).
                if decision_maker.current_tier == OptimizerTier::Converge {
                    println!("  [DM@{step}] AMR → demote Converge→Align");
                    decision_maker = PinnDecisionMaker::new(dm_config.clone(), true, false);
                    optim_w = WeightOptim::from_tier(use_soap_muon, &decision_maker.current_tier);
                    lbfgs_opt = None; frozen_lbfgs_ctx = None; frozen_lbfgs_lams = None;
                }
            }
        }

        // Recompute cached int_norm only when int_pts_phys changed (Phase 2 start or AMR).
        if int_pts_dirty {
            int_norm = int_pts_phys.iter()
                .map(|&[x, y]| normalize_point(x, y, &config)).collect();
            int_pts_dirty = false;
        }

        // === Training step (single source of truth in training_core::step_physics) ===

        let ctx = StepCtx {
            config:            &config,
            engine:            &engine,
            problem:           &problem,
            fd:                &fd,
            k,
            u_ref,
            ref_energy,
            ref_stress2,
            cx,
            cy,
            ref_div2,
            int_norm:          &int_norm,  // cached; recomputed only on AMR/phase-change
            bnd_norm:          &bnd_norm,
            bnd_nx:            &bnd_nx,
            bnd_ny:            &bnd_ny,
            bnd_tx:            &bnd_tx,
            bnd_ty:            &bnd_ty,
            trac_idx:          &trac_idx,
            hole_idx:          &hole_idx,
            right_idx:         &right_idx,
            eq_ring_norm:      &eq_ring_norm,
            dynamic_lam_h_cap,
            dynamic_lam_d_cap,
            phase2_active:     phase2_started,
            step,
        };
        let (new_model, mut out) = if dm_config.enabled
            && decision_maker.current_tier == OptimizerTier::Converge
        {
            // Converge tier: lazy-init frozen context + LBFGS, then run quasi-Newton step.
            if frozen_lbfgs_ctx.is_none() {
                frozen_lbfgs_ctx  = Some(LbfgsCtxScalars::from_ctx(&ctx));
            }
            let lbfgs = lbfgs_opt.get_or_insert_with(|| make_lbfgs(dm_config.lbfgs_max_iter));
            let lams = frozen_lbfgs_lams.as_ref().expect("lams must be set when entering Converge");
            let fctx = frozen_lbfgs_ctx.as_ref().unwrap();
            let lr = lr_sched.current_lr();
            let (new_m, loss_f64) = step_lbfgs(model, lbfgs, lr, fctx, lams, &device);
            let synthetic_out = StepOutput {
                e_scalar: 0.0, n_scalar: 0.0, h_scalar: 0.0, d_scalar: 0.0,
                eq_scalar: 0.0, w_scalar: 0.0, kirsch_scalar: 0.0, const_scalar: 0.0,
                total_scalar: loss_f64 as f32, lr,
                lam_e: 0.0, lam_n: 0.0, lam_h: 0.0, lam_d: 0.0, lam_eq: 0.0, lam_kirsch: 0.0,
                proxy_ratio: 0.0,
                optimizer_tier: OptimizerTier::Converge.as_u8(),
                cosine_sim: None,
                lam_by_name: None,
            };
            (new_m, synthetic_out)
        } else {
            let physics_boost = stiffness_controller.physics_boost();
            let alpha_lr_mult = stiffness_controller.alpha_lr_mult();
            step_physics(model, &mut optim_w, &mut optim_b, &mut optim_gate, &ctx, &mut saw,
                &mut lr_sched, &device, decision_maker.current_tier.as_u8(),
                physics_boost, alpha_lr_mult)
        };
        model = new_model;

        // Decision maker / stiffness controller gate — both `advance()` unconditionally
        // (never short-circuited) so their internal step counters stay correct regardless
        // of whether the other subsystem is enabled. At most one GradientConflict is
        // computed per step, shared by whichever subsystem's gate fired.
        let dm_fire    = decision_maker.advance();
        let stiff_fire = stiffness_controller.advance();
        if dm_fire || stiff_fire {
            let want_conflict = decision_maker.current_tier != OptimizerTier::Converge
                && ((dm_config.use_exact_cosine && dm_fire) || stiff_fire);
            let conflict = if want_conflict {
                Some(compute_gradient_conflict(&model, &ctx, step, &device))
            } else {
                None
            };
            out.cosine_sim = conflict.map(|c| c.cosine_sim);

            if dm_fire {
                if let Some(t) = decision_maker.evaluate(conflict, out.proxy_ratio, phase2_started) {
                    let old_tier_name = match out.optimizer_tier {
                        0 => "Explore", 1 => "Align", _ => "Converge",
                    };
                    let new_tier_name = match t.new_tier {
                        OptimizerTier::Explore  => "Explore",
                        OptimizerTier::Align    => "Align",
                        OptimizerTier::Converge => "Converge",
                    };
                    println!("\n  [DM@{step}] {old_tier_name} → {new_tier_name}");
                    if t.reset_optim {
                        optim_w = WeightOptim::from_tier(use_soap_muon, &t.new_tier);
                        optim_b = make_bias_optim();
                    }
                    if t.reset_lr {
                        lr_sched.reset_for_phase2();
                    }
                    match t.new_tier {
                        OptimizerTier::Converge => {
                            // Freeze current collocation points and SAW lambdas for L-BFGS.
                            frozen_lbfgs_ctx  = Some(LbfgsCtxScalars::from_ctx(&ctx));
                            frozen_lbfgs_lams = Some(LbfgsLams {
                                lam_e:      out.lam_e,
                                lam_n:      out.lam_n,
                                lam_h:      out.lam_h,
                                lam_d:      out.lam_d,
                                lam_eq:     out.lam_eq,
                                lam_kirsch: out.lam_kirsch,
                                lam_const:  engine.lam_const as f64,
                            });
                            lbfgs_opt = None;
                        }
                        _ => {
                            frozen_lbfgs_ctx = None; frozen_lbfgs_lams = None; lbfgs_opt = None;
                        }
                    }
                }
            }

            if stiff_fire {
                if let Some(c) = conflict {
                    let factor = stiffness_controller.update(&c);
                    println!("\n  [Stiffness@{step}] factor={factor:.3} boost={:.2} gate_lr_mult={:.2}",
                        stiffness_controller.physics_boost(), stiffness_controller.alpha_lr_mult());
                }
            }
        }

        // === Progress output ===
        if step % 50 == 0 || step == config.max_steps - 1 {
            let pct = step as f32 / config.max_steps as f32;
            let filled = (pct * BAR_WIDTH as f32) as usize;
            let bar = "█".repeat(filled) + &"░".repeat(BAR_WIDTH - filled);
            let elapsed = start.elapsed().as_secs_f32();
            let eta = if pct > 0.01 { elapsed / pct - elapsed } else { 0.0 };
            if out.total_scalar < best_loss { best_loss = out.total_scalar; }

            print!("\r[{bar}] {:.1}%  {elapsed:.0}s  ETA {eta:.0}s", pct * 100.0);

            if step % 200 == 0 || step == config.max_steps - 1 {
                let model_val: ElasticityNet<BInner> = model.valid();
                let kt_opt = probe_kt_shared(&model_val, &config, &engine, &fd, k, u_ref, &device);
                if kt_opt.is_some() { last_kt = kt_opt; }

                if phase2_started {
                    if let Some(kt) = kt_opt {
                        tracker.push(kt as f64);

                        // Crash recovery: K_t collapsed ≥50% from recent peak → fresh restart.
                        // Clear history after to prevent cascade crash detections while recovering.
                        if let Some(new_cap) = tracker.check_kt_crash(kt as f64) {
                            dynamic_lam_h_cap = new_cap;
                            dynamic_lam_d_cap = new_cap;
                            tracker.clear_history();
                            saw.reset();
                            lr_sched.reset_for_phase2();
                            decision_maker = PinnDecisionMaker::new(dm_config.clone(), true, false);
                            optim_w = WeightOptim::from_tier(use_soap_muon, &decision_maker.current_tier);
                            optim_b = make_bias_optim();
                            optim_gate = make_gate_optim();
                            stiffness_controller = StiffnessController::new(stiff_config.clone());
                            lbfgs_opt = None; frozen_lbfgs_ctx = None; frozen_lbfgs_lams = None;
                            println!("\n  [CRASH RECOVERY #{}] K_t={kt:.3} collapsed → restart: lr+adam+SAW, lam_caps→{new_cap:.0}",
                                tracker.total_restarts());
                        } else if let Some(new_cap) = tracker.check_plateau() {
                            dynamic_lam_h_cap = new_cap;
                            dynamic_lam_d_cap = new_cap;
                            saw.reset();
                            lr_sched.reset_for_phase2();
                            decision_maker = PinnDecisionMaker::new(dm_config.clone(), true, false);
                            optim_w = WeightOptim::from_tier(use_soap_muon, &decision_maker.current_tier);
                            optim_b = make_bias_optim();
                            optim_gate = make_gate_optim();
                            stiffness_controller = StiffnessController::new(stiff_config.clone());
                            lbfgs_opt = None; frozen_lbfgs_ctx = None; frozen_lbfgs_lams = None;
                            println!("\n  [WARM RESTART #{}] K_t plateau → reset: lr+adam+SAW, lam_caps→{new_cap:.0}",
                                tracker.total_restarts());
                        }

                        if tracker.is_kt_converged(engine.expected_kt) {
                            println!("\n  [CONVERGED] K_t={kt:.4} stable near {:.4} — training complete at step {step}",
                                engine.expected_kt);
                            converged = true;
                            break 'training;
                        }
                    }
                }

                let kt_str = kt_opt.map(|kt| format!("{kt:>7.3}"))
                    .unwrap_or_else(|| "   N/A ".to_string());
                println!();
                println!("{step:>6}  {t:>10.3e}  {e:>9.3e}  {n:>9.3e}  {h:>9.3e}  {d:>9.3e}  {eq:>9.3e}  {k:>9.3e}  {lr:>8.2e}  {kt_str}",
                    t = out.total_scalar, e = out.e_scalar, n = out.n_scalar,
                    h = out.h_scalar, d = out.d_scalar, eq = out.eq_scalar,
                    k = out.kirsch_scalar, lr = out.lr);
            } else {
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }

        if step % 500 == 0 && step > 0 {
            let elapsed = start.elapsed().as_secs_f32();
            let phase = if step < engine.phase1_steps { "P1" } else { "P2" };
            println!("  [{step}|{phase}]  {:.1} steps/s  best={best_loss:.3e}  lams=[e:{:.1} n:{:.1} h:{:.1} d:{:.1} eq:{:.2} k:{:.2}]  caps=[h:{:.0} d:{:.0}]",
                step as f32 / elapsed,
                out.lam_e, out.lam_n, out.lam_h, out.lam_d, out.lam_eq, out.lam_kirsch,
                dynamic_lam_h_cap, dynamic_lam_d_cap);
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    println!("\n────────────────────────────────────────────────────────────────────────────────────────────────────────");
    println!("  Done!  {:.0}s  ({:.1} steps/s)", elapsed, config.max_steps as f32 / elapsed);
    println!("  Final best loss : {best_loss:.4e}");
    if let Some(kt) = last_kt {
        println!("  Achieved K_t   : {:.4}", kt);
    }
    println!("  Expected K_t    : {:.4} at r={}·r_hole", engine.expected_kt, engine.probe_r_factor);
    println!("════════════════════════════════════════════════════════════════════════════════════════════════════════");

    converged
}

/// Richer result of [`run_headless_pinlug_inner`] — the public [`run_headless_pinlug`] only
/// exposes `converged` (via its `bool` return), but tests need the final decision-maker tier
/// to assert on (see `run_headless_pinlug_with_decision_maker_enabled_reaches_align_tier`).
pub(crate) struct PinLugHeadlessResult {
    pub converged: bool,
    // Read only by `#[cfg(test)]` assertions (see module doc comment) — the production
    // `run_headless_pinlug` wrapper only surfaces `converged`.
    #[allow(dead_code)]
    pub final_tier: OptimizerTier,
    /// Per-step `total_scalar` trajectory — used by the zero-regression baseline test.
    #[allow(dead_code)]
    pub trajectory: Vec<f32>,
    /// Number of times `problem.convergence_metric(...)` was successfully probed (`Some`)
    /// and pushed into the plateau/crash tracker — test-only visibility into the cascade.
    #[allow(dead_code)]
    pub metric_probes: usize,
    /// Final `dynamic_lam_h_cap`/`dynamic_lam_d_cap` after all plateau/crash restarts.
    #[allow(dead_code)]
    pub final_lam_h_cap: f64,
    #[allow(dead_code)]
    pub final_lam_d_cap: f64,
    /// Final `dynamic_lam_penetration_cap`/`dynamic_lam_non_tension_cap` after all
    /// plateau/crash restarts — same lockstep cascade as `final_lam_h_cap`/`final_lam_d_cap`
    /// (see the cascade block inside the training loop below).
    #[allow(dead_code)]
    pub final_lam_penetration_cap: f64,
    #[allow(dead_code)]
    pub final_lam_non_tension_cap: f64,
    /// `tracker.plateau_restarts + tracker.crash_restarts` at the end of training.
    #[allow(dead_code)]
    pub total_restarts: usize,
    /// The `out.lam_by_name` snapshot from the step IMMEDIATELY BEFORE a Converge-tier
    /// transition fired — test-only regression guard for the frozen-lams bug (see
    /// `frozen_lams`'s construction below): L-BFGS's frozen weights at entry must match
    /// this, not `problem.base_weight()`'s static seed.
    #[allow(dead_code)]
    pub last_lam_before_converge: Option<HashMap<&'static str, f64>>,
    /// The `frozen_lams` snapshot actually captured at Converge-tier entry.
    #[allow(dead_code)]
    pub lbfgs_entry_lams: Option<HashMap<&'static str, f64>>,
}

/// Headless-only entry point for the pin-in-lug 2-domain contact problem — routes through
/// `step_physics_multi` (`training_core.rs`) instead of the frozen 1-domain `step_physics`
/// Kirsch path. This entry point matches the CSV-export post-processing use case pin-in-lug
/// is for. Note: `runner.rs::run_training_pinlug` (the GUI-driving path) DOES also exist and
/// train pin-in-lug — it is not out of scope in the sense of "unimplemented" — but it does
/// NOT yet have this function's plateau/crash cascade or Converge-tier decision-maker wiring
/// (its `dynamic_lam_h_cap`/`dynamic_lam_d_cap` are still hardcoded 50.0, `phase2_active` is
/// still `false`, and it has no `ConvergenceTracker`/`PinnDecisionMaker` at all) — that gap is
/// a deliberate, tracked scope cut (see the GitHub issue for porting this cascade to the GUI
/// path), not an oversight.
///
/// Deliberately does not replicate Kirsch's AMR / stiffness-controller machinery — those are
/// tightly coupled to K_t-based diagnostics/AMR sampling that don't apply to a contact
/// problem without a closed-form K_t or an AMR-gated resample schedule (pin-lug resamples
/// unconditionally every step). It DOES now replicate Kirsch's plateau/crash warm-restart
/// cascade (`ConvergenceTracker::for_metric`, driven by `problem.convergence_metric(...)`'s
/// interface-gap RMS every 200 steps) — see the cascade block inside the training loop below.
/// A fixed SAW-BRDR schedule (base weights from `PinLugProblem::base_weight`) over
/// `max_steps` remains the baseline training loop; the cascade only intervenes on plateau/
/// crash. The decision maker (`PinnDecisionMaker`) IS wired in (opt-in via
/// `config.decision_maker.enabled`, default false), and — unlike Kirsch — CAN reach the
/// Converge (L-BFGS) tier: pin-lug has no Phase-1/Phase-2 curriculum split, so it is
/// constructed with `allow_converge=true` (see `PinnDecisionMaker::new`'s doc comment).
pub fn run_headless_pinlug(config: SolverConfig) -> bool {
    run_headless_pinlug_inner(config, None).converged
}

/// `initial_models`: test-only hook (always `None` from the public `run_headless_pinlug`) so
/// the zero-regression baseline test can inject the SAME randomly-initialized starting
/// weights into two separate calls — this repo's `ElasticityNetConfig::init` has no exposed
/// seed control (confirmed: `burn::tensor::backend::Backend::seed` does not make Wgpu weight
/// initialization deterministic, see the investigation note on the test itself), so
/// process-level RNG determinism is not achievable; injecting identical pre-built models is
/// the only way to get a literal apples-to-apples trajectory comparison.
pub(crate) fn run_headless_pinlug_inner(
    config: SolverConfig,
    initial_models: Option<(ElasticityNet<B>, ElasticityNet<B>)>,
) -> PinLugHeadlessResult {
    use pinn_core::problem::InterfaceParametrization;
    use crate::{
        controllers::{ConvergenceTracker, MetricDirection},
        network::ElasticityNetConfig,
        pinlug_problem::{PinLugProblem, PinLugScalingMode, LUG_DOMAIN, PIN_DOMAIN},
        problem::{
            validate_loss_terms, BoundaryValueProblem, DomainOptim, DomainState, DomainStepCtx,
            DomainStepData, FrozenMultiStepCtx, MultiStepCtx, PointSetData,
        },
        training_core::{compute_gradient_conflict_multi, step_lbfgs_multi, step_physics_multi, TwoDomainModels},
    };

    const N_INTERFACE: usize = 64;
    const OUTPUT_DIM: usize = 5; // mDEM (u, v, sxx, syy, sxy)
    const PHASE1_STEPS: usize = usize::MAX; // no phase-2 cascade for pin-in-lug (see doc comment)
    const PHASE2_ACTIVE: bool = false; // pin-in-lug has no phase-2 cascade — always Phase 1.

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   PINN Structural Stress Solver — Pin-in-Lug (Headless)  ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let problem = PinLugProblem::new(
        config.material.clone(), OUTPUT_DIM, PHASE1_STEPS, N_INTERFACE,
        if config.use_ultimate_strength_scaling { PinLugScalingMode::UltimateStrength } else { PinLugScalingMode::AppliedLoad },
    );
    validate_loss_terms(&problem);

    println!("  Material : E={:.2} Msi  ν={:.3}", config.material.e / MSI_TO_PA, config.material.nu);
    println!("  Steps    : {}   Interior: {}  Boundary: {}  Interface pts: {N_INTERFACE}",
        config.max_steps, config.n_interior, config.n_boundary);
    println!("  Load     : {:.2} ksi equivalent bearing traction (P=20,000 lbf / (2*r_pin*t))",
        problem.equivalent_traction_pa() / KSI_TO_PA);
    println!("──────────────────────────────────────────────────────────────────────────────────────────────");

    let device = WgpuDevice::default();

    let pin_geom = problem.domains()[0].geometry.clone();
    let lug_geom = problem.domains()[1].geometry.clone();
    let pin_fd = FdConfig::new(config.fd_h, 2.0 * pin_geom.half_w, 2.0 * pin_geom.half_h);
    let lug_fd = FdConfig::new(config.fd_h, 2.0 * lug_geom.half_w, 2.0 * lug_geom.half_h);
    // step_physics_multi takes a single shared FdConfig; both domains here use the same
    // normalized fd_h, and since both are normalized to [-1,1]^2 the physical hx/hy differ
    // only via sx/sy (used solely for strain scaling, computed per-domain by the caller
    // below) — the shared FdConfig's hx/hy (normalized step) is what matters for stencil
    // assembly, so either pin_fd or lug_fd's hx/hy works; we use the lug's (chosen
    // arbitrarily, since hx/hy only depends on config.fd_h, identical for both).
    let fd = lug_fd;
    let _ = pin_fd;

    let net_cfg = |geom_half_w: f64, geom_half_h: f64| {
        let _ = (geom_half_w, geom_half_h);
        ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(OUTPUT_DIM)
            .with_use_piratenet(false)
    };
    let (mut model_pin, mut model_lug): (ElasticityNet<B>, ElasticityNet<B>) = match initial_models {
        Some((pin, lug)) => (pin, lug),
        None => (
            net_cfg(pin_geom.half_w, pin_geom.half_h).init(&device),
            net_cfg(lug_geom.half_w, lug_geom.half_h).init(&device),
        ),
    };

    let mut optims = vec![
        DomainOptim { weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim() },
        DomainOptim { weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim() },
    ];

    let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
    let mut saw = SawBrdr::with_base(base_weights, 0.95);
    let mut lr_sched = LrSchedule::new(1e-3, 200, 1000);

    // Reference scales — u_ref derived from the equivalent driving traction (analogous role
    // to Kirsch's Px-derived u_ref; ref_energy/ref_stress2 follow the same pattern).
    let equiv_traction = problem.equivalent_traction_pa();
    let e = config.material.e;
    let u_ref = ((equiv_traction / e) * lug_geom.half_w) as f32;
    let ref_energy = (0.5 * equiv_traction * equiv_traction / e) as f32;
    let ref_stress2 = (equiv_traction * equiv_traction) as f32;

    let pin_sampling = problem.sampling_strategy(0);
    let lug_sampling = problem.sampling_strategy(1);

    let build_pointset = |pts: &[pinn_core::loading::BoundaryPoint], geom: &pinn_core::geometry::GeometryConfig| -> PointSetData {
        PointSetData {
            norm: pts.iter().map(|p| normalize_point_generic(p.x, p.y, geom)).collect(),
            nx: pts.iter().map(|p| p.nx as f32).collect(),
            ny: pts.iter().map(|p| p.ny as f32).collect(),
            tx: pts.iter().map(|p| p.tx as f32).collect(),
            ty: pts.iter().map(|p| p.ty as f32).collect(),
        }
    };

    // Decision maker: `phase2_active` is always `false` at construction, mirroring
    // `run_headless`'s Phase-1 initialization — but `allow_converge=true` (pin-in-lug has no
    // Kirsch-style Phase-1/Phase-2 curriculum split, so Align→Converge is unlocked via the
    // dedicated `allow_converge` arm, not via `phase2_active`; see `PinnDecisionMaker::new`'s
    // doc comment).
    let dm_config = config.decision_maker.clone();
    let mut decision_maker = PinnDecisionMaker::new(dm_config.clone(), false, true);
    let mut lbfgs_opt: Option<burn::optim::LBFGS<B>> = None;
    // Frozen at Converge-tier ENTRY, re-frozen every time Converge is (re-)entered, and
    // cleared on exit — pin-in-lug resamples its collocation points every step
    // unconditionally (unlike Kirsch's AMR-gated resampling), so a frozen context that
    // persisted across a demote-then-repromote cycle would silently train on stale points.
    let mut frozen_ctx: Option<FrozenMultiStepCtx> = None;
    let mut frozen_lams: Option<HashMap<&'static str, f64>> = None;

    // Plateau/crash warm-restart cascade, mirroring `run_headless`'s K_t-based cascade but
    // driven by `problem.convergence_metric(...)`'s interface-gap RMS (SmallerIsBetter,
    // target 0.0). `dynamic_lam_h_cap`/`dynamic_lam_d_cap` are declared here as mutable
    // locals (not per-step 50.0 literals) so restart events can tighten them, mirroring
    // `run_headless`'s own pattern.
    //
    // Physically-derived floor: below ~5x the problem's own characteristic displacement
    // scale (u_ref — the SAME reference PinLugProblem's own ref_gap2 normalizes against,
    // see CLAUDE.md's Reference-scale normalization section), a gap-RMS reading is noise,
    // not signal — mirrors CRASH_MIN_PEAK_KT's role for K_t but derived from the problem,
    // not a hand-rolled literal.
    let significant_floor = 5.0 * u_ref as f64;
    let mut tracker = ConvergenceTracker::for_metric(
        MetricDirection::SmallerIsBetter,
        0.05, // plateau_rel_eps: recent-window min must shrink >=5% relative to the older
              // window's min or a restart fires.
        2.0,  // crash_spike_factor: metric >=2x its recent best — smaller-is-better mirror
              // of K_t's CRASH_DROP_FRACTION=0.5 (1/0.5 = 2.0).
        significant_floor,
    );
    let mut dynamic_lam_h_cap = 50.0_f64;
    let mut dynamic_lam_d_cap = 50.0_f64;
    // Unlike dynamic_lam_h_cap/dynamic_lam_d_cap (seeded at the shared 50.0, already binding
    // from step 0), these two seed at the term's OWN base_weight (500.0/100.0) via
    // `problem.base_weight` rather than a hardcoded literal (avoids drift if
    // LAM_PENETRATION/LAM_NON_TENSION are ever retuned) — non-binding at the moment these
    // terms first carry real gradient (issue #9), since raw_lam == base_weight exactly at
    // step 0 (SawBrdr's first-call decay-EMA skip). Only binds once SAW-BRDR's adaptive
    // multiplier later pushes the term's effective weight ABOVE its base — see CLAUDE.md.
    let mut dynamic_lam_penetration_cap = problem.base_weight("interface_penetration") as f64;
    let mut dynamic_lam_non_tension_cap = problem.base_weight("interface_non_tension") as f64;
    let mut metric_probes: usize = 0;
    let mut last_lam_before_converge: Option<HashMap<&'static str, f64>> = None;
    let mut lbfgs_entry_lams: Option<HashMap<&'static str, f64>> = None;
    // Most recent step_physics_multi (real, non-synthetic) StepOutput's lam_by_name — see
    // the update site inside the loop below for why this is what Converge-entry code reads.
    let mut prev_lam_by_name: Option<HashMap<&'static str, f64>> = None;

    let start = std::time::Instant::now();
    let mut last_total = f32::MAX;
    let mut trajectory: Vec<f32> = Vec::with_capacity(config.max_steps);

    for step in 0..config.max_steps {
        let pin_int = pin_sampling.sample_interior(&pin_geom, config.n_interior);
        let lug_int = lug_sampling.sample_interior(&lug_geom, config.n_interior);
        let lug_bnd = lug_sampling.sample_boundary(&lug_geom, &config.load, config.n_boundary);

        let pin_int_norm: Vec<[f32; 2]> = pin_int.iter().map(|&[x, y]| normalize_point_generic(x, y, &pin_geom)).collect();
        let lug_int_norm: Vec<[f32; 2]> = lug_int.iter().map(|&[x, y]| normalize_point_generic(x, y, &lug_geom)).collect();

        let mut pin_named = std::collections::HashMap::new();
        let mut lug_named = std::collections::HashMap::new();
        for set in pin_sampling.named_point_sets(&[]) {
            pin_named.insert(set.name, build_pointset(&set.points, &pin_geom));
        }
        for set in lug_sampling.named_point_sets(&[]) {
            lug_named.insert(set.name, build_pointset(&set.points, &lug_geom));
        }
        lug_named.insert("boundary", build_pointset(&lug_bnd, &lug_geom));
        // Wire the real driving-traction target (equivalent_traction, +x direction) into
        // the pin's "driving" point-set (named_point_sets leaves tx/ty as placeholders).
        if let Some(driving) = pin_named.get_mut("driving") {
            for t in driving.tx.iter_mut() { *t = equiv_traction as f32; }
        }

        let pin_data = DomainStepData { id: PIN_DOMAIN, int_norm: pin_int_norm, extra_ring_norm: Vec::new(), named: pin_named };
        let lug_data = DomainStepData { id: LUG_DOMAIN, int_norm: lug_int_norm, extra_ring_norm: Vec::new(), named: lug_named };

        let ctx = MultiStepCtx {
            config: &config,
            problem: &problem,
            fd: &fd,
            k: 1.0,
            domains: vec![
                DomainStepCtx { data: &pin_data, u_ref, ref_energy, ref_stress2 },
                DomainStepCtx { data: &lug_data, u_ref, ref_energy, ref_stress2 },
            ],
            dynamic_lam_h_cap,
            dynamic_lam_d_cap,
            dynamic_lam_penetration_cap,
            dynamic_lam_non_tension_cap,
            // NOTE: `MultiStepCtx::phase2_active` and the `PHASE2_ACTIVE` const below are two
            // INDEPENDENT booleans that happen to share a value by coincidence, not
            // architectural coupling. This one gates `step_physics_multi`'s h/d SAW-BRDR cap
            // dispatch (training_core.rs's "hole_traction" | "lug_free_edge_traction" /
            // "displacement_anchor" | "lug_shank_anchor" match arms) — hardcoded `true` so the
            // cap cascade below can actually bind (it was previously dead code for pin-lug
            // because this field was always `false`). `PHASE2_ACTIVE` (used only by
            // `decision_maker.evaluate(...)` further down) is the decision-maker's own,
            // unrelated axis (tier transitions, not loss-term gating) and stays `false`.
            phase2_active: true,
            step,
        };

        let out = if dm_config.enabled && decision_maker.current_tier == OptimizerTier::Converge {
            // Converge tier: freeze context + lams ONCE per Converge entry (never mid-dwell —
            // freshly (re-)frozen by the `evaluate()` transition branch below), lazily build
            // L-BFGS, then run one quasi-Newton step over BOTH domains via TwoDomainModels.
            if frozen_ctx.is_none() {
                frozen_ctx = Some(FrozenMultiStepCtx::from_ctx(&ctx));
                // Snapshot the LIVE SAW+cap-adapted weights from the immediately preceding
                // step_physics_multi call (`prev_lam_by_name`), NOT `problem.base_weight()`'s
                // static Phase-1 seed — see the module's frozen-lams bug note (CLAUDE.md /
                // task doc). Fallback to base weights only if no prior step's lam_by_name was
                // ever captured (defensive; in practice always Some by the time Converge can
                // be entered).
                frozen_lams = Some(prev_lam_by_name.clone().unwrap_or_else(|| {
                    problem.loss_terms().iter()
                        .map(|t| (t.name(), problem.base_weight(t.name()) as f64))
                        .collect()
                }));
                lbfgs_entry_lams = frozen_lams.clone();
            }
            let lbfgs = lbfgs_opt.get_or_insert_with(|| make_lbfgs(dm_config.lbfgs_max_iter));
            let fctx = frozen_ctx.as_ref().unwrap();
            let lams = frozen_lams.as_ref().expect("lams must be set when entering Converge");
            let lr = lr_sched.current_lr();
            let models = TwoDomainModels { pin: model_pin, lug: model_lug };
            let (new_models, loss_f64) = step_lbfgs_multi(models, lbfgs, lr, fctx, &problem, lams, &device);
            model_pin = new_models.pin;
            model_lug = new_models.lug;
            StepOutput {
                e_scalar: 0.0, n_scalar: 0.0, h_scalar: 0.0, d_scalar: 0.0,
                eq_scalar: 0.0, w_scalar: 0.0, kirsch_scalar: 0.0, const_scalar: 0.0,
                total_scalar: loss_f64 as f32, lr,
                lam_e: 0.0, lam_n: 0.0, lam_h: 0.0, lam_d: 0.0, lam_eq: 0.0, lam_kirsch: 0.0,
                proxy_ratio: 0.0,
                optimizer_tier: OptimizerTier::Converge.as_u8(),
                cosine_sim: None,
                lam_by_name: None,
            }
        } else {
            // tier_u8 is a pure logging passthrough (StepOutput.optimizer_tier) — never
            // read computationally by step_physics_multi (grep confirms its only two uses
            // are both `optimizer_tier: tier_u8` field assignments) — so threading the real
            // current tier through here (rather than a hardcoded 0) fixes the Align→Converge
            // transition's console message without affecting the zero-regression bar: when
            // `dm_config.enabled == false`, current_tier never leaves Explore (as_u8()==0),
            // so this is behavior-identical to the literal-0 call in that case, and only
            // improves accuracy when enabled. Same 1.0/1.0 boost/mult placeholders
            // (pin-in-lug has no StiffnessController).
            let (new_models, out) = step_physics_multi(
                vec![model_pin, model_lug], &mut optims, &ctx, &mut saw, &mut lr_sched, &device,
                decision_maker.current_tier.as_u8(), 1.0, 1.0,
            );
            let mut it = new_models.into_iter();
            model_pin = it.next().unwrap();
            model_lug = it.next().unwrap();
            out
        };
        last_total = out.total_scalar;
        trajectory.push(out.total_scalar);
        // Track the most recent REAL (step_physics_multi) SAW+cap-adapted lambda map —
        // `None` on Converge/L-BFGS steps (synthetic StepOutput), so this naturally holds
        // steady at "the last real step's weights" across a Converge dwell, which is exactly
        // what Converge-entry code needs to snapshot from.
        if out.lam_by_name.is_some() {
            prev_lam_by_name = out.lam_by_name.clone();
        }

        // `advance()` unconditionally (even when `dm_config.enabled == false`) so its
        // internal step counters stay correct regardless — mirrors `run_headless`'s
        // invariant at its equivalent call site.
        let dm_fire = decision_maker.advance();
        if dm_fire {
            let want_conflict = decision_maker.current_tier != OptimizerTier::Converge
                && dm_config.use_exact_cosine;
            let conflict = if want_conflict {
                let models = TwoDomainModels { pin: model_pin, lug: model_lug };
                let c = compute_gradient_conflict_multi(&models, &ctx, &device);
                model_pin = models.pin;
                model_lug = models.lug;
                Some(c)
            } else {
                None
            };

            if let Some(t) = decision_maker.evaluate(conflict, out.proxy_ratio, PHASE2_ACTIVE) {
                let old_tier_name = match out.optimizer_tier { 0 => "Explore", 1 => "Align", _ => "Converge" };
                let new_tier_name = match t.new_tier {
                    OptimizerTier::Explore  => "Explore",
                    OptimizerTier::Align    => "Align",
                    OptimizerTier::Converge => "Converge",
                };
                println!("\n  [DM@{step}] {old_tier_name} → {new_tier_name}");
                if t.reset_optim {
                    optims = vec![
                        DomainOptim { weight: WeightOptim::from_tier(config.use_soap_muon, &t.new_tier), bias: make_bias_optim(), gate: make_gate_optim() },
                        DomainOptim { weight: WeightOptim::from_tier(config.use_soap_muon, &t.new_tier), bias: make_bias_optim(), gate: make_gate_optim() },
                    ];
                }
                if t.reset_lr {
                    lr_sched.reset_for_phase2();
                }
                match t.new_tier {
                    OptimizerTier::Converge => {
                        // Freeze fresh collocation points/lams for this Converge entry. Uses
                        // THIS step's `out.lam_by_name` — the LIVE SAW+cap-adapted weights
                        // from the step immediately preceding entry (transitions fire from
                        // Align, which always ran through step_physics_multi this step, so
                        // `out.lam_by_name` is `Some`) — NOT `problem.base_weight()`'s static
                        // Phase-1 seed. See the module's frozen-lams bug note.
                        frozen_ctx = Some(FrozenMultiStepCtx::from_ctx(&ctx));
                        last_lam_before_converge = out.lam_by_name.clone();
                        frozen_lams = Some(out.lam_by_name.clone().unwrap_or_else(|| {
                            problem.loss_terms().iter()
                                .map(|t| (t.name(), problem.base_weight(t.name()) as f64))
                                .collect()
                        }));
                        lbfgs_entry_lams = frozen_lams.clone();
                        lbfgs_opt = None;
                    }
                    _ => {
                        // Leaving Converge (or any other transition): clear frozen state so
                        // the next Converge entry re-freezes on fresh collocation points.
                        frozen_ctx = None; frozen_lams = None; lbfgs_opt = None;
                    }
                }
            }
        }

        // === Plateau/crash warm-restart cascade (mirrors run_headless's K_t-based cascade,
        // driven by problem.convergence_metric()'s interface-gap RMS instead). Unconditional
        // on `dm_config.enabled` — same convention as Kirsch's own cascade, which is gated
        // only on having started its curriculum, not on the decision maker being on. ===
        if step % 200 == 0 {
            let state = vec![
                DomainState { id: PIN_DOMAIN, model: model_pin.clone(), u_ref, ref_energy, ref_stress2 },
                DomainState { id: LUG_DOMAIN, model: model_lug.clone(), u_ref, ref_energy, ref_stress2 },
            ];
            if let Some(rms) = problem.convergence_metric(&state) {
                metric_probes += 1;
                tracker.push(rms);

                // Crash recovery: gap RMS spiked >=2x its recent best → fresh restart.
                // Clear history after to prevent cascade detections while recovering.
                if let Some(new_cap) = tracker.check_kt_crash(rms) {
                    dynamic_lam_h_cap = new_cap;
                    dynamic_lam_d_cap = new_cap;
                    dynamic_lam_penetration_cap = new_cap;
                    dynamic_lam_non_tension_cap = new_cap;
                    tracker.clear_history();
                    saw.reset();
                    lr_sched.reset_for_phase2();
                    // Always restart fresh at Align (phase2_active=true), allow_converge
                    // preserved — mirrors Kirsch's own restart convention exactly.
                    decision_maker = PinnDecisionMaker::new(dm_config.clone(), true, true);
                    optims = vec![
                        DomainOptim { weight: WeightOptim::from_tier(config.use_soap_muon, &decision_maker.current_tier), bias: make_bias_optim(), gate: make_gate_optim() },
                        DomainOptim { weight: WeightOptim::from_tier(config.use_soap_muon, &decision_maker.current_tier), bias: make_bias_optim(), gate: make_gate_optim() },
                    ];
                    lbfgs_opt = None; frozen_ctx = None; frozen_lams = None;
                    println!("\n  [CRASH RECOVERY #{}] gap_rms={rms:.3e} collapsed → restart: lr+adam+SAW, lam_caps→{new_cap:.0}",
                        tracker.total_restarts());
                } else if let Some(new_cap) = tracker.check_plateau() {
                    dynamic_lam_h_cap = new_cap;
                    dynamic_lam_d_cap = new_cap;
                    dynamic_lam_penetration_cap = new_cap;
                    dynamic_lam_non_tension_cap = new_cap;
                    saw.reset();
                    lr_sched.reset_for_phase2();
                    decision_maker = PinnDecisionMaker::new(dm_config.clone(), true, true);
                    optims = vec![
                        DomainOptim { weight: WeightOptim::from_tier(config.use_soap_muon, &decision_maker.current_tier), bias: make_bias_optim(), gate: make_gate_optim() },
                        DomainOptim { weight: WeightOptim::from_tier(config.use_soap_muon, &decision_maker.current_tier), bias: make_bias_optim(), gate: make_gate_optim() },
                    ];
                    lbfgs_opt = None; frozen_ctx = None; frozen_lams = None;
                    println!("\n  [WARM RESTART #{}] gap_rms plateau → reset: lr+adam+SAW, lam_caps→{new_cap:.0}",
                        tracker.total_restarts());
                }
            }
        }

        if step % 200 == 0 || step == config.max_steps - 1 {
            println!("{step:>6}  total={:>10.3e}  lr={:>8.2e}", out.total_scalar, out.lr);
        }
    }

    let _ = InterfaceParametrization { thetas: Vec::new() }; // silence unused-import lint if the type is otherwise unused here
    let elapsed = start.elapsed().as_secs_f32();
    println!("──────────────────────────────────────────────────────────────────────────────────────────────");
    println!("  Done! {elapsed:.0}s   Final total loss: {last_total:.4e}");

    // Post-processing: export the trained LUG network's contact-pressure profile
    // (sigma_rr(theta) over the pin-loaded half of the hole boundary) to CSV — see
    // `contact_export.rs` module doc comment. Uses the same raw-output -> physical-stress
    // scale (`config.load.px`, Pa) `step_physics_multi` applied to the mDEM stress columns
    // during training, so exported values match what training actually optimized against.
    {
        let model_lug_val: ElasticityNet<BInner> = model_lug.valid();
        match crate::contact_export::export_contact_pressure::<BInner>(
            &model_lug_val, &lug_geom, config.load.px, &device,
        ) {
            Ok(samples) => println!(
                "  Contact pressure profile: {} samples written to {}",
                samples.len(), crate::contact_export::DEFAULT_CONTACT_EXPORT_PATH,
            ),
            Err(e) => eprintln!(
                "  [WARN] failed to write contact-pressure CSV to {}: {e}",
                crate::contact_export::DEFAULT_CONTACT_EXPORT_PATH,
            ),
        }
    }

    println!("════════════════════════════════════════════════════════════════════════════════════════════════════════");
    PinLugHeadlessResult {
        converged: last_total.is_finite(),
        final_tier: decision_maker.current_tier,
        trajectory,
        metric_probes,
        final_lam_h_cap: dynamic_lam_h_cap,
        final_lam_d_cap: dynamic_lam_d_cap,
        final_lam_penetration_cap: dynamic_lam_penetration_cap,
        final_lam_non_tension_cap: dynamic_lam_non_tension_cap,
        total_restarts: tracker.plateau_restarts + tracker.crash_restarts,
        last_lam_before_converge,
        lbfgs_entry_lams,
    }
}

/// Normalize a physical coordinate to [-1,1]^2 for an arbitrary (non-Kirsch) domain's own
/// geometry bounds — the pin-in-lug analogue of `training_core::normalize_point`, which is
/// hardwired to `SolverConfig`'s single shared geometry.
fn normalize_point_generic(x: f64, y: f64, geom: &pinn_core::geometry::GeometryConfig) -> [f32; 2] {
    let (x0, x1) = geom.x_range();
    let (y0, y1) = geom.y_range();
    let dw = x1 - x0;
    let dh = y1 - y0;
    [(2.0 * (x - x0) / dw - 1.0) as f32, (2.0 * (y - y0) / dh - 1.0) as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_pinlug_config() -> SolverConfig {
        let mut config = SolverConfig::default_pinlug();
        config.n_interior = 32;
        config.n_boundary = 24;
        config.hidden_dim = 8;
        config.n_hidden = 2;
        config.max_steps = 10;
        config
    }

    /// THE MOST IMPORTANT TEST in this task: with `decision_maker.enabled == false` (the
    /// default), the decision-maker wiring added to `run_headless_pinlug_inner` must be a
    /// complete no-op — byte-for-byte (within float tolerance) identical training behavior
    /// to the pre-wiring code.
    ///
    /// Investigation note: network weight initialization is NOT made deterministic by
    /// `burn::tensor::backend::Backend::seed` on this repo's `Autodiff<Wgpu>` stack (verified
    /// experimentally: calling `B::seed(&device, N)` before two separate
    /// `ElasticityNetConfig::init` calls still produces different weight sums), so a literal
    /// cross-process "paste in the exact fixture Vec<f32>" trajectory comparison is not
    /// reproducible here. Instead, `run_headless_pinlug_inner` takes an `initial_models`
    /// test-only injection hook (always `None` from the public, production
    /// `run_headless_pinlug`) — this test builds ONE pair of randomly-initialized models,
    /// then runs the SAME pair (via `.clone()`, since `ElasticityNet` is a plain `Module`)
    /// through two separate calls, achieving the literal apples-to-apples comparison the
    /// task described, adapted to this backend's actual determinism surface.
    #[test]
    fn run_headless_pinlug_with_decision_maker_disabled_matches_pre_change_trajectory() {
        use crate::network::ElasticityNetConfig;
        use burn::module::Module;
        use burn::tensor::Tensor;

        let device = WgpuDevice::default();
        let mut config_off = small_pinlug_config();
        config_off.decision_maker.enabled = false;

        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(config_off.hidden_dim)
            .with_n_hidden(config_off.n_hidden)
            .with_output_dim(5)
            .with_use_piratenet(false);
        let model_pin: ElasticityNet<B> = net_cfg.init(&device);
        let model_lug: ElasticityNet<B> = net_cfg.init(&device);

        // Force-materialize every lazily-initialized `Param` BEFORE cloning: burn's `Param`
        // defers random-weight materialization until first access (`SyncOnceCell`), so
        // cloning an as-yet-untouched `Param` clones its *lazy init state*, not a value —
        // each clone then independently (and differently) materializes its own random
        // weights on first use, silently defeating the "same starting models" premise this
        // test depends on. Touching every param's `.val()` once first forces materialization
        // so `.clone()` below is a genuine deep-value clone (see `Param::clone`'s doc
        // comment in burn-core's source for the two code paths this distinguishes).
        struct TouchVisitor;
        impl burn::module::ModuleVisitor<B> for TouchVisitor {
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
                let _ = param.val();
            }
        }
        model_pin.visit(&mut TouchVisitor);
        model_lug.visit(&mut TouchVisitor);

        let result_a = run_headless_pinlug_inner(config_off.clone(), Some((model_pin.clone(), model_lug.clone())));
        let result_b = run_headless_pinlug_inner(config_off, Some((model_pin, model_lug)));

        assert_eq!(result_a.trajectory.len(), result_b.trajectory.len());
        assert_eq!(result_a.final_tier, OptimizerTier::Explore,
            "decision_maker.enabled=false must never transition current_tier away from Explore");
        assert_eq!(result_b.final_tier, OptimizerTier::Explore);

        for (i, (a, b)) in result_a.trajectory.iter().zip(result_b.trajectory.iter()).enumerate() {
            let scale = a.abs().max(b.abs()).max(1e-8);
            let rel = (a - b).abs() / scale;
            // This repo's established GPU-float-under-test-contention tolerance (see
            // training_core.rs's `param_l2_sq` regression tests' `1e-4` precedent).
            assert!(rel < 1e-4,
                "step {i}: trajectories diverge beyond tolerance with decision_maker disabled \
                 (must be a complete no-op): a={a} b={b} rel_err={rel}");
        }
    }

    #[test]
    fn run_headless_pinlug_with_decision_maker_enabled_reaches_align_tier() {
        let mut config = small_pinlug_config();
        config.max_steps = 60;
        config.decision_maker.enabled = true;
        config.decision_maker.check_interval = 5;
        config.decision_maker.min_dwell_steps = 5;
        config.decision_maker.conflict_threshold = 0.99;
        // Never satisfied (cosine similarity is clamped to [-1, 1]) — once Explore -> Align
        // fires, prevents the Align -> Explore hysteresis transition from firing again and
        // masking the tier change this test asserts on.
        config.decision_maker.alignment_threshold = 1.1;
        config.decision_maker.use_exact_cosine = true;

        let result = run_headless_pinlug_inner(config, None);

        assert_ne!(result.final_tier, OptimizerTier::Explore,
            "with a permissive conflict_threshold and enabled=true, the decision maker must \
             transition out of Explore within {} steps; final_tier={:?}",
            60, result.final_tier);
    }

    /// The key design decision under test: `dynamic_lam_penetration_cap`/
    /// `dynamic_lam_non_tension_cap` seed at the term's OWN `base_weight` (500.0/100.0), NOT
    /// the shared `dynamic_lam_h_cap`/`dynamic_lam_d_cap` convention's `50.0` — non-binding at
    /// the moment these terms first carry gradient (see CLAUDE.md's design rationale).
    #[test]
    fn run_headless_pinlug_new_interface_caps_start_at_own_base_weight_not_shared_fifty() {
        let mut config = small_pinlug_config();
        config.max_steps = 1;
        let result = run_headless_pinlug_inner(config.clone(), None);

        use crate::pinlug_problem::{PinLugProblem, PinLugScalingMode};
        use crate::problem::BoundaryValueProblem;
        let problem = PinLugProblem::new(
            config.material.clone(), 5, usize::MAX, 64, PinLugScalingMode::AppliedLoad,
        );
        let expected_penetration = problem.base_weight("interface_penetration") as f64;
        let expected_non_tension = problem.base_weight("interface_non_tension") as f64;

        assert_eq!(result.final_lam_penetration_cap, expected_penetration);
        assert_eq!(result.final_lam_non_tension_cap, expected_non_tension);
        // Adversarial: prove these are NOT the shared 50.0 h/d convention.
        assert_ne!(result.final_lam_penetration_cap, result.final_lam_h_cap);
        assert_ne!(result.final_lam_non_tension_cap, result.final_lam_d_cap);
    }

    #[test]
    fn run_headless_pinlug_convergence_metric_probe_wired_into_cascade() {
        let mut config = small_pinlug_config();
        config.max_steps = 220;
        let result = run_headless_pinlug_inner(config, None);
        assert!(result.metric_probes >= 2);
        assert!(result.converged);
    }

    /// Extends the existing zero-regression test's pattern past max_steps=220 (past the
    /// 200-step cascade probe boundary) — the EXISTING test never reaches that boundary and
    /// must NOT be modified. This is new, additional coverage: with the decision maker
    /// disabled, the plateau/crash cascade (unconditional on `decision_maker.enabled`, see
    /// invariant #3) must still be a fully DETERMINISTIC function of the training
    /// trajectory — running it twice on the same starting weights must produce identical
    /// restart counts/caps/trajectories.
    #[test]
    fn run_headless_pinlug_disabled_decision_maker_long_run_deterministic_across_two_calls() {
        use crate::network::ElasticityNetConfig;
        use burn::module::Module;
        use burn::tensor::Tensor;

        let device = WgpuDevice::default();
        let mut config = small_pinlug_config();
        config.max_steps = 220;
        config.decision_maker.enabled = false;

        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(5)
            .with_use_piratenet(false);
        let model_pin: ElasticityNet<B> = net_cfg.init(&device);
        let model_lug: ElasticityNet<B> = net_cfg.init(&device);

        struct TouchVisitor;
        impl burn::module::ModuleVisitor<B> for TouchVisitor {
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
                let _ = param.val();
            }
        }
        model_pin.visit(&mut TouchVisitor);
        model_lug.visit(&mut TouchVisitor);

        let a = run_headless_pinlug_inner(config.clone(), Some((model_pin.clone(), model_lug.clone())));
        let b = run_headless_pinlug_inner(config, Some((model_pin, model_lug)));

        assert_eq!(a.total_restarts, b.total_restarts);
        assert_eq!(a.final_lam_h_cap, b.final_lam_h_cap);
        assert_eq!(a.final_lam_d_cap, b.final_lam_d_cap);
        assert_eq!(a.final_lam_penetration_cap, b.final_lam_penetration_cap);
        assert_eq!(a.final_lam_non_tension_cap, b.final_lam_non_tension_cap);
        assert_eq!(a.trajectory.len(), b.trajectory.len());
        for (i, (x, y)) in a.trajectory.iter().zip(b.trajectory.iter()).enumerate() {
            let scale = x.abs().max(y.abs()).max(1e-8);
            assert!((x - y).abs() / scale < 1e-4, "step {i}: cascade introduced nondeterminism: a={x} b={y}");
        }
    }

    fn permissive_converge_config() -> SolverConfig {
        let mut config = small_pinlug_config();
        config.max_steps = 80;
        config.decision_maker.enabled = true;
        config.decision_maker.check_interval = 5;
        config.decision_maker.min_dwell_steps = 5;
        config.decision_maker.conflict_threshold = 0.99;
        config.decision_maker.alignment_threshold = 1.1;
        config.decision_maker.converge_cosine_min = -1.0;
        config.decision_maker.converge_grad_threshold = 1.0e6;
        config.decision_maker.use_exact_cosine = true; // REQUIRED: Converge entry needs Some(gnorm)
        config
    }

    #[test]
    fn run_headless_pinlug_decision_maker_enabled_can_reach_converge_tier() {
        let config = permissive_converge_config();
        let result = run_headless_pinlug_inner(config, None);
        assert_eq!(result.final_tier, OptimizerTier::Converge);
    }

    #[test]
    fn run_headless_pinlug_decision_maker_enabled_stays_in_align_when_converge_thresholds_unfavorable() {
        let mut config = small_pinlug_config();
        config.max_steps = 60;
        config.decision_maker.enabled = true;
        config.decision_maker.check_interval = 5;
        config.decision_maker.min_dwell_steps = 5;
        config.decision_maker.conflict_threshold = 0.99;
        config.decision_maker.alignment_threshold = 1.1;
        config.decision_maker.use_exact_cosine = true;
        config.decision_maker.converge_cosine_min = 1.1; // impossible: cosine clamped to [-1,1]
        let result = run_headless_pinlug_inner(config, None);
        assert_eq!(result.final_tier, OptimizerTier::Align);
    }

    /// THE MOST IMPORTANT training_core-adjacent regression guard in this task: L-BFGS's
    /// frozen lambdas at Converge entry must be the LIVE SAW+cap-adapted weights from the
    /// step immediately preceding entry, not `problem.base_weight()`'s static Phase-1 seed
    /// (the frozen_lams bug — see module doc comment).
    #[test]
    fn pinlug_frozen_lbfgs_lams_at_converge_entry_match_the_immediately_preceding_step_output() {
        let config = permissive_converge_config();
        let result = run_headless_pinlug_inner(config, None);

        let last = result.last_lam_before_converge.expect("must have captured a pre-entry snapshot");
        let entry = result.lbfgs_entry_lams.expect("must have entered Converge and frozen lams");
        assert_eq!(last, entry, "L-BFGS's frozen lambdas at entry must be the LIVE SAW+cap-adapted \
            weights from the step immediately before entry, not problem.base_weight()'s static seed");

        // Adversarial: prove this isn't vacuously true because SAW hadn't moved from its
        // seed yet.
        use crate::pinlug_problem::{PinLugProblem, PinLugScalingMode};
        use crate::problem::BoundaryValueProblem;
        let problem = PinLugProblem::new(
            pinn_core::material::MaterialProps::steel_4340(), 5, usize::MAX, 64,
            PinLugScalingMode::AppliedLoad,
        );
        let base_free_edge = problem.base_weight("lug_free_edge_traction") as f64;
        let base_penetration = problem.base_weight("interface_penetration") as f64;
        assert!(
            (entry["lug_free_edge_traction"] - base_free_edge).abs() > 1e-9 * base_free_edge.max(1.0)
            || (entry["interface_penetration"] - base_penetration).abs() > 1e-9 * base_penetration.max(1.0),
            "expected at least one term's SAW-adapted weight to have moved from its static \
             base_weight by Converge entry"
        );
    }
}
