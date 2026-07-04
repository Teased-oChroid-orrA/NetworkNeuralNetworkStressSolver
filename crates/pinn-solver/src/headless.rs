/// Headless (terminal-only) training mode with full engine integration.
///
/// Two-phase curriculum:
///   Phase 1 (0..phase1_steps): SAW-BRDR on 5 BC losses (e, n, h, d, eq). No kirsch loss.
///     Smart batch: phase1_n_interior points (<<n_interior) for 16× faster BC convergence.
///   Phase 2 (phase1_steps..): BCs converged → add kirsch_stress_loss (fixed lam_kirsch).
///     Full n_interior batch. lam_h capped at 50 so kirsch gradient dominates h gradient.
///     K_t is REPORTED from probe_kt_shared() as a verification metric — not used as a loss.
/// - No hardcoded constants — everything flows from SolverConfig via EngineParams.

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
    let mut decision_maker = PinnDecisionMaker::new(dm_config.clone(), false);
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
            decision_maker = PinnDecisionMaker::new(dm_config.clone(), true);
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
                    decision_maker = PinnDecisionMaker::new(dm_config.clone(), true);
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
                            decision_maker = PinnDecisionMaker::new(dm_config.clone(), true);
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
                            decision_maker = PinnDecisionMaker::new(dm_config.clone(), true);
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
