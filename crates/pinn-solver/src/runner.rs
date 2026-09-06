use std::collections::HashMap;

use burn::{
    module::AutodiffModule,
    tensor::Tensor,
};
use crossbeam_channel::{Receiver, Sender};
use ndarray::Array2;
use pinn_core::{
    amr::AdaptiveGrid,
    messages::{ControlMsg, SolverConfig, TrainingMsg, TrainingUpdate, VisFields},
    problem_spec::ProblemSpec,
    sampling::{sample_boundary, sample_interior, sample_eq_ring},
};

use crate::{
    bc::apply_dirichlet_ansatz,
    controllers::ConvergenceTracker,
    decision_maker::{OptimizerTier, PinnDecisionMaker},
    engine::EngineParams,
    energy::dem_energy_per_point,
    fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig},
    kirsch_problem::KirschProblem,
    network::{fwd, ElasticityNet, ElasticityNetConfig},
    optim::{make_bias_optim, make_gate_optim, BiasOptim, GateOptim, WeightOptim},
    problem::validate_loss_terms,
    saw_brdr::SawBrdr,
    stiffness::StiffnessController,
    lr_schedule::LrSchedule,
    training_core::{
        build_gathered_boundary_tensors, compute_gradient_conflict, compute_reference_scales,
        extract_boundary_indices, make_lbfgs, normalize_point, probe_kt_shared, step_lbfgs,
        step_physics, GatheredBoundaryTensors, LbfgsCtxScalars, StepCtx, StepOutput, B, BDevice,
        BInner,
    },
};

/// Build the `KirschProblem` driving `step_physics` for the given (post-`apply_to`) config
/// + engine. Rebuilt on warm-start since material/expected_kt can change.
fn make_kirsch_problem(config: &SolverConfig, engine: &EngineParams) -> KirschProblem {
    let problem = KirschProblem::new(
        config.material.clone(), engine.output_dim(), engine.phase1_steps, engine.expected_kt,
    );
    validate_loss_terms(&problem);
    problem
}

// ─── Training state ────────────────────────────────────────────────────────────

/// All mutable state threaded through the training loop, bundled so warm-start doesn't
/// need a 20+ parameter function signature — every field here is either replaced wholesale
/// or recomputed together whenever the problem (load/geometry/material) changes.
struct TrainingState {
    model: ElasticityNet<B>,
    optim_w: WeightOptim,
    optim_b: BiasOptim,
    optim_gate: GateOptim,
    stiffness_controller: StiffnessController,

    u_ref: f32,
    ref_energy: f32,
    ref_stress2: f32,

    saw: SawBrdr,
    lr_sched: LrSchedule,
    phase2_started: bool,
    tracker: ConvergenceTracker,
    dynamic_lam_h_cap: f64,
    dynamic_lam_d_cap: f64,

    int_pts_phys: Vec<[f64; 2]>,
    amr: Option<AdaptiveGrid>,

    /// Cache of normalized interior points — recomputed only when `int_pts_phys` changes
    /// (Phase 2 start, AMR event, or warm-start), not every step. Mirrors
    /// `headless.rs::run_headless`'s identically-named cache, which exists to avoid
    /// ~14 000 `Vec` allocations per run; this GUI-driving path previously lacked the
    /// same optimization despite having the exact same invalidation conditions.
    int_norm: Vec<[f32; 2]>,
    int_pts_dirty: bool,

    bnd_norm: Vec<[f32; 2]>,
    bnd_nx: Vec<f32>,
    bnd_ny: Vec<f32>,
    bnd_tx: Vec<f32>,
    bnd_ty: Vec<f32>,
    trac_idx: Vec<usize>,
    hole_idx: Vec<usize>,
    right_idx: Vec<usize>,
    /// The 6 `trac_idx`/`hole_idx`-gathered boundary tensors `step_physics`/
    /// `compute_gradient_conflict` need every step — rebuilt only in `new()`/`warm_start()`
    /// (when `bnd_nx`/`bnd_ny`/`bnd_tx`/`bnd_ty`/`trac_idx`/`hole_idx` actually change), not
    /// once per step. See `GatheredBoundaryTensors`'s doc comment.
    gathered: GatheredBoundaryTensors,

    eq_ring_norm: Vec<[f32; 2]>,
    vis_pts_norm: Vec<[f32; 2]>,
    vis_mask: Vec<bool>,

    current_config: SolverConfig,
    current_engine: EngineParams,
    current_k: f32,
    current_fd: FdConfig,
    current_cx: f64,
    current_cy: f64,
    current_ref_div2: f64,

    decision_maker:      PinnDecisionMaker,
    lbfgs_opt:           Option<burn::optim::LBFGS<B>>,
    frozen_lbfgs_ctx:    Option<LbfgsCtxScalars>,
    frozen_lbfgs_lams:   Option<HashMap<&'static str, f64>>,

    /// Boundary-value problem driving `step_physics`'s loss-term set/order/base-weights.
    problem: KirschProblem,
}

impl TrainingState {
    fn new(config: &SolverConfig, engine: &EngineParams, net_cfg: &ElasticityNetConfig, device: &BDevice) -> Self {
        let (x0, x1) = config.geometry.x_range();
        let (y0, y1) = config.geometry.y_range();
        let fd = FdConfig::new(config.fd_h, x1 - x0, y1 - y0);
        let cx = fd.sx / (2.0 * fd.hx as f64);
        let cy = fd.sy / (2.0 * fd.hy as f64);
        let ref_div2 = (config.load.px * cx).powi(2).max(1.0);

        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(config);

        let bnd_pts = sample_boundary(&config.geometry, &config.load, config.n_boundary);
        let bnd_norm: Vec<[f32; 2]> = bnd_pts.iter().map(|b| normalize_point(b.x, b.y, config)).collect();
        let bnd_nx: Vec<f32> = bnd_pts.iter().map(|b| b.nx as f32).collect();
        let bnd_ny: Vec<f32> = bnd_pts.iter().map(|b| b.ny as f32).collect();
        let bnd_tx: Vec<f32> = bnd_pts.iter().map(|b| b.tx as f32).collect();
        let bnd_ty: Vec<f32> = bnd_pts.iter().map(|b| b.ty as f32).collect();
        let (trac_idx, hole_idx, right_idx) = extract_boundary_indices(&bnd_pts, &bnd_nx);
        let gathered = build_gathered_boundary_tensors(
            &trac_idx, &hole_idx, &bnd_nx, &bnd_ny, &bnd_tx, &bnd_ty, device,
        );

        let eq_ring_norm: Vec<[f32; 2]> = sample_eq_ring(&config.geometry, engine.n_eq_ring)
            .iter().map(|&[x, y]| normalize_point(x, y, config)).collect();

        let (vis_pts_norm, vis_mask) = build_vis_grid(config);

        let dm_cfg   = config.decision_maker.clone();
        let dm       = PinnDecisionMaker::new(dm_cfg, false, false);
        let optim_w  = WeightOptim::from_tier(config.use_soap_muon, &dm.current_tier);
        let int_pts_phys = sample_interior(&config.geometry, engine.phase1_n_interior);
        let int_norm: Vec<[f32; 2]> = int_pts_phys.iter()
            .map(|&[x, y]| normalize_point(x, y, config)).collect();
        Self {
            model: net_cfg.init(device),
            optim_w,
            optim_b: make_bias_optim(),
            optim_gate: make_gate_optim(),
            stiffness_controller: StiffnessController::new(config.stiffness.clone()),
            u_ref, ref_energy, ref_stress2,
            saw: SawBrdr::with_base(engine.init_weights(), 0.95),
            lr_sched: LrSchedule::new(engine.peak_lr, 200, 1000),
            phase2_started: false,
            tracker: ConvergenceTracker::new(),
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            int_pts_phys,
            amr: None,
            int_norm,
            int_pts_dirty: false,
            bnd_norm, bnd_nx, bnd_ny, bnd_tx, bnd_ty,
            trac_idx, hole_idx, right_idx, gathered,
            eq_ring_norm, vis_pts_norm, vis_mask,
            current_config: config.clone(),
            current_engine: engine.clone(),
            current_k: engine.ansatz_k,
            current_fd: fd,
            current_cx: cx,
            current_cy: cy,
            current_ref_div2: ref_div2,
            decision_maker:    dm,
            lbfgs_opt:         None,
            frozen_lbfgs_ctx:  None,
            frozen_lbfgs_lams: None,
            problem:           make_kirsch_problem(config, engine),
        }
    }

    /// Reconstruct all three optimizers from scratch and reset the decision maker.
    /// Used on warm-start, phase transition, and convergence-cascade restarts.
    fn reset_optimizers(&mut self, use_soap_muon: bool) {
        self.optim_w = WeightOptim::from_tier(use_soap_muon, &self.decision_maker.current_tier);
        self.optim_b = make_bias_optim();
        self.optim_gate = make_gate_optim();
    }

    /// Clear all L-BFGS / Converge-tier state.
    fn clear_lbfgs(&mut self) {
        self.lbfgs_opt = None;
        self.frozen_lbfgs_ctx = None;
        self.frozen_lbfgs_lams = None;
    }

    /// Apply a warm-start: resample the problem from `new_cfg`, reset training progress,
    /// and (if `geometry_changed`) reinitialize the model and visualization grid.
    fn warm_start(
        &mut self,
        new_cfg: SolverConfig,
        net_cfg: &ElasticityNetConfig,
        device: &BDevice,
        geometry_changed: bool,
    ) {
        let new_engine = EngineParams::analyze(&new_cfg);

        if geometry_changed {
            self.model = net_cfg.init(device);
            self.saw.set_base_weights(new_engine.init_weights());
        }

        self.int_pts_phys = sample_interior(&new_cfg.geometry, new_engine.phase1_n_interior);
        // int_pts_phys just changed (new geometry/n_interior) — invalidate the int_norm
        // cache; the main loop lazily recomputes it before the next use.
        self.int_pts_dirty = true;

        let bnd_pts = sample_boundary(&new_cfg.geometry, &new_cfg.load, new_cfg.n_boundary);
        self.bnd_norm = bnd_pts.iter().map(|b| normalize_point(b.x, b.y, &new_cfg)).collect();
        self.bnd_nx   = bnd_pts.iter().map(|b| b.nx as f32).collect();
        self.bnd_ny   = bnd_pts.iter().map(|b| b.ny as f32).collect();
        self.bnd_tx   = bnd_pts.iter().map(|b| b.tx as f32).collect();
        self.bnd_ty   = bnd_pts.iter().map(|b| b.ty as f32).collect();
        (self.trac_idx, self.hole_idx, self.right_idx) = extract_boundary_indices(&bnd_pts, &self.bnd_nx);
        self.gathered = build_gathered_boundary_tensors(
            &self.trac_idx, &self.hole_idx, &self.bnd_nx, &self.bnd_ny, &self.bnd_tx, &self.bnd_ty, device,
        );

        self.eq_ring_norm = sample_eq_ring(&new_cfg.geometry, new_engine.n_eq_ring)
            .iter().map(|&[x, y]| normalize_point(x, y, &new_cfg)).collect();

        (self.u_ref, self.ref_energy, self.ref_stress2) = compute_reference_scales(&new_cfg);

        if geometry_changed {
            (self.vis_pts_norm, self.vis_mask) = build_vis_grid(&new_cfg);
        }

        self.phase2_started = false;
        self.amr = None;
        self.dynamic_lam_h_cap = 50.0;
        self.dynamic_lam_d_cap = 50.0;
        self.tracker = ConvergenceTracker::new();

        self.saw.reset();
        self.lr_sched.reset_for_warmstart();
        self.decision_maker = PinnDecisionMaker::new(new_cfg.decision_maker.clone(), false, false);
        self.stiffness_controller = StiffnessController::new(new_cfg.stiffness.clone());
        self.clear_lbfgs();
        self.reset_optimizers(new_cfg.use_soap_muon);

        // Recompute the FD/equilibrium scales derived from the new geometry/load — was
        // previously the caller's responsibility at both warm-start call sites.
        let (x0, x1) = new_cfg.geometry.x_range();
        let (y0, y1) = new_cfg.geometry.y_range();
        self.current_fd = FdConfig::new(new_cfg.fd_h, x1 - x0, y1 - y0);
        self.current_cx = self.current_fd.sx / (2.0 * self.current_fd.hx as f64);
        self.current_cy = self.current_fd.sy / (2.0 * self.current_fd.hy as f64);
        self.current_ref_div2 = (new_cfg.load.px * self.current_cx).powi(2).max(1.0);
        self.current_k = new_engine.ansatz_k;
        self.problem = make_kirsch_problem(&new_cfg, &new_engine);
        self.current_engine = new_engine;
        self.current_config = new_cfg;
    }
}

// ─── Control-message handling ──────────────────────────────────────────────────

enum ControlAction {
    /// No control message (or an irrelevant one) — proceed with this step normally.
    Continue,
    WarmStart { config: SolverConfig, geometry_changed: bool },
    /// Top-level Stop: break the training loop, but still send `TrainingMsg::Done` after.
    StopAndFinish,
    /// Stop received while paused (or the control channel disconnected while paused):
    /// return immediately without sending `Done`, matching the original behavior.
    StopImmediately,
}

/// Check for a pending control message, blocking (via `stop_rx.recv()`) while paused
/// until Resume, Stop, or WarmStart arrives. Mirrors the original inline match exactly,
/// including the Stop-while-paused vs. top-level-Stop distinction.
fn handle_control_messages(stop_rx: &Receiver<ControlMsg>) -> ControlAction {
    match stop_rx.try_recv() {
        Ok(ControlMsg::Stop) => ControlAction::StopAndFinish,
        Ok(ControlMsg::Pause) => loop {
            match stop_rx.recv() {
                Ok(ControlMsg::Resume) => return ControlAction::Continue,
                Ok(ControlMsg::Stop) => return ControlAction::StopImmediately,
                Ok(ControlMsg::WarmStart { config, geometry_changed }) =>
                    return ControlAction::WarmStart { config, geometry_changed },
                Err(_) => return ControlAction::StopImmediately,
                _ => {}
            }
        },
        Ok(ControlMsg::WarmStart { config, geometry_changed }) =>
            ControlAction::WarmStart { config, geometry_changed },
        // Nothing to export on the single-domain Kirsch path this function drives — a no-op
        // that just proceeds with the current step, same as no control message at all.
        Ok(ControlMsg::ExportContactPressure) => ControlAction::Continue,
        _ => ControlAction::Continue,
    }
}

// ─── Main training loop ─────────────────────────────────────────────────────────

pub fn run_training(
    config: SolverConfig,
    tx: Sender<TrainingMsg>,
    stop_rx: Receiver<ControlMsg>,
) {
    let mut config = config;
    let engine = EngineParams::analyze(&config);
    engine.apply_to(&mut config);

    let device = BDevice::default();
    let net_cfg = ElasticityNetConfig::new()
        .with_input_dim(engine.net_input_dim())
        .with_hidden_dim(config.hidden_dim)
        .with_n_hidden(config.n_hidden)
        .with_output_dim(engine.output_dim())
        .with_use_piratenet(config.use_piratenet);
    let use_soap_muon = config.use_soap_muon;
    let dm_config = config.decision_maker.clone();
    let stiff_config = config.stiffness.clone();

    let mut state = TrainingState::new(&config, &engine, &net_cfg, &device);
    let [nx_vis, ny_vis] = config.vis_grid;

    for step in 0..config.max_steps {
        match handle_control_messages(&stop_rx) {
            ControlAction::StopAndFinish => break,
            ControlAction::StopImmediately => return,
            ControlAction::WarmStart { config: new_cfg, geometry_changed } => {
                state.warm_start(new_cfg, &net_cfg, &device, geometry_changed);
            }
            ControlAction::Continue => {}
        }

        // === Phase transition: replace SAW (5→6 components), activate kirsch + AMR ===
        if step == state.current_engine.phase1_steps && state.current_engine.phase1_steps > 0 && !state.phase2_started {
            state.phase2_started = true;
            state.saw = SawBrdr::with_base(state.current_engine.init_weights_phase2(), 0.95);
            let grid = AdaptiveGrid::new(&state.current_config.geometry, state.current_engine.amr.clone());
            state.int_pts_phys = grid.sample_points();
            state.int_pts_dirty = true;
            state.amr = Some(grid);
            state.lr_sched.reset_for_phase2();
            state.decision_maker = PinnDecisionMaker::new(dm_config.clone(), true, false);
            state.stiffness_controller = StiffnessController::new(stiff_config.clone());
            state.clear_lbfgs();
            state.reset_optimizers(use_soap_muon);
        }

        // === AMR sweep ===
        if step > state.current_engine.phase1_steps
            && (step - state.current_engine.phase1_steps) % state.current_engine.amr.interval_steps == 0
        {
            if let Some(ref mut grid) = state.amr {
                // Use cached int_norm for residual computation (already up-to-date here:
                // phase-transition and AMR sweeps are mutually exclusive on a given step, so
                // int_pts_phys hasn't changed yet this iteration when this block runs).
                let n_amr = state.int_norm.len();
                let model_val: ElasticityNet<BInner> = state.model.valid();
                let amr_residuals: Vec<f32> = {
                    let pts_t = norm_pts_to_tensor::<BInner>(&state.int_norm, &device);
                    let stencil = assemble_stencil::<BInner>(&pts_t, &state.current_fd, &device);
                    let out = apply_dirichlet_ansatz::<BInner>(
                        fwd::<BInner>(&model_val, stencil.clone(), state.current_engine.n_fourier, &device),
                        &stencil,
                        state.current_config.geometry.symmetry, state.current_k,
                    ).mul_scalar(state.u_ref as f64);
                    let (exx, eyy, exy) = compute_strains::<BInner>(out, n_amr, &state.current_fd);
                    dem_energy_per_point::<BInner>(exx, eyy, exy, &state.current_config.material)
                        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_amr])
                        .into_iter().map(|e| e.abs()).collect()
                };
                grid.update_residuals(&amr_residuals);
                grid.adapt();
                let amr_stats = grid.stats();
                state.int_pts_phys = grid.sample_points();
                state.int_pts_dirty = true;
                println!("  [AMR@{step}] cells={} depth={} mean_res={:.3e} pts={}",
                    amr_stats.active_count, amr_stats.max_depth,
                    amr_stats.mean_residual, state.int_pts_phys.len());
                if state.decision_maker.current_tier == OptimizerTier::Converge {
                    println!("  [DM@{step}] AMR → demote Converge→Align");
                    state.decision_maker = PinnDecisionMaker::new(dm_config.clone(), true, false);
                    state.clear_lbfgs();
                    state.reset_optimizers(use_soap_muon);
                }
            }
        }

        // === Training step ===
        // Recompute cached int_norm only when int_pts_phys changed (Phase 2 start, AMR, or
        // warm-start) — mirrors headless.rs's identical dirty-flag pattern.
        if state.int_pts_dirty {
            state.int_norm = state.int_pts_phys.iter()
                .map(|&[x, y]| normalize_point(x, y, &state.current_config)).collect();
            state.int_pts_dirty = false;
        }

        let ctx = StepCtx {
            config:            &state.current_config,
            engine:            &state.current_engine,
            problem:           &state.problem,
            fd:                &state.current_fd,
            k:                 state.current_k,
            u_ref:             state.u_ref,
            ref_energy:        state.ref_energy,
            ref_stress2:       state.ref_stress2,
            cx:                state.current_cx,
            cy:                state.current_cy,
            ref_div2:          state.current_ref_div2,
            int_norm:          &state.int_norm,  // cached; recomputed only on AMR/phase-change/warm-start
            bnd_norm:          &state.bnd_norm,
            bnd_nx:            &state.bnd_nx,
            bnd_ny:            &state.bnd_ny,
            bnd_tx:            &state.bnd_tx,
            bnd_ty:            &state.bnd_ty,
            trac_idx:          &state.trac_idx,
            hole_idx:          &state.hole_idx,
            right_idx:         &state.right_idx,
            gathered:          &state.gathered,
            eq_ring_norm:      &state.eq_ring_norm,
            dynamic_lam_h_cap: state.dynamic_lam_h_cap,
            dynamic_lam_d_cap: state.dynamic_lam_d_cap,
            phase2_active:     state.phase2_started,
            step,
        };
        let (new_model, mut out) = if dm_config.enabled
            && state.decision_maker.current_tier == OptimizerTier::Converge
        {
            if state.frozen_lbfgs_ctx.is_none() {
                state.frozen_lbfgs_ctx = Some(LbfgsCtxScalars::from_ctx(&ctx));
            }
            let lbfgs = state.lbfgs_opt
                .get_or_insert_with(|| make_lbfgs(dm_config.lbfgs_max_iter));
            let lams = state.frozen_lbfgs_lams.as_ref()
                .expect("lams must be set when entering Converge");
            let fctx = state.frozen_lbfgs_ctx.as_ref().unwrap();
            let lr = state.lr_sched.current_lr();
            let (new_m, loss_f64) = step_lbfgs(state.model, lbfgs, lr, fctx, &state.problem, lams, &device);
            let synthetic = StepOutput {
                e_scalar: 0.0, n_scalar: 0.0, h_scalar: 0.0, d_scalar: 0.0,
                eq_scalar: 0.0, w_scalar: 0.0, kirsch_scalar: 0.0, const_scalar: 0.0,
                total_scalar: loss_f64 as f32, lr,
                lam_e: 0.0, lam_n: 0.0, lam_h: 0.0, lam_d: 0.0, lam_eq: 0.0, lam_kirsch: 0.0,
                proxy_ratio: 0.0,
                optimizer_tier: OptimizerTier::Converge.as_u8(),
                cosine_sim: None,
                lam_by_name: None,
                timing: None,
            };
            (new_m, synthetic)
        } else {
            let physics_boost = state.stiffness_controller.physics_boost();
            let alpha_lr_mult = state.stiffness_controller.alpha_lr_mult();
            step_physics(
                state.model, &mut state.optim_w, &mut state.optim_b, &mut state.optim_gate,
                &ctx, &mut state.saw, &mut state.lr_sched, &device,
                state.decision_maker.current_tier.as_u8(),
                physics_boost, alpha_lr_mult,
            )
        };
        state.model = new_model;

        // Decision maker / stiffness controller gate — both `advance()` unconditionally
        // (never short-circuited) so their internal step counters stay correct regardless
        // of whether the other subsystem is enabled. At most one GradientConflict is
        // computed per step, shared by whichever subsystem's gate fired.
        let dm_fire    = state.decision_maker.advance();
        let stiff_fire = state.stiffness_controller.advance();
        if dm_fire || stiff_fire {
            let want_conflict = state.decision_maker.current_tier != OptimizerTier::Converge
                && ((dm_config.use_exact_cosine && dm_fire) || stiff_fire);
            let conflict = if want_conflict {
                Some(compute_gradient_conflict(&state.model, &ctx, step, &device))
            } else {
                None
            };
            out.cosine_sim = conflict.map(|c| c.cosine_sim);

            if dm_fire {
                if let Some(t) = state.decision_maker.evaluate(conflict, out.proxy_ratio, state.phase2_started) {
                    let new_tier_name = match t.new_tier {
                        OptimizerTier::Explore  => "Explore",
                        OptimizerTier::Align    => "Align",
                        OptimizerTier::Converge => "Converge",
                    };
                    println!("\n  [DM@{step}] → {new_tier_name}");
                    if t.reset_optim {
                        state.optim_w = WeightOptim::from_tier(use_soap_muon, &t.new_tier);
                        state.optim_b = make_bias_optim();
                    }
                    if t.reset_lr { state.lr_sched.reset_for_phase2(); }
                    match t.new_tier {
                        OptimizerTier::Converge => {
                            // Keyed by the exact `loss_terms()` names — `lam_const` drops
                            // out of this map entirely: `compute_loss_for_lbfgs` now reads
                            // `ctx.engine.lam_const` directly, same as `step_physics` does.
                            state.frozen_lbfgs_ctx  = Some(LbfgsCtxScalars::from_ctx(&ctx));
                            state.frozen_lbfgs_lams = Some(HashMap::from([
                                ("interior_energy", out.lam_e),
                                ("neumann_traction", out.lam_n),
                                ("hole_traction", out.lam_h),
                                ("displacement_anchor", out.lam_d),
                                ("equilibrium_ring", out.lam_eq),
                                ("kirsch_stress", out.lam_kirsch),
                            ]));
                            state.lbfgs_opt = None;
                        }
                        _ => state.clear_lbfgs(),
                    }
                }
            }

            if stiff_fire {
                if let Some(c) = conflict {
                    let factor = state.stiffness_controller.update(&c);
                    println!("\n  [Stiffness@{step}] factor={factor:.3} boost={:.2} gate_lr_mult={:.2}",
                        state.stiffness_controller.physics_boost(), state.stiffness_controller.alpha_lr_mult());
                }
            }
        }

        // === Visualisation + K_t probe + convergence check every 50 steps ===
        if step % 50 == 0 {
            let model_val: ElasticityNet<BInner> = state.model.valid();
            let vis = evaluate_vis_grid(
                &model_val, &state.vis_pts_norm, &state.vis_mask, &state.current_fd, &state.current_config,
                [nx_vis, ny_vis], &device, state.u_ref, state.current_k, state.current_engine.n_fourier,
            );
            let kt = probe_kt_shared(
                &model_val, &state.current_config, &state.current_engine,
                &state.current_fd, state.current_k, state.u_ref, &device,
            );

            if state.phase2_started {
                if let Some(kt_val) = kt {
                    state.tracker.push(kt_val as f64);

                    if let Some(new_cap) = state.tracker.check_kt_crash(kt_val as f64) {
                        state.dynamic_lam_h_cap = new_cap;
                        state.dynamic_lam_d_cap = new_cap;
                        state.tracker.clear_history();
                        state.saw.reset();
                        state.lr_sched.reset_for_phase2();
                        state.decision_maker = PinnDecisionMaker::new(dm_config.clone(), true, false);
                        state.stiffness_controller = StiffnessController::new(stiff_config.clone());
                        state.clear_lbfgs();
                        state.reset_optimizers(use_soap_muon);
                        println!("\n  [CRASH RECOVERY #{}] K_t={kt_val:.3} collapsed → restart, lam_caps→{new_cap:.0}",
                            state.tracker.total_restarts());
                    } else if let Some(new_cap) = state.tracker.check_plateau() {
                        state.dynamic_lam_h_cap = new_cap;
                        state.dynamic_lam_d_cap = new_cap;
                        state.saw.reset();
                        state.lr_sched.reset_for_phase2();
                        state.decision_maker = PinnDecisionMaker::new(dm_config.clone(), true, false);
                        state.stiffness_controller = StiffnessController::new(stiff_config.clone());
                        state.clear_lbfgs();
                        state.reset_optimizers(use_soap_muon);
                        println!("\n  [WARM RESTART #{}] K_t plateau → reset: lr+adam+SAW, lam_caps→{new_cap:.0}",
                            state.tracker.total_restarts());
                    }

                    if state.tracker.is_kt_converged(state.current_engine.expected_kt) {
                        let _ = tx.send(TrainingMsg::Done);
                        return;
                    }
                }
            }

            let update = TrainingUpdate {
                step,
                total_loss:   out.total_scalar,
                energy_loss:  out.e_scalar,
                neumann_loss: out.n_scalar,
                lr:           out.lr as f32,
                lam_energy:   out.lam_e as f32,
                lam_neumann:  out.lam_n as f32,
                n_colloc:     state.int_pts_phys.len(),
                kt_estimate:  kt,
                vis:          Some(vis),
            };
            let _ = tx.try_send(TrainingMsg::Update(Box::new(update)));
        }
    }

    let _ = tx.send(TrainingMsg::Done);
}

// ─── Pin-in-lug (GUI-driving) training loop ────────────────────────────────────

/// GUI-driving analog of [`run_headless_pinlug`](crate::headless::run_headless_pinlug) — same
/// two-domain `PinLugProblem` / `step_physics_multi` training loop, but wired to the
/// `TrainingMsg`/`ControlMsg` channel pair instead of stdout, matching `run_training`'s
/// zero-stdout-I/O convention.
///
/// Scope cuts (see `run_headless_pinlug`'s doc comment for the shared rationale — none of
/// Kirsch's AMR / decision-maker / stiffness-controller / warm-restart-cascade machinery
/// applies to a contact problem without a closed-form K_t):
/// - `ControlMsg::WarmStart` here only honors the tunable scalar fields pin-lug's config
///   actually reads (`max_steps`, `n_interior`, `n_boundary`, `hidden_dim`, `n_hidden`,
///   `use_soap_muon`) rather than doing a full two-domain resample/reinit. A full pin-lug
///   warm-restart (resampling both domains' interior/boundary/interface point sets and
///   reinitializing both networks) is explicitly OUT OF SCOPE for this slice.
/// - `energy_loss`/`neumann_loss` on the emitted `PinLugTrainingUpdate` are documented
///   approximations, not exact per-term sums: `step_physics_multi`'s `StepOutput` scalar
///   fields (`e_scalar`/`n_scalar`/...) are keyed to Kirsch's single-domain term names
///   ("interior_energy", "neumann_traction", ...), which never match pin-lug's actual term
///   names ("pin_interior_energy", "lug_shank_anchor", ...) — so those fields are always
///   0.0 for this path. `energy_loss` is reported as `e_scalar + eq_scalar` (0.0 today) and
///   `neumann_loss` as `total_scalar - energy_loss`, i.e. "everything else" — an honest
///   approximation given `step_physics_multi`'s current generic-name gap, not a re-derived
///   per-term breakdown (fixing that gap belongs to `training_core.rs`, outside this slice).
pub fn run_training_pinlug(
    config: SolverConfig,
    tx: Sender<TrainingMsg>,
    stop_rx: Receiver<ControlMsg>,
) {
    use pinn_core::messages::{PinLugTrainingUpdate, PinLugVisFields};
    use crate::{
        pinlug_problem::{PinLugProblem, PinLugScalingMode, LUG_DOMAIN, PIN_DOMAIN},
        problem::{
            BoundaryValueProblem, DomainOptim, DomainState, DomainStepCtx, DomainStepData,
            MultiStepCtx, PointSetData,
        },
        training_core::step_physics_multi,
    };

    const N_INTERFACE: usize = 64;
    const OUTPUT_DIM: usize = 5; // mDEM (u, v, sxx, syy, sxy)
    const PHASE1_STEPS: usize = usize::MAX; // no phase-2 cascade for pin-in-lug

    let mut config = config;
    let device = BDevice::default();

    let mut problem = PinLugProblem::new(
        config.material.clone(), OUTPUT_DIM, PHASE1_STEPS, N_INTERFACE,
        if config.use_ultimate_strength_scaling { PinLugScalingMode::UltimateStrength } else { PinLugScalingMode::AppliedLoad },
    );
    validate_loss_terms(&problem);

    let pin_geom = problem.domains()[0].geometry.clone();
    let lug_geom = problem.domains()[1].geometry.clone();
    let lug_fd = FdConfig::new(config.fd_h, 2.0 * lug_geom.half_w, 2.0 * lug_geom.half_h);
    let mut fd = lug_fd;

    let net_cfg = |hidden_dim: usize, n_hidden: usize| {
        ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(hidden_dim)
            .with_n_hidden(n_hidden)
            .with_output_dim(OUTPUT_DIM)
            .with_use_piratenet(config.use_piratenet)
    };
    let mut model_pin: ElasticityNet<B> = net_cfg(config.hidden_dim, config.n_hidden).init(&device);
    let mut model_lug: ElasticityNet<B> = net_cfg(config.hidden_dim, config.n_hidden).init(&device);

    let mut optims = vec![
        DomainOptim { weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim() },
        DomainOptim { weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim() },
    ];

    let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
    let mut saw = SawBrdr::with_base(base_weights, 0.95);
    let mut lr_sched = LrSchedule::new(1e-3, 200, 1000);

    let mut equiv_traction = problem.equivalent_traction_pa();
    let mut e = config.material.e;
    let mut u_ref = ((equiv_traction / e) * lug_geom.half_w) as f32;
    let mut ref_energy = (0.5 * equiv_traction * equiv_traction / e) as f32;
    let mut ref_stress2 = (equiv_traction * equiv_traction) as f32;

    let mut pin_geom = pin_geom;
    let mut lug_geom = lug_geom;

    let build_pointset = |pts: &[pinn_core::loading::BoundaryPoint], geom: &pinn_core::geometry::GeometryConfig| -> PointSetData {
        PointSetData {
            norm: pts.iter().map(|p| normalize_point_generic_pinlug(p.x, p.y, geom)).collect(),
            nx: pts.iter().map(|p| p.nx as f32).collect(),
            ny: pts.iter().map(|p| p.ny as f32).collect(),
            tx: pts.iter().map(|p| p.tx as f32).collect(),
            ty: pts.iter().map(|p| p.ty as f32).collect(),
        }
    };

    let mut last_total = f32::MAX;

    for step in 0..config.max_steps {
        match handle_control_messages_pinlug(&stop_rx) {
            PinLugControlAction::StopAndFinish => break,
            PinLugControlAction::StopImmediately => return,
            PinLugControlAction::ExportContactPressure => {
                let model_lug_val: ElasticityNet<BInner> = model_lug.valid();
                match crate::contact_export::export_contact_pressure::<BInner>(
                    &model_lug_val, &lug_geom, config.load.px, &device,
                ) {
                    Ok(_) => {
                        let _ = tx.send(TrainingMsg::ExportComplete(
                            crate::contact_export::DEFAULT_CONTACT_EXPORT_PATH.to_string(),
                        ));
                    }
                    Err(e) => {
                        let _ = tx.send(TrainingMsg::Error(format!(
                            "failed to write contact-pressure CSV to {}: {e}",
                            crate::contact_export::DEFAULT_CONTACT_EXPORT_PATH,
                        )));
                    }
                }
            }
            PinLugControlAction::WarmStart { config: new_cfg } => {
                // Scope cut (see doc comment): only honor tunable scalar fields, no full
                // two-domain resample/reinit.
                config.max_steps = new_cfg.max_steps;
                config.n_interior = new_cfg.n_interior;
                config.n_boundary = new_cfg.n_boundary;
                config.hidden_dim = new_cfg.hidden_dim;
                config.n_hidden = new_cfg.n_hidden;
                config.use_soap_muon = new_cfg.use_soap_muon;

                problem = PinLugProblem::new(
                    config.material.clone(), OUTPUT_DIM, PHASE1_STEPS, N_INTERFACE,
                    if config.use_ultimate_strength_scaling { PinLugScalingMode::UltimateStrength } else { PinLugScalingMode::AppliedLoad },
                );
                validate_loss_terms(&problem);
                pin_geom = problem.domains()[0].geometry.clone();
                lug_geom = problem.domains()[1].geometry.clone();
                fd = FdConfig::new(config.fd_h, 2.0 * lug_geom.half_w, 2.0 * lug_geom.half_h);
                model_pin = net_cfg(config.hidden_dim, config.n_hidden).init(&device);
                model_lug = net_cfg(config.hidden_dim, config.n_hidden).init(&device);
                optims = vec![
                    DomainOptim { weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim() },
                    DomainOptim { weight: WeightOptim::new(config.use_soap_muon), bias: make_bias_optim(), gate: make_gate_optim() },
                ];
                let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
                saw = SawBrdr::with_base(base_weights, 0.95);
                lr_sched = LrSchedule::new(1e-3, 200, 1000);
                equiv_traction = problem.equivalent_traction_pa();
                e = config.material.e;
                u_ref = ((equiv_traction / e) * lug_geom.half_w) as f32;
                ref_energy = (0.5 * equiv_traction * equiv_traction / e) as f32;
                ref_stress2 = (equiv_traction * equiv_traction) as f32;
            }
            PinLugControlAction::Continue => {}
        }

        let pin_sampling = problem.sampling_strategy(0);
        let lug_sampling = problem.sampling_strategy(1);

        let pin_int = pin_sampling.sample_interior(&pin_geom, config.n_interior);
        let lug_int = lug_sampling.sample_interior(&lug_geom, config.n_interior);
        let lug_bnd = lug_sampling.sample_boundary(&lug_geom, &config.load, config.n_boundary);

        let pin_int_norm: Vec<[f32; 2]> = pin_int.iter().map(|&[x, y]| normalize_point_generic_pinlug(x, y, &pin_geom)).collect();
        let lug_int_norm: Vec<[f32; 2]> = lug_int.iter().map(|&[x, y]| normalize_point_generic_pinlug(x, y, &lug_geom)).collect();
        let n_colloc = pin_int_norm.len() + lug_int_norm.len();

        // Sized to the known closed set of names each domain's `named_point_sets()` populates
        // (pin: "interface" + optionally "driving"; lug: "interface" + "shank_anchor", plus
        // "boundary" inserted below) — avoids the incremental resize/rehash `HashMap::new()`
        // would otherwise pay as entries are inserted one at a time, every step.
        let mut pin_named = std::collections::HashMap::with_capacity(2);
        let mut lug_named = std::collections::HashMap::with_capacity(3);
        for set in pin_sampling.named_point_sets(&[]) {
            pin_named.insert(set.name, build_pointset(&set.points, &pin_geom));
        }
        for set in lug_sampling.named_point_sets(&[]) {
            lug_named.insert(set.name, build_pointset(&set.points, &lug_geom));
        }
        lug_named.insert("boundary", build_pointset(&lug_bnd, &lug_geom));
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
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            // `phase2_active: false` below makes these two caps fully inert (same as the
            // pre-existing h/d caps on this GUI-driving path) — see `run_headless_pinlug_
            // inner`'s equivalent construction for the wired-up cascade version. No cascade
            // logic here; a deliberate, tracked scope cut (see CLAUDE.md's GUI section).
            dynamic_lam_penetration_cap: 500.0,
            dynamic_lam_non_tension_cap: 100.0,
            phase2_active: false,
            step,
        };

        let (new_models, out) = step_physics_multi(
            vec![model_pin, model_lug], &mut optims, &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );
        let mut it = new_models.into_iter();
        model_pin = it.next().unwrap();
        model_lug = it.next().unwrap();
        last_total = out.total_scalar;

        if step % 50 == 0 {
            let model_pin_val: ElasticityNet<BInner> = model_pin.valid();
            let model_lug_val: ElasticityNet<BInner> = model_lug.valid();

            let [nx_vis, ny_vis] = config.vis_grid;
            let pin_vis = evaluate_vis_grid_mdem(
                &model_pin_val, &pin_geom, [nx_vis, ny_vis], &fd, u_ref, config.load.px, &device,
            );
            let lug_vis = evaluate_vis_grid_mdem(
                &model_lug_val, &lug_geom, [nx_vis, ny_vis], &fd, u_ref, config.load.px, &device,
            );

            // Interface-gap RMS via PinLugProblem::convergence_metric — needs autodiff-typed
            // (B) model clones, not the inference-only `.valid()` views above.
            let state = vec![
                DomainState { id: PIN_DOMAIN, model: model_pin.clone(), u_ref, ref_energy, ref_stress2 },
                DomainState { id: LUG_DOMAIN, model: model_lug.clone(), u_ref, ref_energy, ref_stress2 },
            ];
            let convergence_metric = problem.convergence_metric(&state).map(|v| v as f32);

            let energy_loss = out.e_scalar + out.eq_scalar;
            let neumann_loss = out.total_scalar - energy_loss;

            let update = PinLugTrainingUpdate {
                step,
                total_loss: out.total_scalar,
                energy_loss,
                neumann_loss,
                lr: out.lr as f32,
                lam_energy: out.lam_e as f32,
                lam_neumann: out.lam_n as f32,
                n_colloc,
                convergence_metric,
                vis: Some(PinLugVisFields { pin: pin_vis, lug: lug_vis }),
            };
            let _ = tx.try_send(TrainingMsg::PinLugUpdate(Box::new(update)));
        }
    }

    let _ = last_total;
    let _ = tx.send(TrainingMsg::Done);
}

/// GUI-facing runner for a [`crate::user_problem::UserDefinedProblem`] — streams the same
/// `TrainingMsg::Update(Box<TrainingUpdate>)` variant `run_training` sends (not a new
/// variant, and not `PinLugUpdate`), since `TrainingUpdate`'s fields are already generic
/// enough (see `energy_loss`/`neumann_loss`'s role-based names, not Kirsch-specific ones) and
/// this problem is single-domain like Kirsch, not two-domain like pin-lug.
///
/// `neumann_loss = out.total_scalar - out.e_scalar` mirrors `run_training_pinlug`'s own
/// exact convention (`let neumann_loss = out.total_scalar - energy_loss;`) for aggregating
/// an arbitrary number of differently-named BC terms into one number, without needing this
/// problem's own term names (`outer_traction`/`hole_free`/`hole_fixed`) to match any of
/// `StepOutput`'s hardcoded per-name accessors. `out.e_scalar` DOES already match directly —
/// this problem's energy term is named `"interior_energy"`, the same name `StepOutput::
/// e_scalar` looks up for every problem.
///
/// Deliberately no curriculum/decision-maker/AMR (matches `user_runner::
/// run_headless_user_problem`'s same v1 scope cut) — plain constant/scheduled-LR AdamW via
/// `step_physics_multi`, `WarmStart` control messages are accepted but ignored (a loaded
/// `ProblemSpec` isn't a `SolverConfig` there is anything scalar to warm-start into).
pub fn run_training_user_problem(
    spec: ProblemSpec,
    tx: Sender<TrainingMsg>,
    stop_rx: Receiver<ControlMsg>,
) {
    use crate::problem::{
        BoundaryValueProblem, DomainOptim, DomainStepCtx, DomainStepData, MultiStepCtx, PointSetData,
    };
    use crate::training_core::step_physics_multi;
    use crate::user_problem::{evaluate_user_vis_grid, UserDefinedProblem, USER_DOMAIN};

    let device = BDevice::default();
    let half_w = spec.geometry.half_w;
    let half_h = spec.geometry.half_h;

    let problem = UserDefinedProblem::new(spec.clone());
    validate_loss_terms(&problem);

    let mut config = SolverConfig::default_kirsch();
    config.load = spec.load;

    let net_cfg = ElasticityNetConfig::new()
        .with_input_dim(3)
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

    let placeholder_geom = pinn_core::geometry::GeometryConfig::kirsch_plate_inches(); // ignored by UserSamplingStrategy
    let [nx_vis, ny_vis] = config.vis_grid;

    for step in 0..spec.training.max_steps {
        match handle_control_messages(&stop_rx) {
            ControlAction::StopAndFinish => break,
            ControlAction::StopImmediately => return,
            ControlAction::WarmStart { .. } => {} // no-op: see this function's doc comment
            ControlAction::Continue => {}
        }

        let sampling = problem.sampling_strategy(0);
        let int_pts = sampling.sample_interior(&placeholder_geom, spec.training.n_interior);
        let bnd_pts = sampling.sample_boundary(&placeholder_geom, &spec.load, spec.training.n_boundary);
        let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / half_w) as f32, (y / half_h) as f32] };
        let int_norm: Vec<[f32; 2]> = int_pts.iter().map(|&[x, y]| norm_pt(x, y)).collect();
        let to_pointset = |pts: &[pinn_core::loading::BoundaryPoint]| -> PointSetData {
            PointSetData {
                norm: pts.iter().map(|p| norm_pt(p.x, p.y)).collect(),
                nx: pts.iter().map(|p| p.nx as f32).collect(),
                ny: pts.iter().map(|p| p.ny as f32).collect(),
                tx: pts.iter().map(|p| p.tx as f32).collect(),
                ty: pts.iter().map(|p| p.ty as f32).collect(),
            }
        };

        let mut named = HashMap::with_capacity(1 + spec.geometry.holes.len());
        named.insert("outer_boundary", to_pointset(&bnd_pts));
        for set in sampling.named_point_sets(&[]) {
            named.insert(set.name, to_pointset(&set.points));
        }

        let data = DomainStepData { id: USER_DOMAIN, int_norm, extra_ring_norm: Vec::new(), named };
        let ctx = MultiStepCtx {
            config: &config,
            problem: &problem,
            fd: &fd,
            k: 1.0, // IdentityAnsatz ignores k entirely
            domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
            dynamic_lam_h_cap: f64::MAX,
            dynamic_lam_d_cap: f64::MAX,
            dynamic_lam_penetration_cap: f64::MAX,
            dynamic_lam_non_tension_cap: f64::MAX,
            phase2_active: true,
            step,
        };

        let (new_model, out) = step_physics_multi(
            vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, &device,
            0, 1.0, 1.0,
        );
        model = new_model.into_iter().next().unwrap();

        if step % 10 == 0 || step + 1 == spec.training.max_steps {
            let model_val: ElasticityNet<BInner> = model.valid();
            let vis = evaluate_user_vis_grid(&model_val, &spec.geometry, [nx_vis, ny_vis], u_ref, spec.load.px, &device);

            let energy_loss = out.e_scalar;
            let neumann_loss = out.total_scalar - energy_loss;
            let update = TrainingUpdate {
                step,
                total_loss: out.total_scalar,
                energy_loss,
                neumann_loss,
                lr: out.lr as f32,
                lam_energy: out.lam_e as f32,
                lam_neumann: 0.0, // no single generic BC lambda exists for an arbitrary term set
                n_colloc: spec.training.n_interior,
                kt_estimate: None,
                vis: Some(vis),
            };
            let _ = tx.try_send(TrainingMsg::Update(Box::new(update)));
        }
    }

    let _ = tx.send(TrainingMsg::Done);
}

/// `ControlAction` analogue for `run_training_pinlug` — unlike the Kirsch `ControlAction`,
/// `WarmStart` here only carries the new `SolverConfig` (no `geometry_changed` flag: this
/// slice's warm-start scope cut always does a scalar-only update, see the doc comment on
/// `run_training_pinlug`), and there's a dedicated `ExportContactPressure` action instead of
/// treating it as a no-op `Continue` (that no-op behavior is specific to the Kirsch path).
enum PinLugControlAction {
    Continue,
    WarmStart { config: SolverConfig },
    ExportContactPressure,
    StopAndFinish,
    StopImmediately,
}

/// Pin-in-lug's own control-message handler — mirrors `handle_control_messages`'s
/// Stop/Pause/Resume/WarmStart semantics exactly, but additionally acts on
/// `ControlMsg::ExportContactPressure` (a no-op on the Kirsch path) by returning a dedicated
/// action the caller uses to trigger a CSV export.
fn handle_control_messages_pinlug(stop_rx: &Receiver<ControlMsg>) -> PinLugControlAction {
    match stop_rx.try_recv() {
        Ok(ControlMsg::Stop) => PinLugControlAction::StopAndFinish,
        Ok(ControlMsg::Pause) => loop {
            match stop_rx.recv() {
                Ok(ControlMsg::Resume) => return PinLugControlAction::Continue,
                Ok(ControlMsg::Stop) => return PinLugControlAction::StopImmediately,
                Ok(ControlMsg::WarmStart { config, .. }) =>
                    return PinLugControlAction::WarmStart { config },
                Ok(ControlMsg::ExportContactPressure) =>
                    return PinLugControlAction::ExportContactPressure,
                Err(_) => return PinLugControlAction::StopImmediately,
                _ => {}
            }
        },
        Ok(ControlMsg::WarmStart { config, .. }) => PinLugControlAction::WarmStart { config },
        Ok(ControlMsg::ExportContactPressure) => PinLugControlAction::ExportContactPressure,
        _ => PinLugControlAction::Continue,
    }
}

/// Normalize a physical coordinate to [-1,1]^2 for an arbitrary (non-Kirsch) domain's own
/// geometry bounds. Duplicated (deliberately, matching `headless::normalize_point_generic`'s
/// own doc comment rationale) rather than making that helper `pub` across modules for one
/// small, well-understood formula.
fn normalize_point_generic_pinlug(x: f64, y: f64, geom: &pinn_core::geometry::GeometryConfig) -> [f32; 2] {
    let (x0, x1) = geom.x_range();
    let (y0, y1) = geom.y_range();
    let dw = x1 - x0;
    let dh = y1 - y0;
    [(2.0 * (x - x0) / dw - 1.0) as f32, (2.0 * (y - y0) / dh - 1.0) as f32]
}

/// Visualization-grid evaluator for a single plain-mDEM pin-lug domain (identity ansatz, raw
/// output columns `[u, v, sxx, syy, sxy]`) — the pin-lug analogue of `evaluate_vis_grid`,
/// which is Kirsch-ansatz-specific (`apply_dirichlet_ansatz`) and therefore not reusable
/// here. `px_pa` is the same mDEM stress-column scale `step_physics_multi`/`contact_export`
/// use (`config.load.px`).
#[allow(clippy::too_many_arguments)]
fn evaluate_vis_grid_mdem(
    model:    &ElasticityNet<BInner>,
    geom:     &pinn_core::geometry::GeometryConfig,
    [nx, ny]: [usize; 2],
    fd:       &FdConfig,
    u_ref:    f32,
    px_pa:    f64,
    device:   &BDevice,
) -> VisFields {
    let n_total = nx * ny;
    let mut pts = Vec::with_capacity(n_total);
    let mut mask = Vec::with_capacity(n_total);
    for iy in 0..ny {
        for ix in 0..nx {
            let xn = -1.0 + 2.0 * ix as f64 / (nx.max(2) - 1) as f64;
            let yn = -1.0 + 2.0 * iy as f64 / (ny.max(2) - 1) as f64;
            pts.push([xn as f32, yn as f32]);
            let (xp, yp) = geom.denormalize(xn, yn);
            mask.push(geom.contains(xp, yp));
        }
    }

    let mut s_vm = vec![f32::NAN; n_total];
    let mut s_xx = vec![f32::NAN; n_total];
    let mut s_yy = vec![f32::NAN; n_total];
    let mut s_xy = vec![f32::NAN; n_total];
    let mut d_u  = vec![f32::NAN; n_total];
    let mut d_v  = vec![f32::NAN; n_total];

    let active: Vec<usize> = mask.iter().enumerate().filter(|(_, &m)| m).map(|(i, _)| i).collect();
    if active.is_empty() { return make_vis(nx, ny, s_vm, s_xx, s_yy, s_xy, d_u, d_v); }

    let active_pts: Vec<[f32; 2]> = active.iter().map(|&i| pts[i]).collect();
    let n_act = active_pts.len();

    let pts_t = norm_pts_to_tensor::<BInner>(&active_pts, device);
    const N_FOURIER: usize = 0;
    let raw = fwd::<BInner>(model, pts_t, N_FOURIER, device);

    // Batch all 5 field columns into a single `Tensor::cat` + ONE `.into_data()` GPU sync
    // instead of 5 separate syncs (each pays a fixed wgpu queue-flush/buffer-map cost
    // independent of payload size).
    let u_col   = raw.clone().slice([0..n_act, 0..1]).reshape([n_act]);
    let v_col   = raw.clone().slice([0..n_act, 1..2]).reshape([n_act]);
    let sxx_col = raw.clone().slice([0..n_act, 2..3]).reshape([n_act]);
    let syy_col = raw.clone().slice([0..n_act, 3..4]).reshape([n_act]);
    let sxy_col = raw.slice([0..n_act, 4..5]).reshape([n_act]);
    let batched: Vec<f32> = Tensor::cat(vec![u_col, v_col, sxx_col, syy_col, sxy_col], 0)
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; 5 * n_act]);
    let u_vals   = &batched[..n_act];
    let v_vals   = &batched[n_act..2 * n_act];
    let sxx_vals = &batched[2 * n_act..3 * n_act];
    let syy_vals = &batched[3 * n_act..4 * n_act];
    let sxy_vals = &batched[4 * n_act..5 * n_act];

    let _ = fd; // fd is not needed by the plain-mDEM identity-ansatz path (no FD stencil/strains)

    for (i_act, &i_full) in active.iter().enumerate() {
        let u   = u_vals[i_act] * u_ref;
        let v   = v_vals[i_act] * u_ref;
        let sxx = sxx_vals[i_act] as f64 * px_pa;
        let syy = syy_vals[i_act] as f64 * px_pa;
        let sxy = sxy_vals[i_act] as f64 * px_pa;
        let vm  = (sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy).sqrt();
        s_xx[i_full] = sxx as f32; s_yy[i_full] = syy as f32; s_xy[i_full] = sxy as f32;
        s_vm[i_full] = vm as f32;  d_u[i_full]  = u; d_v[i_full] = v;
    }
    make_vis(nx, ny, s_vm, s_xx, s_yy, s_xy, d_u, d_v)
}

// ─── Visualisation grid ───────────────────────────────────────────────────────

/// Build the normalized [-1,1]² visualization grid and its in-domain mask for `config`.
/// Used at initial setup and again on warm-start when geometry changes.
fn build_vis_grid(config: &SolverConfig) -> (Vec<[f32; 2]>, Vec<bool>) {
    let [nx_vis, ny_vis] = config.vis_grid;
    let mut pts = Vec::with_capacity(nx_vis * ny_vis);
    for iy in 0..ny_vis {
        for ix in 0..nx_vis {
            let xn = -1.0 + 2.0 * ix as f64 / (nx_vis.max(2) - 1) as f64;
            let yn = -1.0 + 2.0 * iy as f64 / (ny_vis.max(2) - 1) as f64;
            pts.push([xn as f32, yn as f32]);
        }
    }
    let mask = pts.iter().map(|&[xn, yn]| {
        let (xp, yp) = config.geometry.denormalize(xn as f64, yn as f64);
        config.geometry.contains(xp, yp)
    }).collect();
    (pts, mask)
}

fn evaluate_vis_grid(
    model:     &ElasticityNet<BInner>,
    pts_norm:  &[[f32; 2]],
    mask:      &[bool],
    fd:        &FdConfig,
    cfg:       &SolverConfig,
    [nx, ny]:  [usize; 2],
    device:    &BDevice,
    u_ref:     f32,
    k:         f32,
    n_fourier: usize,
) -> VisFields {
    let n_total = nx * ny;
    assert_eq!(pts_norm.len(), n_total);
    assert_eq!(mask.len(), n_total);

    let active: Vec<usize> = mask.iter().enumerate()
        .filter(|(_, &m)| m).map(|(i, _)| i).collect();

    let mut s_vm = vec![f32::NAN; n_total];
    let mut s_xx = vec![f32::NAN; n_total];
    let mut s_yy = vec![f32::NAN; n_total];
    let mut s_xy = vec![f32::NAN; n_total];
    let mut d_u  = vec![f32::NAN; n_total];
    let mut d_v  = vec![f32::NAN; n_total];

    if active.is_empty() { return make_vis(nx, ny, s_vm, s_xx, s_yy, s_xy, d_u, d_v); }

    let active_pts: Vec<[f32; 2]> = active.iter().map(|&i| pts_norm[i]).collect();
    let n_act = active_pts.len();

    let pts_t = norm_pts_to_tensor::<BInner>(&active_pts, device);
    let stencil_coords = assemble_stencil::<BInner>(&pts_t, fd, device);
    let out = apply_dirichlet_ansatz::<BInner>(
        fwd::<BInner>(model, stencil_coords.clone(), n_fourier, device),
        &stencil_coords, cfg.geometry.symmetry, k,
    ).mul_scalar(u_ref as f64);

    // `u_col`/`v_col` read from `out` and `eps_xx`/`eps_yy`/`eps_xy` (computed from `out` via
    // `compute_strains`, independent of any host read of `u_col`/`v_col`) are batched into a
    // single `Tensor::cat` + ONE `.into_data()` GPU sync instead of 5 separate syncs.
    let u_col = out.clone().slice([0..n_act, 0..1]).reshape([n_act]);
    let v_col = out.clone().slice([0..n_act, 1..2]).reshape([n_act]);

    let (eps_xx, eps_yy, eps_xy) = compute_strains::<BInner>(out, n_act, fd);
    let e  = cfg.material.e  as f32;
    let nu = cfg.material.nu as f32;
    let f  = e / (1.0 - nu * nu);

    let batched: Vec<f32> = Tensor::cat(vec![u_col, v_col, eps_xx, eps_yy, eps_xy], 0)
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; 5 * n_act]);
    let u_vals = &batched[..n_act];
    let v_vals = &batched[n_act..2 * n_act];
    let exx_v  = &batched[2 * n_act..3 * n_act];
    let eyy_v  = &batched[3 * n_act..4 * n_act];
    let exy_v  = &batched[4 * n_act..5 * n_act];

    for (i_act, &i_full) in active.iter().enumerate() {
        let exx = exx_v[i_act]; let eyy = eyy_v[i_act]; let exy = exy_v[i_act];
        let sxx = f * (exx + nu * eyy);
        let syy = f * (eyy + nu * exx);
        let sxy = f / (1.0 + nu) * exy;
        let vm  = (sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy).sqrt();
        s_xx[i_full] = sxx; s_yy[i_full] = syy; s_xy[i_full] = sxy;
        s_vm[i_full] = vm;  d_u[i_full]  = u_vals[i_act]; d_v[i_full] = v_vals[i_act];
    }
    make_vis(nx, ny, s_vm, s_xx, s_yy, s_xy, d_u, d_v)
}

fn make_vis(nx: usize, ny: usize,
            vm: Vec<f32>, sxx: Vec<f32>, syy: Vec<f32>,
            sxy: Vec<f32>, u: Vec<f32>, v: Vec<f32>) -> VisFields {
    let a = |v: Vec<f32>| Array2::from_shape_vec((ny, nx), v).expect("shape mismatch");
    VisFields {
        von_mises: a(vm), sigma_xx: a(sxx), sigma_yy: a(syy), sigma_xy: a(sxy),
        disp_u: a(u), disp_v: a(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_kirsch_config() -> SolverConfig {
        let mut cfg = SolverConfig::default_kirsch();
        cfg.max_steps = 12;
        cfg.n_interior = 32;
        cfg.n_boundary = 16;
        cfg.hidden_dim = 8;
        cfg.n_hidden = 2;
        cfg.vis_grid = [4, 4];
        cfg
    }

    #[test]
    fn build_vis_grid_handles_single_column_grid_without_nan() {
        let mut config = tiny_kirsch_config();
        config.vis_grid = [1, 4];
        let (pts, mask) = build_vis_grid(&config);
        assert_eq!(pts.len(), 4);
        assert_eq!(mask.len(), 4);
        for &[xn, yn] in &pts {
            assert!(xn.is_finite(), "xn must be finite when nx_vis==1, got {xn}");
            assert!(yn.is_finite(), "yn must be finite when nx_vis==1, got {yn}");
            assert_eq!(xn, -1.0, "single-column grid (nx_vis==1) must place every point at xn=-1.0, got {xn}");
        }
    }

    #[test]
    fn build_vis_grid_handles_single_row_grid_without_nan() {
        let mut config = tiny_kirsch_config();
        config.vis_grid = [4, 1];
        let (pts, mask) = build_vis_grid(&config);
        assert_eq!(pts.len(), 4);
        assert_eq!(mask.len(), 4);
        for &[xn, yn] in &pts {
            assert!(xn.is_finite());
            assert!(yn.is_finite());
            assert_eq!(yn, -1.0, "single-row grid (ny_vis==1) must place every point at yn=-1.0, got {yn}");
        }
    }

    fn tiny_pinlug_config() -> SolverConfig {
        let mut cfg = SolverConfig::default_pinlug();
        cfg.max_steps = 12;
        cfg.n_interior = 16;
        cfg.n_boundary = 8;
        cfg.hidden_dim = 8;
        cfg.n_hidden = 2;
        cfg.vis_grid = [4, 4];
        cfg
    }

    /// Drains `rx` on the calling thread while `run_training`/`run_training_pinlug` runs on a
    /// background thread — required because the channel is `bounded(1)` (matching
    /// `pinn-gui`'s real wiring) and the trainer's final `tx.send(TrainingMsg::Done)` blocks
    /// if the channel is full and nobody is reading concurrently.
    ///
    /// The 60s per-message timeout is tuned to Wgpu (GPU-dispatched) throughput. Under the
    /// `ndarray-backend` feature (CPU-only, and dramatically slower again in an unoptimized
    /// `cargo test` debug build with no per-op JIT/shader caching) a full tiny-config training
    /// run can legitimately exceed 60s end-to-end even though it never hangs — so the
    /// ndarray-backend build uses a much longer allowance here. This is a test-harness-only
    /// constant; nothing production-facing has a timeout.
    #[cfg(not(feature = "ndarray-backend"))]
    const RECV_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
    #[cfg(feature = "ndarray-backend")]
    const RECV_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

    fn run_and_drain<F>(train: F) -> Vec<TrainingMsg>
    where
        F: FnOnce(Sender<TrainingMsg>) + Send + 'static,
    {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let handle = std::thread::spawn(move || train(tx));
        let mut msgs = Vec::new();
        while let Ok(msg) = rx.recv_timeout(RECV_TIMEOUT) {
            let is_done = matches!(msg, TrainingMsg::Done);
            msgs.push(msg);
            if is_done { break; }
        }
        handle.join().expect("training thread must not panic");
        msgs
    }

    /// Build a `TrainingState` for a tiny Kirsch config, mirroring `run_training`'s own setup.
    fn tiny_training_state() -> (TrainingState, SolverConfig, EngineParams, ElasticityNetConfig, BDevice) {
        let mut config = tiny_kirsch_config();
        let engine = EngineParams::analyze(&config);
        engine.apply_to(&mut config);
        let device = BDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(engine.net_input_dim())
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(config.use_piratenet);
        let state = TrainingState::new(&config, &engine, &net_cfg, &device);
        (state, config, engine, net_cfg, device)
    }

    #[test]
    fn training_state_new_starts_with_clean_int_norm_cache_matching_int_pts_phys() {
        let (state, config, _engine, _net_cfg, _device) = tiny_training_state();
        assert!(!state.int_pts_dirty, "cache must start clean — new() eagerly computes int_norm");
        assert_eq!(state.int_norm.len(), state.int_pts_phys.len());
        let expected: Vec<[f32; 2]> = state.int_pts_phys.iter()
            .map(|&[x, y]| normalize_point(x, y, &config)).collect();
        assert_eq!(state.int_norm, expected, "cached int_norm must match int_pts_phys under normalize_point");
    }

    #[test]
    fn training_state_warm_start_marks_int_norm_cache_dirty() {
        let (mut state, config, _engine, net_cfg, device) = tiny_training_state();
        assert!(!state.int_pts_dirty);
        state.warm_start(config, &net_cfg, &device, false);
        assert!(state.int_pts_dirty, "warm_start changes int_pts_phys and must invalidate the int_norm cache");
    }

    #[test]
    fn training_state_phase2_start_and_amr_both_mark_int_norm_cache_dirty() {
        // Direct proof that the two headless.rs-mirrored invalidation triggers are wired:
        // Phase-2 start (`AdaptiveGrid::new` + resample) and an AMR sweep (`grid.adapt()` +
        // resample) each replace int_pts_phys and must set int_pts_dirty — asserted here at
        // the same call sites `run_training`'s loop uses, without needing to drive an actual
        // multi-thousand-step run to reach them.
        let (mut state, config, engine, _net_cfg, _device) = tiny_training_state();
        state.int_pts_dirty = false; // simulate "just recomputed" (as after new()/a lazy recompute)

        let grid = AdaptiveGrid::new(&config.geometry, engine.amr.clone());
        state.int_pts_phys = grid.sample_points();
        state.int_pts_dirty = true; // mirrors the phase-transition block in run_training
        assert!(state.int_pts_dirty, "phase-2 start must invalidate the int_norm cache");

        // Recompute (as the loop would before next use), then simulate an AMR event.
        state.int_norm = state.int_pts_phys.iter()
            .map(|&[x, y]| normalize_point(x, y, &config)).collect();
        state.int_pts_dirty = false;
        state.amr = Some(grid);
        if let Some(ref mut g) = state.amr {
            g.adapt();
            state.int_pts_phys = g.sample_points();
            state.int_pts_dirty = true; // mirrors the AMR-sweep block in run_training
        }
        assert!(state.int_pts_dirty, "an AMR sweep must invalidate the int_norm cache");
    }

    #[test]
    fn run_training_signature_is_unchanged() {
        let _f: fn(SolverConfig, crossbeam_channel::Sender<pinn_core::messages::TrainingMsg>,
                   crossbeam_channel::Receiver<pinn_core::messages::ControlMsg>) = run_training;
    }

    #[test]
    fn handle_control_messages_treats_export_contact_pressure_as_continue() {
        let (tx, rx) = crossbeam_channel::unbounded();
        tx.send(ControlMsg::ExportContactPressure).unwrap();
        let action = handle_control_messages(&rx);
        assert!(matches!(action, ControlAction::Continue));
    }

    #[test]
    fn run_training_kirsch_path_still_sends_update_variant_with_vis_and_kt_fields() {
        let config = tiny_kirsch_config();
        let (_stop_tx, stop_rx) = crossbeam_channel::unbounded();
        let msgs = run_and_drain(move |tx| run_training(config, tx, stop_rx));

        assert!(msgs.iter().any(|m| matches!(m, TrainingMsg::Done)), "must send Done");
        let mut saw_update = false;
        for m in &msgs {
            if let TrainingMsg::Update(u) = m {
                saw_update = true;
                assert!(u.total_loss.is_finite(), "total_loss must be finite, got {}", u.total_loss);
                assert!(u.vis.is_some(), "Kirsch Update must carry Some(vis)");
            }
            assert!(!matches!(m, TrainingMsg::PinLugUpdate(_)), "Kirsch path must never send PinLugUpdate");
        }
        assert!(saw_update, "expected at least one TrainingMsg::Update");
    }

    #[test]
    fn run_training_pinlug_completes_without_panic_and_sends_done() {
        let config = tiny_pinlug_config();
        let [nx_vis, ny_vis] = config.vis_grid;
        let n_interior = config.n_interior;
        let (_stop_tx, stop_rx) = crossbeam_channel::unbounded();
        let msgs = run_and_drain(move |tx| run_training_pinlug(config, tx, stop_rx));

        assert!(msgs.iter().any(|m| matches!(m, TrainingMsg::Done)), "must send Done");

        let mut saw_pinlug_update = false;
        for m in &msgs {
            assert!(!matches!(m, TrainingMsg::Update(_)), "pin-lug path must never send plain Update");
            if let TrainingMsg::PinLugUpdate(u) = m {
                saw_pinlug_update = true;
                assert!(u.total_loss.is_finite(), "total_loss must be finite, got {}", u.total_loss);
                let vis = u.vis.as_ref().expect("PinLugUpdate must carry Some(vis)");
                assert_eq!(vis.pin.von_mises.dim(), (ny_vis, nx_vis));
                assert_eq!(vis.lug.von_mises.dim(), (ny_vis, nx_vis));
                // n_colloc == pin + lug interior point counts, summed (both sampled with the
                // same config.n_interior in this tiny config).
                assert_eq!(u.n_colloc, 2 * n_interior);
            }
        }
        assert!(saw_pinlug_update, "expected at least one TrainingMsg::PinLugUpdate");
    }

    /// Issue #51: `run_training_pinlug`'s `net_cfg` closure now reads `config.use_piratenet`
    /// (mirroring `run_training`'s Kirsch pattern and `headless.rs::run_headless_pinlug_inner`)
    /// instead of hardcoding `false`. GUI-path parity check: a short two-domain run with
    /// PirateNet enabled must still complete without panic and emit only finite `total_loss`
    /// values in every `PinLugUpdate`, exercising the same cross-domain Signorini terms
    /// (`interface_penetration`/`interface_non_tension`) the headless path's equivalent test
    /// covers.
    #[test]
    fn run_training_pinlug_with_piratenet_enabled_completes_with_finite_updates() {
        let mut config = tiny_pinlug_config();
        config.use_piratenet = true;
        let (_stop_tx, stop_rx) = crossbeam_channel::unbounded();
        let msgs = run_and_drain(move |tx| run_training_pinlug(config, tx, stop_rx));

        assert!(msgs.iter().any(|m| matches!(m, TrainingMsg::Done)), "must send Done");
        let mut saw_pinlug_update = false;
        for m in &msgs {
            if let TrainingMsg::PinLugUpdate(u) = m {
                saw_pinlug_update = true;
                assert!(u.total_loss.is_finite(),
                    "total_loss must be finite under PirateNet, got {}", u.total_loss);
            }
        }
        assert!(saw_pinlug_update, "expected at least one TrainingMsg::PinLugUpdate");
    }

    #[test]
    fn run_training_pinlug_stop_message_ends_loop_before_max_steps() {
        let mut config = tiny_pinlug_config();
        config.max_steps = 100_000;
        let (stop_tx, stop_rx) = crossbeam_channel::unbounded();
        stop_tx.send(ControlMsg::Stop).unwrap();

        let start = std::time::Instant::now();
        let msgs = run_and_drain(move |tx| run_training_pinlug(config, tx, stop_rx));
        assert!(start.elapsed().as_secs() < 30, "Stop must short-circuit well under 30s");
        assert!(msgs.iter().any(|m| matches!(m, TrainingMsg::Done)), "Done must still be sent after Stop");
    }

    #[test]
    fn run_training_pinlug_export_contact_pressure_writes_csv_and_reports_path() {
        let config = tiny_pinlug_config();
        let (stop_tx, stop_rx) = crossbeam_channel::unbounded();
        stop_tx.send(ControlMsg::ExportContactPressure).unwrap();
        stop_tx.send(ControlMsg::Stop).unwrap();

        let msgs = run_and_drain(move |tx| run_training_pinlug(config, tx, stop_rx));

        let export_path = msgs.iter().find_map(|m| match m {
            TrainingMsg::ExportComplete(p) => Some(p.clone()),
            _ => None,
        }).expect("expected TrainingMsg::ExportComplete");
        assert_eq!(export_path, crate::contact_export::DEFAULT_CONTACT_EXPORT_PATH);
        assert!(std::path::Path::new(&export_path).exists(), "exported CSV must exist on disk");
        let _ = std::fs::remove_file(&export_path);
    }

    #[test]
    fn run_training_pinlug_with_max_steps_one_still_emits_step_zero_update_and_done() {
        let mut config = tiny_pinlug_config();
        config.max_steps = 1;
        let (_stop_tx, stop_rx) = crossbeam_channel::unbounded();
        let msgs = run_and_drain(move |tx| run_training_pinlug(config, tx, stop_rx));

        assert!(msgs.iter().any(|m| matches!(m, TrainingMsg::Done)), "must send Done");
        assert!(
            msgs.iter().any(|m| matches!(m, TrainingMsg::PinLugUpdate(u) if u.step == 0)),
            "max_steps=1 must still hit the step%50==0 branch at step 0",
        );
    }
}
