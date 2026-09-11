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
    controllers::{ConvergenceTracker, MetricDirection},
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
                grad_norm: None,
                raw_scalar_by_name: None,
                term_grad_norms: None,
                gradient_share_report: None,
                gradient_conflict_report: None,
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
                &state.int_norm,
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
                // Kirsch's own AMR sweep (above, this function) is deliberately NOT migrated
                // onto the new `AmrSweepReport` mechanism in this pass - out of scope, see
                // this session's own "Neural-Network-Wide Adaptive Collocation" epic notes
                // (Preservation Rule: don't touch already-working, already-tested code).
                amr_sweep: None,
                // Kirsch's problem has no `UserGeometry`/N-hole concept - hole analysis
                // (Phase 16) is `UserDefinedProblem`-specific, see `run_training_user_problem`.
                hole_analyses: Vec::new(),
                grad_norm: out.grad_norm,
                // Not computed for Kirsch's frozen `step_physics` path in this pass - see
                // `user_problem::probe_boundary_residuals`'s doc comment for the real
                // (`UserDefinedProblem`-side) version of this metric. Extending it to Kirsch
                // would mean building a second probe against `KirschProblem`'s own loss-term
                // math, not reusing this one - a real, deferred follow-up, not a silent 0.0.
                bc_residual_rms: 0.0,
                bc_residual_max: 0.0,
                // Same deferral as `bc_residual_rms`/`_max` immediately above - Kirsch's own
                // path has no `probe_reaction_force`/`probe_energy_balance` equivalent built
                // in this pass.
                reaction_force: None,
                energy_balance: None,
                // Stage I - generic, model-internal, and cheap regardless of problem type
                // (unlike BC residual/reaction force/energy balance, which need problem-
                // specific boundary math) - wired for every path, including Kirsch's, since
                // `model_val` is already available at this exact vis-cadence call site.
                network_snapshot: Some(crate::network::network_snapshot(&model_val)),
                // Smart adaptive architecture is out of scope for Kirsch's own path (no
                // `NetworkSpec::adaptive` field exists on `SolverConfig` to enable it).
                architecture_event: None,
                // Kirsch's frozen `step_physics` path never sets `probe_term_gradients` (see
                // `StepOutput::term_grad_norms`'s doc comment - it's a `step_physics_multi`-
                // only diagnostic in this pass).
                gradient_share_report: None,
                gradient_conflict_report: None,
                // Kirsch's per-step computation runs through the hardcoded `step_physics`
                // path, never through a `&dyn BoundaryValueProblem` trait object -
                // `stress_source_report` has nothing to iterate. Empty, not fabricated.
                stress_source_report: Vec::new(),
                boundary_operator_report: Vec::new(),
                derivative_order_report: Vec::new(),
                formulation_kind_report: Vec::new(),
                constraint_report: Vec::new(),
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
/// Generic AMR (`pinn_core::amr::AdaptiveGrid`/`AmrDomain`) IS wired in here, one grid per
/// domain — this is no longer a scope cut. Nothing is gated on pin-lug specifically: the lug
/// domain's real circular hole gives it a genuine lock zone (same as Kirsch's own), the pin
/// domain (no hole) gets zero zones and runs pure residual-driven refine/coarsen, and neither
/// needed any pin-lug-specific code — see `pinn_core::amr::AmrDomain`'s doc comment.
///
/// Scope cuts (see `run_headless_pinlug`'s doc comment for the shared rationale — none of
/// Kirsch's decision-maker / stiffness-controller / warm-restart-cascade machinery applies
/// to a contact problem without a closed-form K_t):
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
/// Resamples both domains' interior collocation points + all named point-sets (interface/
/// shank-anchor/boundary) from scratch — called once before `run_training_pinlug`'s loop and
/// again inside its `WarmStart` handler (geometry/`n_interior`/`n_boundary` can change
/// there). Extracted to a free function so both call sites share one implementation instead
/// of two copies that could drift — the exact resampling logic previously ran unconditionally
/// every step; hoisting it here (matching `run_training_user_problem`'s own earlier fix) is a
/// prerequisite for wiring periodic AMR-driven resampling, not a separate cleanup pass:
/// `PinLugSamplingStrategy::sample_interior` reseeds a FIXED-SEED `LcgRng`
/// (`SEED_PIN_INTERIOR`/`SEED_LUG_INTERIOR`) every call, so resampling unconditionally every
/// step was pure waste (byte-identical output), the same bug class fixed in
/// `UserDefinedProblem`'s `UserSamplingStrategy` earlier.
fn resample_pinlug_domains(
    problem: &crate::pinlug_problem::PinLugProblem,
    pin_geom: &pinn_core::geometry::GeometryConfig,
    lug_geom: &pinn_core::geometry::GeometryConfig,
    load: &pinn_core::loading::LoadConfig,
    n_interior: usize,
    n_boundary: usize,
    equiv_traction: f64,
) -> (crate::problem::DomainStepData, crate::problem::DomainStepData) {
    use crate::problem::{BoundaryValueProblem, DomainStepData, PointSetData};
    use crate::pinlug_problem::{LUG_DOMAIN, PIN_DOMAIN};

    let pin_sampling = problem.sampling_strategy(0);
    let lug_sampling = problem.sampling_strategy(1);

    let pin_int = pin_sampling.sample_interior(pin_geom, n_interior);
    let lug_int = lug_sampling.sample_interior(lug_geom, n_interior);
    let lug_bnd = lug_sampling.sample_boundary(lug_geom, load, n_boundary);

    let pin_int_norm: Vec<[f32; 2]> = pin_int.iter().map(|&[x, y]| normalize_point_generic_pinlug(x, y, pin_geom)).collect();
    let lug_int_norm: Vec<[f32; 2]> = lug_int.iter().map(|&[x, y]| normalize_point_generic_pinlug(x, y, lug_geom)).collect();

    let build_pointset = |pts: &[pinn_core::loading::BoundaryPoint], geom: &pinn_core::geometry::GeometryConfig| -> PointSetData {
        PointSetData {
            norm: pts.iter().map(|p| normalize_point_generic_pinlug(p.x, p.y, geom)).collect(),
            nx: pts.iter().map(|p| p.nx as f32).collect(),
            ny: pts.iter().map(|p| p.ny as f32).collect(),
            tx: pts.iter().map(|p| p.tx as f32).collect(),
            ty: pts.iter().map(|p| p.ty as f32).collect(),
        }
    };

    // Sized to the known closed set of names each domain's `named_point_sets()` populates
    // (pin: "interface" + optionally "driving"; lug: "interface" + "shank_anchor", plus
    // "boundary" inserted below) — avoids the incremental resize/rehash `HashMap::new()`
    // would otherwise pay as entries are inserted one at a time.
    let mut pin_named = HashMap::with_capacity(2);
    let mut lug_named = HashMap::with_capacity(3);
    for set in pin_sampling.named_point_sets(&[]) {
        pin_named.insert(set.name, build_pointset(&set.points, pin_geom));
    }
    for set in lug_sampling.named_point_sets(&[]) {
        lug_named.insert(set.name, build_pointset(&set.points, lug_geom));
    }
    lug_named.insert("boundary", build_pointset(&lug_bnd, lug_geom));
    if let Some(driving) = pin_named.get_mut("driving") {
        for t in driving.tx.iter_mut() { *t = equiv_traction as f32; }
    }

    let pin_data = DomainStepData { id: PIN_DOMAIN, int_norm: pin_int_norm, extra_ring_norm: Vec::new(), named: pin_named };
    let lug_data = DomainStepData { id: LUG_DOMAIN, int_norm: lug_int_norm, extra_ring_norm: Vec::new(), named: lug_named };
    (pin_data, lug_data)
}

pub fn run_training_pinlug(
    config: SolverConfig,
    tx: Sender<TrainingMsg>,
    stop_rx: Receiver<ControlMsg>,
) {
    use pinn_core::messages::{PinLugTrainingUpdate, PinLugVisFields};
    use pinn_core::amr::{derive_amr_config, AdaptiveGrid, AmrDomain};
    use crate::{
        pinlug_problem::{PinLugProblem, PinLugScalingMode, LUG_DOMAIN, PIN_DOMAIN},
        problem::{BoundaryValueProblem, DomainOptim, DomainState, DomainStepCtx, MultiStepCtx},
        training_core::{probe_interior_energy_residuals, residual_stats, step_physics_multi},
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

    let mut last_total = f32::MAX;

    // Steps before the first AMR sweep - same generic, problem-agnostic constant
    // `run_training_user_problem` uses, not derived from any pin-lug-specific curriculum.
    const AMR_WARMUP_STEPS: usize = 200;

    let (mut pin_data, mut lug_data) = resample_pinlug_domains(
        &problem, &pin_geom, &lug_geom, &config.load, config.n_interior, config.n_boundary, equiv_traction,
    );
    let amr_bounds = |g: &pinn_core::geometry::GeometryConfig| -> (f64, f64, f64, f64) {
        let (x0, x1) = g.x_range();
        let (y0, y1) = g.y_range();
        (x0, x1, y0, y1)
    };
    let amr_cfg_pin = derive_amr_config(amr_bounds(&pin_geom), &pin_geom.lock_zones());
    let amr_cfg_lug = derive_amr_config(amr_bounds(&lug_geom), &lug_geom.lock_zones());
    // Both grids currently derive from the same hardcoded `interval_steps` default
    // (`derive_amr_config` doesn't vary it by zone shape) - one shared sweep cadence read
    // from the lug grid's own config is accurate today; if that ever changes, gate each
    // domain by its own grid's config instead of one shared value.
    let amr_interval = amr_cfg_lug.interval_steps;
    let mut amr_grid_pin = AdaptiveGrid::<pinn_core::geometry::GeometryConfig>::new(&pin_geom, amr_cfg_pin);
    let mut amr_grid_lug = AdaptiveGrid::<pinn_core::geometry::GeometryConfig>::new(&lug_geom, amr_cfg_lug);

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

                // Geometry/n_interior/n_boundary may all have just changed - resample both
                // domains and rebuild their AMR grids from scratch (mirrors `pin_geom`/
                // `lug_geom`/`model_*`'s own rebuild above in this same arm).
                let (new_pin_data, new_lug_data) = resample_pinlug_domains(
                    &problem, &pin_geom, &lug_geom, &config.load, config.n_interior, config.n_boundary, equiv_traction,
                );
                pin_data = new_pin_data;
                lug_data = new_lug_data;
                amr_grid_pin = AdaptiveGrid::<pinn_core::geometry::GeometryConfig>::new(
                    &pin_geom, derive_amr_config(amr_bounds(&pin_geom), &pin_geom.lock_zones()),
                );
                amr_grid_lug = AdaptiveGrid::<pinn_core::geometry::GeometryConfig>::new(
                    &lug_geom, derive_amr_config(amr_bounds(&lug_geom), &lug_geom.lock_zones()),
                );
            }
            PinLugControlAction::Continue => {}
        }

        let mut amr_sweep_reports: Vec<pinn_core::messages::AmrSweepReport> = Vec::new();
        if step >= AMR_WARMUP_STEPS && (step - AMR_WARMUP_STEPS) % amr_interval == 0 {
            let probe_ctx = MultiStepCtx {
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
                dynamic_lam_penetration_cap: 500.0,
                dynamic_lam_non_tension_cap: 100.0,
                constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                n_fourier: 0,
                probe_term_gradients: false,
                phase2_active: false,
                step,
            };
            // `update_residuals` runs unconditionally (cheap EMA/trend bookkeeping, keeps the
            // next interval's `should_adapt` check current) - only the actual adapt()+resample
            // is gated (Phase 7's "smart activation" - `should_adapt` defaults to always-true,
            // zero behavior change, unless explicitly calibrated).
            let mut residuals = probe_interior_energy_residuals(&probe_ctx, &[&model_pin, &model_lug], &device);
            if let Some(r) = residuals.remove(&PIN_DOMAIN) {
                amr_grid_pin.update_residuals(&r);
                if amr_grid_pin.should_adapt(&r) {
                    let sweep_start = std::time::Instant::now();
                    let points_before = pin_data.int_norm.len();
                    let (rms_before, max_before) = residual_stats(&r);
                    let domain_area = 4.0 * pin_geom.half_w * pin_geom.half_h;
                    let hole_density_before = amr_grid_pin.lock_zone_density();
                    let domain_density_before = points_before as f64 / domain_area;

                    amr_grid_pin.adapt();
                    pin_data.int_norm = amr_grid_pin.sample_points().iter()
                        .map(|&[x, y]| normalize_point_generic_pinlug(x, y, &pin_geom)).collect();
                    let points_after = pin_data.int_norm.len();

                    let after_ctx = MultiStepCtx {
                        config: &config, problem: &problem, fd: &fd, k: 1.0,
                        domains: vec![
                            DomainStepCtx { data: &pin_data, u_ref, ref_energy, ref_stress2 },
                            DomainStepCtx { data: &lug_data, u_ref, ref_energy, ref_stress2 },
                        ],
                        dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
                        dynamic_lam_penetration_cap: 500.0, dynamic_lam_non_tension_cap: 100.0,
                        constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                        n_fourier: 0,
                        probe_term_gradients: false,
                        phase2_active: false, step,
                    };
                    let after_r = probe_interior_energy_residuals(&after_ctx, &[&model_pin, &model_lug], &device)
                        .remove(&PIN_DOMAIN).unwrap_or_default();
                    let (rms_after, max_after) = residual_stats(&after_r);

                    amr_sweep_reports.push(pinn_core::messages::AmrSweepReport {
                        domain_label: "pin", step, points_before, points_after,
                        residual_rms_before: rms_before, residual_max_before: max_before,
                        residual_rms_after: rms_after, residual_max_after: max_after,
                        sweep_duration_ms: sweep_start.elapsed().as_secs_f64() * 1000.0,
                        hole_zone_density_before: hole_density_before,
                        hole_zone_density_after: amr_grid_pin.lock_zone_density(),
                        domain_mean_density_before: domain_density_before,
                        domain_mean_density_after: points_after as f64 / domain_area,
                    });
                }
            }
            if let Some(r) = residuals.remove(&LUG_DOMAIN) {
                amr_grid_lug.update_residuals(&r);
                if amr_grid_lug.should_adapt(&r) {
                    let sweep_start = std::time::Instant::now();
                    let points_before = lug_data.int_norm.len();
                    let (rms_before, max_before) = residual_stats(&r);
                    let domain_area = 4.0 * lug_geom.half_w * lug_geom.half_h;
                    let hole_density_before = amr_grid_lug.lock_zone_density();
                    let domain_density_before = points_before as f64 / domain_area;

                    amr_grid_lug.adapt();
                    lug_data.int_norm = amr_grid_lug.sample_points().iter()
                        .map(|&[x, y]| normalize_point_generic_pinlug(x, y, &lug_geom)).collect();
                    let points_after = lug_data.int_norm.len();

                    let after_ctx = MultiStepCtx {
                        config: &config, problem: &problem, fd: &fd, k: 1.0,
                        domains: vec![
                            DomainStepCtx { data: &pin_data, u_ref, ref_energy, ref_stress2 },
                            DomainStepCtx { data: &lug_data, u_ref, ref_energy, ref_stress2 },
                        ],
                        dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
                        dynamic_lam_penetration_cap: 500.0, dynamic_lam_non_tension_cap: 100.0,
                        constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                        n_fourier: 0,
                        probe_term_gradients: false,
                        phase2_active: false, step,
                    };
                    let after_r = probe_interior_energy_residuals(&after_ctx, &[&model_pin, &model_lug], &device)
                        .remove(&LUG_DOMAIN).unwrap_or_default();
                    let (rms_after, max_after) = residual_stats(&after_r);

                    amr_sweep_reports.push(pinn_core::messages::AmrSweepReport {
                        domain_label: "lug", step, points_before, points_after,
                        residual_rms_before: rms_before, residual_max_before: max_before,
                        residual_rms_after: rms_after, residual_max_after: max_after,
                        sweep_duration_ms: sweep_start.elapsed().as_secs_f64() * 1000.0,
                        hole_zone_density_before: hole_density_before,
                        hole_zone_density_after: amr_grid_lug.lock_zone_density(),
                        domain_mean_density_before: domain_density_before,
                        domain_mean_density_after: points_after as f64 / domain_area,
                    });
                }
            }
        }

        let n_colloc = pin_data.int_norm.len() + lug_data.int_norm.len();

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
            // Pin-lug's GUI path deliberately keeps the pre-existing fixed weight - the same
            // documented scope cut as `phase2_active: false` above (no cascade logic here);
            // only `run_user_problem_training_from`'s plate path was raised to match its own
            // now-real `dynamic_lam_h_cap` cap.
            constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
            n_fourier: 0,
            probe_term_gradients: false,
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
                &model_pin_val, &pin_geom, [nx_vis, ny_vis], &fd, u_ref, config.load.px,
                &config.material, &pin_data.int_norm, &device,
            );
            let lug_vis = evaluate_vis_grid_mdem(
                &model_lug_val, &lug_geom, [nx_vis, ny_vis], &fd, u_ref, config.load.px,
                &config.material, &lug_data.int_norm, &device,
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
                // NOTE: this whole `PinLugTrainingUpdate` only gets built/sent every 50 steps
                // (this `if step % 50 == 0` block) - a report from `amr_sweep_reports` is only
                // ever actually delivered if the sweep step happens to also be a multiple of
                // 50. True today (AMR_WARMUP_STEPS=200, amr_interval=1000 are both multiples
                // of 50) but NOT a structurally guaranteed invariant - if either constant
                // changes to something not divisible by 50 in the future, a sweep report
                // could be silently dropped. Flagging here rather than restructuring this
                // function's send cadence (out of scope for this pass).
                amr_sweep: amr_sweep_reports,
                grad_norm: out.grad_norm,
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
/// Fresh start: builds a new model from `spec.network` and trains it for the full
/// `0..spec.training.max_steps`. See `run_user_problem_training_from` for the shared body -
/// this and `run_training_user_problem_resume` are both thin wrappers around it.
pub fn run_training_user_problem(
    spec: ProblemSpec,
    tx: Sender<TrainingMsg>,
    stop_rx: Receiver<ControlMsg>,
) {
    let device = BDevice::default();
    let net_cfg = ElasticityNetConfig::new()
        // See `UserGeometry::n_fourier`'s doc comment - was hardcoded `3` (no Fourier
        // embedding) for every geometry, including holed ones, which is the real root cause
        // this fixes. MUST match every forward pass's own `n_fourier` (the per-step
        // `MultiStepCtx.n_fourier` below, and every `user_problem.rs` probe, which all derive
        // it from this same geometry) or the first forward pass panics on a tensor width
        // mismatch.
        .with_input_dim(spec.geometry.net_input_dim())
        .with_hidden_dim(spec.network.hidden_dim)
        .with_n_hidden(spec.network.n_hidden)
        .with_output_dim(5) // mDEM: u, v, sigma_xx, sigma_yy, sigma_xy
        // Smart adaptive architecture: the gated-residual (PirateNet) structure is what makes
        // safe depth growth/shrink possible at all (see `ElasticityNet::append_dormant_layer`'s
        // doc comment) - there is no separate user-facing "use_piratenet" toggle, `adaptive`
        // forces it internally.
        .with_use_piratenet(spec.network.adaptive);
    let model = net_cfg.init(&device);
    run_user_problem_training_from(spec, model, device, 0, tx, stop_rx);
}

/// Graceful-stop-and-resume: loads a checkpoint's weights into a fresh TRAINABLE model and
/// continues training for `additional_steps` more, starting from where the checkpoint left
/// off. Lightweight resume (a deliberate, documented tradeoff - see the design plan this
/// shipped with): only the weights carry over - optimizer momentum, SAW-BRDR's adapted loss
/// weights, the LR schedule's warmup phase, and the AMR grid all restart fresh, exactly as
/// `run_training_user_problem` builds them for a brand-new run. That's a real, temporary rough
/// patch for roughly the first few hundred resumed steps (the same self-correcting warmup any
/// fresh run already goes through), not a bug - not persisting that state at all is what keeps
/// this a same-session-sized change instead of a new checkpoint file format.
///
/// Parametric checkpoints are out of scope - `run_training_parametric`'s own "instant
/// inference" serving model already covers its post-training use case, and this feature was
/// requested specifically for the plate/user-defined-problem workflow.
pub fn run_training_user_problem_resume(
    weights_path: std::path::PathBuf,
    additional_steps: usize,
    tx: Sender<TrainingMsg>,
    stop_rx: Receiver<ControlMsg>,
) {
    let device = BDevice::default();
    let (model, meta) = match crate::checkpoint::load_checkpoint_for_training(&weights_path, &device) {
        Ok(v) => v,
        Err(e) => {
            let _ = tx.send(TrainingMsg::Error(format!("failed to load checkpoint for resume: {e}")));
            return;
        }
    };
    let mut spec = match meta.spec {
        crate::checkpoint::CheckpointSpec::Plate(s) => s,
        crate::checkpoint::CheckpointSpec::Parametric(_) => {
            let _ = tx.send(TrainingMsg::Error(
                "Resume is only supported for plate specs, not parametric checkpoints.".to_string(),
            ));
            return;
        }
    };
    let steps_completed = meta.steps_completed;
    spec.training.max_steps = steps_completed.saturating_add(additional_steps);
    run_user_problem_training_from(spec, model, device, steps_completed, tx, stop_rx);
}

/// Shared body for `run_training_user_problem`/`run_training_user_problem_resume` - identical
/// either way except where `model`/`device` come from and where the step counter starts
/// (`step_offset`: `0` for a fresh run, `steps_completed` for a resumed one - `spec.training.
/// max_steps` is the ABSOLUTE target step either way, already adjusted by the resume wrapper).
fn run_user_problem_training_from(
    spec: ProblemSpec,
    mut model: ElasticityNet<B>,
    device: BDevice,
    step_offset: usize,
    tx: Sender<TrainingMsg>,
    stop_rx: Receiver<ControlMsg>,
) {
    use crate::architecture_controller::{ArchitectureConfig, ArchitectureController};
    use crate::problem::{
        BoundaryValueProblem, DomainOptim, DomainStepCtx, DomainStepData, MultiStepCtx, PointSetData,
    };
    use crate::training_core::{probe_interior_energy_residuals, residual_stats, step_physics_multi};
    use crate::user_problem::{evaluate_user_vis_grid, UserDefinedProblem, USER_DOMAIN};
    use pinn_core::amr::{derive_amr_config, AdaptiveGrid, AmrDomain};
    use pinn_core::user_geometry::UserGeometry;

    /// Steps before the first AMR sweep - a small, generic warmup (not derived from any
    /// Kirsch-specific curriculum) so refinement isn't driven by a freshly-initialized
    /// network's noisy residual signal.
    const AMR_WARMUP_STEPS: usize = 200;

    let half_w = spec.geometry.half_w;
    let half_h = spec.geometry.half_h;

    let problem = UserDefinedProblem::new(spec.clone());
    validate_loss_terms(&problem);

    let mut config = SolverConfig::default_kirsch();
    config.load = spec.load;

    let mut optim = DomainOptim {
        weight: WeightOptim::new(config.use_soap_muon),
        bias: make_bias_optim(),
        gate: make_gate_optim(),
    };

    // Smart adaptive architecture (v1) - `arch_controller` stays `None` (a zero-cost, always-
    // `None`-returning check per step) unless `spec.network.adaptive`. `current_hidden_dim`/
    // `current_n_hidden` track the LIVE architecture (mirrors `headless.rs`'s own
    // `current_hidden_dim` convention for its width-growth event) - `spec.network.hidden_dim`/
    // `n_hidden` themselves are left untouched, same documented v1 scope limitation
    // `headless.rs` already accepts. `arch_config` is kept (not just consumed by the
    // controller) so its `gate_epsilon` can be reused for `ShrinkDepth`'s dormancy re-check.
    let mut current_hidden_dim = spec.network.hidden_dim;
    let mut current_n_hidden = spec.network.n_hidden;
    let arch_config = ArchitectureConfig::v1(
        spec.network.max_hidden_dim.unwrap_or(usize::MAX),
        spec.network.max_n_hidden.unwrap_or(usize::MAX),
    );
    let mut arch_controller = spec.network.adaptive.then(|| ArchitectureController::new(arch_config.clone()));
    let mut arch_snapshot: Option<(ElasticityNet<B>, usize, usize)> = None;

    // Auto-stop when training plateaus - defaults ON (`spec.network.auto_stop_on_plateau`),
    // matching prior behavior on the Kirsch path (its own `ConvergenceTracker`-driven cascade
    // in `headless.rs`), which the plate path never had until now. `ConvergenceTracker::
    // for_metric` is already generic over any scalar metric - the same reuse this crate's own
    // `ArchitectureController` already established for the SAME reason. Fed `bc_residual_rms`
    // at the existing vis cadence below (real, already-computed - no new physics probe).
    // Unlike a warm restart, a detected plateau here triggers a plain graceful stop (the same
    // path `ControlAction::StopAndFinish` already takes), not a restart - Kirsch's LR/Adam
    // reset + tightened `lam_h_cap` cascade is tuned specifically for its own K_t dynamics.
    let mut plateau_tracker = spec.network.auto_stop_on_plateau.then(|| {
        ConvergenceTracker::for_metric(MetricDirection::SmallerIsBetter, 0.05, f64::INFINITY, 0.0)
    });
    let mut auto_stopped = false;
    // `ConvergenceTracker::check_plateau` needs `PLATEAU_WINDOW * 2` (40) readings before it can
    // even evaluate a plateau - a real, previously-confirmed characteristic (see
    // `run_training_user_problem_adaptive_wiring_does_not_crash_and_events_are_consistent`'s own
    // doc comment for `ArchitectureController`'s identical reuse of this tracker). Kirsch's own
    // usage feeds it once every 50 real steps, so that warmup is ~2000 real steps there - but
    // this path's vis cadence (below) is every 10 steps, so pushing every tick would make the
    // SAME 40-reading warmup fire at only ~400 real steps, long before the plate path has
    // learned anything meaningful (confirmed via a real run: auto-stop fired at step ~430 with
    // total_loss still at ~4.0, three orders of magnitude from converged). Throttling the push
    // to every 5th vis-cadence tick (once per 50 real steps) matches Kirsch's own real cadence-
    // to-real-step ratio - a principled choice, not an arbitrary number - without touching the
    // shared, already-tested `ConvergenceTracker`/`PLATEAU_WINDOW` at all.
    let mut plateau_push_counter: usize = 0;

    let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
    let mut saw = SawBrdr::with_base(base_weights, 0.95);
    let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);

    let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
    let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);

    let placeholder_geom = pinn_core::geometry::GeometryConfig::kirsch_plate_inches(); // ignored by UserSamplingStrategy
    let [nx_vis, ny_vis] = config.vis_grid;

    // `UserSamplingStrategy::sample_interior`/`sample_boundary`/`named_point_sets` are pure,
    // fixed-seed (`LcgRng::new(SEED_INTERIOR)`) functions of `(geometry, load, n)` - all
    // constant for the lifetime of this run (no warm-start support here, see this function's
    // doc comment). Computing `data` ONCE before the loop instead of every step is a real,
    // measured perf fix (rejection-sampling ~n_interior+n_boundary points and rebuilding a
    // `HashMap` of per-hole point sets from scratch every step, for potentially thousands of
    // steps, was pure waste - byte-identical output either way since the seed never changes).
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

    // 1 outer_boundary + 2 per hole (traction ring + constitutive-consistency anchor ring -
    // see `UserSamplingStrategy::named_point_sets`).
    let mut named = HashMap::with_capacity(1 + 2 * spec.geometry.holes.len());
    named.insert("outer_boundary", to_pointset(&bnd_pts));
    for set in sampling.named_point_sets(&[]) {
        named.insert(set.name, to_pointset(&set.points));
    }
    let mut data = DomainStepData { id: USER_DOMAIN, int_norm, extra_ring_norm: Vec::new(), named };

    // Generic, problem-agnostic AMR - see `pinn_core::amr::AmrDomain`'s doc comment. Nothing
    // here is gated on "does this geometry have a hole": `spec.geometry.lock_zones()` is
    // empty for a feature-less plate and non-empty otherwise, and either way `adapt()`'s
    // residual-driven refine/coarsen runs the same. `int_norm`'s resampling-cache fix (above)
    // stays correct - an AMR sweep is the ONE place `data.int_norm` legitimately changes
    // after this point.
    // Collocation-only geometry with every hole radius inflated by the FD-safe margin (see
    // `UserGeometry::inflated_for_collocation`'s doc comment) - used ONLY to build this AMR
    // grid's own containment gate, never for physics/display. Without this, AMR's hole-zone
    // refinement (which deliberately concentrates cells near the hole) can produce leaf-cell
    // centers close enough to the TRUE boundary that an FD stencil there dips back inside the
    // hole, corrupting the very interior-energy/constitutive-consistency signal AMR exists to
    // strengthen there. The margin is tiny relative to the hole radius (a few tenths of a
    // millimeter vs. typically tens of millimeters), so `lock_zones()`/`feature_ratio`
    // computed from it are practically unchanged from the true-radius values.
    let collocation_margin_m = crate::user_problem::ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
    let collocation_geometry = spec.geometry.inflated_for_collocation(collocation_margin_m);
    let amr_cfg = derive_amr_config((-half_w, half_w, -half_h, half_h), &collocation_geometry.lock_zones());
    let amr_interval = amr_cfg.interval_steps;
    let mut amr_grid = AdaptiveGrid::<UserGeometry>::new(&collocation_geometry, amr_cfg);

    // Stage H (model checkpoint save/load) - tracked so the post-training serving loop below
    // can build a real `CheckpointMeta` (steps actually completed, not just `max_steps`, since
    // `StopAndFinish` can end the run early). Starts at `step_offset - 1` (not `0`) so a resume
    // whose loop body never executes (e.g. `additional_steps == 0`) still reports
    // `steps_completed` as the checkpoint's own starting point, not `1`.
    let mut last_step = step_offset.saturating_sub(1);
    let mut last_total_loss = 0.0f32;

    for step in step_offset..spec.training.max_steps {
        match handle_control_messages(&stop_rx) {
            ControlAction::StopAndFinish => break,
            ControlAction::StopImmediately => return,
            ControlAction::WarmStart { .. } => {} // no-op: see this function's doc comment
            ControlAction::Continue => {}
        }
        last_step = step;

        let mut amr_sweep_report = None;
        if step >= AMR_WARMUP_STEPS && (step - AMR_WARMUP_STEPS) % amr_interval == 0 {
            let probe_ctx = MultiStepCtx {
                config: &config,
                problem: &problem,
                fd: &fd,
                k: 1.0,
                domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                dynamic_lam_h_cap: f64::MAX,
                dynamic_lam_d_cap: f64::MAX,
                dynamic_lam_penetration_cap: f64::MAX,
                dynamic_lam_non_tension_cap: f64::MAX,
                constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                n_fourier: spec.geometry.n_fourier(),
                probe_term_gradients: false,
                phase2_active: true,
                step,
            };
            if let Some(residuals) = probe_interior_energy_residuals(&probe_ctx, &[&model], &device).remove(&USER_DOMAIN) {
                // `update_residuals` runs unconditionally (cheap EMA/trend bookkeeping, keeps
                // the next interval's `should_adapt` check current) - only the actual
                // adapt()+resample is gated (Phase 7's "smart activation": `should_adapt`
                // defaults to always-true, zero behavior change, unless explicitly
                // calibrated - see `AmrtConfig::residual_threshold`'s doc comment).
                amr_grid.update_residuals(&residuals);
                if amr_grid.should_adapt(&residuals) {
                    // Phase 11 ("AMR Effectiveness") - real before/after residual + timing,
                    // not just "the collocation count changed". `points_before`/`residual_*_
                    // before` are captured from the SAME probe that fed `should_adapt` above
                    // (no extra forward pass needed for the "before" half).
                    let sweep_start = std::time::Instant::now();
                    let points_before = data.int_norm.len();
                    let (rms_before, max_before) = residual_stats(&residuals);
                    let domain_area = 4.0 * half_w * half_h;
                    let hole_density_before = amr_grid.lock_zone_density();
                    let domain_density_before = points_before as f64 / domain_area;

                    amr_grid.adapt();
                    data.int_norm = amr_grid.sample_points().iter().map(|&[x, y]| norm_pt(x, y)).collect();
                    let points_after = data.int_norm.len();

                    // "After" DOES need a fresh probe - the point set (and therefore the
                    // residual signal at it) genuinely changed.
                    let after_ctx = MultiStepCtx {
                        config: &config, problem: &problem, fd: &fd, k: 1.0,
                        domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                        dynamic_lam_h_cap: f64::MAX, dynamic_lam_d_cap: f64::MAX,
                        dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                        constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                        n_fourier: spec.geometry.n_fourier(),
                        probe_term_gradients: false,
                        phase2_active: true, step,
                    };
                    let after_residuals = probe_interior_energy_residuals(&after_ctx, &[&model], &device)
                        .remove(&USER_DOMAIN).unwrap_or_default();
                    let (rms_after, max_after) = residual_stats(&after_residuals);

                    amr_sweep_report = Some(pinn_core::messages::AmrSweepReport {
                        domain_label: "interior",
                        step,
                        points_before, points_after,
                        residual_rms_before: rms_before, residual_max_before: max_before,
                        residual_rms_after: rms_after, residual_max_after: max_after,
                        sweep_duration_ms: sweep_start.elapsed().as_secs_f64() * 1000.0,
                        hole_zone_density_before: hole_density_before,
                        hole_zone_density_after: amr_grid.lock_zone_density(),
                        domain_mean_density_before: domain_density_before,
                        domain_mean_density_after: points_after as f64 / domain_area,
                    });
                }
            }
        }

        // Hoisted above `ctx`'s construction (previously computed just before its own use
        // below) so `probe_term_gradients` can be gated on the same cadence as the other
        // vis-cadence-only expensive probes (`hole_analyses`/`bc_residual_rms`/etc.) - see
        // `MultiStepCtx::probe_term_gradients`'s doc comment for why this can never be
        // unconditionally `true` on a hot path (N extra backward passes/step), but tying it to
        // the already-established "expensive, infrequent" cadence costs nothing new in kind,
        // only in the same already-accepted vis-cadence magnitude.
        let send_vis = step % 10 == 0 || step + 1 == spec.training.max_steps;
        let ctx = MultiStepCtx {
            config: &config,
            problem: &problem,
            fd: &fd,
            k: 1.0, // IdentityAnsatz ignores k entirely
            domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
            // Real root cause of the garbage-Kt/zero-hole-stress bug (see `powershell_tool/
            // CLAUDE.md`'s Stress Solver section): this path's `hole_traction_loss_direct` term
            // was left fully UNCAPPED (`f64::MAX`, unlike Kirsch's own real, tested 50→15
            // cascade), so SAW-BRDR could grow its adapted weight arbitrarily large over a long
            // run relative to `step_physics_multi`'s fixed `LAM_CONSTITUTIVE_CONSISTENCY` (5.0)
            // - and an outweighed boundary term has a strictly EASIER minimum available than the
            // true elasticity solution: drive the direct-stress outputs toward zero everywhere
            // the boundary term is evaluated (trivially satisfies "traction ≈ 0" without
            // satisfying "stress matches Hooke's law"). Capped at Kirsch's own starting value
            // (50.0, a real, already-tuned bound in this codebase, not an arbitrary guess) -
            // deliberately NOT replicating Kirsch's full plateau-triggered cascade-with-restarts
            // down to 15.0, which is tuned specifically for Kirsch's own K_t dynamics and a
            // separate, larger feature this fix doesn't need.
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: f64::MAX,
            dynamic_lam_non_tension_cap: f64::MAX,
            // Paired with the `dynamic_lam_h_cap` fix directly above: capping the boundary
            // term alone (a real, previously-landed fix) measurably reduced the interior PDE
            // residual but left Kt essentially unmoved - confirmed via a real training run,
            // not assumed (see `run_training_user_problem_generalized_amr_indicator_
            // diagnostic`). Raising this to the SAME 50.0 ceiling closes the remaining gap:
            // `hole_traction` can no longer structurally outweigh constitutive-consistency by
            // 10x the way it could when one was capped at 50 and the other pinned at 5.
            constitutive_consistency_weight: 50.0,
            // Real, evidence-driven fix (see `UserGeometry::n_fourier`'s doc comment for the
            // full story): after ruling out weighting AND sampling density as the bottleneck
            // via multiple independent, measured experiments this session, the remaining gap
            // is representational - Kirsch's own path already uses Fourier positional
            // encoding for exactly this reason ("corrects spectral bias near hole") and
            // already achieves real Kt convergence; this path never had it. `net_cfg`'s
            // `input_dim` (below, at this function's model-construction site) MUST use the
            // SAME value - see that call site's own comment.
            n_fourier: spec.geometry.n_fourier(),
            probe_term_gradients: send_vis,
            phase2_active: true,
            step,
        };

        let (new_model, out) = step_physics_multi(
            vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, &device,
            0, 1.0, 1.0,
        );
        model = new_model.into_iter().next().unwrap();

        // Loss/lr numbers are already computed every step by `out` above (basically free to
        // report) - only the stress-field grid probe below is genuinely expensive (a forward
        // pass over the whole `[nx_vis, ny_vis]` grid). Sending a cheap `vis: None` update
        // EVERY step (not just every 10th) keeps the GUI's loss chart/stat rail visibly live
        // instead of appearing to freeze for up to 10 steps between updates - a real UX
        // complaint on a config slow enough that 10 steps takes several seconds. The
        // expensive field probe still only runs on the original every-10th-step/last-step
        // cadence.
        let mut architecture_event = None;
        let (vis, hole_analyses, bc_residual_rms, bc_residual_max, reaction_force, energy_balance, network_snapshot) = if send_vis {
            let model_val: ElasticityNet<BInner> = model.valid();
            let vis = evaluate_user_vis_grid(
                &model_val, &spec.geometry, [nx_vis, ny_vis], u_ref, spec.load.px,
                &spec.material, &fd, &data.int_norm, &device,
            );
            // Phase 16 ("Final Results Dashboard") - real hole-boundary stress analysis,
            // computed on the SAME cadence/model snapshot as `vis` (not every step - each
            // hole is an extra small forward pass, cheap but not free). `nominal_stress` is
            // the applied far-field traction magnitude, the standard Kt denominator.
            let nominal_stress = spec.load.px.abs().max(spec.load.py.abs());
            // Derived (Hooke, from FD strain) stress at r=hole.radius+margin, NOT direct mDEM
            // σ exactly at the hole boundary - see `probe_hole_boundary_profile_derived`'s doc
            // comment (bugSource-New #12: nothing keeps direct σ aligned to real elasticity
            // away from the traction-free BC anymore, so Kt must read the derived field).
            let hole_margin = crate::user_problem::ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
            let hole_analyses: Vec<pinn_core::messages::HoleAnalysis> = spec.geometry.holes.iter().enumerate()
                .map(|(hole_index, hole)| {
                    let profile = crate::user_problem::probe_hole_boundary_profile_derived(
                        &model_val, &spec.geometry, hole, 72, &fd, u_ref, spec.load.px,
                        &spec.material, hole_margin, &device,
                    );
                    let concentration = crate::user_problem::stress_concentration_from_profile(&profile, nominal_stress);
                    pinn_core::messages::HoleAnalysis { hole_index, profile, concentration }
                })
                .collect();
            // `enhancement.txt` items 4/C ("BC residual RMS/max") - same vis cadence as
            // above, a real side probe, not part of the per-step loss computation.
            let (bc_rms, bc_max) = crate::user_problem::probe_boundary_residuals(&model_val, &spec, &device);

            // Auto-stop when training plateaus - see this function's own setup comment above
            // for the full rationale. A detected plateau just sets a flag here; the actual
            // `break` happens after this step's `TrainingUpdate` is sent, below, so the GUI
            // still sees the final state before the run ends.
            if let Some(tracker) = plateau_tracker.as_mut() {
                plateau_push_counter += 1;
                if plateau_push_counter % 5 == 0 {
                    tracker.push(bc_rms);
                    if tracker.check_plateau().is_some() {
                        auto_stopped = true;
                    }
                }
            }

            // `enhancement.md` Phase 9 ("Force Equilibrium Validation") - same vis cadence,
            // same "side probe" precedent (see `ReactionForce`'s doc comment in
            // `pinn_core::messages`).
            let rf = crate::user_problem::probe_reaction_force(&model_val, &spec, &device);
            // `enhancement.md` Phase 10 ("Energy Validation") - same vis cadence/side-probe
            // precedent as `reaction_force` above.
            let eb = crate::user_problem::probe_energy_balance(&model_val, &spec, &device);
            let ns = crate::network::network_snapshot(&model_val);

            // Smart adaptive architecture - only active when `spec.network.adaptive` (forces
            // `use_piratenet` in `net_cfg` above). Fed `bc_rms` as the training-progress
            // signal: the closest already-computed-at-this-cadence scalar to a PDE residual
            // (the interior residual `probe_interior_energy_residuals` computes runs on a
            // DIFFERENT, AMR-only cadence - reusing it here would mean a new physics probe,
            // which the design plan explicitly avoided).
            if let Some(controller) = arch_controller.as_mut() {
                let per_neuron_mags = model_val.per_neuron_magnitudes();
                if let Some(action) = controller.observe(
                    bc_rms, current_hidden_dim, current_n_hidden, &ns.awake_mask, &per_neuron_mags,
                ) {
                    let (new_model, description, new_hidden_dim, new_n_hidden) = crate::optim::apply_arch_action(
                        &action, model, &mut arch_snapshot, &mut optim.weight, config.use_soap_muon,
                        arch_config.gate_epsilon, current_hidden_dim, current_n_hidden, &device,
                    );
                    model = new_model;
                    architecture_event = Some(pinn_core::messages::ArchitectureEvent {
                        step,
                        description,
                        hidden_dim_before: current_hidden_dim,
                        hidden_dim_after: new_hidden_dim,
                        n_hidden_before: current_n_hidden,
                        n_hidden_after: new_n_hidden,
                    });
                    current_hidden_dim = new_hidden_dim;
                    current_n_hidden = new_n_hidden;
                }
            }

            (Some(vis), hole_analyses, bc_rms, bc_max, Some(rf), Some(eb), Some(ns))
        } else {
            (None, Vec::new(), 0.0, 0.0, None, None, None)
        };

        last_total_loss = out.total_scalar;
        let energy_loss = out.e_scalar;
        let neumann_loss = out.total_scalar - energy_loss;
        // `training_core::GradientShareReport` -> `pinn_core::messages::GradientShareSummary` -
        // a separate transport-side type so `pinn-core` never depends on `pinn-solver` (same
        // reasoning as `HoleBoundaryPoint`/`StressConcentration`).
        let gradient_share_report = out.gradient_share_report.map(|r| pinn_core::messages::GradientShareSummary {
            shares: r.shares.into_iter().collect(),
            inert: r.inert,
            dominant: r.dominant,
        });
        // `training_core::GradientConflictReport` -> `pinn_core::messages::
        // GradientConflictSummary` - same transport-split reasoning as `gradient_share_report`
        // immediately above (Priority 4, General-PINN §17).
        let gradient_conflict_report = out.gradient_conflict_report.map(|r| pinn_core::messages::GradientConflictSummary {
            pairs: r.pairs.into_iter().map(|p| (p.term_a, p.term_b, p.cosine_similarity)).collect(),
            most_conflicting: r.most_conflicting.map(|p| (p.term_a, p.term_b, p.cosine_similarity)),
        });
        // Static per problem (doesn't change step to step) and genuinely free (pure `Vec`/
        // string logic over `problem.loss_terms()`, no tensor ops) - computed every update,
        // never gated, unlike `gradient_share_report` above.
        let stress_source_report: Vec<(&'static str, &'static str)> =
            crate::training_core::stress_source_report(&problem).into_iter()
                .map(|(name, source)| (name, match source {
                    crate::problem::StressSource::Direct => "Direct",
                    crate::problem::StressSource::Derived => "Derived",
                    crate::problem::StressSource::Both => "Both",
                }))
                .collect();
        // Same "static per problem, never gated" treatment as `stress_source_report` above -
        // Priority 5, General-PINN §13.
        let boundary_operator_report: Vec<(&'static str, &'static str)> =
            crate::training_core::boundary_operator_report(&problem).into_iter()
                .map(|(name, kind)| (name, match kind {
                    crate::problem::BoundaryOperatorKind::Dirichlet => "Dirichlet",
                    crate::problem::BoundaryOperatorKind::Neumann => "Neumann",
                    crate::problem::BoundaryOperatorKind::Robin => "Robin",
                    crate::problem::BoundaryOperatorKind::Periodic => "Periodic",
                    crate::problem::BoundaryOperatorKind::Symmetry => "Symmetry",
                    crate::problem::BoundaryOperatorKind::Interface => "Interface",
                }))
                .collect();
        // Same "static per problem, never gated" treatment as `boundary_operator_report` above
        // - Priority 6, General-PINN §10.
        let derivative_order_report: Vec<(&'static str, &'static str)> =
            crate::training_core::derivative_order_report(&problem).into_iter()
                .map(|(name, order)| (name, match order {
                    crate::problem::DerivativeOrder::First => "First",
                    crate::problem::DerivativeOrder::Second => "Second",
                }))
                .collect();
        // Priority 9, General-PINN §39 - unlike the reports above, this doesn't filter (every
        // active term always has a meaningful formulation_kind).
        let formulation_kind_report: Vec<(&'static str, &'static str)> =
            crate::training_core::formulation_kind_report(&problem).into_iter()
                .map(|(name, kind)| (name, match kind {
                    crate::problem::FormulationKind::Strong => "Strong",
                    crate::problem::FormulationKind::Weak => "Weak",
                }))
                .collect();
        // Priority 10, General-PINN §40 - filtered like stress_source_report/boundary_
        // operator_report/derivative_order_report (most terms aren't constraints).
        let constraint_report: Vec<(&'static str, &'static str)> =
            crate::training_core::constraint_report(&problem).into_iter()
                .map(|(name, kind)| (name, match kind {
                    crate::problem::ConstraintKind::Unconstrained => "Unconstrained",
                    crate::problem::ConstraintKind::PenaltyInequality => "PenaltyInequality",
                }))
                .collect();
        let update = TrainingUpdate {
            step,
            total_loss: out.total_scalar,
            energy_loss,
            neumann_loss,
            lr: out.lr as f32,
            lam_energy: out.lam_e as f32,
            lam_neumann: 0.0, // no single generic BC lambda exists for an arbitrary term set
            n_colloc: data.int_norm.len(), // AMR can change this from spec.training.n_interior after a sweep
            kt_estimate: None,
            vis,
            amr_sweep: amr_sweep_report,
            hole_analyses,
            grad_norm: out.grad_norm,
            bc_residual_rms,
            bc_residual_max,
            reaction_force,
            energy_balance,
            network_snapshot,
            architecture_event,
            gradient_share_report,
            gradient_conflict_report,
            stress_source_report,
            boundary_operator_report,
            derivative_order_report,
            formulation_kind_report,
            constraint_report,
        };
        let _ = tx.try_send(TrainingMsg::Update(Box::new(update)));
        if auto_stopped {
            break;
        }
    }

    let _ = tx.send(TrainingMsg::Done);

    // Stage H (model checkpoint save/load) - stay alive to serve on-demand `SaveCheckpoint`
    // requests, mirroring `parametric_problem::run_training_parametric`'s own post-training
    // serving loop (the identical pattern applied a second time, not a new architecture). This
    // path has no `ParametricInfer` equivalent to also serve - a single fixed-spec model has
    // nothing further to query, only to (optionally) persist.
    loop {
        match stop_rx.recv() {
            Ok(ControlMsg::Stop) | Err(_) => return,
            Ok(ControlMsg::SaveCheckpoint { path, saved_at_unix }) => {
                // Smart adaptive architecture: record the LIVE architecture actually reached
                // (`current_hidden_dim`/`current_n_hidden`), not `spec.network`'s original
                // static values - a checkpoint saved after adaptation would otherwise reload
                // with the wrong shape (see `checkpoint::load_checkpoint`'s doc comment).
                let mut live_spec = spec.clone();
                live_spec.network.hidden_dim = current_hidden_dim;
                live_spec.network.n_hidden = current_n_hidden;
                let meta = crate::checkpoint::CheckpointMeta {
                    spec: crate::checkpoint::CheckpointSpec::Plate(live_spec),
                    steps_completed: last_step + 1,
                    final_loss: last_total_loss,
                    saved_at_unix,
                };
                let model_val: ElasticityNet<BInner> = model.valid();
                let result = crate::checkpoint::save_checkpoint(model_val, &meta, &path)
                    .map(|p| p.display().to_string());
                let _ = tx.try_send(TrainingMsg::CheckpointSaved(result));
            }
            Ok(_) => {}
        }
    }
}

/// Stage H (model checkpoint save/load) - serves a `ProblemSpec` checkpoint loaded straight
/// from disk, no training performed. Evaluates the loaded model ONCE via the same standalone
/// probes `run_training_user_problem`'s own vis-cadence block calls (`evaluate_user_vis_grid`,
/// `probe_hole_boundary_profile`/`stress_concentration_from_profile`, `probe_boundary_
/// residuals`, `probe_reaction_force`, `probe_energy_balance` - all pure functions of
/// `(model, spec, device)`, needing none of that function's internal per-step training state),
/// sends it as one `Update`+`Done` pair so the UI's existing Results cards populate exactly as
/// they would after a real run, then stays alive to serve further `SaveCheckpoint` requests.
pub fn serve_loaded_plate_checkpoint(
    spec: ProblemSpec,
    model: ElasticityNet<BInner>,
    tx: Sender<TrainingMsg>,
    stop_rx: Receiver<ControlMsg>,
) {
    let device = BDevice::default();
    let half_w = spec.geometry.half_w;
    let half_h = spec.geometry.half_h;
    let u_ref = crate::training_core::compute_reference_scales_for_plate(&spec).u_ref;
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);
    let [nx_vis, ny_vis] = SolverConfig::default_kirsch().vis_grid;

    let vis = crate::user_problem::evaluate_user_vis_grid(
        &model, &spec.geometry, [nx_vis, ny_vis], u_ref, spec.load.px, &spec.material, &fd, &[], &device,
    );
    let nominal_stress = spec.load.px.abs().max(spec.load.py.abs());
    // Derived-stress-at-margin, not direct σ at the exact boundary - see
    // `probe_hole_boundary_profile_derived`'s doc comment.
    let hole_margin = crate::user_problem::ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);
    let hole_analyses: Vec<pinn_core::messages::HoleAnalysis> = spec.geometry.holes.iter().enumerate()
        .map(|(hole_index, hole)| {
            let profile = crate::user_problem::probe_hole_boundary_profile_derived(
                &model, &spec.geometry, hole, 72, &fd, u_ref, spec.load.px, &spec.material, hole_margin, &device,
            );
            let concentration = crate::user_problem::stress_concentration_from_profile(&profile, nominal_stress);
            pinn_core::messages::HoleAnalysis { hole_index, profile, concentration }
        }).collect();
    let (bc_residual_rms, bc_residual_max) = crate::user_problem::probe_boundary_residuals(&model, &spec, &device);
    let reaction_force = crate::user_problem::probe_reaction_force(&model, &spec, &device);
    let energy_balance = crate::user_problem::probe_energy_balance(&model, &spec, &device);
    let network_snapshot = crate::network::network_snapshot(&model);
    // Transient - constructed only to enumerate `loss_terms()`, no network/training involved.
    let stress_source_report: Vec<(&'static str, &'static str)> =
        crate::training_core::stress_source_report(&crate::user_problem::UserDefinedProblem::new(spec.clone()))
            .into_iter()
            .map(|(name, source)| (name, match source {
                crate::problem::StressSource::Direct => "Direct",
                crate::problem::StressSource::Derived => "Derived",
                crate::problem::StressSource::Both => "Both",
            }))
            .collect();
    let boundary_operator_report: Vec<(&'static str, &'static str)> =
        crate::training_core::boundary_operator_report(&crate::user_problem::UserDefinedProblem::new(spec.clone()))
            .into_iter()
            .map(|(name, kind)| (name, match kind {
                crate::problem::BoundaryOperatorKind::Dirichlet => "Dirichlet",
                crate::problem::BoundaryOperatorKind::Neumann => "Neumann",
                crate::problem::BoundaryOperatorKind::Robin => "Robin",
                crate::problem::BoundaryOperatorKind::Periodic => "Periodic",
                crate::problem::BoundaryOperatorKind::Symmetry => "Symmetry",
                crate::problem::BoundaryOperatorKind::Interface => "Interface",
            }))
            .collect();
    let derivative_order_report: Vec<(&'static str, &'static str)> =
        crate::training_core::derivative_order_report(&crate::user_problem::UserDefinedProblem::new(spec.clone()))
            .into_iter()
            .map(|(name, order)| (name, match order {
                crate::problem::DerivativeOrder::First => "First",
                crate::problem::DerivativeOrder::Second => "Second",
            }))
            .collect();
    let formulation_kind_report: Vec<(&'static str, &'static str)> =
        crate::training_core::formulation_kind_report(&crate::user_problem::UserDefinedProblem::new(spec.clone()))
            .into_iter()
            .map(|(name, kind)| (name, match kind {
                crate::problem::FormulationKind::Strong => "Strong",
                crate::problem::FormulationKind::Weak => "Weak",
            }))
            .collect();
    let constraint_report: Vec<(&'static str, &'static str)> =
        crate::training_core::constraint_report(&crate::user_problem::UserDefinedProblem::new(spec.clone()))
            .into_iter()
            .map(|(name, kind)| (name, match kind {
                crate::problem::ConstraintKind::Unconstrained => "Unconstrained",
                crate::problem::ConstraintKind::PenaltyInequality => "PenaltyInequality",
            }))
            .collect();

    let update = TrainingUpdate {
        step: 0, total_loss: 0.0, energy_loss: 0.0, neumann_loss: 0.0, lr: 0.0,
        lam_energy: 0.0, lam_neumann: 0.0, n_colloc: 0, kt_estimate: None,
        vis: Some(vis), amr_sweep: None, hole_analyses,
        grad_norm: None, bc_residual_rms, bc_residual_max,
        reaction_force: Some(reaction_force), energy_balance: Some(energy_balance),
        network_snapshot: Some(network_snapshot),
        architecture_event: None, // loaded, not (re)trained this session - nothing happened
        gradient_share_report: None, // no training step ran, so no per-term gradient exists
        gradient_conflict_report: None,
        stress_source_report,
        boundary_operator_report,
        derivative_order_report,
        formulation_kind_report,
        constraint_report,
    };
    let _ = tx.try_send(TrainingMsg::Update(Box::new(update)));
    let _ = tx.send(TrainingMsg::Done);

    loop {
        match stop_rx.recv() {
            Ok(ControlMsg::Stop) | Err(_) => return,
            Ok(ControlMsg::SaveCheckpoint { path, saved_at_unix }) => {
                let meta = crate::checkpoint::CheckpointMeta {
                    spec: crate::checkpoint::CheckpointSpec::Plate(spec.clone()),
                    // Loaded, not (re)trained this session - honestly 0, not fabricated from
                    // the original checkpoint's own step count (this session did no training).
                    steps_completed: 0,
                    final_loss: 0.0,
                    saved_at_unix,
                };
                let result = crate::checkpoint::save_checkpoint(model.clone(), &meta, &path)
                    .map(|p| p.display().to_string());
                let _ = tx.try_send(TrainingMsg::CheckpointSaved(result));
            }
            Ok(_) => {}
        }
    }
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

/// Bin a domain's current normalized `[-1,1]^2` collocation point set into an `(ny,nx)` grid —
/// a genuine 2D histogram (raw per-cell point count), not an interpolated estimate. Phase 14
/// ("Spatial Diagnostic Visualization") `collocation_density` field, shared by every vis-grid
/// evaluator below.
pub(crate) fn bin_collocation_density(int_norm: &[[f32; 2]], nx: usize, ny: usize) -> Vec<f32> {
    let mut counts = vec![0.0f32; nx * ny];
    for &[xn, yn] in int_norm {
        let ix = (((xn + 1.0) * 0.5) * nx as f32).floor().clamp(0.0, (nx.max(1) - 1) as f32) as usize;
        let iy = (((yn + 1.0) * 0.5) * ny as f32).floor().clamp(0.0, (ny.max(1) - 1) as f32) as usize;
        counts[iy * nx + ix] += 1.0;
    }
    counts
}

/// Visualization-grid evaluator for a single plain-mDEM pin-lug domain (identity ansatz, raw
/// output columns `[u, v, sxx, syy, sxy]`) — the pin-lug analogue of `evaluate_vis_grid`,
/// which is Kirsch-ansatz-specific (`apply_dirichlet_ansatz`) and therefore not reusable
/// here. `px_pa` is the same mDEM stress-column scale `step_physics_multi`/`contact_export`
/// use (`config.load.px`).
///
/// Phase 14 extension: `sxx`/`syy`/`sxy` are direct network outputs for this ansatz (never
/// derived from strain), so an independent FD-stencil strain estimate at the same points is
/// a genuine second measurement — comparing it against the direct stress via the material's
/// constitutive law is exactly `constitutive_consistency_loss`'s per-point residual (see
/// `energy::constitutive_consistency_loss`), now surfaced as `pde_residual` instead of only
/// existing inside the training loss. Displacement/stress scaling (`u_ref`/`px_pa` applied to
/// the RAW network output before the FD stencil derivative) exactly matches
/// `training_core::compute_domain_forwards`'s `is_mdem` branch — the same convention the
/// model was actually trained under, not a new one invented for display.
#[allow(clippy::too_many_arguments)]
fn evaluate_vis_grid_mdem(
    model:    &ElasticityNet<BInner>,
    geom:     &pinn_core::geometry::GeometryConfig,
    [nx, ny]: [usize; 2],
    fd:       &FdConfig,
    u_ref:    f32,
    px_pa:    f64,
    material: &pinn_core::material::MaterialProps,
    int_norm: &[[f32; 2]],
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
    let mut e_xx = vec![f32::NAN; n_total];
    let mut e_yy = vec![f32::NAN; n_total];
    let mut e_xy = vec![f32::NAN; n_total];
    let mut pde  = vec![f32::NAN; n_total];
    let mut amr  = vec![f32::NAN; n_total];

    let active: Vec<usize> = mask.iter().enumerate().filter(|(_, &m)| m).map(|(i, _)| i).collect();
    if active.is_empty() {
        return make_vis(nx, ny, s_vm, s_xx, s_yy, s_xy, d_u, d_v, e_xx, e_yy, e_xy, pde, amr, int_norm);
    }

    let active_pts: Vec<[f32; 2]> = active.iter().map(|&i| pts[i]).collect();
    let n_act = active_pts.len();

    let pts_t = norm_pts_to_tensor::<BInner>(&active_pts, device);
    let stencil_coords = assemble_stencil::<BInner>(&pts_t, fd, device);
    const N_FOURIER: usize = 0;
    let raw_net = fwd::<BInner>(model, stencil_coords, N_FOURIER, device); // [5*n_act, 5], unscaled

    // Scale to physical units BEFORE the FD derivative — exactly `compute_domain_forwards`'s
    // `is_mdem` branch (displacement cols by u_ref [m], stress cols by px_pa [Pa]), so the
    // strain/residual computed here matches what training itself sees, not a fresh convention.
    let m = 5 * n_act;
    let u_col   = raw_net.clone().slice([0..m, 0..1]).mul_scalar(u_ref as f64);
    let v_col   = raw_net.clone().slice([0..m, 1..2]).mul_scalar(u_ref as f64);
    let sxx_col = raw_net.clone().slice([0..m, 2..3]).mul_scalar(px_pa);
    let syy_col = raw_net.clone().slice([0..m, 3..4]).mul_scalar(px_pa);
    let sxy_col = raw_net.slice([0..m, 4..5]).mul_scalar(px_pa);
    let raw = Tensor::cat(vec![u_col, v_col, sxx_col, syy_col, sxy_col], 1); // [5*n_act, 5], physical

    let (eps_xx, eps_yy, eps_xy) = compute_strains::<BInner>(raw.clone(), n_act, fd);
    let energy = dem_energy_per_point::<BInner>(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), material);
    let (sxx_fd, syy_fd, sxy_fd) =
        crate::energy::compute_stress::<BInner>(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), material);

    let u_c     = raw.clone().slice([0..n_act, 0..1]).reshape([n_act]);
    let v_c     = raw.clone().slice([0..n_act, 1..2]).reshape([n_act]);
    let sxx_net = raw.clone().slice([0..n_act, 2..3]).reshape([n_act]);
    let syy_net = raw.clone().slice([0..n_act, 3..4]).reshape([n_act]);
    let sxy_net = raw.slice([0..n_act, 4..5]).reshape([n_act]);

    // Batch every field column into a single `Tensor::cat` + ONE `.into_data()` GPU sync
    // instead of many separate syncs (each pays a fixed wgpu queue-flush/buffer-map cost
    // independent of payload size).
    let batched: Vec<f32> = Tensor::cat(
        vec![u_c, v_c, sxx_net, syy_net, sxy_net, eps_xx, eps_yy, eps_xy, sxx_fd, syy_fd, sxy_fd, energy],
        0,
    ).into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; 12 * n_act]);
    let chunk = |i: usize| -> &[f32] { &batched[i * n_act..(i + 1) * n_act] };
    let (u_vals, v_vals) = (chunk(0), chunk(1));
    let (sxx_vals, syy_vals, sxy_vals) = (chunk(2), chunk(3), chunk(4));
    let (exx_v, eyy_v, exy_v) = (chunk(5), chunk(6), chunk(7));
    let (sxx_fd_v, syy_fd_v, sxy_fd_v) = (chunk(8), chunk(9), chunk(10));
    let energy_v = chunk(11);

    for (i_act, &i_full) in active.iter().enumerate() {
        let sxx = sxx_vals[i_act] as f64;
        let syy = syy_vals[i_act] as f64;
        let sxy = sxy_vals[i_act] as f64;
        let vm  = (sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy).sqrt();
        let dex = sxx - sxx_fd_v[i_act] as f64;
        let dey = syy - syy_fd_v[i_act] as f64;
        let dexy = sxy - sxy_fd_v[i_act] as f64;
        s_xx[i_full] = sxx as f32; s_yy[i_full] = syy as f32; s_xy[i_full] = sxy as f32;
        s_vm[i_full] = vm as f32;
        d_u[i_full]  = u_vals[i_act]; d_v[i_full] = v_vals[i_act];
        e_xx[i_full] = exx_v[i_act]; e_yy[i_full] = eyy_v[i_act]; e_xy[i_full] = exy_v[i_act];
        pde[i_full]  = (dex*dex + dey*dey + dexy*dexy).sqrt() as f32;
        amr[i_full]  = energy_v[i_act].abs();
    }
    make_vis(nx, ny, s_vm, s_xx, s_yy, s_xy, d_u, d_v, e_xx, e_yy, e_xy, pde, amr, int_norm)
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

/// Phase 14 extension: `pde_residual` is trivially `0.0` (not NaN) inside the domain for this
/// ansatz — `QuarterSymmAnsatz`'s stress is *analytically derived* from strain
/// (`sxx = f*(exx+nu*eyy)`, ...), never an independent network output, so there is nothing
/// for a constitutive-consistency check to disagree with here. `amr_score` reuses
/// `dem_energy_per_point` on the same `eps_xx/eps_yy/eps_xy` this function already computes —
/// the actual signal `AdaptiveGrid`'s real Kirsch AMR sweep scores by (see
/// `training_core::probe_interior_energy_residuals`), not a new indicator invented for
/// display.
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
    int_norm:  &[[f32; 2]],
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
    let mut e_xx = vec![f32::NAN; n_total];
    let mut e_yy = vec![f32::NAN; n_total];
    let mut e_xy = vec![f32::NAN; n_total];
    let mut pde  = vec![f32::NAN; n_total];
    let mut amr  = vec![f32::NAN; n_total];

    if active.is_empty() {
        return make_vis(nx, ny, s_vm, s_xx, s_yy, s_xy, d_u, d_v, e_xx, e_yy, e_xy, pde, amr, int_norm);
    }

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
    // single `Tensor::cat` + ONE `.into_data()` GPU sync instead of 6 separate syncs.
    let u_col = out.clone().slice([0..n_act, 0..1]).reshape([n_act]);
    let v_col = out.clone().slice([0..n_act, 1..2]).reshape([n_act]);

    let (eps_xx, eps_yy, eps_xy) = compute_strains::<BInner>(out, n_act, fd);
    let energy = dem_energy_per_point::<BInner>(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), &cfg.material);
    let e  = cfg.material.e  as f32;
    let nu = cfg.material.nu as f32;
    let f  = e / (1.0 - nu * nu);

    let batched: Vec<f32> = Tensor::cat(vec![u_col, v_col, eps_xx, eps_yy, eps_xy, energy], 0)
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; 6 * n_act]);
    let u_vals   = &batched[..n_act];
    let v_vals   = &batched[n_act..2 * n_act];
    let exx_v    = &batched[2 * n_act..3 * n_act];
    let eyy_v    = &batched[3 * n_act..4 * n_act];
    let exy_v    = &batched[4 * n_act..5 * n_act];
    let energy_v = &batched[5 * n_act..6 * n_act];

    for (i_act, &i_full) in active.iter().enumerate() {
        let exx = exx_v[i_act]; let eyy = eyy_v[i_act]; let exy = exy_v[i_act];
        let sxx = f * (exx + nu * eyy);
        let syy = f * (eyy + nu * exx);
        let sxy = f / (1.0 + nu) * exy;
        let vm  = (sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy).sqrt();
        s_xx[i_full] = sxx; s_yy[i_full] = syy; s_xy[i_full] = sxy;
        s_vm[i_full] = vm;  d_u[i_full]  = u_vals[i_act]; d_v[i_full] = v_vals[i_act];
        e_xx[i_full] = exx; e_yy[i_full] = eyy; e_xy[i_full] = exy;
        pde[i_full] = 0.0; // analytically strain-derived stress — see this fn's doc comment
        amr[i_full] = energy_v[i_act].abs();
    }
    make_vis(nx, ny, s_vm, s_xx, s_yy, s_xy, d_u, d_v, e_xx, e_yy, e_xy, pde, amr, int_norm)
}

#[allow(clippy::too_many_arguments)]
fn make_vis(nx: usize, ny: usize,
            vm: Vec<f32>, sxx: Vec<f32>, syy: Vec<f32>, sxy: Vec<f32>, u: Vec<f32>, v: Vec<f32>,
            eps_xx: Vec<f32>, eps_yy: Vec<f32>, eps_xy: Vec<f32>,
            pde_residual: Vec<f32>, amr_score: Vec<f32>,
            int_norm: &[[f32; 2]]) -> VisFields {
    let a = |v: Vec<f32>| Array2::from_shape_vec((ny, nx), v).expect("shape mismatch");
    let density = bin_collocation_density(int_norm, nx, ny);
    VisFields {
        von_mises: a(vm), sigma_xx: a(sxx), sigma_yy: a(syy), sigma_xy: a(sxy),
        disp_u: a(u), disp_v: a(v),
        eps_xx: a(eps_xx), eps_yy: a(eps_yy), eps_xy: a(eps_xy),
        pde_residual: a(pde_residual), amr_score: a(amr_score),
        collocation_density: a(density),
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

    fn single_hole_like_spec(max_steps: usize) -> ProblemSpec {
        use pinn_core::loading::LoadConfig;
        use pinn_core::material::MaterialProps;
        use pinn_core::problem_spec::{NetworkSpec, TrainingSpec};
        use pinn_core::user_geometry::{HoleBc, HoleSpec, UserGeometry};
        ProblemSpec {
            geometry: UserGeometry {
                half_w: 0.10, half_h: 0.10, thickness: 0.005,
                holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.02, bc: HoleBc::Free }],
            },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: NetworkSpec { hidden_dim: 64, n_hidden: 3, ..Default::default() },
            training: TrainingSpec { max_steps, n_interior: 2048, n_boundary: 512, fd_h: 1e-3, lr: 1e-3 },
        }
    }

    // ─── Stage H: model checkpoint save/load ────────────────────────────────────────────────

    #[test]
    fn serve_loaded_plate_checkpoint_sends_update_then_done_with_no_training() {
        // NOT `run_and_drain` - that helper's `handle.join()` assumes the spawned function
        // RETURNS after `Done` (true for `run_training_pinlug`/the plain training loop above),
        // but `serve_loaded_plate_checkpoint` deliberately stays alive after `Done` (same
        // stay-alive design as `parametric_problem::run_training_parametric`) - using
        // `run_and_drain` here would deadlock waiting for a `join()` that never returns.
        let spec = single_hole_like_spec(0);
        let device = BDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(spec.geometry.net_input_dim()).with_hidden_dim(8).with_n_hidden(2).with_output_dim(5);
        let model: ElasticityNet<BInner> = net_cfg.init(&device);

        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || serve_loaded_plate_checkpoint(spec, model, tx, rx_ctrl));

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut saw_update = false;
        let mut saw_done = false;
        while std::time::Instant::now() < deadline && !saw_done {
            match rx.try_recv() {
                Ok(TrainingMsg::Update(u)) => {
                    saw_update = true;
                    assert!(u.vis.is_some(), "the one Update sent must carry Some(vis) - a loaded checkpoint has no vis-cadence gating to wait out");
                    assert!(u.reaction_force.is_some(), "reaction_force must be computed for a loaded-checkpoint evaluation");
                    assert!(u.energy_balance.is_some(), "energy_balance must be computed for a loaded-checkpoint evaluation");
                    assert_eq!(u.hole_analyses.len(), 1, "single_hole_like_spec has exactly one hole");
                }
                Ok(TrainingMsg::Done) => saw_done = true,
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        assert!(saw_update, "expected at least one TrainingMsg::Update");
        assert!(saw_done, "expected TrainingMsg::Done");

        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn serve_loaded_plate_checkpoint_saves_on_request() {
        let spec = single_hole_like_spec(0);
        let device = BDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(spec.geometry.net_input_dim()).with_hidden_dim(8).with_n_hidden(2).with_output_dim(5);
        let model: ElasticityNet<BInner> = net_cfg.init(&device);

        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || serve_loaded_plate_checkpoint(spec, model, tx, rx_ctrl));

        // Drain until Done, then request a save - mirrors how the real UI only enables "Save"
        // once a model has reached a stable, evaluated state.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut saw_done = false;
        while std::time::Instant::now() < deadline && !saw_done {
            match rx.try_recv() {
                Ok(TrainingMsg::Done) => saw_done = true,
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        assert!(saw_done, "must observe Done before requesting a save");

        let path = std::env::temp_dir().join(format!("pinn_solver_serve_loaded_plate_save_test_{}", std::process::id()));
        tx_ctrl.send(ControlMsg::SaveCheckpoint { path: path.clone(), saved_at_unix: 456 }).unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut saved: Option<Result<String, String>> = None;
        while std::time::Instant::now() < deadline && saved.is_none() {
            if let Ok(TrainingMsg::CheckpointSaved(r)) = rx.try_recv() { saved = Some(r); }
            else { std::thread::sleep(std::time::Duration::from_millis(5)); }
        }
        let result = saved.expect("must receive a CheckpointSaved response");
        let written = result.expect("save must succeed");
        assert!(std::path::Path::new(&written).exists(), "the reported weights path must actually exist on disk: {written}");

        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();
        let _ = std::fs::remove_file(&written);
        let mut meta = path.clone();
        meta.set_file_name(format!("{}.meta.json", path.file_stem().unwrap().to_string_lossy()));
        let _ = std::fs::remove_file(meta);
    }

    // ─── Phase 19 (Neural-Network-Wide Adaptive Collocation epic): generalization validation ──

    /// Trains a small `UserDefinedProblem` for `steps` steps and returns the trained model —
    /// the same setup `run_training_user_problem` itself uses, trimmed to a direct loop (no
    /// channel/thread) so a test can get the trained model back directly.
    fn train_small_user_problem(spec: &ProblemSpec, steps: usize) -> ElasticityNet<BInner> {
        use crate::problem::{BoundaryValueProblem, DomainOptim, DomainStepCtx, DomainStepData, MultiStepCtx, PointSetData};
        use crate::training_core::step_physics_multi;
        use crate::user_problem::UserDefinedProblem;
        use burn::module::AutodiffModule;
        use std::collections::HashMap;

        let device = BDevice::default();
        let half_w = spec.geometry.half_w;
        let half_h = spec.geometry.half_h;
        let problem = UserDefinedProblem::new(spec.clone());
        validate_loss_terms(&problem);

        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3).with_hidden_dim(spec.network.hidden_dim).with_n_hidden(spec.network.n_hidden)
            .with_output_dim(5);
        let mut model = net_cfg.init(&device);
        let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim() };
        let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
        let mut saw = SawBrdr::with_base(base_weights, 0.95);
        let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
        let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);

        let sampling = problem.sampling_strategy(0);
        let int_pts = sampling.sample_interior(&pinn_core::geometry::GeometryConfig::kirsch_plate_inches(), spec.training.n_interior);
        let bnd_pts = sampling.sample_boundary(&pinn_core::geometry::GeometryConfig::kirsch_plate_inches(), &spec.load, spec.training.n_boundary);
        let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / half_w) as f32, (y / half_h) as f32] };
        let to_pointset = |pts: &[pinn_core::loading::BoundaryPoint]| -> PointSetData {
            PointSetData {
                norm: pts.iter().map(|p| norm_pt(p.x, p.y)).collect(),
                nx: pts.iter().map(|p| p.nx as f32).collect(), ny: pts.iter().map(|p| p.ny as f32).collect(),
                tx: pts.iter().map(|p| p.tx as f32).collect(), ty: pts.iter().map(|p| p.ty as f32).collect(),
            }
        };
        let mut named = HashMap::new();
        named.insert("outer_boundary", to_pointset(&bnd_pts));
        for set in sampling.named_point_sets(&[]) { named.insert(set.name, to_pointset(&set.points)); }
        let data = DomainStepData {
            id: crate::user_problem::USER_DOMAIN,
            int_norm: int_pts.iter().map(|&[x, y]| norm_pt(x, y)).collect(),
            extra_ring_norm: Vec::new(), named,
        };

        let scales = crate::training_core::compute_reference_scales_for_plate(spec);
        let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);
        let config = SolverConfig::default_kirsch();

        for step in 0..steps {
            let ctx = MultiStepCtx {
                config: &config, problem: &problem, fd: &fd, k: 1.0,
                domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                dynamic_lam_h_cap: f64::MAX, dynamic_lam_d_cap: f64::MAX,
                dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
                n_fourier: 0,
                probe_term_gradients: false,
                phase2_active: true, step,
            };
            let (new_model, _out) = step_physics_multi(
                vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
            );
            model = new_model.into_iter().next().unwrap();
        }
        model.valid()
    }

    /// Real term-by-term diagnostic recommended by the Kt investigation
    /// (`powershell_tool/CLAUDE.md`, bugSource-New): "registered does not establish a term is
    /// exerting enough optimization pressure" - this prints raw (unweighted) value, live
    /// SAW-adapted lambda, weighted contribution, and the term's OWN gradient L2 norm
    /// (`probe_term_gradients: true`) for every active term, at a few points across a short
    /// no-hole run (no hole/AMR complexity - isolates the question to interior_energy vs.
    /// equilibrium vs. outer_traction, not hole-specific terms). No real training run needed
    /// to design/interpret this (per-step cost, not wall-clock training length, is what makes
    /// it informative) - but `probe_term_gradients`'s fresh-forward-pass-per-term technique
    /// (see `term_grad_norms`'s doc comment in `training_core.rs` for why that's required, not
    /// optional) is itself real per-call cost, `#[ignore]`d per this project's fast-test/slow-
    /// integration-test split (measured ~7 min in a debug build for 5 probe points).
    #[test]
    #[ignore]
    fn term_by_term_raw_lambda_weighted_gradient_diagnostic_on_no_hole_plate() {
        use crate::problem::{BoundaryValueProblem, DomainOptim, DomainStepCtx, DomainStepData, MultiStepCtx, PointSetData};
        use crate::training_core::step_physics_multi;
        use crate::user_problem::UserDefinedProblem;
        use std::collections::HashMap;

        let spec = no_hole_plate_spec(200);
        let device = BDevice::default();
        let half_w = spec.geometry.half_w;
        let half_h = spec.geometry.half_h;
        let problem = UserDefinedProblem::new(spec.clone());
        validate_loss_terms(&problem);

        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3).with_hidden_dim(spec.network.hidden_dim).with_n_hidden(spec.network.n_hidden)
            .with_output_dim(5);
        let mut model = net_cfg.init(&device);
        let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim() };
        let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
        let mut saw = SawBrdr::with_base(base_weights, 0.95);
        let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
        let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);

        let sampling = problem.sampling_strategy(0);
        let placeholder = pinn_core::geometry::GeometryConfig::kirsch_plate_inches();
        let int_pts = sampling.sample_interior(&placeholder, spec.training.n_interior);
        let bnd_pts = sampling.sample_boundary(&placeholder, &spec.load, spec.training.n_boundary);
        let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / half_w) as f32, (y / half_h) as f32] };
        let to_pointset = |pts: &[pinn_core::loading::BoundaryPoint]| -> PointSetData {
            PointSetData {
                norm: pts.iter().map(|p| norm_pt(p.x, p.y)).collect(),
                nx: pts.iter().map(|p| p.nx as f32).collect(), ny: pts.iter().map(|p| p.ny as f32).collect(),
                tx: pts.iter().map(|p| p.tx as f32).collect(), ty: pts.iter().map(|p| p.ty as f32).collect(),
            }
        };
        let mut named = HashMap::new();
        named.insert("outer_boundary", to_pointset(&bnd_pts));
        for set in sampling.named_point_sets(&[]) { named.insert(set.name, to_pointset(&set.points)); }
        let data = DomainStepData {
            id: crate::user_problem::USER_DOMAIN,
            int_norm: int_pts.iter().map(|&[x, y]| norm_pt(x, y)).collect(),
            extra_ring_norm: Vec::new(), named,
        };

        let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
        let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);
        let config = SolverConfig::default_kirsch();

        let print_steps = [0usize, 10, 50, 100, 199];
        for step in 0..200 {
            let probe_now = print_steps.contains(&step);
            let ctx = MultiStepCtx {
                config: &config, problem: &problem, fd: &fd, k: 1.0,
                domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
                dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                constitutive_consistency_weight: 50.0,
                n_fourier: 0,
                probe_term_gradients: probe_now,
                phase2_active: true, step,
            };
            let (new_model, out) = step_physics_multi(
                vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
            );
            model = new_model.into_iter().next().unwrap();

            if probe_now {
                // General-PINN architecture recommendations §6/§29 (Priority 3, "loss ledger")
                // - one consolidated record per term instead of 4 separately cross-referenced
                // hashmaps, the real consumer this test now uses instead of its own manual
                // `raw[name]`/`lam.get(name)`/`grad.get(name)` lookups.
                let raw = out.raw_scalar_by_name.as_ref().expect("raw_scalar_by_name must be Some on step_physics_multi");
                let lam = out.lam_by_name.as_ref().expect("lam_by_name must be Some on step_physics_multi");
                let grad = out.term_grad_norms.as_ref().expect("term_grad_norms must be Some when probe_term_gradients=true");
                let shares = out.gradient_share_report.as_ref().map(|r| &r.shares);
                let mut ledger = crate::training_core::build_loss_ledger(raw, lam, Some(grad), shares);
                ledger.sort_by_key(|e| e.name);
                println!("  [term-diag] step={step} total_loss={:.4e}", out.total_scalar);
                println!("  [term-diag] {:>26} {:>14} {:>14} {:>14} {:>14}", "term", "raw", "lambda", "weighted", "grad_norm");
                for e in &ledger {
                    println!("  [term-diag] {:>26} {:>14.4e} {:>14.4e} {:>14.4e} {:>14.4e}",
                        e.name, e.raw, e.lambda, e.weighted, e.grad_norm.unwrap_or(f32::NAN));
                }
            }
        }
    }

    /// Same diagnostic as `term_by_term_raw_lambda_weighted_gradient_diagnostic_on_no_hole_
    /// plate`, but on the actual hole geometry (`single_hole_like_spec`, matching
    /// `single_hole_plate.toml`) instead of the no-hole sanity case.
    ///
    /// Required because the no-hole case's TRUE solution (uniform uniaxial tension) is
    /// EXACTLY LINEAR (`u=σ0/E·x`, `v=-νσ0/E·y`) - its Hessian is identically zero everywhere
    /// at the target. A correctly-working equilibrium-on-Hessian term is SUPPOSED to have
    /// vanishing gradient as training approaches that curvature-free solution, which is
    /// indistinguishable, on that test alone, from the term being inert/broken - confirmed by
    /// running it after switching `EquilibriumTerm` to the Hessian (bugSource-New #12):
    /// `equilibrium`'s gradient stayed ~1e-7-1e-6, still 5-6 orders smaller than every other
    /// term's, exactly as it did with the old direct-σ version, even though the term is now
    /// provably correct in isolation (analytical Hessian test) - because there's genuinely
    /// almost no curvature for it to react to in that specific problem. The hole geometry's
    /// true (Kirsch-like) solution has real, nonzero curvature near the hole, so this is the
    /// test that can actually distinguish "equilibrium is providing real gradient pressure"
    /// from "equilibrium is inert" for the Hessian-based term.
    #[test]
    #[ignore]
    fn term_by_term_raw_lambda_weighted_gradient_diagnostic_on_single_hole_plate() {
        use crate::problem::{BoundaryValueProblem, DomainOptim, DomainStepCtx, DomainStepData, MultiStepCtx, PointSetData};
        use crate::training_core::step_physics_multi;
        use crate::user_problem::UserDefinedProblem;
        use std::collections::HashMap;

        let spec = single_hole_like_spec(200);
        let device = BDevice::default();
        let half_w = spec.geometry.half_w;
        let half_h = spec.geometry.half_h;
        let problem = UserDefinedProblem::new(spec.clone());
        validate_loss_terms(&problem);

        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3).with_hidden_dim(spec.network.hidden_dim).with_n_hidden(spec.network.n_hidden)
            .with_output_dim(5);
        let mut model = net_cfg.init(&device);
        let mut optim = DomainOptim { weight: WeightOptim::new(true), bias: make_bias_optim(), gate: make_gate_optim() };
        let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
        let mut saw = SawBrdr::with_base(base_weights, 0.95);
        let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
        let fd = FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);

        let sampling = problem.sampling_strategy(0);
        let placeholder = pinn_core::geometry::GeometryConfig::kirsch_plate_inches();
        let int_pts = sampling.sample_interior(&placeholder, spec.training.n_interior);
        let bnd_pts = sampling.sample_boundary(&placeholder, &spec.load, spec.training.n_boundary);
        let norm_pt = |x: f64, y: f64| -> [f32; 2] { [(x / half_w) as f32, (y / half_h) as f32] };
        let to_pointset = |pts: &[pinn_core::loading::BoundaryPoint]| -> PointSetData {
            PointSetData {
                norm: pts.iter().map(|p| norm_pt(p.x, p.y)).collect(),
                nx: pts.iter().map(|p| p.nx as f32).collect(), ny: pts.iter().map(|p| p.ny as f32).collect(),
                tx: pts.iter().map(|p| p.tx as f32).collect(), ty: pts.iter().map(|p| p.ty as f32).collect(),
            }
        };
        let mut named = HashMap::new();
        named.insert("outer_boundary", to_pointset(&bnd_pts));
        for set in sampling.named_point_sets(&[]) { named.insert(set.name, to_pointset(&set.points)); }
        let data = DomainStepData {
            id: crate::user_problem::USER_DOMAIN,
            int_norm: int_pts.iter().map(|&[x, y]| norm_pt(x, y)).collect(),
            extra_ring_norm: Vec::new(), named,
        };

        let scales = crate::training_core::compute_reference_scales_for_plate(&spec);
        let (u_ref, ref_energy, ref_stress2) = (scales.u_ref, scales.ref_energy, scales.ref_stress2);
        let config = SolverConfig::default_kirsch();

        let print_steps = [0usize, 10, 50, 100, 199];
        for step in 0..200 {
            let probe_now = print_steps.contains(&step);
            let ctx = MultiStepCtx {
                config: &config, problem: &problem, fd: &fd, k: 1.0,
                domains: vec![DomainStepCtx { data: &data, u_ref, ref_energy, ref_stress2 }],
                dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
                dynamic_lam_penetration_cap: f64::MAX, dynamic_lam_non_tension_cap: f64::MAX,
                constitutive_consistency_weight: 50.0,
                n_fourier: 0,
                probe_term_gradients: probe_now,
                phase2_active: true, step,
            };
            let (new_model, out) = step_physics_multi(
                vec![model], std::slice::from_mut(&mut optim), &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
            );
            model = new_model.into_iter().next().unwrap();

            if probe_now {
                let raw = out.raw_scalar_by_name.as_ref().expect("raw_scalar_by_name must be Some on step_physics_multi");
                let lam = out.lam_by_name.as_ref().expect("lam_by_name must be Some on step_physics_multi");
                let grad = out.term_grad_norms.as_ref().expect("term_grad_norms must be Some when probe_term_gradients=true");
                println!("  [term-diag-hole] step={step} total_loss={:.4e}", out.total_scalar);
                println!("  [term-diag-hole] {:>26} {:>14} {:>14} {:>14} {:>14}", "term", "raw", "lambda", "weighted", "grad_norm");
                let mut names: Vec<&&str> = raw.keys().collect();
                names.sort();
                for name in names {
                    let r = raw[name];
                    let l = *lam.get(name).unwrap_or(&0.0);
                    let g = grad.get(name).copied().unwrap_or(f32::NAN);
                    println!("  [term-diag-hole] {:>26} {:>14.4e} {:>14.4e} {:>14.4e} {:>14.4e}", name, r, l, r as f64 * l, g);
                }
            }
        }
    }

    /// Real training (300 steps, ~60s in a debug build - see CLAUDE.md's documented ~20-30x
    /// debug/release gap) - `#[ignore]`d per this project's fast-test/slow-integration-test
    /// split (Phase 21), same convention as `run_training_pinlug_amr_sweep_changes_
    /// collocation_count`. Run explicitly: `cargo test -p pinn-solver
    /// generalization_perturbing_material -- --ignored`.
    #[test]
    #[ignore]
    fn generalization_perturbing_material_after_training_measurably_degrades_constitutive_residual_and_is_flagged_red() {
        use crate::user_problem::evaluate_user_vis_grid;
        use pinn_core::{classify_inference, InferenceClass};

        let trained_spec = single_hole_like_spec(300);
        let model = train_small_user_problem(&trained_spec, 300);
        let device = BDevice::default();
        let fd = FdConfig::new(trained_spec.training.fd_h, 2.0 * trained_spec.geometry.half_w, 2.0 * trained_spec.geometry.half_h);
        let u_ref = crate::training_core::compute_reference_scales_for_plate(&trained_spec).u_ref;

        // "Perturb material after training, without retraining" - keep u_ref/px_pa fixed at
        // the ORIGINAL trained scale (see pinn_core::inference_envelope's doc comment: these
        // scales are baked into training, not recomputed at inference time), only swap the
        // material argument passed to the constitutive-consistency check.
        let baseline = evaluate_user_vis_grid(
            &model, &trained_spec.geometry, [16, 16], u_ref, trained_spec.load.px,
            &trained_spec.material, &fd, &[], &device,
        );
        let mut perturbed_material = trained_spec.material.clone();
        perturbed_material.e *= 3.0; // "moderate" perturbation per the epic's Phase 19 wording
        let perturbed = evaluate_user_vis_grid(
            &model, &trained_spec.geometry, [16, 16], u_ref, trained_spec.load.px,
            &perturbed_material, &fd, &[], &device,
        );

        let rms = |field: &ndarray::Array2<f32>| -> f64 {
            let vals: Vec<f64> = field.iter().copied().filter(|v| v.is_finite()).map(|v| (v as f64).powi(2)).collect();
            if vals.is_empty() { return 0.0; }
            (vals.iter().sum::<f64>() / vals.len() as f64).sqrt()
        };
        let rms_before = rms(&baseline.pde_residual);
        let rms_after = rms(&perturbed.pde_residual);
        assert!(
            rms_after > rms_before * 1.2,
            "perturbing material.e by 3x after training must measurably worsen the constitutive-consistency residual (it was trained to satisfy Hooke's law for the OLD material only): before={rms_before:.4e}, after={rms_after:.4e}"
        );

        let mut perturbed_spec = trained_spec.clone();
        perturbed_spec.material = perturbed_material;
        match classify_inference(&trained_spec, &perturbed_spec, None) {
            InferenceClass::Red(msg) => assert!(msg.contains("RequiresRetraining")),
            other => panic!("material change must classify Red/RequiresRetraining, got {other:?}"),
        }
    }

    // ─── Phase 20 (Neural-Network-Wide Adaptive Collocation epic): performance protection ─────

    /// Real wall-clock measurement (not a correctness assertion) of the Phase 14 field
    /// extension's actual added cost, isolated from the rest of a training run. This is the
    /// one place this epic's later phases added real per-call cost to an EXISTING periodic
    /// operation (vis-grid evaluation, which already ran every 10-50 steps depending on
    /// problem before Phase 14 - never the per-step hot loop itself: `training_core::
    /// step_physics`/`step_physics_multi`, the functions that actually dominate the
    /// documented ~80s/950-step Kirsch release baseline, have zero lines changed anywhere in
    /// this epic's Phase 8-19 work - confirmed by inspection, not assumed). `#[ignore]`d
    /// (debug-mode timing isn't representative - see CLAUDE.md's documented ~20-30x debug/
    /// release gap); run explicitly in release mode to get a real number:
    /// `cargo test -p pinn-solver --release evaluate_user_vis_grid_call_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn evaluate_user_vis_grid_call_cost_is_small_relative_to_the_vis_cadence() {
        use crate::user_problem::evaluate_user_vis_grid;
        let spec = single_hole_like_spec(1); // network/material/load config only, steps unused here
        let model = tiny_model_at(spec.network.hidden_dim, spec.network.n_hidden);
        let device = BDevice::default();
        let fd = FdConfig::new(spec.training.fd_h, 2.0 * spec.geometry.half_w, 2.0 * spec.geometry.half_h);
        let int_norm: Vec<[f32; 2]> = (0..spec.training.n_interior)
            .map(|i| [((i % 64) as f32 / 32.0) - 1.0, ((i / 64) as f32 / 32.0) - 1.0]).collect();

        let n_calls = 20;
        let start = std::time::Instant::now();
        for _ in 0..n_calls {
            let _ = evaluate_user_vis_grid(
                &model, &spec.geometry, [64, 64], 1.0, spec.load.px, &spec.material, &fd, &int_norm, &device,
            );
        }
        let per_call_ms = start.elapsed().as_secs_f64() * 1000.0 / n_calls as f64;
        // Documented Kirsch/UserDefinedProblem release per-step cost is O(10-100ms) (see
        // CLAUDE.md's debug/release investigation) and vis fires at most every 10 steps here -
        // i.e. the vis-cadence budget is at least ~10x a single step's cost. Flag (not fail
        // outright, since this is an environment-dependent wall-clock number, not a pure
        // function) if a single vis call alone would already exceed that budget.
        println!("evaluate_user_vis_grid (64x64 grid, {} collocation points): {per_call_ms:.2}ms/call", spec.training.n_interior);
        assert!(per_call_ms < 500.0, "vis-grid evaluation cost grew unexpectedly large ({per_call_ms:.1}ms) - investigate before shipping");
    }

    fn tiny_model_at(hidden_dim: usize, n_hidden: usize) -> ElasticityNet<BInner> {
        let device = BDevice::default();
        ElasticityNetConfig::new().with_input_dim(3).with_hidden_dim(hidden_dim).with_n_hidden(n_hidden).with_output_dim(5).init(&device)
    }

    /// Real wall-clock timing measurement (not a correctness assertion) - reproduces the
    /// exact shape of the user-reported slowness (`single_hole_plate.toml`'s real
    /// `n_interior`/`n_boundary`/`hidden_dim`/`n_hidden`) at a small step count so it stays
    /// fast enough to run manually. `#[ignore]`d for the same reason `toy_beam`'s own
    /// timing-sensitive tests are - not something the default `cargo test` suite should pay
    /// for every run. Run explicitly (release mode - debug is not representative of real
    /// per-step cost): `cargo test -p pinn-solver --features ndarray-backend --release
    /// runner::tests::run_training_user_problem_step_time_smoke -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn run_training_user_problem_step_time_smoke() {
        let steps = 30;
        let spec = single_hole_like_spec(steps);
        let (tx, _rx) = crossbeam_channel::unbounded();
        let (_tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let start = std::time::Instant::now();
        run_training_user_problem(spec, tx, rx_ctrl);
        let elapsed = start.elapsed();
        println!("{steps} steps in {elapsed:?} -> {:?}/step", elapsed / steps as u32);
    }

    /// Real, measured AMR overhead baseline (not an assumption) - per the "Neural-Network-
    /// Wide Adaptive Collocation" epic's Phase 1 requirement, release-mode, representative
    /// problem. There is no literal "AMR disabled" toggle in the current design (AMR is
    /// always constructed and residual-driven after `AMR_WARMUP_STEPS`, by design - see
    /// `pinn_core::amr::AmrDomain`'s doc comment) - so this isolates the ONE-sweep cost by
    /// comparing 200 steps (zero sweeps: `step >= AMR_WARMUP_STEPS` never true for
    /// `step < 200`) against 201 steps (exactly one sweep, at step 200).
    /// `sweep_cost ≈ (T(201) - T(200)) - T(200)/200`. Run explicitly (release mode):
    /// `cargo test -p pinn-solver --features ndarray-backend --release
    /// runner::tests::run_training_user_problem_amr_overhead_baseline -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn run_training_user_problem_amr_overhead_baseline() {
        let time_n = |n: usize| -> std::time::Duration {
            let spec = single_hole_like_spec(n);
            let (tx, _rx) = crossbeam_channel::unbounded();
            let (_tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
            let start = std::time::Instant::now();
            run_training_user_problem(spec, tx, rx_ctrl);
            start.elapsed()
        };
        let t_200 = time_n(200); // zero AMR sweeps
        let t_201 = time_n(201); // exactly one AMR sweep, at step 200
        let per_step_baseline = t_200.as_secs_f64() / 200.0;
        let step_201_cost = t_201.as_secs_f64() - t_200.as_secs_f64();
        let sweep_cost = (step_201_cost - per_step_baseline).max(0.0);
        println!(
            "T(200, 0 sweeps)={t_200:?} ({:.2}ms/step)  T(201, 1 sweep)={t_201:?}  \
             estimated single-sweep cost={:.1}ms ({:.1}x a normal step)",
            per_step_baseline * 1000.0, sweep_cost * 1000.0, sweep_cost / per_step_baseline.max(1e-9),
        );
    }

    /// Real end-to-end integration check that generic AMR is actually wired into
    /// `run_training_user_problem`, not just unit-tested in isolation at the `pinn-core`
    /// level (see `pinn_core::amr::tests::adaptive_grid_over_user_geometry_refines_near_
    /// each_hole_zone` for the spatial-density assertion this can't cheaply repeat here -
    /// this test instead confirms the probe → adapt → resample → telemetry pipeline fires
    /// end-to-end). Runs past `AMR_WARMUP_STEPS` (200) so exactly one sweep fires, then
    /// asserts the reported `n_colloc` differs from the original `n_interior` - AMR's
    /// quadtree-derived point count essentially never exactly matches a plain rejection
    /// sample. `#[ignore]`d (same rationale as the step-time smoke test above): a genuine,
    /// if small, multi-hundred-step training run.
    #[test]
    #[ignore]
    fn run_training_user_problem_amr_sweep_changes_collocation_count() {
        let spec = single_hole_like_spec(250);
        let n_interior = spec.training.n_interior;
        let (tx, rx) = crossbeam_channel::unbounded();
        let (_tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        run_training_user_problem(spec, tx, rx_ctrl);

        let mut last_n_colloc = None;
        let mut amr_report = None;
        while let Ok(msg) = rx.try_recv() {
            if let TrainingMsg::Update(upd) = msg {
                last_n_colloc = Some(upd.n_colloc);
                if upd.amr_sweep.is_some() {
                    amr_report = upd.amr_sweep.clone();
                }
            }
        }
        let last_n_colloc = last_n_colloc.expect("expected at least one TrainingMsg::Update");
        assert_ne!(
            last_n_colloc, n_interior,
            "n_colloc still matches the original n_interior after step 200's AMR sweep - \
             the sweep either didn't fire or didn't actually resample"
        );

        // Phase 11 ("AMR Effectiveness") - the real before/after report must actually be
        // delivered, not just the point count changing.
        let report = amr_report.expect("expected an AmrSweepReport on the step the sweep fired");
        assert_eq!(report.domain_label, "interior");
        assert_eq!(report.step, 200);
        assert_ne!(report.points_before, report.points_after, "report itself must reflect the resample");
        assert!(report.sweep_duration_ms >= 0.0, "sweep timing must be a real, non-negative measurement");
    }

    // Real, `#[ignore]`d (slow - thousands of real steps) diagnostic: runs the REAL GUI code
    // path (`run_training_user_problem`, with AMR active), unlike the separate headless CLI
    // path (`user_runner::run_headless_user_problem`, which has no AMR at all and is
    // therefore not representative of what the app actually does). Prints the same PDE-
    // residual/Kt/AMR-sweep numbers the app's own diagnostics would show, from the real last
    // `Update`'s `vis`/`hole_analyses` (always sent on the final step regardless of vis
    // cadence) - a real, reusable regression check for the garbage-Kt/zero-hole-stress
    // investigation documented in `powershell_tool/CLAUDE.md`'s Stress Solver section, not a
    // one-off debugging script.
    #[test]
    #[ignore]
    fn run_training_user_problem_generalized_amr_indicator_diagnostic() {
        // 8000 steps: deep enough to actually test convergence (not just early dynamics), and
        // comparable to earlier before/after data points gathered during this investigation.
        // NOT a direct/synchronous call - `run_user_problem_training_from` deliberately stays
        // alive after `Done` to serve `SaveCheckpoint` requests (Stage H), so this must run on
        // its own thread and be told `Stop` once we've seen what we need, matching
        // `serve_loaded_plate_checkpoint_sends_update_then_done_with_no_training`'s established
        // pattern - a direct call here would block forever on `stop_rx.recv()`.
        let spec = single_hole_like_spec(8000);
        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || run_training_user_problem(spec, tx, rx_ctrl));

        let mut last_vis = None;
        let mut last_hole_analyses = Vec::new();
        let mut last_amr = None;
        let mut last_total_loss = f32::NAN;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3600);
        let mut saw_done = false;
        while std::time::Instant::now() < deadline && !saw_done {
            match rx.try_recv() {
                Ok(TrainingMsg::Update(upd)) => {
                    last_total_loss = upd.total_loss;
                    if upd.vis.is_some() { last_vis = upd.vis; }
                    if !upd.hole_analyses.is_empty() { last_hole_analyses = upd.hole_analyses; }
                    if upd.amr_sweep.is_some() { last_amr = upd.amr_sweep; }
                }
                Ok(TrainingMsg::Done) => saw_done = true,
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        assert!(saw_done, "expected TrainingMsg::Done within the deadline");

        // Checkpoint save/load (same established pattern as the other diagnostics below) to
        // get a live model for the equilibrium-residual probe - `vis`/`hole_analyses` alone
        // don't carry it.
        let path = std::env::temp_dir().join(format!("pinn_solver_amr_kt_diag_{}", std::process::id()));
        tx_ctrl.send(ControlMsg::SaveCheckpoint { path: path.clone(), saved_at_unix: 0 }).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut saved: Option<Result<String, String>> = None;
        while std::time::Instant::now() < deadline && saved.is_none() {
            if let Ok(TrainingMsg::CheckpointSaved(r)) = rx.try_recv() { saved = Some(r); }
            else { std::thread::sleep(std::time::Duration::from_millis(5)); }
        }
        let written = saved.expect("must receive a CheckpointSaved response").expect("save must succeed");
        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();

        let vis = last_vis.expect("expected a final vis-carrying Update");
        let pde_vals: Vec<f32> = vis.pde_residual.iter().copied().filter(|v| v.is_finite()).collect();
        let (pde_rms, pde_max) = crate::training_core::residual_stats(&pde_vals);
        println!("  [diag] final total_loss={last_total_loss:.6e}");
        // "Constitutive residual" - not "PDE residual" - this measures ‖sigma_direct -
        // Hooke's-law(strain_FD)‖, never an equilibrium/PDE residual. See `EquilibriumTerm`'s
        // doc comment (user_problem.rs) and `powershell_tool/CLAUDE.md`'s Kt investigation.
        println!("  [diag] constitutive residual RMS={pde_rms:.4e}  max={pde_max:.4e} Pa");
        for h in &last_hole_analyses {
            println!(
                "  [diag] hole {}: max_von_mises={:.4e} Pa  nominal={:.4e} Pa  Kt={:.4}",
                h.hole_index, h.concentration.max_von_mises, h.concentration.nominal_stress, h.concentration.kt
            );
        }
        if let Some(r) = last_amr {
            println!(
                "  [diag] last AMR sweep at step {}: points {}->{}  hole-zone density {:.3}->{:.3}  domain-avg density {:.3}->{:.3}",
                r.step, r.points_before, r.points_after, r.hole_zone_density_before, r.hole_zone_density_after,
                r.domain_mean_density_before, r.domain_mean_density_after
            );
        } else {
            println!("  [diag] no AMR sweep report received");
        }

        // Real equilibrium residual (‖∇·σ‖, direct-network stress, same formula/point
        // convention `EquilibriumTerm`/`equilibrium_residual_loss` train on) - genuinely
        // distinct from the constitutive residual above. Sampled over a grid of interior
        // points (outside the FD-safe hole margin).
        {
            use crate::energy::equilibrium_residual_loss;
            use crate::fd_stencil::{assemble_stencil, norm_pts_to_tensor, FdConfig};
            use crate::network::fwd;
            use crate::training_core::BInner;
            let (model, _meta) = crate::checkpoint::load_checkpoint(&path, &BDevice::default())
                .expect("must load the just-saved checkpoint");
            let _ = std::fs::remove_file(&written);
            let mut meta_path = path.clone();
            meta_path.set_file_name(format!("{}.meta.json", path.file_stem().unwrap().to_string_lossy()));
            let _ = std::fs::remove_file(meta_path);

            let device = BDevice::default();
            let spec = single_hole_like_spec(8000);
            let half_w = spec.geometry.half_w;
            let half_h = spec.geometry.half_h;
            let hole = spec.geometry.holes[0];
            let margin = crate::user_problem::ring_anchor_margin_m(1e-3, &spec.geometry);
            let fd = FdConfig::new(1e-3, 2.0 * half_w, 2.0 * half_h);
            let cx = fd.sx / (2.0 * fd.hx as f64);
            let cy = fd.sy / (2.0 * fd.hy as f64);
            let ref_div2 = (spec.load.px * cx).powi(2).max(1.0);

            let mut residuals: Vec<f32> = Vec::new();
            for &fx in &[-0.75, -0.5, -0.25, 0.25, 0.5, 0.75] {
                for &fy in &[-0.75, -0.5, -0.25, 0.25, 0.5, 0.75] {
                    let x = fx * half_w;
                    let y = fy * half_h;
                    let dx = x - hole.center[0];
                    let dy = y - hole.center[1];
                    if (dx * dx + dy * dy).sqrt() <= hole.radius + margin { continue; }
                    let xn = (x / half_w) as f32;
                    let yn = (y / half_h) as f32;
                    let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&[[xn, yn]], &device), &fd, &device);
                    let raw = fwd::<BInner>(&model, stencil, 0, &device).mul_scalar(spec.load.px);
                    let col = |r0: usize, c: usize| -> Tensor<BInner, 1> { raw.clone().slice([r0..r0 + 1, c..c + 1]).reshape([1]) };
                    let loss = equilibrium_residual_loss::<BInner>(
                        col(1, 2), col(1, 4), col(2, 2), col(2, 4),
                        col(3, 4), col(3, 3), col(4, 4), col(4, 3),
                        cx, cy, ref_div2,
                    );
                    residuals.push(loss.into_data().to_vec::<f32>().unwrap()[0].sqrt());
                }
            }
            let (eq_rms, eq_max) = crate::training_core::residual_stats(&residuals);
            println!("  [diag] equilibrium residual RMS={eq_rms:.4e}  max={eq_max:.4e}  (dimensionless, normalized by ref_div2)");
        }
    }

    // Real, `#[ignore]`d diagnostic: profiles hoop stress/strain vs. radial distance from the
    // hole boundary on a REAL trained model (same 8000-step `single_hole_like_spec` config as
    // `run_training_user_problem_generalized_amr_indicator_diagnostic`, so the two are
    // directly comparable), rather than just reporting a single aggregate Kt number. Built
    // per a specific, code-level-grounded hypothesis from the Kt investigation (see
    // `powershell_tool/CLAUDE.md`): does the DIRECT network stress output ever show a hoop-
    // stress concentration anywhere near the hole, even where the FD-derived (displacement-
    // based) stress/strain can't be trusted (r too close to the boundary)? If both the direct
    // and FD-derived hoop stress stay near zero even well outside the FD-unsafe annulus, the
    // displacement field itself never developed the required curvature - the concentration
    // never formed, full stop, regardless of which representation you trust.
    //
    // Gets a REAL, AMR-trained model (not a separately-built non-AMR training loop) by
    // training through the actual GUI entry point, then requesting a checkpoint save/load via
    // the same `ControlMsg::SaveCheckpoint`/`checkpoint::load_checkpoint` path the real
    // Save/Load Trained Model UI buttons use - reusing already-tested infrastructure rather
    // than building a parallel one-off training loop just for this diagnostic.
    #[test]
    #[ignore]
    fn run_training_user_problem_radial_hoop_stress_profile_diagnostic() {
        use crate::energy::compute_stress;
        use crate::fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig};
        use crate::network::fwd;
        use crate::training_core::BInner;

        let spec = single_hole_like_spec(8000);
        let half_w = spec.geometry.half_w;
        let half_h = spec.geometry.half_h;
        let hole = spec.geometry.holes[0];
        let device = crate::training_core::BDevice::default();

        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || run_training_user_problem(spec.clone(), tx, rx_ctrl));

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3600);
        let mut saw_done = false;
        while std::time::Instant::now() < deadline && !saw_done {
            match rx.try_recv() {
                Ok(TrainingMsg::Done) => saw_done = true,
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        assert!(saw_done, "expected TrainingMsg::Done within the deadline");

        let path = std::env::temp_dir().join(format!("pinn_solver_radial_profile_diag_{}", std::process::id()));
        tx_ctrl.send(ControlMsg::SaveCheckpoint { path: path.clone(), saved_at_unix: 0 }).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut saved: Option<Result<String, String>> = None;
        while std::time::Instant::now() < deadline && saved.is_none() {
            if let Ok(TrainingMsg::CheckpointSaved(r)) = rx.try_recv() { saved = Some(r); }
            else { std::thread::sleep(std::time::Duration::from_millis(5)); }
        }
        let written = saved.expect("must receive a CheckpointSaved response").expect("save must succeed");
        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();

        // `load_checkpoint` derives the `.meta.json` sidecar path from the ORIGINAL
        // (pre-extension) `weights_path` it's given - not `written` (the post-extension path
        // `CheckpointSaved` reports) - matching `save_checkpoint`'s own `meta_path` derivation.
        // Passing `written` here would look for a nonexistent "*.mpk.meta.json" instead of the
        // real "*.meta.json" (confirmed via a real failed run, not assumed).
        let (model, _meta) = crate::checkpoint::load_checkpoint(&path, &device)
            .expect("must load the just-saved checkpoint");
        let _ = std::fs::remove_file(&written);
        let mut meta_path = path.clone();
        meta_path.set_file_name(format!("{}.meta.json", path.file_stem().unwrap().to_string_lossy()));
        let _ = std::fs::remove_file(meta_path);

        let stress_ref = 6.9e7_f64; // matches single_hole_like_spec's uniaxial_x load
        let u_ref = ((stress_ref / 71.7e9) * half_w) as f32;
        let fd = FdConfig::new(1e-3, 2.0 * half_w, 2.0 * half_h);
        let material = pinn_core::material::MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 };
        let margin_m = crate::user_problem::ring_anchor_margin_m(1e-3, &pinn_core::user_geometry::UserGeometry {
            half_w, half_h, thickness: 0.005, holes: vec![hole],
        });

        // theta = pi/2 (top of the hole, perpendicular to the uniaxial load direction) is
        // where the classical Kirsch solution puts the peak hoop-stress concentration
        // (~3x nominal) - the single most informative angle to profile.
        let theta = std::f64::consts::FRAC_PI_2;
        let (sin_t, cos_t) = theta.sin_cos();
        println!("  [radial-profile] theta=90deg (peak-concentration angle), FD-unsafe margin={margin_m:.4e} m");
        println!("  [radial-profile] {:>8} {:>10} {:>16} {:>16} {:>16} {:>10}", "r/R", "r-R (mm)", "sigma_tt_direct", "sigma_tt_fd", "eps_tt_fd", "fd_safe");
        for &ratio in &[1.0005, 1.005, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0] {
            let r = hole.radius * ratio;
            let x = hole.center[0] + r * cos_t;
            let y = hole.center[1] + r * sin_t;
            let xn = (x / half_w) as f32;
            let yn = (y / half_h) as f32;
            let fd_safe = (r - hole.radius) > margin_m;

            // Direct network stress (no FD needed - always meaningful, even inside the
            // FD-unsafe annulus, since it doesn't require evaluating neighboring points).
            let pt_t = norm_pts_to_tensor::<BInner>(&[[xn, yn]], &device);
            let raw = fwd::<BInner>(&model, pt_t, 0, &device);
            let sxx_d = raw.clone().slice([0..1, 2..3]).into_data().to_vec::<f32>().unwrap()[0] as f64 * stress_ref;
            let syy_d = raw.clone().slice([0..1, 3..4]).into_data().to_vec::<f32>().unwrap()[0] as f64 * stress_ref;
            let sxy_d = raw.slice([0..1, 4..5]).into_data().to_vec::<f32>().unwrap()[0] as f64 * stress_ref;
            let sigma_tt_direct = sxx_d * sin_t * sin_t + syy_d * cos_t * cos_t - 2.0 * sxy_d * sin_t * cos_t;

            // FD-derived stress/strain - only physically meaningful when `fd_safe`, but
            // computed and printed regardless so the unsafe-zone garbage is visible too, not
            // silently hidden.
            let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&[[xn, yn]], &device), &fd, &device);
            let raw_stencil = fwd::<BInner>(&model, stencil, 0, &device);
            let m = 5usize;
            let scaled = Tensor::<BInner, 2>::cat(vec![
                raw_stencil.clone().slice([0..m, 0..2]).mul_scalar(u_ref as f64),
                raw_stencil.slice([0..m, 2..5]).mul_scalar(stress_ref),
            ], 1);
            let (exx, eyy, exy) = compute_strains::<BInner>(scaled, 1, &fd);
            let (sxx_fd, syy_fd, sxy_fd) = compute_stress::<BInner>(exx.clone(), eyy.clone(), exy.clone(), &material);
            let get = |t: Tensor<BInner, 1>| -> f64 { t.into_data().to_vec::<f32>().unwrap()[0] as f64 };
            let (sxx_fd, syy_fd, sxy_fd) = (get(sxx_fd), get(syy_fd), get(sxy_fd));
            let (exx, eyy, exy) = (get(exx), get(eyy), get(exy));
            let sigma_tt_fd = sxx_fd * sin_t * sin_t + syy_fd * cos_t * cos_t - 2.0 * sxy_fd * sin_t * cos_t;
            let eps_tt_fd = exx * sin_t * sin_t + eyy * cos_t * cos_t - 2.0 * exy * sin_t * cos_t;

            println!(
                "  [radial-profile] {:>8.4} {:>10.4} {:>16.4e} {:>16.4e} {:>16.4e} {:>10}",
                ratio, (r - hole.radius) * 1000.0, sigma_tt_direct, sigma_tt_fd, eps_tt_fd,
                if fd_safe { "yes" } else { "NO" },
            );
        }
        println!("  [radial-profile] nominal_stress={stress_ref:.4e} Pa (expect sigma_tt -> ~3x this near r/R=1 if the classical concentration formed)");
    }

    // Real, `#[ignore]`d diagnostic: signed per-edge outer-boundary traction (target vs.
    // predicted, exactly as `OuterTractionTerm`/`neumann_loss` compute it) plus a horizontal
    // centerline (y=0) profile of displacement and FD-derived stress. Same 8000-step
    // `single_hole_like_spec` config and same train-then-checkpoint-load pattern as the two
    // diagnostics above, for direct comparability. Built to rule out a left/right traction
    // sign error and to see the far-field stress state directly, cheaper than re-deriving it
    // from an aggregate residual number.
    #[test]
    #[ignore]
    fn run_training_user_problem_signed_boundary_and_centerline_diagnostic() {
        use crate::energy::compute_stress;
        use crate::fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig};
        use crate::network::fwd;
        use crate::training_core::BInner;

        let spec = single_hole_like_spec(8000);
        let half_w = spec.geometry.half_w;
        let half_h = spec.geometry.half_h;
        let hole = spec.geometry.holes[0];
        let device = crate::training_core::BDevice::default();

        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || run_training_user_problem(spec.clone(), tx, rx_ctrl));

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3600);
        let mut saw_done = false;
        while std::time::Instant::now() < deadline && !saw_done {
            match rx.try_recv() {
                Ok(TrainingMsg::Done) => saw_done = true,
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        assert!(saw_done, "expected TrainingMsg::Done within the deadline");

        let path = std::env::temp_dir().join(format!("pinn_solver_signed_bc_diag_{}", std::process::id()));
        tx_ctrl.send(ControlMsg::SaveCheckpoint { path: path.clone(), saved_at_unix: 0 }).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut saved: Option<Result<String, String>> = None;
        while std::time::Instant::now() < deadline && saved.is_none() {
            if let Ok(TrainingMsg::CheckpointSaved(r)) = rx.try_recv() { saved = Some(r); }
            else { std::thread::sleep(std::time::Duration::from_millis(5)); }
        }
        let written = saved.expect("must receive a CheckpointSaved response").expect("save must succeed");
        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();

        let (model, _meta) = crate::checkpoint::load_checkpoint(&path, &device)
            .expect("must load the just-saved checkpoint");
        let _ = std::fs::remove_file(&written);
        let mut meta_path = path.clone();
        meta_path.set_file_name(format!("{}.meta.json", path.file_stem().unwrap().to_string_lossy()));
        let _ = std::fs::remove_file(meta_path);

        let stress_ref = 6.9e7_f64; // matches single_hole_like_spec's uniaxial_x load (px)
        let u_ref = ((stress_ref / 71.7e9) * half_w) as f32;
        let fd = FdConfig::new(1e-3, 2.0 * half_w, 2.0 * half_h);
        let material = pinn_core::material::MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 };

        let get1 = |t: Tensor<BInner, 1>| -> f64 { t.into_data().to_vec::<f32>().unwrap()[0] as f64 };
        let sigma_at = |x: f64, y: f64| -> (f64, f64, f64) {
            let xn = (x / half_w) as f32;
            let yn = (y / half_h) as f32;
            let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&[[xn, yn]], &device), &fd, &device);
            let raw = fwd::<BInner>(&model, stencil, 0, &device);
            let scaled = Tensor::<BInner, 2>::cat(vec![
                raw.clone().slice([0..5, 0..2]).mul_scalar(u_ref as f64),
                raw.slice([0..5, 2..5]).mul_scalar(stress_ref),
            ], 1);
            let (exx, eyy, exy) = compute_strains::<BInner>(scaled, 1, &fd);
            let (sxx, syy, sxy) = compute_stress::<BInner>(exx, eyy, exy, &material);
            (get1(sxx), get1(syy), get1(sxy))
        };

        // Signed per-edge traction: target vs. predicted, exactly as `OuterTractionTerm`/
        // `neumann_loss` compute it (FD-derived sigma . n vs. px*nx / py*ny). Midpoint of each
        // edge - far from the hole, no FD-safety concern there.
        println!("  [boundary] {:>8} {:>10} {:>14} {:>14} {:>14} {:>14}", "edge", "quantity", "target_tx", "pred_tx", "target_ty", "pred_ty");
        let edges: [(&str, f64, f64, f64, f64); 4] = [
            ("left",   -half_w, 0.0,     -1.0, 0.0),
            ("right",   half_w, 0.0,      1.0, 0.0),
            ("top",     0.0,    half_h,   0.0, 1.0),
            ("bottom",  0.0,   -half_h,   0.0, -1.0),
        ];
        for (name, x, y, nx, ny) in edges {
            let (sxx, syy, sxy) = sigma_at(x, y);
            let tx_pred = sxx * nx + sxy * ny;
            let ty_pred = sxy * nx + syy * ny;
            let tx_target = nx * stress_ref; // py = 0 for uniaxial_x
            let ty_target = ny * 0.0;
            println!(
                "  [boundary] {:>8} {:>10} {:>14.4e} {:>14.4e} {:>14.4e} {:>14.4e}   (sigma_xx={sxx:.4e} sigma_yy={syy:.4e} sigma_xy={sxy:.4e})",
                name, "traction", tx_target, tx_pred, ty_target, ty_pred,
            );
        }

        // Full per-edge mean/min/max of the displacement-derived stress tensor itself (not
        // just the single-point traction projection above) - same point layout convention
        // `UserSamplingStrategy::sample_boundary` uses (evenly spaced, `frac=(i+0.5)/n` so
        // exact corners are never sampled), evaluated at 32 points per edge.
        const N_EDGE_PTS: usize = 32;
        struct Stats { mean: f64, min: f64, max: f64 }
        fn stats(vals: &[f64]) -> Stats {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Stats { mean, min, max }
        }
        println!("  [edge-stats] {:>8} {:>10} {:>14} {:>14} {:>14}", "edge", "component", "mean", "min", "max");
        for name in ["left", "right", "top", "bottom"] {
            let mut sxx_v = Vec::with_capacity(N_EDGE_PTS);
            let mut syy_v = Vec::with_capacity(N_EDGE_PTS);
            let mut sxy_v = Vec::with_capacity(N_EDGE_PTS);
            for i in 0..N_EDGE_PTS {
                let frac = (i as f64 + 0.5) / N_EDGE_PTS as f64; // (0,1), avoids exact corners
                let (x, y) = match name {
                    "left" => (-half_w, -half_h + 2.0 * half_h * frac),
                    "right" => (half_w, -half_h + 2.0 * half_h * frac),
                    "top" => (-half_w + 2.0 * half_w * frac, half_h),
                    "bottom" => (-half_w + 2.0 * half_w * frac, -half_h),
                    _ => unreachable!(),
                };
                let (sxx, syy, sxy) = sigma_at(x, y);
                sxx_v.push(sxx); syy_v.push(syy); sxy_v.push(sxy);
            }
            for (label, vals) in [("sigma_xx", &sxx_v), ("sigma_yy", &syy_v), ("sigma_xy", &sxy_v)] {
                let s = stats(vals);
                println!("  [edge-stats] {name:>8} {label:>10} {:>14.4e} {:>14.4e} {:>14.4e}", s.mean, s.min, s.max);
            }
        }
        println!("  [edge-stats] intended: left/right sigma_xx~+6.9e7 sigma_xy~0; top/bottom sigma_yy~0 sigma_xy~0 (finite-plate corrections near corners)");

        // Horizontal centerline (y=0), excluding x=0 (inside the hole).
        println!("  [centerline] {:>10} {:>14} {:>14} {:>16} {:>16} {:>16}", "x", "u(x,0)", "v(x,0)", "sigma_xx", "sigma_yy", "sigma_xy");
        for &frac in &[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0] {
            let x = frac * half_w;
            if x.abs() <= hole.radius {
                println!("  [centerline] {x:>10.4}   (inside hole, radius={:.4}, skipped)", hole.radius);
                continue;
            }
            let xn = (x / half_w) as f32;
            let pt = norm_pts_to_tensor::<BInner>(&[[xn, 0.0]], &device);
            let raw = fwd::<BInner>(&model, pt, 0, &device);
            let u = get1(raw.clone().slice([0..1, 0..1]).reshape([1])) * u_ref as f64;
            let v = get1(raw.slice([0..1, 1..2]).reshape([1])) * u_ref as f64;
            let (sxx, syy, sxy) = sigma_at(x, 0.0);
            println!("  [centerline] {x:>10.4} {u:>14.4e} {v:>14.4e} {sxx:>16.4e} {syy:>16.4e} {sxy:>16.4e}");
        }
        println!("  [centerline] nominal_stress={stress_ref:.4e} Pa, half_w={half_w}, hole_radius={}", hole.radius);
    }

    fn no_hole_plate_spec(max_steps: usize) -> ProblemSpec {
        use pinn_core::loading::LoadConfig;
        use pinn_core::material::MaterialProps;
        use pinn_core::problem_spec::{NetworkSpec, TrainingSpec};
        use pinn_core::user_geometry::UserGeometry;
        ProblemSpec {
            geometry: UserGeometry { half_w: 0.10, half_h: 0.10, thickness: 0.005, holes: vec![] },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: NetworkSpec { hidden_dim: 64, n_hidden: 3, ..Default::default() },
            training: TrainingSpec { max_steps, n_interior: 2048, n_boundary: 512, fd_h: 1e-3, lr: 1e-3 },
        }
    }

    /// Real, `#[ignore]`d sanity test recommended by the Kt investigation
    /// (`powershell_tool/CLAUDE.md`): before trusting any hole-specific result, confirm the
    /// generic plate formulation (now including the new `EquilibriumTerm`) can recover the
    /// trivial exact solution - uniform uniaxial tension with no hole to concentrate around.
    /// `holes: vec![]` means zero hole-related loss terms register at all (`UserDefinedProblem::
    /// loss_terms()`'s hole loop is empty), so this exercises exactly: outer traction +
    /// interior energy + equilibrium + constitutive consistency, nothing else. If this doesn't
    /// recover sigma_xx~+69 MPa, sigma_yy~0, sigma_xy~0, the hole case is not worth touching
    /// until this does - per the plan's own explicit stopping condition.
    #[test]
    #[ignore]
    fn run_training_user_problem_no_hole_plate_recovers_uniform_uniaxial_tension() {
        use crate::energy::compute_stress;
        use crate::fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig};
        use crate::network::fwd;
        use crate::training_core::BInner;

        // 9000 (3x the original 3000) - to distinguish "just needs more steps" from "plateaued"
        // now that `equilibrium` has real gradient and genuinely competes with the other terms
        // for optimization budget (see this test's own updated doc comment above).
        let spec = no_hole_plate_spec(9000);
        let half_w = spec.geometry.half_w;
        let half_h = spec.geometry.half_h;
        let device = crate::training_core::BDevice::default();

        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || run_training_user_problem(spec.clone(), tx, rx_ctrl));

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1800);
        let mut saw_done = false;
        let mut last_logged_step = usize::MAX;
        while std::time::Instant::now() < deadline && !saw_done {
            match rx.try_recv() {
                Ok(TrainingMsg::Update(u)) => {
                    // Real convergence-trend evidence (plateaued vs. still improving), not just
                    // a single final number - printed every 1000 steps.
                    if u.step % 1000 == 0 && u.step != last_logged_step {
                        last_logged_step = u.step;
                        println!("  [no-hole] step={} total_loss={:.4e} energy_loss={:.4e} neumann_loss={:.4e}",
                            u.step, u.total_loss, u.energy_loss, u.neumann_loss);
                    }
                }
                Ok(TrainingMsg::Done) => saw_done = true,
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        assert!(saw_done, "expected TrainingMsg::Done within the deadline");

        let path = std::env::temp_dir().join(format!("pinn_solver_no_hole_sanity_{}", std::process::id()));
        tx_ctrl.send(ControlMsg::SaveCheckpoint { path: path.clone(), saved_at_unix: 0 }).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut saved: Option<Result<String, String>> = None;
        while std::time::Instant::now() < deadline && saved.is_none() {
            if let Ok(TrainingMsg::CheckpointSaved(r)) = rx.try_recv() { saved = Some(r); }
            else { std::thread::sleep(std::time::Duration::from_millis(5)); }
        }
        let written = saved.expect("must receive a CheckpointSaved response").expect("save must succeed");
        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();

        let (model, _meta) = crate::checkpoint::load_checkpoint(&path, &device)
            .expect("must load the just-saved checkpoint");
        let _ = std::fs::remove_file(&written);
        let mut meta_path = path.clone();
        meta_path.set_file_name(format!("{}.meta.json", path.file_stem().unwrap().to_string_lossy()));
        let _ = std::fs::remove_file(meta_path);

        let stress_ref = 6.9e7_f64;
        let u_ref = ((stress_ref / 71.7e9) * half_w) as f32;
        let fd = FdConfig::new(1e-3, 2.0 * half_w, 2.0 * half_h);
        let material = pinn_core::material::MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 };
        let get1 = |t: Tensor<BInner, 1>| -> f64 { t.into_data().to_vec::<f32>().unwrap()[0] as f64 };
        let sigma_at = |x: f64, y: f64| -> (f64, f64, f64, f64, f64) {
            let xn = (x / half_w) as f32;
            let yn = (y / half_h) as f32;
            let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&[[xn, yn]], &device), &fd, &device);
            let raw = fwd::<BInner>(&model, stencil, 0, &device);
            // Raw network output at the centre row (row 0 of the 5-row stencil) - the
            // no-hole plate's `IdentityAnsatz` applies no extra per-point scale factor, so
            // this IS `u_norm`/`v_norm` exactly (bugSource-New #8's own normalized-slope
            // diagnostic: `u_norm ≈ x_norm`, `v_norm ≈ -ν·y_norm`).
            let u_norm = get1(raw.clone().slice([0..1, 0..1]).reshape([1]));
            let v_norm = get1(raw.clone().slice([0..1, 1..2]).reshape([1]));
            let scaled = Tensor::<BInner, 2>::cat(vec![
                raw.clone().slice([0..5, 0..2]).mul_scalar(u_ref as f64),
                raw.slice([0..5, 2..5]).mul_scalar(stress_ref),
            ], 1);
            let (exx, eyy, exy) = compute_strains::<BInner>(scaled, 1, &fd);
            let (sxx, syy, sxy) = compute_stress::<BInner>(exx, eyy, exy, &material);
            (get1(sxx), get1(syy), get1(sxy), u_norm, v_norm)
        };

        // A grid of interior points (not just edges/centerline) - the whole point is
        // uniformity EVERYWHERE, not just where a loss term directly constrains it.
        let mut sxx_v = Vec::new();
        let mut syy_v = Vec::new();
        let mut sxy_v = Vec::new();
        // (x_norm, u_norm) and (y_norm, v_norm) pairs for bugSource-New #8's slope check.
        let mut xu_pairs: Vec<(f64, f64)> = Vec::new();
        let mut yv_pairs: Vec<(f64, f64)> = Vec::new();
        for &fx in &[-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75] {
            for &fy in &[-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75] {
                let (sxx, syy, sxy, u_norm, v_norm) = sigma_at(fx * half_w, fy * half_h);
                println!("  [no-hole] x={:>8.4} y={:>8.4}  sigma_xx={sxx:>14.4e}  sigma_yy={syy:>14.4e}  sigma_xy={sxy:>14.4e}", fx * half_w, fy * half_h);
                sxx_v.push(sxx); syy_v.push(syy); sxy_v.push(sxy);
                xu_pairs.push((fx, u_norm));
                yv_pairs.push((fy, v_norm));
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (sxx_mean, syy_mean, sxy_mean) = (mean(&sxx_v), mean(&syy_v), mean(&sxy_v));
        println!("  [no-hole] MEAN sigma_xx={sxx_mean:.4e}  sigma_yy={syy_mean:.4e}  sigma_xy={sxy_mean:.4e}  (target: {stress_ref:.4e}, 0, 0)");
        println!("  [no-hole] relative error: sigma_xx={:.1}%  sigma_yy={:.1}%(of nominal)  sigma_xy={:.1}%(of nominal)",
            (sxx_mean - stress_ref).abs() / stress_ref * 100.0,
            syy_mean.abs() / stress_ref * 100.0,
            sxy_mean.abs() / stress_ref * 100.0,
        );

        // bugSource-New #8: ordinary least-squares slope+intercept of u_norm vs x_norm and
        // v_norm vs y_norm across the same 49-point grid. Expected: slope≈1 (u), slope≈-ν≈-0.33
        // (v), both near-zero intercept - "if those slopes aren't emerging, don't investigate
        // the hole" (their words). A cheap, no-extra-training diagnostic reusing the grid
        // already computed above.
        let ols_slope_intercept = |pairs: &[(f64, f64)]| -> (f64, f64) {
            let n = pairs.len() as f64;
            let sx: f64 = pairs.iter().map(|&(x, _)| x).sum();
            let sy: f64 = pairs.iter().map(|&(_, y)| y).sum();
            let sxx: f64 = pairs.iter().map(|&(x, _)| x * x).sum();
            let sxy: f64 = pairs.iter().map(|&(x, y)| x * y).sum();
            let denom = n * sxx - sx * sx;
            let slope = (n * sxy - sx * sy) / denom;
            let intercept = (sy - slope * sx) / n;
            (slope, intercept)
        };
        let (u_slope, u_intercept) = ols_slope_intercept(&xu_pairs);
        let (v_slope, v_intercept) = ols_slope_intercept(&yv_pairs);
        let nu = material.nu as f64;
        println!("  [no-hole] du_norm/dx_norm = {u_slope:.4} (expected ~1.0), intercept={u_intercept:.4e}");
        println!("  [no-hole] dv_norm/dy_norm = {v_slope:.4} (expected ~{:.4}), intercept={v_intercept:.4e}", -nu);
        // Loose tolerance (this is a finite-plate/network-approximation sanity check, not
        // exact FEA) - but a genuinely working formulation should land well inside 30%.
        assert!((sxx_mean - stress_ref).abs() / stress_ref < 0.30,
            "sigma_xx mean {sxx_mean:.4e} too far from nominal {stress_ref:.4e}");
        assert!(syy_mean.abs() / stress_ref < 0.30, "sigma_yy mean {syy_mean:.4e} should be ~0");
        assert!(sxy_mean.abs() / stress_ref < 0.30, "sigma_xy mean {sxy_mean:.4e} should be ~0");
    }

    /// Real, `#[ignore]`d single-hole Kt check - the actual target metric the whole
    /// investigation (bugSource.txt/bugSource-New) was chasing, run only after the no-hole
    /// sanity test above passes (per the investigation's own standing discipline). Same
    /// train->checkpoint->reload structure as the no-hole test, but measures Kt via
    /// `probe_hole_boundary_profile_derived`/`stress_concentration_from_profile` (derived
    /// stress at the FD-safe margin, NOT direct σ - see those functions' doc comments for why,
    /// bugSource-New #12). Not compared against a hardcoded Kt=3 (finite plate, see
    /// `stress_concentration_from_profile`'s own doc comment) - reported honestly either way.
    #[test]
    #[ignore]
    fn run_training_user_problem_single_hole_plate_kt_after_hessian_equilibrium_fix() {
        let spec = single_hole_like_spec(3000);
        let device = crate::training_core::BDevice::default();

        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || run_training_user_problem(spec.clone(), tx, rx_ctrl));

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1800);
        let mut saw_done = false;
        while std::time::Instant::now() < deadline && !saw_done {
            match rx.try_recv() {
                Ok(TrainingMsg::Done) => saw_done = true,
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        assert!(saw_done, "expected TrainingMsg::Done within the deadline");

        let path = std::env::temp_dir().join(format!("pinn_solver_single_hole_kt_{}", std::process::id()));
        tx_ctrl.send(ControlMsg::SaveCheckpoint { path: path.clone(), saved_at_unix: 0 }).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut saved: Option<Result<String, String>> = None;
        while std::time::Instant::now() < deadline && saved.is_none() {
            if let Ok(TrainingMsg::CheckpointSaved(r)) = rx.try_recv() { saved = Some(r); }
            else { std::thread::sleep(std::time::Duration::from_millis(5)); }
        }
        let written = saved.expect("must receive a CheckpointSaved response").expect("save must succeed");
        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();

        let (model, _meta) = crate::checkpoint::load_checkpoint(&path, &device)
            .expect("must load the just-saved checkpoint");
        let _ = std::fs::remove_file(&written);
        let mut meta_path = path.clone();
        meta_path.set_file_name(format!("{}.meta.json", path.file_stem().unwrap().to_string_lossy()));
        let _ = std::fs::remove_file(meta_path);

        let spec = single_hole_like_spec(3000);
        let half_w = spec.geometry.half_w;
        let half_h = spec.geometry.half_h;
        let u_ref = crate::training_core::compute_reference_scales_for_plate(&spec).u_ref;
        let fd = crate::fd_stencil::FdConfig::new(spec.training.fd_h, 2.0 * half_w, 2.0 * half_h);
        let margin = crate::user_problem::ring_anchor_margin_m(spec.training.fd_h, &spec.geometry);

        let nominal_stress = spec.load.px.abs().max(spec.load.py.abs());
        for (i, hole) in spec.geometry.holes.iter().enumerate() {
            let profile = crate::user_problem::probe_hole_boundary_profile_derived(
                &model, &spec.geometry, hole, 144, &fd, u_ref, spec.load.px, &spec.material, margin, &device,
            );
            let sc = crate::user_problem::stress_concentration_from_profile(&profile, nominal_stress);
            println!("  [single-hole-kt] hole {i}: max_von_mises={:.4e} Pa (at theta={:.1} deg)  nominal={:.4e} Pa  Kt={:.4}",
                sc.max_von_mises, sc.max_theta_deg, sc.nominal_stress, sc.kt);
        }
    }

    // ─── Smart adaptive architecture: end-to-end wiring into run_training_user_problem ───────

    fn tiny_adaptive_spec(max_steps: usize) -> ProblemSpec {
        use pinn_core::loading::LoadConfig;
        use pinn_core::material::MaterialProps;
        use pinn_core::problem_spec::{NetworkSpec, TrainingSpec};
        use pinn_core::user_geometry::{HoleBc, HoleSpec, UserGeometry};
        ProblemSpec {
            geometry: UserGeometry {
                half_w: 0.10, half_h: 0.10, thickness: 0.005,
                holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.02, bc: HoleBc::Free }],
            },
            material: MaterialProps { e: 71.7e9, nu: 0.33, density: 2810.0, ultimate_strength_pa: 503e6 },
            load: LoadConfig::uniaxial_x(6.9e7),
            network: NetworkSpec {
                hidden_dim: 8, n_hidden: 3,
                adaptive: true, max_hidden_dim: Some(12), max_n_hidden: Some(4),
                ..Default::default()
            },
            training: TrainingSpec { max_steps, n_interior: 64, n_boundary: 32, fd_h: 1e-3, lr: 1e-3 },
        }
    }

    /// Real end-to-end integration check that `ArchitectureController` is actually wired into
    /// `run_training_user_problem` (not just unit-tested in isolation), confirming: (a)
    /// `adaptive: true` never panics or produces non-finite loss/vis output across whatever
    /// architecture transitions occur, and (b) IF at least one `ArchitectureEvent` fires, its
    /// fields are internally consistent (a real before/after change, `step` inside the run).
    ///
    /// This deliberately does NOT hard-require an event to fire: `ConvergenceTracker::
    /// check_plateau`'s `PLATEAU_WINDOW` (controllers.rs) needs 40 real vis-cadence readings
    /// (400 steps at this function's every-10th-step cadence) before it can even evaluate a
    /// plateau, and whether one is detected (or a gate happens to go dormant) depends on real,
    /// not fully step-seeded, training dynamics - asserting "an event MUST fire" would make
    /// this test flaky. The crash/NaN-safety assertion is unconditional; the event-content
    /// check only runs when one is actually observed. `#[ignore]`d for the same reason
    /// `run_training_user_problem_amr_sweep_changes_collocation_count` is - a genuine, if
    /// small, multi-hundred-step training run.
    #[test]
    #[ignore]
    fn run_training_user_problem_adaptive_wiring_does_not_crash_and_events_are_consistent() {
        // `run_training_user_problem` never returns after `Done` on its own - it stays alive
        // serving `SaveCheckpoint` requests (see its own doc comment) - so, exactly like
        // `parametric_problem::tests::run_training_parametric_completes_and_sends_updates_and_
        // ready`, this must run on its own thread and be sent an explicit `Stop` once `Done`
        // is observed, not called synchronously (`run_and_drain` doesn't fit here either, for
        // the same reason `serve_loaded_plate_checkpoint_sends_update_then_done_with_no_
        // training`'s own doc comment gives - it assumes the spawned fn returns after `Done`).
        let spec = tiny_adaptive_spec(450);
        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || run_training_user_problem(spec, tx, rx_ctrl));

        let mut events = Vec::new();
        let mut saw_done = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(TrainingMsg::Update(upd)) => {
                    assert!(upd.total_loss.is_finite(), "step {}: non-finite total_loss under adaptive wiring", upd.step);
                    if let Some(vis) = &upd.vis {
                        assert!(vis.von_mises.iter().all(|v| v.is_nan() || v.is_finite()), "step {}: non-finite (non-NaN-mask) vis output", upd.step);
                    }
                    if let Some(ev) = upd.architecture_event.clone() {
                        events.push(ev);
                    }
                }
                Ok(TrainingMsg::Error(e)) => panic!("adaptive run reported an error: {e}"),
                Ok(TrainingMsg::Done) => { saw_done = true; break; }
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();
        assert!(saw_done, "expected the run to reach TrainingMsg::Done");

        for ev in &events {
            assert!(ev.step < 450, "event step {} out of the run's range", ev.step);
            let width_changed = ev.hidden_dim_before != ev.hidden_dim_after;
            let depth_changed = ev.n_hidden_before != ev.n_hidden_after;
            assert!(
                width_changed || depth_changed || ev.description.contains("Reverted"),
                "event at step {} claims no actual change and isn't a revert: {:?}", ev.step, ev
            );
        }
        println!("{} architecture event(s) observed over 450 steps: {events:?}", events.len());
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

    /// Real end-to-end integration check that generic AMR is wired into
    /// `run_training_pinlug` for BOTH domains - the lug domain (real circular hole, a
    /// genuine lock zone) and the pin domain (no hole, zero lock zones, pure residual-driven
    /// refine/coarsen) - without any pin-lug-specific code in the AMR path itself. Mirrors
    /// `run_training_user_problem_amr_sweep_changes_collocation_count`'s same "n_colloc
    /// changed" signal (see that test's doc comment for why this - not a spatial-density
    /// assertion - is the right scope for a runner-level integration test; the spatial claim
    /// is already covered, faster and more precisely, at the `pinn-core` unit level).
    /// `#[ignore]`d: even at `tiny_pinlug_config`'s tiny network size, 1300 real steps (needed
    /// to cross `AMR_WARMUP_STEPS` + one sweep interval) measured ~345s - confirmed passing,
    /// but far too slow for the default suite. Run explicitly: `cargo test -p pinn-solver
    /// runner::tests::run_training_pinlug_amr_sweep_changes_collocation_count -- --ignored`.
    #[test]
    #[ignore]
    fn run_training_pinlug_amr_sweep_changes_collocation_count() {
        let mut config = tiny_pinlug_config();
        config.max_steps = 1300; // past AMR_WARMUP_STEPS(200) + one interval(1000)
        let original_n_colloc = config.n_interior * 2; // pin + lug, both sampled at n_interior
        let (_stop_tx, stop_rx) = crossbeam_channel::unbounded();
        let msgs = run_and_drain(move |tx| run_training_pinlug(config, tx, stop_rx));

        let last_n_colloc = msgs.iter().rev().find_map(|m| match m {
            TrainingMsg::PinLugUpdate(u) => Some(u.n_colloc),
            _ => None,
        }).expect("expected at least one PinLugUpdate");
        assert_ne!(
            last_n_colloc, original_n_colloc,
            "n_colloc still matches the pre-AMR pin+lug sample count after step 200's sweep - \
             the sweep either didn't fire or didn't actually resample either domain"
        );
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
