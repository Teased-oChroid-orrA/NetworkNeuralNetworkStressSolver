use burn::{
    backend::{Autodiff, Wgpu},
    module::AutodiffModule,
};
use burn::backend::wgpu::WgpuDevice;
use crossbeam_channel::{Receiver, Sender};
use ndarray::Array2;
use pinn_core::{
    amr::AdaptiveGrid,
    messages::{ControlMsg, SolverConfig, TrainingMsg, TrainingUpdate, VisFields},
    sampling::{sample_boundary, sample_interior, sample_eq_ring},
};

use crate::{
    bc::apply_dirichlet_ansatz,
    controllers::ConvergenceTracker,
    decision_maker::{OptimizerTier, PinnDecisionMaker},
    engine::EngineParams,
    energy::dem_energy_per_point,
    fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig},
    network::{fwd, ElasticityNet, ElasticityNetConfig},
    optim::{make_bias_optim, BiasOptim, WeightOptim},
    saw_brdr::SawBrdr,
    lr_schedule::LrSchedule,
    training_core::{
        compute_gradient_conflict, compute_reference_scales, extract_boundary_indices,
        make_lbfgs, normalize_point, probe_kt_shared, step_lbfgs, step_physics,
        LbfgsCtxScalars, LbfgsLams, StepCtx, StepOutput,
    },
};

type B = Autodiff<Wgpu>;
type BInner = Wgpu;

// ─── Training state ────────────────────────────────────────────────────────────

/// All mutable state threaded through the training loop, bundled so warm-start doesn't
/// need a 20+ parameter function signature — every field here is either replaced wholesale
/// or recomputed together whenever the problem (load/geometry/material) changes.
struct TrainingState {
    model: ElasticityNet<B>,
    optim_w: WeightOptim,
    optim_b: BiasOptim,

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

    bnd_norm: Vec<[f32; 2]>,
    bnd_nx: Vec<f32>,
    bnd_ny: Vec<f32>,
    bnd_tx: Vec<f32>,
    bnd_ty: Vec<f32>,
    trac_idx: Vec<usize>,
    hole_idx: Vec<usize>,
    right_idx: Vec<usize>,

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
    frozen_lbfgs_lams:   Option<LbfgsLams>,
}

impl TrainingState {
    fn new(config: &SolverConfig, engine: &EngineParams, net_cfg: &ElasticityNetConfig, device: &WgpuDevice) -> Self {
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

        let eq_ring_norm: Vec<[f32; 2]> = sample_eq_ring(&config.geometry, engine.n_eq_ring)
            .iter().map(|&[x, y]| normalize_point(x, y, config)).collect();

        let (vis_pts_norm, vis_mask) = build_vis_grid(config);

        let dm_cfg   = config.decision_maker.clone();
        let dm       = PinnDecisionMaker::new(dm_cfg, false);
        let optim_w  = WeightOptim::from_tier(config.use_soap_muon, &dm.current_tier);
        Self {
            model: net_cfg.init(device),
            optim_w,
            optim_b: make_bias_optim(),
            u_ref, ref_energy, ref_stress2,
            saw: SawBrdr::with_base(engine.init_weights(), 0.95),
            lr_sched: LrSchedule::new(engine.peak_lr, 200, 1000),
            phase2_started: false,
            tracker: ConvergenceTracker::new(),
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            int_pts_phys: sample_interior(&config.geometry, engine.phase1_n_interior),
            amr: None,
            bnd_norm, bnd_nx, bnd_ny, bnd_tx, bnd_ty,
            trac_idx, hole_idx, right_idx,
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
        }
    }

    /// Reconstruct both optimizers from scratch and reset the decision maker.
    /// Used on warm-start, phase transition, and convergence-cascade restarts.
    fn reset_optimizers(&mut self, use_soap_muon: bool) {
        self.optim_w = WeightOptim::from_tier(use_soap_muon, &self.decision_maker.current_tier);
        self.optim_b = make_bias_optim();
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
        device: &WgpuDevice,
        geometry_changed: bool,
    ) {
        let new_engine = EngineParams::analyze(&new_cfg);

        if geometry_changed {
            self.model = net_cfg.init(device);
            self.saw.set_base_weights(new_engine.init_weights());
        }

        self.int_pts_phys = sample_interior(&new_cfg.geometry, new_engine.phase1_n_interior);

        let bnd_pts = sample_boundary(&new_cfg.geometry, &new_cfg.load, new_cfg.n_boundary);
        self.bnd_norm = bnd_pts.iter().map(|b| normalize_point(b.x, b.y, &new_cfg)).collect();
        self.bnd_nx   = bnd_pts.iter().map(|b| b.nx as f32).collect();
        self.bnd_ny   = bnd_pts.iter().map(|b| b.ny as f32).collect();
        self.bnd_tx   = bnd_pts.iter().map(|b| b.tx as f32).collect();
        self.bnd_ty   = bnd_pts.iter().map(|b| b.ty as f32).collect();
        (self.trac_idx, self.hole_idx, self.right_idx) = extract_boundary_indices(&bnd_pts, &self.bnd_nx);

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
        self.decision_maker = PinnDecisionMaker::new(new_cfg.decision_maker.clone(), false);
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

    let device = WgpuDevice::default();
    let net_cfg = ElasticityNetConfig {
        input_dim:  engine.net_input_dim(),
        hidden_dim: config.hidden_dim,
        n_hidden:   config.n_hidden,
        output_dim: engine.output_dim(),
    };
    let use_soap_muon = config.use_soap_muon;
    let dm_config = config.decision_maker.clone();

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
            state.amr = Some(grid);
            state.lr_sched.reset_for_phase2();
            state.decision_maker = PinnDecisionMaker::new(dm_config.clone(), true);
            state.clear_lbfgs();
            state.reset_optimizers(use_soap_muon);
        }

        // === AMR sweep ===
        if step > state.current_engine.phase1_steps
            && (step - state.current_engine.phase1_steps) % state.current_engine.amr.interval_steps == 0
        {
            if let Some(ref mut grid) = state.amr {
                let int_norm_amr: Vec<[f32; 2]> = state.int_pts_phys.iter()
                    .map(|&[x, y]| normalize_point(x, y, &state.current_config)).collect();
                let n_amr = int_norm_amr.len();
                let model_val: ElasticityNet<BInner> = state.model.valid();
                let amr_residuals: Vec<f32> = {
                    let pts_t = norm_pts_to_tensor::<BInner>(&int_norm_amr, &device);
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
                println!("  [AMR@{step}] cells={} depth={} mean_res={:.3e} pts={}",
                    amr_stats.active_count, amr_stats.max_depth,
                    amr_stats.mean_residual, state.int_pts_phys.len());
                if state.decision_maker.current_tier == OptimizerTier::Converge {
                    println!("  [DM@{step}] AMR → demote Converge→Align");
                    state.decision_maker = PinnDecisionMaker::new(dm_config.clone(), true);
                    state.clear_lbfgs();
                    state.reset_optimizers(use_soap_muon);
                }
            }
        }

        // === Training step ===
        let int_norm: Vec<[f32; 2]> = state.int_pts_phys.iter()
            .map(|&[x, y]| normalize_point(x, y, &state.current_config)).collect();

        let ctx = StepCtx {
            config:            &state.current_config,
            engine:            &state.current_engine,
            fd:                &state.current_fd,
            k:                 state.current_k,
            u_ref:             state.u_ref,
            ref_energy:        state.ref_energy,
            ref_stress2:       state.ref_stress2,
            cx:                state.current_cx,
            cy:                state.current_cy,
            ref_div2:          state.current_ref_div2,
            int_norm:          &int_norm,
            bnd_norm:          &state.bnd_norm,
            bnd_nx:            &state.bnd_nx,
            bnd_ny:            &state.bnd_ny,
            bnd_tx:            &state.bnd_tx,
            bnd_ty:            &state.bnd_ty,
            trac_idx:          &state.trac_idx,
            hole_idx:          &state.hole_idx,
            right_idx:         &state.right_idx,
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
            let (new_m, loss_f64) = step_lbfgs(state.model, lbfgs, lr, fctx, lams, &device);
            let synthetic = StepOutput {
                e_scalar: 0.0, n_scalar: 0.0, h_scalar: 0.0, d_scalar: 0.0,
                eq_scalar: 0.0, w_scalar: 0.0, kirsch_scalar: 0.0, const_scalar: 0.0,
                total_scalar: loss_f64 as f32, lr,
                lam_e: 0.0, lam_n: 0.0, lam_h: 0.0, lam_d: 0.0, lam_eq: 0.0, lam_kirsch: 0.0,
                proxy_ratio: 0.0,
                optimizer_tier: OptimizerTier::Converge.as_u8(),
                cosine_sim: None,
            };
            (new_m, synthetic)
        } else {
            step_physics(
                state.model, &mut state.optim_w, &mut state.optim_b,
                &ctx, &mut state.saw, &mut state.lr_sched, &device,
                state.decision_maker.current_tier.as_u8(),
            )
        };
        state.model = new_model;

        // Decision maker gate.
        if state.decision_maker.advance() {
            let conflict = if dm_config.use_exact_cosine
                && state.decision_maker.current_tier != OptimizerTier::Converge
            {
                Some(compute_gradient_conflict(&state.model, &ctx, step, &device))
            } else {
                None
            };
            out.cosine_sim = conflict.map(|c| c.cosine_sim);
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
                        state.frozen_lbfgs_ctx  = Some(LbfgsCtxScalars::from_ctx(&ctx));
                        state.frozen_lbfgs_lams = Some(LbfgsLams {
                            lam_e:      out.lam_e,
                            lam_n:      out.lam_n,
                            lam_h:      out.lam_h,
                            lam_d:      out.lam_d,
                            lam_eq:     out.lam_eq,
                            lam_kirsch: out.lam_kirsch,
                            lam_const:  state.current_engine.lam_const as f64,
                        });
                        state.lbfgs_opt = None;
                    }
                    _ => state.clear_lbfgs(),
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
                        state.decision_maker = PinnDecisionMaker::new(dm_config.clone(), true);
                        state.clear_lbfgs();
                        state.reset_optimizers(use_soap_muon);
                        println!("\n  [CRASH RECOVERY #{}] K_t={kt_val:.3} collapsed → restart, lam_caps→{new_cap:.0}",
                            state.tracker.total_restarts());
                    } else if let Some(new_cap) = state.tracker.check_plateau() {
                        state.dynamic_lam_h_cap = new_cap;
                        state.dynamic_lam_d_cap = new_cap;
                        state.saw.reset();
                        state.lr_sched.reset_for_phase2();
                        state.decision_maker = PinnDecisionMaker::new(dm_config.clone(), true);
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

// ─── Visualisation grid ───────────────────────────────────────────────────────

/// Build the normalized [-1,1]² visualization grid and its in-domain mask for `config`.
/// Used at initial setup and again on warm-start when geometry changes.
fn build_vis_grid(config: &SolverConfig) -> (Vec<[f32; 2]>, Vec<bool>) {
    let [nx_vis, ny_vis] = config.vis_grid;
    let mut pts = Vec::with_capacity(nx_vis * ny_vis);
    for iy in 0..ny_vis {
        for ix in 0..nx_vis {
            let xn = -1.0 + 2.0 * ix as f64 / (nx_vis - 1) as f64;
            let yn = -1.0 + 2.0 * iy as f64 / (ny_vis - 1) as f64;
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
    device:    &WgpuDevice,
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

    let u_vals: Vec<f32> = out.clone().slice([0..n_act, 0..1]).reshape([n_act])
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_act]);
    let v_vals: Vec<f32> = out.clone().slice([0..n_act, 1..2]).reshape([n_act])
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_act]);

    let (eps_xx, eps_yy, eps_xy) = compute_strains::<BInner>(out, n_act, fd);
    let e  = cfg.material.e  as f32;
    let nu = cfg.material.nu as f32;
    let f  = e / (1.0 - nu * nu);

    let exx_v: Vec<f32> = eps_xx.into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_act]);
    let eyy_v: Vec<f32> = eps_yy.into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_act]);
    let exy_v: Vec<f32> = eps_xy.into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_act]);

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
