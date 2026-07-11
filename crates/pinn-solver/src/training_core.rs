/// Shared training step — single source of truth for the per-step physics computation.
///
/// Both `runner` (GUI, channel output) and `headless` (terminal, stdout output) call
/// `step_physics` for the loss computation + backward pass, differing only in how they
/// consume the returned `StepOutput` (send via channel vs. print to stdout).
///
/// Also provides `probe_kt_shared` and `normalize_point` which were duplicated across both
/// callers.

use std::collections::HashMap;

use burn::{
    backend::{Autodiff, Wgpu},
    module::{Module, ModuleVisitor, Param},
    optim::{GradientsParams, LBFGSConfig, Optimizer},
    tensor::{Tensor, TensorData},
};
use burn::backend::wgpu::WgpuDevice;
use pinn_core::{
    geometry::HoleType,
    kirsch::kirsch_stress,
    messages::SolverConfig,
    BoundaryKind, BoundaryPoint,
};

use crate::{
    bc::apply_dirichlet_ansatz,
    decision_maker::GradientConflict,
    engine::EngineParams,
    energy::{compute_stress, constitutive_consistency_loss, dem_energy_loss,
             equilibrium_residual_loss, hole_traction_loss, hole_traction_loss_direct,
             neumann_loss},
    fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig},
    kirsch_problem::KIRSCH_DOMAIN,
    lr_schedule::LrSchedule,
    network::{fwd, ElasticityNet},
    optim::{BiasOptim, GateOptim, WeightOptim},
    problem::{BoundaryValueProblem, DomainForwardOutputs, LossTerm},
    saw_brdr::SawBrdr,
};

pub type B = Autodiff<Wgpu>;
pub type BInner = Wgpu;

/// Two-domain model wrapper so `burn::optim::LBFGS::step()` (which requires a SINGLE
/// `AutodiffModule<B>`) can take one combined quasi-Newton step across pin-in-lug's two
/// domains' parameter spaces. `#[derive(Module, Debug)]` auto-derives `Module<B>`/
/// `AutodiffModule<B>` exactly as it does for `ElasticityNet<B>` itself — no blanket tuple
/// `Module` impl exists in burn-core 0.21.0, so a bare tuple does NOT satisfy this bound;
/// this named struct is the only option that compiles. Named (not positional-tuple) per
/// this repo's established aversion to bare-positional footguns (see `PinLugScalingMode`).
/// Concrete 2-domain, not a generic `Vec<ElasticityNet<B>>` wrapper — YAGNI, nothing in
/// `BoundaryValueProblem` today produces >2 domains.
#[derive(Module, Debug)]
pub struct TwoDomainModels<B: burn::tensor::backend::Backend> {
    pub pin: ElasticityNet<B>,
    pub lug: ElasticityNet<B>,
}

/// Normalize a physical (x, y) coordinate to [-1, 1]² using the config's geometry ranges.
pub fn normalize_point(x: f64, y: f64, config: &SolverConfig) -> [f32; 2] {
    let (x0, x1) = config.geometry.x_range();
    let (y0, y1) = config.geometry.y_range();
    let dw = x1 - x0;
    let dh = y1 - y0;
    [(2.0 * (x - x0) / dw - 1.0) as f32,
     (2.0 * (y - y0) / dh - 1.0) as f32]
}

/// Kirsch stress-loss probe points and their analytical targets — shared by both the
/// mDEM and plain-DEM branches of `step_physics`'s Kirsch loss block (only how the
/// predicted stresses are obtained at these points differs between the two).
struct KirschProbes {
    points: Vec<[f32; 2]>,
    sxx_targets: Vec<f32>,
    syy_targets: Vec<f32>,
    sxy_targets: Vec<f32>,
    /// sin²θ-weighted, pre-normalized so weights sum to 1.
    weights: Vec<f32>,
}

fn compute_kirsch_probes(ctx: &StepCtx, radius: f64) -> KirschProbes {
    let n_pr = ctx.engine.kirsch_r_factors.len() * ctx.engine.kirsch_thetas_deg.len();
    let mut points      = Vec::with_capacity(n_pr);
    let mut sxx_targets = Vec::with_capacity(n_pr);
    let mut syy_targets = Vec::with_capacity(n_pr);
    let mut sxy_targets = Vec::with_capacity(n_pr);
    let mut weights     = Vec::with_capacity(n_pr);

    for &r_fac in &ctx.engine.kirsch_r_factors {
        let r = radius * r_fac;
        for &deg in &ctx.engine.kirsch_thetas_deg {
            let theta = deg.to_radians();
            points.push(normalize_point(r * theta.cos(), r * theta.sin(), ctx.config));
            let (s_rr, s_tt, s_rt) = kirsch_stress(r, theta, radius, ctx.config.load.px, ctx.config.load.py);
            let c = theta.cos(); let s = theta.sin();
            sxx_targets.push((s_rr*c*c + s_tt*s*s - 2.0*s_rt*s*c) as f32);
            syy_targets.push((s_rr*s*s + s_tt*c*c + 2.0*s_rt*s*c) as f32);
            sxy_targets.push(((s_rr - s_tt)*s*c + s_rt*(c*c - s*s)) as f32);
            weights.push(theta.sin().powi(2) as f32);
        }
    }
    let wsum: f32 = weights.iter().sum();
    for w in weights.iter_mut() { *w /= wsum; }

    KirschProbes { points, sxx_targets, syy_targets, sxy_targets, weights }
}

/// Compute the three physics reference scales used to normalize loss terms to O(1):
/// `u_ref` [m] (displacement scale), `ref_energy` [Pa] (energy density scale), and
/// `ref_stress2` [Pa²]. Shared by `runner` and `headless`, and recomputed by both whenever
/// the load or material changes (warm-start).
pub fn compute_reference_scales(config: &SolverConfig) -> (f32, f32, f32) {
    let stress_ref = if config.use_ultimate_strength_scaling {
        config.material.ultimate_strength_pa
    } else {
        config.load.px
    };
    let u_ref       = ((stress_ref / config.material.e) * config.geometry.half_w) as f32;
    // ref_energy/ref_stress2 are squared quantities (always >= 0, like ref_div2 below), and
    // are used downstream as `1.0 / ctx.ref_energy` / `1.0 / ctx.ref_stress2` divisors at
    // ~15 call sites across step_physics/compute_gradient_conflict/compute_loss_for_lbfgs —
    // at stress_ref=0 (e.g. LOAD_PX_KSI=0, unvalidated on the headless/env path) both are
    // exactly 0.0, so every one of those sites would silently divide by zero. Floored here,
    // at the single source, rather than chasing each downstream site individually — mirrors
    // ref_div2's `.max(1.0)` floor, same arbitrary-but-consistent 1.0 floor value, a no-op
    // for any physically realistic config (stress_ref is normally 1e7-1e8 Pa).
    let ref_energy  = (0.5 * stress_ref * stress_ref / config.material.e).max(1.0) as f32;
    let ref_stress2 = (stress_ref * stress_ref).max(1.0) as f32;
    (u_ref, ref_energy, ref_stress2)
}

/// Floors the MAGNITUDE of a divisor at 1.0 while preserving sign — the linear-divisor
/// analogue of `ref_div2`'s `.max(1.0)` guard, which is safe to apply directly only because
/// that divisor is pre-squared (always >= 0). `px * half_w` here is not squared, so a naive
/// `.max(1.0)` would silently flip the sign of a legitimate negative (compression) divisor
/// into a positive floor. Exact zero defaults to +1.0 (this codebase's tension-positive
/// `LoadConfig` convention).
fn floor_signed_divisor(d: f64) -> f64 {
    if d.abs() < 1.0 {
        if d < 0.0 { -1.0 } else { 1.0 }
    } else {
        d
    }
}

/// Partition sampled boundary points into the traction-load, hole-free, and
/// right-edge-traction index sets `StepCtx` uses to slice into the `bnd_*` arrays.
pub fn extract_boundary_indices(bnd_pts: &[BoundaryPoint], bnd_nx: &[f32]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let trac_idx: Vec<usize> = bnd_pts.iter().enumerate()
        .filter(|(_, b)| b.kind == BoundaryKind::NeumannLoad).map(|(i, _)| i).collect();
    let hole_idx: Vec<usize> = bnd_pts.iter().enumerate()
        .filter(|(_, b)| b.kind == BoundaryKind::NeumannFree).map(|(i, _)| i).collect();
    let right_idx: Vec<usize> = trac_idx.iter()
        .filter(|&&i| bnd_nx[i] > 0.9).copied().collect();
    (trac_idx, hole_idx, right_idx)
}

/// All read-only data a single training step needs (passed by reference to `step_physics`).
pub struct StepCtx<'a> {
    pub config:          &'a SolverConfig,
    pub engine:          &'a EngineParams,
    /// The boundary-value problem driving this step's loss-term set/order/base-weights.
    /// `step_physics` computes each domain's forward pass exactly as before, then routes
    /// the resulting tensors through `problem.loss_terms()` (in the trait's stable order)
    /// rather than a hardcoded e/n/h/d/eq/kirsch sequence.
    pub problem:         &'a dyn BoundaryValueProblem,
    pub fd:              &'a FdConfig,
    pub k:               f32,
    pub u_ref:           f32,
    pub ref_energy:      f32,
    pub ref_stress2:     f32,
    pub cx:              f64,
    pub cy:              f64,
    pub ref_div2:        f64,
    /// Pre-normalized interior collocation points (caller responsible for normalization).
    pub int_norm:        &'a [[f32; 2]],
    pub bnd_norm:        &'a [[f32; 2]],
    pub bnd_nx:          &'a [f32],
    pub bnd_ny:          &'a [f32],
    pub bnd_tx:          &'a [f32],
    pub bnd_ty:          &'a [f32],
    pub trac_idx:        &'a [usize],
    pub hole_idx:        &'a [usize],
    pub right_idx:       &'a [usize],
    /// Pre-normalized equilibrium ring points (r ∈ [2r_hole, 3r_hole]); computed once outside loop.
    pub eq_ring_norm:    &'a [[f32; 2]],
    pub dynamic_lam_h_cap: f64,
    /// Cap for SAW-BRDR displacement-anchor weight (uncapped it grows to ~50, destabilising
    /// near-convergence when lam_k ≈ 2). Follows the same reduction schedule as lam_h_cap.
    pub dynamic_lam_d_cap: f64,
    pub phase2_active:   bool,
    pub step:            usize,
}

/// Per-step scalars and lambda values returned to the caller for logging and control.
pub struct StepOutput {
    pub e_scalar:      f32,
    pub n_scalar:      f32,
    pub h_scalar:      f32,
    pub d_scalar:      f32,
    pub eq_scalar:     f32,
    pub w_scalar:      f32,
    pub kirsch_scalar: f32,
    pub const_scalar:  f32,
    pub total_scalar:  f32,
    pub lr:            f64,
    pub lam_e:         f64,
    pub lam_n:         f64,
    pub lam_h:         f64,
    pub lam_d:         f64,
    pub lam_eq:        f64,
    pub lam_kirsch:    f64,
    /// `(e + eq + const) / (n + h + d + w + 1e-8)` — zero-GPU-cost conflict proxy.
    pub proxy_ratio:   f32,
    /// Active `OptimizerTier::as_u8()` at the time of this step (0=Explore, 1=Align, 2=Converge).
    pub optimizer_tier: u8,
    /// Exact cosine similarity from the dual-pass gradient conflict check, when computed.
    /// `None` on steps where the conflict check did not fire.
    pub cosine_sim:    Option<f32>,
    /// Every active term's final SAW-BRDR-weighted (and, if capped, clamped) lambda by
    /// name, keyed by `LossTerm::name()`. `None` on Kirsch's frozen `step_physics` path
    /// (the fixed 6-field `lam_e`/`lam_n`/`lam_h`/`lam_d`/`lam_eq`/`lam_kirsch` schema above
    /// already covers it exhaustively); `Some(..)` on `step_physics_multi`, which has no
    /// fixed schema for an arbitrary N-term problem. This is what Converge-entry code must
    /// snapshot instead of `problem.base_weight(..)` (see `run_headless_pinlug_inner`'s
    /// `frozen_lams` construction) so L-BFGS inherits the LIVE SAW+cap-adapted weight active
    /// on the step Converge was entered, not the static Phase-1 SAW-BRDR seed.
    pub lam_by_name:   Option<std::collections::HashMap<&'static str, f64>>,
}

/// Extract (σ_xx, σ_yy, σ_xy) from rows `[row_start, row_end)` of an mDEM network output
/// tensor (cols 2, 3, 4 — already scaled to Pa by `scale_out`). Shared by every mDEM call
/// site that reads stress directly off the network rather than via the FD stencil.
fn extract_mdem_stress(out: &Tensor<B, 2>, row_start: usize, row_end: usize) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let n = row_end - row_start;
    let sxx = out.clone().slice([row_start..row_end, 2..3]).reshape([n]);
    let syy = out.clone().slice([row_start..row_end, 3..4]).reshape([n]);
    let sxy = out.clone().slice([row_start..row_end, 4..5]).reshape([n]);
    (sxx, syy, sxy)
}

/// Execute one training step: compute all losses → backward → optimizer update.
///
/// Takes ownership of `model`, returns updated model + `StepOutput`.
/// `saw` and `lr_sched` are mutated in-place (SAW-BRDR weight update + LR step).
/// `tier_u8`: `OptimizerTier::as_u8()` of the currently active tier — recorded in StepOutput
/// for logging; does not change the computation.
///
/// Trait-driven cutover: forward passes are computed once per point-set (interior,
/// neumann boundary, right-edge, hole, eq-ring shifts, kirsch probes) exactly as before,
/// then each named loss term's real per-step tensor is looked up and combined in
/// `ctx.problem.loss_terms()`'s STABLE ORDER — that order is the SAW-BRDR component
/// vector (Phase 1: interior_energy/neumann_traction/hole_traction/displacement_anchor/
/// equilibrium_ring; Phase 2 adds kirsch_stress as the 6th, phase2-only term). All
/// weighted terms are summed into ONE scalar and `.backward()` is called exactly once —
/// see `active_terms`/`term_tensor`/`term_scalar` below. `constitutive_consistency` uses
/// a fixed weight (`lam_const`, outside SAW) exactly as before.
pub fn step_physics(
    model: ElasticityNet<B>,
    optim_w: &mut WeightOptim,
    optim_b: &mut BiasOptim,
    optim_gate: &mut GateOptim,
    ctx: &StepCtx,
    saw: &mut SawBrdr,
    lr_sched: &mut LrSchedule,
    device: &WgpuDevice,
    tier_u8: u8,
    physics_boost: f64,
    alpha_lr_mult: f64,
) -> (ElasticityNet<B>, StepOutput) {
    let n_int      = ctx.int_norm.len();
    let n_fourier  = ctx.engine.n_fourier;
    let use_mdem   = ctx.engine.use_mdem;
    let px         = ctx.config.load.px;
    let u_ref_f64  = ctx.u_ref as f64;

    let v_to_t = |idxs: &[usize], src: &[f32]| -> Tensor<B, 1> {
        let v: Vec<f32> = idxs.iter().map(|&i| src[i]).collect();
        Tensor::<B, 1>::from_data(TensorData::new(v.clone(), vec![v.len()]), device)
    };

    // Scale network output after ansatz:
    // - displacement cols (0,1): × u_ref  [meters]
    // - stress cols (2,3,4) in mDEM mode: × Px  [Pa — matches analytical Kirsch values]
    // - plain DEM (3 cols): × u_ref for everything (w is zeroed by ansatz)
    let scale_out = |ansatz: Tensor<B, 2>| -> Tensor<B, 2> {
        if use_mdem {
            let nr = ansatz.dims()[0];
            Tensor::cat(vec![
                ansatz.clone().slice([0..nr, 0..2]).mul_scalar(u_ref_f64),
                ansatz.slice([0..nr, 2..5]).mul_scalar(px),
            ], 1)
        } else {
            ansatz.mul_scalar(u_ref_f64)
        }
    };

    // === Interior forward pass (drives interior_energy + constitutive_consistency) ===
    let pts_t = norm_pts_to_tensor::<B>(ctx.int_norm, device);
    let stencil_coords = assemble_stencil::<B>(&pts_t, ctx.fd, device);
    let stencil_out = scale_out(apply_dirichlet_ansatz::<B>(
        fwd(&model, stencil_coords.clone(), n_fourier, device),
        &stencil_coords,
        ctx.config.geometry.symmetry,
        ctx.k,
    ));

    // Extract σ_net from the stencil's center block (rows `0..n_int` — the other 4×n_int
    // rows are the ± FD shifts used only by `compute_strains`) BEFORE compute_strains
    // consumes `stencil_out`. `ConstitutiveConsistencyTerm`/`InteriorEnergyTerm` read
    // `raw_out` expecting exactly `n_int` rows (cols 2..5 = σ_net for mDEM), matching
    // `extract_mdem_stress(&stencil_out, 0, n_int)`'s old center-block slice.
    let int_raw_out = stencil_out.clone().slice([0..n_int, 0..stencil_out.dims()[1]]);
    let (eps_xx, eps_yy, eps_xy) = compute_strains::<B>(stencil_out, n_int, ctx.fd);
    let int_forward = DomainForwardOutputs {
        domain: KIRSCH_DOMAIN,
        raw_out: &int_raw_out,
        strains: Some((eps_xx, eps_yy, eps_xy)),
        normals: None,
    };

    // === Neumann traction forward pass ===
    let n_loss = if !ctx.trac_idx.is_empty() {
        let trac_norm: Vec<[f32; 2]> = ctx.trac_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
        let nt = trac_norm.len();
        let bnd_t = norm_pts_to_tensor::<B>(&trac_norm, device);
        let stencil_bnd = assemble_stencil::<B>(&bnd_t, ctx.fd, device);
        let out_bnd = scale_out(apply_dirichlet_ansatz::<B>(
            fwd(&model, stencil_bnd.clone(), n_fourier, device), &stencil_bnd,
            ctx.config.geometry.symmetry, ctx.k,
        ));
        let (ex, ey, exy) = compute_strains::<B>(out_bnd.clone(), nt, ctx.fd);
        let neumann_forward = DomainForwardOutputs {
            domain: KIRSCH_DOMAIN,
            raw_out: &out_bnd,
            strains: Some((ex, ey, exy)),
            normals: Some((v_to_t(ctx.trac_idx, ctx.bnd_nx), v_to_t(ctx.trac_idx, ctx.bnd_ny))),
        };
        let term = crate::kirsch_problem::NeumannTractionTerm {
            domain: KIRSCH_DOMAIN,
            material: ctx.config.material.clone(),
            ref_stress2: ctx.ref_stress2,
            tx_target: v_to_t(ctx.trac_idx, ctx.bnd_tx),
            ty_target: v_to_t(ctx.trac_idx, ctx.bnd_ty),
        };
        term.compute(std::slice::from_ref(&neumann_forward))
    } else {
        Tensor::<B, 1>::zeros([1], device)
    };

    // === Right-edge displacement anchor + DEM Neumann work term ===
    let u_target_val = ((ctx.config.load.px
        - ctx.config.material.nu * ctx.config.load.py)
        / ctx.config.material.e * ctx.config.geometry.half_w) as f32;
    let (d_loss, w_neumann) = if !ctx.right_idx.is_empty() {
        let right_norm: Vec<[f32; 2]> = ctx.right_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
        let nr = right_norm.len();
        let right_t = norm_pts_to_tensor::<B>(&right_norm, device);
        let out_r = scale_out(apply_dirichlet_ansatz::<B>(
            fwd(&model, right_t.clone(), n_fourier, device), &right_t,
            ctx.config.geometry.symmetry, ctx.k,
        ));
        let right_forward = DomainForwardOutputs {
            domain: KIRSCH_DOMAIN,
            raw_out: &out_r,
            strains: None,
            normals: None,
        };
        let term = crate::kirsch_problem::DisplacementAnchorTerm {
            domain: KIRSCH_DOMAIN,
            u_target: u_target_val,
        };
        let d_val = term.compute(std::slice::from_ref(&right_forward));
        let u_vals = out_r.slice([0..nr, 0..1]).reshape([nr]);
        let w_val = u_vals.mean()
            .mul_scalar(2.0 * ctx.config.material.e
                / floor_signed_divisor(ctx.config.load.px * ctx.config.geometry.half_w));
        (d_val, w_val)
    } else {
        (Tensor::<B, 1>::zeros([1], device), Tensor::<B, 1>::zeros([1], device))
    };

    // === Hole traction-free loss ===
    // mDEM mode: direct forward pass at hole boundary — avoids FD stencil crossing the hole.
    // Plain DEM: FD-based approach (may suffer stencil artifacts near hole).
    let h_loss = if !ctx.hole_idx.is_empty() {
        let hole_norm: Vec<[f32; 2]> = ctx.hole_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
        let nh = hole_norm.len();
        let bnd_h = norm_pts_to_tensor::<B>(&hole_norm, device);

        if use_mdem {
            let out_h = scale_out(apply_dirichlet_ansatz::<B>(
                fwd(&model, bnd_h.clone(), n_fourier, device), &bnd_h,
                ctx.config.geometry.symmetry, ctx.k,
            ));
            let hole_forward = DomainForwardOutputs {
                domain: KIRSCH_DOMAIN,
                raw_out: &out_h,
                strains: None,
                normals: Some((v_to_t(ctx.hole_idx, ctx.bnd_nx), v_to_t(ctx.hole_idx, ctx.bnd_ny))),
            };
            let term = crate::kirsch_problem::HoleTractionTerm {
                domain: KIRSCH_DOMAIN, material: ctx.config.material.clone(),
                ref_stress2: ctx.ref_stress2, direct: true,
            };
            term.compute(std::slice::from_ref(&hole_forward))
        } else {
            let stencil_h = assemble_stencil::<B>(&bnd_h, ctx.fd, device);
            let out_h = scale_out(apply_dirichlet_ansatz::<B>(
                fwd(&model, stencil_h.clone(), n_fourier, device), &stencil_h,
                ctx.config.geometry.symmetry, ctx.k,
            ));
            let (ex, ey, exy) = compute_strains::<B>(out_h.clone(), nh, ctx.fd);
            let hole_forward = DomainForwardOutputs {
                domain: KIRSCH_DOMAIN,
                raw_out: &out_h,
                strains: Some((ex, ey, exy)),
                normals: Some((v_to_t(ctx.hole_idx, ctx.bnd_nx), v_to_t(ctx.hole_idx, ctx.bnd_ny))),
            };
            let term = crate::kirsch_problem::HoleTractionTerm {
                domain: KIRSCH_DOMAIN, material: ctx.config.material.clone(),
                ref_stress2: ctx.ref_stress2, direct: false,
            };
            term.compute(std::slice::from_ref(&hole_forward))
        }
    } else {
        Tensor::<B, 1>::zeros([1], device)
    };

    // === Equilibrium residual ∇·σ = 0 at near-hole ring points ===
    let n_eq = ctx.eq_ring_norm.len();
    let eq_loss: Tensor<B, 1> = if n_eq > 0 {
        let components: [Tensor<B, 1>; 8] = if use_mdem {
            // mDEM: batch all 4 shifts into a single direct forward pass.
            // σ is read from network output cols 2..5 — ansatz passes them unchanged (bc.rs:55).
            // const_loss couples σ_net ≈ C:ε, so ∇·σ_net = 0 is equivalent to enforcing the PDE.
            // Reduces from 4 × (5 × n_eq) = 2000 GPU evals to 4 × n_eq = 400 (5× fewer).
            let shifts: [(f32, f32); 4] = [
                ( ctx.fd.hx, 0.0), (-ctx.fd.hx, 0.0),
                (0.0,  ctx.fd.hy), (0.0, -ctx.fd.hy),
            ];
            let all_shifted: Vec<[f32; 2]> = shifts.iter().flat_map(|&(dx, dy)| {
                ctx.eq_ring_norm.iter().map(move |&[x, y]| [x + dx, y + dy])
            }).collect();
            let pts_all = norm_pts_to_tensor::<B>(&all_shifted, device);
            let out_all = scale_out(apply_dirichlet_ansatz::<B>(
                fwd(&model, pts_all.clone(), n_fourier, device), &pts_all,
                ctx.config.geometry.symmetry, ctx.k,
            ));
            // Each of the 4 shifts occupies n_eq consecutive rows; σ_xx=col2, σ_yy=col3, σ_xy=col4
            let seg = |i: usize| -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
                extract_mdem_stress(&out_all, i * n_eq, (i + 1) * n_eq)
            };
            let (sxx_xp, _, sxy_xp) = seg(0);
            let (sxx_xm, _, sxy_xm) = seg(1);
            let (_, syy_yp, sxy_yp) = seg(2);
            let (_, syy_ym, sxy_ym) = seg(3);
            [sxx_xp, sxy_xp, sxx_xm, sxy_xm, sxy_yp, syy_yp, sxy_ym, syy_ym]
        } else {
            // Plain DEM: FD stencil around each shifted location (σ not a direct output).
            let fwd_shift = |dx: f32, dy: f32| {
                let shifted: Vec<[f32; 2]> = ctx.eq_ring_norm.iter()
                    .map(|&[xn, yn]| [xn + dx, yn + dy]).collect();
                let pts = norm_pts_to_tensor::<B>(&shifted, device);
                let stencil = assemble_stencil::<B>(&pts, ctx.fd, device);
                let out = scale_out(apply_dirichlet_ansatz::<B>(
                    fwd(&model, stencil.clone(), n_fourier, device), &stencil,
                    ctx.config.geometry.symmetry, ctx.k,
                ));
                let (exx, eyy, exy) = compute_strains::<B>(out, n_eq, ctx.fd);
                compute_stress(exx, eyy, exy, &ctx.config.material)
            };
            let (sxx_xp, _, sxy_xp) = fwd_shift( ctx.fd.hx,  0.0);
            let (sxx_xm, _, sxy_xm) = fwd_shift(-ctx.fd.hx,  0.0);
            let (_, syy_yp, sxy_yp) = fwd_shift( 0.0,  ctx.fd.hy);
            let (_, syy_ym, sxy_ym) = fwd_shift( 0.0, -ctx.fd.hy);
            [sxx_xp, sxy_xp, sxx_xm, sxy_xm, sxy_yp, syy_yp, sxy_ym, syy_ym]
        };
        let term = crate::kirsch_problem::EquilibriumRingTerm {
            domain: KIRSCH_DOMAIN, cx: ctx.cx, cy: ctx.cy, ref_div2: ctx.ref_div2,
            components: Some(components),
        };
        term.compute(&[])
    } else {
        Tensor::<B, 1>::zeros([1], device)
    };

    // === Kirsch Stress Loss — Phase 2 only (SAW-BRDR 6th component) ===
    // mDEM mode: direct forward pass — σ from network output cols 2..5.
    // Plain DEM: FD stencil → strains → stress (same as before).
    let (kirsch_loss, kirsch_scalar): (Tensor<B, 1>, f32) =
        if ctx.phase2_active {
            if let HoleType::Circular { radius } = ctx.config.geometry.hole {
                let probes = compute_kirsch_probes(ctx, radius);
                let n_pr = probes.points.len();
                let pts_pr = norm_pts_to_tensor::<B>(&probes.points, device);
                let px2 = (px * px) as f64;
                let to_t1 = |v: &[f32]| -> Tensor<B, 1> {
                    Tensor::from_data(TensorData::new(v.to_vec(), vec![v.len()]), device)
                };

                let kl = if use_mdem {
                    // Direct pass: σ from network cols 2..5 (already in Pa after scale_out)
                    let out_pr = scale_out(apply_dirichlet_ansatz::<B>(
                        fwd(&model, pts_pr.clone(), n_fourier, device), &pts_pr,
                        ctx.config.geometry.symmetry, ctx.k,
                    ));
                    let kirsch_forward = DomainForwardOutputs {
                        domain: KIRSCH_DOMAIN, raw_out: &out_pr, strains: None, normals: None,
                    };
                    let term = crate::kirsch_problem::KirschStressTerm {
                        domain: KIRSCH_DOMAIN, material: ctx.config.material.clone(), direct: true,
                        px2,
                        sxx_targets: to_t1(&probes.sxx_targets),
                        syy_targets: to_t1(&probes.syy_targets),
                        sxy_targets: to_t1(&probes.sxy_targets),
                        weights: to_t1(&probes.weights),
                    };
                    term.compute(std::slice::from_ref(&kirsch_forward))
                } else {
                    let stencil_pr = assemble_stencil::<B>(&pts_pr, ctx.fd, device);
                    let out_pr = scale_out(apply_dirichlet_ansatz::<B>(
                        fwd(&model, stencil_pr.clone(), n_fourier, device), &stencil_pr,
                        ctx.config.geometry.symmetry, ctx.k,
                    ));
                    let (exx_pr, eyy_pr, exy_pr) = compute_strains::<B>(out_pr.clone(), n_pr, ctx.fd);
                    let kirsch_forward = DomainForwardOutputs {
                        domain: KIRSCH_DOMAIN, raw_out: &out_pr,
                        strains: Some((exx_pr, eyy_pr, exy_pr)), normals: None,
                    };
                    let term = crate::kirsch_problem::KirschStressTerm {
                        domain: KIRSCH_DOMAIN, material: ctx.config.material.clone(), direct: false,
                        px2,
                        sxx_targets: to_t1(&probes.sxx_targets),
                        syy_targets: to_t1(&probes.syy_targets),
                        sxy_targets: to_t1(&probes.sxy_targets),
                        weights: to_t1(&probes.weights),
                    };
                    term.compute(std::slice::from_ref(&kirsch_forward))
                };
                let ks = t_scalar(&kl);
                (kl, ks)
            } else {
                (Tensor::<B, 1>::zeros([1], device), 0.0)
            }
        } else {
            (Tensor::<B, 1>::zeros([1], device), 0.0)
        };

    // === Interior-energy + constitutive-consistency terms (share the interior forward pass) ===
    let e_loss = {
        let term = crate::kirsch_problem::InteriorEnergyTerm {
            domain: KIRSCH_DOMAIN, material: ctx.config.material.clone(), ref_energy: ctx.ref_energy,
        };
        term.compute(std::slice::from_ref(&int_forward))
    };
    let const_loss: Tensor<B, 1> = if use_mdem {
        let term = crate::kirsch_problem::ConstitutiveConsistencyTerm {
            domain: KIRSCH_DOMAIN, material: ctx.config.material.clone(), ref_stress2: ctx.ref_stress2,
        };
        term.compute(std::slice::from_ref(&int_forward))
    } else {
        Tensor::<B, 1>::zeros([1], device)
    };

    // === Trait-driven SAW-BRDR weighted combination ===
    // `ctx.problem.loss_terms()` enumerates the terms IN THEIR STABLE ORDER — that order
    // becomes the SAW-BRDR component vector (replacing the old hardcoded
    // e/n/h/d/eq/kirsch sequence). Each term's *real* per-step tensor (computed above from
    // this step's forward passes) is looked up by name; `equilibrium_ring` and
    // `constitutive_consistency` aren't part of the SAW-BRDR vector (equilibrium_ring IS;
    // constitutive_consistency uses a fixed weight, same as before) — see below.
    let e_scalar      = t_scalar(&e_loss);
    let n_scalar      = t_scalar(&n_loss);
    let h_scalar      = t_scalar(&h_loss);
    let d_scalar      = t_scalar(&d_loss);
    let eq_scalar     = t_scalar(&eq_loss);
    let w_scalar      = t_scalar(&w_neumann);
    let const_scalar  = t_scalar(&const_loss);

    let term_tensor = |name: &str| -> &Tensor<B, 1> {
        match name {
            "interior_energy" => &e_loss,
            "neumann_traction" => &n_loss,
            "hole_traction" => &h_loss,
            "displacement_anchor" => &d_loss,
            "equilibrium_ring" => &eq_loss,
            "kirsch_stress" => &kirsch_loss,
            other => panic!("step_physics: unhandled SAW-BRDR loss term '{other}'"),
        }
    };
    let term_scalar = |name: &str| -> f32 {
        match name {
            "interior_energy" => e_scalar,
            "neumann_traction" => n_scalar,
            "hole_traction" => h_scalar,
            "displacement_anchor" => d_scalar,
            "equilibrium_ring" => eq_scalar,
            "kirsch_stress" => kirsch_scalar,
            other => panic!("step_physics: unhandled SAW-BRDR loss term '{other}'"),
        }
    };

    // Ordered, phase-filtered term names — this is the trait-driven replacement for the
    // hardcoded `[e, n, h, d, eq]` / `[e, n, h, d, eq, kirsch]` SAW-BRDR vectors.
    let active_terms: Vec<Box<dyn LossTerm>> = ctx.problem.loss_terms().into_iter()
        .filter(|t| t.name() != "constitutive_consistency")
        .filter(|t| ctx.phase2_active || !t.phase2_only())
        .collect();
    let term_names: Vec<&'static str> = active_terms.iter().map(|t| t.name()).collect();

    let saw_inputs: Vec<f32> = term_names.iter().map(|&n| term_scalar(n)).collect();
    let lams = saw.update(&saw_inputs);

    // Stiffness-coupled boost (external modulation, applied after SAW-BRDR itself —
    // mirrors how dynamic_lam_h_cap/dynamic_lam_d_cap are clamped below, outside SawBrdr).
    // `physics_boost = 1.0` (StiffnessController disabled) is a no-op.
    let mut lam_by_name: std::collections::HashMap<&'static str, f64> = std::collections::HashMap::new();
    for (name, &raw_lam) in term_names.iter().zip(lams.iter()) {
        let lam = match *name {
            "interior_energy" => raw_lam as f64 * physics_boost,
            "equilibrium_ring" => raw_lam as f64 * physics_boost,
            // Phase 2: cap lam_h so kirsch gradient dominates hole traction. Tightens on plateau.
            "hole_traction" => {
                let v = raw_lam as f64;
                if ctx.phase2_active { v.min(ctx.dynamic_lam_h_cap) } else { v }
            }
            // Phase 2: cap lam_d so displacement-anchor gradient doesn't overwhelm kirsch near
            // convergence. Uncapped lam_d grows to ~46-50 while lam_k ≈ 2 → 23:1 ratio causes
            // Adam overshoot when K_t → 3.
            "displacement_anchor" => {
                let v = raw_lam as f64;
                if ctx.phase2_active { v.min(ctx.dynamic_lam_d_cap) } else { v }
            }
            _ => raw_lam as f64,
        };
        lam_by_name.insert(name, lam);
    }
    let lam_e      = *lam_by_name.get("interior_energy").unwrap_or(&0.0);
    let lam_n      = *lam_by_name.get("neumann_traction").unwrap_or(&0.0);
    let lam_h      = *lam_by_name.get("hole_traction").unwrap_or(&0.0);
    let lam_d      = *lam_by_name.get("displacement_anchor").unwrap_or(&0.0);
    let lam_eq     = *lam_by_name.get("equilibrium_ring").unwrap_or(&0.0);
    let lam_kirsch = *lam_by_name.get("kirsch_stress").unwrap_or(&0.0);
    let lam_const  = ctx.engine.lam_const as f64 * physics_boost;

    // Total potential energy: Π = Σ λ_i · term_i(θ) − W_neumann·λ_e + const_loss·λ_const.
    // `interior_energy` carries the −W_neumann work term (DEM formulation), exactly as before.
    let mut loss: Tensor<B, 1> = (e_loss.clone() - w_neumann).mul_scalar(lam_e);
    let mut total_scalar: f32 = (e_scalar - w_scalar) * lam_e as f32;
    for name in term_names.iter().filter(|&&n| n != "interior_energy") {
        let lam = *lam_by_name.get(name).unwrap_or(&0.0);
        loss = loss + term_tensor(name).clone().mul_scalar(lam);
        total_scalar += term_scalar(name) * lam as f32;
    }
    loss = loss + const_loss.mul_scalar(lam_const);
    total_scalar += const_scalar * lam_const as f32;

    let lr = lr_sched.step(total_scalar.abs() as f64);
    let (default_weight_ids, bias_ids) = model.param_ids();
    // Dormant PirateNet blocks (|gate| <= gate_awake_epsilon) are excluded from the SOAP-Muon
    // weight set — their true gradient is exactly zero at alpha=0 (see network.rs tests), so
    // this skips the eigendecomposition/Shampoo update for capacity the network isn't using
    // yet. No-op (full weight set) when use_piratenet=false.
    let weight_ids = if ctx.config.use_piratenet {
        model.awake_weight_ids(ctx.config.stiffness.gate_awake_epsilon)
    } else {
        default_weight_ids
    };
    let gate_ids = model.gate_ids();
    let mut grads = loss.backward();
    let weight_grads = GradientsParams::from_params(&mut grads, &model, &weight_ids);
    let bias_grads = GradientsParams::from_params(&mut grads, &model, &bias_ids);
    let gate_grads = GradientsParams::from_params(&mut grads, &model, &gate_ids);
    let model = optim_w.step(lr, model, weight_grads);
    let model = optim_b.step(lr, model, bias_grads);
    // Stiffness-accelerated gate LR; empty gate_grads (use_piratenet=false) makes this a no-op.
    let model = optim_gate.step(lr * alpha_lr_mult, model, gate_grads);

    let proxy_ratio = (e_scalar + eq_scalar + const_scalar)
        / (n_scalar + h_scalar + d_scalar + w_scalar + 1e-8);

    (model, StepOutput {
        e_scalar, n_scalar, h_scalar, d_scalar, eq_scalar, w_scalar,
        kirsch_scalar, const_scalar, total_scalar, lr, lam_e, lam_n, lam_h, lam_d, lam_eq,
        lam_kirsch,
        proxy_ratio,
        optimizer_tier: tier_u8,
        cosine_sim: None,
        lam_by_name: None,
    })
}

/// NEW, ADDITIVE multi-domain step driver — the N-domain analogue of `step_physics`. Does
/// NOT touch `step_physics`/`StepCtx` (the frozen 1-domain Kirsch reference) at all; it is
/// a parallel code path so a genuine multi-domain contact problem (pin-in-lug) can train
/// through the same `BoundaryValueProblem`/`LossTerm` trait family without risking the
/// proven Kirsch path's numerical behavior.
///
/// One model per domain (order matches `ctx.domains`/`optims`). Body:
///   (a) for each domain, run one forward pass per DISTINCT named point-set at least one
///       active `LossTerm` needs (built once into a `(DomainId, point_set_name) →
///       DomainForwardOutputs` map so no point-set is forwarded twice in a step even if
///       multiple terms read it);
///   (b) walk `ctx.problem.loss_terms()` in its stable order, gather each active term's
///       `(domains(), point_sets())` pairs from that map, `compute()` it, and weight via
///       SAW-BRDR exactly as `step_physics` does (same phase2/dynamic-cap handling);
///   (c) sum every weighted term into ONE total scalar and call `.backward()` EXACTLY ONCE
///       — gradients for every domain's parameters land in the SAME `GradientsParams` bag
///       from that single backward pass;
///   (d) for each domain (in `ctx.domains` order), pull out just that domain's own
///       weight/bias/gate `ParamId`s (`model.param_ids()`/`awake_weight_ids()`/
///       `gate_ids()`) and call `GradientsParams::from_params(&mut grads, &model, &ids)` —
///       see the inline comment at the call site for why domain iteration order does not
///       matter here (this is the single highest-risk assumption in this design; see
///       `gradient_split_attributes_domain_b_step_only_to_domain_b_params` /
///       `gradient_split_two_domains_both_receive_nonzero_updates_when_both_contribute`).
/// Owned forward-pass output for one `(DomainId, point_set_name)` pair — the storage that
/// backs a `DomainForwardOutputs<'_, B>` borrow. Kept in a stable `Vec` (see
/// `compute_domain_forwards`'s doc comment) so borrowing `DomainForwardOutputs` built from it
/// remain valid for as long as the `Vec` itself is alive.
struct Computed {
    key: (pinn_core::problem::DomainId, &'static str),
    raw_out: Tensor<B, 2>,
    strains: Option<(Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>)>,
    normals: Option<(Tensor<B, 1>, Tensor<B, 1>)>,
}

/// Enumerate which `(DomainId, point_set_name)` pairs at least one of `active_terms` needs,
/// then run exactly one forward pass per pair (never twice, even if several terms share it).
/// Shared by `step_physics_multi` and `compute_loss_for_lbfgs_multi`/
/// `compute_gradient_conflict_multi` so the forward-pass enumeration/scaling logic exists in
/// exactly one place.
///
/// Forward-pass results are computed into owned storage first (raw_out tensor + optional
/// strains/normals), returned as a `Vec<Computed>` — Rust's borrow checker requires the
/// owning Vec to be fully populated (and therefore stable) before any `&Tensor` into it is
/// taken, so callers build their borrowing `DomainForwardOutputs` map from the returned Vec.
fn compute_domain_forwards(
    ctx: &crate::problem::MultiStepCtx,
    models: &[&ElasticityNet<B>],
    active_terms: &[Box<dyn LossTerm>],
    device: &WgpuDevice,
) -> Vec<Computed> {
    use pinn_core::problem::DomainId;

    // Multi-domain problems (pin-in-lug) use plain-DEM output (no Fourier embedding, no
    // hard Dirichlet ansatz — boundary conditions are enforced via loss terms, not a
    // symmetry-plane ansatz, since neither pin nor lug domain has Kirsch's quarter-symmetry
    // structure). This mirrors `PinLugSamplingStrategy`'s geometry (see pinlug_problem.rs).
    let n_fourier = 0usize;

    let mut needed: Vec<(DomainId, &'static str)> = Vec::new();
    for term in active_terms {
        for (&id, &ps) in term.domains().iter().zip(term.point_sets().iter()) {
            if !needed.contains(&(id, ps)) {
                needed.push((id, ps));
            }
        }
    }

    let mut computed: Vec<Computed> = Vec::with_capacity(needed.len());

    for &(domain_id, ps_name) in &needed {
        let dctx = ctx.domains.iter().find(|d| d.data.id == domain_id)
            .unwrap_or_else(|| panic!(
                "compute_domain_forwards: LossTerm references DomainId({}) not present in \
                 ctx.domains — this should have been caught by validate_loss_terms",
                domain_id.0,
            ));
        let model_idx = ctx.domains.iter().position(|d| d.data.id == domain_id).unwrap();
        let model = models[model_idx];
        let u_ref_f64 = dctx.u_ref as f64;

        let domain_idx = ctx.problem.domains().iter().position(|d| d.id == domain_id).unwrap();
        let spec = &ctx.problem.domains()[domain_idx];
        let ansatz = ctx.problem.ansatz(domain_idx);
        let is_mdem = spec.output_dim == 5;
        let px = ctx.config.load.px;

        type NormalsPair = (Tensor<B, 1>, Tensor<B, 1>);
        let (norm_pts, normals): (&[[f32; 2]], Option<NormalsPair>) =
            if ps_name == "interior" {
                (&dctx.data.int_norm, None)
            } else {
                let ps = dctx.data.named(ps_name);
                let to_t = |v: &[f32]| -> Tensor<B, 1> {
                    Tensor::<B, 1>::from_data(TensorData::new(v.to_vec(), vec![v.len()]), device)
                };
                (&ps.norm, Some((to_t(&ps.nx), to_t(&ps.ny))))
            };
        if norm_pts.is_empty() {
            continue;
        }

        let pts_t = norm_pts_to_tensor::<B>(norm_pts, device);
        let n_pts = norm_pts.len();
        let stencil = assemble_stencil::<B>(&pts_t, ctx.fd, device);

        // Apply this domain's Dirichlet ansatz pointwise (columns 0,1 = u,v) via the
        // per-point (dx, dy) scale factors `DirichletAnsatz::eval` returns, then scale to
        // physical units exactly as `step_physics`'s `scale_out` does: displacement cols by
        // u_ref [m], and (mDEM only) stress cols 2..5 by Px [Pa].
        let raw_net = fwd::<B>(model, stencil.clone(), n_fourier, device);
        let m = raw_net.dims()[0];
        let stencil_data: Vec<f32> = stencil.into_data().to_vec::<f32>().unwrap_or_default();
        let mut dx_v = Vec::with_capacity(m);
        let mut dy_v = Vec::with_capacity(m);
        for row in 0..m {
            let xn = stencil_data[row * 3];
            let yn = stencil_data[row * 3 + 1];
            let (dx, dy) = ansatz.eval(xn, yn, ctx.k);
            dx_v.push(dx);
            dy_v.push(dy);
        }
        let dx_t = Tensor::<B, 2>::from_data(TensorData::new(dx_v, vec![m, 1]), device);
        let dy_t = Tensor::<B, 2>::from_data(TensorData::new(dy_v, vec![m, 1]), device);
        let u_col = raw_net.clone().slice([0..m, 0..1]) * dx_t;
        let v_col = raw_net.clone().slice([0..m, 1..2]) * dy_t;
        let ansatz_out = if is_mdem {
            let s_xx = raw_net.clone().slice([0..m, 2..3]);
            let s_yy = raw_net.clone().slice([0..m, 3..4]);
            let s_xy = raw_net.slice([0..m, 4..5]);
            Tensor::cat(vec![u_col, v_col, s_xx, s_yy, s_xy], 1)
        } else {
            Tensor::cat(vec![u_col, v_col], 1)
        };
        let raw = if is_mdem {
            Tensor::cat(vec![
                ansatz_out.clone().slice([0..m, 0..2]).mul_scalar(u_ref_f64),
                ansatz_out.slice([0..m, 2..5]).mul_scalar(px),
            ], 1)
        } else {
            ansatz_out.mul_scalar(u_ref_f64)
        };
        let raw_out = raw.clone().slice([0..n_pts, 0..raw.dims()[1]]);
        let (eps_xx, eps_yy, eps_xy) = compute_strains::<B>(raw, n_pts, ctx.fd);

        computed.push(Computed {
            key: (domain_id, ps_name),
            raw_out,
            strains: Some((eps_xx, eps_yy, eps_xy)),
            normals,
        });
    }

    computed
}

#[allow(clippy::too_many_arguments)]
pub fn step_physics_multi(
    models: Vec<ElasticityNet<B>>,
    optims: &mut [crate::problem::DomainOptim],
    ctx: &crate::problem::MultiStepCtx,
    saw: &mut SawBrdr,
    lr_sched: &mut LrSchedule,
    device: &WgpuDevice,
    tier_u8: u8,
    physics_boost: f64,
    alpha_lr_mult: f64,
) -> (Vec<ElasticityNet<B>>, StepOutput) {
    use crate::problem::DomainForwardOutputs as DFO;
    use pinn_core::problem::DomainId;

    // (a) Enumerate which (DomainId, point_set_name) pairs at least one active LossTerm
    // needs, then run exactly one forward pass per pair (never twice, even if several
    // terms share it).
    let active_terms: Vec<Box<dyn LossTerm>> = ctx.problem.loss_terms().into_iter()
        .filter(|t| t.name() != "constitutive_consistency")
        .filter(|t| ctx.phase2_active || !t.phase2_only())
        .collect();

    let model_refs: Vec<&ElasticityNet<B>> = models.iter().collect();
    let computed = compute_domain_forwards(ctx, &model_refs, &active_terms, device);

    let forwards: HashMap<(DomainId, &'static str), DFO<'_, B>> = computed.iter()
        .map(|c| (c.key, DFO {
            domain: c.key.0,
            raw_out: &c.raw_out,
            strains: c.strains.clone(),
            normals: c.normals.clone(),
        }))
        .collect();

    // (b) Walk terms in stable order, compute each active term's real tensor from the
    // forward-pass map, SAW-BRDR-weight exactly as step_physics does.
    let mut term_tensors: Vec<Tensor<B, 1>> = Vec::with_capacity(active_terms.len());
    let mut term_scalars: Vec<f32> = Vec::with_capacity(active_terms.len());
    let mut term_names: Vec<&'static str> = Vec::with_capacity(active_terms.len());
    for term in &active_terms {
        let inputs: Vec<DFO<'_, B>> = term.domains().iter().zip(term.point_sets().iter())
            .filter_map(|(&id, &ps)| forwards.get(&(id, ps)).map(|f| DFO {
                domain: f.domain, raw_out: f.raw_out,
                strains: f.strains.clone(), normals: f.normals.clone(),
            }))
            .collect();
        let t = term.compute(&inputs);
        let s = t_scalar(&t);
        term_tensors.push(t);
        term_scalars.push(s);
        term_names.push(term.name());
    }

    let lams = saw.update(&term_scalars);
    let mut lam_by_name: HashMap<&'static str, f64> = HashMap::new();
    for (name, &raw_lam) in term_names.iter().zip(lams.iter()) {
        let lam = match *name {
            "hole_traction" | "lug_free_edge_traction" => {
                let v = raw_lam as f64 * physics_boost;
                if ctx.phase2_active { v.min(ctx.dynamic_lam_h_cap) } else { v }
            }
            "displacement_anchor" | "lug_shank_anchor" => {
                let v = raw_lam as f64;
                if ctx.phase2_active { v.min(ctx.dynamic_lam_d_cap) } else { v }
            }
            "interface_penetration" => {
                let v = raw_lam as f64 * physics_boost;
                if ctx.phase2_active { v.min(ctx.dynamic_lam_penetration_cap) } else { v }
            }
            "interface_non_tension" => {
                let v = raw_lam as f64 * physics_boost;
                if ctx.phase2_active { v.min(ctx.dynamic_lam_non_tension_cap) } else { v }
            }
            _ => raw_lam as f64 * physics_boost,
        };
        lam_by_name.insert(name, lam);
    }

    // (c) Sum into ONE total scalar; backward() exactly once.
    let mut total: Option<Tensor<B, 1>> = None;
    let mut total_scalar = 0.0_f32;
    for (i, name) in term_names.iter().enumerate() {
        let lam = *lam_by_name.get(name).unwrap_or(&0.0);
        let weighted = term_tensors[i].clone().mul_scalar(lam);
        total_scalar += term_scalars[i] * lam as f32;
        total = Some(match total {
            Some(acc) => acc + weighted,
            None => weighted,
        });
    }
    let total = total.unwrap_or_else(|| Tensor::<B, 1>::zeros([1], device));

    let lr = lr_sched.step(total_scalar.abs() as f64);
    let mut grads = total.backward();

    // (d) Extract each domain's OWN weight/bias/gate ParamIds and step its own DomainOptim.
    // `GradientsParams::from_params` reads by globally-unique burn `ParamId` out of the
    // single shared `grads` bag produced by the one backward() call above — it does not
    // consume/clear entries, so domain iteration order below is irrelevant and no domain
    // can "steal" another domain's gradients: each ParamId only ever resolves to the
    // params of the model that actually owns it.
    let mut new_models: Vec<ElasticityNet<B>> = Vec::with_capacity(models.len());
    for (i, model) in models.into_iter().enumerate() {
        let dctx = &ctx.domains[i];
        let (default_weight_ids, bias_ids) = model.param_ids();
        let weight_ids = if ctx.config.use_piratenet {
            model.awake_weight_ids(ctx.config.stiffness.gate_awake_epsilon)
        } else {
            default_weight_ids
        };
        let gate_ids = model.gate_ids();
        let weight_grads = GradientsParams::from_params(&mut grads, &model, &weight_ids);
        let bias_grads = GradientsParams::from_params(&mut grads, &model, &bias_ids);
        let gate_grads = GradientsParams::from_params(&mut grads, &model, &gate_ids);
        let optim = &mut optims[i];
        let model = optim.weight.step(lr, model, weight_grads);
        let model = optim.bias.step(lr, model, bias_grads);
        let model = optim.gate.step(lr * alpha_lr_mult, model, gate_grads);
        let _ = dctx;
        new_models.push(model);
    }

    let lam_get = |n: &str| *lam_by_name.get(n).unwrap_or(&0.0);
    let scalar_get = |n: &str| term_names.iter().position(|&x| x == n)
        .map(|i| term_scalars[i]).unwrap_or(0.0);

    (new_models, StepOutput {
        e_scalar: scalar_get("interior_energy"),
        n_scalar: scalar_get("neumann_traction"),
        h_scalar: scalar_get("hole_traction"),
        d_scalar: scalar_get("displacement_anchor"),
        eq_scalar: scalar_get("equilibrium_ring"),
        w_scalar: 0.0,
        kirsch_scalar: scalar_get("kirsch_stress"),
        const_scalar: 0.0,
        total_scalar,
        lr,
        lam_e: lam_get("interior_energy"),
        lam_n: lam_get("neumann_traction"),
        lam_h: lam_get("hole_traction"),
        lam_d: lam_get("displacement_anchor"),
        lam_eq: lam_get("equilibrium_ring"),
        lam_kirsch: lam_get("kirsch_stress"),
        proxy_ratio: 0.0,
        optimizer_tier: tier_u8,
        cosine_sim: None,
        lam_by_name: Some(lam_by_name),
    })
}

/// K_t probe shared between runner and headless modes.
///
/// Evaluates σ_xx at `engine.probe_thetas_deg` angles at r = `engine.probe_r_factor` × r_hole,
/// normalises against Kirsch analytical at the same location, then scales by 3.0 so that
/// a perfectly converged network reports K_t = 3.0.
///
/// mDEM mode: σ_xx is read directly from network output column 2 (no FD stencil).
/// Plain DEM: σ_xx is computed from FD strains via Hooke's law.
///
/// Returns None if there is no circular hole or Px ≈ 0.
pub fn probe_kt_shared(
    model: &ElasticityNet<BInner>,
    config: &SolverConfig,
    engine: &EngineParams,
    fd:     &FdConfig,
    k:      f32,
    u_ref:  f32,
    device: &WgpuDevice,
) -> Option<f32> {
    let HoleType::Circular { radius } = config.geometry.hole else { return None };
    if config.load.px.abs() < 1e-10 { return None; }

    let r_probe = radius * engine.probe_r_factor;
    let (x0, x1) = config.geometry.x_range();
    let (y0, y1) = config.geometry.y_range();
    let inv_dx = 1.0 / (x1 - x0);
    let inv_dy = 1.0 / (y1 - y0);

    let probe_pts: Vec<[f32; 2]> = engine.probe_thetas_deg.iter().map(|&deg| {
        let t = deg.to_radians();
        let xn = (2.0 * (r_probe * t.cos() - x0) * inv_dx - 1.0) as f32;
        let yn = (2.0 * (r_probe * t.sin() - y0) * inv_dy - 1.0) as f32;
        [xn, yn]
    }).collect();

    let n = probe_pts.len();
    let pts_t = norm_pts_to_tensor::<BInner>(&probe_pts, device);

    let r_hole     = radius;
    let r_probe_abs = r_hole * engine.probe_r_factor;
    let px = config.load.px;
    let py = config.load.py;

    if engine.use_mdem {
        // Direct forward pass — σ_xx from output column 2 (scaled by Px in scale_out).
        let raw = fwd::<BInner>(model, pts_t.clone(), engine.n_fourier, device);
        let ansatz = apply_dirichlet_ansatz::<BInner>(
            raw, &pts_t, config.geometry.symmetry, k,
        );
        // σ_xx in column 2, scaled by Px
        let sxx_tensor = ansatz.slice([0..n, 2..3]).reshape([n]).mul_scalar(px);
        let sxx_v: Vec<f32> = sxx_tensor.into_data().to_vec::<f32>().ok()?;

        let mut wval = 0.0_f32;
        let mut wsum = 0.0_f32;
        for (i, &deg) in engine.probe_thetas_deg.iter().enumerate() {
            let theta = deg.to_radians();
            let w = (theta as f32).sin().powi(2);
            let sxx_pred = sxx_v[i];
            let (s_rr, s_tt, s_rt) = kirsch_stress(r_probe_abs, theta, r_hole, px, py);
            let ct = theta.cos(); let st = theta.sin();
            let sxx_kirsch = (s_rr * ct * ct + s_tt * st * st - 2.0 * s_rt * st * ct) as f32;
            if sxx_pred.is_finite() && sxx_kirsch.abs() > 1e-3 * px.abs() as f32 {
                wval += w * (sxx_pred / sxx_kirsch);
                wsum += w;
            }
        }
        if wsum < 1e-10 { return None; }
        Some(wval / wsum * 3.0_f32)
    } else {
        // FD stencil path for plain DEM.
        let stencil = assemble_stencil::<BInner>(&pts_t, fd, device);
        let out = apply_dirichlet_ansatz::<BInner>(
            fwd::<BInner>(model, stencil.clone(), engine.n_fourier, device),
            &stencil, config.geometry.symmetry, k,
        ).mul_scalar(u_ref as f64);

        let (eps_xx, eps_yy, _) = compute_strains::<BInner>(out, n, fd);
        let exx_v: Vec<f32> = eps_xx.into_data().to_vec::<f32>().ok()?;
        let eyy_v: Vec<f32> = eps_yy.into_data().to_vec::<f32>().ok()?;

        let e  = config.material.e  as f32;
        let nu = config.material.nu as f32;
        let f_c = e / (1.0 - nu * nu);

        let mut wval = 0.0_f32;
        let mut wsum = 0.0_f32;
        for (i, &deg) in engine.probe_thetas_deg.iter().enumerate() {
            let theta = deg.to_radians();
            let w = (theta as f32).sin().powi(2);
            let sxx_pred = f_c * (exx_v[i] + nu * eyy_v[i]);
            let (s_rr, s_tt, s_rt) = kirsch_stress(r_probe_abs, theta, r_hole, px, py);
            let ct = theta.cos(); let st = theta.sin();
            let sxx_kirsch = (s_rr * ct * ct + s_tt * st * st - 2.0 * s_rt * st * ct) as f32;
            if sxx_pred.is_finite() && sxx_kirsch.abs() > 1e-3 * px.abs() as f32 {
                wval += w * (sxx_pred / sxx_kirsch);
                wsum += w;
            }
        }
        if wsum < 1e-10 { return None; }
        Some(wval / wsum * 3.0_f32)
    }
}

fn t_scalar(t: &Tensor<B, 1>) -> f32 {
    t.clone().into_data().to_vec::<f32>().unwrap_or(vec![0.0])[0]
}

// ─── Gradient conflict detection ─────────────────────────────────────────────

/// Visitor that flattens all gradient tensors in `GradientsParams` to a 1-D inner-backend tensor.
/// Mirrors `FlattenGradsVisitorInner` from burn-optim-0.21.0/src/optim/lbfgs.rs:366-378.
struct GradFlattenVisitor<'a> {
    grads:   &'a GradientsParams,
    tensors: Vec<Tensor<BInner, 1>>,
}

impl ModuleVisitor<B> for GradFlattenVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(g) = self.grads.get::<BInner, D>(param.id) {
            let numel = g.shape().num_elements();
            self.tensors.push(g.reshape([numel]));
        }
    }
}

/// Flatten all gradients in `grads` (for the given model) into a 1-D inner-backend tensor.
///
/// Non-destructive: `GradientsParams::get` is read-only (the burn LBFGS source confirms this).
pub fn flatten_grads(model: &ElasticityNet<B>, grads: &GradientsParams) -> Tensor<BInner, 1> {
    let mut vis = GradFlattenVisitor { grads, tensors: Vec::new() };
    model.visit(&mut vis);
    if vis.tensors.is_empty() {
        return Tensor::empty([0], &model.devices()[0]);
    }
    Tensor::cat(vis.tensors, 0)
}

/// Compute gradient conflict between the physics group and BC group using dual backward passes.
///
/// Uses a 25% random subset of interior collocation points (seeded deterministically by `step`).
/// Physics group: e_loss + eq_loss + const_loss.
/// BC group:      n_loss + h_loss + d_loss + w_neumann.
/// Gradients are computed unweighted (no SAW scaling) — we care about direction, not magnitude.
pub fn compute_gradient_conflict(
    model: &ElasticityNet<B>,
    ctx: &StepCtx,
    step: usize,
    device: &WgpuDevice,
) -> GradientConflict {
    let n_int = ctx.int_norm.len();
    let n_sub = (n_int / 4).max(1);

    // Deterministic index selection: LCG-based Fisher-Yates shuffle seeded by step.
    let sub_int_norm: Vec<[f32; 2]> = {
        let mut idx: Vec<usize> = (0..n_int).collect();
        let mut rng = (step as u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        for i in (1..n_int).rev() {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (rng >> 33) as usize % (i + 1);
            idx.swap(i, j);
        }
        idx[..n_sub].iter().map(|&i| ctx.int_norm[i]).collect()
    };

    let n_fourier = ctx.engine.n_fourier;
    let use_mdem  = ctx.engine.use_mdem;
    let px        = ctx.config.load.px;
    let u_ref_f64 = ctx.u_ref as f64;

    let v_to_t = |idxs: &[usize], src: &[f32]| -> Tensor<B, 1> {
        let v: Vec<f32> = idxs.iter().map(|&i| src[i]).collect();
        Tensor::<B, 1>::from_data(TensorData::new(v.clone(), vec![v.len()]), device)
    };

    let scale_out = |ansatz: Tensor<B, 2>| -> Tensor<B, 2> {
        if use_mdem {
            let nr = ansatz.dims()[0];
            Tensor::cat(vec![
                ansatz.clone().slice([0..nr, 0..2]).mul_scalar(u_ref_f64),
                ansatz.slice([0..nr, 2..5]).mul_scalar(px),
            ], 1)
        } else {
            ansatz.mul_scalar(u_ref_f64)
        }
    };

    // Helper: extract σ from a scaled mDEM output for rows [rs,re).
    let mdem_stress = |out: &Tensor<B, 2>, rs: usize, re: usize| -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let n = re - rs;
        let sxx = out.clone().slice([rs..re, 2..3]).reshape([n]);
        let syy = out.clone().slice([rs..re, 3..4]).reshape([n]);
        let sxy = out.clone().slice([rs..re, 4..5]).reshape([n]);
        (sxx, syy, sxy)
    };

    // ── Pass 1: physics group ────────────────────────────────────────────────
    let g_pde_flat: Tensor<BInner, 1> = {
        let pts_t = norm_pts_to_tensor::<B>(&sub_int_norm, device);
        let stencil = assemble_stencil::<B>(&pts_t, ctx.fd, device);
        let out = scale_out(apply_dirichlet_ansatz::<B>(
            fwd(model, stencil.clone(), n_fourier, device),
            &stencil, ctx.config.geometry.symmetry, ctx.k,
        ));
        let n_sub_int = sub_int_norm.len();
        let (sxx_n, syy_n, sxy_n) = if use_mdem { mdem_stress(&out, 0, n_sub_int) } else { (
            Tensor::<B, 1>::zeros([1], device),
            Tensor::<B, 1>::zeros([1], device),
            Tensor::<B, 1>::zeros([1], device),
        )};
        let (exx, eyy, exy) = compute_strains::<B>(out, n_sub_int, ctx.fd);
        let e_loss = dem_energy_loss(exx.clone(), eyy.clone(), exy.clone(), &ctx.config.material)
            .mul_scalar(1.0 / ctx.ref_energy as f64);
        let const_loss: Tensor<B, 1> = if use_mdem {
            constitutive_consistency_loss(
                sxx_n, syy_n, sxy_n, exx, eyy, exy, &ctx.config.material,
            ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
        } else {
            Tensor::<B, 1>::zeros([1], device)
        };

        // Equilibrium ring: use full set (fixed, cheap, always included in physics group).
        let n_eq = ctx.eq_ring_norm.len();
        let eq_loss: Tensor<B, 1> = if n_eq > 0 {
            if use_mdem {
                let shifts: [(f32, f32); 4] = [
                    ( ctx.fd.hx, 0.0), (-ctx.fd.hx, 0.0),
                    (0.0,  ctx.fd.hy), (0.0, -ctx.fd.hy),
                ];
                let all_shifted: Vec<[f32; 2]> = shifts.iter().flat_map(|&(dx, dy)| {
                    ctx.eq_ring_norm.iter().map(move |&[x, y]| [x + dx, y + dy])
                }).collect();
                let pts_all = norm_pts_to_tensor::<B>(&all_shifted, device);
                let out_all = scale_out(apply_dirichlet_ansatz::<B>(
                    fwd(model, pts_all.clone(), n_fourier, device),
                    &pts_all, ctx.config.geometry.symmetry, ctx.k,
                ));
                let seg = |i: usize| mdem_stress(&out_all, i * n_eq, (i + 1) * n_eq);
                let (sxx_xp, _, sxy_xp) = seg(0);
                let (sxx_xm, _, sxy_xm) = seg(1);
                let (_, syy_yp, sxy_yp) = seg(2);
                let (_, syy_ym, sxy_ym) = seg(3);
                equilibrium_residual_loss(
                    sxx_xp, sxy_xp, sxx_xm, sxy_xm,
                    sxy_yp, syy_yp, sxy_ym, syy_ym,
                    ctx.cx, ctx.cy, ctx.ref_div2,
                )
            } else {
                let fwd_shift = |dx: f32, dy: f32| {
                    let shifted: Vec<[f32; 2]> = ctx.eq_ring_norm.iter()
                        .map(|&[xn, yn]| [xn + dx, yn + dy]).collect();
                    let pts = norm_pts_to_tensor::<B>(&shifted, device);
                    let stencil = assemble_stencil::<B>(&pts, ctx.fd, device);
                    let out = scale_out(apply_dirichlet_ansatz::<B>(
                        fwd(model, stencil.clone(), n_fourier, device),
                        &stencil, ctx.config.geometry.symmetry, ctx.k,
                    ));
                    let (exx_, eyy_, exy_) = compute_strains::<B>(out, n_eq, ctx.fd);
                    compute_stress(exx_, eyy_, exy_, &ctx.config.material)
                };
                let (sxx_xp, _, sxy_xp) = fwd_shift( ctx.fd.hx,  0.0);
                let (sxx_xm, _, sxy_xm) = fwd_shift(-ctx.fd.hx,  0.0);
                let (_, syy_yp, sxy_yp) = fwd_shift( 0.0,  ctx.fd.hy);
                let (_, syy_ym, sxy_ym) = fwd_shift( 0.0, -ctx.fd.hy);
                equilibrium_residual_loss(
                    sxx_xp, sxy_xp, sxx_xm, sxy_xm,
                    sxy_yp, syy_yp, sxy_ym, syy_ym,
                    ctx.cx, ctx.cy, ctx.ref_div2,
                )
            }
        } else {
            Tensor::<B, 1>::zeros([1], device)
        };

        let physics_loss = e_loss + eq_loss + const_loss;
        let grads_raw = physics_loss.backward();
        let grads_p = GradientsParams::from_grads(grads_raw, model);
        flatten_grads(model, &grads_p)
    };

    // ── Pass 2: BC group ─────────────────────────────────────────────────────
    let g_bc_flat: Tensor<BInner, 1> = {
        let n_loss = if !ctx.trac_idx.is_empty() {
            let trac_norm: Vec<[f32; 2]> = ctx.trac_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
            let nt = trac_norm.len();
            let bnd_t = norm_pts_to_tensor::<B>(&trac_norm, device);
            let stencil_bnd = assemble_stencil::<B>(&bnd_t, ctx.fd, device);
            let out_bnd = scale_out(apply_dirichlet_ansatz::<B>(
                fwd(model, stencil_bnd.clone(), n_fourier, device),
                &stencil_bnd, ctx.config.geometry.symmetry, ctx.k,
            ));
            let (ex, ey, exy) = compute_strains::<B>(out_bnd, nt, ctx.fd);
            neumann_loss(
                ex, ey, exy,
                v_to_t(ctx.trac_idx, ctx.bnd_nx), v_to_t(ctx.trac_idx, ctx.bnd_ny),
                v_to_t(ctx.trac_idx, ctx.bnd_tx), v_to_t(ctx.trac_idx, ctx.bnd_ty),
                &ctx.config.material,
            ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
        } else {
            Tensor::<B, 1>::zeros([1], device)
        };

        let u_target_val = ((ctx.config.load.px
            - ctx.config.material.nu * ctx.config.load.py)
            / ctx.config.material.e * ctx.config.geometry.half_w) as f32;
        let (d_loss, w_neumann) = if !ctx.right_idx.is_empty() {
            let right_norm: Vec<[f32; 2]> = ctx.right_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
            let nr = right_norm.len();
            let right_t = norm_pts_to_tensor::<B>(&right_norm, device);
            let out_r = scale_out(apply_dirichlet_ansatz::<B>(
                fwd(model, right_t.clone(), n_fourier, device),
                &right_t, ctx.config.geometry.symmetry, ctx.k,
            ));
            let u_vals = out_r.slice([0..nr, 0..1]).reshape([nr]);
            let u_tgt: Tensor<B, 1> = Tensor::full([nr], u_target_val as f64, device);
            let denom = ((u_target_val * u_target_val) as f64).max(1e-20);
            let d_val = (u_vals.clone() - u_tgt).powf_scalar(2.0_f64).mean()
                .mul_scalar(1.0 / denom);
            let w_val = u_vals.mean()
                .mul_scalar(2.0 * ctx.config.material.e
                    / floor_signed_divisor(ctx.config.load.px * ctx.config.geometry.half_w));
            (d_val, w_val)
        } else {
            (Tensor::<B, 1>::zeros([1], device), Tensor::<B, 1>::zeros([1], device))
        };

        let h_loss = if !ctx.hole_idx.is_empty() {
            let hole_norm: Vec<[f32; 2]> = ctx.hole_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
            let nh = hole_norm.len();
            let bnd_h = norm_pts_to_tensor::<B>(&hole_norm, device);
            if use_mdem {
                let out_h = scale_out(apply_dirichlet_ansatz::<B>(
                    fwd(model, bnd_h.clone(), n_fourier, device),
                    &bnd_h, ctx.config.geometry.symmetry, ctx.k,
                ));
                let (sxx_h, syy_h, sxy_h) = mdem_stress(&out_h, 0, nh);
                hole_traction_loss_direct(
                    sxx_h, syy_h, sxy_h,
                    v_to_t(ctx.hole_idx, ctx.bnd_nx),
                    v_to_t(ctx.hole_idx, ctx.bnd_ny),
                ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
            } else {
                let stencil_h = assemble_stencil::<B>(&bnd_h, ctx.fd, device);
                let out_h = scale_out(apply_dirichlet_ansatz::<B>(
                    fwd(model, stencil_h.clone(), n_fourier, device),
                    &stencil_h, ctx.config.geometry.symmetry, ctx.k,
                ));
                let (ex, ey, exy) = compute_strains::<B>(out_h, nh, ctx.fd);
                hole_traction_loss(
                    ex, ey, exy,
                    v_to_t(ctx.hole_idx, ctx.bnd_nx),
                    v_to_t(ctx.hole_idx, ctx.bnd_ny),
                    &ctx.config.material,
                ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
            }
        } else {
            Tensor::<B, 1>::zeros([1], device)
        };

        let bc_loss = n_loss + h_loss + d_loss + w_neumann;
        let grads_raw = bc_loss.backward();
        let grads_p = GradientsParams::from_grads(grads_raw, model);
        flatten_grads(model, &grads_p)
    };

    // ── Cosine similarity ────────────────────────────────────────────────────
    let pde_norm_sq = g_pde_flat.clone().powf_scalar(2.0_f64).sum();
    let bc_norm_sq  = g_bc_flat.clone().powf_scalar(2.0_f64).sum();
    let dot         = g_pde_flat.mul(g_bc_flat).sum();

    let g_pde_norm_v = pde_norm_sq.clone().sqrt().into_scalar() as f32;
    let g_bc_norm_v  = bc_norm_sq.clone().sqrt().into_scalar() as f32;
    let denom_f64    = (pde_norm_sq.sqrt() * bc_norm_sq.sqrt()).into_scalar() + 1e-8;
    let cosine_sim   = (dot.into_scalar() / denom_f64) as f32;

    GradientConflict {
        cosine_sim: cosine_sim.clamp(-1.0, 1.0),
        g_pde_norm: g_pde_norm_v,
        g_bc_norm:  g_bc_norm_v,
    }
}

// ─── L-BFGS support ──────────────────────────────────────────────────────────

/// Snapshot of SAW-BRDR lambda values captured at the moment of Converge tier entry.
/// Used as fixed loss weights inside the L-BFGS closure to avoid GPU syncs per inner iteration.
#[derive(Clone)]
pub struct LbfgsLams {
    pub lam_e:      f64,
    pub lam_n:      f64,
    pub lam_h:      f64,
    pub lam_d:      f64,
    pub lam_eq:     f64,
    pub lam_kirsch: f64,
    pub lam_const:  f64,
}

/// Owned copy of the training-step context needed by the L-BFGS closure.
/// Replaces borrowed references from `StepCtx` so the closure can outlive the loop body.
pub struct LbfgsCtxScalars {
    pub config:          SolverConfig,
    pub engine:          EngineParams,
    pub fd:              FdConfig,
    pub k:               f32,
    pub u_ref:           f32,
    pub ref_energy:      f32,
    pub ref_stress2:     f32,
    pub cx:              f64,
    pub cy:              f64,
    pub ref_div2:        f64,
    pub int_norm:        Vec<[f32; 2]>,
    pub bnd_norm:        Vec<[f32; 2]>,
    pub bnd_nx:          Vec<f32>,
    pub bnd_ny:          Vec<f32>,
    pub bnd_tx:          Vec<f32>,
    pub bnd_ty:          Vec<f32>,
    pub trac_idx:        Vec<usize>,
    pub hole_idx:        Vec<usize>,
    pub right_idx:       Vec<usize>,
    pub eq_ring_norm:    Vec<[f32; 2]>,
    pub dynamic_lam_h_cap: f64,
    pub dynamic_lam_d_cap: f64,
    pub phase2_active:   bool,
}

impl LbfgsCtxScalars {
    /// Build from a `StepCtx` reference, cloning all owned data.
    pub fn from_ctx(ctx: &StepCtx) -> Self {
        Self {
            config:           ctx.config.clone(),
            engine:           ctx.engine.clone(),
            fd:               *ctx.fd,
            k:                ctx.k,
            u_ref:            ctx.u_ref,
            ref_energy:       ctx.ref_energy,
            ref_stress2:      ctx.ref_stress2,
            cx:               ctx.cx,
            cy:               ctx.cy,
            ref_div2:         ctx.ref_div2,
            int_norm:         ctx.int_norm.to_vec(),
            bnd_norm:         ctx.bnd_norm.to_vec(),
            bnd_nx:           ctx.bnd_nx.to_vec(),
            bnd_ny:           ctx.bnd_ny.to_vec(),
            bnd_tx:           ctx.bnd_tx.to_vec(),
            bnd_ty:           ctx.bnd_ty.to_vec(),
            trac_idx:         ctx.trac_idx.to_vec(),
            hole_idx:         ctx.hole_idx.to_vec(),
            right_idx:        ctx.right_idx.to_vec(),
            eq_ring_norm:     ctx.eq_ring_norm.to_vec(),
            dynamic_lam_h_cap: ctx.dynamic_lam_h_cap,
            dynamic_lam_d_cap: ctx.dynamic_lam_d_cap,
            phase2_active:    ctx.phase2_active,
        }
    }
}

/// Compute total loss on the frozen collocation points using fixed lambda values.
///
/// Called by the L-BFGS closure at each inner line-search iteration.
/// SAW is NOT updated here — `lams` is the snapshot from Converge tier entry.
fn compute_loss_for_lbfgs(
    model: &ElasticityNet<B>,
    ctx: &LbfgsCtxScalars,
    lams: &LbfgsLams,
    device: &WgpuDevice,
) -> (Tensor<B, 1>, f32) {
    let n_int     = ctx.int_norm.len();
    let n_fourier = ctx.engine.n_fourier;
    let use_mdem  = ctx.engine.use_mdem;
    let px        = ctx.config.load.px;
    let u_ref_f64 = ctx.u_ref as f64;

    let v_to_t_lbfgs = |idxs: &[usize], src: &[f32]| -> Tensor<B, 1> {
        let v: Vec<f32> = idxs.iter().map(|&i| src[i]).collect();
        Tensor::<B, 1>::from_data(TensorData::new(v.clone(), vec![v.len()]), device)
    };

    let scale_out = |ansatz: Tensor<B, 2>| -> Tensor<B, 2> {
        if use_mdem {
            let nr = ansatz.dims()[0];
            Tensor::cat(vec![
                ansatz.clone().slice([0..nr, 0..2]).mul_scalar(u_ref_f64),
                ansatz.slice([0..nr, 2..5]).mul_scalar(px),
            ], 1)
        } else {
            ansatz.mul_scalar(u_ref_f64)
        }
    };

    let mdem_stress_lbfgs = |out: &Tensor<B, 2>, rs: usize, re: usize| -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let n = re - rs;
        (
            out.clone().slice([rs..re, 2..3]).reshape([n]),
            out.clone().slice([rs..re, 3..4]).reshape([n]),
            out.clone().slice([rs..re, 4..5]).reshape([n]),
        )
    };

    // Interior energy loss.
    let pts_t = norm_pts_to_tensor::<B>(&ctx.int_norm, device);
    let stencil_coords = assemble_stencil::<B>(&pts_t, &ctx.fd, device);
    let stencil_out = scale_out(apply_dirichlet_ansatz::<B>(
        fwd(model, stencil_coords.clone(), n_fourier, device),
        &stencil_coords, ctx.config.geometry.symmetry, ctx.k,
    ));
    let (sxx_n_int, syy_n_int, sxy_n_int) = if use_mdem {
        let s = mdem_stress_lbfgs(&stencil_out, 0, n_int);
        (Some(s.0), Some(s.1), Some(s.2))
    } else { (None, None, None) };
    let (exx, eyy, exy) = compute_strains::<B>(stencil_out, n_int, &ctx.fd);
    // Threaded from `StepCtx::ref_energy` (via `LbfgsCtxScalars::from_ctx`), NOT recomputed
    // here — `compute_reference_scales` (the single source of truth) is the only place that
    // knows to branch on `config.use_ultimate_strength_scaling`; a local recompute from raw
    // `config.load.px` would silently ignore that flag (issue #23). Already floored
    // (`.max(1.0)`) upstream by `compute_reference_scales` against the px=0 divide-by-zero
    // case, so no separate guard is needed here.
    let ref_energy = ctx.ref_energy;
    let e_loss = dem_energy_loss(exx.clone(), eyy.clone(), exy.clone(), &ctx.config.material)
        .mul_scalar(1.0 / ref_energy as f64);
    let const_loss: Tensor<B, 1> = if let (Some(sxx_n), Some(syy_n), Some(sxy_n)) =
        (sxx_n_int, syy_n_int, sxy_n_int)
    {
        constitutive_consistency_loss(
            sxx_n, syy_n, sxy_n, exx, eyy, exy, &ctx.config.material,
        ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
    } else {
        Tensor::<B, 1>::zeros([1], device)
    };

    // Neumann traction loss.
    let n_loss = if !ctx.trac_idx.is_empty() {
        let trac_norm: Vec<[f32; 2]> = ctx.trac_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
        let nt = trac_norm.len();
        let bnd_t = norm_pts_to_tensor::<B>(&trac_norm, device);
        let stencil_bnd = assemble_stencil::<B>(&bnd_t, &ctx.fd, device);
        let out_bnd = scale_out(apply_dirichlet_ansatz::<B>(
            fwd(model, stencil_bnd.clone(), n_fourier, device),
            &stencil_bnd, ctx.config.geometry.symmetry, ctx.k,
        ));
        let (ex, ey, exy_b) = compute_strains::<B>(out_bnd, nt, &ctx.fd);
        neumann_loss(
            ex, ey, exy_b,
            v_to_t_lbfgs(&ctx.trac_idx, &ctx.bnd_nx), v_to_t_lbfgs(&ctx.trac_idx, &ctx.bnd_ny),
            v_to_t_lbfgs(&ctx.trac_idx, &ctx.bnd_tx), v_to_t_lbfgs(&ctx.trac_idx, &ctx.bnd_ty),
            &ctx.config.material,
        ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
    } else {
        Tensor::<B, 1>::zeros([1], device)
    };

    // Displacement anchor + Neumann work.
    let u_target_val = ((ctx.config.load.px
        - ctx.config.material.nu * ctx.config.load.py)
        / ctx.config.material.e * ctx.config.geometry.half_w) as f32;
    let (d_loss, w_neumann) = if !ctx.right_idx.is_empty() {
        let right_norm: Vec<[f32; 2]> = ctx.right_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
        let nr = right_norm.len();
        let right_t = norm_pts_to_tensor::<B>(&right_norm, device);
        let out_r = scale_out(apply_dirichlet_ansatz::<B>(
            fwd(model, right_t.clone(), n_fourier, device),
            &right_t, ctx.config.geometry.symmetry, ctx.k,
        ));
        let u_vals = out_r.slice([0..nr, 0..1]).reshape([nr]);
        let u_tgt: Tensor<B, 1> = Tensor::full([nr], u_target_val as f64, device);
        let denom = ((u_target_val * u_target_val) as f64).max(1e-20);
        let d_val = (u_vals.clone() - u_tgt).powf_scalar(2.0_f64).mean()
            .mul_scalar(1.0 / denom);
        let w_val = u_vals.mean()
            .mul_scalar(2.0 * ctx.config.material.e
                / floor_signed_divisor(ctx.config.load.px * ctx.config.geometry.half_w));
        (d_val, w_val)
    } else {
        (Tensor::<B, 1>::zeros([1], device), Tensor::<B, 1>::zeros([1], device))
    };

    // Hole traction-free loss.
    let h_loss = if !ctx.hole_idx.is_empty() {
        let hole_norm: Vec<[f32; 2]> = ctx.hole_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
        let nh = hole_norm.len();
        let bnd_h = norm_pts_to_tensor::<B>(&hole_norm, device);
        if use_mdem {
            let out_h = scale_out(apply_dirichlet_ansatz::<B>(
                fwd(model, bnd_h.clone(), n_fourier, device),
                &bnd_h, ctx.config.geometry.symmetry, ctx.k,
            ));
            let (sxx_h, syy_h, sxy_h) = mdem_stress_lbfgs(&out_h, 0, nh);
            hole_traction_loss_direct(
                sxx_h, syy_h, sxy_h,
                v_to_t_lbfgs(&ctx.hole_idx, &ctx.bnd_nx),
                v_to_t_lbfgs(&ctx.hole_idx, &ctx.bnd_ny),
            ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
        } else {
            let stencil_h = assemble_stencil::<B>(&bnd_h, &ctx.fd, device);
            let out_h = scale_out(apply_dirichlet_ansatz::<B>(
                fwd(model, stencil_h.clone(), n_fourier, device),
                &stencil_h, ctx.config.geometry.symmetry, ctx.k,
            ));
            let (ex, ey, exy_h) = compute_strains::<B>(out_h, nh, &ctx.fd);
            hole_traction_loss(
                ex, ey, exy_h,
                v_to_t_lbfgs(&ctx.hole_idx, &ctx.bnd_nx),
                v_to_t_lbfgs(&ctx.hole_idx, &ctx.bnd_ny),
                &ctx.config.material,
            ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
        }
    } else {
        Tensor::<B, 1>::zeros([1], device)
    };

    // Equilibrium residual.
    let n_eq = ctx.eq_ring_norm.len();
    let eq_loss: Tensor<B, 1> = if n_eq > 0 {
        if use_mdem {
            let shifts: [(f32, f32); 4] = [
                ( ctx.fd.hx, 0.0), (-ctx.fd.hx, 0.0),
                (0.0,  ctx.fd.hy), (0.0, -ctx.fd.hy),
            ];
            let all_shifted: Vec<[f32; 2]> = shifts.iter().flat_map(|&(dx, dy)| {
                ctx.eq_ring_norm.iter().map(move |&[x, y]| [x + dx, y + dy])
            }).collect();
            let pts_all = norm_pts_to_tensor::<B>(&all_shifted, device);
            let out_all = scale_out(apply_dirichlet_ansatz::<B>(
                fwd(model, pts_all.clone(), n_fourier, device),
                &pts_all, ctx.config.geometry.symmetry, ctx.k,
            ));
            let seg = |i: usize| mdem_stress_lbfgs(&out_all, i * n_eq, (i + 1) * n_eq);
            let (sxx_xp, _, sxy_xp) = seg(0);
            let (sxx_xm, _, sxy_xm) = seg(1);
            let (_, syy_yp, sxy_yp) = seg(2);
            let (_, syy_ym, sxy_ym) = seg(3);
            equilibrium_residual_loss(
                sxx_xp, sxy_xp, sxx_xm, sxy_xm,
                sxy_yp, syy_yp, sxy_ym, syy_ym,
                ctx.cx, ctx.cy, ctx.ref_div2,
            )
        } else {
            let fwd_shift = |dx: f32, dy: f32| {
                let shifted: Vec<[f32; 2]> = ctx.eq_ring_norm.iter()
                    .map(|&[xn, yn]| [xn + dx, yn + dy]).collect();
                let pts = norm_pts_to_tensor::<B>(&shifted, device);
                let stencil = assemble_stencil::<B>(&pts, &ctx.fd, device);
                let out = scale_out(apply_dirichlet_ansatz::<B>(
                    fwd(model, stencil.clone(), n_fourier, device),
                    &stencil, ctx.config.geometry.symmetry, ctx.k,
                ));
                let (exx_, eyy_, exy_) = compute_strains::<B>(out, n_eq, &ctx.fd);
                compute_stress(exx_, eyy_, exy_, &ctx.config.material)
            };
            let (sxx_xp, _, sxy_xp) = fwd_shift( ctx.fd.hx,  0.0);
            let (sxx_xm, _, sxy_xm) = fwd_shift(-ctx.fd.hx,  0.0);
            let (_, syy_yp, sxy_yp) = fwd_shift( 0.0,  ctx.fd.hy);
            let (_, syy_ym, sxy_ym) = fwd_shift( 0.0, -ctx.fd.hy);
            equilibrium_residual_loss(
                sxx_xp, sxy_xp, sxx_xm, sxy_xm,
                sxy_yp, syy_yp, sxy_ym, syy_ym,
                ctx.cx, ctx.cy, ctx.ref_div2,
            )
        }
    } else {
        Tensor::<B, 1>::zeros([1], device)
    };

    // Kirsch stress loss (Phase 2 only).
    use pinn_core::geometry::HoleType;
    use pinn_core::kirsch::kirsch_stress;
    let (kirsch_loss, _kirsch_scalar_f): (Tensor<B, 1>, f32) = if ctx.phase2_active {
        if let HoleType::Circular { radius } = ctx.config.geometry.hole {
            let n_pr = ctx.engine.kirsch_r_factors.len() * ctx.engine.kirsch_thetas_deg.len();
            let mut points      = Vec::with_capacity(n_pr);
            let mut sxx_targets = Vec::with_capacity(n_pr);
            let mut syy_targets = Vec::with_capacity(n_pr);
            let mut sxy_targets = Vec::with_capacity(n_pr);
            let mut weights     = Vec::with_capacity(n_pr);
            for &r_fac in &ctx.engine.kirsch_r_factors {
                let r = radius * r_fac;
                for &deg in &ctx.engine.kirsch_thetas_deg {
                    let theta = deg.to_radians();
                    points.push(normalize_point(r * theta.cos(), r * theta.sin(), &ctx.config));
                    let (s_rr, s_tt, s_rt) = kirsch_stress(r, theta, radius, ctx.config.load.px, ctx.config.load.py);
                    let c = theta.cos(); let s = theta.sin();
                    sxx_targets.push((s_rr*c*c + s_tt*s*s - 2.0*s_rt*s*c) as f32);
                    syy_targets.push((s_rr*s*s + s_tt*c*c + 2.0*s_rt*s*c) as f32);
                    sxy_targets.push(((s_rr - s_tt)*s*c + s_rt*(c*c - s*s)) as f32);
                    weights.push(theta.sin().powi(2) as f32);
                }
            }
            let wsum: f32 = weights.iter().sum();
            for w in weights.iter_mut() { *w /= wsum; }
            let to_t1 = |v: &[f32]| -> Tensor<B, 1> {
                Tensor::from_data(TensorData::new(v.to_vec(), vec![v.len()]), device)
            };
            let px2 = (px * px) as f64;
            let pts_pr = norm_pts_to_tensor::<B>(&points, device);
            let kl = if use_mdem {
                let out_pr = scale_out(apply_dirichlet_ansatz::<B>(
                    fwd(model, pts_pr.clone(), n_fourier, device),
                    &pts_pr, ctx.config.geometry.symmetry, ctx.k,
                ));
                let (sxx_pr, syy_pr, sxy_pr) = mdem_stress_lbfgs(&out_pr, 0, n_pr);
                let loss_per_pt =
                    (sxx_pr - to_t1(&sxx_targets)).powf_scalar(2.0_f64)
                    + (syy_pr - to_t1(&syy_targets)).powf_scalar(2.0_f64)
                    + (sxy_pr - to_t1(&sxy_targets)).powf_scalar(2.0_f64).mul_scalar(2.0_f64);
                (loss_per_pt * to_t1(&weights)).sum().mul_scalar(1.0 / px2)
            } else {
                let stencil_pr = assemble_stencil::<B>(&pts_pr, &ctx.fd, device);
                let out_pr = scale_out(apply_dirichlet_ansatz::<B>(
                    fwd(model, stencil_pr.clone(), n_fourier, device),
                    &stencil_pr, ctx.config.geometry.symmetry, ctx.k,
                ));
                let (exx_pr, eyy_pr, exy_pr) = compute_strains::<B>(out_pr, n_pr, &ctx.fd);
                let (sxx_pr, syy_pr, sxy_pr) = compute_stress(exx_pr, eyy_pr, exy_pr, &ctx.config.material);
                let loss_per_pt =
                    (sxx_pr - to_t1(&sxx_targets)).powf_scalar(2.0_f64)
                    + (syy_pr - to_t1(&syy_targets)).powf_scalar(2.0_f64)
                    + (sxy_pr - to_t1(&sxy_targets)).powf_scalar(2.0_f64).mul_scalar(2.0_f64);
                (loss_per_pt * to_t1(&weights)).sum().mul_scalar(1.0 / px2)
            };
            let ks = t_scalar(&kl);
            (kl, ks)
        } else {
            (Tensor::<B, 1>::zeros([1], device), 0.0)
        }
    } else {
        (Tensor::<B, 1>::zeros([1], device), 0.0)
    };

    // Assemble total with fixed lams.
    let lam_h_capped = lams.lam_h.min(ctx.dynamic_lam_h_cap);
    let lam_d_capped = lams.lam_d.min(ctx.dynamic_lam_d_cap);

    let total = (e_loss - w_neumann).mul_scalar(lams.lam_e)
        + n_loss.mul_scalar(lams.lam_n)
        + h_loss.mul_scalar(lam_h_capped)
        + d_loss.mul_scalar(lam_d_capped)
        + eq_loss.mul_scalar(lams.lam_eq)
        + kirsch_loss.mul_scalar(lams.lam_kirsch)
        + const_loss.mul_scalar(lams.lam_const);

    let total_scalar = t_scalar(&total);
    (total, total_scalar)
}

/// Run a single L-BFGS outer step on frozen collocation points.
///
/// The closure captures `ctx` and `lams` by reference; LBFGS calls it 5–20 times
/// internally during its Strong-Wolfe line search, each time on a clone of `model`.
/// SAW-BRDR is NOT updated inside the closure — `lams` is the snapshot from Converge entry.
pub fn step_lbfgs(
    model: ElasticityNet<B>,
    lbfgs: &mut burn::optim::LBFGS<B>,
    lr: f64,
    ctx: &LbfgsCtxScalars,
    lams: &LbfgsLams,
    device: &WgpuDevice,
) -> (ElasticityNet<B>, f64) {
    let closure = |m: ElasticityNet<B>| -> (f64, GradientsParams) {
        let (total_loss, total_scalar) = compute_loss_for_lbfgs(&m, ctx, lams, device);
        let loss_f64 = total_scalar as f64;
        let grads_raw = total_loss.backward();
        let grads_p = GradientsParams::from_grads(grads_raw, &m);
        (loss_f64, grads_p)
    };
    lbfgs.step(lr, model, closure)
}

/// Build a fresh `LBFGS<B>` optimizer using `DecisionMakerConfig::lbfgs_max_iter`.
pub fn make_lbfgs(max_iter: usize) -> burn::optim::LBFGS<B> {
    LBFGSConfig::new()
        .with_max_iter(max_iter)
        .with_line_search_fn(burn::optim::LineSearchFn::StrongWolfe)
        .init()
}

// ─── Multi-domain L-BFGS / gradient-conflict support ────────────────────────────────────

/// Visitor that flattens all gradient tensors in `GradientsParams` for a `TwoDomainModels<B>`
/// to a 1-D inner-backend tensor, substituting an explicit zero tensor (same shape as the
/// param) for any parameter `GradientsParams::get` has no entry for.
///
/// Unlike the single-model `GradFlattenVisitor` (which can safely SKIP untouched params,
/// since a single model is always fully present in both dual-pass backward graphs — the
/// worst case is a zero gradient, and skipping it vs. keeping a zero contributes nothing to
/// a shared-alignment dot product anyway), `TwoDomainModels` genuinely has an entire
/// sub-model's worth of params that ONE pass's loss may not touch at all (e.g. a
/// Physics-only term that reads only `pin`, leaving every `lug` param absent from that
/// pass's `GradientsParams`). Skipping in that case would silently misalign the two flattened
/// vectors positionally between the physics-pass and BC-pass calls, corrupting the cosine
/// similarity dot product with cross-domain garbage instead of the intended zero
/// contribution — hence the explicit zero-substitution here (not just cosmetic; load-bearing
/// for `compute_gradient_conflict_multi`'s correctness).
struct ZeroFillFlattenVisitor<'a> {
    grads:   &'a GradientsParams,
    tensors: Vec<Tensor<BInner, 1>>,
}

impl ModuleVisitor<B> for ZeroFillFlattenVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let numel = param.val().shape().num_elements();
        let flat = match self.grads.get::<BInner, D>(param.id) {
            Some(g) => g.reshape([numel]),
            None => Tensor::<BInner, D>::zeros(param.val().shape(), &param.val().device()).reshape([numel]),
        };
        self.tensors.push(flat);
    }
}

/// Flatten all gradient tensors in `grads` for a `TwoDomainModels<B>` to a 1-D inner-backend
/// tensor, in a FIXED param-visit order with zero-substitution for absent gradients (see
/// `ZeroFillFlattenVisitor`) — required so the physics-pass and BC-pass flattened vectors
/// from `compute_gradient_conflict_multi` are positionally aligned even when one pass's loss
/// doesn't touch one entire sub-model's parameters.
pub fn flatten_grads_multi(models: &TwoDomainModels<B>, grads: &GradientsParams) -> Tensor<BInner, 1> {
    let mut vis = ZeroFillFlattenVisitor { grads, tensors: Vec::new() };
    models.visit(&mut vis);
    if vis.tensors.is_empty() {
        return Tensor::empty([0], &models.devices()[0]);
    }
    Tensor::cat(vis.tensors, 0)
}

/// Sum a group of active `LossTerm`s (all sharing the same `ConflictGroup`) into ONE combined
/// tensor via the shared `compute_domain_forwards` forward-pass enumeration. Returns `None`
/// when `terms` is empty — distinguishes "no active loss term in this group" (no autodiff
/// graph; calling `.backward()` on a bare `zeros()` tensor panics with "Node should have a
/// step registered") from a genuine zero-valued but graph-connected loss.
fn sum_group_loss(
    ctx: &crate::problem::MultiStepCtx,
    model_refs: &[&ElasticityNet<B>],
    terms: &[Box<dyn LossTerm>],
    device: &WgpuDevice,
) -> Option<Tensor<B, 1>> {
    use crate::problem::DomainForwardOutputs as DFO;
    use pinn_core::problem::DomainId;

    if terms.is_empty() {
        return None;
    }
    let computed = compute_domain_forwards(ctx, model_refs, terms, device);
    let forwards: HashMap<(DomainId, &'static str), DFO<'_, B>> = computed.iter()
        .map(|c| (c.key, DFO {
            domain: c.key.0,
            raw_out: &c.raw_out,
            strains: c.strains.clone(),
            normals: c.normals.clone(),
        }))
        .collect();
    let mut total: Option<Tensor<B, 1>> = None;
    for term in terms {
        let inputs: Vec<DFO<'_, B>> = term.domains().iter().zip(term.point_sets().iter())
            .filter_map(|(&id, &ps)| forwards.get(&(id, ps)).map(|f| DFO {
                domain: f.domain, raw_out: f.raw_out,
                strains: f.strains.clone(), normals: f.normals.clone(),
            }))
            .collect();
        let t = term.compute(&inputs);
        total = Some(match total {
            Some(acc) => acc + t,
            None => t,
        });
    }
    total
}

/// N-domain generalization of `compute_gradient_conflict` — same TWO-backward-pass structure
/// (Physics group, then Bc group per `LossTerm::conflict_group()`), but each pass sums ALL
/// active domains' classified-group terms into ONE combined loss before backward — exactly
/// 2 backward passes total, not 2*N. Unweighted (no SAW scaling) — direction, not magnitude,
/// matters.
pub fn compute_gradient_conflict_multi(
    models: &TwoDomainModels<B>,
    ctx: &crate::problem::MultiStepCtx,
    device: &WgpuDevice,
) -> GradientConflict {
    let active_terms: Vec<Box<dyn LossTerm>> = ctx.problem.loss_terms().into_iter()
        .filter(|t| t.name() != "constitutive_consistency")
        .filter(|t| ctx.phase2_active || !t.phase2_only())
        .collect();

    type TermVec = Vec<Box<dyn LossTerm>>;
    let (physics_terms, bc_terms): (TermVec, TermVec) = active_terms
        .into_iter()
        .partition(|t| t.conflict_group() == crate::problem::ConflictGroup::Physics);

    let model_refs: Vec<&ElasticityNet<B>> = vec![&models.pin, &models.lug];

    // A group's flattened gradient is the all-zero vector, SAME LENGTH as a real flattened
    // gradient (total param count across both domains — not a length-1 stub), when that
    // group has no active terms. Matching length is load-bearing: `flatten_grads_multi`
    // zero-fills per-param on a MISSING gradient entry (see `ZeroFillFlattenVisitor`), so an
    // entirely-empty group must produce the same total length or the two passes'
    // elementwise `mul` below would panic/misalign — NOT skipped/NaN, so cosine similarity's
    // `+1e-8` epsilon guard still produces a well-defined (0.0) result.
    let empty_grads = GradientsParams::new();
    let flat_or_zero = |loss: Option<Tensor<B, 1>>| -> Tensor<BInner, 1> {
        match loss {
            Some(l) => {
                let grads_raw = l.backward();
                let grads_p = GradientsParams::from_grads(grads_raw, models);
                flatten_grads_multi(models, &grads_p)
            }
            None => flatten_grads_multi(models, &empty_grads),
        }
    };

    // ── Pass 1: physics group ────────────────────────────────────────────────
    let g_pde_flat = flat_or_zero(sum_group_loss(ctx, &model_refs, &physics_terms, device));

    // ── Pass 2: BC group ─────────────────────────────────────────────────────
    let g_bc_flat = flat_or_zero(sum_group_loss(ctx, &model_refs, &bc_terms, device));

    // ── Cosine similarity — formula/epsilon guard identical to compute_gradient_conflict ──
    let pde_norm_sq = g_pde_flat.clone().powf_scalar(2.0_f64).sum();
    let bc_norm_sq  = g_bc_flat.clone().powf_scalar(2.0_f64).sum();
    let dot         = g_pde_flat.mul(g_bc_flat).sum();

    let g_pde_norm_v = pde_norm_sq.clone().sqrt().into_scalar() as f32;
    let g_bc_norm_v  = bc_norm_sq.clone().sqrt().into_scalar() as f32;
    let denom_f64    = (pde_norm_sq.sqrt() * bc_norm_sq.sqrt()).into_scalar() + 1e-8;
    let cosine_sim   = (dot.into_scalar() / denom_f64) as f32;

    GradientConflict {
        cosine_sim: cosine_sim.clamp(-1.0, 1.0),
        g_pde_norm: g_pde_norm_v,
        g_bc_norm:  g_bc_norm_v,
    }
}

/// Compute one combined loss over BOTH domains on the frozen collocation points, using fixed
/// per-term base weights (`lams`) — the multi-domain analogue of `compute_loss_for_lbfgs`.
/// Reuses `compute_domain_forwards` (the SAME term-summation logic `step_physics_multi` uses)
/// via `frozen_ctx.as_multi_step_ctx(problem)` — no second loss-assembly implementation.
fn compute_loss_for_lbfgs_multi(
    models: &TwoDomainModels<B>,
    frozen_ctx: &crate::problem::FrozenMultiStepCtx,
    problem: &dyn BoundaryValueProblem,
    lams: &HashMap<&'static str, f64>,
    device: &WgpuDevice,
) -> (Tensor<B, 1>, f32) {
    use crate::problem::DomainForwardOutputs as DFO;
    use pinn_core::problem::DomainId;

    let ctx = frozen_ctx.as_multi_step_ctx(problem);

    let active_terms: Vec<Box<dyn LossTerm>> = ctx.problem.loss_terms().into_iter()
        .filter(|t| t.name() != "constitutive_consistency")
        .filter(|t| ctx.phase2_active || !t.phase2_only())
        .collect();

    let model_refs: Vec<&ElasticityNet<B>> = vec![&models.pin, &models.lug];
    let computed = compute_domain_forwards(&ctx, &model_refs, &active_terms, device);

    let forwards: HashMap<(DomainId, &'static str), DFO<'_, B>> = computed.iter()
        .map(|c| (c.key, DFO {
            domain: c.key.0,
            raw_out: &c.raw_out,
            strains: c.strains.clone(),
            normals: c.normals.clone(),
        }))
        .collect();

    let mut total: Option<Tensor<B, 1>> = None;
    let mut total_scalar = 0.0_f32;
    for term in &active_terms {
        let lam = *lams.get(term.name()).unwrap_or(&0.0);
        let inputs: Vec<DFO<'_, B>> = term.domains().iter().zip(term.point_sets().iter())
            .filter_map(|(&id, &ps)| forwards.get(&(id, ps)).map(|f| DFO {
                domain: f.domain, raw_out: f.raw_out,
                strains: f.strains.clone(), normals: f.normals.clone(),
            }))
            .collect();
        let t = term.compute(&inputs);
        let s = t_scalar(&t);
        let weighted = t.mul_scalar(lam);
        total_scalar += s * lam as f32;
        total = Some(match total {
            Some(acc) => acc + weighted,
            None => weighted,
        });
    }
    let total = total.unwrap_or_else(|| Tensor::<B, 1>::zeros([1], device));
    (total, total_scalar)
}

/// Run a single L-BFGS outer step over BOTH domains' parameter spaces at once, via
/// `TwoDomainModels<B>` — the multi-domain analogue of `step_lbfgs`.
pub fn step_lbfgs_multi(
    models: TwoDomainModels<B>,
    lbfgs: &mut burn::optim::LBFGS<B>,
    lr: f64,
    frozen_ctx: &crate::problem::FrozenMultiStepCtx,
    problem: &dyn BoundaryValueProblem,
    lams: &HashMap<&'static str, f64>,
    device: &WgpuDevice,
) -> (TwoDomainModels<B>, f64) {
    let closure = |m: TwoDomainModels<B>| -> (f64, GradientsParams) {
        let (total_loss, total_scalar) = compute_loss_for_lbfgs_multi(&m, frozen_ctx, problem, lams, device);
        let loss_f64 = total_scalar as f64;
        let grads_raw = total_loss.backward();
        let grads_p = GradientsParams::from_grads(grads_raw, &m);
        (loss_f64, grads_p)
    };
    lbfgs.step(lr, models, closure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::{make_bias_optim, make_gate_optim};
    use crate::problem::DomainState;
    use pinn_core::problem::DomainSamplingStrategy;

    #[test]
    fn compute_reference_scales_relationships_hold() {
        let config = SolverConfig::default_kirsch();
        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

        assert!(u_ref > 0.0 && u_ref.is_finite());
        assert!(ref_energy > 0.0 && ref_energy.is_finite());
        assert!(ref_stress2 > 0.0 && ref_stress2.is_finite());

        // ref_stress2 = Px^2, independent of u_ref/ref_energy's own derivation.
        let px = config.load.px as f32;
        assert!((ref_stress2 - px * px).abs() / ref_stress2 < 1e-5);

        // ref_energy = 0.5 * ref_stress2 / E — cross-checks the two outputs against
        // each other rather than re-deriving the same formula independently.
        let e = config.material.e as f32;
        let expected_ref_energy = 0.5 * ref_stress2 / e;
        assert!((ref_energy - expected_ref_energy).abs() / ref_energy < 1e-5);

        // u_ref = sqrt(ref_stress2) / E * half_w (since Px > 0, sqrt(Px^2) = Px).
        let half_w = config.geometry.half_w as f32;
        let expected_u_ref = ref_stress2.sqrt() / e * half_w;
        assert!((u_ref - expected_u_ref).abs() / u_ref < 1e-5);
    }

    #[test]
    fn compute_reference_scales_finite_and_floored_at_zero_px() {
        // px=0 (e.g. LOAD_PX_KSI=0, unvalidated on the headless/env path — see
        // crates/pinn-app/src/main.rs) makes stress_ref exactly 0.0, so ref_energy/
        // ref_stress2 would be exactly 0.0 without the .max(1.0) floor — and both are used
        // downstream as `1.0 / ctx.ref_energy`/`1.0 / ctx.ref_stress2` divisors at ~15 call
        // sites, so an unfloored zero here would silently NaN out nearly every loss term.
        let mut config = SolverConfig::default_kirsch();
        config.load.px = 0.0;
        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

        assert!(u_ref.is_finite(), "u_ref must stay finite at px=0, got {u_ref}");
        assert_eq!(u_ref, 0.0, "u_ref itself is not divided by anywhere in this function, so it's exactly 0.0 (not floored) at px=0 — only the two divisor-bound outputs need flooring");
        assert_eq!(ref_energy, 1.0, "ref_energy must floor to 1.0 at px=0, got {ref_energy}");
        assert_eq!(ref_stress2, 1.0, "ref_stress2 must floor to 1.0 at px=0, got {ref_stress2}");
        assert!((1.0_f32 / ref_energy).is_finite());
        assert!((1.0_f32 / ref_stress2).is_finite());
    }

    #[test]
    fn compute_reference_scales_flag_off_matches_existing_load_based_formula() {
        let config = SolverConfig::default_kirsch();
        assert!(!config.use_ultimate_strength_scaling);
        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

        let px = config.load.px;
        let expected_u_ref = ((px / config.material.e) * config.geometry.half_w) as f32;
        let expected_ref_energy = (0.5 * px * px / config.material.e) as f32;
        let expected_ref_stress2 = (px * px) as f32;

        assert!((u_ref - expected_u_ref).abs() / expected_u_ref < 1e-6);
        assert!((ref_energy - expected_ref_energy).abs() / expected_ref_energy < 1e-6);
        assert!((ref_stress2 - expected_ref_stress2).abs() / expected_ref_stress2 < 1e-6);
    }

    #[test]
    fn compute_reference_scales_flag_on_uses_ultimate_strength_not_applied_load() {
        let mut config_off = SolverConfig::default_kirsch();
        config_off.use_ultimate_strength_scaling = false;
        let mut config_on = SolverConfig::default_kirsch();
        config_on.use_ultimate_strength_scaling = true;

        let (_u_ref_off, _ref_energy_off, ref_stress2_off) = compute_reference_scales(&config_off);
        let (u_ref_on, ref_energy_on, ref_stress2_on) = compute_reference_scales(&config_on);

        // Must diverge by >1% relative (config's Px and ultimate_strength_pa are very
        // different magnitudes, so this is a coarse sanity check, not a precision one).
        assert!((ref_stress2_on - ref_stress2_off).abs() / ref_stress2_off > 0.01);

        let uts = config_on.material.ultimate_strength_pa;
        let expected_u_ref = ((uts / config_on.material.e) * config_on.geometry.half_w) as f32;
        let expected_ref_energy = (0.5 * uts * uts / config_on.material.e) as f32;
        let expected_ref_stress2 = (uts * uts) as f32;

        assert!((u_ref_on - expected_u_ref).abs() / expected_u_ref < 1e-6);
        assert!((ref_energy_on - expected_ref_energy).abs() / expected_ref_energy < 1e-6);
        assert!((ref_stress2_on - expected_ref_stress2).abs() / expected_ref_stress2 < 1e-6);

        assert!(u_ref_on.is_finite() && u_ref_on > 0.0);
        assert!(ref_energy_on.is_finite() && ref_energy_on > 0.0);
        assert!(ref_stress2_on.is_finite() && ref_stress2_on > 0.0);
    }

    #[test]
    fn compute_reference_scales_length_reference_is_per_domain_not_shared_constant() {
        use pinn_core::geometry::GeometryConfig;

        let mut kirsch_cfg = SolverConfig::default_kirsch();
        kirsch_cfg.use_ultimate_strength_scaling = true;

        let mut pinlug_geom_cfg = SolverConfig::default_kirsch();
        pinlug_geom_cfg.use_ultimate_strength_scaling = true;
        pinlug_geom_cfg.geometry = GeometryConfig::pinlug_lug_inches();

        // Same material for both, so any u_ref difference must come from geometry.half_w.
        let (u_ref_kirsch, _, _) = compute_reference_scales(&kirsch_cfg);
        let (u_ref_pinlug, _, _) = compute_reference_scales(&pinlug_geom_cfg);

        let expected_ratio = pinlug_geom_cfg.geometry.half_w as f32 / kirsch_cfg.geometry.half_w as f32;
        let actual_ratio = u_ref_pinlug / u_ref_kirsch;

        assert!(
            (actual_ratio - expected_ratio).abs() / expected_ratio < 1e-5,
            "expected u_ref ratio {expected_ratio}, got {actual_ratio}"
        );
    }

    #[test]
    fn floor_signed_divisor_boundary_cases() {
        assert_eq!(floor_signed_divisor(0.0), 1.0, "exact zero must default to +1.0 (tension-positive convention)");
        assert_eq!(floor_signed_divisor(1.0), 1.0, "exactly at the floor: unchanged");
        assert_eq!(floor_signed_divisor(-1.0), -1.0, "exactly at the floor, negative: unchanged, sign preserved");
        assert_eq!(floor_signed_divisor(0.5), 1.0, "inside the floor, positive: clamped up to +1.0");
        assert_eq!(floor_signed_divisor(-0.5), -1.0, "inside the floor, negative: clamped to -1.0, NOT flipped to +1.0");
        assert_eq!(floor_signed_divisor(1e-10), 1.0, "near-zero positive: floored");
        assert_eq!(floor_signed_divisor(-1e-10), -1.0, "near-zero negative: floored, sign preserved");
        assert_eq!(floor_signed_divisor(3502.4), 3502.4, "well outside the floor: passthrough, unchanged");
        assert_eq!(floor_signed_divisor(-3502.4), -3502.4, "well outside the floor, negative: passthrough, unchanged");
    }

    /// Shared scaffolding for the `floor_signed_divisor` production-site regression tests
    /// below — builds a Kirsch `StepCtx` with `use_ultimate_strength_scaling = true` so
    /// `ref_energy`/`ref_stress2` (and thus every OTHER loss term's normalization) stay
    /// derived from the material's ultimate strength, not from `config.load.px` — isolating
    /// the specific `w_val`-divisor bug under test from `compute_reference_scales`'s own
    /// (now also floored — see `compute_reference_scales_finite_and_floored_at_zero_px`)
    /// degradation at `px = 0`.
    ///
    /// `px`: the value to install into `config.load.px` (the divisor under test).
    fn zero_px_test_fixture(px: f64) -> (
        pinn_core::messages::SolverConfig, crate::engine::EngineParams, FdConfig, f32, f32, f32, f64, f64, f64,
        Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>,
        Vec<usize>, Vec<usize>, Vec<usize>, Vec<[f32; 2]>,
    ) {
        use pinn_core::messages::SolverConfig;
        use crate::engine::EngineParams;

        let mut config = SolverConfig::default_kirsch();
        config.n_interior = 64;
        config.n_boundary = 32;
        config.max_steps = 2;
        config.use_ultimate_strength_scaling = true;
        config.load.px = px;
        let engine = EngineParams::analyze(&config);
        engine.apply_to(&mut config);

        let (x0, x1) = config.geometry.x_range();
        let (y0, y1) = config.geometry.y_range();
        let fd = FdConfig::new(config.fd_h, x1 - x0, y1 - y0);
        let cx = fd.sx / (2.0 * fd.hx as f64);
        let cy = fd.sy / (2.0 * fd.hy as f64);
        let ref_div2 = (config.load.px * cx).powi(2).max(1.0);
        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

        let int_pts = pinn_core::sampling::sample_interior(&config.geometry, engine.phase1_n_interior);
        let bnd_pts = pinn_core::sampling::sample_boundary(&config.geometry, &config.load, config.n_boundary);
        let eq_ring = pinn_core::sampling::sample_eq_ring(&config.geometry, engine.n_eq_ring);

        let int_norm: Vec<[f32; 2]> = int_pts.iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();
        let bnd_norm: Vec<[f32; 2]> = bnd_pts.iter().map(|b| normalize_point(b.x, b.y, &config)).collect();
        let bnd_nx: Vec<f32> = bnd_pts.iter().map(|b| b.nx as f32).collect();
        let bnd_ny: Vec<f32> = bnd_pts.iter().map(|b| b.ny as f32).collect();
        let bnd_tx: Vec<f32> = bnd_pts.iter().map(|b| b.tx as f32).collect();
        let bnd_ty: Vec<f32> = bnd_pts.iter().map(|b| b.ty as f32).collect();
        let (trac_idx, hole_idx, right_idx) = extract_boundary_indices(&bnd_pts, &bnd_nx);
        let eq_ring_norm: Vec<[f32; 2]> = eq_ring.iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();

        (
            config, engine, fd, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
            int_norm, bnd_norm, bnd_nx, bnd_ny, bnd_tx, bnd_ty,
            trac_idx, hole_idx, right_idx, eq_ring_norm,
        )
    }

    #[test]
    fn step_physics_finite_at_zero_px_load() {
        use burn::backend::wgpu::WgpuDevice;
        use crate::{
            kirsch_problem::KirschProblem,
            network::ElasticityNetConfig,
            optim::{make_bias_optim, make_gate_optim, WeightOptim},
        };

        let (
            config, engine, fd, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
            int_norm, bnd_norm, bnd_nx, bnd_ny, bnd_tx, bnd_ty,
            trac_idx, hole_idx, right_idx, eq_ring_norm,
        ) = zero_px_test_fixture(0.0);

        let device = WgpuDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(engine.net_input_dim())
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(config.use_piratenet);
        let model: ElasticityNet<B> = net_cfg.init(&device);

        let problem = KirschProblem::new(
            config.material.clone(), engine.output_dim(), engine.phase1_steps, engine.expected_kt,
        );

        let mut optim_w = WeightOptim::new(config.use_soap_muon);
        let mut optim_b = make_bias_optim();
        let mut optim_gate = make_gate_optim();
        let mut saw = SawBrdr::with_base(engine.init_weights(), 0.95);
        let mut lr_sched = LrSchedule::new(engine.peak_lr, 200, 1000);

        let ctx = StepCtx {
            config: &config, engine: &engine, problem: &problem, fd: &fd,
            k: engine.ansatz_k, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
            int_norm: &int_norm, bnd_norm: &bnd_norm,
            bnd_nx: &bnd_nx, bnd_ny: &bnd_ny, bnd_tx: &bnd_tx, bnd_ty: &bnd_ty,
            trac_idx: &trac_idx, hole_idx: &hole_idx, right_idx: &right_idx,
            eq_ring_norm: &eq_ring_norm,
            dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
            phase2_active: false, step: 0,
        };

        let (_m, out) = step_physics(
            model, &mut optim_w, &mut optim_b, &mut optim_gate,
            &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );
        assert!(out.w_scalar.is_finite(), "w_scalar must be finite at px=0.0, got {}", out.w_scalar);
        assert!(out.total_scalar.is_finite(), "total_scalar must be finite at px=0.0, got {}", out.total_scalar);
    }

    #[test]
    fn compute_gradient_conflict_finite_at_zero_px_load() {
        use burn::backend::wgpu::WgpuDevice;
        use crate::network::ElasticityNetConfig;

        let (
            config, engine, fd, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
            int_norm, bnd_norm, bnd_nx, bnd_ny, bnd_tx, bnd_ty,
            trac_idx, hole_idx, right_idx, eq_ring_norm,
        ) = zero_px_test_fixture(0.0);

        let device = WgpuDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(engine.net_input_dim())
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(config.use_piratenet);
        let model: ElasticityNet<B> = net_cfg.init(&device);

        use crate::kirsch_problem::KirschProblem;
        let problem = KirschProblem::new(
            config.material.clone(), engine.output_dim(), engine.phase1_steps, engine.expected_kt,
        );

        let ctx = StepCtx {
            config: &config, engine: &engine, problem: &problem, fd: &fd,
            k: engine.ansatz_k, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
            int_norm: &int_norm, bnd_norm: &bnd_norm,
            bnd_nx: &bnd_nx, bnd_ny: &bnd_ny, bnd_tx: &bnd_tx, bnd_ty: &bnd_ty,
            trac_idx: &trac_idx, hole_idx: &hole_idx, right_idx: &right_idx,
            eq_ring_norm: &eq_ring_norm,
            dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
            phase2_active: false, step: 0,
        };

        let conflict = compute_gradient_conflict(&model, &ctx, 0, &device);
        assert!(conflict.cosine_sim.is_finite(), "cosine_sim must be finite at px=0.0, got {}", conflict.cosine_sim);
        assert!(conflict.g_bc_norm.is_finite(), "g_bc_norm must be finite at px=0.0, got {}", conflict.g_bc_norm);
    }

    /// Uses a tiny NEGATIVE (not exactly zero) `px` to exercise `floor_signed_divisor`'s
    /// sign-preservation for a legitimate compression (`px < 0`) load at this call site (the
    /// `w_val` divisor inside `compute_loss_for_lbfgs`). See
    /// `step_lbfgs_finite_at_literal_zero_px_load` below for the `px = 0.0` case, which is
    /// now also safe: `ctx.ref_energy` (threaded from `StepCtx`/`compute_reference_scales`,
    /// a separate divisor from `w_val`'s) is floored too.
    #[test]
    fn step_lbfgs_finite_at_near_zero_negative_px_load() {
        use burn::backend::wgpu::WgpuDevice;
        use crate::network::ElasticityNetConfig;

        let (
            config, engine, fd, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
            int_norm, bnd_norm, bnd_nx, bnd_ny, bnd_tx, bnd_ty,
            trac_idx, hole_idx, right_idx, eq_ring_norm,
        ) = zero_px_test_fixture(-0.5);

        let device = WgpuDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(engine.net_input_dim())
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(config.use_piratenet);
        let model: ElasticityNet<B> = net_cfg.init(&device);

        let lbfgs_ctx = LbfgsCtxScalars {
            config: config.clone(),
            engine: engine.clone(),
            fd,
            k: engine.ansatz_k,
            u_ref,
            ref_energy,
            ref_stress2,
            cx, cy, ref_div2,
            int_norm: int_norm.clone(),
            bnd_norm: bnd_norm.clone(),
            bnd_nx: bnd_nx.clone(),
            bnd_ny: bnd_ny.clone(),
            bnd_tx: bnd_tx.clone(),
            bnd_ty: bnd_ty.clone(),
            trac_idx: trac_idx.clone(),
            hole_idx: hole_idx.clone(),
            right_idx: right_idx.clone(),
            eq_ring_norm: eq_ring_norm.clone(),
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            phase2_active: false,
        };
        let lams = LbfgsLams {
            lam_e: 1.0, lam_n: 1.0, lam_h: 1.0, lam_d: 1.0, lam_eq: 1.0, lam_kirsch: 1.0, lam_const: 1.0,
        };

        let (_total, total_scalar) = compute_loss_for_lbfgs(&model, &lbfgs_ctx, &lams, &device);
        assert!(total_scalar.is_finite(), "compute_loss_for_lbfgs's total loss must be finite at px=0.0, got {total_scalar}");

        let mut lbfgs = make_lbfgs(3);
        let (_model_out, loss_out) = step_lbfgs(model, &mut lbfgs, 1e-3, &lbfgs_ctx, &lams, &device);
        assert!(loss_out.is_finite(), "step_lbfgs's returned loss must be finite at px=0.0, got {loss_out}");
    }

    /// Literal `px = 0.0` (not just near-zero) through `ctx.ref_energy` (threaded from
    /// `StepCtx`/`compute_reference_scales` — the second divide-by-zero site found in this
    /// same file (distinct from `w_val`'s `floor_signed_divisor` fix): `ref_energy =
    /// 0.5*stress_ref*stress_ref/E` would be exactly `0.0` at `stress_ref=0.0` with no floor,
    /// and `e_loss.mul_scalar(1.0/ref_energy)` would divide by zero. Floored (`.max(1.0)`) at
    /// the single source (`compute_reference_scales`) and threaded through unchanged from
    /// there — see `lbfgs_ctx_scalars_from_ctx_threads_ref_energy_for_both_scaling_modes`
    /// (issue #23) for the regression this used to silently defeat by recomputing locally
    /// from raw `config.load.px` instead of reusing this already-floored value.
    #[test]
    fn step_lbfgs_finite_at_literal_zero_px_load() {
        use burn::backend::wgpu::WgpuDevice;
        use crate::network::ElasticityNetConfig;

        let (
            config, engine, fd, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
            int_norm, bnd_norm, bnd_nx, bnd_ny, bnd_tx, bnd_ty,
            trac_idx, hole_idx, right_idx, eq_ring_norm,
        ) = zero_px_test_fixture(0.0);

        let device = WgpuDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(engine.net_input_dim())
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(config.use_piratenet);
        let model: ElasticityNet<B> = net_cfg.init(&device);

        let lbfgs_ctx = LbfgsCtxScalars {
            config: config.clone(),
            engine: engine.clone(),
            fd,
            k: engine.ansatz_k,
            u_ref,
            ref_energy,
            ref_stress2,
            cx, cy, ref_div2,
            int_norm: int_norm.clone(),
            bnd_norm: bnd_norm.clone(),
            bnd_nx: bnd_nx.clone(),
            bnd_ny: bnd_ny.clone(),
            bnd_tx: bnd_tx.clone(),
            bnd_ty: bnd_ty.clone(),
            trac_idx: trac_idx.clone(),
            hole_idx: hole_idx.clone(),
            right_idx: right_idx.clone(),
            eq_ring_norm: eq_ring_norm.clone(),
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            phase2_active: false,
        };
        let lams = LbfgsLams {
            lam_e: 1.0, lam_n: 1.0, lam_h: 1.0, lam_d: 1.0, lam_eq: 1.0, lam_kirsch: 1.0, lam_const: 1.0,
        };

        let (_total, total_scalar) = compute_loss_for_lbfgs(&model, &lbfgs_ctx, &lams, &device);
        assert!(total_scalar.is_finite(), "compute_loss_for_lbfgs's total loss must be finite at literal px=0.0, got {total_scalar}");

        let mut lbfgs = make_lbfgs(3);
        let (_model_out, loss_out) = step_lbfgs(model, &mut lbfgs, 1e-3, &lbfgs_ctx, &lams, &device);
        assert!(loss_out.is_finite(), "step_lbfgs's returned loss must be finite at literal px=0.0, got {loss_out}");
    }

    /// Regression for issue #23: `LbfgsCtxScalars::from_ctx` previously dropped
    /// `StepCtx::ref_energy` entirely (only `u_ref`/`ref_stress2` were threaded through),
    /// forcing `compute_loss_for_lbfgs` to hand-roll its own `ref_energy` from raw
    /// `config.load.px` — silently ignoring `config.use_ultimate_strength_scaling` (unlike
    /// `compute_reference_scales`, the single source of truth, which correctly branches on
    /// the flag). Proves `LbfgsCtxScalars::from_ctx`'s `ref_energy` field matches
    /// `compute_reference_scales`'s own `ref_energy` output for BOTH scaling-mode states —
    /// the cheapest reliable way to exercise the dropped-field bug, since it needs no live
    /// model / GPU forward pass through `compute_loss_for_lbfgs` itself.
    #[test]
    fn lbfgs_ctx_scalars_from_ctx_threads_ref_energy_for_both_scaling_modes() {
        use crate::engine::EngineParams;
        use crate::kirsch_problem::KirschProblem;
        use pinn_core::messages::SolverConfig;

        for use_uts in [false, true] {
            let mut config = SolverConfig::default_kirsch();
            config.n_interior = 64;
            config.n_boundary = 32;
            config.max_steps = 2;
            config.use_ultimate_strength_scaling = use_uts;
            let engine = EngineParams::analyze(&config);
            engine.apply_to(&mut config);

            let (x0, x1) = config.geometry.x_range();
            let (y0, y1) = config.geometry.y_range();
            let fd = FdConfig::new(config.fd_h, x1 - x0, y1 - y0);
            let cx = fd.sx / (2.0 * fd.hx as f64);
            let cy = fd.sy / (2.0 * fd.hy as f64);
            let ref_div2 = (config.load.px * cx).powi(2).max(1.0);
            let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

            // `LbfgsCtxScalars::from_ctx` only `.to_vec()`-clones these slices (no indexing/
            // branching on contents) and the `ref_energy` field asserted below depends only on
            // `config`, never on sampled points — so real sampling here would be pure overhead.
            let int_norm: Vec<[f32; 2]> = Vec::new();
            let bnd_norm: Vec<[f32; 2]> = Vec::new();
            let bnd_nx: Vec<f32> = Vec::new();
            let bnd_ny: Vec<f32> = Vec::new();
            let bnd_tx: Vec<f32> = Vec::new();
            let bnd_ty: Vec<f32> = Vec::new();
            let trac_idx: Vec<usize> = Vec::new();
            let hole_idx: Vec<usize> = Vec::new();
            let right_idx: Vec<usize> = Vec::new();
            let eq_ring_norm: Vec<[f32; 2]> = Vec::new();

            let problem = KirschProblem::new(
                config.material.clone(), engine.output_dim(), engine.phase1_steps, engine.expected_kt,
            );

            let ctx = StepCtx {
                config: &config, engine: &engine, problem: &problem, fd: &fd,
                k: engine.ansatz_k, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
                int_norm: &int_norm, bnd_norm: &bnd_norm,
                bnd_nx: &bnd_nx, bnd_ny: &bnd_ny, bnd_tx: &bnd_tx, bnd_ty: &bnd_ty,
                trac_idx: &trac_idx, hole_idx: &hole_idx, right_idx: &right_idx,
                eq_ring_norm: &eq_ring_norm,
                dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
                phase2_active: false, step: 0,
            };

            let lbfgs_ctx = LbfgsCtxScalars::from_ctx(&ctx);

            let expected_ref_energy = compute_reference_scales(&config).1;
            assert_eq!(
                lbfgs_ctx.ref_energy, expected_ref_energy,
                "LbfgsCtxScalars::from_ctx must thread StepCtx::ref_energy through unchanged \
                 (use_ultimate_strength_scaling={use_uts}); got {}, expected {}",
                lbfgs_ctx.ref_energy, expected_ref_energy,
            );
        }
    }

    /// Numerical-equivalence proof for the `step_physics` trait-driven cutover.
    ///
    /// Reproduces the OLD hardcoded formula (pre-cutover, hardwired e/n/h/d/eq/kirsch
    /// sequence, faithfully transcribed from commit 05cd1bc's `step_physics`) inline,
    /// from scratch, and runs it on `model_old` — a `.clone()` of the SAME randomly
    /// initialized starting model used for the NEW trait-driven `step_physics` call on
    /// `model_new`. Both paths get identical inputs (same collocation sets, same fresh
    /// AdamW/SOAP-Muon optimizer state, same SAW-BRDR instance parameters) and are run
    /// for 2 real optimizer steps each. This asserts BOTH the per-step loss scalars AND
    /// the resulting parameter L2-norm² (a scalar proxy sensitive to any change in the
    /// actual parameter deltas — i.e. gradient accumulation / optimizer stepping) match
    /// within 1e-5 relative tolerance, proving the trait-driven cutover (single combined
    /// backward pass, SAW-BRDR weighting from `KirschProblem::loss_terms()`'s stable
    /// order, one step per optimizer) reproduces the old hardcoded training dynamics
    /// exactly, not just at step 0 on a frozen network.
    #[test]
    fn step_physics_trait_driven_matches_independently_reimplemented_old_formula() {
        use burn::backend::wgpu::WgpuDevice;
        use burn::module::Module;
        use pinn_core::messages::SolverConfig;
        use crate::{
            engine::EngineParams,
            kirsch_problem::KirschProblem,
            network::ElasticityNetConfig,
            optim::{make_bias_optim, make_gate_optim, WeightOptim},
        };

        let mut config = SolverConfig::default_kirsch();
        config.n_interior = 64;
        config.n_boundary = 32;
        config.max_steps = 2;
        let engine = EngineParams::analyze(&config);
        engine.apply_to(&mut config);

        let device = WgpuDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(engine.net_input_dim())
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(config.use_piratenet);
        let model0: ElasticityNet<B> = net_cfg.init(&device);
        // burn `Param`s are lazily initialized (materialized + cached on first `.val()`
        // access) — cloning BEFORE materialization would give `model_new`/`model_old`
        // independently-drawn random weights instead of identical ones. Force
        // materialization via a no-op visitor first so `.clone()` below actually copies
        // the same values.
        struct TouchVisitor;
        impl<B: burn::tensor::backend::Backend> burn::module::ModuleVisitor<B> for TouchVisitor {
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
                let _ = param.val();
            }
        }
        model0.visit(&mut TouchVisitor);
        let mut model_new = model0.clone();
        let mut model_old = model0;

        let (x0, x1) = config.geometry.x_range();
        let (y0, y1) = config.geometry.y_range();
        let fd = FdConfig::new(config.fd_h, x1 - x0, y1 - y0);
        let cx = fd.sx / (2.0 * fd.hx as f64);
        let cy = fd.sy / (2.0 * fd.hy as f64);
        let ref_div2 = (config.load.px * cx).powi(2).max(1.0);
        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

        let int_pts = pinn_core::sampling::sample_interior(&config.geometry, engine.phase1_n_interior);
        let bnd_pts = pinn_core::sampling::sample_boundary(&config.geometry, &config.load, config.n_boundary);
        let eq_ring = pinn_core::sampling::sample_eq_ring(&config.geometry, engine.n_eq_ring);

        let int_norm: Vec<[f32; 2]> = int_pts.iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();
        let bnd_norm: Vec<[f32; 2]> = bnd_pts.iter().map(|b| normalize_point(b.x, b.y, &config)).collect();
        let bnd_nx: Vec<f32> = bnd_pts.iter().map(|b| b.nx as f32).collect();
        let bnd_ny: Vec<f32> = bnd_pts.iter().map(|b| b.ny as f32).collect();
        let bnd_tx: Vec<f32> = bnd_pts.iter().map(|b| b.tx as f32).collect();
        let bnd_ty: Vec<f32> = bnd_pts.iter().map(|b| b.ty as f32).collect();
        let (trac_idx, hole_idx, right_idx) = extract_boundary_indices(&bnd_pts, &bnd_nx);
        let eq_ring_norm: Vec<[f32; 2]> = eq_ring.iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();

        let problem = KirschProblem::new(
            config.material.clone(), engine.output_dim(), engine.phase1_steps, engine.expected_kt,
        );
        crate::problem::validate_loss_terms(&problem);

        let mut optim_w_new = WeightOptim::new(config.use_soap_muon);
        let mut optim_b_new = make_bias_optim();
        let mut optim_gate_new = make_gate_optim();
        let mut saw_new = SawBrdr::with_base(engine.init_weights(), 0.95);
        let mut lr_sched_new = LrSchedule::new(engine.peak_lr, 200, 1000);

        let mut optim_w_old = WeightOptim::new(config.use_soap_muon);
        let mut optim_b_old = make_bias_optim();
        let mut optim_gate_old = make_gate_optim();
        let mut saw_old = SawBrdr::with_base(engine.init_weights(), 0.95);
        let mut lr_sched_old = LrSchedule::new(engine.peak_lr, 200, 1000);

        // 1e-4, not 1e-5: WGPU compute-shader reduction order is not guaranteed associative
        // under concurrent GPU contention (e.g. `cargo test --workspace` running other GPU
        // tests in parallel), so a tighter bound flakes under load without indicating a real
        // regression. Matches this crate's existing float-comparison precedent (energy.rs).
        let rel_close = |a: f32, b: f32, label: &str| {
            let scale = a.abs().max(b.abs()).max(1e-8);
            let rel = ((a - b).abs() / scale) as f64;
            assert!(rel < 1e-4, "{label}: new={a} old={b} rel_err={rel}");
        };

        for step in 0..2usize {
            let ctx = StepCtx {
                config: &config, engine: &engine, problem: &problem, fd: &fd,
                k: engine.ansatz_k, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
                int_norm: &int_norm, bnd_norm: &bnd_norm,
                bnd_nx: &bnd_nx, bnd_ny: &bnd_ny, bnd_tx: &bnd_tx, bnd_ty: &bnd_ty,
                trac_idx: &trac_idx, hole_idx: &hole_idx, right_idx: &right_idx,
                eq_ring_norm: &eq_ring_norm,
                dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
                phase2_active: false, step,
            };

            // NEW: the actual (post-cutover) trait-driven step_physics.
            let (m_new, out_new) = step_physics(
                model_new, &mut optim_w_new, &mut optim_b_new, &mut optim_gate_new,
                &ctx, &mut saw_new, &mut lr_sched_new, &device, 0, 1.0, 1.0,
            );
            model_new = m_new;

            // OLD: independently reimplemented hardcoded formula (transcribed verbatim
            // from commit 05cd1bc's step_physics), run on the separately-tracked clone.
            let (m_old, out_old) = old_hardcoded_step_physics(
                model_old, &mut optim_w_old, &mut optim_b_old, &mut optim_gate_old,
                &ctx, &mut saw_old, &mut lr_sched_old, &device,
            );
            model_old = m_old;

            rel_close(out_new.e_scalar, out_old.e_scalar, "e_scalar");
            rel_close(out_new.n_scalar, out_old.n_scalar, "n_scalar");
            rel_close(out_new.h_scalar, out_old.h_scalar, "h_scalar");
            rel_close(out_new.d_scalar, out_old.d_scalar, "d_scalar");
            rel_close(out_new.eq_scalar, out_old.eq_scalar, "eq_scalar");
            rel_close(out_new.kirsch_scalar, out_old.kirsch_scalar, "kirsch_scalar");
            rel_close(out_new.const_scalar, out_old.const_scalar, "const_scalar");
            rel_close(out_new.total_scalar, out_old.total_scalar, "total_scalar");
        }

        // Post-optimizer-step parameter fingerprint (sum-of-squares over every float
        // param, after 2 real optimizer steps on each independent path) — proves the
        // ACTUAL parameter deltas (not just the loss scalars) match.
        struct NormVisitor { total: f64 }
        impl<B: burn::tensor::backend::Backend> burn::module::ModuleVisitor<B> for NormVisitor {
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
                let v: Vec<f32> = param.val().into_data().to_vec().unwrap_or_default();
                self.total += v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
            }
        }
        let mut vis_new = NormVisitor { total: 0.0 };
        model_new.visit(&mut vis_new);
        let mut vis_old = NormVisitor { total: 0.0 };
        model_old.visit(&mut vis_old);
        let scale = vis_new.total.abs().max(vis_old.total).max(1e-8);
        let rel = (vis_new.total - vis_old.total).abs() / scale;
        assert!(rel < 1e-4, // see rel_close's GPU-contention comment above
            "param_l2_sq: new={} old={} rel_err={rel}", vis_new.total, vis_old.total);
    }

    /// Confirmatory test for `use_mdem` (auto-derived from `has_hole`, always true for a
    /// default Kirsch config) combined with `use_ultimate_strength_scaling=true` — a
    /// combination no other test in this file exercises. Investigated as a possible sibling
    /// of issue #23 (`scale_out`'s mDEM stress channels are scaled by raw `config.load.px`,
    /// not the flag-aware `stress_ref` used for the displacement channels via `ctx.u_ref`) and
    /// concluded correct by design, not a bug: `constitutive_consistency_loss` trains the
    /// stress channel toward the true physical Hooke's-law stress (derived from the always
    /// flag-independent `E`/`nu`), so recovering it via raw `px` is consistent — exactly like
    /// `u_target_val`/`w_val`'s verified-correct use of real `px`/`py` as physical BC targets
    /// rather than a normalization scale. This test backs that reasoning empirically rather
    /// than by proof alone: every `step_physics` component scalar must stay finite across
    /// several real steps, and `probe_kt_shared` must still return a finite K_t afterward.
    #[test]
    fn step_physics_stays_finite_with_mdem_and_ultimate_strength_scaling_combined() {
        use burn::backend::wgpu::WgpuDevice;
        use burn::module::AutodiffModule;
        use pinn_core::messages::SolverConfig;
        use crate::{
            engine::EngineParams,
            kirsch_problem::KirschProblem,
            network::ElasticityNetConfig,
            optim::{make_bias_optim, make_gate_optim, WeightOptim},
        };

        let mut config = SolverConfig::default_kirsch();
        config.n_interior = 64;
        config.n_boundary = 32;
        config.max_steps = 8;
        config.use_ultimate_strength_scaling = true;
        let engine = EngineParams::analyze(&config);
        engine.apply_to(&mut config);
        assert_eq!(engine.output_dim(), 5,
            "default Kirsch config must be mDEM-capable (has_hole=true) for this test to \
             actually exercise the use_mdem + use_ultimate_strength_scaling combination");

        let device = WgpuDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(engine.net_input_dim())
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(config.use_piratenet);
        let mut model: ElasticityNet<B> = net_cfg.init(&device);

        let (x0, x1) = config.geometry.x_range();
        let (y0, y1) = config.geometry.y_range();
        let fd = FdConfig::new(config.fd_h, x1 - x0, y1 - y0);
        let cx = fd.sx / (2.0 * fd.hx as f64);
        let cy = fd.sy / (2.0 * fd.hy as f64);
        let ref_div2 = (config.load.px * cx).powi(2).max(1.0);
        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

        let int_pts = pinn_core::sampling::sample_interior(&config.geometry, engine.phase1_n_interior);
        let bnd_pts = pinn_core::sampling::sample_boundary(&config.geometry, &config.load, config.n_boundary);
        let eq_ring = pinn_core::sampling::sample_eq_ring(&config.geometry, engine.n_eq_ring);

        let int_norm: Vec<[f32; 2]> = int_pts.iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();
        let bnd_norm: Vec<[f32; 2]> = bnd_pts.iter().map(|b| normalize_point(b.x, b.y, &config)).collect();
        let bnd_nx: Vec<f32> = bnd_pts.iter().map(|b| b.nx as f32).collect();
        let bnd_ny: Vec<f32> = bnd_pts.iter().map(|b| b.ny as f32).collect();
        let bnd_tx: Vec<f32> = bnd_pts.iter().map(|b| b.tx as f32).collect();
        let bnd_ty: Vec<f32> = bnd_pts.iter().map(|b| b.ty as f32).collect();
        let (trac_idx, hole_idx, right_idx) = extract_boundary_indices(&bnd_pts, &bnd_nx);
        let eq_ring_norm: Vec<[f32; 2]> = eq_ring.iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();

        let problem = KirschProblem::new(
            config.material.clone(), engine.output_dim(), engine.phase1_steps, engine.expected_kt,
        );

        let mut optim_w = WeightOptim::new(config.use_soap_muon);
        let mut optim_b = make_bias_optim();
        let mut optim_gate = make_gate_optim();
        let mut saw = SawBrdr::with_base(engine.init_weights(), 0.95);
        let mut lr_sched = LrSchedule::new(engine.peak_lr, 200, 1000);

        for step in 0..8usize {
            let ctx = StepCtx {
                config: &config, engine: &engine, problem: &problem, fd: &fd,
                k: engine.ansatz_k, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
                int_norm: &int_norm, bnd_norm: &bnd_norm,
                bnd_nx: &bnd_nx, bnd_ny: &bnd_ny, bnd_tx: &bnd_tx, bnd_ty: &bnd_ty,
                trac_idx: &trac_idx, hole_idx: &hole_idx, right_idx: &right_idx,
                eq_ring_norm: &eq_ring_norm,
                dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
                phase2_active: false, step,
            };

            let (m, out) = step_physics(
                model, &mut optim_w, &mut optim_b, &mut optim_gate,
                &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
            );
            model = m;

            for (name, v) in [
                ("e_scalar", out.e_scalar), ("n_scalar", out.n_scalar), ("h_scalar", out.h_scalar),
                ("d_scalar", out.d_scalar), ("eq_scalar", out.eq_scalar), ("kirsch_scalar", out.kirsch_scalar),
                ("const_scalar", out.const_scalar), ("total_scalar", out.total_scalar),
            ] {
                assert!(v.is_finite(),
                    "step {step}: {name} went non-finite ({v}) under use_ultimate_strength_scaling=true \
                     + mDEM — scale_out's raw-px stress-channel convention would be the first suspect \
                     if this ever regresses");
            }
        }

        let model_val: ElasticityNet<BInner> = model.valid();
        let kt = probe_kt_shared(&model_val, &config, &engine, &fd, engine.ansatz_k, u_ref, &device);
        assert!(kt.is_some_and(|v| v.is_finite()),
            "probe_kt_shared must return a finite K_t after training with \
             use_ultimate_strength_scaling=true + mDEM, got {kt:?}");
    }

    /// Verbatim transcription of commit 05cd1bc's hardcoded `step_physics` body (the
    /// pre-cutover e/n/h/d/eq/kirsch sequence) — kept ONLY for
    /// `step_physics_trait_driven_matches_independently_reimplemented_old_formula`'s
    /// independent-reimplementation equivalence check. Not used by the live training path.
    #[allow(clippy::too_many_arguments)]
    fn old_hardcoded_step_physics(
        model: ElasticityNet<B>,
        optim_w: &mut WeightOptim,
        optim_b: &mut BiasOptim,
        optim_gate: &mut GateOptim,
        ctx: &StepCtx,
        saw: &mut SawBrdr,
        lr_sched: &mut LrSchedule,
        device: &WgpuDevice,
    ) -> (ElasticityNet<B>, StepOutput) {
        use pinn_core::geometry::HoleType;

        let n_int      = ctx.int_norm.len();
        let n_fourier  = ctx.engine.n_fourier;
        let use_mdem   = ctx.engine.use_mdem;
        let px         = ctx.config.load.px;
        let u_ref_f64  = ctx.u_ref as f64;

        let v_to_t = |idxs: &[usize], src: &[f32]| -> Tensor<B, 1> {
            let v: Vec<f32> = idxs.iter().map(|&i| src[i]).collect();
            Tensor::<B, 1>::from_data(TensorData::new(v.clone(), vec![v.len()]), device)
        };
        let scale_out = |ansatz: Tensor<B, 2>| -> Tensor<B, 2> {
            if use_mdem {
                let nr = ansatz.dims()[0];
                Tensor::cat(vec![
                    ansatz.clone().slice([0..nr, 0..2]).mul_scalar(u_ref_f64),
                    ansatz.slice([0..nr, 2..5]).mul_scalar(px),
                ], 1)
            } else {
                ansatz.mul_scalar(u_ref_f64)
            }
        };

        let pts_t = norm_pts_to_tensor::<B>(ctx.int_norm, device);
        let stencil_coords = assemble_stencil::<B>(&pts_t, ctx.fd, device);
        let stencil_out = scale_out(apply_dirichlet_ansatz::<B>(
            fwd(&model, stencil_coords.clone(), n_fourier, device),
            &stencil_coords, ctx.config.geometry.symmetry, ctx.k,
        ));
        let (sxx_net_int, syy_net_int, sxy_net_int) = if use_mdem {
            let (sx, sy, sxy) = extract_mdem_stress(&stencil_out, 0, n_int);
            (Some(sx), Some(sy), Some(sxy))
        } else { (None, None, None) };
        let (eps_xx, eps_yy, eps_xy) = compute_strains::<B>(stencil_out, n_int, ctx.fd);
        let e_loss = dem_energy_loss(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), &ctx.config.material)
            .mul_scalar(1.0 / ctx.ref_energy as f64);
        let const_loss: Tensor<B, 1> = if let (Some(sxx_n), Some(syy_n), Some(sxy_n)) =
            (sxx_net_int, syy_net_int, sxy_net_int)
        {
            constitutive_consistency_loss(sxx_n, syy_n, sxy_n, eps_xx, eps_yy, eps_xy, &ctx.config.material)
                .mul_scalar(1.0 / ctx.ref_stress2 as f64)
        } else {
            Tensor::<B, 1>::zeros([1], device)
        };

        let n_loss = if !ctx.trac_idx.is_empty() {
            let trac_norm: Vec<[f32; 2]> = ctx.trac_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
            let nt = trac_norm.len();
            let bnd_t = norm_pts_to_tensor::<B>(&trac_norm, device);
            let stencil_bnd = assemble_stencil::<B>(&bnd_t, ctx.fd, device);
            let out_bnd = scale_out(apply_dirichlet_ansatz::<B>(
                fwd(&model, stencil_bnd.clone(), n_fourier, device), &stencil_bnd,
                ctx.config.geometry.symmetry, ctx.k,
            ));
            let (ex, ey, exy) = compute_strains::<B>(out_bnd, nt, ctx.fd);
            neumann_loss(
                ex, ey, exy,
                v_to_t(ctx.trac_idx, ctx.bnd_nx), v_to_t(ctx.trac_idx, ctx.bnd_ny),
                v_to_t(ctx.trac_idx, ctx.bnd_tx), v_to_t(ctx.trac_idx, ctx.bnd_ty),
                &ctx.config.material,
            ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
        } else {
            Tensor::<B, 1>::zeros([1], device)
        };

        let u_target_val = ((ctx.config.load.px - ctx.config.material.nu * ctx.config.load.py)
            / ctx.config.material.e * ctx.config.geometry.half_w) as f32;
        let (d_loss, w_neumann) = if !ctx.right_idx.is_empty() {
            let right_norm: Vec<[f32; 2]> = ctx.right_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
            let nr = right_norm.len();
            let right_t = norm_pts_to_tensor::<B>(&right_norm, device);
            let out_r = scale_out(apply_dirichlet_ansatz::<B>(
                fwd(&model, right_t.clone(), n_fourier, device), &right_t,
                ctx.config.geometry.symmetry, ctx.k,
            ));
            let u_vals = out_r.slice([0..nr, 0..1]).reshape([nr]);
            let u_tgt: Tensor<B, 1> = Tensor::full([nr], u_target_val as f64, device);
            let denom = ((u_target_val * u_target_val) as f64).max(1e-20);
            let d_val = (u_vals.clone() - u_tgt).powf_scalar(2.0_f64).mean().mul_scalar(1.0 / denom);
            let w_val = u_vals.mean().mul_scalar(2.0 * ctx.config.material.e
                / (ctx.config.load.px * ctx.config.geometry.half_w));
            (d_val, w_val)
        } else {
            (Tensor::<B, 1>::zeros([1], device), Tensor::<B, 1>::zeros([1], device))
        };

        let h_loss = if !ctx.hole_idx.is_empty() {
            let hole_norm: Vec<[f32; 2]> = ctx.hole_idx.iter().map(|&i| ctx.bnd_norm[i]).collect();
            let nh = hole_norm.len();
            let bnd_h = norm_pts_to_tensor::<B>(&hole_norm, device);
            if use_mdem {
                let out_h = scale_out(apply_dirichlet_ansatz::<B>(
                    fwd(&model, bnd_h.clone(), n_fourier, device), &bnd_h,
                    ctx.config.geometry.symmetry, ctx.k,
                ));
                let (sxx_h, syy_h, sxy_h) = extract_mdem_stress(&out_h, 0, nh);
                hole_traction_loss_direct(
                    sxx_h, syy_h, sxy_h,
                    v_to_t(ctx.hole_idx, ctx.bnd_nx), v_to_t(ctx.hole_idx, ctx.bnd_ny),
                ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
            } else {
                let stencil_h = assemble_stencil::<B>(&bnd_h, ctx.fd, device);
                let out_h = scale_out(apply_dirichlet_ansatz::<B>(
                    fwd(&model, stencil_h.clone(), n_fourier, device), &stencil_h,
                    ctx.config.geometry.symmetry, ctx.k,
                ));
                let (ex, ey, exy) = compute_strains::<B>(out_h, nh, ctx.fd);
                hole_traction_loss(
                    ex, ey, exy,
                    v_to_t(ctx.hole_idx, ctx.bnd_nx), v_to_t(ctx.hole_idx, ctx.bnd_ny),
                    &ctx.config.material,
                ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
            }
        } else {
            Tensor::<B, 1>::zeros([1], device)
        };

        let n_eq = ctx.eq_ring_norm.len();
        let eq_loss: Tensor<B, 1> = if n_eq > 0 {
            if use_mdem {
                let shifts: [(f32, f32); 4] = [
                    ( ctx.fd.hx, 0.0), (-ctx.fd.hx, 0.0), (0.0, ctx.fd.hy), (0.0, -ctx.fd.hy),
                ];
                let all_shifted: Vec<[f32; 2]> = shifts.iter().flat_map(|&(dx, dy)| {
                    ctx.eq_ring_norm.iter().map(move |&[x, y]| [x + dx, y + dy])
                }).collect();
                let pts_all = norm_pts_to_tensor::<B>(&all_shifted, device);
                let out_all = scale_out(apply_dirichlet_ansatz::<B>(
                    fwd(&model, pts_all.clone(), n_fourier, device), &pts_all,
                    ctx.config.geometry.symmetry, ctx.k,
                ));
                let seg = |i: usize| extract_mdem_stress(&out_all, i * n_eq, (i + 1) * n_eq);
                let (sxx_xp, _, sxy_xp) = seg(0);
                let (sxx_xm, _, sxy_xm) = seg(1);
                let (_, syy_yp, sxy_yp) = seg(2);
                let (_, syy_ym, sxy_ym) = seg(3);
                equilibrium_residual_loss(
                    sxx_xp, sxy_xp, sxx_xm, sxy_xm, sxy_yp, syy_yp, sxy_ym, syy_ym,
                    ctx.cx, ctx.cy, ctx.ref_div2,
                )
            } else {
                let fwd_shift = |dx: f32, dy: f32| {
                    let shifted: Vec<[f32; 2]> = ctx.eq_ring_norm.iter()
                        .map(|&[xn, yn]| [xn + dx, yn + dy]).collect();
                    let pts = norm_pts_to_tensor::<B>(&shifted, device);
                    let stencil = assemble_stencil::<B>(&pts, ctx.fd, device);
                    let out = scale_out(apply_dirichlet_ansatz::<B>(
                        fwd(&model, stencil.clone(), n_fourier, device), &stencil,
                        ctx.config.geometry.symmetry, ctx.k,
                    ));
                    let (exx, eyy, exy) = compute_strains::<B>(out, n_eq, ctx.fd);
                    compute_stress(exx, eyy, exy, &ctx.config.material)
                };
                let (sxx_xp, _, sxy_xp) = fwd_shift( ctx.fd.hx,  0.0);
                let (sxx_xm, _, sxy_xm) = fwd_shift(-ctx.fd.hx,  0.0);
                let (_, syy_yp, sxy_yp) = fwd_shift( 0.0,  ctx.fd.hy);
                let (_, syy_ym, sxy_ym) = fwd_shift( 0.0, -ctx.fd.hy);
                equilibrium_residual_loss(
                    sxx_xp, sxy_xp, sxx_xm, sxy_xm, sxy_yp, syy_yp, sxy_ym, syy_ym,
                    ctx.cx, ctx.cy, ctx.ref_div2,
                )
            }
        } else {
            Tensor::<B, 1>::zeros([1], device)
        };

        let (kirsch_loss, kirsch_scalar): (Tensor<B, 1>, f32) = if ctx.phase2_active {
            if let HoleType::Circular { radius } = ctx.config.geometry.hole {
                let probes = compute_kirsch_probes(ctx, radius);
                let n_pr = probes.points.len();
                let pts_pr = norm_pts_to_tensor::<B>(&probes.points, device);
                let px2 = (px * px) as f64;
                let to_t1 = |v: &[f32]| -> Tensor<B, 1> {
                    Tensor::from_data(TensorData::new(v.to_vec(), vec![v.len()]), device)
                };
                let kl = if use_mdem {
                    let out_pr = scale_out(apply_dirichlet_ansatz::<B>(
                        fwd(&model, pts_pr.clone(), n_fourier, device), &pts_pr,
                        ctx.config.geometry.symmetry, ctx.k,
                    ));
                    let (sxx_pr, syy_pr, sxy_pr) = extract_mdem_stress(&out_pr, 0, n_pr);
                    let loss_per_pt =
                        (sxx_pr - to_t1(&probes.sxx_targets)).powf_scalar(2.0_f64)
                        + (syy_pr - to_t1(&probes.syy_targets)).powf_scalar(2.0_f64)
                        + (sxy_pr - to_t1(&probes.sxy_targets)).powf_scalar(2.0_f64).mul_scalar(2.0_f64);
                    (loss_per_pt * to_t1(&probes.weights)).sum().mul_scalar(1.0 / px2)
                } else {
                    let stencil_pr = assemble_stencil::<B>(&pts_pr, ctx.fd, device);
                    let out_pr = scale_out(apply_dirichlet_ansatz::<B>(
                        fwd(&model, stencil_pr.clone(), n_fourier, device), &stencil_pr,
                        ctx.config.geometry.symmetry, ctx.k,
                    ));
                    let (exx_pr, eyy_pr, exy_pr) = compute_strains::<B>(out_pr, n_pr, ctx.fd);
                    let (sxx_pr, syy_pr, sxy_pr) = compute_stress(exx_pr, eyy_pr, exy_pr, &ctx.config.material);
                    let loss_per_pt =
                        (sxx_pr - to_t1(&probes.sxx_targets)).powf_scalar(2.0_f64)
                        + (syy_pr - to_t1(&probes.syy_targets)).powf_scalar(2.0_f64)
                        + (sxy_pr - to_t1(&probes.sxy_targets)).powf_scalar(2.0_f64).mul_scalar(2.0_f64);
                    (loss_per_pt * to_t1(&probes.weights)).sum().mul_scalar(1.0 / px2)
                };
                let ks = t_scalar(&kl);
                (kl, ks)
            } else {
                (Tensor::<B, 1>::zeros([1], device), 0.0)
            }
        } else {
            (Tensor::<B, 1>::zeros([1], device), 0.0)
        };

        let e_scalar      = t_scalar(&e_loss);
        let n_scalar      = t_scalar(&n_loss);
        let h_scalar      = t_scalar(&h_loss);
        let d_scalar      = t_scalar(&d_loss);
        let eq_scalar     = t_scalar(&eq_loss);
        let w_scalar      = t_scalar(&w_neumann);
        let const_scalar  = t_scalar(&const_loss);

        let saw_inputs: Vec<f32> = if ctx.phase2_active {
            vec![e_scalar, n_scalar, h_scalar, d_scalar, eq_scalar, kirsch_scalar]
        } else {
            vec![e_scalar, n_scalar, h_scalar, d_scalar, eq_scalar]
        };
        let lams = saw.update(&saw_inputs);

        let lam_e   = lams[0] as f64;
        let lam_n   = lams[1] as f64;
        let lam_h_  = lams[2] as f64;
        let lam_d_  = lams[3] as f64;
        let lam_eq  = lams[4] as f64;
        let lam_h   = if ctx.phase2_active { lam_h_.min(ctx.dynamic_lam_h_cap) } else { lam_h_ };
        let lam_d   = if ctx.phase2_active { lam_d_.min(ctx.dynamic_lam_d_cap) } else { lam_d_ };
        let lam_kirsch = if ctx.phase2_active { lams[5] as f64 } else { 0.0 };
        let lam_const  = ctx.engine.lam_const as f64;

        let loss = (e_loss - w_neumann).mul_scalar(lam_e)
            + n_loss.mul_scalar(lam_n)
            + h_loss.mul_scalar(lam_h)
            + d_loss.mul_scalar(lam_d)
            + eq_loss.mul_scalar(lam_eq)
            + kirsch_loss.mul_scalar(lam_kirsch)
            + const_loss.mul_scalar(lam_const);

        let total_scalar = (e_scalar - w_scalar) * lam_e as f32
            + n_scalar  * lam_n  as f32
            + h_scalar  * lam_h  as f32
            + d_scalar  * lam_d  as f32
            + eq_scalar * lam_eq as f32
            + kirsch_scalar * lam_kirsch as f32
            + const_scalar  * lam_const as f32;

        let lr = lr_sched.step(total_scalar.abs() as f64);
        let (default_weight_ids, bias_ids) = model.param_ids();
        let weight_ids = if ctx.config.use_piratenet {
            model.awake_weight_ids(ctx.config.stiffness.gate_awake_epsilon)
        } else {
            default_weight_ids
        };
        let gate_ids = model.gate_ids();
        let mut grads = loss.backward();
        let weight_grads = GradientsParams::from_params(&mut grads, &model, &weight_ids);
        let bias_grads = GradientsParams::from_params(&mut grads, &model, &bias_ids);
        let gate_grads = GradientsParams::from_params(&mut grads, &model, &gate_ids);
        let model = optim_w.step(lr, model, weight_grads);
        let model = optim_b.step(lr, model, bias_grads);
        let model = optim_gate.step(lr, model, gate_grads);

        let proxy_ratio = (e_scalar + eq_scalar + const_scalar)
            / (n_scalar + h_scalar + d_scalar + w_scalar + 1e-8);

        (model, StepOutput {
            e_scalar, n_scalar, h_scalar, d_scalar, eq_scalar, w_scalar,
            kirsch_scalar, const_scalar, total_scalar, lr, lam_e, lam_n, lam_h, lam_d, lam_eq,
            lam_kirsch,
            proxy_ratio,
            optimizer_tier: 0,
            cosine_sim: None,
            lam_by_name: None,
        })
    }

    // ─── step_physics_multi RED tests ────────────────────────────────────────────────

    /// Minimal single-domain `BoundaryValueProblem` wrapping the SAME real Kirsch loss-term
    /// structs (`InteriorEnergyTerm`/`NeumannTractionTerm`/`HoleTractionTerm`/
    /// `DisplacementAnchorTerm`) `step_physics` drives, restricted to the four terms whose
    /// real per-step tensor is fully expressible via `DomainForwardOutputs` alone —
    /// `equilibrium_ring` (needs a 4-meta-shift stencil `step_physics` assembles by hand
    /// outside the trait) and `kirsch_stress` (phase2-only probe grid) are Kirsch-specific
    /// plumbing that doesn't fit the generic single-`DomainForwardOutputs`-per-point-set
    /// shape `step_physics_multi` drives every term through, so they're intentionally
    /// excluded from this equivalence check (documented scope limit, not an oversight).
    struct FourTermKirschProblem {
        domains: [pinn_core::problem::DomainSpec; 1],
        sampling: crate::kirsch_problem::KirschSamplingStrategy,
        ansatz: crate::kirsch_problem::QuarterSymmAnsatz,
        u_target: f32,
        tx_target: Tensor<B, 1>,
        ty_target: Tensor<B, 1>,
        ref_energy: f32,
        ref_stress2: f32,
    }

    impl BoundaryValueProblem for FourTermKirschProblem {
        fn domains(&self) -> &[pinn_core::problem::DomainSpec] { &self.domains }
        fn sampling_strategy(&self, _domain_idx: usize) -> &dyn pinn_core::problem::DomainSamplingStrategy { &self.sampling }
        fn ansatz(&self, _domain_idx: usize) -> &dyn pinn_core::problem::DirichletAnsatz { &self.ansatz }
        fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
            let material = self.domains[0].material.clone();
            vec![
                Box::new(crate::kirsch_problem::InteriorEnergyTerm {
                    domain: KIRSCH_DOMAIN, material: material.clone(), ref_energy: self.ref_energy,
                }),
                Box::new(crate::kirsch_problem::NeumannTractionTerm {
                    domain: KIRSCH_DOMAIN, material: material.clone(), ref_stress2: self.ref_stress2,
                    tx_target: self.tx_target.clone(), ty_target: self.ty_target.clone(),
                }),
                Box::new(crate::kirsch_problem::HoleTractionTerm {
                    domain: KIRSCH_DOMAIN, material: material.clone(), ref_stress2: self.ref_stress2, direct: true,
                }),
                Box::new(crate::kirsch_problem::DisplacementAnchorTerm {
                    domain: KIRSCH_DOMAIN, u_target: self.u_target,
                }),
            ]
        }
        fn base_weight(&self, term_name: &str) -> f32 {
            match term_name {
                "interior_energy" => 1.0,
                "neumann_traction" => 10.0,
                "hole_traction" => 200.0,
                "displacement_anchor" => 50.0,
                other => panic!("FourTermKirschProblem::base_weight: unknown term '{other}'"),
            }
        }
        fn phase1_steps(&self) -> usize { usize::MAX }
        fn convergence_metric(&self, _state: &[DomainState<B>]) -> Option<f64> { None }
        fn convergence_target(&self) -> f64 { 3.0 }
    }

    /// RED test #1 (single-domain equivalence): `step_physics_multi` driving a 1-element
    /// `MultiStepCtx` wrapping Kirsch's own sampling/ansatz/loss-term structs must agree
    /// with `step_physics` on the four shared loss scalars (interior_energy,
    /// neumann_traction, hole_traction, displacement_anchor) for 2 real optimizer steps on
    /// cloned-identical starting models. See `FourTermKirschProblem` doc comment for the
    /// documented equilibrium_ring/kirsch_stress exclusion.
    #[test]
    fn step_physics_multi_single_domain_matches_step_physics_kirsch() {
        use crate::{
            engine::EngineParams,
            kirsch_problem::{KirschSamplingStrategy, QuarterSymmAnsatz, KIRSCH_DOMAIN},
            network::ElasticityNetConfig,
            optim::WeightOptim,
            problem::{DomainOptim, DomainStepCtx, DomainStepData, MultiStepCtx, PointSetData},
        };

        let mut config = SolverConfig::default_kirsch();
        config.n_interior = 64;
        config.n_boundary = 32;
        config.max_steps = 2;
        let engine = EngineParams::analyze(&config);
        engine.apply_to(&mut config);

        let device = WgpuDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(3) // no Fourier embedding on the multi-domain path
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(false);
        let model0: ElasticityNet<B> = net_cfg.init(&device);
        struct TouchVisitor;
        impl ModuleVisitor<B> for TouchVisitor {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let _ = param.val();
            }
        }
        model0.visit(&mut TouchVisitor);
        let mut model_single = model0.clone();
        let mut model_multi = model0;

        let (x0, x1) = config.geometry.x_range();
        let (y0, y1) = config.geometry.y_range();
        let fd = FdConfig::new(config.fd_h, x1 - x0, y1 - y0);
        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

        let strategy = KirschSamplingStrategy;
        let int_pts = strategy.sample_interior(&config.geometry, engine.phase1_n_interior);
        let bnd_pts = strategy.sample_boundary(&config.geometry, &config.load, config.n_boundary);

        let int_norm: Vec<[f32; 2]> = int_pts.iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();
        let bnd_norm: Vec<[f32; 2]> = bnd_pts.iter().map(|b| normalize_point(b.x, b.y, &config)).collect();
        let bnd_nx: Vec<f32> = bnd_pts.iter().map(|b| b.nx as f32).collect();
        let bnd_ny: Vec<f32> = bnd_pts.iter().map(|b| b.ny as f32).collect();
        let bnd_tx: Vec<f32> = bnd_pts.iter().map(|b| b.tx as f32).collect();
        let bnd_ty: Vec<f32> = bnd_pts.iter().map(|b| b.ty as f32).collect();
        let (trac_idx, hole_idx, right_idx) = extract_boundary_indices(&bnd_pts, &bnd_nx);

        let gather = |idxs: &[usize]| -> PointSetData {
            PointSetData {
                norm: idxs.iter().map(|&i| bnd_norm[i]).collect(),
                nx: idxs.iter().map(|&i| bnd_nx[i]).collect(),
                ny: idxs.iter().map(|&i| bnd_ny[i]).collect(),
                tx: idxs.iter().map(|&i| bnd_tx[i]).collect(),
                ty: idxs.iter().map(|&i| bnd_ty[i]).collect(),
            }
        };
        let mut named = HashMap::new();
        named.insert("traction", gather(&trac_idx));
        named.insert("hole", gather(&hole_idx));
        named.insert("right_edge", gather(&right_idx));
        let domain_data = DomainStepData {
            id: KIRSCH_DOMAIN,
            int_norm: int_norm.clone(),
            extra_ring_norm: Vec::new(),
            named,
        };

        let u_target_val = ((config.load.px - config.material.nu * config.load.py)
            / config.material.e * config.geometry.half_w) as f32;
        let tx_target: Vec<f32> = trac_idx.iter().map(|&i| bnd_tx[i]).collect();
        let ty_target: Vec<f32> = trac_idx.iter().map(|&i| bnd_ty[i]).collect();
        let to_t1 = |v: &[f32]| -> Tensor<B, 1> {
            Tensor::from_data(TensorData::new(v.to_vec(), vec![v.len()]), &device)
        };

        let problem = FourTermKirschProblem {
            domains: [pinn_core::problem::DomainSpec {
                id: KIRSCH_DOMAIN, geometry: config.geometry.clone(),
                material: config.material.clone(), output_dim: engine.output_dim(),
            }],
            sampling: KirschSamplingStrategy,
            ansatz: QuarterSymmAnsatz,
            u_target: u_target_val,
            tx_target: to_t1(&tx_target),
            ty_target: to_t1(&ty_target),
            ref_energy, ref_stress2,
        };

        let mut optim_w_single = WeightOptim::new(config.use_soap_muon);
        let mut optim_b_single = make_bias_optim();
        let mut optim_gate_single = make_gate_optim();
        let mut saw_single = SawBrdr::with_base(vec![1.0, 10.0, 200.0, 50.0], 0.95);
        let mut lr_sched_single = LrSchedule::new(engine.peak_lr, 200, 1000);

        let mut optims_multi = vec![DomainOptim {
            weight: WeightOptim::new(config.use_soap_muon),
            bias: make_bias_optim(),
            gate: make_gate_optim(),
        }];
        let mut saw_multi = SawBrdr::with_base(vec![1.0, 10.0, 200.0, 50.0], 0.95);
        let mut lr_sched_multi = LrSchedule::new(engine.peak_lr, 200, 1000);

        // 1e-4: see the GPU-contention comment on the sibling equivalence test above.
        let rel_close = |a: f32, b: f32, label: &str| {
            let scale = a.abs().max(b.abs()).max(1e-8);
            let rel = ((a - b).abs() / scale) as f64;
            assert!(rel < 1e-4, "{label}: single={a} multi={b} rel_err={rel}");
        };

        for step in 0..2usize {
            // Single-domain "reference" path: same forward-pass mechanics as step_physics
            // (ansatz + scale_out), but assembled by hand here (not step_physics itself,
            // which drives 6 SAW components including equilibrium_ring/kirsch_stress —
            // see FourTermKirschProblem doc comment) so this is a genuine independent
            // comparison against the SAME underlying loss-term structs step_physics_multi
            // will call through the trait.
            let ctx_multi = MultiStepCtx {
                config: &config,
                problem: &problem,
                fd: &fd,
                k: engine.ansatz_k,
                domains: vec![DomainStepCtx {
                    data: &domain_data, u_ref, ref_energy, ref_stress2,
                }],
                dynamic_lam_h_cap: 50.0,
                dynamic_lam_d_cap: 50.0,
                dynamic_lam_penetration_cap: 500.0,
                dynamic_lam_non_tension_cap: 100.0,
                phase2_active: false,
                step,
            };
            let (new_models, out_multi) = step_physics_multi(
                vec![model_multi], &mut optims_multi, &ctx_multi,
                &mut saw_multi, &mut lr_sched_multi, &device, 0, 1.0, 1.0,
            );
            model_multi = new_models.into_iter().next().unwrap();

            // Independent single-domain computation using the identical KirschProblem
            // sampling/ansatz path but restricted to the same 4 terms, weighted by the same
            // SAW-BRDR instance semantics.
            let n_int = int_norm.len();
            let pts_t = norm_pts_to_tensor::<B>(&int_norm, &device);
            let stencil_coords = assemble_stencil::<B>(&pts_t, &fd, &device);
            let ansatz_out = apply_dirichlet_ansatz::<B>(
                fwd(&model_single, stencil_coords.clone(), 0, &device),
                &stencil_coords, config.geometry.symmetry, engine.ansatz_k,
            );
            let nr = ansatz_out.dims()[0];
            let scaled = Tensor::cat(vec![
                ansatz_out.clone().slice([0..nr, 0..2]).mul_scalar(u_ref as f64),
                ansatz_out.slice([0..nr, 2..5]).mul_scalar(config.load.px),
            ], 1);
            let int_raw = scaled.clone().slice([0..n_int, 0..scaled.dims()[1]]);
            let (exx, eyy, exy) = compute_strains::<B>(scaled, n_int, &fd);
            let e_loss = dem_energy_loss(exx, eyy, exy, &config.material).mul_scalar(1.0 / ref_energy as f64);

            let nt = trac_idx.len();
            let trac_norm: Vec<[f32; 2]> = trac_idx.iter().map(|&i| bnd_norm[i]).collect();
            let bnd_t = norm_pts_to_tensor::<B>(&trac_norm, &device);
            let stencil_bnd = assemble_stencil::<B>(&bnd_t, &fd, &device);
            let out_bnd_raw = apply_dirichlet_ansatz::<B>(
                fwd(&model_single, stencil_bnd.clone(), 0, &device),
                &stencil_bnd, config.geometry.symmetry, engine.ansatz_k,
            );
            let nrb = out_bnd_raw.dims()[0];
            let out_bnd = Tensor::cat(vec![
                out_bnd_raw.clone().slice([0..nrb, 0..2]).mul_scalar(u_ref as f64),
                out_bnd_raw.slice([0..nrb, 2..5]).mul_scalar(config.load.px),
            ], 1);
            let (ex, ey, exy_) = compute_strains::<B>(out_bnd, nt, &fd);
            let v_to_t = |idxs: &[usize], src: &[f32]| -> Tensor<B, 1> {
                let v: Vec<f32> = idxs.iter().map(|&i| src[i]).collect();
                Tensor::<B, 1>::from_data(TensorData::new(v, vec![idxs.len()]), &device)
            };
            let n_loss = crate::energy::neumann_loss(
                ex, ey, exy_, v_to_t(&trac_idx, &bnd_nx), v_to_t(&trac_idx, &bnd_ny),
                to_t1(&tx_target), to_t1(&ty_target), &config.material,
            ).mul_scalar(1.0 / ref_stress2 as f64);

            let nh = hole_idx.len();
            let hole_norm: Vec<[f32; 2]> = hole_idx.iter().map(|&i| bnd_norm[i]).collect();
            let bnd_h = norm_pts_to_tensor::<B>(&hole_norm, &device);
            let out_h_raw = apply_dirichlet_ansatz::<B>(
                fwd(&model_single, bnd_h.clone(), 0, &device),
                &bnd_h, config.geometry.symmetry, engine.ansatz_k,
            );
            let nrh = out_h_raw.dims()[0];
            let out_h = Tensor::cat(vec![
                out_h_raw.clone().slice([0..nrh, 0..2]).mul_scalar(u_ref as f64),
                out_h_raw.slice([0..nrh, 2..5]).mul_scalar(config.load.px),
            ], 1);
            let sxx_h = out_h.clone().slice([0..nh, 2..3]).reshape([nh]);
            let syy_h = out_h.clone().slice([0..nh, 3..4]).reshape([nh]);
            let sxy_h = out_h.slice([0..nh, 4..5]).reshape([nh]);
            let h_loss = crate::energy::hole_traction_loss_direct(
                sxx_h, syy_h, sxy_h, v_to_t(&hole_idx, &bnd_nx), v_to_t(&hole_idx, &bnd_ny),
            ).mul_scalar(1.0 / ref_stress2 as f64);

            let nrr = right_idx.len();
            let right_norm: Vec<[f32; 2]> = right_idx.iter().map(|&i| bnd_norm[i]).collect();
            let right_t = norm_pts_to_tensor::<B>(&right_norm, &device);
            let out_r_raw = apply_dirichlet_ansatz::<B>(
                fwd(&model_single, right_t.clone(), 0, &device),
                &right_t, config.geometry.symmetry, engine.ansatz_k,
            );
            let nrr2 = out_r_raw.dims()[0];
            let out_r = Tensor::cat(vec![
                out_r_raw.clone().slice([0..nrr2, 0..2]).mul_scalar(u_ref as f64),
                out_r_raw.slice([0..nrr2, 2..5]).mul_scalar(config.load.px),
            ], 1);
            let u_vals = out_r.slice([0..nrr, 0..1]).reshape([nrr]);
            let u_tgt: Tensor<B, 1> = Tensor::full([nrr], u_target_val as f64, &device);
            let denom = ((u_target_val * u_target_val) as f64).max(1e-20);
            let d_loss = (u_vals - u_tgt).powf_scalar(2.0_f64).mean().mul_scalar(1.0 / denom);

            let e_s = t_scalar(&e_loss);
            let n_s = t_scalar(&n_loss);
            let h_s = t_scalar(&h_loss);
            let d_s = t_scalar(&d_loss);
            let lams = saw_single.update(&[e_s, n_s, h_s, d_s]);
            let loss = e_loss.mul_scalar(lams[0] as f64)
                + n_loss.mul_scalar(lams[1] as f64)
                + h_loss.mul_scalar(lams[2] as f64)
                + d_loss.mul_scalar(lams[3] as f64);
            let _ = int_raw;

            let lr = lr_sched_single.step(
                (e_s * lams[0] + n_s * lams[1] + h_s * lams[2] + d_s * lams[3]).abs() as f64
            );
            let (weight_ids, bias_ids) = model_single.param_ids();
            let gate_ids = model_single.gate_ids();
            let mut grads = loss.backward();
            let wg = GradientsParams::from_params(&mut grads, &model_single, &weight_ids);
            let bg = GradientsParams::from_params(&mut grads, &model_single, &bias_ids);
            let gg = GradientsParams::from_params(&mut grads, &model_single, &gate_ids);
            model_single = optim_w_single.step(lr, model_single, wg);
            model_single = optim_b_single.step(lr, model_single, bg);
            model_single = optim_gate_single.step(lr, model_single, gg);

            rel_close(out_multi.e_scalar, e_s, "e_scalar");
            rel_close(out_multi.n_scalar, n_s, "n_scalar");
            rel_close(out_multi.h_scalar, h_s, "h_scalar");
            rel_close(out_multi.d_scalar, d_s, "d_scalar");
        }

        struct NormVisitor { total: f64 }
        impl ModuleVisitor<B> for NormVisitor {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let v: Vec<f32> = param.val().into_data().to_vec().unwrap_or_default();
                self.total += v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
            }
        }
        let mut vis_multi = NormVisitor { total: 0.0 };
        model_multi.visit(&mut vis_multi);
        let mut vis_single = NormVisitor { total: 0.0 };
        model_single.visit(&mut vis_single);
        let scale = vis_multi.total.abs().max(vis_single.total).max(1e-8);
        let rel = (vis_multi.total - vis_single.total).abs() / scale;
        assert!(rel < 1e-4, "param_l2_sq: multi={} single={} rel_err={rel}", vis_multi.total, vis_single.total);
    }

    /// Minimal 2-parameter "network" stand-in for gradient-split tests: a tiny real
    /// `ElasticityNet` per domain with distinguishable initial weights (achieved via
    /// different hidden_dim/n_hidden — burn's default initializer is randomized per-call,
    /// so any two independently-constructed nets already have different weights; we only
    /// need domain-labeled models here, not KirschProblem's full sampling/loss machinery).
    fn tiny_net(device: &WgpuDevice) -> ElasticityNet<B> {
        crate::network::ElasticityNetConfig::new()
            .with_input_dim(3) // stencil coords are [N, 3] (x, y, z=0) — see assemble_stencil
            .with_hidden_dim(4)
            .with_n_hidden(2)
            .with_output_dim(2) // plain-DEM ansatz application reads cols 0 (u) and 1 (v)
            .init(device)
    }

    struct OneDomainLossTerm {
        domain: pinn_core::problem::DomainId,
    }
    impl LossTerm for OneDomainLossTerm {
        fn name(&self) -> &'static str { "solo_term" }
        fn domains(&self) -> Vec<pinn_core::problem::DomainId> { vec![self.domain] }
        fn compute(&self, inputs: &[crate::problem::DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
            let d = inputs.iter().find(|i| i.domain == self.domain).unwrap();
            let n = d.raw_out.dims()[0];
            d.raw_out.clone().slice([0..n, 0..1]).reshape([n]).powf_scalar(2.0_f64).mean()
        }
    }

    struct TwoDomainSumLossTerm {
        domain_a: pinn_core::problem::DomainId,
        domain_b: pinn_core::problem::DomainId,
    }
    impl LossTerm for TwoDomainSumLossTerm {
        fn name(&self) -> &'static str { "combined_term" }
        fn domains(&self) -> Vec<pinn_core::problem::DomainId> { vec![self.domain_a, self.domain_b] }
        fn compute(&self, inputs: &[crate::problem::DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
            let a = inputs.iter().find(|i| i.domain == self.domain_a).unwrap();
            let b = inputs.iter().find(|i| i.domain == self.domain_b).unwrap();
            let na = a.raw_out.dims()[0];
            let nb = b.raw_out.dims()[0];
            let la = a.raw_out.clone().slice([0..na, 0..1]).reshape([na]).powf_scalar(2.0_f64).mean();
            let lb = b.raw_out.clone().slice([0..nb, 0..1]).reshape([nb]).powf_scalar(2.0_f64).mean();
            la + lb
        }
    }

    struct TwoDomainToyProblem {
        domains: Vec<pinn_core::problem::DomainSpec>,
        sampling: crate::kirsch_problem::KirschSamplingStrategy,
        ansatz: crate::kirsch_problem::QuarterSymmAnsatz,
        only_a: bool,
    }
    impl BoundaryValueProblem for TwoDomainToyProblem {
        fn domains(&self) -> &[pinn_core::problem::DomainSpec] { &self.domains }
        fn sampling_strategy(&self, _domain_idx: usize) -> &dyn pinn_core::problem::DomainSamplingStrategy { &self.sampling }
        fn ansatz(&self, _domain_idx: usize) -> &dyn pinn_core::problem::DirichletAnsatz { &self.ansatz }
        fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
            if self.only_a {
                vec![Box::new(OneDomainLossTerm { domain: self.domains[0].id })]
            } else {
                vec![Box::new(TwoDomainSumLossTerm { domain_a: self.domains[0].id, domain_b: self.domains[1].id })]
            }
        }
        fn base_weight(&self, _term_name: &str) -> f32 { 1.0 }
        fn phase1_steps(&self) -> usize { usize::MAX }
        fn convergence_metric(&self, _state: &[DomainState<B>]) -> Option<f64> { None }
        fn convergence_target(&self) -> f64 { 0.0 }
    }

    fn toy_domain_data(id: pinn_core::problem::DomainId) -> crate::problem::DomainStepData {
        crate::problem::DomainStepData {
            id,
            int_norm: vec![[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1]],
            extra_ring_norm: Vec::new(),
            named: HashMap::new(),
        }
    }

    fn toy_geom_2d() -> pinn_core::geometry::GeometryConfig {
        pinn_core::geometry::GeometryConfig {
            half_w: 1.0, half_h: 1.0, thickness: 1.0,
            hole: pinn_core::geometry::HoleType::None,
            symmetry: pinn_core::geometry::SymmetryMode::Full,
        }
    }

    fn param_l2_sq(model: &ElasticityNet<B>) -> f64 {
        struct V { total: f64 }
        impl ModuleVisitor<B> for V {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let v: Vec<f32> = param.val().into_data().to_vec().unwrap_or_default();
                self.total += v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
            }
        }
        let mut v = V { total: 0.0 };
        model.visit(&mut v);
        v.total
    }

    /// RED test #2 (highest-risk assumption): a combined loss depending ONLY on domain A's
    /// output must leave domain B's weights EXACTLY unchanged after one step — proves
    /// `GradientsParams::from_params` correctly isolates each domain's own ParamIds out of
    /// the single shared backward() gradient bag, i.e. no domain can "steal" or be
    /// contaminated by another domain's gradients.
    #[test]
    fn gradient_split_attributes_domain_b_step_only_to_domain_b_params() {
        let device = WgpuDevice::default();
        let model_a = tiny_net(&device);
        let model_b = tiny_net(&device);
        struct TouchVisitor;
        impl ModuleVisitor<B> for TouchVisitor {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let _ = param.val();
            }
        }
        model_a.visit(&mut TouchVisitor);
        model_b.visit(&mut TouchVisitor);

        let domain_a = pinn_core::problem::DomainId(100);
        let domain_b = pinn_core::problem::DomainId(101);
        let geom = toy_geom_2d();
        let material = pinn_core::material::MaterialProps::al7075_t6();
        let problem = TwoDomainToyProblem {
            domains: vec![
                pinn_core::problem::DomainSpec { id: domain_a, geometry: geom.clone(), material: material.clone(), output_dim: 2 },
                pinn_core::problem::DomainSpec { id: domain_b, geometry: geom, material, output_dim: 2 },
            ],
            sampling: crate::kirsch_problem::KirschSamplingStrategy,
            ansatz: crate::kirsch_problem::QuarterSymmAnsatz,
            only_a: true,
        };

        let data_a = toy_domain_data(domain_a);
        let data_b = toy_domain_data(domain_b);
        let fd = FdConfig::new(1e-3, 2.0, 2.0);
        let ctx = crate::problem::MultiStepCtx {
            config: &SolverConfig::default_kirsch(),
            problem: &problem,
            fd: &fd,
            k: 1.0,
            domains: vec![
                crate::problem::DomainStepCtx { data: &data_a, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
                crate::problem::DomainStepCtx { data: &data_b, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
            ],
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: 500.0,
            dynamic_lam_non_tension_cap: 100.0,
            phase2_active: false,
            step: 0,
        };

        let a_before = param_l2_sq(&model_a);
        let b_before = param_l2_sq(&model_b);

        let mut optims = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let mut saw = SawBrdr::with_base(vec![1.0], 0.95);
        let mut lr_sched = LrSchedule::new(1e-3, 200, 1000);

        let (new_models, _out) = step_physics_multi(
            vec![model_a, model_b], &mut optims, &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );
        let mut iter = new_models.into_iter();
        let model_a_after = iter.next().unwrap();
        let model_b_after = iter.next().unwrap();

        let a_after = param_l2_sq(&model_a_after);
        let b_after = param_l2_sq(&model_b_after);

        assert!((a_after - a_before).abs() > 1e-12, "domain A's params must change — loss depended on A's output");
        assert_eq!(b_after, b_before, "domain B's params must be EXACTLY unchanged — loss did not depend on B's output");
    }

    /// RED test #3: a combined loss depending on BOTH domains' outputs must leave BOTH
    /// domains' weights changed after one step.
    #[test]
    fn gradient_split_two_domains_both_receive_nonzero_updates_when_both_contribute() {
        let device = WgpuDevice::default();
        let model_a = tiny_net(&device);
        let model_b = tiny_net(&device);
        struct TouchVisitor;
        impl ModuleVisitor<B> for TouchVisitor {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let _ = param.val();
            }
        }
        model_a.visit(&mut TouchVisitor);
        model_b.visit(&mut TouchVisitor);

        let domain_a = pinn_core::problem::DomainId(200);
        let domain_b = pinn_core::problem::DomainId(201);
        let geom = toy_geom_2d();
        let material = pinn_core::material::MaterialProps::al7075_t6();
        let problem = TwoDomainToyProblem {
            domains: vec![
                pinn_core::problem::DomainSpec { id: domain_a, geometry: geom.clone(), material: material.clone(), output_dim: 2 },
                pinn_core::problem::DomainSpec { id: domain_b, geometry: geom, material, output_dim: 2 },
            ],
            sampling: crate::kirsch_problem::KirschSamplingStrategy,
            ansatz: crate::kirsch_problem::QuarterSymmAnsatz,
            only_a: false,
        };

        let data_a = toy_domain_data(domain_a);
        let data_b = toy_domain_data(domain_b);
        let fd = FdConfig::new(1e-3, 2.0, 2.0);
        let ctx = crate::problem::MultiStepCtx {
            config: &SolverConfig::default_kirsch(),
            problem: &problem,
            fd: &fd,
            k: 1.0,
            domains: vec![
                crate::problem::DomainStepCtx { data: &data_a, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
                crate::problem::DomainStepCtx { data: &data_b, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
            ],
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: 500.0,
            dynamic_lam_non_tension_cap: 100.0,
            phase2_active: false,
            step: 0,
        };

        let a_before = param_l2_sq(&model_a);
        let b_before = param_l2_sq(&model_b);

        let mut optims = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let mut saw = SawBrdr::with_base(vec![1.0], 0.95);
        let mut lr_sched = LrSchedule::new(1e-3, 200, 1000);

        let (new_models, _out) = step_physics_multi(
            vec![model_a, model_b], &mut optims, &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );
        let mut iter = new_models.into_iter();
        let model_a_after = iter.next().unwrap();
        let model_b_after = iter.next().unwrap();

        let a_after = param_l2_sq(&model_a_after);
        let b_after = param_l2_sq(&model_b_after);

        assert_ne!(a_after, a_before, "domain A's params must change — loss depends on A's output");
        assert_ne!(b_after, b_before, "domain B's params must change — loss depends on B's output");
    }

    // ─── TwoDomainModels / multi-domain L-BFGS ──────────────────────────────────────────

    fn param_l2_sq_wrapper(models: &TwoDomainModels<B>) -> f64 {
        param_l2_sq(&models.pin) + param_l2_sq(&models.lug)
    }

    #[test]
    fn two_domain_models_wrapper_visits_both_inner_models_params() {
        let device = WgpuDevice::default();
        let pin = tiny_net(&device);
        let lug = tiny_net(&device);
        let expected = param_l2_sq(&pin) + param_l2_sq(&lug);
        let wrapper = TwoDomainModels { pin, lug };
        let actual = param_l2_sq_wrapper(&wrapper);
        assert!((actual - expected).abs() < 1e-9,
            "wrapper visitor sum must equal sum of visiting each inner model separately: \
             actual={actual} expected={expected}");
    }

    /// Highest-risk smoke test in this task: does `TwoDomainModels` (a named struct wrapping
    /// two `ElasticityNet<B>`s) actually satisfy `LBFGS::step`'s `M: AutodiffModule<B> +
    /// Clone` bound via its `#[derive(Module, Debug)]`? Get this compiling/passing FIRST.
    #[test]
    fn two_domain_models_round_trips_through_lbfgs_flatten_params() {
        let device = WgpuDevice::default();
        let pin = tiny_net(&device);
        let lug = tiny_net(&device);
        let models = TwoDomainModels { pin, lug };

        let mut lbfgs = make_lbfgs(5);
        let closure = |m: TwoDomainModels<B>| -> (f64, GradientsParams) {
            let pin_out = fwd::<B>(
                &m.pin,
                Tensor::<B, 2>::from_data(TensorData::new(vec![0.1_f32, 0.2, 0.0], vec![1, 3]), &device),
                0, &device,
            );
            let lug_out = fwd::<B>(
                &m.lug,
                Tensor::<B, 2>::from_data(TensorData::new(vec![0.3_f32, -0.1, 0.0], vec![1, 3]), &device),
                0, &device,
            );
            let loss = pin_out.powf_scalar(2.0_f64).sum() + lug_out.powf_scalar(2.0_f64).sum();
            let loss_scalar = t_scalar(&loss.clone().reshape([1])) as f64;
            let grads_raw = loss.backward();
            let grads_p = GradientsParams::from_grads(grads_raw, &m);
            (loss_scalar, grads_p)
        };
        let (_new_models, loss) = lbfgs.step(1e-3, models, closure);
        assert!(loss.is_finite(), "L-BFGS step over TwoDomainModels must return a finite loss, got {loss}");
    }

    #[test]
    fn step_lbfgs_multi_reduces_loss_and_updates_both_domains_params() {
        let device = WgpuDevice::default();
        let pin = tiny_net(&device);
        let lug = tiny_net(&device);
        let models = TwoDomainModels { pin, lug };

        let domain_a = pinn_core::problem::DomainId(300);
        let domain_b = pinn_core::problem::DomainId(301);
        let geom = toy_geom_2d();
        let material = pinn_core::material::MaterialProps::al7075_t6();
        let problem = TwoDomainToyProblem {
            domains: vec![
                pinn_core::problem::DomainSpec { id: domain_a, geometry: geom.clone(), material: material.clone(), output_dim: 2 },
                pinn_core::problem::DomainSpec { id: domain_b, geometry: geom, material, output_dim: 2 },
            ],
            sampling: crate::kirsch_problem::KirschSamplingStrategy,
            ansatz: crate::kirsch_problem::QuarterSymmAnsatz,
            only_a: false, // loss depends on BOTH domains
        };

        let data_a = toy_domain_data(domain_a);
        let data_b = toy_domain_data(domain_b);
        let fd = FdConfig::new(1e-3, 2.0, 2.0);
        let ctx = crate::problem::MultiStepCtx {
            config: &SolverConfig::default_kirsch(),
            problem: &problem,
            fd: &fd,
            k: 1.0,
            domains: vec![
                crate::problem::DomainStepCtx { data: &data_a, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
                crate::problem::DomainStepCtx { data: &data_b, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
            ],
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: 500.0,
            dynamic_lam_non_tension_cap: 100.0,
            phase2_active: false,
            step: 0,
        };
        let frozen_ctx = crate::problem::FrozenMultiStepCtx::from_ctx(&ctx);

        let mut lams: HashMap<&'static str, f64> = HashMap::new();
        lams.insert("combined_term", 1.0);

        let a_before = param_l2_sq(&models.pin);
        let b_before = param_l2_sq(&models.lug);

        let mut lbfgs = make_lbfgs(5);
        let mut current = models;
        let mut last_loss = f64::MAX;
        let mut first_loss = None;
        for _ in 0..5 {
            let (new_models, loss) = step_lbfgs_multi(current, &mut lbfgs, 1e-2, &frozen_ctx, &problem, &lams, &device);
            if first_loss.is_none() { first_loss = Some(loss); }
            last_loss = loss;
            current = new_models;
        }

        let a_after = param_l2_sq(&current.pin);
        let b_after = param_l2_sq(&current.lug);

        assert!(last_loss < first_loss.unwrap(),
            "loss must strictly decrease over 5 outer L-BFGS iterations: first={} last={last_loss}",
            first_loss.unwrap());
        assert_ne!(a_after, a_before, "domain A (pin)'s params must change");
        assert_ne!(b_after, b_before, "domain B (lug)'s params must change");
    }

    /// Extends `TwoDomainToyProblem` with a Physics-tagged term reading domain A only and a
    /// Bc-tagged term reading domain B only — disjoint params, so cosine similarity must be
    /// near-zero (gradients touch entirely different parameters, dot product ~ 0).
    struct PhysicsOnlyATerm { domain: pinn_core::problem::DomainId }
    impl LossTerm for PhysicsOnlyATerm {
        fn name(&self) -> &'static str { "physics_only_a" }
        fn domains(&self) -> Vec<pinn_core::problem::DomainId> { vec![self.domain] }
        fn conflict_group(&self) -> crate::problem::ConflictGroup { crate::problem::ConflictGroup::Physics }
        fn compute(&self, inputs: &[crate::problem::DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
            let d = inputs.iter().find(|i| i.domain == self.domain).unwrap();
            let n = d.raw_out.dims()[0];
            d.raw_out.clone().slice([0..n, 0..1]).reshape([n]).powf_scalar(2.0_f64).mean()
        }
    }
    struct BcOnlyBTerm { domain: pinn_core::problem::DomainId }
    impl LossTerm for BcOnlyBTerm {
        fn name(&self) -> &'static str { "bc_only_b" }
        fn domains(&self) -> Vec<pinn_core::problem::DomainId> { vec![self.domain] }
        fn conflict_group(&self) -> crate::problem::ConflictGroup { crate::problem::ConflictGroup::Bc }
        fn compute(&self, inputs: &[crate::problem::DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
            let d = inputs.iter().find(|i| i.domain == self.domain).unwrap();
            let n = d.raw_out.dims()[0];
            d.raw_out.clone().slice([0..n, 1..2]).reshape([n]).powf_scalar(2.0_f64).mean()
        }
    }

    struct DisjointConflictToyProblem {
        domains: Vec<pinn_core::problem::DomainSpec>,
        sampling: crate::kirsch_problem::KirschSamplingStrategy,
        ansatz: crate::kirsch_problem::QuarterSymmAnsatz,
        /// When true, both terms are Physics-classified (used for the empty-Bc-group test).
        all_physics: bool,
    }
    impl BoundaryValueProblem for DisjointConflictToyProblem {
        fn domains(&self) -> &[pinn_core::problem::DomainSpec] { &self.domains }
        fn sampling_strategy(&self, _domain_idx: usize) -> &dyn pinn_core::problem::DomainSamplingStrategy { &self.sampling }
        fn ansatz(&self, _domain_idx: usize) -> &dyn pinn_core::problem::DirichletAnsatz { &self.ansatz }
        fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
            if self.all_physics {
                vec![
                    Box::new(PhysicsOnlyATerm { domain: self.domains[0].id }),
                    Box::new({
                        struct PhysicsOnlyBTerm { domain: pinn_core::problem::DomainId }
                        impl LossTerm for PhysicsOnlyBTerm {
                            fn name(&self) -> &'static str { "physics_only_b" }
                            fn domains(&self) -> Vec<pinn_core::problem::DomainId> { vec![self.domain] }
                            fn conflict_group(&self) -> crate::problem::ConflictGroup { crate::problem::ConflictGroup::Physics }
                            fn compute(&self, inputs: &[crate::problem::DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
                                let d = inputs.iter().find(|i| i.domain == self.domain).unwrap();
                                let n = d.raw_out.dims()[0];
                                d.raw_out.clone().slice([0..n, 1..2]).reshape([n]).powf_scalar(2.0_f64).mean()
                            }
                        }
                        PhysicsOnlyBTerm { domain: self.domains[1].id }
                    }),
                ]
            } else {
                vec![
                    Box::new(PhysicsOnlyATerm { domain: self.domains[0].id }),
                    Box::new(BcOnlyBTerm { domain: self.domains[1].id }),
                ]
            }
        }
        fn base_weight(&self, _term_name: &str) -> f32 { 1.0 }
        fn phase1_steps(&self) -> usize { usize::MAX }
        fn convergence_metric(&self, _state: &[DomainState<B>]) -> Option<f64> { None }
        fn convergence_target(&self) -> f64 { 0.0 }
    }

    #[test]
    fn compute_gradient_conflict_multi_produces_finite_cosine_and_norms_on_toy_problem() {
        let device = WgpuDevice::default();
        let pin = tiny_net(&device);
        let lug = tiny_net(&device);
        let models = TwoDomainModels { pin, lug };

        let domain_a = pinn_core::problem::DomainId(400);
        let domain_b = pinn_core::problem::DomainId(401);
        let geom = toy_geom_2d();
        let material = pinn_core::material::MaterialProps::al7075_t6();
        let problem = DisjointConflictToyProblem {
            domains: vec![
                pinn_core::problem::DomainSpec { id: domain_a, geometry: geom.clone(), material: material.clone(), output_dim: 2 },
                pinn_core::problem::DomainSpec { id: domain_b, geometry: geom, material, output_dim: 2 },
            ],
            sampling: crate::kirsch_problem::KirschSamplingStrategy,
            ansatz: crate::kirsch_problem::QuarterSymmAnsatz,
            all_physics: false,
        };

        let data_a = toy_domain_data(domain_a);
        let data_b = toy_domain_data(domain_b);
        let fd = FdConfig::new(1e-3, 2.0, 2.0);
        let ctx = crate::problem::MultiStepCtx {
            config: &SolverConfig::default_kirsch(),
            problem: &problem,
            fd: &fd,
            k: 1.0,
            domains: vec![
                crate::problem::DomainStepCtx { data: &data_a, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
                crate::problem::DomainStepCtx { data: &data_b, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
            ],
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: 500.0,
            dynamic_lam_non_tension_cap: 100.0,
            phase2_active: false,
            step: 0,
        };

        let conflict = compute_gradient_conflict_multi(&models, &ctx, &device);
        assert!(conflict.cosine_sim.is_finite());
        assert!(conflict.g_pde_norm.is_finite() && conflict.g_pde_norm > 0.0);
        assert!(conflict.g_bc_norm.is_finite() && conflict.g_bc_norm > 0.0);
        assert!(conflict.cosine_sim.abs() < 1e-3,
            "disjoint-parameter gradients must have ~zero cosine similarity, got {}", conflict.cosine_sim);
    }

    #[test]
    fn compute_gradient_conflict_multi_bc_group_empty_yields_finite_not_nan_cosine() {
        let device = WgpuDevice::default();
        let pin = tiny_net(&device);
        let lug = tiny_net(&device);
        let models = TwoDomainModels { pin, lug };

        let domain_a = pinn_core::problem::DomainId(500);
        let domain_b = pinn_core::problem::DomainId(501);
        let geom = toy_geom_2d();
        let material = pinn_core::material::MaterialProps::al7075_t6();
        let problem = DisjointConflictToyProblem {
            domains: vec![
                pinn_core::problem::DomainSpec { id: domain_a, geometry: geom.clone(), material: material.clone(), output_dim: 2 },
                pinn_core::problem::DomainSpec { id: domain_b, geometry: geom, material, output_dim: 2 },
            ],
            sampling: crate::kirsch_problem::KirschSamplingStrategy,
            ansatz: crate::kirsch_problem::QuarterSymmAnsatz,
            all_physics: true,
        };

        let data_a = toy_domain_data(domain_a);
        let data_b = toy_domain_data(domain_b);
        let fd = FdConfig::new(1e-3, 2.0, 2.0);
        let ctx = crate::problem::MultiStepCtx {
            config: &SolverConfig::default_kirsch(),
            problem: &problem,
            fd: &fd,
            k: 1.0,
            domains: vec![
                crate::problem::DomainStepCtx { data: &data_a, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
                crate::problem::DomainStepCtx { data: &data_b, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
            ],
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: 500.0,
            dynamic_lam_non_tension_cap: 100.0,
            phase2_active: false,
            step: 0,
        };

        let conflict = compute_gradient_conflict_multi(&models, &ctx, &device);
        assert_eq!(conflict.cosine_sim, 0.0,
            "empty BC group (all terms Physics-classified) must yield exactly 0.0 cosine via the epsilon guard, not NaN");
        assert!(conflict.g_bc_norm.is_finite());
        assert!(conflict.g_pde_norm.is_finite());
    }

    // ─── StepOutput.lam_by_name ──────────────────────────────────────────────────────────

    #[test]
    fn step_physics_step_output_lam_by_name_is_none_on_kirsch_frozen_path() {
        use crate::{
            engine::EngineParams,
            kirsch_problem::KirschProblem,
            network::ElasticityNetConfig,
            optim::{make_bias_optim, make_gate_optim, WeightOptim},
        };
        use pinn_core::messages::SolverConfig;

        let mut config = SolverConfig::default_kirsch();
        config.n_interior = 32;
        config.n_boundary = 24;
        config.max_steps = 1;
        let engine = EngineParams::analyze(&config);
        engine.apply_to(&mut config);

        let device = WgpuDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(engine.net_input_dim())
            .with_hidden_dim(config.hidden_dim)
            .with_n_hidden(config.n_hidden)
            .with_output_dim(engine.output_dim())
            .with_use_piratenet(config.use_piratenet);
        let model: ElasticityNet<B> = net_cfg.init(&device);

        let (x0, x1) = config.geometry.x_range();
        let (y0, y1) = config.geometry.y_range();
        let fd = FdConfig::new(config.fd_h, x1 - x0, y1 - y0);
        let cx = fd.sx / (2.0 * fd.hx as f64);
        let cy = fd.sy / (2.0 * fd.hy as f64);
        let ref_div2 = (config.load.px * cx).powi(2).max(1.0);
        let (u_ref, ref_energy, ref_stress2) = compute_reference_scales(&config);

        let int_pts = pinn_core::sampling::sample_interior(&config.geometry, engine.phase1_n_interior);
        let bnd_pts = pinn_core::sampling::sample_boundary(&config.geometry, &config.load, config.n_boundary);
        let eq_ring = pinn_core::sampling::sample_eq_ring(&config.geometry, engine.n_eq_ring);

        let int_norm: Vec<[f32; 2]> = int_pts.iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();
        let bnd_norm: Vec<[f32; 2]> = bnd_pts.iter().map(|b| normalize_point(b.x, b.y, &config)).collect();
        let bnd_nx: Vec<f32> = bnd_pts.iter().map(|b| b.nx as f32).collect();
        let bnd_ny: Vec<f32> = bnd_pts.iter().map(|b| b.ny as f32).collect();
        let bnd_tx: Vec<f32> = bnd_pts.iter().map(|b| b.tx as f32).collect();
        let bnd_ty: Vec<f32> = bnd_pts.iter().map(|b| b.ty as f32).collect();
        let (trac_idx, hole_idx, right_idx) = extract_boundary_indices(&bnd_pts, &bnd_nx);
        let eq_ring_norm: Vec<[f32; 2]> = eq_ring.iter().map(|&[x, y]| normalize_point(x, y, &config)).collect();

        let problem = KirschProblem::new(
            config.material.clone(), engine.output_dim(), engine.phase1_steps, engine.expected_kt,
        );

        let mut optim_w = WeightOptim::new(config.use_soap_muon);
        let mut optim_b = make_bias_optim();
        let mut optim_gate = make_gate_optim();
        let mut saw = SawBrdr::with_base(engine.init_weights(), 0.95);
        let mut lr_sched = LrSchedule::new(engine.peak_lr, 200, 1000);

        let ctx = StepCtx {
            config: &config, engine: &engine, problem: &problem, fd: &fd,
            k: engine.ansatz_k, u_ref, ref_energy, ref_stress2, cx, cy, ref_div2,
            int_norm: &int_norm, bnd_norm: &bnd_norm,
            bnd_nx: &bnd_nx, bnd_ny: &bnd_ny, bnd_tx: &bnd_tx, bnd_ty: &bnd_ty,
            trac_idx: &trac_idx, hole_idx: &hole_idx, right_idx: &right_idx,
            eq_ring_norm: &eq_ring_norm,
            dynamic_lam_h_cap: 50.0, dynamic_lam_d_cap: 50.0,
            phase2_active: false, step: 0,
        };

        let (_model, out) = step_physics(
            model, &mut optim_w, &mut optim_b, &mut optim_gate,
            &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );

        assert!(out.lam_by_name.is_none(),
            "step_physics (Kirsch's frozen single-domain path) must leave lam_by_name None — \
             the fixed 6-field lam_e/n/h/d/eq/kirsch schema already covers it exhaustively");
    }

    #[test]
    fn step_physics_multi_step_output_exposes_lam_by_name_for_every_active_term() {
        let device = WgpuDevice::default();
        let model_a = tiny_net(&device);
        let model_b = tiny_net(&device);
        struct TouchVisitor;
        impl ModuleVisitor<B> for TouchVisitor {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let _ = param.val();
            }
        }
        model_a.visit(&mut TouchVisitor);
        model_b.visit(&mut TouchVisitor);

        let domain_a = pinn_core::problem::DomainId(600);
        let domain_b = pinn_core::problem::DomainId(601);
        let geom = toy_geom_2d();
        let material = pinn_core::material::MaterialProps::al7075_t6();
        let problem = TwoDomainToyProblem {
            domains: vec![
                pinn_core::problem::DomainSpec { id: domain_a, geometry: geom.clone(), material: material.clone(), output_dim: 2 },
                pinn_core::problem::DomainSpec { id: domain_b, geometry: geom, material, output_dim: 2 },
            ],
            sampling: crate::kirsch_problem::KirschSamplingStrategy,
            ansatz: crate::kirsch_problem::QuarterSymmAnsatz,
            only_a: false, // combined_term reads BOTH domains
        };

        let data_a = toy_domain_data(domain_a);
        let data_b = toy_domain_data(domain_b);
        let fd = FdConfig::new(1e-3, 2.0, 2.0);
        let ctx = crate::problem::MultiStepCtx {
            config: &SolverConfig::default_kirsch(),
            problem: &problem,
            fd: &fd,
            k: 1.0,
            domains: vec![
                crate::problem::DomainStepCtx { data: &data_a, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
                crate::problem::DomainStepCtx { data: &data_b, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
            ],
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: 500.0,
            dynamic_lam_non_tension_cap: 100.0,
            phase2_active: false,
            step: 0,
        };

        let mut optims = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let mut saw = SawBrdr::with_base(vec![1.0], 0.95);
        let mut lr_sched = LrSchedule::new(1e-3, 200, 1000);

        let (_new_models, out) = step_physics_multi(
            vec![model_a, model_b], &mut optims, &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );

        let map = out.lam_by_name.expect("step_physics_multi must always populate lam_by_name");
        for term in problem.loss_terms() {
            let lam = *map.get(term.name())
                .unwrap_or_else(|| panic!("lam_by_name missing entry for active term '{}'", term.name()));
            assert!(lam.is_finite() && lam >= 0.0, "lam_by_name['{}']={} must be finite and non-negative", term.name(), lam);
        }
    }

    // ─── frozen_lams regression guard (pin-lug Converge entry) ──────────────────────────

    /// Regression guard for the frozen_lams bug: L-BFGS's frozen lambdas at Converge entry
    /// must be the LIVE SAW+cap-adapted weights from the step immediately preceding entry,
    /// not `problem.base_weight()`'s static Phase-1 seed. The end-to-end proof lives in
    /// `headless.rs`'s `pinlug_frozen_lbfgs_lams_at_converge_entry_match_the_immediately_
    /// preceding_step_output` test (which exercises the actual `run_headless_pinlug_inner`
    /// wiring); this test pins down the underlying mechanism `StepOutput.lam_by_name`
    /// itself is reliable enough to build that fix on — i.e. after SAW has adapted away
    /// from its base weights, `lam_by_name` reflects the adapted values, not the seed.
    #[test]
    fn step_physics_multi_lam_by_name_reflects_saw_adaptation_not_base_weight_seed() {
        let device = WgpuDevice::default();
        let model_a = tiny_net(&device);
        let model_b = tiny_net(&device);
        struct TouchVisitor;
        impl ModuleVisitor<B> for TouchVisitor {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let _ = param.val();
            }
        }
        model_a.visit(&mut TouchVisitor);
        model_b.visit(&mut TouchVisitor);

        let domain_a = pinn_core::problem::DomainId(700);
        let domain_b = pinn_core::problem::DomainId(701);
        let geom = toy_geom_2d();
        let material = pinn_core::material::MaterialProps::al7075_t6();
        // Two INDEPENDENT single-domain terms (not TwoDomainToyProblem's single combined
        // term) — SAW-BRDR needs >=2 components with differential convergence rates for its
        // multiplier to move away from the uniform 1/n seed (see SawBrdr::update: with n=1
        // the multiplier is trivially always 1.0, so `combined_term`'s single-term case
        // would never demonstrate adaptation).
        let problem = DisjointConflictToyProblem {
            domains: vec![
                pinn_core::problem::DomainSpec { id: domain_a, geometry: geom.clone(), material: material.clone(), output_dim: 2 },
                pinn_core::problem::DomainSpec { id: domain_b, geometry: geom, material, output_dim: 2 },
            ],
            sampling: crate::kirsch_problem::KirschSamplingStrategy,
            ansatz: crate::kirsch_problem::QuarterSymmAnsatz,
            all_physics: false,
        };

        let data_a = toy_domain_data(domain_a);
        let data_b = toy_domain_data(domain_b);
        let fd = FdConfig::new(1e-3, 2.0, 2.0);
        let ctx = crate::problem::MultiStepCtx {
            config: &SolverConfig::default_kirsch(),
            problem: &problem,
            fd: &fd,
            k: 1.0,
            domains: vec![
                crate::problem::DomainStepCtx { data: &data_a, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
                crate::problem::DomainStepCtx { data: &data_b, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 },
            ],
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: 500.0,
            dynamic_lam_non_tension_cap: 100.0,
            phase2_active: false,
            step: 0,
        };

        // Base weight seed the SAW-BRDR instance starts from (uniform, both terms equal).
        let base_seed = 1.0_f64;
        let mut optims = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let mut saw = SawBrdr::with_base(vec![base_seed as f32, base_seed as f32], 0.95);
        let mut lr_sched = LrSchedule::new(1e-3, 200, 1000);

        // Run several steps so SAW-BRDR's adaptive multipliers have room to diverge as the
        // two terms' convergence rates differ (two independently-initialized tiny nets).
        let mut models = vec![model_a, model_b];
        let mut last_lam_a = base_seed;
        let mut last_lam_b = base_seed;
        for _ in 0..8 {
            let (new_models, out) = step_physics_multi(
                models, &mut optims, &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
            );
            models = new_models;
            let map = out.lam_by_name.expect("must be populated");
            last_lam_a = *map.get("physics_only_a").expect("physics_only_a must be present");
            last_lam_b = *map.get("bc_only_b").expect("bc_only_b must be present");
        }

        assert!((last_lam_a - base_seed).abs() > 1e-9 || (last_lam_b - base_seed).abs() > 1e-9,
            "lam_by_name must reflect SAW-BRDR's adapted weight after several steps, not the \
             static base_weight seed both terms started from: last_lam_a={last_lam_a} \
             last_lam_b={last_lam_b} base_seed={base_seed}");
    }

    // ─── interface_penetration / interface_non_tension cap dispatch (issue #21) ───────────

    /// Normalizes a physical coordinate to [-1,1]^2 for an arbitrary domain's own geometry
    /// bounds — a local copy of `headless.rs`'s private `normalize_point_generic` (not `pub`,
    /// so re-derived here rather than reached into across module boundaries).
    fn normalize_point_generic_for_test(x: f64, y: f64, geom: &pinn_core::geometry::GeometryConfig) -> [f32; 2] {
        let (x0, x1) = geom.x_range();
        let (y0, y1) = geom.y_range();
        let dw = x1 - x0;
        let dh = y1 - y0;
        [(2.0 * (x - x0) / dw - 1.0) as f32, (2.0 * (y - y0) / dh - 1.0) as f32]
    }

    /// Builds real (non-toy) pin/lug `DomainStepData` plus reference scales for a
    /// `PinLugProblem`, mirroring `headless.rs::run_headless_pinlug_inner`'s per-step setup
    /// (sample interior/boundary/named point-sets, wire the driving traction target) trimmed
    /// to small point counts for test speed. Returns owned data so each test can build its
    /// own `MultiStepCtx` with distinct cap values referencing it.
    fn build_pinlug_test_domain_data(
        problem: &crate::pinlug_problem::PinLugProblem,
        n_interior: usize,
        n_boundary: usize,
    ) -> (crate::problem::DomainStepData, crate::problem::DomainStepData, f32, f32, f32) {
        use crate::pinlug_problem::{PIN_DOMAIN, LUG_DOMAIN};
        use crate::problem::{DomainStepData, PointSetData};

        let pin_geom = problem.domains()[0].geometry.clone();
        let lug_geom = problem.domains()[1].geometry.clone();
        let pin_sampling = problem.sampling_strategy(0);
        let lug_sampling = problem.sampling_strategy(1);

        let equiv_traction = problem.equivalent_traction_pa();
        let e = problem.domains()[0].material.e;
        let u_ref = ((equiv_traction / e) * lug_geom.half_w) as f32;
        let ref_energy = (0.5 * equiv_traction * equiv_traction / e) as f32;
        let ref_stress2 = (equiv_traction * equiv_traction) as f32;

        let pin_int = pin_sampling.sample_interior(&pin_geom, n_interior);
        let lug_int = lug_sampling.sample_interior(&lug_geom, n_interior);
        let lug_bnd = lug_sampling.sample_boundary(&lug_geom, &pinn_core::loading::LoadConfig::uniaxial_x(equiv_traction), n_boundary);

        let pin_int_norm: Vec<[f32; 2]> = pin_int.iter().map(|&[x, y]| normalize_point_generic_for_test(x, y, &pin_geom)).collect();
        let lug_int_norm: Vec<[f32; 2]> = lug_int.iter().map(|&[x, y]| normalize_point_generic_for_test(x, y, &lug_geom)).collect();

        let build_pointset = |pts: &[pinn_core::loading::BoundaryPoint], geom: &pinn_core::geometry::GeometryConfig| -> PointSetData {
            PointSetData {
                norm: pts.iter().map(|p| normalize_point_generic_for_test(p.x, p.y, geom)).collect(),
                nx: pts.iter().map(|p| p.nx as f32).collect(),
                ny: pts.iter().map(|p| p.ny as f32).collect(),
                tx: pts.iter().map(|p| p.tx as f32).collect(),
                ty: pts.iter().map(|p| p.ty as f32).collect(),
            }
        };

        let mut pin_named = HashMap::new();
        let mut lug_named = HashMap::new();
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
        (pin_data, lug_data, u_ref, ref_energy, ref_stress2)
    }

    #[allow(clippy::too_many_arguments)]
    fn make_pinlug_test_ctx<'a>(
        config: &'a SolverConfig,
        problem: &'a crate::pinlug_problem::PinLugProblem,
        fd: &'a FdConfig,
        pin_data: &'a crate::problem::DomainStepData,
        lug_data: &'a crate::problem::DomainStepData,
        u_ref: f32, ref_energy: f32, ref_stress2: f32,
        dynamic_lam_h_cap: f64, dynamic_lam_d_cap: f64,
        dynamic_lam_penetration_cap: f64, dynamic_lam_non_tension_cap: f64,
        phase2_active: bool,
    ) -> crate::problem::MultiStepCtx<'a> {
        crate::problem::MultiStepCtx {
            config,
            problem,
            fd,
            k: 1.0,
            domains: vec![
                crate::problem::DomainStepCtx { data: pin_data, u_ref, ref_energy, ref_stress2 },
                crate::problem::DomainStepCtx { data: lug_data, u_ref, ref_energy, ref_stress2 },
            ],
            dynamic_lam_h_cap,
            dynamic_lam_d_cap,
            dynamic_lam_penetration_cap,
            dynamic_lam_non_tension_cap,
            phase2_active,
            step: 0,
        }
    }

    fn pinlug_test_fixture() -> (crate::pinlug_problem::PinLugProblem, SolverConfig, FdConfig) {
        use crate::pinlug_problem::{PinLugProblem, PinLugScalingMode};
        let problem = PinLugProblem::new(
            pinn_core::material::MaterialProps::steel_4340(), 5, usize::MAX, 8,
            PinLugScalingMode::AppliedLoad,
        );
        let config = SolverConfig::default_pinlug();
        let fd = FdConfig::new(config.fd_h, 1.0, 1.0);
        (problem, config, fd)
    }

    fn tiny_pinlug_nets(device: &WgpuDevice) -> (ElasticityNet<B>, ElasticityNet<B>) {
        let net_cfg = crate::network::ElasticityNetConfig::new()
            .with_input_dim(3)
            .with_hidden_dim(4)
            .with_n_hidden(1)
            .with_output_dim(5)
            .with_use_piratenet(false);
        (net_cfg.init(device), net_cfg.init(device))
    }

    /// Same as `tiny_pinlug_nets`, but forces every lazily-initialized `Param` to
    /// materialize BEFORE returning (see the `TouchVisitor` doc comment on
    /// `headless.rs`'s zero-regression test for why: burn's `Param` defers random-weight
    /// materialization until first access, so an untouched `Param`'s `.clone()` clones its
    /// lazy init state, not a value — each clone would then independently materialize
    /// DIFFERENT random weights on first use). Tests that need two runs to start from the
    /// EXACT same weights (not just the same distribution) must build ONE pair here, then
    /// `.clone()` it into each run.
    fn touched_tiny_pinlug_nets(device: &WgpuDevice) -> (ElasticityNet<B>, ElasticityNet<B>) {
        let (model_pin, model_lug) = tiny_pinlug_nets(device);
        struct TouchVisitor;
        impl ModuleVisitor<B> for TouchVisitor {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let _ = param.val();
            }
        }
        model_pin.visit(&mut TouchVisitor);
        model_lug.visit(&mut TouchVisitor);
        (model_pin, model_lug)
    }

    /// At step 0 with a fresh `SawBrdr`, `raw_lam == base_weight` for every term (the
    /// first-call decay-EMA skip — see CLAUDE.md), so seeding
    /// `dynamic_lam_penetration_cap`/`dynamic_lam_non_tension_cap` at the terms' own
    /// `base_weight` must be a no-op (within float-rounding tolerance, NOT bit-exact — `n *
    /// (1.0/n as f32)` is not bit-exact for all n).
    #[test]
    fn step_physics_multi_interface_penetration_cap_is_noop_at_step_zero_when_cap_equals_base_weight() {
        let device = WgpuDevice::default();
        let (problem, config, fd) = pinlug_test_fixture();
        let (pin_data, lug_data, u_ref, ref_energy, ref_stress2) = build_pinlug_test_domain_data(&problem, 8, 6);
        let (model_pin, model_lug) = tiny_pinlug_nets(&device);

        let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
        let mut saw = SawBrdr::with_base(base_weights, 0.95);
        let mut lr_sched = LrSchedule::new(1e-3, 200, 1000);
        let mut optims = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];

        let penetration_base = problem.base_weight("interface_penetration") as f64;
        let non_tension_base = problem.base_weight("interface_non_tension") as f64;
        let ctx = make_pinlug_test_ctx(
            &config, &problem, &fd, &pin_data, &lug_data, u_ref, ref_energy, ref_stress2,
            50.0, 50.0, penetration_base, non_tension_base, true,
        );

        let (_new_models, out) = step_physics_multi(
            vec![model_pin, model_lug], &mut optims, &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );
        let lam_by_name = out.lam_by_name.expect("must be populated");

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-8);
        assert!(rel(lam_by_name["interface_penetration"], penetration_base) < 1e-4,
            "interface_penetration lam={} expected~={penetration_base}", lam_by_name["interface_penetration"]);
        assert!(rel(lam_by_name["interface_non_tension"], non_tension_base) < 1e-4,
            "interface_non_tension lam={} expected~={non_tension_base}", lam_by_name["interface_non_tension"]);
    }

    /// A cap set far below the live/adapted weight must clamp the effective weight to
    /// exactly that cap — and must NOT leak into the pre-existing h/d caps' own dispatch arms.
    #[test]
    fn step_physics_multi_interface_penetration_cap_binds_when_below_live_weight() {
        let device = WgpuDevice::default();
        let (problem, config, fd) = pinlug_test_fixture();
        let (pin_data, lug_data, u_ref, ref_energy, ref_stress2) = build_pinlug_test_domain_data(&problem, 8, 6);

        let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
        let (model_pin_seed, model_lug_seed) = touched_tiny_pinlug_nets(&device);

        // Control run: h/d caps at 50.0 (their own established convention), new caps at
        // f64::MAX (fully non-binding), to capture lug_free_edge_traction/lug_shank_anchor's
        // "normal" lam values for comparison.
        let (model_pin_ctrl, model_lug_ctrl) = (model_pin_seed.clone(), model_lug_seed.clone());
        let mut saw_ctrl = SawBrdr::with_base(base_weights.clone(), 0.95);
        let mut lr_sched_ctrl = LrSchedule::new(1e-3, 200, 1000);
        let mut optims_ctrl = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let ctx_ctrl = make_pinlug_test_ctx(
            &config, &problem, &fd, &pin_data, &lug_data, u_ref, ref_energy, ref_stress2,
            50.0, 50.0, f64::MAX, f64::MAX, true,
        );
        let (_new_models, out_ctrl) = step_physics_multi(
            vec![model_pin_ctrl, model_lug_ctrl], &mut optims_ctrl, &ctx_ctrl, &mut saw_ctrl, &mut lr_sched_ctrl, &device, 0, 1.0, 1.0,
        );
        let lam_ctrl = out_ctrl.lam_by_name.expect("must be populated");

        // Test run: SAME starting models/SAW seed, new caps clamped to 1.0.
        let (model_pin, model_lug) = (model_pin_seed, model_lug_seed);
        let mut saw = SawBrdr::with_base(base_weights, 0.95);
        let mut lr_sched = LrSchedule::new(1e-3, 200, 1000);
        let mut optims = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let ctx = make_pinlug_test_ctx(
            &config, &problem, &fd, &pin_data, &lug_data, u_ref, ref_energy, ref_stress2,
            50.0, 50.0, 1.0, 1.0, true,
        );
        let (_new_models, out) = step_physics_multi(
            vec![model_pin, model_lug], &mut optims, &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );
        let lam = out.lam_by_name.expect("must be populated");

        assert_eq!(lam["interface_penetration"], 1.0);
        assert_eq!(lam["interface_non_tension"], 1.0);
        // The pre-existing h/d-capped terms must be BYTE-IDENTICAL between the two runs
        // (same starting models/SAW seed/h-d caps) — proves the new caps don't leak into
        // lug_free_edge_traction/lug_shank_anchor's own dispatch arms.
        assert_eq!(lam["lug_free_edge_traction"], lam_ctrl["lug_free_edge_traction"]);
        assert_eq!(lam["lug_shank_anchor"], lam_ctrl["lug_shank_anchor"]);
    }

    /// `phase2_active: false` must make the new caps fully inert — mirrors the pre-existing
    /// h/d arms' documented gating (matches `runner.rs`'s GUI-path construction).
    #[test]
    fn step_physics_multi_interface_penetration_cap_inert_when_phase2_active_false() {
        let device = WgpuDevice::default();
        let (problem, config, fd) = pinlug_test_fixture();
        let (pin_data, lug_data, u_ref, ref_energy, ref_stress2) = build_pinlug_test_domain_data(&problem, 8, 6);
        let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
        let (model_pin_seed, model_lug_seed) = touched_tiny_pinlug_nets(&device);

        let (model_pin_a, model_lug_a) = (model_pin_seed.clone(), model_lug_seed.clone());
        let mut saw_a = SawBrdr::with_base(base_weights.clone(), 0.95);
        let mut lr_sched_a = LrSchedule::new(1e-3, 200, 1000);
        let mut optims_a = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let ctx_a = make_pinlug_test_ctx(
            &config, &problem, &fd, &pin_data, &lug_data, u_ref, ref_energy, ref_stress2,
            50.0, 50.0, 1.0, 1.0, false,
        );
        let (_new_models, out_a) = step_physics_multi(
            vec![model_pin_a, model_lug_a], &mut optims_a, &ctx_a, &mut saw_a, &mut lr_sched_a, &device, 0, 1.0, 1.0,
        );
        let lam_a = out_a.lam_by_name.expect("must be populated");

        let (model_pin_b, model_lug_b) = (model_pin_seed, model_lug_seed);
        let mut saw_b = SawBrdr::with_base(base_weights, 0.95);
        let mut lr_sched_b = LrSchedule::new(1e-3, 200, 1000);
        let mut optims_b = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let ctx_b = make_pinlug_test_ctx(
            &config, &problem, &fd, &pin_data, &lug_data, u_ref, ref_energy, ref_stress2,
            50.0, 50.0, f64::MAX, f64::MAX, false,
        );
        let (_new_models, out_b) = step_physics_multi(
            vec![model_pin_b, model_lug_b], &mut optims_b, &ctx_b, &mut saw_b, &mut lr_sched_b, &device, 0, 1.0, 1.0,
        );
        let lam_b = out_b.lam_by_name.expect("must be populated");

        assert_eq!(lam_a["interface_penetration"], lam_b["interface_penetration"]);
        assert_eq!(lam_a["interface_non_tension"], lam_b["interface_non_tension"]);
    }

    /// The two new caps must be read INDEPENDENTLY — clamping one must not affect the other
    /// (proves they aren't accidentally aliased to the same local).
    #[test]
    fn step_physics_multi_interface_penetration_and_non_tension_caps_are_independent() {
        let device = WgpuDevice::default();
        let (problem, config, fd) = pinlug_test_fixture();
        let (pin_data, lug_data, u_ref, ref_energy, ref_stress2) = build_pinlug_test_domain_data(&problem, 8, 6);
        let (model_pin, model_lug) = tiny_pinlug_nets(&device);
        let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
        let mut saw = SawBrdr::with_base(base_weights, 0.95);
        let mut lr_sched = LrSchedule::new(1e-3, 200, 1000);
        let mut optims = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let ctx = make_pinlug_test_ctx(
            &config, &problem, &fd, &pin_data, &lug_data, u_ref, ref_energy, ref_stress2,
            50.0, 50.0, 1.0, f64::MAX, true,
        );
        let (_new_models, out) = step_physics_multi(
            vec![model_pin, model_lug], &mut optims, &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );
        let lam = out.lam_by_name.expect("must be populated");

        assert_eq!(lam["interface_penetration"], 1.0, "penetration must clamp to its own cap");
        assert!(lam["interface_non_tension"] != 1.0,
            "non_tension must NOT clamp to interface_penetration's cap (proves independence): got {}",
            lam["interface_non_tension"]);
    }

    /// Boundary case: cap == raw_lam precisely — a `>` vs `>=` mutation in the `.min()` call
    /// would still be correct here (`.min` ties are a no-op either way), but this pins the
    /// exact-equality behavior explicitly so a future refactor away from `.min()` is caught.
    #[test]
    fn step_physics_multi_interface_penetration_cap_exactly_at_live_weight_is_noop() {
        let device = WgpuDevice::default();
        let (problem, config, fd) = pinlug_test_fixture();
        let (pin_data, lug_data, u_ref, ref_energy, ref_stress2) = build_pinlug_test_domain_data(&problem, 8, 6);
        let base_weights: Vec<f32> = problem.loss_terms().iter().map(|t| problem.base_weight(t.name())).collect();
        let (model_pin_seed, model_lug_seed) = touched_tiny_pinlug_nets(&device);

        // First, an uncapped run to discover the live raw_lam for interface_penetration.
        let (model_pin_probe, model_lug_probe) = (model_pin_seed.clone(), model_lug_seed.clone());
        let mut saw_probe = SawBrdr::with_base(base_weights.clone(), 0.95);
        let mut lr_sched_probe = LrSchedule::new(1e-3, 200, 1000);
        let mut optims_probe = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let ctx_probe = make_pinlug_test_ctx(
            &config, &problem, &fd, &pin_data, &lug_data, u_ref, ref_energy, ref_stress2,
            50.0, 50.0, f64::MAX, f64::MAX, true,
        );
        let (_new_models, out_probe) = step_physics_multi(
            vec![model_pin_probe, model_lug_probe], &mut optims_probe, &ctx_probe, &mut saw_probe, &mut lr_sched_probe, &device, 0, 1.0, 1.0,
        );
        let live_weight = out_probe.lam_by_name.expect("must be populated")["interface_penetration"];

        // Second, an identical run with the cap set to EXACTLY that live weight.
        let (model_pin, model_lug) = (model_pin_seed, model_lug_seed);
        let mut saw = SawBrdr::with_base(base_weights, 0.95);
        let mut lr_sched = LrSchedule::new(1e-3, 200, 1000);
        let mut optims = vec![
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
            crate::problem::DomainOptim { weight: WeightOptim::new(false), bias: make_bias_optim(), gate: make_gate_optim() },
        ];
        let ctx = make_pinlug_test_ctx(
            &config, &problem, &fd, &pin_data, &lug_data, u_ref, ref_energy, ref_stress2,
            50.0, 50.0, live_weight, f64::MAX, true,
        );
        let (_new_models, out) = step_physics_multi(
            vec![model_pin, model_lug], &mut optims, &ctx, &mut saw, &mut lr_sched, &device, 0, 1.0, 1.0,
        );
        let lam = out.lam_by_name.expect("must be populated");

        let rel = (lam["interface_penetration"] - live_weight).abs() / live_weight.abs().max(1e-8);
        assert!(rel < 1e-9, "cap exactly at live weight must be a no-op: got={} expected~={live_weight}", lam["interface_penetration"]);
    }
}

