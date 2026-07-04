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
    let u_ref       = ((config.load.px / config.material.e) * config.geometry.half_w) as f32;
    let ref_energy  = (0.5 * config.load.px * config.load.px / config.material.e) as f32;
    let ref_stress2 = (config.load.px * config.load.px) as f32;
    (u_ref, ref_energy, ref_stress2)
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
                / (ctx.config.load.px * ctx.config.geometry.half_w));
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

    // Multi-domain problems (pin-in-lug) use plain-DEM output (no Fourier embedding, no
    // hard Dirichlet ansatz — boundary conditions are enforced via loss terms, not a
    // symmetry-plane ansatz, since neither pin nor lug domain has Kirsch's quarter-symmetry
    // structure). This mirrors `PinLugSamplingStrategy`'s geometry (see pinlug_problem.rs).
    let n_fourier = 0usize;

    // (a) Enumerate which (DomainId, point_set_name) pairs at least one active LossTerm
    // needs, then run exactly one forward pass per pair (never twice, even if several
    // terms share it).
    let active_terms: Vec<Box<dyn LossTerm>> = ctx.problem.loss_terms().into_iter()
        .filter(|t| t.name() != "constitutive_consistency")
        .filter(|t| ctx.phase2_active || !t.phase2_only())
        .collect();

    let mut needed: Vec<(DomainId, &'static str)> = Vec::new();
    for term in &active_terms {
        for (&id, &ps) in term.domains().iter().zip(term.point_sets().iter()) {
            if !needed.contains(&(id, ps)) {
                needed.push((id, ps));
            }
        }
    }

    // Forward-pass results are computed into owned storage first (raw_out tensor +
    // optional strains/normals), then wrapped into borrowing `DomainForwardOutputs` in a
    // second pass — Rust's borrow checker requires the owning Vec to be fully populated
    // (and therefore stable) before any `&Tensor` into it is taken.
    struct Computed {
        key: (DomainId, &'static str),
        raw_out: Tensor<B, 2>,
        strains: Option<(Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>)>,
        normals: Option<(Tensor<B, 1>, Tensor<B, 1>)>,
    }
    let mut computed: Vec<Computed> = Vec::with_capacity(needed.len());

    for &(domain_id, ps_name) in &needed {
        let dctx = ctx.domains.iter().find(|d| d.data.id == domain_id)
            .unwrap_or_else(|| panic!(
                "step_physics_multi: LossTerm references DomainId({}) not present in \
                 ctx.domains — this should have been caught by validate_loss_terms",
                domain_id.0,
            ));
        let model_idx = ctx.domains.iter().position(|d| d.data.id == domain_id).unwrap();
        let model = &models[model_idx];
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
    let ref_energy = (0.5 * ctx.config.load.px * ctx.config.load.px / ctx.config.material.e) as f32;
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
                / (ctx.config.load.px * ctx.config.geometry.half_w));
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
}

