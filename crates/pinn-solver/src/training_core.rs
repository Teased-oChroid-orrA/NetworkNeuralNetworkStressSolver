/// Shared training step — single source of truth for the per-step physics computation.
///
/// Both `runner` (GUI, channel output) and `headless` (terminal, stdout output) call
/// `step_physics` for the loss computation + backward pass, differing only in how they
/// consume the returned `StepOutput` (send via channel vs. print to stdout).
///
/// Also provides `probe_kt_shared` and `normalize_point` which were duplicated across both
/// callers.

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
    lr_schedule::LrSchedule,
    network::{fwd, ElasticityNet},
    optim::{BiasOptim, GateOptim, WeightOptim},
    saw_brdr::SawBrdr,
};

pub(crate) type B = Autodiff<Wgpu>;
pub(crate) type BInner = Wgpu;

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

/// sin²θ-weighted MSE between predicted and target Kirsch stress components, normalized
/// by `Px²` so the loss is O(1). Shared by the mDEM and plain-DEM Kirsch loss branches —
/// they differ only in how `(sxx_pr, syy_pr, sxy_pr)` were obtained, not in this aggregation.
fn kirsch_weighted_loss(
    sxx_pr: Tensor<B, 1>, syy_pr: Tensor<B, 1>, sxy_pr: Tensor<B, 1>,
    probes: &KirschProbes,
    px2: f64,
    device: &WgpuDevice,
) -> Tensor<B, 1> {
    let to_t1 = |v: &[f32]| -> Tensor<B, 1> {
        Tensor::from_data(TensorData::new(v.to_vec(), vec![v.len()]), device)
    };
    let loss_per_pt =
        (sxx_pr - to_t1(&probes.sxx_targets)).powf_scalar(2.0_f64)
        + (syy_pr - to_t1(&probes.syy_targets)).powf_scalar(2.0_f64)
        + (sxy_pr - to_t1(&probes.sxy_targets)).powf_scalar(2.0_f64).mul_scalar(2.0_f64);
    (loss_per_pt * to_t1(&probes.weights)).sum().mul_scalar(1.0 / px2)
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
/// SAW component counts:
/// - Phase 1: 5 components [e, n, h, d, eq]
/// - Phase 2: 6 components [e, n, h, d, eq, kirsch] — SAW replaced by caller at transition
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

    // === Interior energy loss ===
    let pts_t = norm_pts_to_tensor::<B>(ctx.int_norm, device);
    let stencil_coords = assemble_stencil::<B>(&pts_t, ctx.fd, device);
    let stencil_out = scale_out(apply_dirichlet_ansatz::<B>(
        fwd(&model, stencil_coords.clone(), n_fourier, device),
        &stencil_coords,
        ctx.config.geometry.symmetry,
        ctx.k,
    ));

    // Extract σ_net from center block BEFORE compute_strains consumes stencil_out.
    // compute_strains only reads cols 0,1 (u,v) — cols 2..5 are ignored — but it moves the tensor.
    let (sxx_net_int, syy_net_int, sxy_net_int) = if use_mdem {
        let (sx, sy, sxy) = extract_mdem_stress(&stencil_out, 0, n_int);
        (Some(sx), Some(sy), Some(sxy))
    } else {
        (None, None, None)
    };

    let (eps_xx, eps_yy, eps_xy) = compute_strains::<B>(stencil_out, n_int, ctx.fd);
    let e_loss = dem_energy_loss(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), &ctx.config.material)
        .mul_scalar(1.0 / ctx.ref_energy as f64);

    // === Constitutive consistency loss (mDEM only) ===
    let const_loss: Tensor<B, 1> = if let (Some(sxx_n), Some(syy_n), Some(sxy_n)) =
        (sxx_net_int, syy_net_int, sxy_net_int)
    {
        // eps_xx, eyy, exy are dimensionless; C:eps has units [Pa] same as σ_net (scaled by Px)
        constitutive_consistency_loss(
            sxx_n, syy_n, sxy_n,
            eps_xx, eps_yy, eps_xy,
            &ctx.config.material,
        ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
    } else {
        Tensor::<B, 1>::zeros([1], device)
    };

    // === Neumann traction loss ===
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
            let (sxx_h, syy_h, sxy_h) = extract_mdem_stress(&out_h, 0, nh);
            hole_traction_loss_direct(
                sxx_h, syy_h, sxy_h,
                v_to_t(ctx.hole_idx, ctx.bnd_nx),
                v_to_t(ctx.hole_idx, ctx.bnd_ny),
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
                v_to_t(ctx.hole_idx, ctx.bnd_nx),
                v_to_t(ctx.hole_idx, ctx.bnd_ny),
                &ctx.config.material,
            ).mul_scalar(1.0 / ctx.ref_stress2 as f64)
        }
    } else {
        Tensor::<B, 1>::zeros([1], device)
    };

    // === Equilibrium residual ∇·σ = 0 at near-hole ring points ===
    let n_eq = ctx.eq_ring_norm.len();
    let eq_loss: Tensor<B, 1> = if n_eq > 0 {
        if use_mdem {
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
            equilibrium_residual_loss(
                sxx_xp, sxy_xp, sxx_xm, sxy_xm,
                sxy_yp, syy_yp, sxy_ym, syy_ym,
                ctx.cx, ctx.cy, ctx.ref_div2,
            )
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
            equilibrium_residual_loss(
                sxx_xp, sxy_xp, sxx_xm, sxy_xm,
                sxy_yp, syy_yp, sxy_ym, syy_ym,
                ctx.cx, ctx.cy, ctx.ref_div2,
            )
        }
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

                let kl = if use_mdem {
                    // Direct pass: σ from network cols 2..5 (already in Pa after scale_out)
                    let out_pr = scale_out(apply_dirichlet_ansatz::<B>(
                        fwd(&model, pts_pr.clone(), n_fourier, device), &pts_pr,
                        ctx.config.geometry.symmetry, ctx.k,
                    ));
                    let (sxx_pr, syy_pr, sxy_pr) = extract_mdem_stress(&out_pr, 0, n_pr);
                    kirsch_weighted_loss(sxx_pr, syy_pr, sxy_pr, &probes, px2, device)
                } else {
                    let stencil_pr = assemble_stencil::<B>(&pts_pr, ctx.fd, device);
                    let out_pr = scale_out(apply_dirichlet_ansatz::<B>(
                        fwd(&model, stencil_pr.clone(), n_fourier, device), &stencil_pr,
                        ctx.config.geometry.symmetry, ctx.k,
                    ));
                    let (exx_pr, eyy_pr, exy_pr) = compute_strains::<B>(out_pr, n_pr, ctx.fd);
                    let (sxx_pr, syy_pr, sxy_pr) = compute_stress(
                        exx_pr, eyy_pr, exy_pr, &ctx.config.material,
                    );
                    kirsch_weighted_loss(sxx_pr, syy_pr, sxy_pr, &probes, px2, device)
                };
                let ks = t_scalar(&kl);
                (kl, ks)
            } else {
                (Tensor::<B, 1>::zeros([1], device), 0.0)
            }
        } else {
            (Tensor::<B, 1>::zeros([1], device), 0.0)
        };

    // === SAW-BRDR adaptive weight update ===
    // Phase 1: 5 components [e, n, h, d, eq].
    // Phase 2: 6 components [e, n, h, d, eq, kirsch] — SAW replaced at transition by caller.
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

    // Stiffness-coupled boost (external modulation, applied after SAW-BRDR itself —
    // mirrors how dynamic_lam_h_cap/dynamic_lam_d_cap are clamped below, outside SawBrdr).
    // `physics_boost = 1.0` (StiffnessController disabled) is a no-op.
    let lam_e   = lams[0] as f64 * physics_boost;
    let lam_n   = lams[1] as f64;
    let lam_h_  = lams[2] as f64;
    let lam_d_  = lams[3] as f64;
    let lam_eq  = lams[4] as f64 * physics_boost;
    // Phase 2: cap lam_h so kirsch gradient dominates hole traction. Tightens on plateau.
    let lam_h   = if ctx.phase2_active { lam_h_.min(ctx.dynamic_lam_h_cap) } else { lam_h_ };
    // Phase 2: cap lam_d so displacement-anchor gradient doesn't overwhelm kirsch at near-convergence.
    // Uncapped lam_d grows to ~46-50 while lam_k ≈ 2 → 23:1 ratio causes Adam overshoot when K_t → 3.
    let lam_d   = if ctx.phase2_active { lam_d_.min(ctx.dynamic_lam_d_cap) } else { lam_d_ };
    let lam_kirsch = if ctx.phase2_active { lams[5] as f64 } else { 0.0 };
    let lam_const  = ctx.engine.lam_const as f64 * physics_boost;

    // Total potential energy: Π = (e_loss − W_neumann)·λ_e + BC penalties + kirsch + const
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
}
