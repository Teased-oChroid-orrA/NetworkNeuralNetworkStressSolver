//! Parametric PINN training: `enhancement.txt` items 7/8/17 - "train once across a range of
//! `(E, nu, Px)`, then infer a new solution instantly at any point in that range, with an
//! honest validity check instead of a silent guess." See `pinn_core::parametric_spec`'s
//! module doc for the exact v1 scope (only material.e/nu and load.px are parametric;
//! geometry is fixed).
//!
//! Deliberately self-contained, NOT built on `step_physics_multi`/`compute_domain_forwards`/
//! the `LossTerm`/`BoundaryValueProblem` trait machinery `UserDefinedProblem` uses - those
//! assume ONE fixed material/load baked into `DomainSpec` at construction time, with no path
//! for a per-step-varying, network-INPUT-conditioning value (the whole point of a parametric
//! PINN: the material/load must be part of what the network actually SEES, not just what the
//! loss is computed against). Reaching that would mean modifying `compute_domain_forwards`
//! itself - a large, heavily-tested function shared by Kirsch/pin-lug/`UserDefinedProblem` -
//! for the sake of one new caller. Per this project's own established discipline (a new
//! sibling instead of touching working, tested code), this module instead reuses only the
//! PURE, already-tested primitives that don't care where their inputs come from:
//! `fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor}`, every function in
//! `energy.rs`, `SawBrdr`, `LrSchedule`, `WeightOptim`/`BiasOptim`/`GateOptim`, and
//! `UserSamplingStrategy` (geometry is fixed, so its `(x, y)` point generation is unchanged).
//!
//! ## What "parametric" actually means here
//!
//! `ElasticityNet`'s input grows from 3 columns (`x, y, z=0`) to 6 (`x, y, z=0, e_n, nu_n,
//! p_n` - the trained `(E, nu, Px)` sample for this forward pass, each linearly mapped to
//! `[-1, 1]` via `ParamRange::normalize`). Every forward pass this module makes appends the
//! SAME triple to every row (the parameter doesn't vary spatially within one step - see
//! `tile_params`). One (E, nu, Px) sample is drawn PER STEP (not per point) via `LcgRng`,
//! deterministic and seeded by the step index - a real, stated simplification versus sampling
//! an independent triple per point, chosen because it keeps every point-set's loss term
//! computation byte-identical to `UserDefinedProblem`'s existing formulas (just fed a
//! step-local `MaterialProps`/`LoadConfig` instead of a spec-fixed one), rather than requiring
//! every `energy.rs` function to become per-point-material-aware. Over many steps the network
//! still sees the full parameter range, just one point in it per step rather than a full
//! per-point sweep every step.
//!
//! ## Instant inference after training
//!
//! There is no model checkpoint save/load anywhere in this codebase (see
//! `pinn_core::inference_envelope`'s doc comment) - so "instant inference" is implemented by
//! keeping the training thread ALIVE after `training.max_steps`, blocked on
//! `ControlMsg::ParametricInfer`/`Stop` (see `run_training_parametric`'s tail loop). The model
//! never leaves that thread; each inference request is answered over the same channel pair
//! the training loop already used.

use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    tensor::{backend::Backend, Tensor, TensorData},
};
use crossbeam_channel::{Receiver, Sender};
use ndarray::Array2;

use pinn_core::{
    loading::{BoundaryPoint, LoadConfig},
    material::MaterialProps,
    messages::{
        ControlMsg, HoleAnalysis, HoleBoundaryPoint, ParametricInferenceResult,
        ParametricTrainingUpdate, StressConcentration, TrainingMsg, VisFields,
    },
    parametric_spec::ParametricProblemSpec,
    problem::DomainSamplingStrategy,
    user_geometry::{HoleBc, HoleSpec, UserGeometry},
    LcgRng,
};

use crate::{
    architecture_controller::{ArchitectureConfig, ArchitectureController},
    energy::{compute_stress, constitutive_consistency_loss, dem_energy_loss, dem_energy_per_point, hole_traction_loss_direct, neumann_loss},
    fd_stencil::{assemble_stencil, compute_strains, norm_pts_to_tensor, FdConfig},
    lr_schedule::LrSchedule,
    network::{fwd, ElasticityNet, ElasticityNetConfig},
    optim::{apply_arch_action, make_bias_optim, make_gate_optim, BiasOptim, GateOptim, WeightOptim},
    runner::bin_collocation_density,
    saw_brdr::SawBrdr,
    training_core::{BDevice, BInner, B},
    user_problem::UserSamplingStrategy,
};

// SAW-BRDR base weights - deliberately the SAME values `user_problem.rs`'s own
// `LAM_INTERIOR_ENERGY`/`LAM_OUTER_TRACTION`/`LAM_HOLE_FREE`/`LAM_HOLE_FIXED` constants use
// (that module's constants are private, so these are a small, intentional, documented
// duplication rather than a visibility change to already-tested code).
const LAM_INTERIOR_ENERGY: f32 = 1.0;
const LAM_OUTER_TRACTION: f32 = 10.0;
const LAM_HOLE_FREE: f32 = 100.0;
const LAM_HOLE_FIXED: f32 = 50.0;
const LAM_CONSTITUTIVE_CONSISTENCY: f64 = 5.0;
const HOLE_RING_POINTS: usize = 64;
const SEED_PARAM_SAMPLE: u64 = 731_991;

/// Tiles one `(e_n, nu_n, p_n)` triple to `rows` identical rows - the parameter doesn't vary
/// spatially within a single forward pass. Generic over backend: used both with the
/// autodiff-enabled `B` during training and the inference-only `BInner` for vis/inference.
fn tile_params<Bk: Backend>(e_n: f32, nu_n: f32, p_n: f32, rows: usize, device: &Bk::Device) -> Tensor<Bk, 2> {
    let mut data = Vec::with_capacity(rows * 3);
    for _ in 0..rows { data.push(e_n); data.push(nu_n); data.push(p_n); }
    Tensor::<Bk, 2>::from_data(TensorData::new(data, vec![rows, 3]), device)
}

/// Builds the 6-wide network input for a set of already-3-wide (x,y,z) coordinate rows.
fn with_params<Bk: Backend>(xyz: Tensor<Bk, 2>, e_n: f32, nu_n: f32, p_n: f32, device: &Bk::Device) -> Tensor<Bk, 2> {
    let rows = xyz.dims()[0];
    Tensor::cat(vec![xyz, tile_params::<Bk>(e_n, nu_n, p_n, rows, device)], 1)
}

fn t_scalar<Bk: burn::tensor::backend::Backend>(t: &Tensor<Bk, 1>) -> f32 {
    t.clone().into_data().to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0)
}

/// Fixed (x,y)-only point sets for the trained geometry - sampled once, reused every step
/// (geometry is fixed in v1 - see this module's doc comment).
struct FixedPoints {
    interior: Vec<[f64; 2]>,
    boundary: Vec<BoundaryPoint>,
    holes: Vec<(HoleSpec, Vec<BoundaryPoint>)>,
}

fn build_fixed_points(geometry: &UserGeometry, n_interior: usize, n_boundary: usize, fd_h: f32) -> FixedPoints {
    let sampling = UserSamplingStrategy::new(geometry.clone(), fd_h);
    let placeholder = geometry.to_placeholder();
    let interior = sampling.sample_interior(&placeholder, n_interior);
    let boundary = sampling.sample_boundary(&placeholder, &LoadConfig::uniaxial_x(0.0), n_boundary);
    let named = sampling.named_point_sets(&[]);
    let holes = geometry.holes.iter().zip(named.into_iter())
        .map(|(hole, set)| (*hole, set.points)).collect();
    FixedPoints { interior, boundary, holes }
}

#[allow(clippy::too_many_arguments)]
struct StepScales {
    half_w: f64,
    half_h: f64,
    u_ref: f32,
    /// Fixed physical stress scale the network's raw stress columns are multiplied by -
    /// derived from the RANGE's worst-case magnitude (`ParamRange::max_abs`), not the
    /// current step's sampled load - a property of the normalization scheme, matching how
    /// `u_ref`/`ref_energy`/`ref_stress2` are fixed once per training on every other path in
    /// this codebase, not recomputed per step.
    stress_scale: f64,
    ref_energy: f32,
    ref_stress2: f32,
}

fn compute_scales(spec: &ParametricProblemSpec) -> StepScales {
    let stress_ref = spec.load_range.max_abs().max(1.0);
    let e_ref = spec.e_range.max_abs().max(1.0);
    StepScales {
        half_w: spec.geometry.half_w,
        half_h: spec.geometry.half_h,
        u_ref: ((stress_ref / e_ref) * spec.geometry.half_w) as f32,
        stress_scale: stress_ref,
        ref_energy: (0.5 * stress_ref * stress_ref / e_ref).max(1.0) as f32,
        ref_stress2: (stress_ref * stress_ref).max(1.0) as f32,
    }
}

fn norm_pt(x: f64, y: f64, scales: &StepScales) -> [f32; 2] {
    [(x / scales.half_w) as f32, (y / scales.half_h) as f32]
}

/// `step_parametric`'s return - a named struct (not a growing positional tuple) so adding a
/// new telemetry field (gradient norm, BC residual) doesn't require re-deriving every call
/// site's argument order from scratch.
struct StepParametricOutput {
    model: ElasticityNet<B>,
    e: f64,
    nu: f64,
    px: f64,
    energy_scalar: f32,
    boundary_scalar: f32,
    total_scalar: f32,
    lr: f32,
    /// `enhancement.txt` item B ("Gradient Norm") - L2 norm of every weight gradient this
    /// step, via the already-existing, already-tested `training_core::flatten_grads` (reused
    /// as-is, not reimplemented).
    grad_norm: f32,
    /// `enhancement.txt` items 4/C ("BC residual RMS/max") - real per-point traction/
    /// displacement residual at the outer boundary AND every hole ring, combined. Computed
    /// from the SAME per-point tensors already built for `boundary_loss`/`hole_terms` below
    /// (cloned before those tensors are consumed by `neumann_loss`/`hole_traction_loss_
    /// direct`) - not a second forward pass, and not the interior `pde_residual` field (a
    /// distinct quantity - see `VisFields::pde_residual`'s doc comment).
    bc_residual_rms: f64,
    bc_residual_max: f64,
}

/// One training step: samples a fresh `(E, nu, Px)` triple, runs interior/boundary/hole
/// forward passes conditioned on it, computes every loss term with the SAME formulas
/// `user_problem.rs` uses (just fed step-local material/load instead of spec-fixed), SAW-BRDR
/// weights them, backpropagates once, and steps the optimizer. Returns the new model plus
/// telemetry for this step.
#[allow(clippy::too_many_arguments)]
fn step_parametric(
    model: ElasticityNet<B>,
    weight_optim: &mut WeightOptim,
    bias_optim: &mut BiasOptim,
    gate_optim: &mut GateOptim,
    saw: &mut SawBrdr,
    lr_sched: &mut LrSchedule,
    spec: &ParametricProblemSpec,
    scales: &StepScales,
    fd: &FdConfig,
    points: &FixedPoints,
    rng: &mut LcgRng,
    device: &BDevice,
) -> StepParametricOutput {
    let e = spec.e_range.sample(rng.next_f64());
    let nu = spec.nu_range.sample(rng.next_f64());
    let px = spec.load_range.sample(rng.next_f64());
    let (e_n, nu_n, p_n) = (
        spec.e_range.normalize(e) as f32,
        spec.nu_range.normalize(nu) as f32,
        spec.load_range.normalize(px) as f32,
    );
    let material = spec.material_at(e, nu);

    // Interior: stencil (for FD strain), scale, split raw stress vs FD-derived strain.
    let int_pts: Vec<[f32; 2]> = points.interior.iter().map(|&[x, y]| norm_pt(x, y, scales)).collect();
    let n_int = int_pts.len();
    let int_stencil = assemble_stencil::<B>(&norm_pts_to_tensor::<B>(&int_pts, device), fd, device);
    let int_raw = fwd::<B>(&model, with_params(int_stencil, e_n, nu_n, p_n, device), 0, device);
    let m_int = 5 * n_int;
    let int_scaled = Tensor::cat(vec![
        int_raw.clone().slice([0..m_int, 0..2]).mul_scalar(scales.u_ref as f64),
        int_raw.slice([0..m_int, 2..5]).mul_scalar(scales.stress_scale),
    ], 1);
    let (int_exx, int_eyy, int_exy) = compute_strains::<B>(int_scaled.clone(), n_int, fd);
    let energy_loss = dem_energy_loss(int_exx.clone(), int_eyy.clone(), int_exy.clone(), &material)
        .mul_scalar(1.0 / scales.ref_energy as f64);
    let int_center = int_scaled.slice([0..n_int, 0..5]);
    let sxx_net = int_center.clone().slice([0..n_int, 2..3]).reshape([n_int]);
    let syy_net = int_center.clone().slice([0..n_int, 3..4]).reshape([n_int]);
    let sxy_net = int_center.slice([0..n_int, 4..5]).reshape([n_int]);
    let const_loss = constitutive_consistency_loss(sxx_net, syy_net, sxy_net, int_exx, int_eyy, int_exy, &material)
        .mul_scalar(1.0 / scales.ref_stress2 as f64);

    // Outer boundary: stencil (for FD strain), far-field Neumann traction target = px * n.
    let bnd_pts: Vec<[f32; 2]> = points.boundary.iter().map(|p| norm_pt(p.x, p.y, scales)).collect();
    let n_bnd = bnd_pts.len();
    let bnd_stencil = assemble_stencil::<B>(&norm_pts_to_tensor::<B>(&bnd_pts, device), fd, device);
    let bnd_raw = fwd::<B>(&model, with_params(bnd_stencil, e_n, nu_n, p_n, device), 0, device);
    let m_bnd = 5 * n_bnd;
    let bnd_scaled = Tensor::cat(vec![
        bnd_raw.clone().slice([0..m_bnd, 0..2]).mul_scalar(scales.u_ref as f64),
        bnd_raw.slice([0..m_bnd, 2..5]).mul_scalar(scales.stress_scale),
    ], 1);
    let (bnd_exx, bnd_eyy, bnd_exy) = compute_strains::<B>(bnd_scaled, n_bnd, fd);
    let bnd_nx: Vec<f32> = points.boundary.iter().map(|p| p.nx as f32).collect();
    let bnd_ny: Vec<f32> = points.boundary.iter().map(|p| p.ny as f32).collect();
    let nx_t = Tensor::<B, 1>::from_data(TensorData::new(bnd_nx, vec![n_bnd]), device);
    let ny_t = Tensor::<B, 1>::from_data(TensorData::new(bnd_ny, vec![n_bnd]), device);
    let tx_target = nx_t.clone().mul_scalar(px);
    let ty_target = ny_t.clone().mul_scalar(0.0); // py fixed at 0 - see this module's doc comment

    // BC residual (per-point, for RMS/max telemetry) - same math `neumann_loss` computes
    // internally, kept pre-mean here so a real distribution stat is possible, not just the
    // SAW-BRDR-weighted scalar loss.
    let mut bc_residuals: Vec<f32> = Vec::new();
    {
        let (sxx, syy, sxy) = compute_stress::<B>(bnd_exx.clone(), bnd_eyy.clone(), bnd_exy.clone(), &material);
        let tx_pred = sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone();
        let ty_pred = sxy * nx_t.clone() + syy * ny_t.clone();
        let ex = tx_pred - tx_target.clone();
        let ey = ty_pred - ty_target.clone();
        let mag = (ex.clone() * ex + ey.clone() * ey).sqrt();
        bc_residuals.extend(mag.into_data().to_vec::<f32>().unwrap_or_default());
    }
    let boundary_loss = neumann_loss(bnd_exx, bnd_eyy, bnd_exy, nx_t, ny_t, tx_target, ty_target, &material)
        .mul_scalar(1.0 / scales.ref_stress2 as f64);

    // Per-hole: direct mDEM columns, no stencil needed.
    let mut hole_terms: Vec<Tensor<B, 1>> = Vec::with_capacity(points.holes.len());
    let mut hole_base_weights: Vec<f32> = Vec::with_capacity(points.holes.len());
    for (hole, ring) in &points.holes {
        let ring_pts: Vec<[f32; 2]> = ring.iter().map(|p| norm_pt(p.x, p.y, scales)).collect();
        let n_h = ring_pts.len();
        let raw = fwd::<B>(&model, with_params(norm_pts_to_tensor::<B>(&ring_pts, device), e_n, nu_n, p_n, device), 0, device);
        let scaled = Tensor::cat(vec![
            raw.clone().slice([0..n_h, 0..2]).mul_scalar(scales.u_ref as f64),
            raw.slice([0..n_h, 2..5]).mul_scalar(scales.stress_scale),
        ], 1);
        match hole.bc {
            HoleBc::Free => {
                let nx: Vec<f32> = ring.iter().map(|p| p.nx as f32).collect();
                let ny: Vec<f32> = ring.iter().map(|p| p.ny as f32).collect();
                let nx_t = Tensor::<B, 1>::from_data(TensorData::new(nx, vec![n_h]), device);
                let ny_t = Tensor::<B, 1>::from_data(TensorData::new(ny, vec![n_h]), device);
                let sxx = scaled.clone().slice([0..n_h, 2..3]).reshape([n_h]);
                let syy = scaled.clone().slice([0..n_h, 3..4]).reshape([n_h]);
                let sxy = scaled.slice([0..n_h, 4..5]).reshape([n_h]);
                let tx = sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone();
                let ty = sxy.clone() * nx_t.clone() + syy.clone() * ny_t.clone();
                let mag = (tx.clone() * tx + ty.clone() * ty).sqrt();
                bc_residuals.extend(mag.into_data().to_vec::<f32>().unwrap_or_default());
                hole_terms.push(hole_traction_loss_direct(sxx, syy, sxy, nx_t, ny_t).mul_scalar(1.0 / scales.ref_stress2 as f64));
                hole_base_weights.push(LAM_HOLE_FREE);
            }
            HoleBc::Fixed => {
                let u = scaled.clone().slice([0..n_h, 0..1]).reshape([n_h]);
                let v = scaled.slice([0..n_h, 1..2]).reshape([n_h]);
                let mag = (u.clone() * u.clone() + v.clone() * v.clone()).sqrt();
                bc_residuals.extend(mag.into_data().to_vec::<f32>().unwrap_or_default());
                hole_terms.push((u.clone() * u + v.clone() * v).mean());
                hole_base_weights.push(LAM_HOLE_FIXED);
            }
        }
    }
    let (bc_residual_rms, bc_residual_max) = crate::training_core::residual_stats(&bc_residuals);

    // SAW-BRDR weighting - [interior_energy, outer_traction, hole terms...], stable order.
    let mut term_tensors = vec![energy_loss, boundary_loss];
    term_tensors.extend(hole_terms);
    let term_scalars: Vec<f32> = term_tensors.iter().map(t_scalar).collect();
    // `saw` was already constructed with the right base-weight count before the training loop
    // started (holes/geometry are fixed for the whole run - see this module's doc comment) -
    // `term_scalars.len()` here always matches what `saw` was built with.
    let lams = saw.update(&term_scalars);

    let mut total: Tensor<B, 1> = Tensor::zeros([1], device);
    let mut total_scalar = 0.0_f64;
    for (t, &lam) in term_tensors.into_iter().zip(lams.iter()) {
        total_scalar += t_scalar(&t) as f64 * lam as f64;
        total = total + t.mul_scalar(lam as f64);
    }
    let const_scalar = t_scalar(&const_loss);
    total_scalar += const_scalar as f64 * LAM_CONSTITUTIVE_CONSISTENCY;
    total = total + const_loss.mul_scalar(LAM_CONSTITUTIVE_CONSISTENCY);

    let lr = lr_sched.step(total_scalar.abs());
    let mut grads = total.backward();

    let (weight_ids, bias_ids) = model.param_ids();
    let gate_ids = model.gate_ids();
    let weight_grads = GradientsParams::from_params(&mut grads, &model, &weight_ids);
    let bias_grads = GradientsParams::from_params(&mut grads, &model, &bias_ids);
    let gate_grads = GradientsParams::from_params(&mut grads, &model, &gate_ids);

    // `enhancement.txt` item B ("Gradient Norm") - L2 norm of every weight/bias/gate
    // gradient this step, via the already-tested `training_core::flatten_grads` (reused
    // verbatim, not reimplemented). `weight_grads`/`bias_grads`/`gate_grads` each only carry
    // entries for their own param ids - `flatten_grads` treats any of the model's OTHER
    // params as zero (confirmed by its own sibling `flatten_grads_multi`'s doc comment: "any
    // param `GradientsParams::get` has no entry for" defaults out), so summing the three
    // flattened squared-norms reconstructs the exact total gradient norm with no double
    // counting - computed by reference here, BEFORE the three `GradientsParams` are moved
    // into the optimizer `.step()` calls below.
    let grad_norm: f32 = {
        let wn = crate::training_core::flatten_grads(&model, &weight_grads).powf_scalar(2.0_f64).sum();
        let bn = crate::training_core::flatten_grads(&model, &bias_grads).powf_scalar(2.0_f64).sum();
        let gn = crate::training_core::flatten_grads(&model, &gate_grads).powf_scalar(2.0_f64).sum();
        (wn + bn + gn).sqrt().into_scalar()
    };

    let model = weight_optim.step(lr, model, weight_grads);
    let model = bias_optim.step(lr, model, bias_grads);
    let model = gate_optim.step(lr, model, gate_grads);

    StepParametricOutput {
        model, e, nu, px,
        energy_scalar: term_scalars[0],
        boundary_scalar: term_scalars[1..].iter().sum(),
        total_scalar: total_scalar as f32,
        lr: lr as f32,
        grad_norm,
        bc_residual_rms,
        bc_residual_max,
    }
}

/// Evaluate the trained model at a specific `(e, nu, px)` over the visualization grid - the
/// "instant inference" primitive. Mirrors `user_problem::evaluate_user_vis_grid`'s Phase 14
/// field set/masking exactly, with every forward pass additionally conditioned on the query
/// parameters.
#[allow(clippy::too_many_arguments)]
fn evaluate_parametric_vis_grid(
    model: &ElasticityNet<BInner>,
    geometry: &UserGeometry,
    [nx, ny]: [usize; 2],
    scales: &StepScales,
    e_n: f32, nu_n: f32, p_n: f32,
    material: &MaterialProps,
    fd: &FdConfig,
    int_norm: &[[f32; 2]],
    device: &BDevice,
) -> VisFields {
    let n_total = nx * ny;
    let mut pts = Vec::with_capacity(n_total);
    let mut mask = Vec::with_capacity(n_total);
    for iy in 0..ny {
        for ix in 0..nx {
            let xn = -1.0 + 2.0 * ix as f64 / (nx.max(2) - 1) as f64;
            let yn = -1.0 + 2.0 * iy as f64 / (ny.max(2) - 1) as f64;
            pts.push([xn as f32, yn as f32]);
            mask.push(geometry.contains(xn * geometry.half_w, yn * geometry.half_h));
        }
    }
    let mut s_vm = vec![f32::NAN; n_total]; let mut s_xx = vec![f32::NAN; n_total];
    let mut s_yy = vec![f32::NAN; n_total]; let mut s_xy = vec![f32::NAN; n_total];
    let mut d_u = vec![f32::NAN; n_total]; let mut d_v = vec![f32::NAN; n_total];
    let mut e_xx = vec![f32::NAN; n_total]; let mut e_yy = vec![f32::NAN; n_total]; let mut e_xy = vec![f32::NAN; n_total];
    let mut pde = vec![f32::NAN; n_total]; let mut amr = vec![f32::NAN; n_total];

    let make = |vm, sxx, syy, sxy, u, v, exx, eyy, exy, pde, amr| {
        let a = |x: Vec<f32>| Array2::from_shape_vec((ny, nx), x).expect("shape mismatch");
        let density = bin_collocation_density(int_norm, nx, ny);
        VisFields {
            von_mises: a(vm), sigma_xx: a(sxx), sigma_yy: a(syy), sigma_xy: a(sxy),
            disp_u: a(u), disp_v: a(v), eps_xx: a(exx), eps_yy: a(eyy), eps_xy: a(exy),
            pde_residual: a(pde), amr_score: a(amr), collocation_density: a(density),
        }
    };
    let active: Vec<usize> = mask.iter().enumerate().filter(|(_, &m)| m).map(|(i, _)| i).collect();
    if active.is_empty() {
        return make(s_vm, s_xx, s_yy, s_xy, d_u, d_v, e_xx, e_yy, e_xy, pde, amr);
    }
    let active_pts: Vec<[f32; 2]> = active.iter().map(|&i| pts[i]).collect();
    let n_act = active_pts.len();
    let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&active_pts, device), fd, device);
    let raw = fwd::<BInner>(model, with_params(stencil, e_n, nu_n, p_n, device), 0, device);
    let m = 5 * n_act;
    let scaled = Tensor::cat(vec![
        raw.clone().slice([0..m, 0..2]).mul_scalar(scales.u_ref as f64),
        raw.slice([0..m, 2..5]).mul_scalar(scales.stress_scale),
    ], 1);
    let (eps_xx, eps_yy, eps_xy) = compute_strains::<BInner>(scaled.clone(), n_act, fd);
    let energy = dem_energy_per_point::<BInner>(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), material);
    let (sxx_fd, syy_fd, sxy_fd) = compute_stress::<BInner>(eps_xx.clone(), eps_yy.clone(), eps_xy.clone(), material);
    let center = scaled.slice([0..n_act, 0..5]);
    let batched: Vec<f32> = Tensor::cat(
        vec![center.reshape([5 * n_act]), eps_xx, eps_yy, eps_xy, sxx_fd, syy_fd, sxy_fd, energy], 0,
    ).into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; 12 * n_act]);
    let center_vals = &batched[..5 * n_act];
    let chunk = |i: usize| -> &[f32] { &batched[5 * n_act + i * n_act..5 * n_act + (i + 1) * n_act] };
    let (exx_v, eyy_v, exy_v) = (chunk(0), chunk(1), chunk(2));
    let (sxx_fd_v, syy_fd_v, sxy_fd_v) = (chunk(3), chunk(4), chunk(5));
    let energy_v = chunk(6);
    for (i_act, &i_full) in active.iter().enumerate() {
        let sxx = center_vals[i_act * 5 + 2] as f64;
        let syy = center_vals[i_act * 5 + 3] as f64;
        let sxy = center_vals[i_act * 5 + 4] as f64;
        s_xx[i_full] = sxx as f32; s_yy[i_full] = syy as f32; s_xy[i_full] = sxy as f32;
        s_vm[i_full] = (sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy).sqrt() as f32;
        d_u[i_full] = center_vals[i_act * 5]; d_v[i_full] = center_vals[i_act * 5 + 1];
        e_xx[i_full] = exx_v[i_act]; e_yy[i_full] = eyy_v[i_act]; e_xy[i_full] = exy_v[i_act];
        let dex = sxx - sxx_fd_v[i_act] as f64; let dey = syy - syy_fd_v[i_act] as f64; let dexy = sxy - sxy_fd_v[i_act] as f64;
        pde[i_full] = (dex*dex + dey*dey + dexy*dexy).sqrt() as f32;
        amr[i_full] = energy_v[i_act].abs();
    }
    make(s_vm, s_xx, s_yy, s_xy, d_u, d_v, e_xx, e_yy, e_xy, pde, amr)
}

/// `enhancement.txt` items 4/C ("BC residual") - real per-point traction/displacement
/// residual at the outer boundary AND every hole ring, combined into one RMS/max pair.
/// Generic over backend (`Bk`) so both the inference-only `BInner` call sites (periodic
/// status updates, on-demand `ParametricInfer` queries) share this ONE implementation - a
/// deliberate dedup `step_parametric`'s own (autodiff-`B`, backprop-entangled) inline
/// computation doesn't share, to avoid touching that already-verified training-step code a
/// second time for the sake of code reuse alone.
#[allow(clippy::too_many_arguments)]
fn bc_residual_stats<Bk: Backend>(
    model: &ElasticityNet<Bk>,
    points: &FixedPoints,
    scales: &StepScales,
    fd: &FdConfig,
    e_n: f32, nu_n: f32, p_n: f32,
    material: &MaterialProps,
    px: f64,
    device: &Bk::Device,
) -> (f64, f64) {
    let mut residuals: Vec<f32> = Vec::new();

    let bnd_pts: Vec<[f32; 2]> = points.boundary.iter().map(|p| norm_pt(p.x, p.y, scales)).collect();
    let n_bnd = bnd_pts.len();
    if n_bnd > 0 {
        let bnd_stencil = assemble_stencil::<Bk>(&norm_pts_to_tensor::<Bk>(&bnd_pts, device), fd, device);
        let bnd_raw = fwd::<Bk>(model, with_params::<Bk>(bnd_stencil, e_n, nu_n, p_n, device), 0, device);
        let m_bnd = 5 * n_bnd;
        let bnd_scaled = Tensor::cat(vec![
            bnd_raw.clone().slice([0..m_bnd, 0..2]).mul_scalar(scales.u_ref as f64),
            bnd_raw.slice([0..m_bnd, 2..5]).mul_scalar(scales.stress_scale),
        ], 1);
        let (exx, eyy, exy) = compute_strains::<Bk>(bnd_scaled, n_bnd, fd);
        let (sxx, syy, sxy) = compute_stress::<Bk>(exx, eyy, exy, material);
        let bnd_nx: Vec<f32> = points.boundary.iter().map(|p| p.nx as f32).collect();
        let bnd_ny: Vec<f32> = points.boundary.iter().map(|p| p.ny as f32).collect();
        let nx_t = Tensor::<Bk, 1>::from_data(TensorData::new(bnd_nx, vec![n_bnd]), device);
        let ny_t = Tensor::<Bk, 1>::from_data(TensorData::new(bnd_ny, vec![n_bnd]), device);
        let tx_target = nx_t.clone().mul_scalar(px);
        let ty_target = ny_t.clone().mul_scalar(0.0);
        let tx_pred = sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone();
        let ty_pred = sxy * nx_t + syy * ny_t;
        let ex = tx_pred - tx_target;
        let ey = ty_pred - ty_target;
        let mag = (ex.clone() * ex + ey.clone() * ey).sqrt();
        residuals.extend(mag.into_data().to_vec::<f32>().unwrap_or_default());
    }

    for (hole, ring) in &points.holes {
        let ring_pts: Vec<[f32; 2]> = ring.iter().map(|p| norm_pt(p.x, p.y, scales)).collect();
        let n_h = ring_pts.len();
        let raw = fwd::<Bk>(model, with_params::<Bk>(norm_pts_to_tensor::<Bk>(&ring_pts, device), e_n, nu_n, p_n, device), 0, device);
        let scaled = Tensor::cat(vec![
            raw.clone().slice([0..n_h, 0..2]).mul_scalar(scales.u_ref as f64),
            raw.slice([0..n_h, 2..5]).mul_scalar(scales.stress_scale),
        ], 1);
        match hole.bc {
            HoleBc::Free => {
                let nx: Vec<f32> = ring.iter().map(|p| p.nx as f32).collect();
                let ny: Vec<f32> = ring.iter().map(|p| p.ny as f32).collect();
                let nx_t = Tensor::<Bk, 1>::from_data(TensorData::new(nx, vec![n_h]), device);
                let ny_t = Tensor::<Bk, 1>::from_data(TensorData::new(ny, vec![n_h]), device);
                let sxx = scaled.clone().slice([0..n_h, 2..3]).reshape([n_h]);
                let syy = scaled.clone().slice([0..n_h, 3..4]).reshape([n_h]);
                let sxy = scaled.slice([0..n_h, 4..5]).reshape([n_h]);
                let tx = sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone();
                let ty = sxy * nx_t + syy * ny_t;
                let mag = (tx.clone() * tx + ty.clone() * ty).sqrt();
                residuals.extend(mag.into_data().to_vec::<f32>().unwrap_or_default());
            }
            HoleBc::Fixed => {
                let u = scaled.clone().slice([0..n_h, 0..1]).reshape([n_h]);
                let v = scaled.slice([0..n_h, 1..2]).reshape([n_h]);
                let mag = (u.clone() * u.clone() + v.clone() * v.clone()).sqrt();
                residuals.extend(mag.into_data().to_vec::<f32>().unwrap_or_default());
            }
        }
    }
    crate::training_core::residual_stats(&residuals)
}

/// `enhancement.md` Phase 9 - parametric analogue of `user_problem::probe_reaction_force`.
/// Generic over backend for the same reason `bc_residual_stats` is (shared by the periodic
/// training-time probe and the on-demand `ParametricInfer` query). See
/// `user_problem::probe_reaction_force`'s doc comment for why the meaningful check is "is the
/// PREDICTED net boundary force close to zero", not a comparison against a nonzero applied
/// resultant (the far-field target is self-canceling around the whole closed rectangle by
/// construction). `py` is always `0.0` in v1's parametric scope (see `ParametricProblemSpec`'s
/// module doc), so `reference_force` here only ever depends on `px`/`half_h`.
#[allow(clippy::too_many_arguments)]
fn reaction_force_stats<Bk: Backend>(
    model: &ElasticityNet<Bk>,
    points: &FixedPoints,
    scales: &StepScales,
    fd: &FdConfig,
    e_n: f32, nu_n: f32, p_n: f32,
    material: &MaterialProps,
    px: f64,
    thickness: f64,
    device: &Bk::Device,
) -> pinn_core::messages::ReactionForce {
    let reference_force = (px * 2.0 * scales.half_h * thickness).abs().max(1e-30);
    let n_bnd = points.boundary.len();
    if n_bnd == 0 {
        return pinn_core::messages::ReactionForce { net_fx: 0.0, net_fy: 0.0, reference_force, equilibrium_error: 0.0 };
    }
    let per_edge = (n_bnd / 4).max(1);
    let ds_x_normal = 2.0 * scales.half_h / per_edge as f64;
    let ds_y_normal = 2.0 * scales.half_w / per_edge as f64;

    let bnd_pts: Vec<[f32; 2]> = points.boundary.iter().map(|p| norm_pt(p.x, p.y, scales)).collect();
    let bnd_stencil = assemble_stencil::<Bk>(&norm_pts_to_tensor::<Bk>(&bnd_pts, device), fd, device);
    let bnd_raw = fwd::<Bk>(model, with_params::<Bk>(bnd_stencil, e_n, nu_n, p_n, device), 0, device);
    let m_bnd = 5 * n_bnd;
    let bnd_scaled = Tensor::cat(vec![
        bnd_raw.clone().slice([0..m_bnd, 0..2]).mul_scalar(scales.u_ref as f64),
        bnd_raw.slice([0..m_bnd, 2..5]).mul_scalar(scales.stress_scale),
    ], 1);
    let (exx, eyy, exy) = compute_strains::<Bk>(bnd_scaled, n_bnd, fd);
    let (sxx, syy, sxy) = compute_stress::<Bk>(exx, eyy, exy, material);
    let bnd_nx: Vec<f32> = points.boundary.iter().map(|p| p.nx as f32).collect();
    let bnd_ny: Vec<f32> = points.boundary.iter().map(|p| p.ny as f32).collect();
    let nx_t = Tensor::<Bk, 1>::from_data(TensorData::new(bnd_nx.clone(), vec![n_bnd]), device);
    let ny_t = Tensor::<Bk, 1>::from_data(TensorData::new(bnd_ny.clone(), vec![n_bnd]), device);
    let tx_pred = (sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone())
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
    let ty_pred = (sxy * nx_t + syy * ny_t)
        .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);

    let mut net_fx = 0.0f64;
    let mut net_fy = 0.0f64;
    for i in 0..n_bnd {
        let ds = if bnd_nx[i].abs() > 0.5 { ds_x_normal } else { ds_y_normal };
        net_fx += tx_pred[i] as f64 * ds * thickness;
        net_fy += ty_pred[i] as f64 * ds * thickness;
    }
    let equilibrium_error = (net_fx * net_fx + net_fy * net_fy).sqrt() / reference_force;
    pinn_core::messages::ReactionForce { net_fx, net_fy, reference_force, equilibrium_error }
}

/// `enhancement.md` Phase 10 - parametric analogue of `user_problem::probe_energy_balance`.
/// See that function's doc comment for the internal-energy/external-work formulation; `area`
/// is passed in (rather than recomputed from `points.holes` each call) since the fixed
/// geometry's area never changes across steps in v1.
#[allow(clippy::too_many_arguments)]
fn energy_balance_stats<Bk: Backend>(
    model: &ElasticityNet<Bk>,
    points: &FixedPoints,
    scales: &StepScales,
    fd: &FdConfig,
    e_n: f32, nu_n: f32, p_n: f32,
    material: &MaterialProps,
    area: f64,
    thickness: f64,
    device: &Bk::Device,
) -> pinn_core::messages::EnergyBalance {
    let internal_energy = if points.interior.is_empty() {
        0.0
    } else {
        let n_int = points.interior.len();
        let int_norm: Vec<[f32; 2]> = points.interior.iter().map(|&[x, y]| norm_pt(x, y, scales)).collect();
        let stencil = assemble_stencil::<Bk>(&norm_pts_to_tensor::<Bk>(&int_norm, device), fd, device);
        let raw = fwd::<Bk>(model, with_params::<Bk>(stencil, e_n, nu_n, p_n, device), 0, device);
        let m = 5 * n_int;
        let scaled = Tensor::cat(vec![
            raw.clone().slice([0..m, 0..2]).mul_scalar(scales.u_ref as f64),
            raw.slice([0..m, 2..5]).mul_scalar(scales.stress_scale),
        ], 1);
        let (exx, eyy, exy) = compute_strains::<Bk>(scaled, n_int, fd);
        let energy_density = dem_energy_per_point::<Bk>(exx, eyy, exy, material);
        let mean_density: f64 = energy_density.into_data().to_vec::<f32>().unwrap_or_default()
            .iter().map(|&v| v as f64).sum::<f64>() / n_int as f64;
        mean_density * area * thickness
    };

    let n_bnd = points.boundary.len();
    let external_work = if n_bnd == 0 {
        0.0
    } else {
        let per_edge = (n_bnd / 4).max(1);
        let ds_x_normal = 2.0 * scales.half_h / per_edge as f64;
        let ds_y_normal = 2.0 * scales.half_w / per_edge as f64;
        let bnd_pts: Vec<[f32; 2]> = points.boundary.iter().map(|p| norm_pt(p.x, p.y, scales)).collect();
        let bnd_stencil = assemble_stencil::<Bk>(&norm_pts_to_tensor::<Bk>(&bnd_pts, device), fd, device);
        let bnd_raw = fwd::<Bk>(model, with_params::<Bk>(bnd_stencil, e_n, nu_n, p_n, device), 0, device);
        let m_bnd = 5 * n_bnd;
        let bnd_scaled = Tensor::cat(vec![
            bnd_raw.clone().slice([0..m_bnd, 0..2]).mul_scalar(scales.u_ref as f64),
            bnd_raw.slice([0..m_bnd, 2..5]).mul_scalar(scales.stress_scale),
        ], 1);
        let (exx, eyy, exy) = compute_strains::<Bk>(bnd_scaled.clone(), n_bnd, fd);
        let (sxx, syy, sxy) = compute_stress::<Bk>(exx, eyy, exy, material);
        let bnd_nx: Vec<f32> = points.boundary.iter().map(|p| p.nx as f32).collect();
        let bnd_ny: Vec<f32> = points.boundary.iter().map(|p| p.ny as f32).collect();
        let nx_t = Tensor::<Bk, 1>::from_data(TensorData::new(bnd_nx.clone(), vec![n_bnd]), device);
        let ny_t = Tensor::<Bk, 1>::from_data(TensorData::new(bnd_ny.clone(), vec![n_bnd]), device);
        let tx_pred = (sxx.clone() * nx_t.clone() + sxy.clone() * ny_t.clone())
            .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
        let ty_pred = (sxy * nx_t + syy * ny_t)
            .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
        let u_vals: Vec<f32> = bnd_scaled.clone().slice([0..n_bnd, 0..1]).reshape([n_bnd])
            .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
        let v_vals: Vec<f32> = bnd_scaled.slice([0..n_bnd, 1..2]).reshape([n_bnd])
            .into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n_bnd]);
        let mut work = 0.0f64;
        for i in 0..n_bnd {
            let ds = if bnd_nx[i].abs() > 0.5 { ds_x_normal } else { ds_y_normal };
            work += (tx_pred[i] as f64 * u_vals[i] as f64 + ty_pred[i] as f64 * v_vals[i] as f64) * ds * thickness;
        }
        0.5 * work
    };

    let denom = external_work.abs().max(1e-30);
    let energy_balance_error = (internal_energy - external_work).abs() / denom;
    pinn_core::messages::EnergyBalance { internal_energy, external_work, energy_balance_error }
}

/// Hole-boundary stress profile at a specific `(e, nu, px)` - parametric analogue of
/// `user_problem::probe_hole_boundary_profile`.
fn probe_hole_profile_parametric(
    model: &ElasticityNet<BInner>,
    hole: &HoleSpec,
    n_theta: usize,
    fd: &FdConfig,
    scales: &StepScales,
    e_n: f32, nu_n: f32, p_n: f32,
    device: &BDevice,
) -> Vec<HoleBoundaryPoint> {
    let n = n_theta.max(1);
    let mut thetas = Vec::with_capacity(n);
    let mut pts_phys = Vec::with_capacity(n);
    let mut pts_norm = Vec::with_capacity(n);
    for i in 0..n {
        let theta_deg = 360.0 * i as f64 / n as f64;
        let theta = theta_deg.to_radians();
        let x = hole.center[0] + hole.radius * theta.cos();
        let y = hole.center[1] + hole.radius * theta.sin();
        thetas.push(theta_deg); pts_phys.push((x, y));
        pts_norm.push([(x / scales.half_w) as f32, (y / scales.half_h) as f32]);
    }
    let stencil = assemble_stencil::<BInner>(&norm_pts_to_tensor::<BInner>(&pts_norm, device), fd, device);
    let raw = fwd::<BInner>(model, with_params(stencil, e_n, nu_n, p_n, device), 0, device);
    let m = 5 * n;
    let scaled = Tensor::cat(vec![
        raw.clone().slice([0..m, 0..2]).mul_scalar(scales.u_ref as f64),
        raw.slice([0..m, 2..5]).mul_scalar(scales.stress_scale),
    ], 1);
    let center = scaled.clone().slice([0..n, 0..5]);
    let (eps_xx, eps_yy, eps_xy) = compute_strains::<BInner>(scaled, n, fd);
    let center_vals: Vec<f32> = center.into_data().to_vec().unwrap_or_else(|_| vec![0.0; 5 * n]);
    let exx_vals: Vec<f32> = eps_xx.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    let eyy_vals: Vec<f32> = eps_yy.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    let exy_vals: Vec<f32> = eps_xy.into_data().to_vec().unwrap_or_else(|_| vec![0.0; n]);
    (0..n).map(|i| {
        let (ux, uy) = (center_vals[i*5], center_vals[i*5+1]);
        let (sxx, syy, sxy) = (center_vals[i*5+2], center_vals[i*5+3], center_vals[i*5+4]);
        let vm = ((sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy) as f64).sqrt() as f32;
        let (x, y) = pts_phys[i];
        HoleBoundaryPoint { theta_deg: thetas[i], x, y, ux, uy, eps_xx: exx_vals[i], eps_yy: eyy_vals[i], eps_xy: exy_vals[i], sxx, syy, sxy, von_mises: vm }
    }).collect()
}

fn stress_concentration(profile: &[HoleBoundaryPoint], nominal_stress: f64) -> StressConcentration {
    let (mut max_vm, mut max_theta) = (f64::NEG_INFINITY, 0.0);
    for p in profile {
        if (p.von_mises as f64) > max_vm { max_vm = p.von_mises as f64; max_theta = p.theta_deg; }
    }
    let kt = if nominal_stress.abs() > 1e-300 { max_vm / nominal_stress } else { 0.0 };
    StressConcentration {
        nominal_stress, max_von_mises: max_vm, max_theta_deg: max_theta, kt,
        stress_projection: "VonMises",
        // Issue #62 PH3-15: the parametric path has no derived-stress probe (`probe_hole_
        // profile_parametric` reads the network's DIRECT stress output only) to run `user_
        // problem::kt_convergence_check`'s angular/radial re-probing against - `None` is a
        // real "not computed for this path", not a fabricated non-convergence claim.
        angular_refinement_relative_change: None,
        radial_offset_refinement_relative_change: None,
        refinement_converged: None,
        domain_classification: "FiniteDomainReference",
    }
}

fn hole_analyses_at(
    model: &ElasticityNet<BInner>, geometry: &UserGeometry, fd: &FdConfig, scales: &StepScales,
    e_n: f32, nu_n: f32, p_n: f32, px: f64, device: &BDevice,
) -> Vec<HoleAnalysis> {
    geometry.holes.iter().enumerate().map(|(hole_index, hole)| {
        let profile = probe_hole_profile_parametric(model, hole, HOLE_RING_POINTS, fd, scales, e_n, nu_n, p_n, device);
        let concentration = stress_concentration(&profile, px.abs());
        HoleAnalysis { hole_index, profile, concentration }
    }).collect()
}

/// Drives a parametric PINN training run, then keeps the model alive to answer
/// `ControlMsg::ParametricInfer` requests - see this module's doc comment.
pub fn run_training_parametric(spec: ParametricProblemSpec, tx: Sender<TrainingMsg>, stop_rx: Receiver<ControlMsg>) {
    let device = BDevice::default();
    let scales = compute_scales(&spec);
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * spec.geometry.half_w, 2.0 * spec.geometry.half_h);
    let points = build_fixed_points(&spec.geometry, spec.training.n_interior, spec.training.n_boundary, spec.training.fd_h);
    let int_norm: Vec<[f32; 2]> = points.interior.iter().map(|&[x, y]| norm_pt(x, y, &scales)).collect();
    // Fixed geometry (v1 scope) - the plate's area never changes across steps, so this is
    // computed once here rather than inside the per-step/per-query probe (`energy_balance_
    // stats`'s `area` parameter).
    let area = 4.0 * spec.geometry.half_w * spec.geometry.half_h
        - spec.geometry.holes.iter().map(|h| std::f64::consts::PI * h.radius * h.radius).sum::<f64>();

    let net_cfg = ElasticityNetConfig::new()
        .with_input_dim(6).with_hidden_dim(spec.network.hidden_dim).with_n_hidden(spec.network.n_hidden)
        .with_output_dim(5)
        // Smart adaptive architecture: forces the gated-residual (PirateNet) structure
        // internally whenever `adaptive` - see `runner::run_training_user_problem`'s own
        // `net_cfg` for the identical rationale (no separate user-facing toggle exists).
        .with_use_piratenet(spec.network.adaptive);
    // Issue #62 PH3-11 - same fix as `runner::run_training_user_problem`, see that call site's
    // own comment.
    B::seed(&device, spec.network.model_init_seed);
    let mut model = net_cfg.init(&device);
    let mut weight_optim = WeightOptim::new(true);
    let mut bias_optim = make_bias_optim();
    let mut gate_optim = make_gate_optim();

    // Smart adaptive architecture (v1) - see `runner::run_training_user_problem`'s identical
    // setup for the full rationale on each of these.
    let mut current_hidden_dim = spec.network.hidden_dim;
    let mut current_n_hidden = spec.network.n_hidden;
    let arch_config = ArchitectureConfig::v1(
        spec.network.max_hidden_dim.unwrap_or(usize::MAX),
        spec.network.max_n_hidden.unwrap_or(usize::MAX),
    );
    let mut arch_controller = spec.network.adaptive.then(|| ArchitectureController::new(arch_config.clone()));
    let mut arch_snapshot: Option<(ElasticityNet<B>, usize, usize)> = None;
    let base_weights = { let mut w = vec![LAM_INTERIOR_ENERGY, LAM_OUTER_TRACTION]; w.extend(points.holes.iter().map(|(h, _)| match h.bc { HoleBc::Free => LAM_HOLE_FREE, HoleBc::Fixed => LAM_HOLE_FIXED })); w };
    let mut saw = SawBrdr::with_base(base_weights, 0.95);
    let mut lr_sched = LrSchedule::new(spec.training.lr, 100, 500);
    let mut rng = LcgRng::new(SEED_PARAM_SAMPLE);

    // `enhancement.md` Phase 21 ("Do Not Rely Only on Min/Max") - a bounded, recent-history
    // reservoir of every normalized `(e_n, nu_n, p_n)` triple actually drawn this run, used to
    // answer "how close is this instant-inference query to what training actually saw" with a
    // real nearest-neighbor distance instead of only a min/max range check. Bounded (not
    // unbounded) since a long run could otherwise accumulate an ever-growing Vec for the
    // lifetime of the training thread - FIFO eviction keeps memory flat while still reflecting
    // "recent" coverage, which is what matters for a still-live model.
    const PARAM_RESERVOIR_CAP: usize = 500;
    let mut param_reservoir: std::collections::VecDeque<[f32; 3]> = std::collections::VecDeque::with_capacity(PARAM_RESERVOIR_CAP);
    // Stage H (model checkpoint save/load) - tracked for the same reason `runner::
    // run_training_user_problem` tracks its own `last_step`/`last_total_loss`.
    let mut last_step = 0usize;
    let mut last_total_loss = 0.0f32;

    for step in 0..spec.training.max_steps {
        if let ControlAction::StopImmediately = handle_control_messages(&stop_rx) {
            return;
        }
        last_step = step;

        let out = step_parametric(
            model, &mut weight_optim, &mut bias_optim, &mut gate_optim, &mut saw, &mut lr_sched,
            &spec, &scales, &fd, &points, &mut rng, &device,
        );
        model = out.model;
        let (e, nu, px) = (out.e, out.nu, out.px);

        param_reservoir.push_back([spec.e_range.normalize(e) as f32, spec.nu_range.normalize(nu) as f32, spec.load_range.normalize(px) as f32]);
        if param_reservoir.len() > PARAM_RESERVOIR_CAP { param_reservoir.pop_front(); }

        let send_vis = step % 10 == 0 || step + 1 == spec.training.max_steps;
        let mut architecture_event = None;
        let (vis, hole_analyses, reaction_force, energy_balance, network_snapshot) = if send_vis {
            let model_val: ElasticityNet<BInner> = model.valid();
            let (e_n, nu_n, p_n) = (spec.e_range.normalize(e) as f32, spec.nu_range.normalize(nu) as f32, spec.load_range.normalize(px) as f32);
            let material = spec.material_at(e, nu);
            let vis = evaluate_parametric_vis_grid(&model_val, &spec.geometry, [64, 64], &scales, e_n, nu_n, p_n, &material, &fd, &int_norm, &device);
            let analyses = hole_analyses_at(&model_val, &spec.geometry, &fd, &scales, e_n, nu_n, p_n, px, &device);
            let rf = reaction_force_stats(&model_val, &points, &scales, &fd, e_n, nu_n, p_n, &material, px, spec.geometry.thickness, &device);
            let eb = energy_balance_stats(&model_val, &points, &scales, &fd, e_n, nu_n, p_n, &material, area, spec.geometry.thickness, &device);
            let ns = crate::network::network_snapshot(&model_val);

            // Smart adaptive architecture - see `runner::run_training_user_problem`'s
            // identical block for the full rationale. `out.bc_residual_rms` is already
            // computed every step here (unlike the plate path), so no extra probe is needed.
            if let Some(controller) = arch_controller.as_mut() {
                let per_neuron_mags = model_val.per_neuron_magnitudes();
                if let Some(action) = controller.observe(
                    out.bc_residual_rms, current_hidden_dim, current_n_hidden, &ns.awake_mask, &per_neuron_mags,
                ) {
                    let (new_model, description, new_hidden_dim, new_n_hidden) = apply_arch_action(
                        &action, model, &mut arch_snapshot, &mut weight_optim, true,
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

            (Some(vis), analyses, Some(rf), Some(eb), Some(ns))
        } else { (None, Vec::new(), None, None, None) };

        last_total_loss = out.total_scalar;
        let _ = tx.try_send(TrainingMsg::ParametricUpdate(Box::new(ParametricTrainingUpdate {
            step, total_loss: out.total_scalar, energy_loss: out.energy_scalar, boundary_loss: out.boundary_scalar,
            lr: out.lr, e_this_step: e, nu_this_step: nu, load_this_step: px,
            grad_norm: out.grad_norm, bc_residual_rms: out.bc_residual_rms, bc_residual_max: out.bc_residual_max,
            reaction_force, energy_balance, network_snapshot,
            architecture_event,
            vis, hole_analyses,
        })));
    }

    let _ = tx.try_send(TrainingMsg::ParametricReady);
    let model_val: ElasticityNet<BInner> = model.valid();
    serve_parametric_inference(
        &spec, model_val, &scales, &fd, &points, &int_norm, area, &param_reservoir,
        last_step, last_total_loss, current_hidden_dim, current_n_hidden, &device, &tx, &stop_rx,
    );
}

/// Shared instant-inference serving loop - answers `ParametricInfer`/`SaveCheckpoint` until
/// `Stop`/disconnect. Used both by `run_training_parametric`'s post-training tail AND by
/// `serve_loaded_checkpoint` (a checkpoint loaded straight from disk, no training performed) -
/// the model doesn't care whether it just finished training or was deserialized from a save
/// file, so this is the ONE place that serving logic lives, not duplicated per entry point.
#[allow(clippy::too_many_arguments)]
fn serve_parametric_inference(
    spec: &ParametricProblemSpec,
    model: ElasticityNet<BInner>,
    scales: &StepScales,
    fd: &FdConfig,
    points: &FixedPoints,
    int_norm: &[[f32; 2]],
    area: f64,
    param_reservoir: &std::collections::VecDeque<[f32; 3]>,
    last_step: usize,
    last_total_loss: f32,
    // Smart adaptive architecture: the LIVE architecture reached (may differ from
    // `spec.network.hidden_dim`/`n_hidden` after adaptation) - see `CheckpointMeta`'s own
    // doc comment on why a checkpoint must record this, not the original spec's static
    // values. Callers with no adaptation to report (e.g. `serve_loaded_checkpoint`) just
    // pass `spec.network.hidden_dim`/`n_hidden` through unchanged.
    current_hidden_dim: usize,
    current_n_hidden: usize,
    device: &BDevice,
    tx: &Sender<TrainingMsg>,
    stop_rx: &Receiver<ControlMsg>,
) {
    loop {
        match stop_rx.recv() {
            Ok(ControlMsg::Stop) | Err(_) => return,
            Ok(ControlMsg::ParametricInfer { e, nu, px }) => {
                let (e_n, nu_n, p_n) = (spec.e_range.normalize(e) as f32, spec.nu_range.normalize(nu) as f32, spec.load_range.normalize(px) as f32);
                let material = spec.material_at(e, nu);
                let vis = evaluate_parametric_vis_grid(&model, &spec.geometry, [64, 64], scales, e_n, nu_n, p_n, &material, fd, int_norm, device);
                let hole_analyses = hole_analyses_at(&model, &spec.geometry, fd, scales, e_n, nu_n, p_n, px, device);
                let (bc_residual_rms, bc_residual_max) = bc_residual_stats(&model, points, scales, fd, e_n, nu_n, p_n, &material, px, device);
                let reaction_force = reaction_force_stats(&model, points, scales, fd, e_n, nu_n, p_n, &material, px, spec.geometry.thickness, device);
                let energy_balance = energy_balance_stats(&model, points, scales, fd, e_n, nu_n, p_n, &material, area, spec.geometry.thickness, device);
                let reservoir_slice: Vec<[f32; 3]> = param_reservoir.iter().copied().collect();
                let nearest_sample_distance = pinn_core::param_distance::nearest_neighbor_distance([e_n, nu_n, p_n], &reservoir_slice) as f64;
                let typical_sample_spacing = pinn_core::param_distance::median_nn_spacing(&reservoir_slice) as f64;
                let in_range = spec.in_range(e, nu, px);
                let _ = tx.try_send(TrainingMsg::ParametricInferResult(Box::new(ParametricInferenceResult {
                    e, nu, px, in_range, bc_residual_rms, bc_residual_max, reaction_force, energy_balance,
                    nearest_sample_distance, typical_sample_spacing, vis, hole_analyses,
                })));
            }
            Ok(ControlMsg::SaveCheckpoint { path, saved_at_unix }) => {
                let mut live_spec = spec.clone();
                live_spec.network.hidden_dim = current_hidden_dim;
                live_spec.network.n_hidden = current_n_hidden;
                let provenance = crate::provenance::compute_run_provenance(&live_spec, None, Some(live_spec.network.model_init_seed), &live_spec.network, &live_spec.training);
                let meta = crate::checkpoint::CheckpointMeta {
                    spec: crate::checkpoint::CheckpointSpec::Parametric(live_spec),
                    steps_completed: last_step + 1,
                    final_loss: last_total_loss,
                    saved_at_unix,
                    provenance,
                    // Issue #62 PH3-13: the full authoritative report is scoped to the plate
                    // path this pass (see `run_user_problem_training_from`'s own SaveCheckpoint
                    // handler) - the parametric path has no AMR mechanism and no measure-aware
                    // integration upgrade (see this module's own doc comment: it reuses
                    // `energy.rs` functions directly, never `step_physics_multi`/`measure_
                    // integral`), so claiming an `integration_mode`/`sampling_mode` for it here
                    // would misrepresent behavior this path doesn't actually have.
                    report: None,
                };
                let result = crate::checkpoint::save_checkpoint(model.clone(), &meta, &path)
                    .map(|p| p.display().to_string());
                let _ = tx.try_send(TrainingMsg::CheckpointSaved(result));
            }
            Ok(_) => {}
        }
    }
}

/// Stage H (model checkpoint save/load) - serves a `ParametricProblemSpec` checkpoint loaded
/// straight from disk, no training performed. Sends `ParametricReady` immediately (the model
/// is already trained) and enters the same [`serve_parametric_inference`] loop a just-finished
/// training run would.
pub fn serve_loaded_checkpoint(
    spec: ParametricProblemSpec,
    model: ElasticityNet<BInner>,
    tx: Sender<TrainingMsg>,
    stop_rx: Receiver<ControlMsg>,
) {
    let device = BDevice::default();
    let scales = compute_scales(&spec);
    let fd = FdConfig::new(spec.training.fd_h, 2.0 * spec.geometry.half_w, 2.0 * spec.geometry.half_h);
    let points = build_fixed_points(&spec.geometry, spec.training.n_interior, spec.training.n_boundary, spec.training.fd_h);
    let int_norm: Vec<[f32; 2]> = points.interior.iter().map(|&[x, y]| norm_pt(x, y, &scales)).collect();
    let area = 4.0 * spec.geometry.half_w * spec.geometry.half_h
        - spec.geometry.holes.iter().map(|h| std::f64::consts::PI * h.radius * h.radius).sum::<f64>();
    // Empty reservoir, honestly - no training happened this session, so there is genuinely no
    // per-step sample history to report coverage against (the distance-to-training-
    // distribution check degrades to "unknown" via `nearest_neighbor_distance`'s own
    // documented `f64::INFINITY`-on-empty behavior, not a fabricated value).
    let param_reservoir: std::collections::VecDeque<[f32; 3]> = std::collections::VecDeque::new();

    let _ = tx.try_send(TrainingMsg::ParametricReady);
    let (hidden_dim, n_hidden) = (spec.network.hidden_dim, spec.network.n_hidden);
    serve_parametric_inference(
        &spec, model, &scales, &fd, &points, &int_norm, area, &param_reservoir, 0, 0.0,
        hidden_dim, n_hidden, &device, &tx, &stop_rx,
    );
}

enum ControlAction { Continue, StopImmediately }

fn handle_control_messages(stop_rx: &Receiver<ControlMsg>) -> ControlAction {
    while let Ok(msg) = stop_rx.try_recv() {
        if let ControlMsg::Stop = msg {
            return ControlAction::StopImmediately;
        }
    }
    ControlAction::Continue
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::{
        parametric_spec::ParamRange,
        problem_spec::{NetworkSpec, TrainingSpec},
        user_geometry::HoleSpec,
    };

    fn tiny_spec(max_steps: usize) -> ParametricProblemSpec {
        ParametricProblemSpec {
            geometry: UserGeometry {
                half_w: 0.1, half_h: 0.1, thickness: 0.005,
                holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.02, bc: HoleBc::Free }],
            },
            e_range: ParamRange::new(50e9, 100e9),
            nu_range: ParamRange::new(0.25, 0.35),
            load_range: ParamRange::new(40e6, 80e6),
            density: 2810.0,
            ultimate_strength_pa: 503e6,
            network: NetworkSpec { hidden_dim: 16, n_hidden: 2, ..Default::default() },
            training: TrainingSpec { max_steps, n_interior: 128, n_boundary: 32, fd_h: 1e-3, lr: 1e-3, measure_aware_training: false, derivative_operator_diagnostic: false, amr_enabled: true },
        }
    }

    #[test]
    fn run_training_parametric_completes_and_sends_updates_and_ready() {
        // `run_training_parametric` deliberately stays alive after training (blocked on
        // `ControlMsg::ParametricInfer`/`Stop` - see this module's doc comment on "instant
        // inference after training"), so it must run on its own thread here and be sent an
        // explicit `Stop` once `ParametricReady` is observed - it does not return on its own.
        //
        // 6 steps (not more) - kept deliberately small since this test's actual wall-clock
        // cost is backend-dependent (the default `wgpu` backend's fixed per-dispatch
        // overhead dominates far more than `ndarray`'s for tensors this tiny - a real,
        // confirmed difference, not assumed), and the poll deadline below is a genuine wall-
        // clock budget (60s), not a fixed iteration count, so it stays correct either way.
        let spec = tiny_spec(6);
        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || run_training_parametric(spec, tx, rx_ctrl));

        let mut saw_update = false;
        let mut saw_ready = false;
        let mut steps_seen: Vec<usize> = Vec::new();
        let mut params_seen: std::collections::HashSet<(u64, u64, u64)> = std::collections::HashSet::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(TrainingMsg::ParametricUpdate(u)) => {
                    saw_update = true;
                    steps_seen.push(u.step);
                    assert!(u.total_loss.is_finite(), "total_loss must be finite");
                    assert!(u.grad_norm.is_finite() && u.grad_norm >= 0.0, "grad_norm must be finite and non-negative, got {}", u.grad_norm);
                    assert!(u.bc_residual_rms.is_finite() && u.bc_residual_rms >= 0.0, "bc_residual_rms must be finite and non-negative");
                    assert!(u.bc_residual_max >= u.bc_residual_rms - 1e-6, "bc_residual_max must be >= rms");
                    if let Some(rf) = &u.reaction_force {
                        assert!(rf.net_fx.is_finite() && rf.net_fy.is_finite(), "reaction_force net force must be finite");
                        assert!(rf.equilibrium_error.is_finite() && rf.equilibrium_error >= 0.0, "equilibrium_error must be finite and non-negative");
                    }
                    params_seen.insert((u.e_this_step.to_bits(), u.nu_this_step.to_bits(), u.load_this_step.to_bits()));
                }
                Ok(TrainingMsg::ParametricReady) => { saw_ready = true; break; }
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();

        assert!(saw_update, "must send at least one ParametricUpdate");
        assert!(saw_ready, "must send ParametricReady after training completes");
        assert!(params_seen.len() > 1, "different steps must sample different (E,nu,Px) triples - got {}", params_seen.len());
    }

    fn tiny_adaptive_spec(max_steps: usize) -> ParametricProblemSpec {
        ParametricProblemSpec {
            geometry: UserGeometry {
                half_w: 0.1, half_h: 0.1, thickness: 0.005,
                holes: vec![HoleSpec { center: [0.0, 0.0], radius: 0.02, bc: HoleBc::Free }],
            },
            e_range: ParamRange::new(50e9, 100e9),
            nu_range: ParamRange::new(0.25, 0.35),
            load_range: ParamRange::new(40e6, 80e6),
            density: 2810.0,
            ultimate_strength_pa: 503e6,
            network: NetworkSpec {
                hidden_dim: 8, n_hidden: 3,
                adaptive: true, max_hidden_dim: Some(12), max_n_hidden: Some(4),
                ..Default::default()
            },
            training: TrainingSpec { max_steps, n_interior: 64, n_boundary: 32, fd_h: 1e-3, lr: 1e-3, measure_aware_training: false, derivative_operator_diagnostic: false, amr_enabled: true },
        }
    }

    /// Real end-to-end integration check that `ArchitectureController` is actually wired into
    /// `run_training_parametric` - see `runner::tests::
    /// run_training_user_problem_adaptive_wiring_does_not_crash_and_events_are_consistent`'s
    /// doc comment for why this deliberately doesn't hard-require an event to fire (real,
    /// not fully step-seeded, training dynamics). `#[ignore]`d as a genuine multi-hundred-step
    /// run: `cargo test -p pinn-solver --features ndarray-backend --release
    /// parametric_problem::tests::run_training_parametric_adaptive_wiring -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn run_training_parametric_adaptive_wiring_does_not_crash_and_events_are_consistent() {
        let spec = tiny_adaptive_spec(450);
        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || run_training_parametric(spec, tx, rx_ctrl));

        let mut events = Vec::new();
        let mut saw_ready = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(TrainingMsg::ParametricUpdate(u)) => {
                    assert!(u.total_loss.is_finite(), "step {}: non-finite total_loss under adaptive wiring", u.step);
                    if let Some(ev) = u.architecture_event.clone() {
                        events.push(ev);
                    }
                }
                Ok(TrainingMsg::Error(e)) => panic!("adaptive run reported an error: {e}"),
                Ok(TrainingMsg::ParametricReady) => { saw_ready = true; break; }
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();

        assert!(saw_ready, "expected the run to reach ParametricReady");
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
    fn run_training_parametric_serves_instant_inference_after_ready() {
        let spec = tiny_spec(5);
        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();

        let spec_clone = spec.clone();
        let handle = std::thread::spawn(move || run_training_parametric(spec_clone, tx, rx_ctrl));

        assert!(wait_for_ready(&rx), "training did not reach ParametricReady in time");

        let mid_e = spec.e_range.mid();
        let mid_nu = spec.nu_range.mid();
        let mid_px = spec.load_range.mid();
        tx_ctrl.send(ControlMsg::ParametricInfer { e: mid_e, nu: mid_nu, px: mid_px }).unwrap();

        let result = wait_for_result(&rx).expect("must receive a ParametricInferResult");
        assert_eq!(result.e, mid_e);
        assert!(result.in_range, "midpoint of every range must be in_range");
        assert!(result.vis.von_mises.iter().any(|v| v.is_finite()), "vis must contain at least one real value");
        assert!(result.bc_residual_rms.is_finite() && result.bc_residual_rms >= 0.0, "bc_residual_rms must be finite and non-negative");
        assert!(result.bc_residual_max >= result.bc_residual_rms - 1e-6, "bc_residual_max must be >= rms");
        assert!(result.reaction_force.net_fx.is_finite() && result.reaction_force.net_fy.is_finite(), "reaction_force net force must be finite");
        assert!(result.reaction_force.equilibrium_error.is_finite() && result.reaction_force.equilibrium_error >= 0.0, "equilibrium_error must be finite and non-negative");
        assert!(result.energy_balance.energy_balance_error.is_finite() && result.energy_balance.energy_balance_error >= 0.0, "energy_balance_error must be finite and non-negative");
        assert!(result.nearest_sample_distance.is_finite() && result.nearest_sample_distance >= 0.0, "nearest_sample_distance must be finite and non-negative for a non-empty reservoir, got {}", result.nearest_sample_distance);
        assert!(result.typical_sample_spacing.is_finite() && result.typical_sample_spacing >= 0.0, "typical_sample_spacing must be finite and non-negative");

        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn run_training_parametric_out_of_range_query_is_flagged() {
        let spec = tiny_spec(5);
        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let spec_clone = spec.clone();
        let handle = std::thread::spawn(move || run_training_parametric(spec_clone, tx, rx_ctrl));

        assert!(wait_for_ready(&rx), "training did not reach ParametricReady in time");

        tx_ctrl.send(ControlMsg::ParametricInfer { e: spec.e_range.max + 1e10, nu: spec.nu_range.mid(), px: spec.load_range.mid() }).unwrap();
        let result = wait_for_result(&rx).expect("must receive a result");
        assert!(!result.in_range, "E far outside range must be flagged out-of-range");

        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();
    }

    // ─── Stage H: model checkpoint save/load ────────────────────────────────────────────────

    #[test]
    fn serve_loaded_checkpoint_is_ready_immediately_and_answers_inference() {
        // No training at all here - a fresh, never-trained model, loaded exactly as
        // `serve_loaded_checkpoint` would receive one from `checkpoint::load_checkpoint`.
        // Proves the "load from disk, skip training, serve immediately" path actually works
        // end to end, not just that the underlying serving loop (already covered by the
        // run_training_parametric tests above) is correct in isolation.
        let spec = tiny_spec(0);
        let device = BDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(6).with_hidden_dim(spec.network.hidden_dim).with_n_hidden(spec.network.n_hidden).with_output_dim(5);
        let model: ElasticityNet<BInner> = net_cfg.init(&device);

        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let spec_clone = spec.clone();
        let handle = std::thread::spawn(move || serve_loaded_checkpoint(spec_clone, model, tx, rx_ctrl));

        assert!(wait_for_ready(&rx), "a loaded checkpoint must send ParametricReady immediately, no training required");

        let mid_e = spec.e_range.mid();
        let mid_nu = spec.nu_range.mid();
        let mid_px = spec.load_range.mid();
        tx_ctrl.send(ControlMsg::ParametricInfer { e: mid_e, nu: mid_nu, px: mid_px }).unwrap();
        let result = wait_for_result(&rx).expect("must receive a ParametricInferResult from a loaded checkpoint");
        assert_eq!(result.e, mid_e);
        assert!(result.in_range, "midpoint of every range must be in_range");
        // Empty reservoir (no training happened) - nearest_sample_distance must honestly
        // report "no coverage information" (infinite), not a fabricated finite value.
        assert_eq!(result.nearest_sample_distance, f64::INFINITY);
        assert_eq!(result.typical_sample_spacing, 0.0);

        tx_ctrl.send(ControlMsg::Stop).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn serve_loaded_checkpoint_saves_on_request() {
        let spec = tiny_spec(0);
        let device = BDevice::default();
        let net_cfg = ElasticityNetConfig::new()
            .with_input_dim(6).with_hidden_dim(spec.network.hidden_dim).with_n_hidden(spec.network.n_hidden).with_output_dim(5);
        let model: ElasticityNet<BInner> = net_cfg.init(&device);

        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_ctrl, rx_ctrl) = crossbeam_channel::unbounded();
        let handle = std::thread::spawn(move || serve_loaded_checkpoint(spec, model, tx, rx_ctrl));
        assert!(wait_for_ready(&rx), "must reach ParametricReady before a save can be requested");

        let path = std::env::temp_dir().join(format!("pinn_solver_serve_loaded_save_test_{}", std::process::id()));
        tx_ctrl.send(ControlMsg::SaveCheckpoint { path: path.clone(), saved_at_unix: 123 }).unwrap();

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

    /// Wall-clock deadline (not a fixed iteration count) - the default `wgpu` backend's
    /// fixed per-dispatch overhead is real and much larger than `ndarray`'s for tensors this
    /// tiny (confirmed: a fixed 2000x5ms budget flaked under `wgpu` for a 15-step run even
    /// though the same run completed in under 2s under `ndarray-backend`), so tests must not
    /// assume a specific backend's speed.
    fn wait_for_ready(rx: &crossbeam_channel::Receiver<TrainingMsg>) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while std::time::Instant::now() < deadline {
            if let Ok(TrainingMsg::ParametricReady) = rx.try_recv() { return true; }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        false
    }

    fn wait_for_result(rx: &crossbeam_channel::Receiver<TrainingMsg>) -> Option<Box<pinn_core::messages::ParametricInferenceResult>> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while std::time::Instant::now() < deadline {
            if let Ok(TrainingMsg::ParametricInferResult(r)) = rx.try_recv() { return Some(r); }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        None
    }
}
