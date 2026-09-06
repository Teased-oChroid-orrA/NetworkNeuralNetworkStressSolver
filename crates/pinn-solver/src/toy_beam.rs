//! Standalone 1D Euler-Bernoulli beam sanity check — deliberately decoupled from
//! `pinn_core::problem::BoundaryValueProblem`/`DomainSamplingStrategy` (those are
//! irreducibly 2D-plate-shaped; see `pinn-solver::execution`'s own module doc for the
//! same kind of scope-discipline note). Exists purely so the core training methodology
//! (does the network actually minimize the physics residual, or can it collapse to a
//! near-zero/trivial deflection that happens to satisfy the boundary conditions cheaply)
//! can be checked in seconds, not the ~70-90 minutes the real Kirsch/pin-lug default
//! config takes (`MAX_STEPS=28000` at ~150-200ms/step, per Phase 2's own measured
//! per-step timing — that cost is real and expected, not a bug).
//!
//! Problem: `d^4 w/dx^4 = q` on `x in (0,1)`, `E=I=q=1`, two boundary-condition cases
//! with known closed-form solutions (see [`BeamBc::exact_solution`]).
//!
//! Formulation: energy minimization (DEM), not a strong-form residual. A strong-form
//! residual `(d^4 w/dx^4 - q)^2` would need a novel 4th-derivative stencil — nothing
//! above 1st-derivative (strain) exists anywhere else in this codebase. The weak
//! (variational) form only needs the 2nd derivative (curvature `w''`), because
//! integration by parts drops the order by 2, and critically: natural boundary
//! conditions (moment/shear at a free end) are automatically satisfied at the energy
//! minimum and need no explicit loss term at all — only essential boundary conditions
//! (displacement/slope) need enforcing, via the same hard-ansatz pattern
//! `pinn_core::problem::DirichletAnsatz` already establishes ("return a scale factor the
//! raw network output gets multiplied by"): `w(x) = p(x) * NN(x)` for a polynomial `p`
//! chosen so the essential BCs hold at every `x`, not just approximately.
//!
//! Reuses `ElasticityNetConfig`/`ElasticityNet` directly (its `forward` has no hardcoded
//! input/output width — only the optional Fourier embedding assumes 3 input columns, and
//! it's skipped when `n_fourier=0`) and plain `burn::optim::AdamW` (not
//! `pinn_solver::optim`'s SOAP-Muon weight/bias split, which exists for elasticity's own
//! rationale and is irrelevant here).

use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};

use crate::network::{ElasticityNet, ElasticityNetConfig};
use crate::training_core::{sync_device, BDevice, B};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeamBc {
    /// Clamped-free: `w(0)=0, w'(0)=0` (essential) / `w''(1)=0, w'''(1)=0` (natural).
    Cantilever,
    /// Pinned-pinned: `w(0)=w(1)=0` (essential) / `w''(0)=w''(1)=0` (natural).
    SimplySupported,
}

impl BeamBc {
    /// `p(x)` in `w(x) = p(x) * NN(x)` — satisfies this case's essential BCs exactly,
    /// for any `NN`, at every `x` (not penalized, not approximate).
    fn essential_factor(self, x: f32) -> f32 {
        match self {
            BeamBc::Cantilever => x * x,
            BeamBc::SimplySupported => x * (1.0 - x),
        }
    }

    /// Closed-form exact deflection, user-provided, used as the pass/fail oracle.
    pub fn exact_solution(self, x: f64) -> f64 {
        match self {
            BeamBc::Cantilever => (1.0 / 24.0) * x * x * (x * x - 4.0 * x + 6.0),
            BeamBc::SimplySupported => (1.0 / 24.0) * x * (x * x * x - 2.0 * x * x + 1.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ToyBeamResult {
    pub final_loss: f64,
    /// Max |w_net(x) - w_exact(x)| over a fixed evaluation grid, x in [0.05, 1.0].
    pub max_abs_error: f64,
    /// Max |w_net(x)| over the same grid — the actual trivial-solution check: a genuine
    /// w=0 collapse makes this ~0 regardless of how `max_abs_error` is computed.
    pub max_abs_deflection: f64,
    /// `(x, w_net(x), w_exact(x))` at each evaluation-grid point — lets a caller (e.g. the
    /// `toy_beam` example) print a side-by-side comparison without re-running training.
    pub eval_points: Vec<(f64, f64, f64)>,
}

/// Evaluates `p(x) * NN(x)` for a batch of `x` values in one forward pass. `xs` must be
/// non-empty. Returns a `[xs.len()]` plain `Vec<f32>` (already-detached scalar values —
/// callers doing further autodiff work should use [`forward_tensor`] instead).
fn eval_batch(model: &ElasticityNet<B>, bc: BeamBc, xs: &[f32], device: &BDevice) -> Vec<f32> {
    let raw = forward_tensor(model, bc, xs, device);
    raw.into_data().to_vec::<f32>().unwrap()
}

/// `p(x) * NN(x)` for a batch of `x` values, kept as a tensor (autodiff graph intact) —
/// the shared core of both training (needs the graph) and evaluation (doesn't).
fn forward_tensor(model: &ElasticityNet<B>, bc: BeamBc, xs: &[f32], device: &BDevice) -> Tensor<B, 2> {
    let n = xs.len();
    let input = Tensor::<B, 2>::from_data(TensorData::new(xs.to_vec(), vec![n, 1]), device);
    let raw_out = model.forward(input);
    let factors: Vec<f32> = xs.iter().map(|&x| bc.essential_factor(x)).collect();
    let factor_t = Tensor::<B, 2>::from_data(TensorData::new(factors, vec![n, 1]), device);
    raw_out * factor_t
}

/// Trains a tiny network to minimize the beam's discretized potential energy
/// `Pi[w] ~= sum_i [ 0.5*w''(x_i)^2 - q*w(x_i) ] * dx` over `n_points` midpoint-rule
/// collocation points in `(0,1)`, `q=1`. `w''` via a 3-point central stencil (step `h`),
/// mirroring `fd_stencil::assemble_stencil`'s "stack all offset points into one batched
/// forward pass, then combine" pattern — just 1D/3-point here instead of 2D/5-point.
pub fn train_toy_beam(
    bc: BeamBc,
    steps: usize,
    n_points: usize,
    hidden_dim: usize,
    n_hidden: usize,
) -> ToyBeamResult {
    let device = BDevice::default();
    let h: f32 = 1e-2;
    let dx = 1.0_f64 / n_points as f64;

    let net_cfg = ElasticityNetConfig::new()
        .with_input_dim(1)
        .with_hidden_dim(hidden_dim)
        .with_n_hidden(n_hidden)
        .with_output_dim(1);
    let mut model: ElasticityNet<B> = net_cfg.init(&device);
    let mut optimizer = AdamWConfig::new().init();
    let lr = 1e-3;

    // Midpoint-rule collocation points, fixed (no RNG needed for a 1D deterministic grid).
    let centers: Vec<f32> = (0..n_points)
        .map(|i| (i as f32 + 0.5) / n_points as f32)
        .collect();
    let mut stencil_xs = Vec::with_capacity(3 * n_points);
    stencil_xs.extend(centers.iter().map(|&x| x - h));
    stencil_xs.extend(centers.iter().copied());
    stencil_xs.extend(centers.iter().map(|&x| x + h));

    let mut final_loss = 0.0_f64;
    for _ in 0..steps {
        let w_all = forward_tensor(&model, bc, &stencil_xs, &device); // [3*n, 1]
        let w_minus = w_all.clone().slice([0..n_points, 0..1]);
        let w_center = w_all.clone().slice([n_points..2 * n_points, 0..1]);
        let w_plus = w_all.slice([2 * n_points..3 * n_points, 0..1]);

        let w_pp = (w_minus - w_center.clone().mul_scalar(2.0_f64) + w_plus)
            .div_scalar((h * h) as f64);
        let energy_density = w_pp.powf_scalar(2.0_f64).mul_scalar(0.5_f64) - w_center;
        let loss = energy_density.sum().mul_scalar(dx);

        final_loss = loss.clone().into_scalar() as f64;
        let grads_raw = loss.backward();
        let grads = GradientsParams::from_grads(grads_raw, &model);
        model = optimizer.step(lr, model, grads);
    }
    sync_device(&device);

    let eval_xs: Vec<f32> = (1..=20).map(|i| i as f32 * 0.05).collect(); // [0.05 .. 1.0]
    let net_vals = eval_batch(&model, bc, &eval_xs, &device);
    let mut max_abs_error = 0.0_f64;
    let mut max_abs_deflection = 0.0_f64;
    let mut eval_points = Vec::with_capacity(eval_xs.len());
    for (&x, &w) in eval_xs.iter().zip(net_vals.iter()) {
        let exact = bc.exact_solution(x as f64);
        max_abs_error = max_abs_error.max((w as f64 - exact).abs());
        max_abs_deflection = max_abs_deflection.max((w as f64).abs());
        eval_points.push((x as f64, w as f64, exact));
    }

    ToyBeamResult { final_loss, max_abs_error, max_abs_deflection, eval_points }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `#[ignore]`: this test's ~15s Wgpu training loop, run concurrently with the rest of
    // `cargo test --workspace`'s default parallel execution, was observed to tip two
    // separate byte-exact `training_core` oracle tests (thin-margin, ~0.01-0.1% relative
    // error assertions) into spurious failure across two different full-suite runs — each
    // failing test passed cleanly standalone. Consistent with this project's own documented
    // Wgpu weight-init non-determinism under added GPU contention, not a toy_beam logic bug.
    // Run explicitly: `cargo test -p pinn-solver toy_beam:: -- --ignored`.
    #[test]
    #[ignore]
    fn toy_beam_cantilever_converges_near_exact_and_is_not_trivial() {
        let result = train_toy_beam(BeamBc::Cantilever, 3000, 32, 48, 3);
        assert!(
            result.max_abs_error < 0.01,
            "cantilever max_abs_error too large: {result:?}"
        );
        // Exact tip deflection (x=1) is 0.125 — a genuine w=0 collapse would make
        // max_abs_deflection ~0, which would ALSO show up as max_abs_error ~0.125 (since
        // exact deflection is that large near x=1), but assert the floor directly so this
        // check doesn't rely on the error-tolerance math happening to catch it.
        assert!(
            result.max_abs_deflection > 0.02,
            "suspiciously small deflection, possible trivial-solution collapse: {result:?}"
        );
    }

    // See the `#[ignore]` note on `toy_beam_cantilever_converges_near_exact_and_is_not_trivial`.
    #[test]
    #[ignore]
    fn toy_beam_simply_supported_converges_near_exact_and_is_not_trivial() {
        let result = train_toy_beam(BeamBc::SimplySupported, 3000, 32, 48, 3);
        assert!(
            result.max_abs_error < 0.005,
            "simply-supported max_abs_error too large: {result:?}"
        );
        // Exact max deflection (near x=0.5) is ~0.013 — smaller than the cantilever case,
        // so the floor is set tighter accordingly.
        assert!(
            result.max_abs_deflection > 0.003,
            "suspiciously small deflection, possible trivial-solution collapse: {result:?}"
        );
    }

    #[test]
    fn essential_factor_vanishes_at_required_boundaries() {
        for bc in [BeamBc::Cantilever, BeamBc::SimplySupported] {
            assert_eq!(bc.essential_factor(0.0), 0.0);
        }
        assert_eq!(BeamBc::SimplySupported.essential_factor(1.0), 0.0);
    }

    #[test]
    fn exact_solution_matches_user_provided_formulas_at_known_points() {
        // Cantilever tip (x=1): (1/24)*1*(1-4+6) = 3/24 = 0.125
        assert!((BeamBc::Cantilever.exact_solution(1.0) - 0.125).abs() < 1e-12);
        // Simply-supported midspan (x=0.5): (1/24)*0.5*(0.125-0.5+1) = 0.013020833...
        assert!((BeamBc::SimplySupported.exact_solution(0.5) - 0.0130208333333).abs() < 1e-10);
        // Both cases: w(0) = 0 exactly.
        assert_eq!(BeamBc::Cantilever.exact_solution(0.0), 0.0);
        assert_eq!(BeamBc::SimplySupported.exact_solution(0.0), 0.0);
    }
}
