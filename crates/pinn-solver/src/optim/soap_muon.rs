//! Hybrid SOAP + Muon optimizer for 2D weight matrices.
//!
//! Ported from the reference implementations at `github.com/nikhilvyas/SOAP` (`soap.py`)
//! and `github.com/nikhilvyas/SOAP_MUON` (`nanogpt_optimizer.py`), per "Improving SOAP
//! Using Iterative Whitening and Muon" (Vyas et al.).
//!
//! Each step: SOAP maintains per-dimension Shampoo-style preconditioners (`gg0`, `gg1`,
//! an EMA of `grad @ grad^T` / `grad^T @ grad`), projects the gradient into the current
//! eigenbasis (`q0`, `q1`), runs a standard Adam moment update in that rotated space, and
//! projects back. The result is then passed through Muon's Newton-Schulz orthogonalization
//! as an iterative-whitening refinement pass — this is the paper's actual hybrid mechanism,
//! not a per-dimension split between the two optimizers.
//!
//! Simplification vs. the original SOAP: the eigenbasis is refreshed via a full
//! eigendecomposition every `precondition_frequency` steps rather than SOAP's cheaper
//! power-iteration+QR incremental update. That optimization exists in the reference
//! implementation to amortize cost on LLM-scale matrices (thousands of rows/columns); this
//! network's weight matrices are at most `hidden_dim`×`hidden_dim` (≤ 256×256), where a full
//! eigh is microseconds on CPU, so the approximation isn't needed here.
//!
//! Only valid for 2D parameters (weight matrices) — biases/1D params must be optimized
//! separately (e.g. with `AdamW`). Panics if `step` is called with `D != 2`, mirroring
//! burn's own native `Muon` optimizer.

use burn::{
    config::Config,
    optim::{SimpleOptimizer, adaptor::OptimizerAdaptor},
    record::Record,
    tensor::{Tensor, TensorData, backend::Backend, backend::AutodiffBackend, ops::Device},
};
use burn::module::AutodiffModule;
use burn::optim::LearningRate;

/// Configuration for the [`SoapMuon`] hybrid optimizer.
#[derive(Config, Debug)]
pub struct SoapMuonConfig {
    /// First-moment decay for the Adam update performed in SOAP's rotated eigenbasis.
    #[config(default = 0.95)]
    pub beta1: f32,
    /// Second-moment decay for the Adam update performed in SOAP's rotated eigenbasis.
    #[config(default = 0.95)]
    pub beta2: f32,
    /// EMA decay for the Shampoo-style preconditioner accumulators (`gg0`/`gg1`).
    #[config(default = 0.95)]
    pub shampoo_beta: f32,
    /// Numerical-stability epsilon for the Adam update.
    #[config(default = 1e-8)]
    pub eps: f32,
    /// Refresh the eigenbasis every this many steps.
    #[config(default = 10)]
    pub precondition_frequency: usize,
    /// Decoupled (AdamW-style) weight decay applied to the parameter directly.
    pub weight_decay: Option<f32>,
    /// Newton-Schulz iteration coefficients (a, b, c) — defaults match burn's native Muon.
    #[config(default = "(3.4445, -4.775, 2.0315)")]
    pub ns_coefficients: (f32, f32, f32),
    /// Number of Newton-Schulz iteration steps.
    #[config(default = 5)]
    pub ns_steps: usize,
    /// Epsilon used when normalizing by the Frobenius norm in the Newton-Schulz pass.
    #[config(default = 1e-7)]
    pub ns_eps: f32,
}

impl SoapMuonConfig {
    /// Build a [`SoapMuon`] from the config.
    pub fn build<B: Backend>(&self) -> SoapMuon<B> {
        SoapMuon {
            beta1: self.beta1,
            beta2: self.beta2,
            shampoo_beta: self.shampoo_beta,
            eps: self.eps,
            precondition_frequency: self.precondition_frequency.max(1),
            weight_decay: self.weight_decay,
            ns_coefficients: self.ns_coefficients,
            ns_steps: self.ns_steps,
            ns_eps: self.ns_eps,
            _marker: std::marker::PhantomData,
        }
    }

    /// Initialize the SOAP-Muon optimizer, wrapped in burn's `OptimizerAdaptor`.
    ///
    /// Only feed this optimizer gradients for 2D weight-matrix parameters — partition
    /// `GradientsParams` before calling `.step()` so biases go through a separate `AdamW`.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<SoapMuon<B::InnerBackend>, M, B> {
        OptimizerAdaptor::from(self.build())
    }
}

/// Hybrid SOAP + Muon optimizer. See module docs for the algorithm.
pub struct SoapMuon<B: Backend> {
    beta1: f32,
    beta2: f32,
    shampoo_beta: f32,
    eps: f32,
    precondition_frequency: usize,
    weight_decay: Option<f32>,
    ns_coefficients: (f32, f32, f32),
    ns_steps: usize,
    ns_eps: f32,
    _marker: std::marker::PhantomData<B>,
}

// Manual Clone impl: every field is `Copy` regardless of `B`, so this avoids over-constraining
// with a `B: Clone` bound the way `#[derive(Clone)]` would.
impl<B: Backend> Clone for SoapMuon<B> {
    fn clone(&self) -> Self {
        Self {
            beta1: self.beta1,
            beta2: self.beta2,
            shampoo_beta: self.shampoo_beta,
            eps: self.eps,
            precondition_frequency: self.precondition_frequency,
            weight_decay: self.weight_decay,
            ns_coefficients: self.ns_coefficients,
            ns_steps: self.ns_steps,
            ns_eps: self.ns_eps,
            _marker: std::marker::PhantomData,
        }
    }
}

/// State carried between steps for one 2D weight-matrix parameter.
#[derive(Record, Clone)]
pub struct SoapMuonState<B: Backend> {
    pub gg0: Tensor<B, 2>,
    pub gg1: Tensor<B, 2>,
    pub q0: Tensor<B, 2>,
    pub q1: Tensor<B, 2>,
    pub exp_avg: Tensor<B, 2>,
    pub exp_avg_sq: Tensor<B, 2>,
    pub step: usize,
}

impl<B: Backend> SoapMuonState<B> {
    fn init(d0: usize, d1: usize, device: &Device<B>) -> Self {
        Self {
            gg0: Tensor::zeros([d0, d0], device),
            gg1: Tensor::zeros([d1, d1], device),
            q0: Tensor::eye(d0, device),
            q1: Tensor::eye(d1, device),
            exp_avg: Tensor::zeros([d0, d1], device),
            exp_avg_sq: Tensor::zeros([d0, d1], device),
            step: 0,
        }
    }
}

/// Full eigendecomposition of a symmetric matrix via `nalgebra`, returning the orthonormal
/// eigenvector matrix (eigenvalues sorted descending for run-to-run ordering stability).
///
/// `burn-tensor` has no eigendecomposition primitive, so this round-trips the (small,
/// `hidden_dim`×`hidden_dim`) matrix through the CPU.
fn eigh_eigenvectors<B: Backend>(m: &Tensor<B, 2>, device: &Device<B>) -> Tensor<B, 2> {
    let [n, _] = m.dims();
    let data = m.clone().into_data().to_vec::<f32>().unwrap();
    let mat = nalgebra::DMatrix::<f32>::from_row_slice(n, n, &data);
    let eig = nalgebra::SymmetricEigen::new(mat);

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = vec![0f32; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            out[row * n + new_col] = eig.eigenvectors[(row, old_col)];
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(out, vec![n, n]), device)
}

/// Newton-Schulz quintic orthogonalization (Keller Jordan's iteration, as used by Muon).
/// Mirrors burn's native `Muon::zeropower_via_newtonschulz`, applied here to SOAP's
/// project-Adam-project-back update rather than a raw momentum-smoothed gradient.
fn newton_schulz<B: Backend>(
    g: Tensor<B, 2>,
    coefficients: (f32, f32, f32),
    steps: usize,
    eps: f32,
) -> Tensor<B, 2> {
    let [d0, d1] = g.dims();
    let (mut x, needs_transpose) = if d0 > d1 {
        (g.transpose(), true)
    } else {
        (g, false)
    };

    let norm = x
        .clone()
        .powf_scalar(2.0)
        .sum()
        .sqrt()
        .clamp_min(eps)
        .unsqueeze();
    x = x.div(norm);

    let (a, b, c) = coefficients;
    for _ in 0..steps {
        let x_t = x.clone().transpose();
        let a_matrix = x.clone().matmul(x_t);
        let a_squared = a_matrix.clone().matmul(a_matrix.clone());
        let b_matrix = a_matrix.mul_scalar(b).add(a_squared.mul_scalar(c));
        x = x.clone().mul_scalar(a).add(b_matrix.matmul(x.clone()));
    }

    if needs_transpose { x.transpose() } else { x }
}

impl<B: Backend> SimpleOptimizer<B> for SoapMuon<B> {
    type State<const D: usize> = SoapMuonState<B>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        assert!(
            D == 2,
            "SoapMuon requires 2D weight-matrix tensors, got {D}D — route 1D/biases through a \
             separate optimizer (e.g. AdamW) by partitioning GradientsParams before calling step()"
        );

        let device = tensor.device();
        // SAFETY (type-level only): `K::Primitive` is rank-erased, and we've just asserted
        // D == 2 above, so this relabel is exact — same pattern `GradientsParams` itself
        // uses internally to store/retrieve tensors of varying rank.
        let t2: Tensor<B, 2> = Tensor::from_primitive(tensor.into_primitive());
        let g2: Tensor<B, 2> = Tensor::from_primitive(grad.into_primitive());
        let [d0, d1] = t2.dims();

        let mut st = state.unwrap_or_else(|| SoapMuonState::init(d0, d1, &device));

        // 1. EMA-update the Shampoo-style preconditioner accumulators.
        let gb = self.shampoo_beta;
        st.gg0 = st.gg0.mul_scalar(gb).add(
            g2.clone().matmul(g2.clone().transpose()).mul_scalar(1.0 - gb),
        );
        st.gg1 = st.gg1.mul_scalar(gb).add(
            g2.clone().transpose().matmul(g2.clone()).mul_scalar(1.0 - gb),
        );
        st.step += 1;

        // 2. Refresh the eigenbasis periodically (full eigh — see module docs).
        if st.step % self.precondition_frequency == 0 {
            st.q0 = eigh_eigenvectors(&st.gg0, &device);
            st.q1 = eigh_eigenvectors(&st.gg1, &device);
        }

        // 3. Project gradient into the eigenbasis: q0^T @ grad @ q1
        let grad_proj = st.q0.clone().transpose().matmul(g2.clone()).matmul(st.q1.clone());

        // 4. Adam moment update in the rotated space (bias-corrected, as in burn's Adam).
        let b1 = self.beta1;
        let b2 = self.beta2;
        st.exp_avg = st.exp_avg.clone().mul_scalar(b1).add(grad_proj.clone().mul_scalar(1.0 - b1));
        st.exp_avg_sq = st
            .exp_avg_sq
            .clone()
            .mul_scalar(b2)
            .add(grad_proj.clone().powf_scalar(2.0).mul_scalar(1.0 - b2));

        let t = st.step as i32;
        let bias_correction2_sqrt = (1.0 - b2.powi(t)).sqrt();
        let combined_factor = bias_correction2_sqrt / (1.0 - b1.powi(t));
        let update_rotated = st
            .exp_avg
            .clone()
            .mul_scalar(combined_factor)
            .div(st.exp_avg_sq.clone().sqrt().add_scalar(self.eps * bias_correction2_sqrt));

        // 5. Project back: q0 @ update_rotated @ q1^T
        let update = st.q0.clone().matmul(update_rotated).matmul(st.q1.clone().transpose());

        // 6. Newton-Schulz refinement pass (the paper's "iterative whitening" contribution).
        let update = newton_schulz(update, self.ns_coefficients, self.ns_steps, self.ns_eps);

        // 7. Optional decoupled (AdamW-style) weight decay, then apply the update.
        let t2 = if let Some(wd) = self.weight_decay {
            t2.mul_scalar(1.0 - lr * wd as f64)
        } else {
            t2
        };
        let new_tensor = t2 - update.mul_scalar(lr);

        let new_tensor_d: Tensor<B, D> = Tensor::from_primitive(new_tensor.into_primitive());
        (new_tensor_d, Some(st))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.gg0 = state.gg0.to_device(device);
        state.gg1 = state.gg1.to_device(device);
        state.q0 = state.q0.to_device(device);
        state.q1 = state.q1.to_device(device);
        state.exp_avg = state.exp_avg.to_device(device);
        state.exp_avg_sq = state.exp_avg_sq.to_device(device);
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Wgpu};

    type TB = Autodiff<Wgpu>;

    #[test]
    fn newton_schulz_orthogonalizes() {
        // Same reference matrix and tolerance as burn's own native Muon test
        // (`test_newton_schulz_orthogonalization`) — 5 quintic NS steps drive the diagonal
        // of `ortho @ ortho^T` toward 1.0 but, for an anisotropic 2x2 input like this one,
        // don't fully converge off-diagonal entries to 0 in that few steps. That's expected
        // behavior for a fixed-iteration-count approximation, not a bug.
        let device = Default::default();
        let matrix = Tensor::<TB, 2>::from_floats([[1.0, 0.5], [0.5, 1.0]], &device);
        let ortho = newton_schulz(matrix, (3.4445, -4.775, 2.0315), 5, 1e-7);
        let product = ortho.clone().matmul(ortho.transpose());
        let data = product.into_data();
        let values = data.as_slice::<f32>().unwrap();
        assert!((values[0] - 1.0).abs() < 0.1, "expected ~1.0, got {}", values[0]);
        assert!((values[3] - 1.0).abs() < 0.1, "expected ~1.0, got {}", values[3]);
    }

    #[test]
    fn eigh_eigenvectors_orthonormal() {
        let device = Default::default();
        let m = Tensor::<TB, 2>::from_floats([[2.0, 1.0], [1.0, 2.0]], &device);
        let q = eigh_eigenvectors(&m, &device);
        let product = q.clone().matmul(q.transpose());
        let data = product.into_data();
        let values = data.as_slice::<f32>().unwrap();
        assert!((values[0] - 1.0).abs() < 1e-4, "Q^T Q [0,0] should be ~1.0, got {}", values[0]);
        assert!((values[3] - 1.0).abs() < 1e-4, "Q^T Q [1,1] should be ~1.0, got {}", values[3]);
        assert!(values[1].abs() < 1e-4, "Q^T Q off-diagonal should be ~0, got {}", values[1]);
    }

    #[test]
    #[should_panic(expected = "SoapMuon requires 2D")]
    fn panics_on_1d_tensor() {
        let device = Default::default();
        let optim: SoapMuon<Wgpu> = SoapMuonConfig::new().build();
        let t = Tensor::<Wgpu, 1>::zeros([8], &device);
        let g = Tensor::<Wgpu, 1>::ones([8], &device);
        let _ = optim.step(0.01, t, g, None);
    }

    #[test]
    fn multi_step_changes_weights_without_nan() {
        let device = Default::default();
        let optim: SoapMuon<Wgpu> = SoapMuonConfig::new()
            .with_precondition_frequency(2)
            .build();

        let mut tensor = Tensor::<Wgpu, 2>::from_floats(
            [[1.0, 0.5, -0.3, 0.2], [0.5, 1.0, 0.1, -0.4], [0.2, 0.1, 1.0, 0.3]],
            &device,
        );
        let initial = tensor.clone().into_data();
        let mut state = None;

        for i in 0..8 {
            let grad = Tensor::<Wgpu, 2>::from_floats(
                [[0.1, -0.2, 0.05, 0.3], [0.2, 0.1, -0.1, 0.05], [-0.05, 0.15, 0.2, -0.1]],
                &device,
            )
            .mul_scalar(1.0 + i as f32 * 0.1);
            let (new_tensor, new_state) = optim.step(0.01, tensor, grad, state);
            tensor = new_tensor;
            state = new_state;
        }

        let final_data = tensor.into_data();
        let final_vals = final_data.as_slice::<f32>().unwrap();
        let initial_vals = initial.as_slice::<f32>().unwrap();

        for v in final_vals {
            assert!(!v.is_nan(), "weight became NaN after SOAP-Muon steps");
        }
        assert_ne!(
            final_vals, initial_vals,
            "weights should change after 8 optimizer steps"
        );
        assert!(state.is_some(), "state should be returned after a step");
    }
}
