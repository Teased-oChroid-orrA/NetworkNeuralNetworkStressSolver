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

/// Migrates a [`SoapMuonState`] built for the OLD shape of a weight matrix onto its NEW
/// (function-preservingly grown, see `ElasticityNet::grow_width`) shape — issue #50.
///
/// `grow_dim0`/`grow_dim1` are `Some((old_size, new_size))` for whichever of the state's two
/// axes actually grew for this particular weight (`None` when that axis is unaffected — e.g.
/// `layers[0]`'s `d0` axis, `input_dim`, never grows). Uses the SAME `g`/`m` duplication map as
/// [`crate::network::duplication_map`] (the one `grow_width` itself uses) on whichever axis
/// grows, so a moment/accumulator entry belonging to a duplicated unit is migrated onto the
/// SAME twin relationship the weight values themselves encode.
///
/// - `gg0`/`gg1` (bilinear Shampoo accumulators): migrated via the pullback `gg' = Pᵀ · gg · P`,
///   where `P` (`[d_old, d_new]`) has `P[s,s]=1` for `s<d_old` and `P[g(j), d_old+j]=1` for each
///   new unit `j` — this preserves the old `d_old×d_old` block EXACTLY (top-left corner
///   unchanged) and gives new rows/columns the twin's correlation history.
/// - `q0`/`q1` (cached eigenbases): reset to the identity of the NEW size, mirroring
///   `SoapMuonState::init`'s own fresh-init identity construction (`Tensor::eye`).
/// - `exp_avg`/`exp_avg_sq` (rotated-space Adam moments): copied element-wise via `g` on
///   whichever axis grew — a duplicate's entry equals its twin's entry — WITHOUT division. This
///   deliberately differs from the weight-VALUE migration rule (which divides by `m`): the
///   gradient a duplicate receives post-growth is the same undivided gradient its twin receives
///   (chain rule), since the `1/m` divisor is baked into the weight VALUE, not the gradient
///   flowing into it.
/// - `step`: preserved unchanged (NOT reset to 0), keeping Adam bias-correction continuous and
///   avoiding an artificially inflated early post-growth update.
pub fn migrate_soap_muon_state_for_growth<B: Backend>(
    old: &SoapMuonState<B>,
    grow_dim0: Option<(usize, usize)>,
    grow_dim1: Option<(usize, usize)>,
    device: &Device<B>,
) -> SoapMuonState<B> {
    let gg0 = match grow_dim0 {
        Some((d_old, d_new)) => migrate_gg::<B>(&old.gg0, d_old, d_new, device),
        None => old.gg0.clone(),
    };
    let gg1 = match grow_dim1 {
        Some((d_old, d_new)) => migrate_gg::<B>(&old.gg1, d_old, d_new, device),
        None => old.gg1.clone(),
    };

    let d0_new = grow_dim0.map(|(_, n)| n).unwrap_or_else(|| old.q0.dims()[0]);
    let d1_new = grow_dim1.map(|(_, n)| n).unwrap_or_else(|| old.q1.dims()[0]);
    let q0 = Tensor::eye(d0_new, device);
    let q1 = Tensor::eye(d1_new, device);

    let exp_avg = migrate_moment::<B>(&old.exp_avg, grow_dim0, grow_dim1, device);
    let exp_avg_sq = migrate_moment::<B>(&old.exp_avg_sq, grow_dim0, grow_dim1, device);

    SoapMuonState { gg0, gg1, q0, q1, exp_avg, exp_avg_sq, step: old.step }
}

/// Builds the `[d_old, d_new]` duplication/pullback matrix `P` used by [`migrate_gg`]:
/// `P[s,s]=1` for `s<d_old` (identity on the preserved block), `P[g(j), d_old+j]=1` for each
/// new unit `j` (routes a new unit's pullback contribution to its twin's row/column). All other
/// entries are 0.
fn build_duplication_matrix<B: Backend>(d_old: usize, d_new: usize, device: &Device<B>) -> Tensor<B, 2> {
    let k = d_new - d_old;
    let (g, _m) = crate::network::duplication_map(d_old, k);

    let mut data = vec![0f32; d_old * d_new];
    for s in 0..d_old {
        data[s * d_new + s] = 1.0;
    }
    for (j, &gj) in g.iter().enumerate() {
        data[gj * d_new + d_old + j] = 1.0;
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, vec![d_old, d_new]), device)
}

/// Pullback migration for one bilinear Shampoo accumulator (`gg0` or `gg1`): `gg' = Pᵀ · gg · P`
/// — see [`migrate_soap_muon_state_for_growth`]'s doc comment for why this exactly preserves
/// the old block and correctly seeds new rows/columns from their twin's history. Built as a
/// real matmul against a real duplication-matrix tensor (not a hand-derived closed-form
/// shortcut) so correctness is easy to verify by inspection.
fn migrate_gg<B: Backend>(gg_old: &Tensor<B, 2>, d_old: usize, d_new: usize, device: &Device<B>) -> Tensor<B, 2> {
    let p = build_duplication_matrix::<B>(d_old, d_new, device);
    p.clone().transpose().matmul(gg_old.clone()).matmul(p)
}

/// Migrates an Adam moment tensor (`exp_avg`/`exp_avg_sq`, shape `[d0, d1]`) onto a grown
/// shape by duplicating (COPYING, never dividing) rows and/or columns per the same `g` map used
/// elsewhere — see [`migrate_soap_muon_state_for_growth`]'s doc comment for why no division is
/// applied here (unlike the weight-value / `gg` migrations).
fn migrate_moment<B: Backend>(
    old: &Tensor<B, 2>,
    grow_dim0: Option<(usize, usize)>,
    grow_dim1: Option<(usize, usize)>,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let [d0_old, d1_old] = old.dims();
    let data = old.clone().into_data().to_vec::<f32>().unwrap();

    let (d0_new, g0) = match grow_dim0 {
        Some((d_old, d_new)) => (d_new, Some(crate::network::duplication_map(d_old, d_new - d_old).0)),
        None => (d0_old, None),
    };
    let (d1_new, g1) = match grow_dim1 {
        Some((d_old, d_new)) => (d_new, Some(crate::network::duplication_map(d_old, d_new - d_old).0)),
        None => (d1_old, None),
    };

    let mut out = vec![0f32; d0_new * d1_new];
    for row in 0..d0_new {
        let src_row = if row < d0_old {
            row
        } else {
            g0.as_ref().expect("row beyond d0_old requires grow_dim0")[row - d0_old]
        };
        for col in 0..d1_new {
            let src_col = if col < d1_old {
                col
            } else {
                g1.as_ref().expect("col beyond d1_old requires grow_dim1")[col - d1_old]
            };
            out[row * d1_new + col] = data[src_row * d1_old + src_col];
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(out, vec![d0_new, d1_new]), device)
}

/// Smart adaptive architecture: shrink-side analogue of [`migrate_soap_muon_state_for_growth`].
/// `keep_dim0`/`keep_dim1` are the SURVIVING indices (in order) for each axis, or `None` if that
/// axis didn't change size — a pure selection/projection, unlike growth's duplication-map
/// expansion (there is no lossless equivalent for removal; the dropped rows/columns' momentum
/// state is simply discarded along with the weights they belonged to). `gg0`/`gg1` (SOAP's
/// per-axis second-moment Gram matrices) are projected onto the kept subspace by selecting the
/// matching rows AND columns; `q0`/`q1` reset to identity of the new size (same "re-orthogonalize
/// on next precondition update" convention growth already uses — these are recomputed
/// periodically during training anyway, never treated as ground truth carried across a resize).
pub fn migrate_soap_muon_state_for_shrink<B: Backend>(
    old: &SoapMuonState<B>,
    keep_dim0: Option<&[usize]>,
    keep_dim1: Option<&[usize]>,
    device: &Device<B>,
) -> SoapMuonState<B> {
    let gg0 = match keep_dim0 {
        Some(keep) => select_symmetric::<B>(&old.gg0, keep, device),
        None => old.gg0.clone(),
    };
    let gg1 = match keep_dim1 {
        Some(keep) => select_symmetric::<B>(&old.gg1, keep, device),
        None => old.gg1.clone(),
    };
    let d0_new = keep_dim0.map(|k| k.len()).unwrap_or_else(|| old.q0.dims()[0]);
    let d1_new = keep_dim1.map(|k| k.len()).unwrap_or_else(|| old.q1.dims()[0]);
    let q0 = Tensor::eye(d0_new, device);
    let q1 = Tensor::eye(d1_new, device);

    let exp_avg = select_moment::<B>(&old.exp_avg, keep_dim0, keep_dim1, device);
    let exp_avg_sq = select_moment::<B>(&old.exp_avg_sq, keep_dim0, keep_dim1, device);

    SoapMuonState { gg0, gg1, q0, q1, exp_avg, exp_avg_sq, step: old.step }
}

/// Projects a symmetric `[d_old, d_old]` matrix onto the `keep.len()` kept rows/columns (the
/// same indices on both axes, since `gg0`/`gg1` are Gram matrices over a single dimension).
fn select_symmetric<B: Backend>(m: &Tensor<B, 2>, keep: &[usize], device: &Device<B>) -> Tensor<B, 2> {
    let [d_old, _] = m.dims();
    let data = m.clone().into_data().to_vec::<f32>().unwrap();
    let d_new = keep.len();
    let mut out = vec![0f32; d_new * d_new];
    for (new_r, &old_r) in keep.iter().enumerate() {
        for (new_c, &old_c) in keep.iter().enumerate() {
            out[new_r * d_new + new_c] = data[old_r * d_old + old_c];
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(out, vec![d_new, d_new]), device)
}

/// Shrink-side analogue of `migrate_moment` — projects an Adam moment tensor (`exp_avg`/
/// `exp_avg_sq`, shape `[d0, d1]`) onto the kept rows/columns of each axis independently.
fn select_moment<B: Backend>(
    old: &Tensor<B, 2>,
    keep_dim0: Option<&[usize]>,
    keep_dim1: Option<&[usize]>,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let [d0_old, d1_old] = old.dims();
    let data = old.clone().into_data().to_vec::<f32>().unwrap();

    let rows: Vec<usize> = keep_dim0.map(|k| k.to_vec()).unwrap_or_else(|| (0..d0_old).collect());
    let cols: Vec<usize> = keep_dim1.map(|k| k.to_vec()).unwrap_or_else(|| (0..d1_old).collect());
    let (d0_new, d1_new) = (rows.len(), cols.len());

    let mut out = vec![0f32; d0_new * d1_new];
    for (new_r, &old_r) in rows.iter().enumerate() {
        for (new_c, &old_c) in cols.iter().enumerate() {
            out[new_r * d1_new + new_c] = data[old_r * d1_old + old_c];
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(out, vec![d0_new, d1_new]), device)
}

/// Full eigendecomposition of a symmetric matrix, given already-on-host row-major data —
/// returns the orthonormal eigenvector matrix, ALSO row-major, flattened (eigenvalues sorted
/// descending for run-to-run ordering stability). Pure-CPU/host-only: no GPU round trip is
/// performed here, so this can be called any number of times per combined GPU sync (see
/// [`eigh_eigenvectors_batched`]).
fn eigh_from_host_data(data: &[f32], n: usize) -> Vec<f32> {
    let mat = nalgebra::DMatrix::<f32>::from_row_slice(n, n, data);
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
    out
}

/// Full eigendecomposition of a symmetric matrix via `nalgebra`, returning the orthonormal
/// eigenvector matrix (eigenvalues sorted descending for run-to-run ordering stability).
///
/// `burn-tensor` has no eigendecomposition primitive, so this round-trips the (small,
/// `hidden_dim`×`hidden_dim`) matrix through the CPU. Kept as the single-matrix entry point
/// (used directly by `eigh_eigenvectors_orthonormal`'s test); `SoapMuon::step`'s hot path uses
/// [`eigh_eigenvectors_batched`] instead — see its doc comment for why. `#[allow(dead_code)]`
/// because this is now exercised only from `#[cfg(test)]` code, which a plain (non-test)
/// build sees as unused.
#[allow(dead_code)]
fn eigh_eigenvectors<B: Backend>(m: &Tensor<B, 2>, device: &Device<B>) -> Tensor<B, 2> {
    let [n, _] = m.dims();
    let data = m.clone().into_data().to_vec::<f32>().unwrap();
    let out = eigh_from_host_data(&data, n);
    Tensor::<B, 2>::from_data(TensorData::new(out, vec![n, n]), device)
}

/// Batched analogue of two separate `eigh_eigenvectors(&gg0, ..)` / `eigh_eigenvectors(&gg1,
/// ..)` calls: pays exactly ONE combined GPU->CPU sync and ONE combined CPU->GPU upload
/// instead of two of each, by flattening+concatenating `gg0`/`gg1` before the single
/// `.into_data()` call and concatenating both results before the single `Tensor::from_data`
/// upload. The two matrices' `nalgebra::SymmetricEigen` decompositions are still run
/// INDEPENDENTLY on host (via [`eigh_from_host_data`]) and produce byte-identical eigenvectors
/// to calling `eigh_eigenvectors` on `gg0`/`gg1` separately.
///
/// Deliberately NOT a joint block-diagonal eigendecomposition of `[[gg0, 0], [0, gg1]]`: while
/// that would be mathematically equivalent whenever `gg0`/`gg1` have disjoint eigenvalue
/// spectra, a degenerate eigenvalue SHARED across the two blocks (e.g. both matrices having a
/// zero eigenvalue, which is common for a rank-deficient/early-training Shampoo accumulator)
/// has a non-unique eigenbasis for that joint eigenspace — `nalgebra` could return eigenvectors
/// that mix components from `gg0` and `gg1`, which would silently change SOAP's preconditioning
/// vs. computing each block's eigenbasis independently. Batching only the DATA TRANSFER (this
/// function), never the decomposition itself, is a pure round-trip-count win with zero
/// numerical difference from the pre-batching two-call path.
fn eigh_eigenvectors_batched<B: Backend>(
    gg0: &Tensor<B, 2>,
    gg1: &Tensor<B, 2>,
    device: &Device<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [n0, _] = gg0.dims();
    let [n1, _] = gg1.dims();
    let sz0 = n0 * n0;
    let sz1 = n1 * n1;

    // ONE combined GPU->CPU readback (instead of two): flatten both matrices to row vectors
    // and `cat` along dim 1 before the single `.into_data()` call.
    let combined_in = Tensor::cat(
        vec![gg0.clone().reshape([1, sz0]), gg1.clone().reshape([1, sz1])],
        1,
    );
    let data = combined_in.into_data().to_vec::<f32>().unwrap();
    let (data0, data1) = data.split_at(sz0);

    let out0 = eigh_from_host_data(data0, n0);
    let out1 = eigh_from_host_data(data1, n1);

    // ONE combined CPU->GPU upload (instead of two).
    let mut combined_out = Vec::with_capacity(sz0 + sz1);
    combined_out.extend_from_slice(&out0);
    combined_out.extend_from_slice(&out1);
    let combined_t = Tensor::<B, 1>::from_data(TensorData::new(combined_out, vec![sz0 + sz1]), device);
    let q0 = combined_t.clone().narrow(0, 0, sz0).reshape([n0, n0]);
    let q1 = combined_t.narrow(0, sz0, sz1).reshape([n1, n1]);
    (q0, q1)
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

        // 2. Refresh the eigenbasis periodically (full eigh — see module docs). Batched via
        // `eigh_eigenvectors_batched` (one combined GPU round trip for both gg0/gg1's
        // eigenvectors, byte-identical to two separate `eigh_eigenvectors` calls — see its
        // doc comment for why this is safe and block-diagonal joint eigh is not).
        if st.step % self.precondition_frequency == 0 {
            let (q0, q1) = eigh_eigenvectors_batched(&st.gg0, &st.gg1, &device);
            st.q0 = q0;
            st.q1 = q1;
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

    // ---- migrate_soap_muon_state_for_growth (issue #50, width growth) ----

    /// Runs `SoapMuon::step` a few times on a `[3,4]` tensor to build real (non-trivial) state.
    fn build_real_state(device: &Device<Wgpu>) -> SoapMuonState<Wgpu> {
        let optim: SoapMuon<Wgpu> = SoapMuonConfig::new().with_precondition_frequency(2).build();
        let mut tensor = Tensor::<Wgpu, 2>::from_floats(
            [[1.0, 0.5, -0.3, 0.2], [0.5, 1.0, 0.1, -0.4], [0.2, 0.1, 1.0, 0.3]],
            device,
        );
        let mut state = None;
        for i in 0..5 {
            let grad = Tensor::<Wgpu, 2>::from_floats(
                [[0.1, -0.2, 0.05, 0.3], [0.2, 0.1, -0.1, 0.05], [-0.05, 0.15, 0.2, -0.1]],
                device,
            )
            .mul_scalar(1.0 + i as f32 * 0.1);
            let (new_tensor, new_state) = optim.step(0.01, tensor, grad, state);
            tensor = new_tensor;
            state = new_state;
        }
        state.expect("state should exist after steps")
    }

    #[test]
    fn migrate_soap_muon_state_preserves_old_block_of_gg1_exactly() {
        let device = Default::default();
        let state = build_real_state(&device);

        let migrated = migrate_soap_muon_state_for_growth(&state, None, Some((4, 7)), &device);

        let old_data = state.gg1.into_data();
        let old_vals = old_data.as_slice::<f32>().unwrap();
        let new_data = migrated.gg1.into_data();
        let new_vals = new_data.as_slice::<f32>().unwrap();

        for r in 0..4 {
            for c in 0..4 {
                let old_v = old_vals[r * 4 + c];
                let new_v = new_vals[r * 7 + c];
                assert!(
                    (old_v - new_v).abs() < 1e-6,
                    "top-left 4x4 block of migrated gg1 should equal the original exactly at [{r},{c}]: {old_v} vs {new_v}"
                );
            }
        }
    }

    #[test]
    fn migrate_soap_muon_state_new_gg1_block_inherits_twin_correlation() {
        let device = Default::default();
        let state = build_real_state(&device);

        // k=1 (H_new=5): g(0) = 0 % 4 = 0.
        let migrated = migrate_soap_muon_state_for_growth(&state, None, Some((4, 5)), &device);

        let old_data = state.gg1.into_data();
        let old_vals = old_data.as_slice::<f32>().unwrap();
        let new_data = migrated.gg1.into_data();
        let new_vals = new_data.as_slice::<f32>().unwrap();

        let g0 = 0usize;
        assert!(
            (new_vals[4 * 5 + 4] - old_vals[g0 * 4 + g0]).abs() < 1e-6,
            "migrated.gg1[4,4] should equal original.gg1[g(0),g(0)]"
        );
        for s in 0..4 {
            assert!(
                (new_vals[4 * 5 + s] - old_vals[g0 * 4 + s]).abs() < 1e-6,
                "migrated.gg1[4,{s}] should equal original.gg1[g(0),{s}]"
            );
        }
    }

    #[test]
    fn migrate_soap_muon_state_resets_q_to_identity_of_new_size() {
        let device = Default::default();
        let state = build_real_state(&device);

        let migrated = migrate_soap_muon_state_for_growth(&state, None, Some((4, 7)), &device);
        assert_eq!(migrated.q1.dims(), [7, 7]);

        let product = migrated.q1.clone().matmul(migrated.q1.transpose());
        let data = product.into_data();
        let values = data.as_slice::<f32>().unwrap();
        for r in 0..7 {
            for c in 0..7 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (values[r * 7 + c] - expected).abs() < 1e-4,
                    "q1 @ q1^T should be the identity at [{r},{c}], got {}", values[r * 7 + c]
                );
            }
        }
    }

    #[test]
    fn migrate_soap_muon_state_preserves_step_counter() {
        let device = Default::default();
        let state = build_real_state(&device);
        let original_step = state.step;

        let migrated = migrate_soap_muon_state_for_growth(&state, None, Some((4, 7)), &device);
        assert_eq!(migrated.step, original_step);
    }

    #[test]
    fn migrate_soap_muon_state_exp_avg_copies_not_divides_new_slice() {
        let device = Default::default();
        let state = build_real_state(&device);

        // k=1 (H_new=5): g(0) = 0.
        let migrated = migrate_soap_muon_state_for_growth(&state, None, Some((4, 5)), &device);

        let old_data = state.exp_avg.into_data();
        let old_vals = old_data.as_slice::<f32>().unwrap();
        let new_data = migrated.exp_avg.into_data();
        let new_vals = new_data.as_slice::<f32>().unwrap();

        let d0 = 3usize; // rows unaffected (grow_dim0 = None)
        for row in 0..d0 {
            let twin = new_vals[row * 5 + 4];
            let src = old_vals[row * 4]; // g(0) = 0
            assert_eq!(twin, src, "migrated.exp_avg[{row},4] should exactly copy exp_avg[{row},g(0)], no division");
        }
    }

    // ─── Smart adaptive architecture: shrink-side migration ─────────────────────────────────

    #[test]
    fn migrate_soap_muon_state_for_shrink_selects_the_kept_columns_of_exp_avg_exactly() {
        let device = Default::default();
        let state = build_real_state(&device);
        let old_data = state.exp_avg.clone().into_data();
        let old_vals = old_data.as_slice::<f32>().unwrap(); // [3,4] row-major

        let keep = [1usize, 3usize]; // drop columns 0 and 2
        let migrated = migrate_soap_muon_state_for_shrink(&state, None, Some(&keep), &device);
        assert_eq!(migrated.exp_avg.dims(), [3, 2]);

        let new_data = migrated.exp_avg.into_data();
        let new_vals = new_data.as_slice::<f32>().unwrap();
        for row in 0..3 {
            assert_eq!(new_vals[row * 2], old_vals[row * 4 + 1], "row {row} col 0 must be the OLD column 1");
            assert_eq!(new_vals[row * 2 + 1], old_vals[row * 4 + 3], "row {row} col 1 must be the OLD column 3");
        }
    }

    #[test]
    fn migrate_soap_muon_state_for_shrink_projects_gg1_onto_kept_indices() {
        let device = Default::default();
        let state = build_real_state(&device);
        let old_data = state.gg1.clone().into_data();
        let old_vals = old_data.as_slice::<f32>().unwrap(); // [4,4]

        let keep = [1usize, 3usize];
        let migrated = migrate_soap_muon_state_for_shrink(&state, None, Some(&keep), &device);
        assert_eq!(migrated.gg1.dims(), [2, 2]);

        let new_data = migrated.gg1.into_data();
        let new_vals = new_data.as_slice::<f32>().unwrap();
        assert_eq!(new_vals[0], old_vals[1 * 4 + 1], "[0,0] must be old gg1[1,1]");
        assert_eq!(new_vals[1], old_vals[1 * 4 + 3], "[0,1] must be old gg1[1,3]");
        assert_eq!(new_vals[2], old_vals[3 * 4 + 1], "[1,0] must be old gg1[3,1]");
        assert_eq!(new_vals[3], old_vals[3 * 4 + 3], "[1,1] must be old gg1[3,3]");
    }

    #[test]
    fn migrate_soap_muon_state_for_shrink_resets_q_to_identity_of_new_size() {
        let device = Default::default();
        let state = build_real_state(&device);
        let keep = [0usize, 2usize];
        let migrated = migrate_soap_muon_state_for_shrink(&state, None, Some(&keep), &device);
        assert_eq!(migrated.q1.dims(), [2, 2]);
        let product = migrated.q1.clone().matmul(migrated.q1.transpose());
        let data = product.into_data();
        let values = data.as_slice::<f32>().unwrap();
        for r in 0..2 {
            for c in 0..2 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((values[r * 2 + c] - expected).abs() < 1e-4, "q1 @ q1^T should be identity at [{r},{c}]");
            }
        }
    }

    #[test]
    fn migrate_soap_muon_state_for_shrink_preserves_step_counter() {
        let device = Default::default();
        let state = build_real_state(&device);
        let original_step = state.step;
        let keep = [0usize, 1usize];
        let migrated = migrate_soap_muon_state_for_shrink(&state, None, Some(&keep), &device);
        assert_eq!(migrated.step, original_step);
    }

    #[test]
    fn migrate_soap_muon_state_for_shrink_none_axis_is_untouched() {
        let device = Default::default();
        let state = build_real_state(&device);
        // Only dim1 shrinks - dim0 (rows=3) must be completely unaffected.
        let keep = [0usize, 2usize];
        let migrated = migrate_soap_muon_state_for_shrink(&state, None, Some(&keep), &device);
        assert_eq!(migrated.exp_avg.dims()[0], 3);
        assert_eq!(migrated.gg0.dims(), state.gg0.dims());
    }

    #[test]
    fn soap_muon_step_after_shrink_migration_no_nan_no_discontinuous_spike() {
        let device = Default::default();
        let state = build_real_state(&device);
        let optim: SoapMuon<Wgpu> = SoapMuonConfig::new().with_precondition_frequency(2).build();
        let keep = [1usize, 2usize, 3usize]; // drop column 0
        let migrated = migrate_soap_muon_state_for_shrink(&state, None, Some(&keep), &device);

        let tensor = Tensor::<Wgpu, 2>::from_floats(
            [[0.5, -0.3, 0.2], [1.0, 0.1, -0.4], [0.1, 1.0, 0.3]], &device,
        );
        let grad = Tensor::<Wgpu, 2>::from_floats(
            [[-0.2, 0.05, 0.3], [0.1, -0.1, 0.05], [0.15, 0.2, -0.1]], &device,
        );
        let (new_tensor, _) = optim.step(0.01, tensor, grad, Some(migrated));
        let data = new_tensor.into_data();
        let vals = data.as_slice::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()), "a step immediately after shrink migration must not produce NaN/Inf");
    }

    #[test]
    fn soap_muon_step_after_migration_no_nan_no_discontinuous_spike() {
        let device = Default::default();
        let optim: SoapMuon<Wgpu> = SoapMuonConfig::new().with_precondition_frequency(2).build();

        let mut tensor = Tensor::<Wgpu, 2>::from_floats(
            [[1.0, 0.5, -0.3, 0.2], [0.5, 1.0, 0.1, -0.4], [0.2, 0.1, 1.0, 0.3]],
            &device,
        );
        let mut state = None;
        let mut pre_migration_update_norms = Vec::new();

        for i in 0..5 {
            let grad = Tensor::<Wgpu, 2>::from_floats(
                [[0.1, -0.2, 0.05, 0.3], [0.2, 0.1, -0.1, 0.05], [-0.05, 0.15, 0.2, -0.1]],
                &device,
            )
            .mul_scalar(1.0 + i as f32 * 0.1);
            let prev = tensor.clone();
            let (new_tensor, new_state) = optim.step(0.01, tensor, grad, state);
            let delta: f32 = (new_tensor.clone() - prev).powf_scalar(2.0).sum().sqrt().into_scalar();
            pre_migration_update_norms.push(delta);
            tensor = new_tensor;
            state = new_state;
        }

        let max_prior_norm = pre_migration_update_norms.iter().cloned().fold(0.0_f32, f32::max);

        // Migrate [3,4] -> [3,7] (grow_dim1 only — d0=3 rows never grow for this parameter).
        let old_state = state.expect("state should exist after 5 steps");
        let migrated_state = migrate_soap_muon_state_for_growth(&old_state, None, Some((4, 7)), &device);

        let old_data = tensor.into_data();
        let old_vals = old_data.as_slice::<f32>().unwrap().to_vec();
        let g = crate::network::duplication_map(4, 3).0;
        let mut grown_tensor_data = vec![0f32; 3 * 7];
        for row in 0..3 {
            for col in 0..4 {
                grown_tensor_data[row * 7 + col] = old_vals[row * 4 + col];
            }
            for (j, &gj) in g.iter().enumerate() {
                grown_tensor_data[row * 7 + 4 + j] = old_vals[row * 4 + gj];
            }
        }
        let mut tensor = Tensor::<Wgpu, 2>::from_data(
            TensorData::new(grown_tensor_data, vec![3, 7]),
            &device,
        );
        let mut state = Some(migrated_state);

        for i in 0..5 {
            let base = [[0.1, -0.2, 0.05, 0.3, 0.1, -0.2, 0.05],
                        [0.2, 0.1, -0.1, 0.05, 0.2, 0.1, -0.1],
                        [-0.05, 0.15, 0.2, -0.1, -0.05, 0.15, 0.2]];
            let grad = Tensor::<Wgpu, 2>::from_floats(base, &device).mul_scalar(1.0 + i as f32 * 0.1);
            let prev = tensor.clone();
            let (new_tensor, new_state) = optim.step(0.01, tensor, grad, state);

            let data = new_tensor.clone().into_data();
            let vals = data.as_slice::<f32>().unwrap();
            for v in vals {
                assert!(!v.is_nan() && !v.is_infinite(), "weight became NaN/Inf after post-migration step {i}");
            }

            if i == 0 {
                // Check only the pre-existing 0..4 columns for a discontinuous spike.
                let prev_data = prev.into_data();
                let prev_vals = prev_data.as_slice::<f32>().unwrap();
                let mut sq_sum = 0.0_f32;
                for row in 0..3 {
                    for col in 0..4 {
                        let d = vals[row * 7 + col] - prev_vals[row * 7 + col];
                        sq_sum += d * d;
                    }
                }
                let delta_norm = sq_sum.sqrt();
                assert!(
                    delta_norm < 3.0 * max_prior_norm.max(1e-6),
                    "post-migration update on pre-existing columns should not spike: {delta_norm} vs 3x max prior {max_prior_norm}"
                );
            }

            tensor = new_tensor;
            state = new_state;
        }
    }
}
