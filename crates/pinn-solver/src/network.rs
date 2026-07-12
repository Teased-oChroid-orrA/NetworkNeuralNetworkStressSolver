use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    nn::{Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, ElementConversion, Tensor},
};

/// Physics-informed displacement (+ optional stress) network.
///
/// Input:  [N, D_in] — D_in depends on Fourier embedding (3 without, 4*n_fourier with).
/// Output: [N, 3] (displacement: u, v, w) or [N, 5] (mDEM: u, v, σ_xx, σ_yy, σ_xy).
///
/// The Fourier embedding is applied OUTSIDE this module (see `fourier_embed`), so the
/// network itself remains a plain MLP. Callers are responsible for embedding the input
/// before calling `forward()`.
///
/// `gates` holds one learnable scalar per hidden→hidden layer (`layers[1..]`) for the
/// opt-in PirateNet adaptive-residual mode (see [`ElasticityNetConfig::use_piratenet`]).
/// It is empty when that mode is disabled, in which case `forward()` is byte-for-byte
/// the plain MLP path. `layers[0]` (input_dim→hidden_dim) is never gated — its input and
/// output dimensions differ, so it has no residual identity to fall back to.
#[derive(Module, Debug)]
pub struct ElasticityNet<B: Backend> {
    layers: Vec<Linear<B>>,
    gates:  Vec<Param<Tensor<B, 1>>>,
    out:    Linear<B>,
}

impl<B: Backend> ElasticityNet<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_masked(x, None)
    }

    /// Identical to [`Self::forward`], except that when `mask` is `Some`, hidden block
    /// `i` (i.e. `self.layers[i]` for `i in 1..self.layers.len()`, mask index `i-1`) is
    /// *structurally* skipped when `mask[i-1] == false`: `self.layers[i].forward(...)` is
    /// never called for that block, so its ops never enter the autodiff graph (a real
    /// compute-skip, not a zeroed contribution to it). `h` passes through unchanged for a
    /// skipped block, matching the mathematically-exact `alpha=0` residual identity that
    /// makes this lossless (see `dormant_block_gradient_is_exactly_zero`).
    ///
    /// `mask` must have length `self.gates.len()` (`self.layers.len() - 1`) when `Some`.
    /// `None` (or an empty mask when `self.gates` is empty, i.e. `use_piratenet=false`)
    /// computes every block, byte-identical to the pre-mask `forward()` path.
    pub fn forward_masked(&self, x: Tensor<B, 2>, mask: Option<&[bool]>) -> Tensor<B, 2> {
        let mut h = self.layers[0].forward(x).tanh();
        for i in 1..self.layers.len() {
            let awake = match mask {
                Some(m) => m.get(i - 1).copied().unwrap_or(true),
                None => true,
            };
            if !awake {
                continue;
            }
            let f = self.layers[i].forward(h.clone()).tanh();
            h = match self.gates.get(i - 1) {
                Some(alpha) => h.clone() + f.mul(alpha.val().reshape([1, 1])),
                None => f,
            };
        }
        self.out.forward(h)
    }

    /// Per-block "awake" classification (length `self.gates.len()`, i.e. `n_hidden - 1`):
    /// `true` iff `gate_values()[i].abs() > gate_epsilon`. Empty when `use_piratenet=false`
    /// (gates empty) — the single source of classification logic reused by both
    /// `awake_weight_ids` (optimizer parameter-set exclusion) and `forward_masked`'s
    /// compute-skip mask, so both enforcement points always agree.
    pub fn awake_mask(&self, gate_epsilon: f32) -> Vec<bool> {
        if self.gates.is_empty() {
            return Vec::new();
        }
        self.gate_values()
            .into_iter()
            .map(|g| g.abs() > gate_epsilon)
            .collect()
    }

    /// Returns (weight_param_ids, bias_param_ids) across all `Linear` layers — used to
    /// partition `GradientsParams` between the SOAP-Muon optimizer (2D weights) and AdamW
    /// (1D biases) in a single training step. Gate params are NOT included here — see
    /// [`Self::gate_ids`].
    pub fn param_ids(&self) -> (Vec<ParamId>, Vec<ParamId>) {
        let mut weight_ids = Vec::with_capacity(self.layers.len() + 1);
        let mut bias_ids = Vec::new();
        for layer in self.layers.iter().chain(std::iter::once(&self.out)) {
            weight_ids.push(layer.weight.id);
            if let Some(b) = &layer.bias {
                bias_ids.push(b.id);
            }
        }
        (weight_ids, bias_ids)
    }

    /// ParamIds of the PirateNet gate scalars — empty when `use_piratenet=false`.
    pub fn gate_ids(&self) -> Vec<ParamId> {
        self.gates.iter().map(|g| g.id).collect()
    }

    /// Test-only: force `gates[i]` to `value` — direct field assignment (`gates` has no
    /// production setter; training mutates it via the optimizer, never by direct
    /// assignment) so cross-module tests (e.g. `training_core`'s compute-skip tests) can
    /// deterministically exercise a dormant/awake block without a full training loop.
    #[cfg(test)]
    pub(crate) fn force_gate_for_test(&mut self, i: usize, value: f32, device: &B::Device) {
        self.gates[i] = Param::from_tensor(Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(vec![value], vec![1]),
            device,
        ));
    }

    /// Current gate values (cheap CPU read — call only at an infrequent check cadence,
    /// not on every step, since `.into_scalar()`/`.to_data()` forces a GPU sync).
    pub fn gate_values(&self) -> Vec<f32> {
        self.gates
            .iter()
            .map(|g| g.val().into_scalar().elem::<f32>())
            .collect()
    }

    /// Weight/bias ParamIds for layers that are still "awake" — i.e. `layers[0]`, `out`,
    /// and any `layers[1..]` whose gate magnitude exceeds `gate_epsilon`. When
    /// `use_piratenet=false` (gates empty), this is identical to `param_ids().0` — a
    /// dormant block's true gradient is exactly zero at `alpha=0` (see network.rs tests),
    /// so excluding it from the optimizer's parameter set is a correctness-preserving
    /// compute-skip, not an approximation.
    pub fn awake_weight_ids(&self, gate_epsilon: f32) -> Vec<ParamId> {
        let mask = self.awake_mask(gate_epsilon);
        self.awake_weight_ids_from_mask(&mask)
    }

    /// Same result as [`Self::awake_weight_ids`], but takes an already-computed
    /// [`Self::awake_mask`] instead of reading `gate_values()` itself — lets a caller that
    /// also needs the mask for [`Self::forward_masked`] (e.g. `training_core::step_physics`)
    /// take a single gate-value GPU-sync snapshot per step and reuse it for both purposes.
    pub fn awake_weight_ids_from_mask(&self, mask: &[bool]) -> Vec<ParamId> {
        if mask.is_empty() {
            return self.param_ids().0;
        }
        let mut ids = Vec::with_capacity(self.layers.len() + 1);
        ids.push(self.layers[0].weight.id);
        for i in 1..self.layers.len() {
            if mask[i - 1] {
                ids.push(self.layers[i].weight.id);
            }
        }
        ids.push(self.out.weight.id);
        ids
    }
}

#[derive(Config, Debug)]
pub struct ElasticityNetConfig {
    /// Input dimension. With Fourier embedding (n_fourier>0): 4*n_fourier. Without: 3.
    #[config(default = 3)]
    pub input_dim:  usize,
    #[config(default = 128)]
    pub hidden_dim: usize,
    #[config(default = 5)]
    pub n_hidden:   usize,
    /// 3 for plain DEM (u,v,w); 5 for mDEM (u,v,σ_xx,σ_yy,σ_xy).
    #[config(default = 3)]
    pub output_dim: usize,
    /// Opt-in PirateNet adaptive-residual mode: `layers[1..]` become gated residual
    /// blocks (`h += tanh(layer(h)) * alpha`), `alpha` starting at exactly 0.0 so the
    /// network launches as a single-hidden-layer MLP (`layers[0] -> out`) regardless of
    /// `n_hidden`. Default `false` — plain MLP, byte-for-byte the pre-existing behavior.
    #[config(default = false)]
    pub use_piratenet: bool,
    /// Gain for the `KaimingNormal` initializer applied to `layers[1..]` when
    /// `use_piratenet=true` (ignored otherwise). Default `5/3`, the standard gain for
    /// tanh-activated layers.
    #[config(default = 1.6666666666666667)]
    pub kaiming_gain: f64,
}

impl ElasticityNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ElasticityNet<B> {
        assert!(self.n_hidden >= 1, "need at least 1 hidden layer");

        let mut layers: Vec<Linear<B>> = Vec::with_capacity(self.n_hidden);
        layers.push(LinearConfig::new(self.input_dim, self.hidden_dim).init(device));
        let hidden_config = if self.use_piratenet {
            LinearConfig::new(self.hidden_dim, self.hidden_dim).with_initializer(
                Initializer::KaimingNormal { gain: self.kaiming_gain, fan_out_only: false },
            )
        } else {
            LinearConfig::new(self.hidden_dim, self.hidden_dim)
        };
        for _ in 1..self.n_hidden {
            layers.push(hidden_config.init(device));
        }
        let out = LinearConfig::new(self.hidden_dim, self.output_dim).init(device);

        let gates = if self.use_piratenet {
            (0..self.n_hidden.saturating_sub(1))
                .map(|_| Param::from_tensor(Tensor::<B, 1>::zeros([1], device)))
                .collect()
        } else {
            Vec::new()
        };

        ElasticityNet { layers, gates, out }
    }
}

/// Sinusoidal log-scale Fourier feature embedding for spectral-bias correction.
///
/// Maps [N, 3] coords (x_norm, y_norm, z=0) → [N, 4*n_fourier] features:
///   for l = 0..n_fourier: [sin(2^l π x), cos(2^l π x), sin(2^l π y), cos(2^l π y)]
///
/// The z column is dropped (always 0 in plane stress; adds no information).
/// With n_fourier=8 the frequency range is [π, 128π], covering the near-hole
/// stress gradient scale in the normalized domain.
pub fn fourier_embed<B: Backend>(
    coords: Tensor<B, 2>,  // [N, 3]
    n_fourier: usize,
    _device: &B::Device,
) -> Tensor<B, 2> {
    let n = coords.dims()[0];
    let x = coords.clone().slice([0..n, 0..1]);  // [N, 1]
    let y = coords.slice([0..n, 1..2]);           // [N, 1]

    let mut feats: Vec<Tensor<B, 2>> = Vec::with_capacity(4 * n_fourier);
    for l in 0..n_fourier {
        let scale = 2.0_f64.powi(l as i32) * std::f64::consts::PI;
        let xs = x.clone().mul_scalar(scale);
        let ys = y.clone().mul_scalar(scale);
        feats.push(xs.clone().sin());
        feats.push(xs.cos());
        feats.push(ys.clone().sin());
        feats.push(ys.cos());
    }
    Tensor::cat(feats, 1)  // [N, 4*n_fourier]
}

/// Forward helper: apply optional Fourier embedding then run the network.
///
/// `input` is the raw stencil coordinate tensor [M, 3]. The B matrix for Fourier
/// features is deterministic (seeded by frequency level) and recreated each call.
/// Cost is negligible for M ~ N_pts × 5 (stencil batch size).
pub fn fwd<Bk: Backend>(
    model: &ElasticityNet<Bk>,
    input: Tensor<Bk, 2>,
    n_fourier: usize,
    device: &Bk::Device,
) -> Tensor<Bk, 2> {
    if n_fourier > 0 {
        model.forward(fourier_embed(input, n_fourier, device))
    } else {
        model.forward(input)
    }
}

/// `fwd`'s masked analogue — applies the same optional Fourier embedding, then calls
/// `ElasticityNet::forward_masked` instead of `forward`. `None` is byte-identical to `fwd`.
pub fn fwd_masked<Bk: Backend>(
    model: &ElasticityNet<Bk>,
    input: Tensor<Bk, 2>,
    n_fourier: usize,
    device: &Bk::Device,
    mask: Option<&[bool]>,
) -> Tensor<Bk, 2> {
    if n_fourier > 0 {
        model.forward_masked(fourier_embed(input, n_fourier, device), mask)
    } else {
        model.forward_masked(input, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Wgpu};
    use burn::backend::wgpu::WgpuDevice;
    use burn::optim::GradientsParams;

    type TB = Autodiff<Wgpu>;
    type TBInner = Wgpu;

    fn piratenet_config() -> ElasticityNetConfig {
        ElasticityNetConfig::new()
            .with_input_dim(2)
            .with_hidden_dim(4)
            .with_n_hidden(3)
            .with_output_dim(2)
            .with_use_piratenet(true)
    }

    /// At init (all gates = 0.0), a PirateNet-mode network must be mathematically
    /// identical to a single-hidden-layer MLP (layers[0] -> out) regardless of
    /// n_hidden — the "perfectly shallow launch" property the gate design relies on.
    #[test]
    fn shallow_launch_matches_single_hidden_layer() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = piratenet_config().init(&device);

        assert_eq!(model.gates.len(), 2); // n_hidden - 1
        for g in &model.gates {
            assert_eq!(g.val().into_scalar().elem::<f32>(), 0.0);
        }

        let x: Tensor<TB, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(vec![0.3_f32, -0.7, 0.1, 0.9], vec![2, 2]),
            &device,
        );

        let actual = model.forward(x.clone());
        // Manually walk the "shallow" path using the same layers[0]/out weights.
        let expected = model.out.forward(model.layers[0].forward(x).tanh());

        let diff: f32 = (actual - expected).abs().sum().into_scalar();
        assert!(diff < 1e-5, "shallow-launch mismatch: diff={diff}");
    }

    /// A dormant block's true gradient is exactly zero at alpha=0 (chain rule: the
    /// entire nonlinear branch is multiplied by alpha before being added to the
    /// residual stream). This is the correctness property `awake_weight_ids` relies
    /// on to safely exclude dormant blocks from the SOAP-Muon optimizer step.
    #[test]
    fn dormant_block_gradient_is_exactly_zero() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = piratenet_config().init(&device);

        let x: Tensor<TB, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(vec![0.3_f32, -0.7, 0.1, 0.9], vec![2, 2]),
            &device,
        );
        let out = model.forward(x);
        let loss = out.powf_scalar(2.0_f64).sum();

        let mut grads = loss.backward();
        for i in 1..model.layers.len() {
            let id = model.layers[i].weight.id;
            let g = GradientsParams::from_params(&mut grads, &model, &[id])
                .get::<TBInner, 2>(id)
                .expect("gradient should exist for a param that participated in forward");
            let max_abs: f32 = g.abs().max().into_scalar();
            assert_eq!(max_abs, 0.0, "layers[{i}] weight gradient should be exactly zero at alpha=0");
        }
    }

    /// Sets `model.gates[i]` to `value` — thin wrapper over `force_gate_for_test`.
    fn set_gate<B: Backend>(model: &mut ElasticityNet<B>, i: usize, value: f32, device: &B::Device) {
        model.force_gate_for_test(i, value, device);
    }

    fn fixed_input(device: &WgpuDevice) -> Tensor<TB, 2> {
        Tensor::from_data(
            burn::tensor::TensorData::new(vec![0.3_f32, -0.7, 0.1, 0.9], vec![2, 2]),
            device,
        )
    }

    #[test]
    fn awake_mask_matches_awake_weight_ids_classification() {
        let device = WgpuDevice::default();
        let mut model: ElasticityNet<TB> = piratenet_config().init(&device);
        set_gate(&mut model, 0, 0.0, &device);
        set_gate(&mut model, 1, 5e-4, &device);

        let mask = model.awake_mask(1e-4);
        assert_eq!(mask, vec![false, true]);

        let ids = model.awake_weight_ids(1e-4);
        assert!(ids.contains(&model.layers[0].weight.id));
        assert!(ids.contains(&model.layers[2].weight.id));
        assert!(ids.contains(&model.out.weight.id));
        assert!(!ids.contains(&model.layers[1].weight.id));
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn dormant_block_forward_is_structurally_skipped_not_computed() {
        let device = WgpuDevice::default();
        let mut model: ElasticityNet<TB> = piratenet_config().init(&device);
        set_gate(&mut model, 0, 0.0, &device); // layers[1] dormant
        set_gate(&mut model, 1, 1.0, &device); // layers[2] awake

        let mask = model.awake_mask(1e-4);
        assert_eq!(mask, vec![false, true]);

        let x = fixed_input(&device);
        let out = model.forward_masked(x, Some(&mask));
        let loss = out.powf_scalar(2.0_f64).sum();
        let mut grads = loss.backward();

        let id = model.layers[1].weight.id;
        let g = GradientsParams::from_params(&mut grads, &model, &[id]).get::<TBInner, 2>(id);
        assert!(
            g.is_none(),
            "structurally-skipped block should receive no gradient entry at all, got {g:?}"
        );
    }

    #[test]
    fn awake_block_forward_masked_still_computed_and_gradient_flows() {
        let device = WgpuDevice::default();
        let mut model: ElasticityNet<TB> = piratenet_config().init(&device);
        set_gate(&mut model, 0, 0.0, &device); // layers[1] dormant
        set_gate(&mut model, 1, 1.0, &device); // layers[2] awake

        let mask = model.awake_mask(1e-4);
        let x = fixed_input(&device);
        let out = model.forward_masked(x, Some(&mask));
        let loss = out.powf_scalar(2.0_f64).sum();
        let mut grads = loss.backward();

        let id = model.layers[2].weight.id;
        let g = GradientsParams::from_params(&mut grads, &model, &[id])
            .get::<TBInner, 2>(id)
            .expect("awake block should have a gradient");
        let max_abs: f32 = g.abs().max().into_scalar();
        assert!(max_abs > 0.0, "awake block gradient should be nonzero, got {max_abs}");
    }

    #[test]
    fn forward_masked_none_is_byte_identical_to_forward() {
        let device = WgpuDevice::default();
        let mut model: ElasticityNet<TB> = piratenet_config().init(&device);
        set_gate(&mut model, 0, 0.37, &device);
        set_gate(&mut model, 1, -0.91, &device);

        let x = fixed_input(&device);
        let a = model.forward(x.clone());
        let b = model.forward_masked(x, None);

        let diff: f32 = (a - b).abs().sum().into_scalar();
        assert_eq!(diff, 0.0, "forward_masked(x, None) must be byte-identical to forward(x)");
    }

    #[test]
    fn forward_masked_all_true_matches_forward_all_gates_nonzero() {
        let device = WgpuDevice::default();
        let mut model: ElasticityNet<TB> = piratenet_config().init(&device);
        set_gate(&mut model, 0, 0.37, &device);
        set_gate(&mut model, 1, -0.91, &device);

        let x = fixed_input(&device);
        let a = model.forward(x.clone());
        let b = model.forward_masked(x, Some(&[true, true]));

        let diff: f32 = (a - b).abs().sum().into_scalar();
        assert!(diff < 1e-6, "all-true mask should numerically match forward(), diff={diff}");
    }

    #[test]
    fn use_piratenet_disabled_makes_awake_mask_empty_and_forward_masked_ignores_it() {
        let device = WgpuDevice::default();
        let config = ElasticityNetConfig::new()
            .with_input_dim(2)
            .with_hidden_dim(4)
            .with_n_hidden(3)
            .with_output_dim(2)
            .with_use_piratenet(false);
        let model: ElasticityNet<TB> = config.init(&device);

        assert!(model.gates.is_empty());
        assert_eq!(model.awake_mask(1e-4), Vec::<bool>::new());

        let x = fixed_input(&device);
        let a = model.forward(x.clone());
        let b = model.forward_masked(x, Some(&[]));

        let diff: f32 = (a - b).abs().sum().into_scalar();
        assert_eq!(diff, 0.0);
    }
}
