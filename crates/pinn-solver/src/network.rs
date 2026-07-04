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
        let mut h = self.layers[0].forward(x).tanh();
        for i in 1..self.layers.len() {
            let f = self.layers[i].forward(h.clone()).tanh();
            h = match self.gates.get(i - 1) {
                Some(alpha) => h.clone() + f.mul(alpha.val().reshape([1, 1])),
                None => f,
            };
        }
        self.out.forward(h)
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
        if self.gates.is_empty() {
            return self.param_ids().0;
        }
        let gate_values = self.gate_values();
        let mut ids = Vec::with_capacity(self.layers.len() + 1);
        ids.push(self.layers[0].weight.id);
        for i in 1..self.layers.len() {
            if gate_values[i - 1].abs() > gate_epsilon {
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
}
