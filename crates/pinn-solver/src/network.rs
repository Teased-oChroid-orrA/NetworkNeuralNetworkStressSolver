use burn::{
    config::Config,
    module::{Module, ParamId},
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

/// Physics-informed displacement (+ optional stress) network.
///
/// Input:  [N, D_in] — D_in depends on Fourier embedding (3 without, 4*n_fourier with).
/// Output: [N, 3] (displacement: u, v, w) or [N, 5] (mDEM: u, v, σ_xx, σ_yy, σ_xy).
///
/// The Fourier embedding is applied OUTSIDE this module (see `fourier_embed`), so the
/// network itself remains a plain MLP. Callers are responsible for embedding the input
/// before calling `forward()`.
#[derive(Module, Debug)]
pub struct ElasticityNet<B: Backend> {
    layers: Vec<Linear<B>>,
    out:    Linear<B>,
}

impl<B: Backend> ElasticityNet<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(h).tanh();
        }
        self.out.forward(h)
    }

    /// Returns (weight_param_ids, bias_param_ids) across all `Linear` layers — used to
    /// partition `GradientsParams` between the SOAP-Muon optimizer (2D weights) and AdamW
    /// (1D biases) in a single training step.
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
}

impl ElasticityNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ElasticityNet<B> {
        assert!(self.n_hidden >= 1, "need at least 1 hidden layer");

        let mut layers: Vec<Linear<B>> = Vec::with_capacity(self.n_hidden);
        layers.push(LinearConfig::new(self.input_dim, self.hidden_dim).init(device));
        for _ in 1..self.n_hidden {
            layers.push(LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device));
        }
        let out = LinearConfig::new(self.hidden_dim, self.output_dim).init(device);
        ElasticityNet { layers, out }
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
