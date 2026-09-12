use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    nn::{Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, ElementConversion, Tensor, TensorData},
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

    /// Stage I (live network-evolution visualization) - per-layer `(mean |weight|, max
    /// |weight|)`, read directly from already-computed parameter tensors: no forward pass, no
    /// gradient computation, safe to call at any point without perturbing training. One entry
    /// per `self.layers` element (input layer first, then each hidden->hidden layer in order);
    /// `self.out` is deliberately excluded, matching `awake_mask`'s own scope (`gates` only
    /// ever covers `layers[1..]`) - the fixed-size output projection is less informative for
    /// "network evolution" than the hidden representation layers.
    pub fn layer_weight_stats(&self) -> Vec<(f32, f32)> {
        self.layers.iter().map(|l| {
            let data = l.weight.val().abs().into_data().to_vec::<f32>().unwrap_or_default();
            if data.is_empty() {
                return (0.0, 0.0);
            }
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let max = data.iter().copied().fold(0.0f32, f32::max);
            (mean, max)
        }).collect()
    }

    /// Per-output-neuron mean |weight| for each prunable layer (`self.layers`, i.e. every entry
    /// EXCEPT `out` — mirrors `layer_weight_stats`'/`awake_mask`'s own "excludes `out`" scope),
    /// used by `architecture_controller::ArchitectureController::plan_prune` to rank INDIVIDUAL
    /// hidden units for width-shrink, not just whole layers the way `layer_weight_stats` does.
    /// Each inner `Vec<f32>` has one entry per output column (hidden unit) of that layer, in the
    /// SAME index space [`Self::prune_width`]'s `drop_indices` uses (this network's one shared
    /// `hidden_dim` — see `prune_width`'s doc comment for why every layer must share one).
    pub fn per_neuron_magnitudes(&self) -> Vec<Vec<f32>> {
        self.layers.iter().map(|l| {
            let w = l.weight.val(); // [d_input, d_output]
            let dims = w.dims();
            let data = w.abs().into_data().to_vec::<f32>().unwrap_or_default();
            if dims[0] == 0 || data.len() != dims[0] * dims[1] {
                return vec![0.0; dims[1]];
            }
            (0..dims[1]).map(|col| {
                let sum: f32 = (0..dims[0]).map(|row| data[row * dims[1] + col]).sum();
                sum / dims[0] as f32
            }).collect()
        }).collect()
    }

    /// Real, end-to-end weight matrices for the network-diagram visualization (approved neuron-
    /// and-edge design, replacing the earlier per-layer bar-chart aggregate) — `self.layers`
    /// (input layer first) THEN `self.out`, unlike [`Self::layer_weight_stats`]/[`Self::
    /// awake_mask`], which deliberately exclude the output projection to mirror `awake_mask`'s
    /// own scope. A wiring diagram needs the complete input-to-output path to mean anything, so
    /// this is a genuinely different scope, not an oversight in the other two. Each matrix is
    /// `[d_input, d_output]`, matching burn's own `Linear::weight` layout, so `M[[i, j]]` is the
    /// weight from input neuron `i` to output neuron `j` — exactly what an edge in the diagram
    /// needs, with no transposition.
    pub fn all_weight_matrices(&self) -> Vec<ndarray::Array2<f32>> {
        self.layers.iter().chain(std::iter::once(&self.out)).map(|l| {
            let w = l.weight.val();
            let dims = w.dims();
            let data = w.into_data().to_vec::<f32>().unwrap_or_default();
            ndarray::Array2::from_shape_vec((dims[0], dims[1]), data)
                .unwrap_or_else(|_| ndarray::Array2::zeros((0, 0)))
        }).collect()
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

    /// Net2WiderNet-style function-preserving width growth (issue #50). Grows `hidden_dim`
    /// from its current value to `new_hidden_dim` by deterministically DUPLICATING hidden
    /// units (see [`duplication_map`]) — never zero- or random-padding — which is what makes
    /// `grown.forward(x) == self.forward(x)` (up to float rounding) hold immediately after
    /// growth, before any further training.
    ///
    /// - `layers[0].weight`'s input side (`input_dim`) never grows — only its output side
    ///   (`hidden_dim`) does (grow-output: new columns are exact duplicates of existing ones).
    /// - `layers[1..]`'s weight grows on BOTH sides (`hidden_dim` is both its input and output
    ///   dimension): grow-output on its own columns, grow-input-split on its own rows (divided
    ///   by each source unit's duplication multiplicity), using the SAME duplication map on
    ///   both axes — this is what keeps `h_new[H_old+j] == h_new[g(j)]` an exact identity at
    ///   every hop, including through PirateNet's gated residual (`h += f * alpha`), since `h`
    ///   and `f` are duplicated identically at every layer.
    /// - `out.weight` grows only on its input side (`hidden_dim`; grow-input-split). `output_dim`
    ///   never changes, so `out.bias` is untouched. `gates` (one scalar per hidden→hidden
    ///   block) are orthogonal to width and are moved through unchanged.
    ///
    /// Every weight `Param` keeps its original [`ParamId`] (via [`Param::map`], which preserves
    /// `self.id`) so ParamId-keyed optimizer bookkeeping (`param_ids()`, `awake_weight_ids*`,
    /// `GradientsParams::from_params`) continues to resolve correctly without any call-site
    /// change. Every bias `Param` gets a FRESH `ParamId` instead — biases cold-start their
    /// AdamW moments on growth rather than being migrated (see
    /// `pinn_solver::optim::soap_muon::migrate_soap_muon_state_for_growth` for the weight-side
    /// SOAP-Muon moment migration, which the fresh bias ids deliberately do NOT need/use).
    ///
    /// # Panics
    /// If `new_hidden_dim` does not strictly exceed the current `hidden_dim` — a programmer-
    /// error precondition (this must be a strict *growth*), matching this file's existing
    /// `assert!`-on-invariant convention rather than a recoverable `Result`.
    pub fn grow_width(&self, new_hidden_dim: usize, device: &B::Device) -> ElasticityNet<B> {
        let h_old = self.layers[0].weight.val().dims()[1];
        assert!(
            new_hidden_dim > h_old,
            "grow_width: new_hidden_dim ({new_hidden_dim}) must be strictly greater than the \
             current hidden_dim ({h_old})"
        );
        let k = new_hidden_dim - h_old;
        let (g, m) = duplication_map(h_old, k);

        let mut new_layers: Vec<Linear<B>> = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            // Force-materialize the lazily-initialized `Param` on `self` BEFORE cloning it:
            // `Initializer::init_with` (what `LinearConfig::init` uses for weight/bias) builds
            // an UNINITIALIZED `Param` whose value is drawn from RNG on first `.val()` call.
            // `Param::clone()` on a still-uninitialized `Param` clones the lazy *closure*, not
            // a value — so if we cloned first and called `.val()` only on the clone (inside
            // `Param::map`), the clone would independently re-roll its own random draw,
            // silently diverging from `self`'s own (separately-triggered) value. Calling
            // `.val()` here first caches the value into `self`'s own `Param`, so the
            // subsequent `.clone()` takes the "already initialized" branch and clones the
            // cached tensor byte-for-byte instead.
            let _ = layer.weight.val();
            if let Some(b) = &layer.bias {
                let _ = b.val();
            }

            let grow_input = i >= 1; // layers[0]'s input_dim side never grows
            let weight = layer.weight.clone().map(|w| {
                let w = if grow_input { split_rows::<B>(&w, &g, &m, device) } else { w };
                duplicate_columns::<B>(&w, &g, device)
            });
            let bias = layer.bias.clone().map(|b| {
                Param::initialized(ParamId::new(), duplicate_1d::<B>(&b.val(), &g, device))
            });
            new_layers.push(Linear { weight, bias });
        }

        let _ = self.out.weight.val();
        let out_weight = self.out.weight.clone().map(|w| split_rows::<B>(&w, &g, &m, device));
        // `out.bias` is passed through UNCHANGED (output_dim never grows) — still force-
        // materialize before cloning, same lazy-Param-clone hazard as above, so
        // `grown.out.bias` is byte-identical to `self.out.bias`, not an independent draw.
        if let Some(b) = &self.out.bias {
            let _ = b.val();
        }
        let out = Linear { weight: out_weight, bias: self.out.bias.clone() };

        // `gates` are orthogonal to width and moved through unchanged — force-materialize for
        // the same lazy-Param-clone reason (defensive: `ElasticityNetConfig::init`/
        // `force_gate_for_test` currently build gates eagerly via `Param::from_tensor`, but
        // this keeps `grow_width` correct even if that ever changes).
        for gate in &self.gates {
            let _ = gate.val();
        }
        ElasticityNet { layers: new_layers, gates: self.gates.clone(), out }
    }

    /// Depth growth (Net2DeeperNet-style) — safe ONLY because of PirateNet's gated-residual
    /// form: `h += tanh(layer(h)) * alpha` degenerates to EXACTLY `h` at `alpha=0`, regardless
    /// of `tanh`'s nonlinearity. A plain stacked-`tanh` MLP (`use_piratenet=false`) has no
    /// equivalent safe insertion point — `tanh(x) != x` in general, so there is no identity
    /// layer to insert there, unlike ReLU-based Net2DeeperNet. Appends one new
    /// `hidden_dim -> hidden_dim` `Linear` (same `KaimingNormal` init `ElasticityNetConfig::
    /// init` uses for gated blocks) to `layers`, and a new gate initialized to EXACTLY `0.0` to
    /// `gates` — the new block computes nothing that reaches the output until something later
    /// ungates it (proven by `append_dormant_layer_is_output_identical_at_insertion`).
    ///
    /// # Panics
    /// If `self.gates.len() != self.layers.len() - 1` — i.e. the network is not FULLY gated
    /// (every `layers[1..]` block already has a gate). This is the precondition that keeps
    /// `forward_masked`'s positional `gates[i-1]` lookup aligned with `layers[i]`; a partially-
    /// gated network (which never occurs from ordinary construction — `ElasticityNetConfig::
    /// init` builds either zero gates or exactly `n_hidden-1`) would silently misalign the two
    /// after this call, so this is checked, not assumed.
    pub fn append_dormant_layer(&self, kaiming_gain: f64, device: &B::Device) -> ElasticityNet<B> {
        assert_eq!(
            self.gates.len(), self.layers.len() - 1,
            "append_dormant_layer requires a fully-gated network (use_piratenet=true) - got {} \
             gates for {} layers", self.gates.len(), self.layers.len()
        );
        let hidden_dim = self.layers[0].weight.val().dims()[1];
        let new_layer = LinearConfig::new(hidden_dim, hidden_dim)
            .with_initializer(Initializer::KaimingNormal { gain: kaiming_gain, fan_out_only: false })
            .init::<B>(device);

        let mut layers = self.layers.clone();
        layers.push(new_layer);
        let mut gates = self.gates.clone();
        gates.push(Param::from_tensor(Tensor::<B, 1>::zeros([1], device)));

        ElasticityNet { layers, gates, out: self.out.clone() }
    }

    /// Depth shrink — the inverse of [`Self::append_dormant_layer`], safe ONLY when
    /// `layers[idx]`'s gate is genuinely dormant (`|alpha| <= gate_epsilon`): removal of a
    /// truly-dormant block changes nothing, since the block's contribution to `h` was already
    /// exactly zero (the same `alpha=0` identity `append_dormant_layer`'s doc comment explains,
    /// in reverse). The caller (`ArchitectureController`) decides WHEN a block has been dormant
    /// long enough to remove; this function only enforces that the gate is dormant RIGHT NOW,
    /// as a defensive check, not a policy decision.
    ///
    /// # Panics
    /// If `idx == 0` (`layers[0]` is never gated — no dormancy concept, must never be removed),
    /// if `idx >= self.layers.len()`, if the network is not fully gated (same precondition as
    /// `append_dormant_layer`), or if the gate at `idx` is not currently dormant within
    /// `gate_epsilon`.
    pub fn remove_layer(&self, idx: usize, gate_epsilon: f32) -> ElasticityNet<B> {
        assert!(idx >= 1 && idx < self.layers.len(), "remove_layer: idx {idx} out of range (must be 1..{})", self.layers.len());
        assert_eq!(
            self.gates.len(), self.layers.len() - 1,
            "remove_layer requires a fully-gated network (use_piratenet=true)"
        );
        let alpha = self.gates[idx - 1].val().into_scalar().elem::<f32>();
        assert!(
            alpha.abs() <= gate_epsilon,
            "remove_layer: gate at idx {idx} is not dormant (|alpha|={} > {gate_epsilon})", alpha.abs()
        );

        let mut layers = self.layers.clone();
        layers.remove(idx);
        let mut gates = self.gates.clone();
        gates.remove(idx - 1);

        ElasticityNet { layers, gates, out: self.out.clone() }
    }

    /// Width shrink — the GLOBAL, uniform-hidden_dim mirror of [`Self::grow_width`], in
    /// reverse. Explicitly LOSSY, unlike [`Self::append_dormant_layer`]/[`Self::remove_layer`]:
    /// removing an established, contributing neuron necessarily changes what the network
    /// computes (there is no rescaling trick analogous to `grow_width`'s `split_rows` division
    /// that makes removal function-preserving — the removed neuron's contribution is simply
    /// gone, not redistributed).
    ///
    /// Must be global, not per-layer: `forward_masked`'s gated residual sum
    /// (`h = h + tanh(layers[i](h)) * alpha`) requires `layers[i]`'s OUTPUT width to equal
    /// `h`'s width for every gated `i`, which in turn requires every `layers[1..]` block to
    /// share the SAME `hidden_dim` as `layers[0]`'s output and `out`'s input. Pruning only one
    /// layer's boundary (an earlier, incorrect version of this function) would desync that
    /// shared width and panic on the very next `forward_masked` call — this version removes
    /// `drop_indices` from every layer uniformly, exactly mirroring `grow_width`'s per-layer
    /// row/column treatment (`layers[0]`: output columns only; `layers[1..]`: input rows AND
    /// output columns; `out`: input rows only), just with `remove_*` in place of `duplicate_*`/
    /// `split_*` and no rescaling (removal has no lossless equivalent). `gates` are untouched
    /// (pruning neurons within every block doesn't change the block count).
    ///
    /// # Panics
    /// If any `drop_indices` entry is out of range for the current `hidden_dim`, or if dropping
    /// every entry in `drop_indices` would leave zero hidden units.
    pub fn prune_width(&self, drop_indices: &[usize], device: &B::Device) -> ElasticityNet<B> {
        let hidden_dim = self.layers[0].weight.val().dims()[1];
        let mut sorted_drop: Vec<usize> = drop_indices.to_vec();
        sorted_drop.sort_unstable();
        sorted_drop.dedup();
        assert!(
            sorted_drop.iter().all(|&i| i < hidden_dim),
            "prune_width: drop index out of range (hidden_dim {hidden_dim})"
        );
        assert!(sorted_drop.len() < hidden_dim, "prune_width: cannot drop every hidden unit");

        let mut new_layers: Vec<Linear<B>> = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            // Same lazy-Param-clone hazard as `grow_width` - force-materialize before cloning.
            let _ = layer.weight.val();
            if let Some(b) = &layer.bias {
                let _ = b.val();
            }
            let drop_input = i >= 1; // layers[0]'s input_dim side never shrinks
            let weight = layer.weight.clone().map(|w| {
                let w = if drop_input { remove_rows::<B>(&w, &sorted_drop, device) } else { w };
                remove_columns::<B>(&w, &sorted_drop, device)
            });
            let bias = layer.bias.clone().map(|b| {
                Param::initialized(ParamId::new(), remove_1d::<B>(&b.val(), &sorted_drop, device))
            });
            new_layers.push(Linear { weight, bias });
        }

        let _ = self.out.weight.val();
        let out_weight = self.out.weight.clone().map(|w| remove_rows::<B>(&w, &sorted_drop, device));
        if let Some(b) = &self.out.bias {
            let _ = b.val();
        }
        let out = Linear { weight: out_weight, bias: self.out.bias.clone() };

        for gate in &self.gates {
            let _ = gate.val();
        }
        ElasticityNet { layers: new_layers, gates: self.gates.clone(), out }
    }
}

/// Deterministic Net2WiderNet duplication map for growing a dimension of size `h_old` by `k`
/// new units, with no RNG: `g[j] = j mod h_old` (for `j` in `0..k`) is the ORIGINAL index that
/// new unit `h_old + j` duplicates; `m[s]` (for `s` in `0..h_old`) is `s`'s resulting
/// multiplicity — `1 + |{ j : g[j] == s }|` — i.e. how many total units (the original plus any
/// duplicates) now share unit `s`'s identity. `m` is what an input-side weight divides by
/// (grow-input-split) so a duplicated unit's *combined* downstream contribution equals the
/// original single unit's contribution, preserving the network's function exactly.
pub(crate) fn duplication_map(h_old: usize, k: usize) -> (Vec<usize>, Vec<usize>) {
    let g: Vec<usize> = (0..k).map(|j| j % h_old).collect();
    let mut m = vec![1usize; h_old];
    for &s in &g {
        m[s] += 1;
    }
    (g, m)
}

/// Grow-output: append `k = g.len()` new columns to a `[d0, d1_old]` weight, each an EXACT
/// (undivided) duplicate of source column `g[j]` — no float math on this axis, a pure copy.
fn duplicate_columns<B: Backend>(w: &Tensor<B, 2>, g: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let [d0, d1_old] = w.dims();
    let k = g.len();
    let d1_new = d1_old + k;
    let data = w.clone().into_data().to_vec::<f32>().unwrap();

    let mut out = vec![0f32; d0 * d1_new];
    for row in 0..d0 {
        out[row * d1_new..row * d1_new + d1_old]
            .copy_from_slice(&data[row * d1_old..row * d1_old + d1_old]);
        for (j, &gj) in g.iter().enumerate() {
            out[row * d1_new + d1_old + j] = data[row * d1_old + gj];
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(out, vec![d0, d1_new]), device)
}

/// Grow-input-split: append `k = g.len()` new rows to a `[d0_old, d1]` weight, and rescale
/// every row (existing rows AND the new duplicate rows) by `1 / m[source_index]` so that the
/// SUM of a duplicated unit's fan-out contributions equals the original single unit's
/// contribution — this is the division half of Net2WiderNet duplication.
fn split_rows<B: Backend>(w: &Tensor<B, 2>, g: &[usize], m: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let [d0_old, d1] = w.dims();
    let k = g.len();
    let d0_new = d0_old + k;
    let data = w.clone().into_data().to_vec::<f32>().unwrap();

    let mut out = vec![0f32; d0_new * d1];
    for s in 0..d0_old {
        let mult = m[s] as f32;
        for col in 0..d1 {
            out[s * d1 + col] = data[s * d1 + col] / mult;
        }
    }
    for (j, &gj) in g.iter().enumerate() {
        let mult = m[gj] as f32;
        for col in 0..d1 {
            out[(d0_old + j) * d1 + col] = data[gj * d1 + col] / mult;
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(out, vec![d0_new, d1]), device)
}

/// Bias analogue of [`duplicate_columns`] (1D, no division — bias duplication is always an
/// exact copy, matching the weight's grow-OUTPUT rule, never grow-input-split's division).
fn duplicate_1d<B: Backend>(b: &Tensor<B, 1>, g: &[usize], device: &B::Device) -> Tensor<B, 1> {
    let d_old = b.dims()[0];
    let data = b.clone().into_data().to_vec::<f32>().unwrap();

    let mut out = data.clone();
    for &gj in g {
        out.push(data[gj]);
    }
    Tensor::<B, 1>::from_data(TensorData::new(out, vec![d_old + g.len()]), device)
}

/// `prune_width`'s column-removal primitive — the reverse of [`duplicate_columns`], but with no
/// rescaling counterpart (unlike `split_rows`'s division on growth, there is no way to make
/// removal lossless; the dropped columns' contribution is simply gone).
fn remove_columns<B: Backend>(w: &Tensor<B, 2>, drop: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let [d0, d1_old] = w.dims();
    let data = w.clone().into_data().to_vec::<f32>().unwrap();
    let keep: Vec<usize> = (0..d1_old).filter(|c| !drop.contains(c)).collect();
    let d1_new = keep.len();
    let mut out = vec![0f32; d0 * d1_new];
    for row in 0..d0 {
        for (new_c, &old_c) in keep.iter().enumerate() {
            out[row * d1_new + new_c] = data[row * d1_old + old_c];
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(out, vec![d0, d1_new]), device)
}

/// `prune_width`'s row-removal primitive — the reverse of [`split_rows`], with no rescaling
/// (see [`remove_columns`]'s doc comment for why removal has no lossless equivalent).
fn remove_rows<B: Backend>(w: &Tensor<B, 2>, drop: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let [d0_old, d1] = w.dims();
    let data = w.clone().into_data().to_vec::<f32>().unwrap();
    let keep: Vec<usize> = (0..d0_old).filter(|r| !drop.contains(r)).collect();
    let d0_new = keep.len();
    let mut out = vec![0f32; d0_new * d1];
    for (new_r, &old_r) in keep.iter().enumerate() {
        out[new_r * d1..new_r * d1 + d1].copy_from_slice(&data[old_r * d1..old_r * d1 + d1]);
    }
    Tensor::<B, 2>::from_data(TensorData::new(out, vec![d0_new, d1]), device)
}

/// Bias analogue of [`remove_columns`] (1D).
fn remove_1d<B: Backend>(b: &Tensor<B, 1>, drop: &[usize], device: &B::Device) -> Tensor<B, 1> {
    let d_old = b.dims()[0];
    let data = b.clone().into_data().to_vec::<f32>().unwrap();
    let keep: Vec<f32> = (0..d_old).filter(|i| !drop.contains(i)).map(|i| data[i]).collect();
    let d_new = keep.len();
    Tensor::<B, 1>::from_data(TensorData::new(keep, vec![d_new]), device)
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
/// Same threshold `training_core.rs`'s Kirsch/pin-lug paths pass as `gate_awake_epsilon`
/// (`SolverConfig::stiffness`) - `UserDefinedProblem`/parametric specs have no such config
/// field (`NetworkSpec` never sets `use_piratenet`, so `awake_mask` is always empty for them
/// regardless of this value), but reusing the same constant keeps the classification
/// consistent if that ever changes rather than introducing an unrelated second threshold.
const GATE_AWAKE_EPSILON: f32 = 1e-4;

/// Stage I ("live network-evolution visualization") - builds a [`pinn_core::messages::
/// NetworkSnapshot`] from an already-available model reference. Reads ONLY existing parameter
/// tensors (`layer_weight_stats`/`awake_mask`, both pure reads with zero side effects) - adds
/// no forward pass, no gradient computation. Callers must gate this behind the SAME vis-cadence
/// throttle `VisFields` itself is built on (never call this every training step) - the cost is
/// small per call, but "small per call, called every step for thousands of steps" is exactly
/// the class of waste this project's own CLAUDE.md documents finding and fixing before.
pub fn network_snapshot<Bk: Backend>(model: &ElasticityNet<Bk>) -> pinn_core::messages::NetworkSnapshot {
    let stats = model.layer_weight_stats();
    pinn_core::messages::NetworkSnapshot {
        layer_mean_abs_weight: stats.iter().map(|&(mean, _)| mean).collect(),
        layer_max_abs_weight: stats.iter().map(|&(_, max)| max).collect(),
        awake_mask: model.awake_mask(GATE_AWAKE_EPSILON),
        layer_weights: model.all_weight_matrices(),
    }
}

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

    /// Issue #62 PH3-11: real, verified evidence that `Backend::seed` immediately before
    /// `ElasticityNetConfig::init` makes weight initialization reproducible - this codebase's
    /// own previously-documented negative finding (`provenance::RunProvenance::
    /// model_init_seeded`'s old doc comment) is now closed for real, not just claimed.
    ///
    /// Deliberately uses `training_core::BInner` (the SHIPPED backend - `NdArray` under
    /// `--features ndarray-backend`, the only configuration `app-egui`/the headless CLI
    /// actually build with, per `powershell_tool/CLAUDE.md`'s own "Debug vs release" section),
    /// not this file's own hardcoded `Wgpu` test backend used everywhere else in this module.
    /// A first version of this test used `Wgpu` and found genuine, deterministic-but-different
    /// weights across two identically-seeded runs, even after adding `Backend::sync` calls -
    /// traced to burn-wgpu's fusion/cubecl execution layer, where `Backend::seed`'s global
    /// mutable state mutation isn't tracked by the lazy op-fusion graph, so its ordering
    /// relative to queued `float_random` kernel dispatches isn't guaranteed by program order
    /// alone. `NdArray` has no such concern (eager, single-threaded CPU execution, its own
    /// separate `SEED` static in `burn-ndarray`) - and is the only backend this reproducibility
    /// claim needs to hold for, since it's the only one real runs ever use.
    #[test]
    fn seeding_before_init_produces_byte_identical_initial_weights() {
        use crate::training_core::{BDevice, BInner};
        let device = BDevice::default();
        let config = plain_mlp_config(6, 3);

        BInner::seed(&device, 12345);
        let model_a: ElasticityNet<BInner> = config.init(&device);
        // Force every lazily-initialized `Param` to materialize its actual random draw NOW,
        // before `seed` is called again - `Param` values in this burn version are lazily
        // computed on first access, so reading them only AFTER building both models would let
        // the two models' random draws interleave against a single shared RNG stream in
        // whatever order the reads happen to occur, not the order each model was built in
        // (confirmed the hard way: an earlier version of this test read both models' weights
        // only at the end and got real, deterministic-but-wrong divergence from exactly this).
        let wa: Vec<Vec<f32>> = model_a.layers.iter().map(|l| l.weight.val().into_data().to_vec().unwrap()).collect();
        let oa: Vec<f32> = model_a.out.weight.val().into_data().to_vec().unwrap();

        BInner::seed(&device, 12345);
        let model_b: ElasticityNet<BInner> = config.init(&device);
        let wb: Vec<Vec<f32>> = model_b.layers.iter().map(|l| l.weight.val().into_data().to_vec().unwrap()).collect();
        let ob: Vec<f32> = model_b.out.weight.val().into_data().to_vec().unwrap();

        assert_eq!(wa, wb, "same seed must reproduce byte-identical layer weights");
        assert_eq!(oa, ob, "same seed must reproduce byte-identical output-layer weights");
    }

    /// Sanity check that seeding actually has an effect (not a silent no-op that happens to
    /// produce identical weights regardless of seed) - same `BInner` rationale as the test
    /// immediately above.
    #[test]
    fn different_seeds_before_init_produce_different_initial_weights() {
        use crate::training_core::{BDevice, BInner};
        let device = BDevice::default();
        let config = plain_mlp_config(6, 3);

        BInner::seed(&device, 111);
        let model_a: ElasticityNet<BInner> = config.init(&device);
        BInner::seed(&device, 222);
        let model_b: ElasticityNet<BInner> = config.init(&device);

        let wa: Vec<f32> = model_a.layers[0].weight.val().into_data().to_vec().unwrap();
        let wb: Vec<f32> = model_b.layers[0].weight.val().into_data().to_vec().unwrap();
        assert_ne!(wa, wb, "different seeds should (overwhelmingly likely) produce different weights");
    }

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

    // ---- grow_width (issue #50, Net2WiderNet-style function-preserving width growth) ----

    fn plain_mlp_config(hidden_dim: usize, n_hidden: usize) -> ElasticityNetConfig {
        ElasticityNetConfig::new()
            .with_input_dim(2)
            .with_hidden_dim(hidden_dim)
            .with_n_hidden(n_hidden)
            .with_output_dim(2)
            .with_use_piratenet(false)
    }

    #[test]
    fn grow_width_preserves_function_plain_mlp() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = plain_mlp_config(4, 3).init(&device);
        let x = fixed_input(&device);

        // k=3: g = {0,1,2}, all distinct — no multiplicity > 1.
        let grown7 = model.grow_width(7, &device);
        let diff7: f32 = (model.forward(x.clone()) - grown7.forward(x.clone())).abs().sum().into_scalar();
        assert!(diff7 < 1e-5, "k=3 growth changed the function: diff={diff7}");

        // k=6 (4 -> 10): g = {0,1,2,3,0,1} forces m(0)=m(1)=2 > 1.
        let grown10 = model.grow_width(10, &device);
        let diff10: f32 = (model.forward(x.clone()) - grown10.forward(x)).abs().sum().into_scalar();
        assert!(diff10 < 1e-5, "k=6 growth changed the function: diff={diff10}");
    }

    #[test]
    fn grow_width_preserves_function_piratenet_gated() {
        let device = WgpuDevice::default();
        let mut model: ElasticityNet<TB> = ElasticityNetConfig::new()
            .with_input_dim(2)
            .with_hidden_dim(4)
            .with_n_hidden(4)
            .with_output_dim(2)
            .with_use_piratenet(true)
            .init(&device);
        // Force nonzero gates so the residual path is actually exercised (n_hidden=4 -> 3 gates).
        set_gate(&mut model, 0, 0.6, &device);
        set_gate(&mut model, 1, -0.3, &device);
        set_gate(&mut model, 2, 0.9, &device);

        let x = fixed_input(&device);
        let grown = model.grow_width(9, &device); // k=5 (4->9)

        let diff: f32 = (model.forward(x.clone()) - grown.forward(x)).abs().sum().into_scalar();
        assert!(diff < 1e-5, "PirateNet gated growth changed the function: diff={diff}");
    }

    #[test]
    fn grow_width_boundary_k_equals_1() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = plain_mlp_config(4, 3).init(&device);
        let x = fixed_input(&device);

        let grown = model.grow_width(5, &device); // k=1
        let diff: f32 = (model.forward(x.clone()) - grown.forward(x)).abs().sum().into_scalar();
        assert!(diff < 1e-5, "k=1 growth changed the function: diff={diff}");
    }

    #[test]
    fn grow_width_dims_are_correct() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = plain_mlp_config(4, 3).init(&device);
        let grown = model.grow_width(7, &device);

        assert_eq!(grown.layers[0].weight.val().dims(), [2, 7]);
        assert_eq!(grown.layers[1].weight.val().dims(), [7, 7]);
        assert_eq!(grown.out.weight.val().dims(), [7, 2]);
        assert_eq!(grown.layers[0].bias.as_ref().unwrap().val().dims(), [7]);
        assert_eq!(grown.out.bias.as_ref().unwrap().val().dims(), [2]);
        assert_eq!(grown.gates.len(), model.gates.len());
    }

    #[test]
    fn grow_width_preserves_param_ids_for_weights() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = plain_mlp_config(4, 3).init(&device);
        let grown = model.grow_width(7, &device);

        for i in 0..model.layers.len() {
            assert_eq!(
                grown.layers[i].weight.id, model.layers[i].weight.id,
                "layers[{i}].weight ParamId should be preserved across growth"
            );
        }
        assert_eq!(grown.out.weight.id, model.out.weight.id);
    }

    #[test]
    fn grow_width_gives_bias_params_fresh_ids() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = plain_mlp_config(4, 3).init(&device);
        let grown = model.grow_width(7, &device);

        for i in 0..model.layers.len() {
            let old_id = model.layers[i].bias.as_ref().unwrap().id;
            let new_id = grown.layers[i].bias.as_ref().unwrap().id;
            assert_ne!(old_id, new_id, "layers[{i}].bias should get a fresh ParamId on growth");
        }
    }

    #[test]
    fn grow_width_new_columns_are_exact_duplicates_not_zero() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = plain_mlp_config(4, 3).init(&device);
        let h_old = 4;
        let grown = model.grow_width(7, &device); // k=3, g = {0,1,2}

        let w_old = model.layers[0].weight.val().into_data();
        let w_old = w_old.as_slice::<f32>().unwrap();
        let w_new = grown.layers[0].weight.val().into_data();
        let w_new = w_new.as_slice::<f32>().unwrap();
        let d0 = 2usize; // input_dim
        let d1_new = 7usize;

        let g = [0usize, 1, 2];
        let mut any_nonzero = false;
        for row in 0..d0 {
            for (j, &gj) in g.iter().enumerate() {
                let new_val = w_new[row * d1_new + h_old + j];
                let src_val = w_old[row * h_old + gj];
                assert_eq!(new_val, src_val, "row {row} new col {j} should exactly duplicate source col {gj}");
                if new_val != 0.0 {
                    any_nonzero = true;
                }
            }
        }
        assert!(any_nonzero, "regression guard: new columns must not be all-zero (rejected zero-padding scheme)");
    }

    #[test]
    fn grow_width_next_layer_rows_sum_to_original_row() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = plain_mlp_config(4, 3).init(&device);
        let h_old = 4;
        // k=6 (4->10): g = [0,1,2,3,0,1] (j%4 for j in 0..6) => s=0's duplicates are at j=0 and
        // j=4 (new rows h_old+0=4 and h_old+4=8), giving m(0)=3 (a genuine multiplicity>2 case,
        // not just the m=2 case already covered by `grow_width_preserves_function_plain_mlp`).
        let grown = model.grow_width(10, &device);

        let w_old = model.layers[1].weight.val().into_data();
        let w_old = w_old.as_slice::<f32>().unwrap();
        let w_new = grown.layers[1].weight.val().into_data();
        let w_new = w_new.as_slice::<f32>().unwrap();
        let d1 = h_old; // layers[1]'s output side also grew, but the first h_old columns are
                         // the pre-existing (index-stable) output columns, unaffected by this
                         // row-only (input-side) check.
        let d1_new = 10usize;

        // s=0's twins are at new row indices h_old+0=4 and h_old+4=8 (see g above).
        let s = 0usize;
        let twin_rows = [h_old, h_old + 4];
        for col in 0..d1 {
            let orig = w_old[s * d1 + col];
            let a = w_new[s * d1_new + col];
            let sum_twins: f32 = twin_rows.iter().map(|&r| w_new[r * d1_new + col]).sum();
            assert!(
                (a + sum_twins - orig).abs() < 1e-6,
                "row {s} split across original+{} twins should sum back to the original row at col {col}: {a}+{sum_twins} != {orig}",
                twin_rows.len()
            );
        }
    }

    #[test]
    #[should_panic]
    fn grow_width_panics_when_shrinking_or_equal() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = plain_mlp_config(4, 3).init(&device);
        let _ = model.grow_width(4, &device); // equal — must panic
    }

    #[test]
    #[should_panic]
    fn grow_width_panics_when_shrinking() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TB> = plain_mlp_config(4, 3).init(&device);
        let _ = model.grow_width(3, &device); // shrink — must panic
    }

    // ─── Stage I: live network-evolution visualization ──────────────────────────────────────

    #[test]
    fn layer_weight_stats_returns_one_entry_per_layer_with_finite_nonnegative_values() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TBInner> = plain_mlp_config(4, 3).init(&device);
        let stats = model.layer_weight_stats();
        assert_eq!(stats.len(), 3, "one entry per `layers` element (n_hidden), matching plain_mlp_config(4, 3)");
        for (mean, max) in stats {
            assert!(mean.is_finite() && mean >= 0.0, "mean |weight| must be finite and non-negative, got {mean}");
            assert!(max.is_finite() && max >= 0.0, "max |weight| must be finite and non-negative, got {max}");
            assert!(max >= mean - 1e-6, "max must be >= mean, got mean={mean} max={max}");
        }
    }

    #[test]
    fn per_neuron_magnitudes_has_one_vec_per_prunable_layer_each_hidden_dim_long() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TBInner> = plain_mlp_config(4, 3).init(&device);
        let mags = model.per_neuron_magnitudes();
        assert_eq!(mags.len(), 3, "one entry per `layers` element, `out` excluded");
        for layer_mags in &mags {
            assert_eq!(layer_mags.len(), 4, "one score per hidden unit (hidden_dim=4)");
            assert!(layer_mags.iter().all(|m| m.is_finite() && *m >= 0.0));
        }
    }

    #[test]
    fn per_neuron_magnitudes_averages_the_right_column_not_the_whole_matrix() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TBInner> = plain_mlp_config(2, 2).init(&device);
        let expected: Vec<f32> = {
            let w = model.all_weight_matrices()[0].clone(); // layers[0], [d_input, d_output]
            (0..w.ncols()).map(|col| {
                let sum: f32 = (0..w.nrows()).map(|row| w[[row, col]].abs()).sum();
                sum / w.nrows() as f32
            }).collect()
        };
        let mags = model.per_neuron_magnitudes();
        for (got, want) in mags[0].iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn network_snapshot_awake_mask_is_empty_when_piratenet_disabled() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TBInner> = plain_mlp_config(4, 3).init(&device);
        let snap = network_snapshot(&model);
        assert_eq!(snap.layer_mean_abs_weight.len(), 3);
        assert_eq!(snap.layer_max_abs_weight.len(), 3);
        assert!(snap.awake_mask.is_empty(), "plain MLP (use_piratenet=false) must report an empty awake_mask, not fabricated values");
    }

    #[test]
    fn network_snapshot_awake_mask_matches_the_model_s_own_awake_mask_when_piratenet_enabled() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TBInner> = piratenet_config().init(&device);
        let snap = network_snapshot(&model);
        assert_eq!(snap.awake_mask, model.awake_mask(GATE_AWAKE_EPSILON), "network_snapshot must reuse the model's own classification, not a second divergent one");
    }

    // ─── Network diagram (approved neuron-and-edge design): real weight matrices ────────────

    #[test]
    fn all_weight_matrices_includes_the_output_layer_with_correct_shapes() {
        let device = WgpuDevice::default();
        // input_dim=2, hidden_dim=4, n_hidden=3, output_dim=2 (see plain_mlp_config)
        let model: ElasticityNet<TBInner> = plain_mlp_config(4, 3).init(&device);
        let mats = model.all_weight_matrices();
        assert_eq!(mats.len(), 4, "n_hidden (3) layers + 1 output layer");
        assert_eq!(mats[0].dim(), (2, 4), "layers[0]: input_dim -> hidden_dim");
        assert_eq!(mats[1].dim(), (4, 4), "layers[1]: hidden_dim -> hidden_dim");
        assert_eq!(mats[2].dim(), (4, 4), "layers[2]: hidden_dim -> hidden_dim");
        assert_eq!(mats[3].dim(), (4, 2), "out: hidden_dim -> output_dim - INCLUDED, unlike layer_weight_stats/awake_mask");
        for m in &mats {
            assert!(m.iter().all(|v| v.is_finite()), "every weight must be finite for a freshly-initialized model");
        }
    }

    #[test]
    fn all_weight_matrices_values_match_the_model_s_own_layer_weight_stats() {
        // Cross-check: the (mean, max) `layer_weight_stats` reports for the non-output layers
        // must be hand-derivable from the SAME raw values `all_weight_matrices` exposes - proves
        // the two views aren't drifting apart (e.g. a future edit to one that forgets the other).
        let device = WgpuDevice::default();
        let model: ElasticityNet<TBInner> = plain_mlp_config(4, 3).init(&device);
        let mats = model.all_weight_matrices();
        let stats = model.layer_weight_stats();
        assert_eq!(stats.len(), 3, "layer_weight_stats excludes the output layer");
        for (i, (mean, max)) in stats.iter().enumerate() {
            let vals: Vec<f32> = mats[i].iter().map(|v| v.abs()).collect();
            let hand_mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let hand_max = vals.iter().copied().fold(0.0f32, f32::max);
            assert!((mean - hand_mean).abs() < 1e-6, "layer {i} mean mismatch: {mean} vs {hand_mean}");
            assert!((max - hand_max).abs() < 1e-6, "layer {i} max mismatch: {max} vs {hand_max}");
        }
    }

    #[test]
    fn network_snapshot_layer_weights_matches_all_weight_matrices() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TBInner> = plain_mlp_config(4, 3).init(&device);
        let snap = network_snapshot(&model);
        let direct = model.all_weight_matrices();
        assert_eq!(snap.layer_weights.len(), direct.len());
        for (a, b) in snap.layer_weights.iter().zip(direct.iter()) {
            assert_eq!(a, b, "network_snapshot must carry the exact same matrices all_weight_matrices returns");
        }
    }

    // ─── Smart adaptive architecture: depth growth/shrink, width shrink ─────────────────────

    fn probe_input() -> Tensor<TBInner, 2> {
        let device = WgpuDevice::default();
        Tensor::from_data(TensorData::new(vec![0.3_f32, -0.7, 0.1, 0.9, -0.2, 0.5], vec![3, 2]), &device)
    }

    #[test]
    fn append_dormant_layer_is_output_identical_at_insertion() {
        let device = WgpuDevice::default();
        let mut model: ElasticityNet<TBInner> = piratenet_config().init(&device);
        // Force real (non-zero) gate values first - proves appending doesn't disturb EXISTING
        // awake blocks' contributions, not just a degenerate all-zero-gates case.
        model.force_gate_for_test(0, 0.6, &device);
        model.force_gate_for_test(1, -0.3, &device);

        let x = probe_input();
        let before = model.forward(x.clone());
        let grown = model.append_dormant_layer(1.6666666666666667, &device);

        assert_eq!(grown.layers.len(), model.layers.len() + 1);
        assert_eq!(grown.gates.len(), model.gates.len() + 1);
        assert_eq!(grown.gates.last().unwrap().val().into_scalar().elem::<f32>(), 0.0);

        let after = grown.forward(x);
        let diff: f32 = (before - after).abs().sum().into_scalar();
        assert!(diff < 1e-5, "appending a dormant layer must not change the network's output: diff={diff}");
    }

    #[test]
    #[should_panic(expected = "requires a fully-gated network")]
    fn append_dormant_layer_panics_on_a_plain_mlp() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TBInner> = plain_mlp_config(4, 3).init(&device);
        let _ = model.append_dormant_layer(1.6666666666666667, &device);
    }

    #[test]
    fn remove_layer_is_output_identical_when_gate_is_dormant() {
        let device = WgpuDevice::default();
        let mut model: ElasticityNet<TBInner> = piratenet_config().init(&device);
        model.force_gate_for_test(0, 0.6, &device); // layers[1]: awake
        model.force_gate_for_test(1, 0.0, &device); // layers[2]: dormant

        let x = probe_input();
        let before = model.forward(x.clone());
        let shrunk = model.remove_layer(2, 1e-4);

        assert_eq!(shrunk.layers.len(), model.layers.len() - 1);
        assert_eq!(shrunk.gates.len(), model.gates.len() - 1);

        let after = shrunk.forward(x);
        let diff: f32 = (before - after).abs().sum().into_scalar();
        assert!(diff < 1e-5, "removing an already-dormant block must not change the network's output: diff={diff}");
    }

    #[test]
    #[should_panic(expected = "is not dormant")]
    fn remove_layer_panics_when_gate_is_not_dormant() {
        let device = WgpuDevice::default();
        let mut model: ElasticityNet<TBInner> = piratenet_config().init(&device);
        model.force_gate_for_test(0, 0.6, &device);
        let _ = model.remove_layer(1, 1e-4);
    }

    #[test]
    #[should_panic]
    fn remove_layer_panics_on_layer_zero() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TBInner> = piratenet_config().init(&device);
        let _ = model.remove_layer(0, 1e-4);
    }

    #[test]
    fn prune_width_removes_the_intended_indices_uniformly_across_every_layer() {
        let device = WgpuDevice::default();
        // input_dim=2, hidden_dim=4, n_hidden=3, output_dim=2 (plain_mlp_config)
        let model: ElasticityNet<TBInner> = plain_mlp_config(4, 3).init(&device);
        let before = model.all_weight_matrices();

        let pruned = model.prune_width(&[0, 2], &device);
        let after = pruned.all_weight_matrices();

        // layers[0]: output columns only (input_dim untouched).
        assert_eq!(after[0].dim(), (before[0].nrows(), before[0].ncols() - 2));
        // layers[1..]: BOTH input rows and output columns shrink - required for
        // `forward_masked`'s residual sum (`h = h + tanh(layer(h)) * alpha`) to stay shape-
        // consistent when this network is gated.
        assert_eq!(after[1].dim(), (before[1].nrows() - 2, before[1].ncols() - 2));
        assert_eq!(after[2].dim(), (before[2].nrows() - 2, before[2].ncols() - 2));
        // `out`: input rows only.
        let out_idx = after.len() - 1;
        assert_eq!(after[out_idx].dim(), (before[out_idx].nrows() - 2, before[out_idx].ncols()));

        // Correctness, not just shape: the SURVIVING columns (1, 3) of layers[0] must be
        // byte-identical to the originals, in order - proves the right global indices were
        // dropped everywhere, not just shape-compatible ones.
        for row in 0..before[0].nrows() {
            assert_eq!(after[0][[row, 0]], before[0][[row, 1]]);
            assert_eq!(after[0][[row, 1]], before[0][[row, 3]]);
        }
        // Same check on `out`'s surviving INPUT rows (1, 3).
        for col in 0..before[out_idx].ncols() {
            assert_eq!(after[out_idx][[0, col]], before[out_idx][[1, col]]);
            assert_eq!(after[out_idx][[1, col]], before[out_idx][[3, col]]);
        }
    }

    #[test]
    #[should_panic(expected = "cannot drop every hidden unit")]
    fn prune_width_panics_when_dropping_every_hidden_unit() {
        let device = WgpuDevice::default();
        let model: ElasticityNet<TBInner> = plain_mlp_config(4, 3).init(&device);
        let _ = model.prune_width(&[0, 1, 2, 3], &device);
    }

    #[test]
    fn prune_width_on_a_gated_network_does_not_break_forward_masked() {
        // Regression test for the bug an earlier, per-layer version of `prune_width` had:
        // shrinking only one layer's boundary desynced the shared hidden_dim the gated
        // residual sum requires, which would panic on the very next forward pass. This
        // network has REAL nonzero gates (not the degenerate all-zero launch state), so the
        // residual sum is actually exercised, not skipped.
        let device = WgpuDevice::default();
        let mut model: ElasticityNet<TBInner> = piratenet_config().init(&device);
        model.force_gate_for_test(0, 0.6, &device);
        model.force_gate_for_test(1, -0.3, &device);

        let pruned = model.prune_width(&[0, 2], &device);
        let y = pruned.forward(probe_input());
        assert_eq!(y.dims(), [3, 2]);
        let data = y.into_data();
        let values = data.as_slice::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()), "pruned gated network produced non-finite output");
    }
}

