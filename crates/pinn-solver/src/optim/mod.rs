pub mod soap_muon;

pub use soap_muon::{
    SoapMuon, SoapMuonConfig, SoapMuonState, migrate_soap_muon_state_for_growth,
    migrate_soap_muon_state_for_shrink,
};

use burn::module::ParamId;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor};
use burn::tensor::ops::Device;

use crate::architecture_controller::ArchAction;
use crate::decision_maker::OptimizerTier;
use crate::network::ElasticityNet;
use crate::training_core::{B, BInner};

/// Optimizer for 2D weight-matrix parameters: either the SOAP-Muon hybrid, or — as an
/// explicit fallback, per [`pinn_core::messages::SolverConfig::use_soap_muon`] — plain
/// AdamW for every parameter. Kept as a runtime choice (not just a config default) so a
/// run can fall back without a rebuild if the hybrid proves unstable on some configuration.
pub enum WeightOptim {
    SoapMuon(OptimizerAdaptor<SoapMuon<BInner>, ElasticityNet<B>, B>),
    AdamWOnly(OptimizerAdaptor<burn::optim::AdamW, ElasticityNet<B>, B>),
}

impl WeightOptim {
    pub fn new(use_soap_muon: bool) -> Self {
        if use_soap_muon {
            WeightOptim::SoapMuon(SoapMuonConfig::new().init())
        } else {
            WeightOptim::AdamWOnly(AdamWConfig::new().init())
        }
    }

    /// Construct the appropriate optimizer variant for a given tier.
    ///
    /// `Explore` → `AdamWOnly` (cheap; near-convex Phase 1 landscape).
    /// `Align | Converge` → SOAP-Muon (curvature preconditioning; Converge is a safety net
    ///   for the rare case that L-BFGS can't be invoked on the current step).
    ///
    /// When `use_soap_muon = false` the escape hatch maps every tier to `AdamWOnly`.
    pub fn from_tier(use_soap_muon: bool, tier: &OptimizerTier) -> Self {
        match tier {
            OptimizerTier::Explore => WeightOptim::AdamWOnly(AdamWConfig::new().init()),
            OptimizerTier::Align | OptimizerTier::Converge => WeightOptim::new(use_soap_muon),
        }
    }

    pub fn step(
        &mut self,
        lr: f64,
        model: ElasticityNet<B>,
        grads: GradientsParams,
    ) -> ElasticityNet<B> {
        match self {
            WeightOptim::SoapMuon(o) => o.step(lr, model, grads),
            WeightOptim::AdamWOnly(o) => o.step(lr, model, grads),
        }
    }

    /// Migrates one weight parameter's persisted optimizer state onto its new (grown) shape,
    /// after `ElasticityNet::grow_width` (issue #50). `weight_id` is the (growth-preserving,
    /// see `grow_width`'s doc comment) `ParamId` shared by the pre- and post-growth weight
    /// `Param`; `grow_dim0`/`grow_dim1` mirror
    /// [`soap_muon::migrate_soap_muon_state_for_growth`]'s parameters of the same name.
    ///
    /// Only meaningful for `SoapMuon` mode, which carries real per-parameter moment/
    /// accumulator history worth migrating. **No-op for `AdamWOnly` mode** — deliberately:
    /// AdamW-only weights are meant to cold-start on growth instead, the same treatment
    /// `grow_width` already gives every bias `Param` (a fresh `ParamId`, discarding its old
    /// moments) rather than attempting a SoapMuon-grade migration for a mode that has no
    /// Shampoo/eigenbasis state to migrate in the first place. The caller is responsible for
    /// actually discarding `AdamWOnly`'s now-shape-mismatched records after growth by
    /// rebuilding a fresh `WeightOptim::AdamWOnly` (e.g. `WeightOptim::new(false)`) rather
    /// than calling this method in that mode — see `headless.rs`'s growth-trigger site, which
    /// does exactly that.
    pub fn migrate_for_growth(
        &mut self,
        weight_id: ParamId,
        grow_dim0: Option<(usize, usize)>,
        grow_dim1: Option<(usize, usize)>,
        device: &Device<BInner>,
    ) {
        if let WeightOptim::SoapMuon(o) = self {
            // `OptimizerAdaptor` exposes its per-ParamId state only via `to_record()`/
            // `load_record()` (burn 0.21's `Optimizer` trait) — `to_record()` clones the
            // `HashMap<ParamId, AdaptorRecord<..>>` without consuming `o`, so we can read it
            // through `&*o`, then rebuild a fresh adaptor (from a clone of the wrapped
            // `SoapMuon` config/state, via the only public constructor, `From<O>`) and load
            // the migrated map back in — `load_record` requires ownership of `self`, which
            // `*o` (a `&mut` field inside this enum, not an owned local) can't give up
            // directly, so we construct the replacement in a local and assign it back.
            let optim_clone = o.optim().clone();
            let mut records = o.to_record();
            if let Some(record) = records.remove(&weight_id) {
                let state: SoapMuonState<BInner> = record.into_state::<2>();
                let migrated = migrate_soap_muon_state_for_growth(&state, grow_dim0, grow_dim1, device);
                records.insert(
                    weight_id,
                    burn::optim::record::AdaptorRecord::from_state::<2>(migrated),
                );
            }
            let fresh: OptimizerAdaptor<SoapMuon<BInner>, ElasticityNet<B>, B> =
                OptimizerAdaptor::from(optim_clone);
            *o = fresh.load_record(records);
        }
    }

    /// Smart adaptive architecture: shrink-side analogue of [`Self::migrate_for_growth`] — used
    /// after [`crate::network::ElasticityNet::prune_width`] narrows a weight `Param` that KEEPS
    /// its original `ParamId` (unlike [`crate::network::ElasticityNet::remove_layer`], which
    /// deletes an entire layer/`ParamId` outright and therefore needs no migration call at all —
    /// the orphaned state simply stops being looked up). `shrink_dim0`/`shrink_dim1` are the
    /// SURVIVING indices for that axis (in order), or `None` if that axis didn't change size.
    /// Same `AdamWOnly` no-op / caller-rebuilds-fresh convention as `migrate_for_growth`.
    pub fn migrate_for_shrink(
        &mut self,
        weight_id: ParamId,
        shrink_dim0: Option<&[usize]>,
        shrink_dim1: Option<&[usize]>,
        device: &Device<BInner>,
    ) {
        if let WeightOptim::SoapMuon(o) = self {
            let optim_clone = o.optim().clone();
            let mut records = o.to_record();
            if let Some(record) = records.remove(&weight_id) {
                let state: SoapMuonState<BInner> = record.into_state::<2>();
                let migrated = migrate_soap_muon_state_for_shrink(&state, shrink_dim0, shrink_dim1, device);
                records.insert(
                    weight_id,
                    burn::optim::record::AdaptorRecord::from_state::<2>(migrated),
                );
            }
            let fresh: OptimizerAdaptor<SoapMuon<BInner>, ElasticityNet<B>, B> =
                OptimizerAdaptor::from(optim_clone);
            *o = fresh.load_record(records);
        }
    }
}

/// Bias optimizer (1D parameters) — always plain AdamW.
pub type BiasOptim = OptimizerAdaptor<burn::optim::AdamW, ElasticityNet<B>, B>;

pub fn make_bias_optim() -> BiasOptim {
    AdamWConfig::new().init()
}

/// PirateNet gate-scalar optimizer — always plain AdamW, stepped with a stiffness-scaled
/// learning rate. Its gradient set is empty when `use_piratenet=false`, making `step()` a
/// no-op in that case.
pub type GateOptim = OptimizerAdaptor<burn::optim::AdamW, ElasticityNet<B>, B>;

pub fn make_gate_optim() -> GateOptim {
    AdamWConfig::new().init()
}

/// Smart adaptive architecture — applies one [`ArchAction`] (from `ArchitectureController::
/// observe`) to a live model + its weight optimizer. Shared by `run_training_user_problem` and
/// `run_training_parametric` so their adaptive behavior can't silently drift apart; mirrors the
/// exact per-layer grow/shrink axis rules `headless.rs`'s pre-existing width-growth site already
/// established (see that module's own doc comment on its growth block) — `layers[0]` only ever
/// touches its output side, `layers[1..]` touch both sides, `out` only ever touches its input
/// side, for both growth (`grow_width`) and shrink (`prune_width`).
///
/// `snapshot` holds the model + its `(hidden_dim, n_hidden)` from immediately BEFORE the most
/// recent speculative action (`GrowWidth`/`GrowDepth`/`PruneWidth`) — this function sets it on
/// those three actions and consumes it on `RevertLastChange`. `ShrinkDepth` never touches it
/// (a dormant gated block is provably zero-effect to remove — see `ElasticityNet::remove_layer`'s
/// doc comment — so it is never watched/reverted). `gate_epsilon` should be the SAME value the
/// caller's `ArchitectureConfig` was constructed with, so `ShrinkDepth`'s dormancy re-check
/// agrees with the controller's own classification.
///
/// Returns the updated model, a human-readable description for `ArchitectureEvent`, and the new
/// `(hidden_dim, n_hidden)` — the caller updates its own tracking locals from these (the same
/// `current_hidden_dim` convention `headless.rs` already uses for its own growth event, needed
/// so a later checkpoint save records the LIVE architecture, not the original spec's static one).
#[allow(clippy::too_many_arguments)]
pub fn apply_arch_action(
    action: &ArchAction,
    model: ElasticityNet<B>,
    snapshot: &mut Option<(ElasticityNet<B>, usize, usize)>,
    optim_w: &mut WeightOptim,
    use_soap_muon: bool,
    gate_epsilon: f32,
    hidden_dim: usize,
    n_hidden: usize,
    device: &Device<BInner>,
) -> (ElasticityNet<B>, String, usize, usize) {
    match action {
        ArchAction::GrowWidth(new_hidden_dim) => {
            *snapshot = Some((model.clone(), hidden_dim, n_hidden));
            let (weight_ids, _) = model.param_ids();
            let grown = model.grow_width(*new_hidden_dim, device);
            if use_soap_muon {
                let n_layers = weight_ids.len() - 1; // last id is `out`
                for (i, &id) in weight_ids.iter().enumerate() {
                    if i == n_layers {
                        optim_w.migrate_for_growth(id, Some((hidden_dim, *new_hidden_dim)), None, device);
                    } else if i == 0 {
                        optim_w.migrate_for_growth(id, None, Some((hidden_dim, *new_hidden_dim)), device);
                    } else {
                        optim_w.migrate_for_growth(
                            id, Some((hidden_dim, *new_hidden_dim)), Some((hidden_dim, *new_hidden_dim)), device,
                        );
                    }
                }
            } else {
                *optim_w = WeightOptim::new(false);
            }
            (grown, format!("Grew width {hidden_dim} -> {new_hidden_dim}"), *new_hidden_dim, n_hidden)
        }
        ArchAction::GrowDepth => {
            *snapshot = Some((model.clone(), hidden_dim, n_hidden));
            // A brand-new layer's `Param`s get fresh `ParamId`s (see `append_dormant_layer`'s
            // doc comment) - naturally absent from the optimizer's record map, so no migration
            // call is needed; every EXISTING layer's shape/`ParamId` is untouched.
            let grown = model.append_dormant_layer(1.6666666666666667, device);
            (grown, format!("Grew depth {n_hidden} -> {}", n_hidden + 1), hidden_dim, n_hidden + 1)
        }
        ArchAction::ShrinkDepth(idx) => {
            // Never watched/reverted (provably zero-effect) - no snapshot taken. The removed
            // layer's whole `ParamId` disappears with it; its now-orphaned optimizer record is
            // simply never looked up again - no migration call needed either.
            let shrunk = model.remove_layer(*idx, gate_epsilon);
            (shrunk, format!("Removed dormant layer {idx} ({n_hidden} -> {})", n_hidden - 1), hidden_dim, n_hidden - 1)
        }
        ArchAction::PruneWidth { drop_indices } => {
            *snapshot = Some((model.clone(), hidden_dim, n_hidden));
            let (weight_ids, _) = model.param_ids();
            let keep: Vec<usize> = (0..hidden_dim).filter(|i| !drop_indices.contains(i)).collect();
            let new_hidden_dim = keep.len();
            let pruned = model.prune_width(drop_indices, device);
            if use_soap_muon {
                let n_layers = weight_ids.len() - 1;
                for (i, &id) in weight_ids.iter().enumerate() {
                    if i == n_layers {
                        optim_w.migrate_for_shrink(id, Some(&keep), None, device);
                    } else if i == 0 {
                        optim_w.migrate_for_shrink(id, None, Some(&keep), device);
                    } else {
                        optim_w.migrate_for_shrink(id, Some(&keep), Some(&keep), device);
                    }
                }
            } else {
                *optim_w = WeightOptim::new(false);
            }
            (pruned, format!("Pruned {} neurons ({hidden_dim} -> {new_hidden_dim})", drop_indices.len()), new_hidden_dim, n_hidden)
        }
        ArchAction::RevertLastChange => match snapshot.take() {
            Some((restored, h, n)) => {
                // Accepted cost (see the design plan): momentum state re-warms from scratch,
                // but the actual trained WEIGHTS are fully restored from the snapshot - that's
                // what "not throwing away trained progress" means here.
                *optim_w = WeightOptim::new(use_soap_muon);
                (restored, "Reverted last architecture change (no improvement)".to_string(), h, n)
            }
            // Should not happen in practice (the controller only emits `RevertLastChange`
            // after a watched action, which always sets `snapshot`) - defensive no-op instead
            // of a panic if it ever does.
            None => (model, "Revert requested but no change was pending (no-op)".to_string(), hidden_dim, n_hidden),
        },
    }
}
