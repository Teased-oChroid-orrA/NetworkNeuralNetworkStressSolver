pub mod soap_muon;

pub use soap_muon::{SoapMuon, SoapMuonConfig, SoapMuonState, migrate_soap_muon_state_for_growth};

use burn::module::ParamId;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor};
use burn::tensor::ops::Device;

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
