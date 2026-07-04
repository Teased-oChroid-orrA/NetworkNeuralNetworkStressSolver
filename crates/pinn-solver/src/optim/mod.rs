pub mod soap_muon;

pub use soap_muon::{SoapMuon, SoapMuonConfig, SoapMuonState};

use burn::optim::{AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor};

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
