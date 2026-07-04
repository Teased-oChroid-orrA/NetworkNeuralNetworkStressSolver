/// Generic boundary-value-problem trait family (tensor-graph layer). Builds on
/// `pinn_core::problem` (geometry/sampling/ansatz, burn-free) to let the training loop
/// (`training_core`/`runner`/`headless`) drive an arbitrary number of physical domains and
/// loss terms instead of being hardwired to the single-domain Kirsch problem.
///
/// `KirschProblem` (`kirsch_problem.rs`) is the reference implementation — exactly one
/// domain. A follow-up pin-in-lug problem adds a second domain plus cross-domain
/// (`LossTerm::domains().len() == 2`) contact loss terms on top of the same traits.
use std::collections::HashMap;

use burn::tensor::{backend::Backend, Tensor};

use pinn_core::messages::SolverConfig;
use pinn_core::problem::{DirichletAnsatz, DomainId, DomainSamplingStrategy, DomainSpec};

use crate::fd_stencil::FdConfig;
use crate::network::ElasticityNet;
use crate::optim::{BiasOptim, GateOptim, WeightOptim};

/// Autodiff-enabled backend used throughout the training loop. Re-exported from
/// `training_core` (the pre-existing single source of truth) rather than redefined here.
pub use crate::training_core::B;

/// Mutable per-domain state threaded through training: the domain's own network plus its
/// normalization/reference scales (analogous to today's Kirsch-only `u_ref`/`ref_energy`/
/// `ref_stress2` in `TrainingState`, generalized to N domains).
pub struct DomainState<Bk: Backend> {
    pub id: DomainId,
    pub model: ElasticityNet<Bk>,
    /// Displacement scale [m] this domain's raw network output is multiplied by.
    pub u_ref: f32,
    /// Strain-energy-density scale [Pa] used to normalize this domain's energy loss to O(1).
    pub ref_energy: f32,
    /// Stress-squared scale [Pa^2] used to normalize this domain's stress-based losses to O(1).
    pub ref_stress2: f32,
}

/// Per-domain forward-pass outputs handed to `LossTerm::compute`. `strains`/`normals` are
/// `None` when the term doesn't need them (e.g. a term operating purely on `raw_out`).
pub struct DomainForwardOutputs<'a, Bk: Backend> {
    pub domain: DomainId,
    pub raw_out: &'a Tensor<Bk, 2>,
    pub strains: Option<(Tensor<Bk, 1>, Tensor<Bk, 1>, Tensor<Bk, 1>)>,
    pub normals: Option<(Tensor<Bk, 1>, Tensor<Bk, 1>)>,
}

/// One additive term of the total physics loss. Implementations describe *which* domain(s)
/// they read from and *how* to compute their (unweighted) scalar loss from those domains'
/// forward outputs; the training loop is responsible for weighting (SAW-BRDR / fixed) and
/// summing all active terms into a single tensor before calling `.backward()` once per step
/// — terms are not each independently backpropagated.
pub trait LossTerm: Send + Sync {
    fn name(&self) -> &'static str;

    /// Domain(s) this term reads from. A single-domain problem's terms return one id; a
    /// cross-domain term (e.g. pin-in-lug contact) returns two.
    fn domains(&self) -> Vec<DomainId>;

    fn compute(&self, inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1>;

    /// True if this term is only active once Phase 2 begins (e.g. Kirsch's stress-probe
    /// loss). Defaults to always-active.
    fn phase2_only(&self) -> bool {
        false
    }

    /// Named point-set (one per entry in [`Self::domains`], same order) this term's
    /// forward pass should be built from — e.g. "interior", "hole", "interface". Defaults
    /// to `"interior"` for every domain, matching every existing single-domain Kirsch term
    /// (whose real forward pass is still wired by hand in `training_core::step_physics`,
    /// unaffected by this method — it only matters to `step_physics_multi`'s generic
    /// point-set lookup).
    fn point_sets(&self) -> Vec<&'static str> {
        self.domains().iter().map(|_| "interior").collect()
    }

    /// Classifies this term as either enforcing interior PDE/equilibrium physics or a
    /// boundary/interface condition — used by `compute_gradient_conflict_multi`'s dual-backward
    /// split. Defaults to `Bc`. Every concrete LossTerm impl in pinlug_problem.rs must override
    /// this explicitly (do not rely on the default silently) — a RED test pins each term's
    /// expected classification so a future term added without an override is caught immediately.
    fn conflict_group(&self) -> ConflictGroup {
        ConflictGroup::Bc
    }
}

/// Classifies a [`LossTerm`] as either enforcing interior PDE/equilibrium physics or a
/// boundary/interface condition. See [`LossTerm::conflict_group`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConflictGroup { Physics, Bc }

/// A complete boundary-value problem: its domain(s), their sampling/ansatz strategies, loss
/// terms, base SAW-BRDR weights, curriculum length, and convergence metric/target.
pub trait BoundaryValueProblem: Send + Sync {
    fn domains(&self) -> &[DomainSpec];

    fn sampling_strategy(&self, domain_idx: usize) -> &dyn DomainSamplingStrategy;

    fn ansatz(&self, domain_idx: usize) -> &dyn DirichletAnsatz;

    fn loss_terms(&self) -> Vec<Box<dyn LossTerm>>;

    /// Initial (Phase 1) SAW-BRDR base weight for the named loss term.
    fn base_weight(&self, term_name: &str) -> f32;

    /// Number of Phase-1 (BC-only) steps before phase2-only loss terms activate.
    fn phase1_steps(&self) -> usize;

    /// Problem-specific convergence metric (e.g. K_t for Kirsch), evaluated from the
    /// current per-domain state. `None` when the metric can't be evaluated yet (e.g. no
    /// load applied).
    fn convergence_metric(&self, state: &[DomainState<B>]) -> Option<f64>;

    /// Target value `convergence_metric` should approach at convergence.
    fn convergence_target(&self) -> f64;
}

/// Panics with a clear message if any `LossTerm::domains()` references a `DomainId` not
/// present in `problem.domains()`. Called once during driver setup (not lazily per-step) so
/// a misconfigured problem fails fast instead of silently indexing into nonexistent state.
pub fn validate_loss_terms(problem: &dyn BoundaryValueProblem) {
    let valid_ids: Vec<DomainId> = problem.domains().iter().map(|d| d.id).collect();
    for term in problem.loss_terms() {
        for id in term.domains() {
            if !valid_ids.contains(&id) {
                panic!(
                    "LossTerm '{}' references DomainId({}) which is not present in this \
                     problem's domains() (valid ids: {:?}) — this is a configuration bug, \
                     not a runtime condition to recover from.",
                    term.name(),
                    id.0,
                    valid_ids.iter().map(|d| d.0).collect::<Vec<_>>()
                );
            }
        }
    }
}

// ─── Multi-domain step driver support (additive; `step_physics`/`StepCtx` unaffected) ────

/// Per-named-point-set arrays a `LossTerm` reads from — normalized coordinates plus normal
/// and tangential-traction targets. Mirrors the `bnd_norm`/`bnd_nx`/`bnd_ny`/`bnd_tx`/
/// `bnd_ty` arrays `StepCtx` carries flat (single point-set) for Kirsch, but keyed by name
/// so a multi-domain problem can have more than one flavor of boundary point per domain
/// (e.g. pin-in-lug's "interface" vs. "free_edge" vs. "driving_load").
#[derive(Clone, Default)]
pub struct PointSetData {
    pub norm: Vec<[f32; 2]>,
    pub nx: Vec<f32>,
    pub ny: Vec<f32>,
    pub tx: Vec<f32>,
    pub ty: Vec<f32>,
}

/// All per-step sampled data for one domain, keyed by named point-set (`"interior"`,
/// `"interface"`, etc. — see [`LossTerm::point_sets`]).
#[derive(Clone)]
pub struct DomainStepData {
    pub id: DomainId,
    /// Normalized interior collocation points for this domain.
    pub int_norm: Vec<[f32; 2]>,
    /// Normalized extra-ring points (generalizes Kirsch's near-hole equilibrium ring).
    pub extra_ring_norm: Vec<[f32; 2]>,
    pub named: HashMap<&'static str, PointSetData>,
}

impl DomainStepData {
    /// Look up a named point-set, panicking (fail-fast, mirrors `validate_loss_terms`'s
    /// tone) with a message naming both the domain and the missing point-set — a term
    /// requesting a point-set its domain's sampling strategy never populated is a wiring
    /// bug, not a runtime condition to recover from.
    pub fn named(&self, name: &str) -> &PointSetData {
        self.named.get(name).unwrap_or_else(|| {
            panic!(
                "DomainStepData::named: DomainId({}) has no point-set named '{name}' — this \
                 is a wiring bug (the domain's sampling strategy must populate every \
                 point-set name its LossTerm::point_sets() declare), not a runtime \
                 condition to recover from.",
                self.id.0
            )
        })
    }
}

/// Mutable per-domain optimizer triple (weight/bias/gate), analogous to the three
/// standalone optimizer arguments `step_physics` takes for its single (implicit) domain.
pub struct DomainOptim {
    pub weight: WeightOptim,
    pub bias: BiasOptim,
    pub gate: GateOptim,
}

/// Read-only per-domain data threaded through `step_physics_multi` — the multi-domain
/// analogue of the scalar fields (`u_ref`/`ref_energy`/`ref_stress2`) `StepCtx` carries for
/// Kirsch's single implicit domain.
pub struct DomainStepCtx<'a> {
    pub data: &'a DomainStepData,
    pub u_ref: f32,
    pub ref_energy: f32,
    pub ref_stress2: f32,
}

/// All read-only data one multi-domain training step needs. The N-domain analogue of
/// `training_core::StepCtx` — see `training_core::step_physics_multi`.
pub struct MultiStepCtx<'a> {
    pub config: &'a SolverConfig,
    pub problem: &'a dyn BoundaryValueProblem,
    pub fd: &'a FdConfig,
    pub k: f32,
    pub domains: Vec<DomainStepCtx<'a>>,
    pub dynamic_lam_h_cap: f64,
    pub dynamic_lam_d_cap: f64,
    pub phase2_active: bool,
    pub step: usize,
}

/// Owned, per-domain analogue of `DomainStepCtx` — clones `data` instead of borrowing it, so
/// it can be stored alongside `FrozenMultiStepCtx` (which must outlive the loop body that
/// produced the original `MultiStepCtx`).
#[derive(Clone)]
pub struct FrozenDomainStepCtx {
    pub data: DomainStepData,
    pub u_ref: f32,
    pub ref_energy: f32,
    pub ref_stress2: f32,
}

/// Owned, 'static-lifetime copy of MultiStepCtx's per-step data, frozen at Converge-tier
/// entry so the L-BFGS closure can outlive the loop body — the multi-domain analogue of
/// training_core::LbfgsCtxScalars::from_ctx.
#[derive(Clone)]
pub struct FrozenMultiStepCtx {
    pub config: SolverConfig,
    pub fd: FdConfig,
    pub k: f32,
    pub domains: Vec<FrozenDomainStepCtx>,
    pub dynamic_lam_h_cap: f64,
    pub dynamic_lam_d_cap: f64,
    pub phase2_active: bool,
}

impl FrozenMultiStepCtx {
    /// Clone every field of `ctx` into an owned, borrow-free snapshot.
    pub fn from_ctx(ctx: &MultiStepCtx) -> Self {
        Self {
            config: ctx.config.clone(),
            fd: *ctx.fd,
            k: ctx.k,
            domains: ctx.domains.iter().map(|d| FrozenDomainStepCtx {
                data: d.data.clone(),
                u_ref: d.u_ref,
                ref_energy: d.ref_energy,
                ref_stress2: d.ref_stress2,
            }).collect(),
            dynamic_lam_h_cap: ctx.dynamic_lam_h_cap,
            dynamic_lam_d_cap: ctx.dynamic_lam_d_cap,
            phase2_active: ctx.phase2_active,
        }
    }

    /// Rebuild a borrowing `MultiStepCtx` view over this frozen snapshot's owned data, so the
    /// existing `step_physics_multi`/`compute_domain_forwards` machinery (which takes
    /// `&MultiStepCtx`) can be reused verbatim inside the L-BFGS closure. `step` is not
    /// tracked by `FrozenMultiStepCtx` (irrelevant once frozen — no phase transitions happen
    /// mid-Converge), so it is always reported as `0`.
    pub fn as_multi_step_ctx<'a>(&'a self, problem: &'a dyn BoundaryValueProblem) -> MultiStepCtx<'a> {
        MultiStepCtx {
            config: &self.config,
            problem,
            fd: &self.fd,
            k: self.k,
            domains: self.domains.iter().map(|d| DomainStepCtx {
                data: &d.data,
                u_ref: d.u_ref,
                ref_energy: d.ref_energy,
                ref_stress2: d.ref_stress2,
            }).collect(),
            dynamic_lam_h_cap: self.dynamic_lam_h_cap,
            dynamic_lam_d_cap: self.dynamic_lam_d_cap,
            phase2_active: self.phase2_active,
            step: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinn_core::{
        geometry::GeometryConfig,
        material::MaterialProps,
        problem::DirichletAnsatz,
    };

    struct DummyAnsatz;
    impl DirichletAnsatz for DummyAnsatz {
        fn eval(&self, _xn: f32, _yn: f32, _k: f32) -> (f32, f32) { (1.0, 1.0) }
    }

    struct DummySampling;
    impl DomainSamplingStrategy for DummySampling {
        fn sample_interior(&self, _geom: &GeometryConfig, _n: usize) -> Vec<[f64; 2]> { Vec::new() }
        fn sample_boundary(&self, _geom: &GeometryConfig, _load: &pinn_core::loading::LoadConfig, _n: usize) -> Vec<pinn_core::loading::BoundaryPoint> { Vec::new() }
        fn amr_lock_zone(&self, _geom: &GeometryConfig, _cell_center: [f64; 2]) -> bool { false }
        fn sample_extra_ring(&self, _geom: &GeometryConfig, _n: usize) -> Vec<[f64; 2]> { Vec::new() }
    }

    struct BadDomainTerm {
        domains: Vec<DomainId>,
    }
    impl LossTerm for BadDomainTerm {
        fn name(&self) -> &'static str { "bad_domain_term" }
        fn domains(&self) -> Vec<DomainId> { self.domains.clone() }
        fn compute(&self, _inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
            Tensor::<B, 1>::zeros([1], &Default::default())
        }
    }

    struct MismatchedDomainProblem {
        domains: Vec<DomainSpec>,
        sampling: DummySampling,
        ansatz: DummyAnsatz,
        term_domains: Vec<DomainId>,
    }
    impl BoundaryValueProblem for MismatchedDomainProblem {
        fn domains(&self) -> &[DomainSpec] { &self.domains }
        fn sampling_strategy(&self, _domain_idx: usize) -> &dyn DomainSamplingStrategy { &self.sampling }
        fn ansatz(&self, _domain_idx: usize) -> &dyn DirichletAnsatz { &self.ansatz }
        fn loss_terms(&self) -> Vec<Box<dyn LossTerm>> {
            vec![Box::new(BadDomainTerm { domains: self.term_domains.clone() })]
        }
        fn base_weight(&self, _term_name: &str) -> f32 { 1.0 }
        fn phase1_steps(&self) -> usize { 0 }
        fn convergence_metric(&self, _state: &[DomainState<B>]) -> Option<f64> { None }
        fn convergence_target(&self) -> f64 { 0.0 }
    }

    fn dummy_domain_spec(id: u32) -> DomainSpec {
        DomainSpec {
            id: DomainId(id),
            geometry: GeometryConfig::kirsch_plate_inches(),
            material: MaterialProps::al7075_t6(),
            output_dim: 5,
        }
    }

    /// Extends the existing `validate_loss_terms` coverage: the invalid `DomainId` sits at
    /// index 1 (not 0) of a two-element `domains()` — proves the validator doesn't
    /// short-circuit after finding the first VALID id and skip checking the rest.
    #[test]
    #[should_panic(expected = "DomainId(99)")]
    fn validate_loss_terms_catches_invalid_domain_id_not_at_index_zero() {
        let problem = MismatchedDomainProblem {
            domains: vec![dummy_domain_spec(0)],
            sampling: DummySampling,
            ansatz: DummyAnsatz,
            term_domains: vec![DomainId(0), DomainId(99)],
        };
        validate_loss_terms(&problem);
    }

    #[test]
    #[should_panic(expected = "DomainId(7) has no point-set named 'interface'")]
    fn domain_step_data_named_panics_with_domain_and_pointset_name_on_missing_key() {
        let data = DomainStepData {
            id: DomainId(7),
            int_norm: Vec::new(),
            extra_ring_norm: Vec::new(),
            named: HashMap::new(),
        };
        let _ = data.named("interface");
    }

    /// Minimal fixture `LossTerm` that does NOT override `conflict_group()` — must fall back
    /// to the trait default, `ConflictGroup::Bc`.
    struct DefaultConflictGroupTerm;
    impl LossTerm for DefaultConflictGroupTerm {
        fn name(&self) -> &'static str { "default_conflict_group_term" }
        fn domains(&self) -> Vec<DomainId> { vec![DomainId(0)] }
        fn compute(&self, _inputs: &[DomainForwardOutputs<'_, B>]) -> Tensor<B, 1> {
            Tensor::<B, 1>::zeros([1], &Default::default())
        }
    }

    #[test]
    fn loss_term_conflict_group_defaults_to_bc_when_not_overridden() {
        let term = DefaultConflictGroupTerm;
        assert_eq!(term.conflict_group(), ConflictGroup::Bc);
    }
}
