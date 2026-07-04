/// Generic boundary-value-problem trait family (tensor-graph layer). Builds on
/// `pinn_core::problem` (geometry/sampling/ansatz, burn-free) to let the training loop
/// (`training_core`/`runner`/`headless`) drive an arbitrary number of physical domains and
/// loss terms instead of being hardwired to the single-domain Kirsch problem.
///
/// `KirschProblem` (`kirsch_problem.rs`) is the reference implementation — exactly one
/// domain. A follow-up pin-in-lug problem adds a second domain plus cross-domain
/// (`LossTerm::domains().len() == 2`) contact loss terms on top of the same traits.
use burn::tensor::{backend::Backend, Tensor};

use pinn_core::problem::{DirichletAnsatz, DomainId, DomainSamplingStrategy, DomainSpec};

use crate::network::ElasticityNet;

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
}

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
