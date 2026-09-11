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
pub use crate::training_core::{BDevice, B};

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

/// Direct mDEM stress (σxx,σyy,σxy - only the 3 components `energy::equilibrium_residual_loss`
/// needs per shifted position) at the 4 FD-shifted positions `assemble_stencil` already
/// evaluates - `(sxx_xp, sxy_xp, sxx_xm, sxy_xm, sxy_yp, syy_yp, sxy_ym, syy_ym)`, matching
/// that function's parameter order exactly so callers never need to reshuffle. See
/// `training_core::compute_domain_forwards`'s doc comment for how this is populated at zero
/// extra forward-pass cost (the same stencil rows already computed for FD strains).
pub type ShiftedStress<Bk> = (
    Tensor<Bk, 1>, Tensor<Bk, 1>, Tensor<Bk, 1>, Tensor<Bk, 1>,
    Tensor<Bk, 1>, Tensor<Bk, 1>, Tensor<Bk, 1>, Tensor<Bk, 1>,
);

/// Displacement Hessian `(u_xx, u_yy, u_xy, v_xx, v_yy, v_xy)` from `fd_stencil::
/// compute_hessian`'s 9-point stencil — the derived-stress equilibrium alternative to
/// `ShiftedStress` (see `energy::equilibrium_from_displacement_hessian_loss`'s doc comment
/// for why this exists: the plate's direct-σ-based equilibrium term was found functionally
/// inert, bugSource-New #12). Unlike `shifted_stress`, this is NOT free to populate — it needs
/// a genuinely new 9-point forward pass (the existing 5-point stencil for `strains`/
/// `shifted_stress` doesn't include the 4 diagonal points a mixed partial needs) — see
/// `LossTerm::needs_hessian`.
pub type HessianData<Bk> = (
    Tensor<Bk, 1>, Tensor<Bk, 1>, Tensor<Bk, 1>,
    Tensor<Bk, 1>, Tensor<Bk, 1>, Tensor<Bk, 1>,
);

/// Per-domain forward-pass outputs handed to `LossTerm::compute`. `strains`/`normals`/
/// `shifted_stress`/`hessian` are `None` when the term doesn't need them (e.g. a term
/// operating purely on `raw_out`).
pub struct DomainForwardOutputs<'a, Bk: Backend> {
    pub domain: DomainId,
    pub raw_out: &'a Tensor<Bk, 2>,
    pub strains: Option<(Tensor<Bk, 1>, Tensor<Bk, 1>, Tensor<Bk, 1>)>,
    pub normals: Option<(Tensor<Bk, 1>, Tensor<Bk, 1>)>,
    pub shifted_stress: Option<ShiftedStress<Bk>>,
    pub hessian: Option<HessianData<Bk>>,
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

    /// Computes this term's (unweighted) scalar loss from the given domains' forward-pass
    /// outputs. The returned `Tensor<B, 1>` MUST remain connected to the live autodiff graph
    /// rooted at `inputs`'s `raw_out`/`strains`/`normals` tensors — build the ENTIRE
    /// computation (including any nonlinear penalty function) out of `Tensor` ops on backend
    /// `B` all the way to the returned scalar. Calling `.into_data()`/`.to_vec()` (or any
    /// other host-transferring op) on an operand that needs gradient, then rebuilding a fresh
    /// tensor from the resulting `Vec` via `Tensor::from_data`, creates a brand-new autodiff
    /// LEAF with no edge back to `inputs` — the term's scalar VALUE will still look correct in
    /// console logging / SAW-BRDR bookkeeping (which only reads the value), but it supplies
    /// ZERO gradient to `.backward()`, silently making the term inert to gradient descent (see
    /// the pin-lug Signorini-term autodiff-detachment bug, Issue #9, this note exists to
    /// prevent a repeat of). Converting to a plain `Vec`/scalar is fine ONLY for host-side data
    /// that never re-enters the returned tensor's graph (e.g. fixed non-learned constants like
    /// `thetas`, or values used purely for diagnostic printing).
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

    /// True if this term needs `DomainForwardOutputs::hessian` populated. Defaults to `false`
    /// (zero behavior/cost change for every existing term — Kirsch/pin-lug never override
    /// this). `compute_domain_forwards` runs the additional 9-point Hessian stencil forward
    /// pass ONLY for `(domain, point_set)` pairs whose active term(s) include one that
    /// returns `true` here — see that function's doc comment.
    fn needs_hessian(&self) -> bool {
        false
    }

    /// Classifies this term as either enforcing interior PDE/equilibrium physics or a
    /// boundary/interface condition — used by `compute_gradient_conflict_multi`'s dual-backward
    /// split. Defaults to `Bc`. Every concrete LossTerm impl in pinlug_problem.rs must override
    /// this explicitly (do not rely on the default silently) — a RED test pins each term's
    /// expected classification so a future term added without an override is caught immediately.
    fn conflict_group(&self) -> ConflictGroup {
        ConflictGroup::Bc
    }

    /// Which stress representation this term's physics actually depends on, if any -
    /// General-PINN architecture recommendations §4's "every derived quantity must identify
    /// its source", narrowed to the ONE dependency edge this codebase's own real bugs have
    /// repeatedly been about (bugSource-New #2/#11/#12: does a term read the network's direct
    /// mDEM σ output, or σ derived via `energy::compute_stress` from strain/Hessian?).
    /// Defaults to `None` - NOT a cop-out default: a term that operates purely on strain or
    /// displacement (no stress quantity anywhere in its `compute()`) genuinely has no stress
    /// source to report, and should leave this at the default rather than picking one
    /// arbitrarily. Every concrete `LossTerm` impl that DOES read a stress value must override
    /// this explicitly, classified by actually reading that term's `compute()` body - same
    /// "no silent default" discipline `conflict_group` already established, extended to this
    /// new axis. See [`crate::training_core::stress_source_report`] for the generic consumer.
    fn stress_source(&self) -> Option<StressSource> {
        None
    }
}

/// Classifies a [`LossTerm`] as either enforcing interior PDE/equilibrium physics or a
/// boundary/interface condition. See [`LossTerm::conflict_group`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConflictGroup { Physics, Bc }

/// Which stress representation a [`LossTerm`] (or a standalone diagnostic probe, e.g.
/// `user_problem::probe_hole_boundary_profile`/`probe_hole_boundary_profile_derived`) actually
/// reads. See [`LossTerm::stress_source`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StressSource {
    /// Read directly from the network's own mDEM output columns (`raw_out`/`shifted_stress`) -
    /// never validated against Hooke's law by anything outside a `constitutive_consistency`
    /// term, if one happens to be registered.
    Direct,
    /// Computed via `energy::compute_stress` (`σ=C:ε`) from FD- or Hessian-derived strain.
    Derived,
    /// Reads AND compares both representations against each other - `constitutive_consistency`
    /// is the one real example (its entire purpose is `‖σ_direct − σ_derived‖²`), added as a
    /// third variant rather than forcing this term into `Direct` or `Derived` alone, which
    /// would misreport exactly the term whose job is to police the gap between them.
    Both,
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
    /// Cap on `interface_penetration`'s SAW-BRDR effective weight, mirroring
    /// `dynamic_lam_h_cap`'s role for `hole_traction`/`lug_free_edge_traction` — see
    /// `step_physics_multi`'s dispatch match. Only binds once SAW-BRDR pushes the term's
    /// live weight above whatever this is currently set to; the caller (`headless.rs`)
    /// seeds it at the term's own `base_weight` (500.0), not the shared 50.0 h/d convention.
    pub dynamic_lam_penetration_cap: f64,
    /// Cap on `interface_non_tension`'s SAW-BRDR effective weight — see
    /// `dynamic_lam_penetration_cap`'s doc comment (same convention, seeded at 100.0).
    pub dynamic_lam_non_tension_cap: f64,
    /// Fixed weight `step_physics_multi` applies to the `constitutive_consistency` term for
    /// every mDEM (`output_dim == 5`) domain — see `training_core::LAM_CONSTITUTIVE_
    /// CONSISTENCY`'s own doc comment for why this is a plain ctx field (not adaptive/
    /// per-step) and why every caller except `run_user_problem_training_from`'s real
    /// per-step ctx (and the headless CLI path that mirrors it) sets it to that constant,
    /// unchanged from before this field existed.
    pub constitutive_consistency_weight: f64,
    /// Positional-Fourier-feature count every domain's forward pass uses (see
    /// `UserGeometry::n_fourier`'s doc comment for the full root-cause story). MUST match
    /// what every domain's model was actually constructed with (`net_input_dim()`) - a
    /// mismatch panics on the first forward pass (wrong input tensor width), not silently
    /// corrupts anything. Global across all domains in this ctx (not per-domain) since the
    /// only caller that sets this non-zero (`run_user_problem_training_from`) is always
    /// single-domain; every other caller keeps this `0`, byte-identical to before this field
    /// existed.
    pub n_fourier: usize,
    /// Diagnostic-only, opt-in (default `false` everywhere except dedicated diagnostics): when
    /// true, `step_physics_multi` computes each active term's OWN gradient L2 norm (an extra
    /// `.backward()` pass per term) and populates `StepOutput.term_grad_norms`. Real cost (N
    /// extra backward passes per step) - never set true on a hot training path. See
    /// `StepOutput::term_grad_norms`'s doc comment.
    pub probe_term_gradients: bool,
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
    pub dynamic_lam_penetration_cap: f64,
    pub dynamic_lam_non_tension_cap: f64,
    pub constitutive_consistency_weight: f64,
    pub n_fourier: usize,
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
            dynamic_lam_penetration_cap: ctx.dynamic_lam_penetration_cap,
            dynamic_lam_non_tension_cap: ctx.dynamic_lam_non_tension_cap,
            constitutive_consistency_weight: ctx.constitutive_consistency_weight,
            n_fourier: ctx.n_fourier,
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
            dynamic_lam_penetration_cap: self.dynamic_lam_penetration_cap,
            dynamic_lam_non_tension_cap: self.dynamic_lam_non_tension_cap,
            constitutive_consistency_weight: self.constitutive_consistency_weight,
            n_fourier: self.n_fourier,
            // Not tracked by `FrozenMultiStepCtx` (same rationale as `step` above) - term-
            // gradient probing is a dedicated-diagnostic-only concern, never needed on the
            // L-BFGS/Converge path this reconstructs for.
            probe_term_gradients: false,
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

    /// `dynamic_lam_penetration_cap`/`dynamic_lam_non_tension_cap` must round-trip through
    /// `FrozenMultiStepCtx::from_ctx` -> `as_multi_step_ctx` bit-exact (plain f64 field
    /// copies, no arithmetic) — proves the two new cap fields are threaded through the
    /// freeze/thaw path alongside the pre-existing `dynamic_lam_h_cap`/`dynamic_lam_d_cap`.
    /// Uses arbitrary distinct-from-h/d-cap values so a copy-paste field-swap bug (e.g.
    /// `dynamic_lam_penetration_cap` accidentally reading `dynamic_lam_h_cap`) is detectable.
    #[test]
    fn frozen_multi_step_ctx_round_trips_the_two_new_interface_caps() {
        let problem = MismatchedDomainProblem {
            domains: vec![dummy_domain_spec(0)],
            sampling: DummySampling,
            ansatz: DummyAnsatz,
            term_domains: vec![DomainId(0)],
        };
        let config = SolverConfig::default_pinlug();
        let fd = FdConfig::new(1e-3, 2.0, 2.0);
        let data = DomainStepData {
            id: DomainId(0),
            int_norm: Vec::new(),
            extra_ring_norm: Vec::new(),
            named: HashMap::new(),
        };
        let ctx = MultiStepCtx {
            config: &config,
            problem: &problem,
            fd: &fd,
            k: 1.0,
            domains: vec![DomainStepCtx { data: &data, u_ref: 1.0, ref_energy: 1.0, ref_stress2: 1.0 }],
            dynamic_lam_h_cap: 50.0,
            dynamic_lam_d_cap: 50.0,
            dynamic_lam_penetration_cap: 500.0,
            dynamic_lam_non_tension_cap: 100.0,
            constitutive_consistency_weight: crate::training_core::LAM_CONSTITUTIVE_CONSISTENCY,
            n_fourier: 0,
            probe_term_gradients: false,
            phase2_active: true,
            step: 0,
        };

        let frozen = FrozenMultiStepCtx::from_ctx(&ctx);
        let thawed = frozen.as_multi_step_ctx(&problem);

        assert_eq!(thawed.dynamic_lam_penetration_cap, 500.0);
        assert_eq!(thawed.dynamic_lam_non_tension_cap, 100.0);
        // Adversarial: prove the two new fields aren't aliased to the pre-existing h/d caps.
        assert_ne!(thawed.dynamic_lam_penetration_cap, thawed.dynamic_lam_h_cap);
        assert_ne!(thawed.dynamic_lam_non_tension_cap, thawed.dynamic_lam_d_cap);
    }
}
