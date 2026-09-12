//! Issue #61 EPIC P2-03: authoritative field dependency graph.
//!
//! The prior General-PINN pass's Priority 1 (`crate::problem::StressSource`,
//! `training_core::stress_source_report`) gave every stress-reading `LossTerm` a queryable
//! `Direct`/`Derived`/`Both` label — a real, useful capability, but only a two-node
//! classification (`raw network output` vs. `C:ε`), not the full pipeline issue #61 §3
//! describes: `NN -> displacement -> strain -> constitutive -> stress`. It also never
//! *enforced* anything: a term could read the network's raw stress channel (`Direct`) while
//! another term relied on the constitutive-derived value (`Derived`) with nothing checking
//! that the two representations were required to agree wherever both are in play.
//!
//! This module closes both gaps at the scope this codebase's real architecture actually
//! needs (per issue #61 §1.4's "no destructive refactor" — this does NOT rip out or replace
//! `StressSource`, it generalizes it):
//!
//! 1. [`FieldKind`] names every node in the real dependency chain and can compute the
//!    ordered path from the network to any downstream field ([`FieldKind::dependency_chain`]).
//!    [`FieldKind::from_stress_source`] maps the existing `StressSource` classification onto
//!    the two stress-producing leaves (`DirectStress`/`ConstitutiveStress`) — the graph
//!    *generalizes* the existing label, it does not duplicate it.
//! 2. [`check_mixed_stress_source_compatibility`] makes "mixed formulations enforce sigma_aux
//!    <-> constitutive compatibility explicitly" (P2-03's own acceptance wording) a real,
//!    testable predicate: given a `stress_source_report`-shaped list AND whether this
//!    codebase's existing compatibility mechanism (`ConstitutiveConsistencyTerm`, applied to
//!    every mDEM domain in `step_physics_multi`) is active, it answers whether the mix is
//!    *covered*. `step_physics_multi` calls this every step (cheap — reuses the already-built
//!    `active_terms` list, no extra `loss_terms()` construction) and panics if a future
//!    formulation change ever produces a genuinely uncovered mix — see that call site's own
//!    comment for why this is a real enforcement point, not a diagnostic-only report.

use crate::problem::StressSource;

/// A node in the authoritative field dependency graph (issue #61 §3's pipeline diagram,
/// narrowed to the fields this codebase's plate/Kirsch/pin-lug problems actually compute).
/// `NetworkOutput` is the graph's one root; every other variant has exactly one upstream
/// dependency (see [`FieldKind::depends_on`]), matching the fact that this codebase's real
/// pipeline is a chain, not a general DAG — `DirectStress` is the one exception, branching
/// straight off `NetworkOutput` (mDEM's raw stress output columns) rather than descending
/// through `Strain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldKind {
    /// The network's raw forward-pass output tensor (displacement columns, and for mDEM
    /// domains, raw stress columns too).
    NetworkOutput,
    /// Displacement field, read directly from `NetworkOutput`'s first two columns.
    Displacement,
    /// Strain, computed from `Displacement` via FD or AD differentiation
    /// (`differential_operator::fd_strain_via`/`ad_strain`, or the historical inline
    /// `fd_stencil` call sites they generalize).
    Strain,
    /// Stress derived from `Strain` via the constitutive law (`energy::compute_stress`,
    /// `σ = C:ε`). The authoritative field for any consumer that wants Hooke's-law-consistent
    /// stress.
    ConstitutiveStress,
    /// Stress read directly from the network's own mDEM output columns, with no constitutive
    /// derivation involved. Only meaningful for `output_dim == 5` domains.
    DirectStress,
}

impl FieldKind {
    /// This field's single upstream dependency, or `None` for the graph's root
    /// (`NetworkOutput`).
    pub fn depends_on(self) -> Option<FieldKind> {
        match self {
            FieldKind::NetworkOutput => None,
            FieldKind::Displacement => Some(FieldKind::NetworkOutput),
            FieldKind::Strain => Some(FieldKind::Displacement),
            FieldKind::ConstitutiveStress => Some(FieldKind::Strain),
            FieldKind::DirectStress => Some(FieldKind::NetworkOutput),
        }
    }

    /// The full ordered dependency chain from the graph's root down to (and including) this
    /// field — e.g. `ConstitutiveStress.dependency_chain()` returns
    /// `[NetworkOutput, Displacement, Strain, ConstitutiveStress]`. This is the generic
    /// primitive `user_problem::dependency_chain_for_kt` is built from (see that function) —
    /// proof this graph is load-bearing, not a dead parallel abstraction.
    pub fn dependency_chain(self) -> Vec<FieldKind> {
        let mut chain = vec![self];
        let mut cur = self;
        while let Some(dep) = cur.depends_on() {
            chain.push(dep);
            cur = dep;
        }
        chain.reverse();
        chain
    }

    /// Maps the existing (narrower) [`StressSource`] classification onto this graph's two
    /// stress-producing leaves. `Both` maps to `ConstitutiveStress` as the "authoritative"
    /// side of the pair — a term classified `Both` (e.g. `ConstitutiveConsistencyTerm`) reads
    /// both representations by definition, but its role in the graph is to hold the
    /// constitutive one authoritative and check the direct one against it, not the reverse.
    pub fn from_stress_source(source: StressSource) -> FieldKind {
        match source {
            StressSource::Direct => FieldKind::DirectStress,
            StressSource::Derived | StressSource::Both => FieldKind::ConstitutiveStress,
        }
    }

    /// Human-readable node name for chain-printing (`dependency_chain_for_kt` and similar).
    pub fn label(self) -> &'static str {
        match self {
            FieldKind::NetworkOutput => "network_output",
            FieldKind::Displacement => "displacement",
            FieldKind::Strain => "strain",
            FieldKind::ConstitutiveStress => "constitutive_stress",
            FieldKind::DirectStress => "direct_stress",
        }
    }
}

/// Result of [`check_mixed_stress_source_compatibility`] — whether a problem's active terms
/// mix `Direct` and `Derived`/`Both` stress sources, and if so, whether a compatibility
/// mechanism (something classified `StressSource::Both`, or the caller's separately-tracked
/// `external_compatibility_active` flag for mechanisms that live outside the `LossTerm` list,
/// e.g. `step_physics_multi`'s `ConstitutiveConsistencyTerm` injection) is present to police
/// the gap between them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedFormulationCheck {
    pub direct_terms: Vec<&'static str>,
    pub derived_or_both_terms: Vec<&'static str>,
    /// True iff at least one `Direct` term AND at least one `Derived`/`Both` term are active
    /// simultaneously — i.e. this problem genuinely relies on both representations at once.
    pub mixed: bool,
    /// True iff `mixed` is false (nothing to enforce), OR a `Both`-classified term is present,
    /// OR the caller declared an external compatibility mechanism active.
    pub compatibility_enforced: bool,
}

/// Issue #61 P2-03's own acceptance wording: "mixed formulations enforce sigma_aux <->
/// constitutive compatibility explicitly." `report` is a `stress_source_report`-shaped list
/// (name, source) for a problem's currently-active terms. `external_compatibility_active`
/// lets a caller declare that a compatibility mechanism outside the `LossTerm` list is
/// running this step (`step_physics_multi` passes its own `any_mdem` — true exactly when
/// `ConstitutiveConsistencyTerm` was applied to at least one domain this step).
pub fn check_mixed_stress_source_compatibility(
    report: &[(&'static str, StressSource)],
    external_compatibility_active: bool,
) -> MixedFormulationCheck {
    let direct_terms: Vec<&'static str> = report.iter()
        .filter(|(_, s)| matches!(s, StressSource::Direct))
        .map(|(name, _)| *name)
        .collect();
    let derived_or_both_terms: Vec<&'static str> = report.iter()
        .filter(|(_, s)| matches!(s, StressSource::Derived | StressSource::Both))
        .map(|(name, _)| *name)
        .collect();
    let has_both_term = report.iter().any(|(_, s)| matches!(s, StressSource::Both));

    let mixed = !direct_terms.is_empty() && !derived_or_both_terms.is_empty();
    let compatibility_enforced = !mixed || has_both_term || external_compatibility_active;

    MixedFormulationCheck { direct_terms, derived_or_both_terms, mixed, compatibility_enforced }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dependency_chain_matches_issue_61_section_3_pipeline_for_constitutive_stress() {
        let chain = FieldKind::ConstitutiveStress.dependency_chain();
        assert_eq!(chain, vec![
            FieldKind::NetworkOutput,
            FieldKind::Displacement,
            FieldKind::Strain,
            FieldKind::ConstitutiveStress,
        ]);
    }

    #[test]
    fn dependency_chain_for_direct_stress_branches_straight_off_network_output() {
        let chain = FieldKind::DirectStress.dependency_chain();
        assert_eq!(chain, vec![FieldKind::NetworkOutput, FieldKind::DirectStress]);
    }

    #[test]
    fn from_stress_source_maps_derived_and_both_onto_the_same_authoritative_leaf() {
        assert_eq!(FieldKind::from_stress_source(StressSource::Direct), FieldKind::DirectStress);
        assert_eq!(FieldKind::from_stress_source(StressSource::Derived), FieldKind::ConstitutiveStress);
        assert_eq!(FieldKind::from_stress_source(StressSource::Both), FieldKind::ConstitutiveStress);
    }

    /// Reproduces the actual single-hole plate configuration's real term classification
    /// (`docs/investigations/kt-investigation-bugsource-new.md`'s own written conclusion,
    /// already regression-guarded by `user_problem::stress_source_report_matches_the_kt_
    /// investigation_docs_written_conclusion`): `equilibrium`/`outer_traction` derived,
    /// `hole_free` direct, no `Both`-classified term in `loss_terms()` itself. This mix IS
    /// covered in the real running system, but only via `step_physics_multi`'s *external*
    /// `ConstitutiveConsistencyTerm` injection (`any_mdem`) — not via anything in
    /// `loss_terms()`. Proves the check correctly requires the external flag here.
    #[test]
    fn plate_mdem_mix_is_uncovered_without_the_external_flag_and_covered_with_it() {
        let report: Vec<(&'static str, StressSource)> = vec![
            ("equilibrium", StressSource::Derived),
            ("outer_traction", StressSource::Derived),
            ("hole_free", StressSource::Direct),
        ];

        let without_external = check_mixed_stress_source_compatibility(&report, false);
        assert!(without_external.mixed);
        assert!(!without_external.compatibility_enforced, "a real architecture gap: mixed sources with no compatibility mechanism must not be silently reported as fine");

        let with_external = check_mixed_stress_source_compatibility(&report, true);
        assert!(with_external.mixed);
        assert!(with_external.compatibility_enforced);
    }

    #[test]
    fn kirsch_both_classified_term_covers_the_mix_with_no_external_flag_needed() {
        let report: Vec<(&'static str, StressSource)> = vec![
            ("equilibrium_ring", StressSource::Direct),
            ("kirsch_constitutive_consistency", StressSource::Both),
        ];
        let check = check_mixed_stress_source_compatibility(&report, false);
        assert!(check.mixed);
        assert!(check.compatibility_enforced);
    }

    #[test]
    fn single_source_reports_are_never_flagged_as_mixed() {
        let all_derived: Vec<(&'static str, StressSource)> =
            vec![("equilibrium", StressSource::Derived), ("outer_traction", StressSource::Derived)];
        let check = check_mixed_stress_source_compatibility(&all_derived, false);
        assert!(!check.mixed);
        assert!(check.compatibility_enforced);

        let empty: Vec<(&'static str, StressSource)> = vec![];
        let check_empty = check_mixed_stress_source_compatibility(&empty, false);
        assert!(!check_empty.mixed);
        assert!(check_empty.compatibility_enforced);
    }
}
